import torch
from RL4CRN.utils.ffnn import FFNN
from torch.distributions import Categorical, LogNormal
from RL4CRN.utils.utils import batch_multi_hot
from RL4CRN.policies.parameter_generator_from_distribution import ParameterGeneratorFromDistribution
import numpy as np

class AddReactionByIndex(torch.nn.Module):
    """ A policy that generates a reaction by its index to an IOCRN in batch mode.
        The policy consists of a neural network with multiple heads that outputs the reaction structure, 
        the reaction parameters (continuous and discrete, if applicable), and the input influence (if 
        applicable). The reaction structure is represented as a categorical distribution over the reactions 
        in the indexed set, from which a reaction is sampled. The reaction parameters are generated from 
        specified distributions (e.g., log-normal for continuous parameters and categorical for discrete 
        parameters) using separate heads. #TODO The input influence generation is not implemented yet. """
    def __init__(self, num_reactions, num_parameters, num_inputs, 
                 encoder_attributes, deep_layer_size, structure_head_attributes, parameter_head_attributes, 
                 input_influence_head_attributes, masks=None,
                 continuous_distribution={"type": 'lognormal'}, 
                 discrete_distribution={"type": 'categorical', "categories": [torch.tensor([1, 2])]}, 
                 allow_input_influence=False, device=None):
        """ Initializes the AddReactionByIndex policy.
        Arguments:
        - num_reactions: total number of reactions to select from (assumed to be the same for all IOCRNs in the batch).
        - num_parameters: total number of parameters (continuous + discrete) across all possible reactions.
        - num_inputs: number of inputs in the IOCRN (assumed to be the same for all IOCRNs in the batch).
        - encoder_attributes: a dictionary containing the attributes of the encoder neural network (hidden_size, num_layers).
        - deep_layer_size: size of the deep layer representation of the IOCRN.
        - structure_head_attributes: a dictionary containing the attributes of the reaction structure head neural network (hidden_size, num_layers).
        - parameter_head_attributes: a dictionary containing the attributes of the reaction parameter head neural network (hidden_size, num_layers).
        - input_influence_head_attributes: a dictionary containing the attributes of the input influence head neural network (hidden_size, num_layers).
        - masks: a dictionary containing the masks for the continuous parameters, discrete parameters, and logits (default is None, which means no masks are applied). The keys are:
            - 'continuous': a binary numpy array of shape (num_reactions, max_num_continuous_parameters) indicating the presence of continuous parameters for each reaction.
            - 'discrete': a binary numpy array of shape (num_reactions, max_num_discrete_parameters) indicating the presence of discrete parameters for each reaction.
            - 'logit': a binary numpy array of shape (num_reactions, total_num_categories_for_all_discrete_parameters) indicating the valid logits for the discrete parameters for each reaction.   
        - continuous_distribution: a dictionary specifying the type of the continuous parameter distribution (default is log-normal).
        - discrete_distribution: a dictionary specifying the type and categories of the discrete parameter distribution (default is categorical).
        - allow_input_influence: if True, the policy will include an input influence head (default is False).
        - device: device to run the policy on (default is None, which uses CPU). """

        super().__init__()

        # Record the IOCRN attributes
        self.M = num_reactions                                              # Total number of reactions
        self.K = num_parameters                                             # Total number of parameters (continuous + discrete) across all reactions
        self.p = num_inputs                                                 # Number of inputs in the IOCRN

        # Record the neural network attributes
        self.encoder_attributes = encoder_attributes
        self.deep_layer_size = deep_layer_size
        self.structure_head_attributes = structure_head_attributes
        self.parameter_head_attributes = parameter_head_attributes
        self.input_influence_head_attributes = input_influence_head_attributes
        self.allow_input_influence = allow_input_influence
        self.device = device if device is not None else torch.device('cpu')

        # Record the distribution attributes
        self.continuous_distribution = continuous_distribution
        self.discrete_distribution = discrete_distribution

        # Tensorize the masks, if provided, and tensorize them into the specified device
        if masks is None:
            masks = {'continuous': None, 'discrete': None, 'logit': None}
        self.continuous_parameter_mask = masks['continuous']
        self.continuous_parameter_mask = torch.tensor(self.continuous_parameter_mask).to(self.device) if self.continuous_parameter_mask is not None else None # Shape: (M, max_num_continuous_parameters)
        self.discrete_parameter_mask = masks['discrete']
        self.discrete_parameter_mask = torch.tensor(self.discrete_parameter_mask).to(self.device) if self.discrete_parameter_mask is not None else None  # Shape: (M, max_num_discrete_parameters)
        self.logit_mask = masks['logit']

        # Define the encoder that encodes the IOCRN observation into a deep layer representation
        self.encoder = FFNN(input_size=self.M + (self.p + 1) * self.K, output_size=deep_layer_size, hidden_size=encoder_attributes["hidden_size"], num_layers=encoder_attributes["num_layers"]).to(device=device)
        
        # Define the reaction structure head that reads the deep layer representation to output the logits for the reaction structure
        self.reaction_structure_head = FFNN(input_size=deep_layer_size, output_size=self.M, hidden_size=structure_head_attributes["hidden_size"], num_layers=structure_head_attributes["num_layers"]).to(device=device)
        
        # Define the continuous parameter head that reads the deep layer representation and the reaction structure to deliver to the continuous parameter generator
        if self.continuous_parameter_mask is not None:

            # Set the dimension of the continuous distribution to the maximum number of continuous parameters across all reactions in the library
            self.continuous_distribution["dim"] = self.continuous_parameter_mask.shape[1] 

            # Construct the continuous parameter generator
            self.continuous_parameter_generator = ParameterGeneratorFromDistribution(distribution=self.continuous_distribution, 
                                                                                     backbone_attributes={  "input_size": deep_layer_size + self.M, 
                                                                                                            "hidden_size": parameter_head_attributes["hidden_size"], 
                                                                                                            "num_layers": parameter_head_attributes["num_layers"]
                                                                                                            }, 
                                                                                    device=self.device
                                                                                    ).to(device=self.device)   
        else:
            self.continuous_parameter_generator = None

        # Define the discrete parameter head that reads the deep layer representation, the reaction structure, and the continuous parameters to deliver to the discrete parameter generator
        if self.discrete_parameter_mask is not None:
            
            # Set the dimension of the discrete distribution to the maximum number of discrete parameters across all reactions in the library
            self.discrete_distribution["dim"] = self.discrete_parameter_mask.shape[1]

            # Make the discrete distribution categories the same for all discrete parameters #TODO Allow different categories for different discrete parameters
            self.discrete_distribution["categories"] = self.discrete_distribution["dim"] * self.discrete_distribution["categories"] # List of tensors, each of shape (num_categories,) where num_categories is the number of categories for each discrete parameter

            # Calculate the input size of the discrete parameter head
            discrete_parameter_head_input_size = deep_layer_size + self.M + (self.continuous_parameter_mask.shape[1] if self.continuous_parameter_mask is not None else 0)

            # Construct the discrete parameter generator
            self.discrete_parameter_generator = ParameterGeneratorFromDistribution(distribution=self.discrete_distribution, 
                                                                                     backbone_attributes={  "input_size": discrete_parameter_head_input_size, 
                                                                                                            "hidden_size": parameter_head_attributes["hidden_size"], 
                                                                                                            "num_layers": parameter_head_attributes["num_layers"]
                                                                                                            }, 
                                                                                    device=self.device
                                                                                    ).to(device=self.device)   
        else:
            self.discrete_parameter_generator = None
            
        # Define the input influence head that reads the deep layer representation, the reaction structure and the reaction rate to output the logits for the input influence
        if allow_input_influence is True:
            raise NotImplementedError("The input influence head is not implemented yet.")
        else:
            self.input_influence_head = None
                  
    def forward(self, state, mode='full', action=None):
        """ Generates an action (reaction structure, parameters and input influence) given the observation of the state received from the observer.
        Args:
        - state (torch.Tensor): The observation (state) of the IOCRN. Shape: (N, M + (p+1)*K)), where N is the batch size, M is the number of reactions in the library, p is the number of inputs in the IOCRN, and K is the total number of parameters in the IOCRN. 
        The first M entries correspond to the multi-hot encoding of the reactions in the IOCRN, the next K entries correspond to the reaction parameters (0 if the reaction is not present), and the last K*p entries correspond to the multi-hot encoding of the parameters influenced by each input.
        - mode (str): The mode of the policy. Can be 'full' or 'partial'. Default is 'full'.
        The 'full' mode considers both the reaction structure and the reaction parameters.
        The 'partial' mode considers only the reaction parameters, assuming the reaction structure is given.
        - action (list): A list of dictionaries containing the actions in the batch. Each dictionary contains:
            - 'reaction index': The index of the reaction to be added (if mode is 'full').
            - 'parameters': A list of the reaction parameters (continuous and discrete, if applicable).
            - 'continuous parameters': A list of the continuous parameters (if applicable).
            - 'discrete parameters': A list of the discrete parameters (if applicable).
            If action is provided, the policy will compute only the log probabilities of the provided actions (used for computing the probability of an external action). Default is None.
        Returns:
        - actions (list): A list of dictionaries containing the actions in the batch. Each dictionary contains:
            - 'reaction index': The index of the sampled reaction (if mode is 'full').
            - 'parameters': A list of the sampled reaction parameters (continuous and discrete, if applicable).
            - 'continuous parameters': A list of the sampled continuous parameters (if applicable).
            - 'discrete parameters': A list of the sampled discrete parameters (if applicable).
        - log_probabilities (torch.Tensor): The log probability of the actions in the batch. Shape: (N,).
        - entropies (torch.Tensor): The entropy of the actions in the batch. Shape: (N,). """

        # Validate the input has no NaNs
        assert state.isnan().sum() == 0, "Input contains NaN values."

        # Encode the observation
        encoded = self.encoder(state) # shape: (N, deep_layer_size)
        
        # Run the reaction structure head to generate the structure of the next reaction while masking out already existing reactions in the IOCRN 
        entropies = 0
        log_probabilities = 0
        if mode == 'full':
            reaction_structure_logits = self.reaction_structure_head(encoded) # shape: (N, M)

            # Mask out already existing reactions in the IOCRN
            masked_reaction_structure_logits = reaction_structure_logits.masked_fill(state[:,:self.M].bool(), float('-inf')) # shape: (N, M)

            # Construct the categorical distribution over the library reactions and compute their entropies
            reaction_structure_distribution = Categorical(logits=masked_reaction_structure_logits) # batch of N categorical distributions, each over M categories
            entropies = reaction_structure_distribution.entropy() # shape: (N,)

            # Sample the reaction structure from the distribution and compute the log probabilities of the sampled reactions
            samples_reaction_idx = reaction_structure_distribution.sample() if action is None else torch.tensor([a['reaction index'] for a in action], requires_grad=False).to(self.device)  # shape: (N,)
            log_probabilities = reaction_structure_distribution.log_prob(samples_reaction_idx) # shape: (N,)

            # Create the one-hot encoding of the sampled reactions
            samples_reaction_hot = batch_multi_hot(samples_reaction_idx.unsqueeze(-1).cpu().numpy(), self.M, intensities=None, device=self.device) # shape: (N, M)
               
        # Create parameter masks corresponding to the sampled reactions
        continuous_parameter_mask_subset = self.continuous_parameter_mask[samples_reaction_idx] if self.continuous_parameter_mask is not None else None # Shape: (N, max_num_continuous_parameters) or None
        discrete_parameter_mask_subset = self.discrete_parameter_mask[samples_reaction_idx] if self.discrete_parameter_mask is not None else None  # Shape: (N, max_num_discrete_parameters) or None
        logit_mask_subset = self.logit_mask[samples_reaction_idx] if self.logit_mask is not None else None  # Shape: (N, total_num_categories_for_all_discrete_parameters) or None

        # Initialize the sampled parameters
        if action is None:
            samples_continuous_parameters = None
            samples_discrete_parameters = None
        else:
            samples_continuous_parameters = torch.tensor([a['continuous parameters'] for a in action], requires_grad=False).to(self.device) if self.continuous_parameter_generator is not None else None # shape: (N, max_num_continuous_parameters) or None
            samples_discrete_parameters = torch.tensor([a['discrete parameters'] for a in action], requires_grad=False).to(self.device) if self.discrete_parameter_generator is not None else None # shape: (N, max_num_discrete_parameters) or None

        # Concatenate the encoded IOCRN with the one-hot encoding of the sampled reaction structure to form the input to the continuous parameter generator
        x = torch.cat([encoded, samples_reaction_hot], dim=-1) # shape: (N, deep_layer_size + M)

        # Run the continuous and discrete parameter generators to generate the parameters of the sampled reactions
        parameter_types = ['continuous', 'discrete']
        for type in parameter_types:
            match type:
                case 'continuous':
                    # Skip if there are no continuous parameters to generate
                    if self.continuous_parameter_generator is None:
                        continue

                    # Generate the continuous parameters samples, their log probabilities and entropies of their respective distributions
                    samples_continuous_parameters, log_probs_continuous_parameters, entropies_continuous_parameters = self.continuous_parameter_generator(x, mask=continuous_parameter_mask_subset, samples=samples_continuous_parameters) # shapes: (N, max_num_continuous_parameters), (N,), (N,)

                    # Accumulate the log probabilities and entropies
                    entropies = entropies + entropies_continuous_parameters # shape: (N,)
                    log_probabilities = log_probabilities + log_probs_continuous_parameters # shape: (N,)

                    # Mask out the parameters that do not exist for the sampled reactions
                    if action is None:
                        samples_continuous_parameters = samples_continuous_parameters * continuous_parameter_mask_subset if continuous_parameter_mask_subset is not None else samples_continuous_parameters

                    # Concatenate the encoded IOCRN, the one-hot encoding of the sampled reaction structure and the sampled continuous parameters to form the input to the discrete parameter generator
                    x = torch.cat([x, samples_continuous_parameters], dim=-1)

                case 'discrete':
                    # Skip if there are no discrete parameters to generate
                    if self.discrete_parameter_generator is None:
                        continue

                    # Generate the discrete parameters samples, their log probabilities and entropies of their respective distributions
                    samples_discrete_parameters, log_probs_discrete_parameters, entropies_discrete_parameters = self.discrete_parameter_generator(x, logit_mask=logit_mask_subset, dimension_mask=discrete_parameter_mask_subset, samples=samples_discrete_parameters) # shapes: (N, max_num_discrete_parameters), (N,), (N,)
                    
                    # Accumulate the log probabilities and entropies
                    entropies = entropies + entropies_discrete_parameters # shape: (N,)
                    log_probabilities = log_probabilities + log_probs_discrete_parameters # shape: (N,)
                    
                    # Mask out the parameters that do not exist for the sampled reactions
                    samples_discrete_parameters = samples_discrete_parameters * discrete_parameter_mask_subset if discrete_parameter_mask_subset is not None else samples_discrete_parameters
                    
                    # Concatenate the encoded IOCRN, the one-hot encoding of the sampled reaction structure, the sampled continuous parameters and the sampled discrete parameters to form the input to the input influence head, if applicable
                    x = torch.cat([x, samples_discrete_parameters], dim=-1)

        # Run the input influence head, if applicable, to generate the input influence 
        if self.allow_input_influence is True:
            raise NotImplementedError("The input influence head is not implemented yet.")

        # If action is provided, return only the log probabilities (used for computing the probability of an external action)
        if action is not None:
            return log_probabilities
        
        # Otherwise, return the sampled actions, their log probabilities and entropies
        # Process the sampled parameters to return only the parameters that exist for the sampled reactions #TODO: This part should belong to the actuator not the agent as it is environment-specific? 
        if samples_continuous_parameters is not None:
            if continuous_parameter_mask_subset is not None:
                samples_continuous_parameters = [samples_continuous_parameters[i, continuous_parameter_mask_subset[i].bool()].cpu().numpy().tolist() for i in range(samples_continuous_parameters.shape[0])] # N-List of lists of continuous parameters, each sublist containing only the continuous parameters that exist for the sampled reaction
            else:
                samples_continuous_parameters = samples_continuous_parameters.cpu().numpy().tolist() # N-List of lists of continuous parameters, each sublist containing all the continuous parameters (no masking)
            
        if samples_discrete_parameters is not None:
            if discrete_parameter_mask_subset is not None:
                samples_discrete_parameters = [samples_discrete_parameters[i, discrete_parameter_mask_subset[i].bool()].cpu().numpy().tolist() for i in range(samples_discrete_parameters.shape[0])] # N-List of lists of discrete parameters, each sublist containing only the discrete parameters that exist for the sampled reaction
            else:
                samples_discrete_parameters = samples_discrete_parameters.cpu().numpy().tolist() # N-List of lists of discrete parameters, each sublist containing all the discrete parameters (no masking)

        # Collect the policy output, convert to numpy and move to CPU 
        samples_reaction_idx = samples_reaction_idx.cpu().numpy()

        if self.allow_input_influence is True:
            raise NotImplementedError("The input influence head is not implemented yet.")
            
        else:
            if mode == 'full': # structure and parameters
                if samples_discrete_parameters is None:
                    action = [
                        {
                            'reaction index': r_idx,
                            'parameters': param_continuous,
                            'continuous parameters': param_continuous, 
                            'discrete parameters': None
                        }
                        for r_idx, param_continuous in zip(samples_reaction_idx, samples_continuous_parameters)
                    ] # N dictionaries representing a batch of reactions, each containing the reaction index and continuous parameters
                else:
                    action = [
                        {
                            'reaction index': r_idx,
                            'parameters': np.concatenate([param_continuous, param_discrete]), 
                            'continuous parameters': param_continuous, 
                            'discrete parameters': param_discrete
                        }
                        for r_idx, param_continuous, param_discrete in zip(samples_reaction_idx, samples_continuous_parameters, samples_discrete_parameters)
                    ] # N dictionaries representing a batch of reactions, each containing the reaction index, continuous and discrete parameters

            elif mode == 'partial': # rates
                raise NotImplementedError("The 'partial' mode is not implemented yet.")

            else:
                raise ValueError(f"Unknown mode: {mode}. Supported modes are: 'full', 'partial'.")
            
        # Return the samples, log probability and entropy
        return action, log_probabilities, entropies