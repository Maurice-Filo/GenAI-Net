import torch
from RL4CRN.utils.ffnn import FFNN
from torch.distributions import Categorical, LogNormal
from RL4CRN.utils.utils import batch_multi_hot
from RL4CRN.policies.parameter_generator_from_distribution import ParameterGeneratorFromDistribution
import numpy as np

class AddReactionFromLibrary(torch.nn.Module):
    def __init__(self, reaction_library, num_inputs, encoder_attributes, deep_layer_size, structure_head_attributes, parameter_head_attributes, input_influence_head_attributes, 
                 continuous_distribution={"type": 'lognormal'}, discrete_distribution={"type": 'categorical', "categories": [torch.tensor([1, 2])]}, allow_input_influence=False, device=None):
        """ A policy that generates a reaction from a reaction library to an IOCRN in batch mode.
        The policy consists of a neural network with multiple heads that outputs the reaction structure, the reaction parameters (continuous and discrete, if applicable), and the input influence (if applicable). 
        The reaction structure is represented as a categorical distribution over the reactions in the library, from which a reaction is sampled.
        The reaction parameters are generated from specified distributions (e.g., log-normal for continuous parameters and categorical for discrete parameters) using separate heads.
        The input influence is represented as a categorical distribution over the inputs and no input, from which an input influence is sampled.
        Arguments:
        - reaction_library: an instance of ReactionLibrary containing the reactions to sample from.
        - num_inputs: number of inputs in the IOCRN (assumed to be the same for all IOCRNs in the batch).
        - encoder_attributes: a dictionary containing the attributes of the encoder neural network (hidden_size, num_layers).
        - deep_layer_size: size of the deep layer representation of the IOCRN.
        - structure_head_attributes: a dictionary containing the attributes of the reaction structure head neural network (hidden_size, num_layers).
        - parameter_head_attributes: a dictionary containing the attributes of the reaction parameter head neural network (hidden_size, num_layers).
        - input_influence_head_attributes: a dictionary containing the attributes of the input influence head neural network (hidden_size, num_layers).
        - continuous_distribution: a dictionary specifying the type of the continuous parameter distribution (default is log-normal).
        - discrete_distribution: a dictionary specifying the type and categories of the discrete parameter distribution (default is categorical).
        - allow_input_influence: if True, the policy will include an input influence head (default is False).
        - device: device to run the policy on (default is None, which uses CPU).
        """
        super().__init__()

        # Record the IOCRN and ReactionLibrary attributes
        self.reaction_library = reaction_library                        # An instance of ReactionLibrary containing the reactions to sample from    
        self.K = reaction_library.get_num_parameters()                  # Total number of parameters (continuous + discrete) across all reactions in the library
        self.M = len(reaction_library)                                  # Total number of reactions in the library
        self.p = num_inputs                                             # Number of inputs in the IOCRN

        # Record the neural network attributes
        self.encoder_attributes = encoder_attributes
        self.deep_layer_size = deep_layer_size
        self.structure_head_attributes = structure_head_attributes
        self.rate_head_attributes = parameter_head_attributes
        self.input_influence_head_attributes = input_influence_head_attributes
        self.allow_input_influence = allow_input_influence
        self.device = device

        # Record the distribution attributes
        self.continuous_distribution = continuous_distribution
        self.discrete_distribution = discrete_distribution

        # Get the masks from the reaction library and tensorize them
        self.continuous_parameter_mask = reaction_library.get_parameter_mask(mode='continuous')
        self.continuous_parameter_mask = torch.tensor(self.continuous_parameter_mask).to(self.device) if self.continuous_parameter_mask is not None else None # Shape: (M, max_num_continuous_parameters)
        self.discrete_parameter_mask = reaction_library.get_parameter_mask(mode='discrete')
        self.discrete_parameter_mask = torch.tensor(self.discrete_parameter_mask).to(self.device) if self.discrete_parameter_mask is not None else None  # Shape: (M, max_num_discrete_parameters)
        self.logit_mask = reaction_library.get_logit_mask()

        # Define the encoder that encodes the IOCRN observation into a deep layer representation
        self.encoder = FFNN(input_size=self.M + (self.p + 1) * self.K, output_size=deep_layer_size, hidden_size=encoder_attributes["hidden_size"], num_layers=encoder_attributes["num_layers"]).to(device=device)
        
        # Define the reaction structure head that reads the deep layer representation to output the logits for the reaction structure
        self.reaction_structure_head = FFNN(input_size=deep_layer_size, output_size=self.M, hidden_size=structure_head_attributes["hidden_size"], num_layers=structure_head_attributes["num_layers"]).to(device=device)
        
        # Define the continuous parameter head that reads the deep layer representation and the reaction structure to deliver to the continuous parameter generator
        if self.continuous_parameter_mask is not None:
            self.continuous_parameter_head = FFNN(input_size=deep_layer_size + self.M, output_size=parameter_head_attributes["hidden_size"], hidden_size=parameter_head_attributes["hidden_size"], num_layers=parameter_head_attributes["num_layers"]).to(device=device) 
            self.continuous_distribution["dim"] = self.continuous_parameter_mask.shape[1] # Set the dimension of the continuous distribution to the maximum number of continuous parameters across all reactions in the library
            self.continuous_parameter_generator = ParameterGeneratorFromDistribution(distribution=self.continuous_distribution, backbone=self.continuous_parameter_head, device=self.device).to(device=self.device)   
        else:
            self.continuous_parameter_head = None   
            self.continuous_parameter_generator = None

        # Define the discrete parameter head that reads the deep layer representation, the reaction structure, and the continuous parameters to deliver to the discrete parameter generator
        if self.discrete_parameter_mask is not None:
            input_size = deep_layer_size + self.M + (self.continuous_parameter_mask.shape[1] if self.continuous_parameter_mask is not None else 0)
            self.discrete_parameter_head = FFNN(input_size=input_size, output_size=parameter_head_attributes["hidden_size"], hidden_size=parameter_head_attributes["hidden_size"], num_layers=parameter_head_attributes["num_layers"]).to(device=device) 
            self.discrete_distribution["dim"] = self.discrete_parameter_mask.shape[1]
            self.discrete_distribution["categories"] = self.discrete_distribution["dim"] * self.discrete_distribution["categories"] # Make the discrete distribution categories the same for all discrete parameters
            self.discrete_parameter_generator = ParameterGeneratorFromDistribution(distribution=self.discrete_distribution, backbone=self.discrete_parameter_head, device=self.device).to(device=self.device)
        else:
            self.discrete_parameter_head = None   
            self.discrete_parameter_generator = None
            
        # Define the input influence head that reads the deep layer representation, the reaction structure and the reaction rate to output the logits for the input influence
        if allow_input_influence is True:
            raise NotImplementedError("The input influence head is not implemented yet.")
        else:
            self.input_influence_head = None
                  
    def forward(self, x, mode='full'):
        """ Generates an action (reaction structure, parameters and input influence) given the observation of the state received from the observer.
        Args:
        - x (torch.Tensor): The observation (state) of the IOCRN. Shape: (N, M + (p+1)*K)), where N is the batch size, M is the number of reactions in the library, p is the number of inputs in the IOCRN, and K is the total number of parameters in the IOCRN. 
        The first M entries correspond to the multi-hot encoding of the reactions in the IOCRN, the next K entries correspond to the reaction parameters (0 if the reaction is not present), and the last K*p entries correspond to the multi-hot encoding of the parameters influenced by each input.
        - mode (str): The mode of the policy. Can be 'full' or 'partial'. Default is 'full'.
        The 'full' mode considers both the reaction structure and the reaction parameters.
        The 'partial' mode considers only the reaction parameters, assuming the reaction structure is given.
        Returns:
        - actions (list): A list of dictionaries containing the actions. Each dictionary contains:
            - 'reaction index': The index of the sampled reaction (if mode is 'full').
            - 'parameters': A list of the sampled reaction parameters (continuous and discrete, if applicable).
        - log_probabilities (torch.Tensor): The log probability of the actions in the batch. Shape: (N,).
        - entropies (torch.Tensor): The entropy of the actions in the batch. Shape: (N,).
        """
        # Validate the input has no NaNs
        assert x.isnan().sum() == 0, "Input contains NaN values."

        # Encode the observation
        encoded = self.encoder(x) # shape: (N, deep_layer_size)
        
        # Run the reaction structure head to generate the structure of the next reaction while masking out already existing reactions in the IOCRN 
        entropies = 0
        log_probabilities = 0
        if mode == 'full':
            reaction_structure_logits = self.reaction_structure_head(encoded) # shape: (N, M)

            # Mask out already existing reactions in the IOCRN
            masked_reaction_structure_logits = reaction_structure_logits.masked_fill(x[:,:self.M].bool(), float('-inf')) # shape: (N, M)
            # masked_reaction_structure_logits = reaction_structure_logits.exp() * (1 - x[:, :self.M]) + 1e-20
            # masked_reaction_structure_logits = torch.log(masked_reaction_structure_logits)
            reaction_structure_distribution = Categorical(logits=masked_reaction_structure_logits) # batch of N categorical distributions, each over M categories
            samples_reaction_idx = reaction_structure_distribution.sample() # shape: (N,)
            samples_reaction_hot = batch_multi_hot(samples_reaction_idx.unsqueeze(-1).cpu().numpy(), self.M, intensities=None, device=self.device) # shape: (N, M)
            entropies = reaction_structure_distribution.entropy() # shape: (N,)
            log_probabilities = reaction_structure_distribution.log_prob(samples_reaction_idx) # shape: (N,)

        # Create masks corresponding to the sampled reactions
        self.continuous_parameter_mask = None #TODO Remove this line after testing
        continuous_parameter_mask_subset = self.continuous_parameter_mask[samples_reaction_idx] if self.continuous_parameter_mask is not None else None # Shape: (N, max_num_continuous_parameters) or None
        discrete_parameter_mask_subset = self.discrete_parameter_mask[samples_reaction_idx] if self.discrete_parameter_mask is not None else None  # Shape: (N, max_num_discrete_parameters) or None
        logit_mask_subset = self.logit_mask[samples_reaction_idx] if self.logit_mask is not None else None  # Shape: (N, total_num_categories_for_all_discrete_parameters) or None

        # Initialize the sampled parameters
        samples_continuous_parameters = None
        samples_discrete_parameters = None

        # Run the continuous and discrete parameter heads and generators to generate the parameters of the sampled reactions
        parameter_types = ['continuous', 'discrete']
        x = torch.cat([encoded, samples_reaction_hot], dim=-1) # shape: (N, deep_layer_size + M)
        for type in parameter_types:
            match type:
                case 'continuous':
                    if self.continuous_parameter_generator is None:
                        continue
                    samples_continuous_parameters, log_probs_continuous_parameters, entropies_continuous_parameters = self.continuous_parameter_generator(x, mask=continuous_parameter_mask_subset) # shapes: (N, max_num_continuous_parameters), (N,), (N,)
                    entropies = entropies + entropies_continuous_parameters # shape: (N,)
                    log_probabilities = log_probabilities + log_probs_continuous_parameters # shape: (N,)
                    # Mask out the parameters that do not exist for the sampled reactions
                    samples_continuous_parameters = samples_continuous_parameters * continuous_parameter_mask_subset if continuous_parameter_mask_subset is not None else samples_continuous_parameters
                    x = torch.cat([x, samples_continuous_parameters], dim=-1)

                case 'discrete':
                    if self.discrete_parameter_generator is None:
                        continue
                    samples_discrete_parameters, log_probs_discrete_parameters, entropies_discrete_parameters = self.discrete_parameter_generator(x, logit_mask=logit_mask_subset, dimension_mask=discrete_parameter_mask_subset) # shapes: (N, max_num_discrete_parameters), (N,), (N,)
                    entropies = entropies + entropies_discrete_parameters # shape: (N,)
                    log_probabilities = log_probabilities + log_probs_discrete_parameters # shape: (N,)
                    # Mask out the parameters that do not exist for the sampled reactions
                    samples_discrete_parameters = samples_discrete_parameters * discrete_parameter_mask_subset if discrete_parameter_mask_subset is not None else samples_discrete_parameters
                    x = torch.cat([x, samples_discrete_parameters], dim=-1)

        # Run the input influence head, if applicable, to generate the input influence 
        if self.allow_input_influence is True:
            raise NotImplementedError("The input influence head is not implemented yet.")

        # Process the sampled parameters to return only the parameters that exist for the sampled reactions
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
                    actions = [
                        {
                            'reaction index': r_idx,
                            'parameters': param_continuous,
                        }
                        for r_idx, param_continuous in zip(samples_reaction_idx, samples_continuous_parameters)
                    ] # N dictionaries representing a batch of reactions, each containing the reaction index and continuous parameters
                else:
                    actions = [
                        {
                            'reaction index': r_idx,
                            'parameters': np.concatenate([param_continuous, param_discrete])
                        }
                        for r_idx, param_continuous, param_discrete in zip(samples_reaction_idx, samples_continuous_parameters, samples_discrete_parameters)
                    ] # N dictionaries representing a batch of reactions, each containing the reaction index, continuous and discrete parameters

            elif mode == 'partial': # rates
                raise NotImplementedError("The 'partial' mode is not implemented yet.")

            else:
                raise ValueError(f"Unknown mode: {mode}. Supported modes are: 'full', 'partial'.")
            
        # Return the samples, log probability and entropy
        return actions, log_probabilities, entropies
    
    def compute_action_probability(self, states, actions, mode='full'): #TODO Work on this
        """
        Compute the log probability of the actions in the batch given the observations (states).
        Args:
            - states (tuple): A tuple containing the following elements:
                - reactions_indices_batch: A numpy array representing the reactions indices in the batch of IOCRNs. Shape: (N, m).
                - rate_constants_batch: A numpy array representing the reaction rate constants in the batch of IOCRNs. Shape: (N, m).
                - reactions_indices_influenced_by_inputs_batch: A list of p numpy arrays, each containing the influenced reactions for a specific input. 
                Each numpy array is associated with a specific input and has shape (N, #), where # is the maximum number of reactions in any CRN in the batch influenced by this input.
            - actions (list): A list of dictionaries containing the actions. Each dictionary contains:
                - 'reaction index': The index of the sampled reaction (if mode is 'full').
                - 'rate constant': The sampled reaction rate.
                - 'input influence index': The index of the input influence (if allow_input_influence is True).
            - mode (str): The mode of the policy. Can be 'full' or 'partial'. Default is 'full'.
            The 'full' mode considers both the reaction structure and the reaction rates, and optionally the input influence.
            The 'partial' mode considers only the reaction rates and optionally the input influence.
        Returns:
            - log_probabilities (torch.Tensor): The log probability of the actions in the batch. Shape: (N,).
        """
        reactions_indices_batch, rate_constants_batch, reactions_indices_influenced_by_inputs_batch = states

        # Compute the multi-hot encoding of the observations
        reactions_indices_batch_hot, rates_batch_hot = batch_multi_hot(reactions_indices_batch, self.M, rate_constants_batch, device=self.device)
        reactions_indices_influenced_by_inputs_batch_hot = [batch_multi_hot(reactions_indices_influenced_by_inputs_batch[i], self.M, device=self.device) for i in range(self.p)]

        # Construct the encoder input
        x_structure = reactions_indices_batch_hot # shape: (N, M)
        x_rate = rates_batch_hot # shape: (N, M)
        x_input_influence = torch.cat(reactions_indices_influenced_by_inputs_batch_hot, dim=1) # shape: (N, M*p)

        # Encode the policy input
        x = torch.cat([x_structure, x_rate, x_input_influence], dim=1) # shape: (N, M*(p+2))
        encoded = self.encoder(x) # shape: (N, deep_layer_size)

        # Run the reaction structure head to obtain the probability distribution over the reaction structure and get the log probability of the reaction structures in the batch of actions
        log_probabilities = 0
        if mode == 'full':
            reaction_structure_logits = self.reaction_structure_head(encoded) # shape: (N, M)
            masked_reaction_structure_logits = reaction_structure_logits.masked_fill(x_structure.bool(), float('-inf')) # shape: (N, M)
            reaction_structure_distribution = Categorical(logits=masked_reaction_structure_logits) # batch of N categorical distributions, each over M categories
            samples_reaction_idx = torch.tensor([a['reaction index'] for a in actions], requires_grad=False).to(self.device) # shape: (N,)
            samples_reaction_hot = batch_multi_hot(samples_reaction_idx.unsqueeze(-1).cpu().numpy(), self.M, intensities=None, device=self.device) # shape: (N, M)
            log_probabilities = reaction_structure_distribution.log_prob(samples_reaction_idx) # shape: (N,)

        # Run the reaction rate head to obtain the probability distribution over the reaction rates and get the log probability of the rate constants in the batch of actions
        x1 = torch.cat([encoded, samples_reaction_hot], dim=-1) # shape: (N, deep_layer_size + M)
        continuous_distribution_parameters = self.reaction_rate_head(x1) # shape: (N, number of parameters for the continuous distribution)
        match self.continuous_distribution:
            case 'lognormal': # Parameters are mean and log(stddev)
                continuous_distribution_parameters = torch.nn.functional.softplus(continuous_distribution_parameters)
                mu_log_normal, sigma_log_normal = continuous_distribution_parameters[:, 0], continuous_distribution_parameters[:, 1] # shape: (N,)
                mu_normal = torch.log(mu_log_normal**2 / torch.sqrt(mu_log_normal**2 + sigma_log_normal**2)) 
                sigma_normal = torch.log(1 + sigma_log_normal**2 / mu_log_normal**2)
                reaction_rate_distribution = LogNormal(mu_normal, sigma_normal) # batch of N LogNormal distributions
                samples_reaction_rate = torch.tensor([a['rate constant'] for a in actions], requires_grad=False).to(self.device) # shape: (N,)
                log_probabilities = log_probabilities + reaction_rate_distribution.log_prob(samples_reaction_rate) # shape: (N,)
            case _:
                raise ValueError(f"Unknown continuous distribution: {self.continuous_distribution}. Supported distributions are: 'lognormal'.")
            
        # Run the input influence head, if applicable, to obtain the probability distribution over the input influence and get the log probability of the input influence in the batch of actions
        if self.allow_input_influence is True:
            x2 = torch.cat([x1, samples_reaction_rate.unsqueeze(-1)], dim=-1) # shape: (N, deep_layer_size + M + 1)
            input_influence_logits = self.input_influence_head(x2) # shape: (N, p+1)
            input_influence_distribution = Categorical(logits=input_influence_logits) # batch of N categorical distributions, each over p+1 categories
            samples_input_influence_idx = torch.tensor([a['input influence index'] for a in actions], requires_grad=False).to(self.device) # shape: (N,)
            log_probabilities = log_probabilities + input_influence_distribution.log_prob(samples_input_influence_idx) # shape: (N,)

        # Return the log probability of the actions 
        return log_probabilities