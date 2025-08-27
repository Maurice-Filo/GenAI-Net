import torch
from RL4CRN.utils.ffnn import FFNN
from torch.distributions import Categorical, LogNormal
from RL4CRN.utils.utils import batch_multi_hot

class AddReactionFromLibrary(torch.nn.Module):
    """
    A class representing a policy to sample reactions from a library, given an observation of the IOCRN. The policy supports batching.
    The policy input is a representation of the current batch of IOCRNs (observation from environment) which is organized as follows:
        - M inputs for the multi-hot encoding of the reaction structure
        - M inputs for the intensity multi-hot encoding of the reaction rates
        - M*p inputs for the multi-hot encoding of the input influence
    The policy output is a representation of a batch of reactions which is organized as follows:
        - samples: A list of dictionaries, each containing:
            - 'reaction index': The index of the sampled reaction (if mode is 'full').
            - 'rate constant': The sampled reaction rate.
            - 'input influence index': The index of the input influence (if allow_input_influence is True).

    This policy is composed of an encoder and three heads:
        - An encoder that encodes the IOCRN observation into a deep layer representation.
        - A reaction structure head that reads the deep layer representation to output the logits for the reaction structure, which is a one-hot encoding of the reactions. 
        - A reaction rate head that reads the deep layer representation and the reaction structure to output the parameters of a continuous distribution for the reaction rates.
        - An input influence head that reads the deep layer representation, the reaction structure and the reaction rate to output the logits for the input influence, which is a one-hot encoding of the inputs.

    The policy supports two modes:
        - 'full': samples both the reaction structure and the reaction rates, and optionally the input influence.
        - 'partial': samples only the reaction rates and optionally the input influence.

    The policy can be used to generate reactions with or without input influence, depending on the allow_input_influence parameter.
    The policy uses a continuous distribution for the reaction rates, which can be specified by the continuous_distribution parameter.
    """
    def __init__(self, M, p, encoder_attributes, deep_layer_size, structure_head_attributes, rate_head_attributes, input_influence_head_attributes, continuous_distribution='lognormal', allow_input_influence=False, device=None):
        """
        Initialize the AddReactionFromLibrary policy.
        Args:
            - M (int): The number of all possible reactions in the library.
            - p (int): The number of inputs to the IOCRN.
            - encoder_attributes (dict): Attributes for the encoder, including hidden size and number of layers.
            - deep_layer_size (int): The size of the deep layer representing an embedding for the IOCRN.
            - structure_head_attributes (dict): Attributes for the reaction structure head, including hidden size and number of layers.
            - rate_head_attributes (dict): Attributes for the reaction rate head, including hidden size and number of layers.
            - input_influence_head_attributes (dict): Attributes for the input influence head, including hidden size and number of layers.
            - continuous_distribution (str): The type of continuous distribution to use for the reaction rates. Default is 'lognormal'.
            - allow_input_influence (bool): Whether to allow input influence in the generated reactions by the policy. Default is False.
            - device (torch.device): The device to run the policy on. Default is None, which uses the current device.
        The encoder and three heads attributes are dictionaries with the following keys:
            - hidden_size (int): The size of the hidden layers in the feedforward neural network.
            - num_layers (int): The number of hidden layers in the feedforward neural network.
        """
        super().__init__()
        self.M = M
        self.p = p
        self.encoder_attributes = encoder_attributes
        self.deep_layer_size = deep_layer_size
        self.structure_head_attributes = structure_head_attributes
        self.rate_head_attributes = rate_head_attributes
        self.input_influence_head_attributes = input_influence_head_attributes
        self.continuous_distribution = continuous_distribution
        self.allow_input_influence = allow_input_influence
        self.device = device

        # Define the encoder that encodes the IOCRN observation into a deep layer representation
        self.encoder = FFNN(input_size=M * (p + 2), output_size=deep_layer_size, hidden_size=encoder_attributes["hidden_size"], num_layers=encoder_attributes["num_layers"]).to(device=device)
        
        # Define the reaction structure head that reads the deep layer representation to output the logits for the reaction structure
        self.reaction_structure_head = FFNN(input_size=deep_layer_size, output_size=M, hidden_size=structure_head_attributes["hidden_size"], num_layers=structure_head_attributes["num_layers"]).to(device=device)
        
        # Define the reaction rate head that reads the deep layer representation and the reaction structure to output the parameters of a continuous distribution for the reaction rates
        match continuous_distribution:
            case 'lognormal': # Parameters are mean and log(stddev) 
                self.reaction_rate_head = FFNN(input_size=deep_layer_size + M, output_size=2, hidden_size=rate_head_attributes["hidden_size"], num_layers=rate_head_attributes["num_layers"]).to(device=device)         
            case _:
                raise ValueError(f"Unknown continuous distribution: {continuous_distribution}. Supported distributions are: 'lognormal'.")
        
        # Define the input influence head that reads the deep layer representation, the reaction structure and the reaction rate to output the logits for the input influence
        if allow_input_influence is True:
            self.input_influence_head = FFNN(input_size=deep_layer_size + M + 1, output_size=p+1, hidden_size=input_influence_head_attributes["hidden_size"], num_layers=input_influence_head_attributes["num_layers"]).to(device=device)
        else:
            self.input_influence_head = None
                  
    def forward(self, states, mode='full'):
        """
        Forward pass of the AddReactionFromLibrary policy.
        Args:
            - states (tuple): A tuple containing the following elements:
                - reactions_indices_batch: A numpy array representing the reactions indices in the batch of IOCRNs. Shape: (N, m).
                - rate_constants_batch: A numpy array representing the reaction rate constants in the batch of IOCRNs. Shape: (N, m).
                - reactions_indices_influenced_by_inputs_batch: A list of p numpy arrays, each containing the influenced reactions for a specific input. 
                Each numpy array is associated with a specific input and has shape (N, #), where # is the maximum number of reactions in any CRN in the batch influenced by this input.
            - mode (str): The mode of the policy. Can be 'full' or 'partial'. Default is 'full'.
            The 'full' mode samples both the reaction structure and the reaction rates, and optionally the input influence.
            The 'partial' mode samples only the reaction rates and optionally the input influence.
        Returns:
            - actions (list): A list of dictionaries containing the sampled reactions. Each dictionary contains:
                - 'reaction index': The index of the sampled reaction (if mode is 'full').
                - 'rate constant': The sampled reaction rate.
                - 'input influence index': The index of the input influence (if allow_input_influence is True).
            - log_probabilities (torch.Tensor): The log probability of the sampled reactions.
            - entropies (torch.Tensor): The entropy of the sampled reactions.
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
        
        # Run the reaction structure head to generate the structure of the next reaction while masking out already existing reactions in the IOCRN 
        entropies = 0
        log_probabilities = 0
        if mode == 'full':
            reaction_structure_logits = self.reaction_structure_head(encoded) # shape: (N, M)
            masked_reaction_structure_logits = reaction_structure_logits.masked_fill(x_structure.bool(), float('-inf')) # shape: (N, M)
            reaction_structure_distribution = Categorical(logits=masked_reaction_structure_logits) # batch of N categorical distributions, each over M categories
            samples_reaction_idx = reaction_structure_distribution.sample() # shape: (N,)
            samples_reaction_hot = batch_multi_hot(samples_reaction_idx.unsqueeze(-1).cpu().numpy(), self.M, intensities=None, device=self.device) # shape: (N, M)
            entropies = reaction_structure_distribution.entropy() # shape: (N,)
            log_probabilities = reaction_structure_distribution.log_prob(samples_reaction_idx) # shape: (N,)
        
        # Run the reaction rate head to generate the reaction rate
        x1 = torch.cat([encoded, samples_reaction_hot], dim=-1) # shape: (N, deep_layer_size + M)
        continuous_distribution_parameters = self.reaction_rate_head(x1) # shape: (N, number of parameters for the continuous distribution)
        match self.continuous_distribution:
            case 'lognormal': # Parameters are mean and log(stddev)
                # log_mu, log_sigma = continuous_distribution_parameters[:, 0], continuous_distribution_parameters[:, 1]
                # mu = torch.exp(log_mu)
                # sigma = torch.exp(log_sigma)
                # reaction_rate_distribution = LogNormal(mu, sigma)
                continuous_distribution_parameters = torch.nn.functional.softplus(continuous_distribution_parameters)
                mu_log_normal, sigma_log_normal = continuous_distribution_parameters[:, 0], continuous_distribution_parameters[:, 1] # shape: (N,)
                mu_normal = torch.log(mu_log_normal**2 / torch.sqrt(mu_log_normal**2 + sigma_log_normal**2)) 
                sigma_normal = torch.log(1 + sigma_log_normal**2 / mu_log_normal**2)
                reaction_rate_distribution = LogNormal(mu_normal, sigma_normal) # batch of N LogNormal distributions
                samples_reaction_rate = reaction_rate_distribution.sample() # shape: (N,)
                entropies = entropies + reaction_rate_distribution.entropy() # shape: (N,)
                log_probabilities = log_probabilities + reaction_rate_distribution.log_prob(samples_reaction_rate) # shape: (N,)
            case _:
                raise ValueError(f"Unknown continuous distribution: {self.continuous_distribution}. Supported distributions are: 'lognormal'.")

        # Run the input influence head, if applicable, to generate the input influence 
        if self.allow_input_influence is True:
            x2 = torch.cat([x1, samples_reaction_rate.unsqueeze(-1)], dim=-1) # shape: (N, deep_layer_size + M + 1)
            input_influence_logits = self.input_influence_head(x2) # shape: (N, p+1)
            input_influence_distribution = Categorical(logits=input_influence_logits) # batch of N categorical distributions, each over p+1 categories
            samples_input_influence_idx = input_influence_distribution.sample() # shape: (N,)
            entropies = entropies + input_influence_distribution.entropy() # shape: (N,)
            log_probabilities = log_probabilities + input_influence_distribution.log_prob(samples_input_influence_idx) # shape: (N,)

        # Collect the policy output, convert to numpy and move to CPU
        samples_reaction_rate = samples_reaction_rate.cpu().numpy()
        if self.allow_input_influence is True:
            samples_input_influence_idx = samples_input_influence_idx.cpu().numpy()
            if mode == 'full': # structure and rates
                samples_reaction_idx = samples_reaction_idx.cpu().numpy()
                actions = [
                    {
                        'reaction index': r_idx,
                        'rate constant': r_rate,
                        'input influence index': infl_idx
                    }
                    for r_idx, r_rate, infl_idx in zip(samples_reaction_idx, samples_reaction_rate, samples_input_influence_idx)
                ] # N dictionaries representing a batch of reactions, each containing the reaction index, rate constant and input influence index

            elif mode == 'partial': #rates
                actions = [
                    {
                        'rate constant': r_rate,
                        'input influence index': infl_idx
                    }
                    for r_rate, infl_idx in zip(samples_reaction_rate, samples_input_influence_idx)
                ] # N dictionaries representing a batch of reactions, each containing the rate constant and input influence index
                
            else:
                raise ValueError(f"Unknown mode: {mode}. Supported modes are: 'full', 'partial'.")
            
        else:
            if mode == 'full': # structure and rates
                samples_reaction_idx = samples_reaction_idx.cpu().numpy()
                actions = [
                    {
                        'reaction index': r_idx,
                        'rate constant': r_rate,
                    }
                    for r_idx, r_rate in zip(samples_reaction_idx, samples_reaction_rate)
                ] # N dictionaries representing a batch of reactions, each containing the reaction index and rate constant

            elif mode == 'partial': # rates
                actions = [
                    {
                        'rate constant': r_rate,
                    }
                    for r_rate in samples_reaction_rate
                ]  # N dictionaries representing a batch of reactions, each containing the rate constant

            else:
                raise ValueError(f"Unknown mode: {mode}. Supported modes are: 'full', 'partial'.")
            
        # Return the samples, log probability and entropy
        return actions, log_probabilities, entropies
    
    def compute_action_probability(self, states, actions, mode='full'):
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