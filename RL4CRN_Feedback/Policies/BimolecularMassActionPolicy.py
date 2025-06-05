import torch
from RL4CRN_Feedback.Policies.FFNN import FFNN
from torch.distributions import Categorical, LogNormal
from RL4CRN_Feedback.Utils.Utils import batch_multi_hot

class BimolecularMassActionPolicy(torch.nn.Module):
    def __init__(self, num_possible_reactions, num_inputs, encoder_attributes, hidden_size, structure_decoder_attributes, rate_decoder_attributes, input_influence_decoder_attributes, continuous_distribution='lognormal', allow_input_influence=False, device=None):
        super().__init__()
        self.num_possible_reactions = num_possible_reactions
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self.encoder_attributes = encoder_attributes
        self.structure_decoder_attributes = structure_decoder_attributes
        self.rate_decoder_attributes = rate_decoder_attributes
        self.input_influence_decoder_attributes = input_influence_decoder_attributes
        self.continuous_distribution = continuous_distribution
        self.allow_input_influence = allow_input_influence
        self.device = device

        # Define the encoder
        self.encoder = FFNN(input_size=num_possible_reactions * (num_inputs + 2), output_size=hidden_size, hidden_size=encoder_attributes["hidden_size"], num_layers=encoder_attributes["num_layers"]).to(device=device)
        
        # Define the reaction structure head
        self.reaction_structure_head = FFNN(input_size=hidden_size, output_size=num_possible_reactions, hidden_size=structure_decoder_attributes["hidden_size"], num_layers=structure_decoder_attributes["num_layers"]).to(device=device)
        
        # Define the reaction rate head
        match continuous_distribution:
            case 'lognormal':
                self.reaction_rate_head = FFNN(input_size=hidden_size + num_possible_reactions, output_size=2, hidden_size=rate_decoder_attributes["hidden_size"], num_layers=rate_decoder_attributes["num_layers"]).to(device=device)         
            case _:
                raise ValueError(f"Unknown continuous distribution: {continuous_distribution}. Supported distributions are: 'lognormal'.")
        
        # Define the input influence head
        if allow_input_influence is True:
            self.input_influence_head = FFNN(input_size=hidden_size + num_possible_reactions + 1, output_size=num_inputs+1, hidden_size=input_influence_decoder_attributes["hidden_size"], num_layers=input_influence_decoder_attributes["num_layers"]).to(device=device)
        else:
            self.input_influence_head = None

    def verify(self, observation_batch, action, mode='full'):
        """
        compute the probability of an action given a state
        :param state: the state of the environment
        :param action: the action to compute the probability for
        :return: the probability of the action given the state
        """
        # TODO implement this function
        # print(self.net(state).shape)
        # print(action.shape)
        # print(self.net(state)[action].shape)
        # o = self.net(state)
        # return o.gather(2, action.unsqueeze(2)).squeeze(2)

        # simulate a forward pass

        reactions_indices_batch, parameters_batch, reactions_indices_influenced_by_inputs_batch = observation_batch
        # Compute the multi-hot encoding of the observations
        reactions_indices_batch_hot, rates_batch_hot = batch_multi_hot(reactions_indices_batch, self.num_possible_reactions, parameters_batch, device=self.device)
        reactions_indices_influenced_by_inputs_batch_hot = [batch_multi_hot(reactions_indices_influenced_by_inputs_batch[i], self.num_possible_reactions, device=self.device) for i in range(self.num_inputs)]
        # Construct the input of the neural network
        x_structure = reactions_indices_batch_hot.to(dtype=torch.float32)
        x_rate = rates_batch_hot.to(dtype=torch.float32)
        x_input_influence = torch.cat(reactions_indices_influenced_by_inputs_batch_hot, dim=1).to(dtype=torch.float32)
        # Encode the input of the neural network
        x = torch.cat([x_structure, x_rate, x_input_influence], dim=1)
        encoded = self.encoder(x)


        entropy = 0
        log_probability = 0
        if mode == 'full':

            ### REACTION STRUCTURE ###

            reaction_structure_logits = self.reaction_structure_head(encoded)
            masked_reaction_structure_logits = reaction_structure_logits.masked_fill(x_structure.bool(), float('-inf'))
            reaction_structure_distribution = Categorical(logits=masked_reaction_structure_logits)
            
            # get the reaction index from the action
            if self.allow_input_influence is True:
                if mode == 'full': # structure and rates
                    
                    reaction_index = torch.tensor([a['reaction index'] for a in action], requires_grad=False).to(self.device)
                elif mode == 'partial': #rates
                    reaction_index = None
                else:
                    raise ValueError(f"Unknown mode: {mode}. Supported modes are: 'full', 'partial'.")
            else:
                if mode == 'full': # structure and rates
                    
                    reaction_index = torch.tensor([a['reaction index'] for a in action], requires_grad=False).to(self.device)
                elif mode == 'partial': # rates
                    reaction_index = None
                else:
                    raise ValueError(f"Unknown mode: {mode}. Supported modes are: 'full', 'partial'.")
            
            # get the probability of the action
            log_probability_of_reaction = reaction_structure_distribution.log_prob(reaction_index)
            reaction_index_hot = batch_multi_hot(reaction_index.unsqueeze(-1).cpu().numpy(), self.num_possible_reactions, intensities=None, device=self.device)

            ## REACTION RATE ##
        
            x1 = torch.cat([encoded, reaction_index_hot], dim=-1)
            continuous_distribution_parameters = self.reaction_rate_head(x1)
            match self.continuous_distribution:
                case 'lognormal': # Parameters are mean and log(stddev)
                    # log_mu, log_sigma = continuous_distribution_parameters[:, 0], continuous_distribution_parameters[:, 1]
                    # mu = torch.exp(log_mu)
                    # sigma = torch.exp(log_sigma)
                    # reaction_rate_distribution = LogNormal(mu, sigma)
                    continuous_distribution_parameters = torch.nn.functional.softplus(continuous_distribution_parameters)
                    mu_log_normal, sigma_log_normal = continuous_distribution_parameters[:, 0], continuous_distribution_parameters[:, 1]
                    mu_normal = torch.log(mu_log_normal**2 / torch.sqrt(mu_log_normal**2 + sigma_log_normal**2)) 
                    sigma_normal = torch.log(1 + sigma_log_normal**2 / mu_log_normal**2)
                    reaction_rate_distribution = LogNormal(mu_normal, sigma_normal)
                    # get the reaction rate from the action
                    action_rate = torch.tensor([a['rate constant'] for a in action], requires_grad=False).to(self.device)
                    # get the probability of the action
                    log_probability_of_rate = reaction_rate_distribution.log_prob(action_rate)
                case _:
                    raise ValueError(f"Unknown continuous distribution: {self.continuous_distribution}. Supported distributions are: 'lognormal'.")


            ### INPUT INFLUENCE ###

            # Decode the input influence if applicable
            if self.allow_input_influence is True:
                x2 = torch.cat([x1, action_rate.unsqueeze(-1)], dim=-1)
                input_influence_logits = self.input_influence_head(x2)
                input_influence_distribution = Categorical(logits=input_influence_logits)
                # get the input influence index from the action
                if mode == 'full': # structure and rates
                    input_influence_index = torch.tensor([a['input influence index'] for a in action], requires_grad=False).to(self.device)
                else:
                    raise ValueError(f"Unknown mode: {mode}. Supported modes are: 'full', and NOT 'partial'.")
                # get the probability of the action
                log_probability_of_input_influence = input_influence_distribution.log_prob(input_influence_index)
                return log_probability_of_reaction + log_probability_of_rate + log_probability_of_input_influence
            else:
                return log_probability_of_reaction + log_probability_of_rate
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported modes are: 'full', and NOT 'partial'.")
            

        
    def forward(self, observation_batch, mode='full'):
        reactions_indices_batch, parameters_batch, reactions_indices_influenced_by_inputs_batch = observation_batch
        # Compute the multi-hot encoding of the observations
        reactions_indices_batch_hot, rates_batch_hot = batch_multi_hot(reactions_indices_batch, self.num_possible_reactions, parameters_batch, device=self.device)
        reactions_indices_influenced_by_inputs_batch_hot = [batch_multi_hot(reactions_indices_influenced_by_inputs_batch[i], self.num_possible_reactions, device=self.device) for i in range(self.num_inputs)]
        # Construct the input of the neural network
        x_structure = reactions_indices_batch_hot.to(dtype=torch.float32)
        x_rate = rates_batch_hot.to(dtype=torch.float32)
        x_input_influence = torch.cat(reactions_indices_influenced_by_inputs_batch_hot, dim=1).to(dtype=torch.float32)
        # Encode the input of the neural network
        x = torch.cat([x_structure, x_rate, x_input_influence], dim=1)
        encoded = self.encoder(x)
        
        # Decode the reaction structure and mask out already existing reactions
        entropy = 0
        log_probability = 0
        if mode == 'full':
            reaction_structure_logits = self.reaction_structure_head(encoded)
            masked_reaction_structure_logits = reaction_structure_logits.masked_fill(x_structure.bool(), float('-inf'))
            reaction_structure_distribution = Categorical(logits=masked_reaction_structure_logits)
            samples_reaction_idx = reaction_structure_distribution.sample()
            samples_reaction_hot = batch_multi_hot(samples_reaction_idx.unsqueeze(-1).cpu().numpy(), self.num_possible_reactions, intensities=None, device=self.device)
            entropy = reaction_structure_distribution.entropy()
            log_probability = reaction_structure_distribution.log_prob(samples_reaction_idx)
        
        # Decode the reaction rate
        x1 = torch.cat([encoded, samples_reaction_hot], dim=-1)
        continuous_distribution_parameters = self.reaction_rate_head(x1)
        match self.continuous_distribution:
            case 'lognormal': # Parameters are mean and log(stddev)
                # log_mu, log_sigma = continuous_distribution_parameters[:, 0], continuous_distribution_parameters[:, 1]
                # mu = torch.exp(log_mu)
                # sigma = torch.exp(log_sigma)
                # reaction_rate_distribution = LogNormal(mu, sigma)
                continuous_distribution_parameters = torch.nn.functional.softplus(continuous_distribution_parameters)
                mu_log_normal, sigma_log_normal = continuous_distribution_parameters[:, 0], continuous_distribution_parameters[:, 1]
                mu_normal = torch.log(mu_log_normal**2 / torch.sqrt(mu_log_normal**2 + sigma_log_normal**2)) 
                sigma_normal = torch.log(1 + sigma_log_normal**2 / mu_log_normal**2)
                reaction_rate_distribution = LogNormal(mu_normal, sigma_normal)
                samples_reaction_rate = reaction_rate_distribution.sample()
                entropy = entropy + reaction_rate_distribution.entropy()
                log_probability = log_probability + reaction_rate_distribution.log_prob(samples_reaction_rate)
            case _:
                raise ValueError(f"Unknown continuous distribution: {self.continuous_distribution}. Supported distributions are: 'lognormal'.")

        # Decode the input influence if applicable 
        if self.allow_input_influence is True:
            x2 = torch.cat([x1, samples_reaction_rate.unsqueeze(-1)], dim=-1)
            input_influence_logits = self.input_influence_head(x2)
            input_influence_distribution = Categorical(logits=input_influence_logits)
            samples_input_influence_idx = input_influence_distribution.sample()
            entropy = entropy + input_influence_distribution.entropy()
            log_probability = log_probability + input_influence_distribution.log_prob(samples_input_influence_idx)

        # Construct the output of the neural network, Convert to numpy and move to CPU
        samples_reaction_rate = samples_reaction_rate.cpu().numpy()
        if self.allow_input_influence is True:
            samples_input_influence_idx = samples_input_influence_idx.cpu().numpy()
            if mode == 'full': # structure and rates
                samples_reaction_idx = samples_reaction_idx.cpu().numpy()
                samples = [
                    {
                        'reaction index': r_idx,
                        'rate constant': r_rate,
                        'input influence index': infl_idx
                    }
                    for r_idx, r_rate, infl_idx in zip(samples_reaction_idx, samples_reaction_rate, samples_input_influence_idx)
                ]
            elif mode == 'partial': #rates
                samples = [
                    {
                        'rate constant': r_rate,
                        'input influence index': infl_idx
                    }
                    for r_rate, infl_idx in zip(samples_reaction_rate, samples_input_influence_idx)
                ]
            else:
                raise ValueError(f"Unknown mode: {mode}. Supported modes are: 'full', 'partial'.")
        else:
            if mode == 'full': # structure and rates
                samples_reaction_idx = samples_reaction_idx.cpu().numpy()
                samples = [
                    {
                        'reaction index': r_idx,
                        'rate constant': r_rate,
                    }
                    for r_idx, r_rate in zip(samples_reaction_idx, samples_reaction_rate)
                ]
            elif mode == 'partial': # rates
                samples = [
                    {
                        'rate constant': r_rate,
                    }
                    for r_rate in samples_reaction_rate
                ]
            else:
                raise ValueError(f"Unknown mode: {mode}. Supported modes are: 'full', 'partial'.")
        return samples, log_probability, entropy