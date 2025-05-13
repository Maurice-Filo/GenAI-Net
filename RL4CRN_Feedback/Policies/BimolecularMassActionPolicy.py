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
        self.encoder = FFNN(input_size=num_possible_reactions * (num_inputs + 2), output_size=hidden_size, hidden_size=encoder_attributes["hidden_size"], num_layers=encoder_attributes["num_layers"])
        
        # Define the reaction structure head
        self.reaction_structure_head = FFNN(input_size=hidden_size, output_size=num_possible_reactions, hidden_size=structure_decoder_attributes["hidden_size"], num_layers=structure_decoder_attributes["num_layers"])
        
        # Define the reaction rate head
        match continuous_distribution:
            case 'lognormal':
                self.reaction_rate_head = FFNN(input_size=hidden_size + num_possible_reactions, output_size=2, hidden_size=rate_decoder_attributes["hidden_size"], num_layers=rate_decoder_attributes["num_layers"])
            case _:
                raise ValueError(f"Unknown continuous distribution: {continuous_distribution}. Supported distributions are: 'lognormal'.")
        
        # Define the input influence head
        if allow_input_influence is True:
            self.input_influence_head = FFNN(input_size=hidden_size + num_possible_reactions + 1, output_size=num_inputs+1, hidden_size=input_influence_decoder_attributes["hidden_size"], num_layers=input_influence_decoder_attributes["num_layers"])
        else:
            self.input_influence_head = None

    def forward(self, x_structure, x_rate, x_input_influence, mode='full'):
        batch_size = x_structure.size(0)
        
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
            sample_reaction_idx = reaction_structure_distribution.sample()
            sample_reaction_hot = batch_multi_hot(sample_reaction_idx.unsqueeze(-1), self.num_possible_reactions, intensities=None, device=self.device)
            entropy = reaction_structure_distribution.entropy()
            log_probability = reaction_structure_distribution.log_prob(sample_reaction_idx)
        
        # Decode the reaction rate
        x1 = torch.cat([encoded, sample_reaction_hot], dim=-1)
        continuous_distribution_parameters = self.reaction_rate_head(x1)
        match self.continuous_distribution:
            case 'lognormal': # Parameters are mean and log(stddev)
                mu, log_sigma = continuous_distribution_parameters[:, 0], continuous_distribution_parameters[:, 1]
                sigma = torch.exp(log_sigma)
                reaction_rate_distribution = LogNormal(mu, sigma)
                sample_reaction_rate = reaction_rate_distribution.sample()
                entropy = entropy + reaction_rate_distribution.entropy()
                log_probability = log_probability + reaction_rate_distribution.log_prob(sample_reaction_rate)
            case _:
                raise ValueError(f"Unknown continuous distribution: {self.continuous_distribution}. Supported distributions are: 'lognormal'.")

        # Decode the input influence if applicable
        if self.allow_input_influence is True:
            x2 = torch.cat([x1, sample_reaction_rate.unsqueeze(-1)], dim=-1)
            input_influence_logits = self.input_influence_head(x2)
            input_influence_distribution = Categorical(logits=input_influence_logits)
            sample_input_influence_idx = input_influence_distribution.sample()
            entropy = entropy + input_influence_distribution.entropy()
            log_probability = log_probability + input_influence_distribution.log_prob(sample_input_influence_idx)

        # Construct the output of the neural network
        if self.allow_input_influence is True:
            if mode == 'full': # structure and rates
                Sample = {
                    'reaction_idx': sample_reaction_idx,
                    'reaction_rate': sample_reaction_rate,
                    'input_influence_idx': sample_input_influence_idx
                }
            elif mode == 'partial': #rates
                Sample = {
                    'reaction_rate': sample_reaction_rate,
                    'input_influence_idx': sample_input_influence_idx
                }
            else:
                raise ValueError(f"Unknown mode: {mode}. Supported modes are: 'full', 'partial'.")
        else:
            if mode == 'full': # structure and rates
                Sample = {
                    'reaction_idx': sample_reaction_idx,
                    'reaction_rate': sample_reaction_rate,
                }
            elif mode == 'partial': # rates
                Sample = {
                    'reaction_rate': sample_reaction_rate,
                }
            else:
                raise ValueError(f"Unknown mode: {mode}. Supported modes are: 'full', 'partial'.")
        return Sample, log_probability, entropy