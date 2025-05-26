import torch
from RL4CRN_Feedback.Policies.FFNN import FFNN
from torch.distributions import Categorical, LogNormal
from RL4CRN_Feedback.Utils.Utils import batch_multi_hot

# TODO cleanup useless arguments
# similar to Policy, but with a single output head for the value function
class BimolecularMassActionValue(torch.nn.Module):
    def __init__(self, num_possible_reactions, num_inputs, encoder_attributes, hidden_size, structure_decoder_attributes, rate_decoder_attributes, input_influence_decoder_attributes, continuous_distribution='lognormal', allow_input_influence=False, device=None):
        # call super constructor
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
        self.output_head = FFNN(input_size=hidden_size, output_size=1, hidden_size=structure_decoder_attributes["hidden_size"], num_layers=structure_decoder_attributes["num_layers"]).to(device=device)


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
        return self.output_head(encoded).squeeze(-1) # remove last dimension
    

    