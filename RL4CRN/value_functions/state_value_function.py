import torch
from RL4CRN.utils.ffnn import FFNN
from torch.distributions import Categorical, LogNormal
from RL4CRN.utils.utils import batch_multi_hot

class StateValueFunction(torch.nn.Module):
    
    def __init__(self, num_reactions, num_parameters, num_inputs, 
                 encoder_attributes, deep_layer_size, output_head_attributes, device=None):
        """ Initialize the StateValueFunction.
        Args:
        - num_reactions: total number of reactions to select from (assumed to be the same for all IOCRNs in the batch).
        - num_parameters: total number of parameters (continuous + discrete) across all possible reactions.
        - num_inputs: number of inputs in the IOCRN (assumed to be the same for all IOCRNs in the batch).
        - encoder_attributes: a dictionary containing the attributes of the encoder neural network (hidden_size, num_layers).
        - deep_layer_size: size of the deep layer representation of the IOCRN.
        - output_head_attributes: a dictionary containing the attributes of the reaction structure head neural network (hidden_size, num_layers).
        - device (torch.device): The device to run on. Default is None, which uses the current device. """

        super().__init__()
        
        # Record the IOCRN attributes
        self.M = num_reactions                                              # Total number of reactions
        self.K = num_parameters                                             # Total number of parameters (continuous + discrete) across all reactions
        self.p = num_inputs                                                 # Number of inputs in the IOCRN

        # Record the neural network attributes
        self.encoder_attributes = encoder_attributes
        self.deep_layer_size = deep_layer_size
        self.output_head_attributes = output_head_attributes
        self.device = device if device is not None else torch.device('cpu')

        # Define the encoder that encodes the IOCRN observation into a deep layer representation
        self.encoder = FFNN(input_size=self.M + (self.p + 1) * self.K, output_size=deep_layer_size, hidden_size=encoder_attributes["hidden_size"], num_layers=encoder_attributes["num_layers"]).to(device=device)
        
        # Define the output head that computes the value of the state at time t
        self.output_head = FFNN(input_size=deep_layer_size, output_size=1, hidden_size=output_head_attributes["hidden_size"], num_layers=output_head_attributes["num_layers"]).to(device=device)
                  
    def forward(self, state):
        """ Forward pass through the StateValueFunction.
        Args:
        - state (torch.Tensor): The observation (state) of the IOCRN. Shape: (N, M + (p+1)*K)), where N is the batch size, M is the number of reactions in the library, p is the number of inputs in the IOCRN, and K is the total number of parameters in the IOCRN. 
        Returns:
        - state_value (torch.Tensor): The value of the state. Shape: (N,). """

        # Validate the input has no NaNs
        assert state.isnan().sum() == 0, "Input contains NaN values."

        # Encode the observation
        encoded = self.encoder(state) # shape: (N, deep_layer_size)
        
        # Compute and return the state value
        state_value = self.output_head(encoded).squeeze(-1) # shape: (N,)
        return state_value