import torch
from RL4CRN.utils.ffnn import FFNN
from torch.distributions import Categorical, LogNormal
from RL4CRN.utils.utils import batch_multi_hot

class StateValueFunction(torch.nn.Module):
    """
    A class representing a state-value function for an IOCRN (Input-Output Chemical Reaction Network). 
    In finite-horizon settings, the value function takes as input the state and time step, but for our IOCRN setting, it only takes the state as input since the time step is already encoded in the number of reactions in the IOCRN.
    The input is a representation of the current batch of IOCRNs (observation from environment) which is organized as follows:
        - M inputs for the multi-hot encoding of the reaction structure
        - M inputs for the intensity multi-hot encoding of the reaction rates
        - M*p inputs for the multi-hot encoding of the input influence
    The output is the values of the states (IOCRNs) in the batch.

    This state-value function is composed of an encoder and one output head:
        - An encoder that encodes the IOCRN observation into a deep layer representation.
        - An output head that reads the deep layer representation to output the value of the state. 
    """
    def __init__(self, M, p, encoder_attributes, deep_layer_size, output_head_attributes, device=None):
        """
        Initialize the StateValueFunction.
        Args:
            - M (int): The number of all possible reactions in the library.
            - p (int): The number of inputs to the IOCRN.
            - encoder_attributes (dict): Attributes for the encoder, including hidden size and number of layers.
            - deep_layer_size (int): The size of the deep layer representing an embedding for the IOCRN.
            - output_head_attributes (dict): Attributes for the output head, including hidden size and number of layers.
            - device (torch.device): The device to run on. Default is None, which uses the current device.
        The encoder and the output head attributes are dictionaries with the following keys:
            - hidden_size (int): The size of the hidden layers in the feedforward neural network.
            - num_layers (int): The number of hidden layers in the feedforward neural network.
        """
        super().__init__()
        self.M = M
        self.p = p
        self.encoder_attributes = encoder_attributes
        self.deep_layer_size = deep_layer_size
        self.output_head_attributes = output_head_attributes
        self.device = device if device is not None else torch.device('cpu')

        # Define the encoder that encodes the IOCRN observation into a deep layer representation
        self.encoder = FFNN(input_size=M * (p + 2), output_size=deep_layer_size, hidden_size=encoder_attributes["hidden_size"], num_layers=encoder_attributes["num_layers"]).to(device=device)
        
        # Define the output head that computes the value of the state at time t
        self.output_head = FFNN(input_size=deep_layer_size, output_size=1, hidden_size=output_head_attributes["hidden_size"], num_layers=output_head_attributes["num_layers"]).to(device=device)
                  
    def forward(self, states):
        """
        Forward pass of the StateValueFunction.
        Args:
            - states (tuple): A tuple containing the following elements:
                - reactions_indices_batch: A numpy array representing the reactions indices in the batch of IOCRNs. Shape: (N, # of reactions).
                - rate_constants_batch: A numpy array representing the reaction rate constants in the batch of IOCRNs. Shape: (N, # of reactions).
                - reactions_indices_influenced_by_inputs_batch: A list of p numpy arrays, each containing the influenced reactions for a specific input. 
                Each numpy array is associated with a specific input and has shape (N, #), where # is the maximum number of reactions in any CRN in the batch influenced by this input.
        Returns:
            - state_values (torch.Tensor): A tensor of shape (N,) containing the state values for each IOCRN in the batch.
        """
        reactions_indices_batch, rate_constants_batch, reactions_indices_influenced_by_inputs_batch = states

        # Compute the multi-hot encoding of the observations
        reactions_indices_batch_hot, rates_batch_hot = batch_multi_hot(reactions_indices_batch, self.M, rate_constants_batch, device=self.device)
        reactions_indices_influenced_by_inputs_batch_hot = [batch_multi_hot(reactions_indices_influenced_by_inputs_batch[i], self.M, device=self.device) for i in range(self.p)]

        # Construct the encoder input
        x_structure = reactions_indices_batch_hot # shape: (N, M)
        x_rate = rates_batch_hot # shape: (N, M)
        x_input_influence = torch.cat(reactions_indices_influenced_by_inputs_batch_hot, dim=1) # shape: (N, M*p)

        # Encode the state-value function input
        x = torch.cat([x_structure, x_rate, x_input_influence], dim=1) # shape: (N, M*(p+2))
        encoded = self.encoder(x) # shape: (N, deep_layer_size)
        
        # Compute and return the state value
        return self.output_head(encoded).squeeze(-1) # shape: (N,)