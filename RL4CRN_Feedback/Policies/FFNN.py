import torch

class FFNN(torch.nn.Module):
    """
    A Feed-Forward Neural Network (FFNN) for reinforcement learning tasks.
    This network consists of a series of linear layers with ReLU activations,
    followed by additional hidden layers with Tanh activations, and ends with
    a final linear layer to produce the output.
    Args:
        input_size (int): Size of the input layer.
        output_size (int): Size of the output layer.
        hidden_size (int): Size of the hidden layers.
        num_layers (int): Number of hidden layers in the network.
    """
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size), 
            torch.nn.ReLU(), 
            *[torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.Tanh() 
            ) for _ in range(num_layers)],
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)