import torch

class FFNN(torch.nn.Module):
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