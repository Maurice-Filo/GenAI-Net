"""
Feed-forward neural network utilities.

This module defines a lightweight fully-connected neural network backbone used
throughout RL4CRN (e.g., policy encoders, parameter heads, and other small MLP
components). It provides a single `FFNN` class that builds an MLP with a
configurable number of hidden layers and activations, intended as a simple,
general-purpose function approximator.
"""

import torch

class FFNN(torch.nn.Module):
    """
    Simple feed-forward neural network (MLP) backbone.

    The network is a stack of fully-connected layers mapping an input vector to
    an output vector. It is primarily used as a reusable building block for
    encoders and heads in policies/value functions.

    Architecture:
        - Linear(input_size -> hidden_size) + ReLU
        - `num_layers` blocks of: Linear(hidden_size -> hidden_size) + Tanh
        - Linear(hidden_size -> output_size)

    Notes:
        - This class does not apply any output activation; callers should apply
          any required squashing (e.g., tanh/softplus) externally.
        - Input tensors are expected to be of shape (N, input_size), where N is
          the batch dimension.

    Args:
        input_size (int): Dimensionality of the input features.
        output_size (int): Dimensionality of the output features.
        hidden_size (int): Width of the hidden layers.
        num_layers (int): Number of additional hidden blocks after the first
            layer (each block is Linear + Tanh).

    Attributes:
        input_size (int): Stored input dimensionality.
        output_size (int): Stored output dimensionality.
        hidden_size (int): Stored hidden width.
        num_layers (int): Stored number of hidden blocks.
        model (torch.nn.Sequential): The assembled PyTorch module.

    Example:
        ```
        net = FFNN(input_size=16, output_size=4, hidden_size=64, num_layers=2)
        x = torch.randn(32, 16)
        y = net(x)  # (32, 4)
        ```
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