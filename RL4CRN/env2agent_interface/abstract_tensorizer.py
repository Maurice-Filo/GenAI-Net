"""
Abstract tensorizer interface.

Defines the minimal interface for a *tensorizer*, a component that converts
environment observations (typically produced by an observer) into PyTorch
tensors suitable for neural network policies.

Tensorizers belong to the environment-to-agent (env2agent) interface and
encapsulate device placement and any preprocessing needed to obtain the final
tensor representation.
"""

import torch
import numpy as np

class AbstractTensorizer():
    """Base class for tensorizers mapping observations to torch tensors."""

    def __init__(self, device='cpu'):
        """Initialize the tensorizer.

        Args:
            device: Target device for produced tensors (e.g., `'cpu'`, `'cuda'`).
        """
        self.device = device

    def tensorize(self, observation):
        """Convert an observation into a torch tensor representation.

        Concrete tensorizers should override this method and document the expected
        structure of `observation` as well as the returned tensor shapes/dtypes.

        Args:
            observation: Observation object produced by an observer.

        Returns:
            A torch tensor (or a collection of tensors) placed on `self.device`.
        """
        pass