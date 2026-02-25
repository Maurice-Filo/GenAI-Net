"""
Explicit tensorizer.

This module defines `ExplicitTensorizer`, a tensorizer that converts the
explicit observation produced by `RL4CRN.env2agent_interface.explicit_observer.ExplicitObserver`
into a single flat PyTorch tensor.

The expected observation is a tuple of numpy arrays (or array-like objects),
which are concatenated and converted to `torch.float32` on the configured device.
"""

import torch
import numpy as np
from RL4CRN.env2agent_interface.abstract_tensorizer import AbstractTensorizer

class ExplicitTensorizer(AbstractTensorizer):
    """Tensorizer that flattens an explicit observation into a 1D float tensor."""

    def __init__(self, device='cpu'):
        """Initialize the tensorizer.

        Args:
            device: Target device for the returned tensor (e.g., `'cpu'`, `'cuda'`).
        """
        super().__init__(device)

    def tensorize(self, observation):
        """Concatenate an explicit observation and convert it to a torch tensor.

        The input `observation` is expected to be a tuple/list of array-like
        components (e.g., `(reaction_multihot, params_cross_multihot, ...)`).
        The components are concatenated along the last axis and returned as a
        single tensor.

        Args:
            observation: Tuple/list of numpy arrays (or array-like) representing
                an explicit IOCRN observation.

        Returns:
            `torch.float32` tensor on `self.device` containing the concatenated
                observation.
        """
        return torch.as_tensor(np.concatenate(observation), dtype=torch.float32, device=self.device)