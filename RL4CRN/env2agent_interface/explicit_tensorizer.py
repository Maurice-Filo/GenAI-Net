import torch
import numpy as np
from RL4CRN.env2agent_interface.abstract_tensorizer import AbstractTensorizer

class ExplicitTensorizer(AbstractTensorizer):
    def __init__(self, device='cpu'):
        super().__init__(device)

    def tensorize(self, observation):
        """
        Translate the observation into a format suitable for the agent.
        """
        return torch.as_tensor(np.concatenate(observation), dtype=torch.float32, device=self.device)