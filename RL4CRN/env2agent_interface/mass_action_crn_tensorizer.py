import torch
import numpy as np
from RL4CRN.env2agent_interface.abstract_tensorizer import AbstractTensorizer
from RL4CRN.utils.utils import batch_multi_hot

class MassActionCRNTensorizer(AbstractTensorizer):
    def __init__(self, device='cpu'):
        super().__init__(device)

    def tensorize(self, observation):
        """
        Translate the observation into a format suitable for the agent.
        """
        reactions_indices, rate_constants, reactions_indices_influenced_by_inputs, M = observation
        p = len(reactions_indices_influenced_by_inputs)

        # Compute the multi-hot encoding of the observation
        reactions_indices_hot, rates_hot = batch_multi_hot(np.expand_dims(reactions_indices, axis=0), M, np.expand_dims(rate_constants, axis=0), device=self.device)
        reactions_indices_influenced_by_inputs_hot = [batch_multi_hot(np.expand_dims(reactions_indices_influenced_by_inputs[i], axis=0), M, device=self.device) for i in range(p)]
        reactions_indices_hot = torch.squeeze(reactions_indices_hot)
        rates_hot = torch.squeeze(rates_hot)
        reactions_indices_influenced_by_inputs_hot = torch.squeeze(torch.cat(reactions_indices_influenced_by_inputs_hot, dim=1))

        return torch.cat((reactions_indices_hot, rates_hot, reactions_indices_influenced_by_inputs_hot), dim=0)