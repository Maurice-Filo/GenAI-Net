import numpy as np
import torch
from Index_Generators.IndexSequenceGenerator import IndexSequenceGenerator

class ParameterSequenceGenerator(torch.nn.Module):
    def __init__(self, IOCRN, parameter_grid, LSTM_hidden_size, FFNN_hidden_size, FFNN_num_layers, num_samples=1, weight=None):
        super().__init__()
        self.IOCRN = IOCRN
        self.parameter_grid = parameter_grid
        self.parameter_size = parameter_grid.shape[0]
        self.grid_size = parameter_grid.shape[1]
        self.LSTM_hidden_size = LSTM_hidden_size
        self.num_samples = num_samples
        self.model = IndexSequenceGenerator(LSTM_hidden_size, FFNN_hidden_size, self.grid_size, FFNN_num_layers, self.parameter_size, num_samples, weight=None)
        
    def forward(self, h0, c0, logP0=None, H0=None, I0=None, manifold_dim=None):
        indices, total_logP, total_entropy, h, c = self.model(h0, c0, logP0=None, H0=None, I0=None, index_start=manifold_dim)
        return indices, total_logP, total_entropy, h, c
        
    def compute_loss(self, CRN_inputs, CRN_targeted_outputs, h0, c0, manifold_dim):
        indices, total_logP, total_entropy, _, _ = self.forward(h0, c0, manifold_dim)
        parameter = map_index_to_parameter(indices, self.parameter_grid)

        # CRN_outputs = []
        # for p in parameter.T:
        #     self.IOCRN.parameters = p
        #     y = self.IOCRN.dose_response(CRN_inputs, initial_guess, plot_flag = False, axis=None)
        #     CRN_outputs.append(y)
        # CRN_outputs = torch.stack(CRN_outputs).T

        CRN_outputs = self.IOCRN.dose_response(CRN_inputs, parameter)
        loss_for_each_sample = torch.tensor(np.mean(np.abs(CRN_outputs - np.tile(CRN_targeted_outputs, (parameter.shape[1], 1)).T), axis=0)).to(h0)
        loss_mean = torch.mean(loss_for_each_sample)
        return loss_for_each_sample, loss_mean, total_logP, total_entropy, parameter, CRN_outputs
    
def map_index_to_parameter(indices, parameter_grid):
    # indices is a 2D integer tensor with shape (num_samples, parameter_size)
    # parameter_grid is a 2D array with shape (parameter_size, grid_size)
    indices = indices.cpu()
    return parameter_grid[np.arange(parameter_grid.shape[0]).reshape(-1, 1), indices.detach().T.numpy()]