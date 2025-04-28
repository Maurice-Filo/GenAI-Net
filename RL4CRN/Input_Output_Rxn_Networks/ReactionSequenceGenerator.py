import numpy as np
import torch
from RL4CRN.Index_Generators.MultIndexSequenceGenerator import MultIndexSequenceGenerator

class ReactionSequenceGenerator(torch.nn.Module):
    def __init__(self, num_species, num_reactions, num_inputs, num_samples, LSTM_hidden_size, FFNN_hidden_size, FFNN_num_layers, weight=None):
        super().__init__()
        self.num_species = num_species
        self.num_reactions = num_reactions
        self.num_inputs = num_inputs
        self.num_samples = num_samples
        self.LSTM_hidden_size = LSTM_hidden_size
        num_categories = [(num_species + 1) * (num_species + 2) // 2, (num_species + 1) * (num_species + 2) // 2, num_inputs + 1]
        self.model = MultIndexSequenceGenerator(LSTM_hidden_size, FFNN_hidden_size, num_categories, FFNN_num_layers, num_reactions, num_samples, weight=weight)

    def forward(self, h0, c0, logP0=None, H0=None, I0=None, manifold_dim=None):
        indices, total_logP, total_entropy, h, c = self.model(h0, c0, logP0, H0, I0, manifold_dim)
        stoichiometry_reactants, stoichiometry_products, input_influence_matrix = map_indices_to_reactions(indices, self.num_samples, self.num_species, self.num_reactions, self.num_inputs)
        return stoichiometry_reactants, stoichiometry_products, input_influence_matrix, total_logP, total_entropy, h, c

def map_indices_to_reactions(indices, num_samples, num_species, num_reactions, num_inputs):
        # indices is a 3D integer tensor with shape (num_samples, 3, num_reactions)
        stoichiometry_reactants1 = np.zeros((num_samples, num_species, num_reactions), dtype=np.int16)
        stoichiometry_reactants2 = np.zeros((num_samples, num_species, num_reactions), dtype=np.int16)
        stoichiometry_products1 = np.zeros((num_samples, num_species, num_reactions), dtype=np.int16)
        stoichiometry_products2 = np.zeros((num_samples, num_species, num_reactions), dtype=np.int16)
        input_influence_matrix = np.zeros((num_samples, num_inputs, num_reactions), dtype=np.int16)
         # Extract species indices
        species1_indices = torch.floor((2*num_species + 3 - torch.sqrt((2*num_species + 3)**2 - 8 * indices[:, :2, :])) / 2).long()
        species2_indices = (indices[:, :2, :] - (species1_indices * (2*num_species + 1 - species1_indices)) / 2).long()
        # Create masks for reactants, products, and inputs where indices > 0
        mask_reactants1 = species1_indices[:, 0, :] > 0
        mask_reactants2 = species2_indices[:, 0, :] > 0
        mask_products1 = species1_indices[:, 1, :] > 0
        mask_products2 = species2_indices[:, 1, :] > 0
        mask_inputs = indices[:, 2, :] > 0
        # Adjust indices to be zero-based for correct indexing
        index_reactants1 = (species1_indices[:, 0, :] - 1) * mask_reactants1
        index_reactants2 = (species2_indices[:, 0, :] - 1) * mask_reactants2
        index_products1 = (species1_indices[:, 1, :] - 1) * mask_products1
        index_products2 = (species2_indices[:, 1, :] - 1) * mask_products2
        index_inputs = (indices[:, 2, :] - 1) * mask_inputs
         # Use advanced indexing to set the values
        sample_idx = np.arange(num_samples).reshape(-1, 1).repeat(num_reactions, axis=1)

        # move to CPU
        mask_inputs = mask_inputs.cpu()
        mask_reactants1 = mask_reactants1.cpu()
        mask_reactants2 = mask_reactants2.cpu()
        mask_products1 = mask_products1.cpu()
        mask_products2 = mask_products2.cpu()
        index_reactants1 = index_reactants1.cpu()
        index_reactants2 = index_reactants2.cpu()
        index_products1 = index_products1.cpu()
        index_products2 = index_products2.cpu()
        index_inputs = index_inputs.cpu()

        stoichiometry_reactants1[sample_idx, index_reactants1, np.arange(num_reactions)] = mask_reactants1.long()
        stoichiometry_reactants2[sample_idx, index_reactants2, np.arange(num_reactions)] = mask_reactants2.long()
        stoichiometry_reactants = stoichiometry_reactants1 + stoichiometry_reactants2
        stoichiometry_products1[sample_idx, index_products1, np.arange(num_reactions)] = mask_products1.long()
        stoichiometry_products2[sample_idx, index_products2, np.arange(num_reactions)] = mask_products2.long()
        stoichiometry_products = stoichiometry_products1 + stoichiometry_products2
        input_influence_matrix[sample_idx, index_inputs, np.arange(num_reactions)] = mask_inputs
        return stoichiometry_reactants, stoichiometry_products, input_influence_matrix