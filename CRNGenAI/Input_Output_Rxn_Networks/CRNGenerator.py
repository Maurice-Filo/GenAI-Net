import torch
from Index_Generators.IndexSequenceGenerator import IndexSequenceGenerator
from Index_Generators.MultIndexSequenceGenerator import MultIndexSequenceGenerator
from Input_Output_Rxn_Networks.ParameterSequenceGenerator import map_index_to_parameter
from Input_Output_Rxn_Networks.ReactionSequenceGenerator import map_indices_to_reactions

class CRNGenerator(torch.nn.Module):
    def __init__(self, num_species, num_reactions, num_free_parameters, num_inputs, parameter_grid, num_samples, RSG_Attributes, PSG_Attributes):
        super().__init__()
        self.num_species = num_species
        self.num_reactions = num_reactions
        self.num_free_parameters = num_free_parameters
        self.num_inputs = num_inputs
        self.num_samples = num_samples
        self.parameter_grid = parameter_grid
        self.parameter_grid_size = parameter_grid.shape[1]
        num_categories = [(num_species + 1) * (num_species + 2) // 2, (num_species + 1) * (num_species + 2) // 2, num_inputs + 1]
        self.MISG = MultIndexSequenceGenerator(RSG_Attributes.LSTM_hidden_size, RSG_Attributes.FFNN_hidden_size, num_categories, RSG_Attributes.FFNN_num_layers, num_reactions, num_samples, RSG_Attributes.weight)
        self.ISG = IndexSequenceGenerator(PSG_Attributes.LSTM_hidden_size, PSG_Attributes.FFNN_hidden_size, self.parameter_grid_size, PSG_Attributes.FFNN_num_layers, num_reactions+num_free_parameters, num_samples, PSG_Attributes.weight)
    
    def forward(self, h0, c0, logP0=None, H0=None, I0=None, index_start=None):
        indices_structure, total_logP, total_entropy, h, c = self.MISG(h0, c0, logP0, H0, I0, index_start)
        indices_parameters, total_logP, total_entropy, h, c = self.ISG(h, c, logP0=total_logP, H0=total_entropy, I0=None, index_start=index_start)
        stoichiometry_reactants, stoichiometry_products, input_influence_matrix = map_indices_to_reactions(indices_structure, self.num_samples, self.num_species, self.num_reactions, self.num_inputs)
        parameter = map_index_to_parameter(indices_parameters, self.parameter_grid)
        return stoichiometry_reactants, stoichiometry_products, input_influence_matrix, parameter, total_logP, total_entropy, h, c
