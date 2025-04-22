import numpy as np
import sympy as sp
from scipy.optimize import fsolve
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import copy
from scipy.integrate import solve_ivp

class IOCRN_MassAction:
    def __init__(self, stoichiometry_reactants, stoichiometry_products, parameters, input_influence_matrix, output_species, species_labels, inputs_labels):
        self.species_labels = species_labels
        self.inputs_labels = inputs_labels
        self.num_species = stoichiometry_reactants.shape[0]
        self.num_reactions = stoichiometry_reactants.shape[1]
        self.num_inputs = input_influence_matrix.shape[0]
        self.num_outputs = len(output_species)
        self.stoichiometry_reactants = stoichiometry_reactants
        self.stoichiometry_products = stoichiometry_products
        self.stoichiometry_matrix = stoichiometry_products - stoichiometry_reactants
        self.parameters = parameters
        self.input_influence_matrix = input_influence_matrix
        self.output_species = output_species
        self.num_unknown_parameters = np.isnan(parameters).sum()
        self.nan_indices = np.where(np.isnan(parameters))[0]

    def clone(self):
        return copy.deepcopy(self)
    
    def get_complexes_range(self):
        return self.num_species * (self.num_species + 3) // 2 + 1
    
    def map_index_to_species(self, index):
        species1_index = np.floor((2*self.num_species + 3 - np.sqrt((2*self.num_species + 3)**2 - 8 * index)) / 2).astype(np.uint64)
        species2_index = (index - (species1_index * (2*self.num_species + 1 - species1_index)) / 2).astype(np.uint64)
        return species1_index, species2_index

    def set_next_unknown_parameter(self, value):
        if self.num_unknown_parameters == 0:
            raise ValueError("All parameters are set.")
        if isinstance(value, dict):
            raise Exception(str(self.nan_indices)+str(type(value))+str(self.num_unknown_parameters))
        self.set_parameters(self.nan_indices[0], value)

    def tune_reaction(self, reaction_index, value):
        if reaction_index < 0 or reaction_index >= self.num_reactions:
            raise ValueError("Reaction index out of bounds.")
        if value <= 0:
            raise ValueError("Rate constant must be positive.")
        self.parameters[reaction_index] = value

    def set_parameters(self, index, value):
        self.parameters[index] = value
        self.num_unknown_parameters = np.isnan(self.parameters).sum()
        self.nan_indices = np.where(np.isnan(self.parameters))[0]

    def add_reaction(self, reaction):
        # reaction is a dictionary with keys 'reactants index', 'products index', 'input influence index', 'rate constant'
        reactant1_index, reactant2_index = self.map_index_to_species(reaction['reactants index'].cpu().numpy())
        product1_index, product2_index = self.map_index_to_species(reaction['products index'].cpu().numpy())
        self.stoichiometry_reactants = np.pad(self.stoichiometry_reactants, ((0, 0), (0, 1)), mode='constant')
        self.stoichiometry_products = np.pad(self.stoichiometry_products, ((0, 0), (0, 1)), mode='constant')
        self.input_influence_matrix = np.pad(self.input_influence_matrix, ((0, 0), (0, 1)), mode='constant')
        if reactant1_index > 0:     # reactant1_index is 0 if it is the empty set
            self.stoichiometry_reactants[np.uint64(reactant1_index-1), -1] += 1
        if reactant2_index > 0:     # reactant2_index is 0 if it is the empty set
            self.stoichiometry_reactants[np.uint64(reactant2_index-1), -1] += 1
        if product1_index > 0:      # product1_index is 0 if it is the empty set
            self.stoichiometry_products[np.uint64(product1_index-1), -1] += 1
        if product2_index > 0:      # product2_index is 0 if it is the empty set
            self.stoichiometry_products[np.uint64(product2_index-1), -1] += 1
        if reaction['input influence index'] > 0:   # input influence index is 0 if no input influences the reaction
            self.input_influence_matrix[np.uint64(reaction['input influence index']-1), -1] = 1
        self.stoichiometry_matrix = self.stoichiometry_products - self.stoichiometry_reactants
        self.num_unknown_parameters = np.isnan(self.parameters).sum()
        self.nan_indices = np.where(np.isnan(self.parameters))[0]
        self.num_reactions += 1
        self.parameters = np.append(self.parameters, reaction['rate constant'])

    def propensity_function(self, concentrations, inputs):
        return self.parameters * np.prod(np.power(concentrations, self.stoichiometry_reactants.T), axis=1) * np.prod(np.power(inputs, self.input_influence_matrix.T), axis=1)
    
    def symbolic_propensity_function(self, concentrations, parameters, inputs):
        stoichiometry_reactants = sp.Matrix(self.stoichiometry_reactants)
        input_influence_matrix = sp.Matrix(self.input_influence_matrix)
        propensity = sp.zeros(self.num_reactions, 1)
        for j in range(self.num_reactions):
            propensity_reactants = sp.prod([c**s for c, s in zip(concentrations, stoichiometry_reactants[:, j])])
            propensity_inputs = sp.prod([i**inf for i, inf in zip(inputs, input_influence_matrix[:, j])])
            propensity[j] = parameters[j] * propensity_reactants * propensity_inputs
        return propensity
    
    def rate_function(self, time, concentrations, inputs):
        return np.matmul(self.stoichiometry_matrix, self.propensity_function(concentrations, inputs))

    def transient_response(self, inputs, initial_condition, time_horizon, return_states=False):
        outputs = []

        def stop_if_unstable(t, y):
            """Event function to stop integration if solution becomes unstable."""
            threshold = 10000  # Adjust as needed
            return threshold - np.max(y)
        
        stop_if_unstable.terminal = True  # Stop integration if triggered
        stop_if_unstable.direction = -1   # Trigger when exceeding threshold

        for input in inputs:
            solution = solve_ivp(
                lambda t, y: self.rate_function(t, y, input),  # ODE function
                (time_horizon[0], time_horizon[-1]),  # Time span
                initial_condition,  # Initial conditions
                t_eval=time_horizon,  # Output time points
                method="LSODA",  # Use LSODA for adaptive stepping
                events=stop_if_unstable  # Add event to stop on instability
            ).y.T
            output = solution[:, self.output_species - 1]
            if output.shape[0] < time_horizon.shape[0]:
                output = np.pad(output, ((0, time_horizon.shape[0] - output.shape[0]), (0,0)), mode='constant', constant_values=1000.0)
            outputs.append(output)  

        if return_states:
            return outputs, solution
        return outputs
  
    # def transient_response(self, inputs, initial_condition, time_horizon, return_states=False):
    #     outputs = []
    #     for input in inputs:
    #         # solution = odeint(lambda concentrations, time: self.rate_function(time, concentrations, input), initial_condition, time_horizon)
    #         # rewrite with solve_ivp
    #         solution = solve_ivp(
    #             lambda t, y: self.rate_function(t, y, input),  # Function to integrate
    #             (time_horizon[0], time_horizon[-1]),  # Time span
    #             initial_condition,  # Initial condition
    #             t_eval=time_horizon,  # Specific time points where solution is computed
    #             method="LSODA",  # Use LSODA to match odeint behavior
    #         ).y.T


    #         output = solution[:, self.output_species - 1]
    #         outputs.append(output)  
    #     if return_states:
    #         return outputs, solution
    #     return outputs

    def dose_response(self, inputs, initial_guess, plot_flag = False, axis=None):
        outputs = []
        for input in inputs:
            solution = fsolve(lambda concentrations, input: self.rate_function(0, concentrations, input), initial_guess, args=(input,))
            output = solution[self.output_species - 1]
            outputs.append(output) 
            initial_guess = solution
        if plot_flag:
            if axis is None:
                axis = plt.subplot()
            axis.set_xlabel('Input')
            axis.set_ylabel('Output')
            axis.set_title('Dose Response')
            label = f'Params {self.parameters}'
            axis.plot(inputs, outputs, label=label)
        return outputs
    
    def print_reactions(self):
        print(f'Inputs: {self.inputs_labels}')
        print(f'Species: {self.species_labels}')
        print(f'Output Species: {[self.species_labels[i-1] for i in self.output_species]}')
        for j in range(self.num_reactions):
            reactants = []
            products = []
            influencing_inputs = []
            for i in range(self.num_species):
                if self.stoichiometry_reactants[i, j] > 0:
                    reactants.append((self.species_labels[i], self.stoichiometry_reactants[i, j]))
                if self.stoichiometry_products[i, j] > 0:
                    products.append((self.species_labels[i], self.stoichiometry_products[i, j]))
            for k in range(self.num_inputs):
                if self.input_influence_matrix[k, j] > 0:
                    influencing_inputs.append(self.inputs_labels[k])
            reactant_str = ' + '.join(f'{coeff} {sp}' if coeff > 1 else sp for sp, coeff in reactants)
            product_str = ' + '.join(f'{coeff} {sp}' if coeff > 1 else sp for sp, coeff in products)
            influencing_inputs_str = ' '.join(f'{inp}' for inp in influencing_inputs)
            if not reactant_str:
                reactant_str = '0'
            if not product_str:
                product_str = '0'
            print(f'Reaction {j}: {reactant_str} -> {product_str} ; Rate Constant: {self.parameters[j]}{influencing_inputs_str}')

    def print_ODEs(self, species, parameters, inputs):
        S = sp.Matrix(self.stoichiometry_matrix)
        prop = self.symbolic_propensity_function(species, parameters, inputs)
        ODEs = S * prop
        sp.pprint(ODEs)
        return ODEs
    
    def linearize_ODEs(self, species, parameters, inputs):
        S = sp.Matrix(self.stoichiometry_matrix)
        prop = self.symbolic_propensity_function(species, parameters, inputs)
        ODEs = S * prop
        A = ODEs.jacobian(species)
        B = ODEs.jacobian(inputs)
        return A, B