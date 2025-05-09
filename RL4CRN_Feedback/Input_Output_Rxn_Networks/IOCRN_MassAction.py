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
        self.last_task_info = {}
        # Construct the reactions indices


    def clone(self):
        return copy.deepcopy(self)
    
    def set_next_unknown_parameter(self, value):
        if self.num_unknown_parameters == 0:
            raise ValueError("All parameters are set.")
        if isinstance(value, dict):
            raise Exception(str(self.nan_indices)+str(type(value))+str(self.num_unknown_parameters))
        self.set_parameters(self.nan_indices[0], value)

    def tune_reaction(self, reaction_index, value):
        # Flush the trajectory memory
        self.last_task_info = {}

        if reaction_index < 0 or reaction_index >= self.num_reactions:
            raise ValueError("Reaction index out of bounds.")
        if value <= 0:
            raise ValueError("Rate constant must be positive.")
        self.parameters[reaction_index] = value

    def set_parameters(self, index, value):
        # Flush the trajectory memory
        self.last_task_info = {}
        
        self.parameters[index] = value
        self.num_unknown_parameters = np.isnan(self.parameters).sum()
        self.nan_indices = np.where(np.isnan(self.parameters))[0]

    def get_complexes_range(self):
        return self.num_species * (self.num_species + 3) // 2 + 1
    
    def get_reactions_range(self):
        N = self.get_complexes_range() - 1
        return (N+1)*N -2 + 1
    
    def map_complex_to_species(self, index):
        """Maps a complex index to the corresponding species indices.
        Args:
            index (int): The complex index.
        Returns:
            tuple: A tuple containing the indices of the two species involved in the complex.
        """
        species1_index = np.floor((2*self.num_species + 3 - np.sqrt((2*self.num_species + 3)**2 - 8 * index)) / 2).astype(np.uint64)
        species2_index = (index - (species1_index * (2*self.num_species + 1 - species1_index)) / 2).astype(np.uint64)
        return species1_index, species2_index
    
    def map_species_to_complex(self, species_indices):
        """Maps a pair of species indices to the corresponding complex index.
        Args:
            species_indices (tuple): A tuple containing the indices of the two species.
        Returns:
            int: The complex index.
        """
        return ((species_indices[0] * (2*self.num_species + 1 - species_indices[0])) / 2 + species_indices[1]).astype(np.uint64)
    
    def map_reaction_to_complexes(self, index):
        """Maps a reaction index to the corresponding reactants and products complexes indices.
        Args:
            index (int): The reaction index.
        Returns:
            tuple: A tuple containing the indices of the reactants and products complexes.
        """
        N = self.get_complexes_range() - 1
        reactants_index = np.floor(index/N).astype(np.uint64)
        remainder = index % N
        products_index = remainder if remainder < reactants_index else remainder + 1
        return reactants_index, products_index
    
    def map_complexes_to_reaction(self, complexes_indices):
        """Maps a pair of complexes indices to the corresponding reaction index.
        Args:
            complexes_indices (tuple): A tuple containing the indices of the reactants and products complexes.
        Returns:
            int: The reaction index.
        """
        N = self.get_complexes_range() - 1
        return complexes_indices[0] * N + complexes_indices[1] if complexes_indices[0] > complexes_indices[1] else complexes_indices[0] * N + complexes_indices[1] - 1
    
    def map_reaction_to_species(self, index):
        """Maps a reaction index to the corresponding reactants and products species indices.
        Args:
            index (int): The reaction index.
        Returns:
            tuple: A tuple containing the indices of the reactants and products species.
        """
        reactants_index, products_index = self.map_reaction_to_complexes(index)
        reactant1_index, reactant2_index = self.map_complex_to_species(reactants_index)
        product1_index, product2_index = self.map_complex_to_species(products_index)
        return reactant1_index, reactant2_index, product1_index, product2_index

    def add_reaction(self, reaction, mode='complex index'):
        # Flush the trajectory memory
        self.last_task_info = {}

        # For each mode, we extract the indices of the reactants and products species
        match mode:
            case 'complex index':
                # reaction is a dictionary with keys 'reactants index', 'products index', 'input influence index', 'rate constant'
                reactant1_index, reactant2_index = self.map_complex_to_species(reaction['reactants index'])
                product1_index, product2_index = self.map_complex_to_species(reaction['products index'])
            case 'reaction index':
                # reaction is a dictionary with keys 'reaction index', 'input influence index', 'rate constant'
                reactant1_index, reactant2_index, product1_index, product2_index = self.map_reaction_to_species(reaction['reaction index'])
            case 'species index':
                # reaction is a dictionary with keys 'reactant1_index', 'reactant2_index', 'product1_index', 'product2_index', 'input influence index', 'rate constant'
                reactant1_index = reaction['reactant1 index']
                reactant2_index = reaction['reactant2 index']
                product1_index = reaction['product1 index']
                product2_index = reaction['product2 index']
            case _:
                raise ValueError("Invalid mode for adding reactions. Use 'species index', 'complex index', or 'reaction index'.")
        
        # Construct the stoichiometry and input influence matrices
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

        # Update the IOCRN   
        self.stoichiometry_matrix = self.stoichiometry_products - self.stoichiometry_reactants
        self.num_unknown_parameters = np.isnan(self.parameters).sum()
        self.nan_indices = np.where(np.isnan(self.parameters))[0]
        self.num_reactions += 1
        self.parameters = np.append(self.parameters, reaction['rate constant'])

    def map_stoichiometry_to_species(self, stoichiometry):
        """Maps the stoichiometry coefficient matrix to species indices.
        Args:
            stoichiometry (np.ndarray): The stoichiometry matrix.
        Returns:
            np.ndarray: A 2D array with the species indices for each reaction. Dimensions: (2, num_reactions).
        """
        n_species, n_reactions = stoichiometry.shape
        species_indices = np.zeros((2, n_reactions), dtype=int)
        colsum = stoichiometry.sum(axis=0)
        is_1 = stoichiometry == 1
        is_2 = stoichiometry == 2

        # CASE 1: two distinct species with coefficient 1
        idx_two_ones = np.where((colsum == 2) & (is_1.sum(axis=0) == 2))[0]
        if idx_two_ones.size > 0:
            rows, cols = np.where(is_1[:, idx_two_ones])
            species = rows.reshape(2, -1) + 1
            species_indices[:, idx_two_ones] = np.sort(species, axis=0)

        # CASE 2: one species with coefficient 2
        idx_two_same = np.where((colsum == 2) & (is_2.sum(axis=0) == 1))[0]
        if idx_two_same.size > 0:
            species = np.argmax(stoichiometry[:, idx_two_same] == 2, axis=0) + 1
            species_indices[:, idx_two_same] = np.stack([species, species], axis=0)

        # CASE 3: one species with only coefficient 1
        idx_single_one = np.where((colsum == 1) & (is_1.sum(axis=0) == 1))[0]
        if idx_single_one.size > 0:
            species = np.argmax(stoichiometry[:, idx_single_one] == 1, axis=0) + 1
            species_indices[0, idx_single_one] = 0
            species_indices[1, idx_single_one] = species

        return species_indices

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
        if not return_states and self.last_task_info is not None:
            return self.last_task_info['trajectories']
        
        outputs = []
        def stop_if_unstable(t, y):
            """Event function to stop integration if solution becomes unstable."""
            threshold = 10000  # Adjust as needed
            outputs = threshold - np.max(y)
            self.last_task_info['trajectories'] = outputs
            self.last_task_info['time_horizon'] = time_horizon
            return outputs
        
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

        self.last_task_info['trajectories'] = outputs
        self.last_task_info['time_horizon'] = time_horizon
        if return_states:
            return outputs, solution
        
        return outputs
    
    def plot_transient_response(self, fig=None, axes=None):
        if self.last_task_info is None:
            raise ValueError("No transient response data available. Run transient_response() first.")
        if fig is None and axes is None:
            fig, axes = plt.subplots(self.num_outputs, 1, figsize=(10, 5 * self.num_outputs))
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
        for i in range(self.num_outputs):
            for j in range(len(self.last_task_info['trajectories'])):
                axes[i].plot(self.last_task_info['time_horizon'], self.last_task_info['trajectories'][j], alpha=0.1)
                axes[i].set_title(f"Transient Response of Output Species {self.output_species[i]}")
                axes[i].set_xlabel("Time")
                axes[i].set_ylabel("Concentration")
        plt.tight_layout()
        return fig, axes

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

    def __str__(self):
        out = f'Inputs: {self.inputs_labels} \n'
        out += f'Species: {self.species_labels} \n'
        out += f'Output Species: {[self.species_labels[i-1] for i in self.output_species]} \n'
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
            out += f'Reaction {j}: {reactant_str} -> {product_str} ; Rate Constant: {self.parameters[j]}{influencing_inputs_str} \n'
        return out
    
    def print_reactions(self):
        print(self)

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