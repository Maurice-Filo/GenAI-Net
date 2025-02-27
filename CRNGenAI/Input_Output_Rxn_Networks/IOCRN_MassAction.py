import numpy as np
import sympy as sp
from scipy.optimize import fsolve
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import copy
from scipy.integrate import solve_ivp

class IOCRN_MassAction:
    def __init__(self, stoichiometry_reactants, stoichiometry_products, parameters, input_influence_matrix, output_species):
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

    def clone(self):
        return copy.deepcopy(self)

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
                output = np.pad(output, ((0, time_horizon.shape[0] - output.shape[0]), (0,0)), mode='edge')
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
    
    def print_reactions(self, species, inputs):
        print(f'Inputs: {inputs}')
        print(f'Species: {species}')
        print(f'Output Species: {[species[i-1] for i in self.output_species]}')
        for j in range(self.num_reactions):
            reactants = []
            products = []
            influencing_inputs = []
            for i in range(self.num_species):
                if self.stoichiometry_reactants[i, j] > 0:
                    reactants.append((species[i], self.stoichiometry_reactants[i, j]))
                if self.stoichiometry_products[i, j] > 0:
                    products.append((species[i], self.stoichiometry_products[i, j]))
            for k in range(self.num_inputs):
                if self.input_influence_matrix[k, j] > 0:
                    influencing_inputs.append(inputs[k])
            reactant_str = ' + '.join(f'{coeff} {sp}' if coeff > 1 else sp for sp, coeff in reactants)
            product_str = ' + '.join(f'{coeff} {sp}' if coeff > 1 else sp for sp, coeff in products)
            influencing_inputs_str = ' '.join(f'{inp}' for inp in influencing_inputs)
            if not reactant_str:
                reactant_str = '0'
            if not product_str:
                product_str = '0'
            print(f'Reaction {j + 1}: {reactant_str} -> {product_str} ; Rate Constant: {self.parameters[j]}{influencing_inputs_str}')

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