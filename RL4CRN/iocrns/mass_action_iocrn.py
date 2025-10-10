import numpy as np
import sympy as sp
from itertools import product
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import copy
from scipy.integrate import solve_ivp

class MassActionIOCRN:
    """
    A class representing an Input-Output Chemical Reaction Network (IOCRN) with mass action kinetics and with at most two reactants and two products per reaction.
    The IOCRN is fully defined by:
    - Number of species:                n
    - Number of reactions:              m
    - Number of inputs:                 p   
    - Number of outputs:                q
    - Reactants stoichiometry matrix:   S_R     (n x m)
    - Products stoichiometry matrix:    S_P     (n x m)
    - Rate constants:                   c       (m x 1)
    - Input influence matrix:           S_I     (p x m)
    - Output species indeces:           o       (q x 1), elements in {1, ..., n}
    """
    # ------------------------ Construction Methods ------------------------
    def __init__(self, S_R, S_P, c, S_I, o, species_labels, inputs_labels):
        """
        Initializes the IOCRN with the given stoichiometry matrices, rate constants, input influence matrix, output species indices, and labels.
        Arguments:
        - S_R: numpy array of shape (n, m) representing the reactants stoichiometry matrix.
        - S_P: numpy array of shape (n, m) representing the products stoichiometry matrix.
        - c: numpy array of shape (m,) representing the rate constants for each reaction.
        - S_I: numpy array of shape (p, m) representing the input influence matrix.
        - o: numpy array of shape (q,) representing the indices of the output species.
        - species_labels: list of strings representing the labels of the species.
        - inputs_labels: list of strings representing the labels of the inputs.
        """
        # Record the stoichiometry matrices, rate constants, input influence matrix, output species indices, and labels
        self.S_R = S_R
        self.S_P = S_P
        self.c = c
        self.S_I = S_I
        self.o = o
        self.species_labels = species_labels
        self.inputs_labels = inputs_labels

        # Get the number of species, reactions, inputs, and outputs
        self.n = S_R.shape[0]
        self.m = S_R.shape[1]
        self.p = S_I.shape[0]
        self.num_outputs = len(o)

        # Compute the stoichiometry matrix S    
        self.S = S_P - S_R

        # Map the reactants and products stoichiometry matrices to reactions indices
        self.reactions_indices = self.map_stoichiometry_to_reactions(S_R, S_P)

        # Map the input influence matrix to a list where each element corresponds to an input and contains the indices of the reactions influenced by that input
        self.list_influenced_reactions = self.map_input_influence_matrix_to_reactions(self.reactions_indices, S_I)

        # Get the number of unknown rate constants and their indices: these correspond to reactions with NaN rate constants
        self.num_unknown_rates = np.isnan(c).sum()
        self.nan_indices = np.where(np.isnan(c))[0]

        # Initialize a dictionary to store the last task information
        self.last_task_info = {}
        self.last_task_info['type'] = None
        
    def clone(self):
        return copy.deepcopy(self)
    
    def reset(self):
        """
        Resets the IOCRN to its initial state by clearing the last task information.
        """
        self.last_task_info = {}
        self.last_task_info['type'] = None

    def get_complexes_range(self):
        """
        Returns the total number of complexes that can be formed with the given number of species of the IOCRN.
        """
        return self.n * (self.n + 3) // 2 + 1
    
    def get_reactions_range(self):
        """
        Returns the total number of reactions M that can be formed with the given number of species of the IOCRN.
        """
        N = self.get_complexes_range() - 1
        return (N+1)*N + 1
    
    def add_reaction(self, reaction, mode='complex index'):
        """
        Adds a new reaction to the IOCRN.
        Arguments:
        - reaction: A dictionary containing the reaction information. The keys depend on the mode:
            - 'reactants index' and 'products index' for 'complex index'
            - 'reaction index' for 'reaction index'
            - 'reactant1 index', 'reactant2 index', 'product1 index', 'product2 index' for 'species index'
            - 'input influence index' for the input influence index (optional)
            - 'rate constant' for the rate constant of the reaction.
        - mode: A string indicating the mode of the reaction. It can be 'complex index', 'reaction index', or 'species index'.
        The indeces can be numpy integers, integers, or torch tensors.
        """
        # Flush the last task information
        self.last_task_info = {}
        self.last_task_info['type'] = None

        # For each mode, extract the indices of the reactants and products species
        match mode:
            case 'complex index': # reaction is a dictionary with keys 'reactants index', 'products index', 'input influence index', 'rate constant'
                reactant1_idx, reactant2_idx = self.map_complex_to_species(reaction['reactants index'])
                product1_idx, product2_idx = self.map_complex_to_species(reaction['products index'])
            case 'reaction index': # reaction is a dictionary with keys 'reaction index', 'input influence index', 'rate constant'
                reactant1_idx, reactant2_idx, product1_idx, product2_idx = self.map_reaction_to_species(reaction['reaction index'])
            case 'species index': # reaction is a dictionary with keys 'reactant1_index', 'reactant2_index', 'product1_index', 'product2_index', 'input influence index', 'rate constant'
                reactant1_idx = np.int64(reaction['reactant1 index'])
                reactant2_idx = np.int64(reaction['reactant2 index'])
                product1_idx = np.int64(reaction['product1 index'])
                product2_idx = np.int64(reaction['product2 index'])
                if reactant1_idx > reactant2_idx or product1_idx > product2_idx:
                    raise ValueError("Species indices must be in ascending order.")
            case _:
                raise ValueError("Invalid mode for adding reactions. Use 'species index', 'complex index', or 'reaction index'.")
        
        # Construct the stoichiometry matrices and the input influence matrix of the new IOCRN obtained by adding the new reaction
        self.S_R = np.pad(self.S_R, ((0, 0), (0, 1)), mode='constant')
        self.S_P = np.pad(self.S_P, ((0, 0), (0, 1)), mode='constant')
        self.S_I = np.pad(self.S_I, ((0, 0), (0, 1)), mode='constant')
        self.S_R[:, -1] = self.map_species_to_stoichiometry_vector(reactant1_idx, reactant2_idx)
        self.S_P[:, -1] = self.map_species_to_stoichiometry_vector(product1_idx, product2_idx)

        # If the reaction has an input influence index, update the input influence matrix
        if 'input influence index' in reaction.keys(): 
            if reaction['input influence index'] > 0: # an input in {1, ..., p} is selected
                self.S_I[np.int64(reaction['input influence index']-1), -1] = 1 

        # Update the IOCRN 
        self.c = np.append(self.c, reaction['rate constant'])  
        self.m += 1
        self.S = self.S_P - self.S_R
        self.num_unknown_rates = np.isnan(self.c).sum()
        self.nan_indices = np.where(np.isnan(self.c))[0]
        reaction_idx = self.map_species_to_reaction(reactant1_idx, reactant2_idx, product1_idx, product2_idx)
        self.reactions_indices = np.append(self.reactions_indices, reaction_idx)
        if 'input influence index' in reaction.keys():
            if reaction['input influence index'] > 0:
                self.add_input_influence(reaction['input influence index'], reaction_idx)

    def add_input_influence(self, input_influence_idx, reaction_idx):
        """
        Adds the reaction reaction index to the list of influenced reactions for the given input influence index.
        Arguments:
        - input_influence_idx: Index of the input influence, an integer in the range [1, p]. Can be numpy integer, int, or torch tensor.
        - reaction_idx: Index of the reaction, an integer. Can be numpy integer, int, or torch tensor.
        """
        idx = input_influence_idx - 1
        row = self.list_influenced_reactions[idx]
        if reaction_idx in row:
            return 
        self.list_influenced_reactions[idx] = np.append(row, reaction_idx)

    def set_rates(self, index, value):
        """
        Sets the rate constant for the reaction at the given index to the specified value.
        Arguments:
        - index: Index of the reaction, an integer. Can be numpy integer, int, or torch tensor.
        - value: The new rate constant value, a positive float.
        """
        # Flush the last task information
        self.last_task_info = {}
        self.last_task_info['type'] = None

        # Set the rate constant at the specified index to the new value
        self.c[index] = value

        # Update the IOCRN
        self.num_unknown_rates = np.isnan(self.c).sum()
        self.nan_indices = np.where(np.isnan(self.c))[0]

    def set_next_unknown_rate(self, value):
        """
        Sets the next unknown rate constant to the specified value.
        Arguments:
        - value: The new rate constant value, a positive float.
        """
        # If there are no unknown rates, raise an error
        if self.num_unknown_rates == 0:
            raise ValueError("All parameters are set.")
        
        # If the value is not a positive float, raise an error
        if isinstance(value, dict):
            raise Exception(str(self.nan_indices)+str(type(value))+str(self.num_unknown_rates))
        
        # Set the next unknown rate constant to the specified value
        self.set_rates(self.nan_indices[0], value)
    
    # ------------------------ Printing Methods ------------------------
    def __str__(self):
        """
        When print() is called, this function is executed to print a string representation of the IOCRN, including the inputs, species, output species, and reactions.
        """
        out = f'Inputs: {self.inputs_labels} \n'
        out += f'Species: {self.species_labels} \n'
        out += f'Output Species: {[self.species_labels[i-1] for i in self.o]} \n'
        for j in range(self.m):
            reactants = []
            products = []
            influencing_inputs = []
            for i in range(self.n):
                if self.S_R[i, j] > 0:
                    reactants.append((self.species_labels[i], self.S_R[i, j]))
                if self.S_P[i, j] > 0:
                    products.append((self.species_labels[i], self.S_P[i, j]))
            for k in range(self.p):
                if self.S_I[k, j] > 0:
                    influencing_inputs.append(self.inputs_labels[k])
            reactant_str = ' + '.join(f'{coeff} {sp}' if coeff > 1 else sp for sp, coeff in reactants)
            product_str = ' + '.join(f'{coeff} {sp}' if coeff > 1 else sp for sp, coeff in products)
            influencing_inputs_str = ' '.join(f'{inp}' for inp in influencing_inputs)
            if not reactant_str:
                reactant_str = '0'
            if not product_str:
                product_str = '0'
            out += f'Reaction {j}: {reactant_str} -> {product_str} ; Rate Constant: {self.c[j]}{influencing_inputs_str} \n'
        return out
    
    def print_task_info(self, mode='sizes'):
        """
        Prints the information of the last task performed on the IOCRN.
        Arguments:
        - mode: A string indicating the mode of printing. It can be 'sizes' to print the sizes and types of the last task information, or 'values' to print the values of the last task information.
        If no task has been performed yet, it prints a message indicating that.
        """
        if not self.last_task_info:
            print("No task has been performed yet.")
            return
        
        if mode == 'sizes':
            for key, value in self.last_task_info.items():
                value_type = type(value).__name__
                if isinstance(value, list):
                    value_size = len(value) 
                    if all(isinstance(v, np.ndarray) for v in value):
                        shapes = [v.shape for v in value]
                        if all(shape == shapes[0] for shape in shapes):
                            array_shape = shapes[0]
                        else:
                            array_shape = "Variable shapes"
                        print(f"{key} --- Type: {value_type} of numpy arrays, List size: {value_size}, Numpy Arrays shape: {array_shape}")
                    else:
                        print(f"{key} --- Type: {value_type}, Size: {value_size}")
                elif isinstance(value, np.ndarray):
                    print(f"{key} --- Type: {value_type}, Shape: {value.shape}")
                else:
                    print(f"{key} --- Type: {value_type}, Value: {value}")
        else:
            for key, value in self.last_task_info.items():
                print(f"{key}: {value}")
    
    # ------------------------ Computation Methods ------------------------
    def propensity_function(self, x, u):
        """
        Computes the propensity function for the IOCRN given concentrations x and inputs u.
        Arguments:
        - x: numpy array of shape (n,) representing the concentrations of the species.
        - u: numpy array of shape (p,) representing the inputs to the IOCRN.
        Returns:
        - A numpy array of shape (m,) representing the propensity of each reaction.
        """
        return self.c * np.prod(np.power(x, self.S_R.T), axis=1) * np.prod(np.power(u, self.S_I.T), axis=1)
    
    def rate_function(self, t, x, u):
        """
        Computes the rate of change of concentrations for the IOCRN given time t, concentrations x, and inputs u.
        Arguments:
        - t: float representing the current time (not used in mass action kinetics).
        - x: numpy array of shape (n,) representing the concentrations of the species.
        - u: numpy array of shape (p,) representing the inputs to the IOCRN.
        Returns:
        - A numpy array of shape (n,) representing the rate of change of concentrations.
        """
        return np.matmul(self.S, self.propensity_function(x, u))
    
    def dose_response(self, u_dose, u_list, initial_guess):
        """
        Computes the dose response of the IOCRN given a list of input doses, a list of input scenarios, and an initial guess for the concentrations.
        If the CRN dose response has been simulated and stored before, it returns the stored results instead of recomputing them.
        The results are stored in the last_task_info dictionary for future reference.
        Arguments:
        - u_dose: numpy array of shape (num_doses,) representing the input doses to the IOCRN.
        - u_list: A list of numpy arrays, each of shape (p,) representing the constant inputs to the IOCRN for each scenario.
        The first element of each input array corresponds to the dose, and the rest correspond to other inputs.
        - initial_guess: numpy array of shape (n,) representing the initial guess for the concentrations of the species.
        Returns a tupple containing:
            - x_list: A list of numpy arrays of shape (n, num_doses) representing the state for each input scenario.
            - y_list: A list of numpy arrays of shape (q, num_doses) representing the output for each input scenario.
        """
        # If the CRN dose response has been simulated and stored before, return the stored results
        if self.last_task_info:
            if self.last_task_info['type'] == 'dose response':
                return self.last_task_info['states'], self.last_task_info['outputs']
            
        # Check if the IOCRN has unknown rate constants
        if self.num_unknown_rates > 0:
            raise ValueError("The IOCRN has unknown rate constants. Please set them before running the transient response.")
        
        # Do the simulation for each input scenario and store the results in lists
        x_list = []
        y_list = []
        num_doses = u_dose.shape[0]
        for u_e in u_list:
            x = np.zeros((self.n, num_doses), dtype=np.float64) # numpy array of shape (n, num_doses)
            y = np.zeros((self.num_outputs, num_doses), dtype=np.float64) # numpy array of shape (q, num_doses)
            x_0 = initial_guess
            for i in range(num_doses):
                u = np.concatenate(([u_dose[i]], u_e))
                x[:,i] = fsolve(lambda x, u: self.rate_function(0, x, u), x_0, args=(u,))
                y[:,i] = x[self.o - 1, i] 
                x_0 = x[:,i]  
            # Append the states and outputs to the lists
            x_list.append(x) 
            y_list.append(y) 

        # Store and return the last task information
        self.last_task_info = {}
        self.last_task_info['type'] = 'dose response'
        self.last_task_info['input doses'] = u_dose
        self.last_task_info['input scenarios'] = u_list
        self.last_task_info['states'] = x_list
        self.last_task_info['outputs'] = y_list
        return x_list, y_list, self.last_task_info
    
    def transient_response(self, u_list, x0_list, time_horizon, LARGE_NUMBER=1e4):
        """
        Computes the transient response of the IOCRN given a list of inputs, a list of initial conditions, and a time horizon. 
        The combination of inputs and initial conditions are taken as the cartesian product of the two lists.
        If the CRN has been simulated and stored before, it returns the stored results instead of recomputing them.
        If the integration fails or becomes unstable, it fills the remaining time points with large numbers. 
        The results are stored in the last_task_info dictionary for future reference.
        Arguments:
        - u_list: A list of numpy arrays, each of shape (p,) representing the constant inputs to the IOCRN for each input scenario.
        - x0_list: A list of numpy arrays, each of shape (n,) representing the initial conditions for the concentrations of the species for each initial condition scenario.
        - time_horizon: numpy array of shape (T,) representing the time points at which to evaluate the system.
        Returns a tupple containing:
            - time_horizon: numpy array of shape (T,) representing the time points at which the system was evaluated.
            - x_list: A list of numpy arrays of shape (n, T) representing the full state trajectories for each input and initial condition scenario.
            - y_list: A list of numpy arrays of shape (q, T) representing the output trajectories for each input and initial condition scenario.
        """
        # If the CRN dynamics has been simulated and stored before, return the stored results
        if self.last_task_info['type'] == 'transient response':
            return self.last_task_info['time_horizon'], self.last_task_info['trajectories'], self.last_task_info['outputs'], self.last_task_info
        
        # Check if the IOCRN has unknown rate constants
        if self.num_unknown_rates > 0:
            raise ValueError("The IOCRN has unknown rate constants. Please set them before running the transient response.")

        # Event function
        def stop_if_unstable(t, x):
            """Event function to stop if any state becomes unstable."""
            max_val = np.max(np.abs(x))
            if not np.isfinite(max_val):
                return 0 
            return LARGE_NUMBER - max_val 
        stop_if_unstable.terminal = True
        stop_if_unstable.direction = 0

        # Do the simulation for each input and initial condition scenario and store the results in lists
        x_list = []
        y_list = []
        for u, x0 in product(u_list, x0_list):
            solution = solve_ivp(lambda t, x: self.rate_function(t, x, u), (time_horizon[0], time_horizon[-1]), x0, t_eval=time_horizon, method="LSODA", events=stop_if_unstable)

            if solution.status == -1: # if the integration failed, return large numbers for all species and outputs
                x = np.full((self.n, time_horizon.shape[0]), LARGE_NUMBER) # numpy array of shape (n, steps)
            else:
                x = solution.y # numpy array of shape (n, steps)
                if solution.status == 1: # if the integration was stopped due to an event, fill the remaining time points after the event with large numbers
                    x = np.concatenate([x, np.full((self.n, time_horizon.shape[0] - x.shape[1]), LARGE_NUMBER)], axis=1)
            y = x[self.o - 1, :] # select the output species from the state trajectory

            # Append the state trajectory and output trajectory to the lists
            x_list.append(x)
            y_list.append(y)

        # Store and return the last task information
        self.last_task_info = {}
        self.last_task_info['type'] = 'transient response'
        self.last_task_info['inputs'] = u_list
        self.last_task_info['initial conditions'] = x0_list
        self.last_task_info['time_horizon'] = time_horizon
        self.last_task_info['trajectories'] = x_list
        self.last_task_info['outputs'] = y_list
        return time_horizon, x_list, y_list, self.last_task_info
    
    # ------------------------ Plotting Methods ------------------------
    def plot_dose_response(self, fig=None, axes=None, alpha=0.5):
        """
        Plots the dose response of the IOCRN for each output species. The dose response for each output species, for each input scenario is plotted versus the input dose.
        Arguments:
        - fig: matplotlib figure object to plot on. If None, a new figure is created.
        - axes: matplotlib axes object to plot on. If None, a new set of axes is created.
        - alpha: float, transparency level for the plot lines.
        Returns:
        - fig: matplotlib figure object containing the plots.
        - axes: matplotlib axes object containing the plots.
        """
        # Check if dose response data is available
        if self.last_task_info.get('type') != 'dose response':
            raise ValueError("No dose response data available. Run dose_response() first.")
        
        # If no figure or axes are provided, create a new figure and axes
        if fig is None and axes is None:
            fig, axes = plt.subplots(self.num_outputs, 1, figsize=(10, 5 * self.num_outputs))
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
        
        # Plot the dose responses for each output species and return the figure and axes
        u_dose = self.last_task_info['input doses']
        for i in range(self.num_outputs):
            for j in range(len(self.last_task_info['input scenarios'])):
                axes[i].plot(u_dose, self.last_task_info['outputs'][j][i,:], alpha=alpha)
                axes[i].set_title(f"Dose Response of Output Species {self.species_labels[self.o[i]-1]}")
                axes[i].set_xlabel("Input Dose")
                axes[i].set_ylabel("Concentration")
        plt.tight_layout()
        return fig, axes
    
    def plot_transient_response(self, fig=None, axes=None, alpha=0.1):
        """
        Plots the transient response of the IOCRN for each output species.
        Arguments:
        - fig: matplotlib figure object to plot on. If None, a new figure is created.
        - axes: matplotlib axes object to plot on. If None, a new set of axes is created.
        - alpha: float, transparency level for the plot lines.
        Returns:
        - fig: matplotlib figure object containing the plots.
        - axes: matplotlib axes object containing the plots.
        """
        # Check if transient response data is available
        if self.last_task_info.get('type') != 'transient response':
            raise ValueError("No transient response data available. Run transient_response() first.")
        
        # If no figure or axes are provided, create a new figure and axes
        if fig is None and axes is None:
            fig, axes = plt.subplots(self.num_outputs, 1, figsize=(10, 5 * self.num_outputs))
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
        
        # Plot the transient response for each output species and return the figure and axes
        for i in range(self.num_outputs):
            for j in range(len(self.last_task_info['outputs'])):
                axes[i].plot(self.last_task_info['time_horizon'], self.last_task_info['outputs'][j][i,:], alpha=alpha)
                axes[i].set_title(f"Transient Response of Output Species {self.species_labels[self.o[i]-1]}")
                axes[i].set_xlabel("Time")
                axes[i].set_ylabel("Concentration")
        plt.tight_layout()
        return fig, axes

    # ------------------------ Symbolic Operations ------------------------
    def symbolic_propensity_function(self, x, c, u):
        """
        Computes the symbolic propensity function for the IOCRN given symbolic variables for concentrations x, rate constants c, and inputs u.
        Arguments:
        - x: A list of symbolic variables representing the concentrations of the species.
        - c: A list of symbolic variables representing the rate constants for each reaction.
        - u: A list of symbolic variables representing the inputs to the IOCRN.
        Returns:
        - A sympy Matrix of shape (m, 1) representing the propensity of each reaction.
        """
        S_R = sp.Matrix(self.S_R)
        S_I = sp.Matrix(self.S_I)
        propensity = sp.zeros(self.m, 1)
        for j in range(self.m):
            propensity_reactants = sp.prod([c**s for c, s in zip(x, S_R[:, j])])
            propensity_inputs = sp.prod([i**inf for i, inf in zip(u, S_I[:, j])])
            propensity[j] = c[j] * propensity_reactants * propensity_inputs
        return propensity

    def print_odes(self, x, c, u):
        """
        Prints the ODEs of the IOCRN given symbolic variables for concentrations x, rate constants c, and inputs u.
        Arguments:
        - x: A list of symbolic variables representing the concentrations of the species.
        - c: A list of symbolic variables representing the rate constants for each reaction.
        - u: A list of symbolic variables representing the inputs to the IOCRN.
        Returns:
        - A sympy Matrix of shape (n, 1) representing the ODEs for each species.
        """
        S = sp.Matrix(self.S)
        prop = self.symbolic_propensity_function(x, c, u)
        ODEs = S * prop
        sp.pprint(ODEs)
        return ODEs
    
    def linearize_odes(self, species, parameters, inputs):
        S = sp.Matrix(self.S)
        prop = self.symbolic_propensity_function(species, parameters, inputs)
        ODEs = S * prop
        A = ODEs.jacobian(species)
        B = ODEs.jacobian(inputs)
        return A, B

    # ------------------------ Mapping functions ------------------------
    def map_complex_to_species(self, idx):
        """
        Maps a complex index to the indices of the two species that form the complex.
        Arguments:
        - idx: Index of the complex, an integer in the range [0, n*(n+3)/2]. Can be numpy integer, int, or torch tensor.
        Returns:
        - species1_idx: Index of the first species in the complex. Type: numpy integer
        - species2_idx: Index of the second species in the complex. Type: numpy integer.
        By construction, species1_idx <= species2_idx.
        """
        species1_idx = np.floor((2*self.n + 3 - np.sqrt((2*self.n + 3)**2 - 8 * idx)) / 2).astype(np.int64)
        species2_idx = (idx - (species1_idx * (2*self.n + 1 - species1_idx)) / 2).astype(np.int64)
        return species1_idx, species2_idx
    
    def map_species_to_complex(self, species1_idx, species2_idx):
        """
        Maps the indices of two species to the index of the complex they form.
        Arguments:
        - species1_idx: Index of the first species in the complex. Type: numpy integer, or int.
        - species2_idx: Index of the second species in the complex. Type: numpy integer, or int.
        Returns:
        - idx: Index of the complex formed by the two species. Type: numpy integer.
        The species indices must satisfy 0 <= species1_idx <= species2_idx < n.
        """
        if species1_idx > species2_idx:
            raise ValueError("species1_idx must be less than or equal to species2_idx.")
        return ((species1_idx * (2*self.n + 1 - species1_idx)) / 2 + species2_idx).astype(np.int64)
    
    def map_reaction_to_complexes(self, idx):
        """
        Maps a reaction index to the indices of the reactant complex and product complex that form the reaction.
        Arguments:
        - idx: Index of the reaction, an integer in the range [0, n(n+1)(n+3)(n+4)/4]. Can be numpy integer, int, or torch tensor.
        Returns:
        - reactants_idx: Index of the reactants complex. Type: numpy integer.
        - products_idx: Index of the products complex. Type: numpy integer.
        The reactants_idx and products_idx are in the range [0, n(n+1)/2].
        """
        N = self.get_complexes_range() - 1
        if idx <= N:
            reactants_idx = 0
            products_idx = idx
        else: 
            reactants_idx = np.int64(np.floor((idx - N - 1)/N) + 1)
            remainder = np.int64((idx - N - 1) % N)
            products_idx = remainder + (remainder >= reactants_idx).astype(np.int64)
        return reactants_idx, products_idx
    
    def map_complexes_to_reaction(self, reactants_idx, products_idx):
        """
        Maps the indices of the reactants complex and products complex to the index of the reaction they form.
        Arguments:
        - reactants_idx: Index of the reactants complex. Type: numpy integer, or int.
        - products_idx: Index of the products complex. Type: numpy integer, or int.
        Returns:
        - reaction_idx: Index of the reaction formed by the two complexes. Type: numpy integer.
        """
        N = self.get_complexes_range() - 1
        if reactants_idx == 0:
            reaction_idx = products_idx
        else:
            reaction_idx = np.int64(reactants_idx * N + products_idx + 1 - (products_idx > reactants_idx))
        return reaction_idx
    
    def map_reaction_to_species(self, idx):
        """
        Maps a reaction index to the indices of each of the two reactants and two products.
        Arguments:
        - idx: Index of the reaction, an integer in the range [0, n(n+1)(n+3)(n+4)/4]. Can be numpy integer, int, or torch tensor.
        Returns:
        - reactant1_idx: Index of the first reactant species. Type: numpy integer.
        - reactant2_idx: Index of the second reactant species. Type: numpy integer.
        - product1_idx: Index of the first product species. Type: numpy integer.
        - product2_idx: Index of the second product species. Type: numpy integer.
        The species indices satisfy reactant1_idx <= reactant2_idx and product1_idx <= product2_idx.
        """
        reactants_idx, products_idx = self.map_reaction_to_complexes(idx)
        reactant1_idx, reactant2_idx = self.map_complex_to_species(reactants_idx)
        product1_idx, product2_idx = self.map_complex_to_species(products_idx)
        return reactant1_idx, reactant2_idx, product1_idx, product2_idx
    
    def map_species_to_reaction(self, reactant1_idx, reactant2_idx, product1_idx, product2_idx):
        """
        Maps the indices of each of the two reactants and two products to the index of the reaction they form.
        Arguments:
        - reactant1_idx: Index of the first reactant species. Type: numpy integer, or int.
        - reactant2_idx: Index of the second reactant species. Type: numpy integer, or int.
        - product1_idx: Index of the first product species. Type: numpy integer, or int.
        - product2_idx: Index of the second product species. Type: numpy integer, or int.
        Returns:
        - reaction_idx: Index of the reaction formed by the two reactants and two products. Type: numpy integer.
        The species indices must satisfy reactant1_idx <= reactant2_idx and product1_idx <= product2_idx.
        """
        reactants_idx = self.map_species_to_complex(reactant1_idx, reactant2_idx)
        products_idx = self.map_species_to_complex(product1_idx, product2_idx)
        return self.map_complexes_to_reaction(reactants_idx, products_idx)
    
    def map_species_to_stoichiometry_vector(self, species1_idx, species2_idx):
        """
        Maps the indices of two species to a stoichiometry vector for a reaction with those two species as reactants or products.
        Arguments:
        - species1_idx: Index of the first species. Type: numpy integer, or int.
        - species2_idx: Index of the second species. Type: numpy integer, or int.
        Returns:
        - stoichiometry_vector: A numpy array of shape (n,) representing the stoichiometry vector for the reaction with the two species.
        The species indices must satisfy 0 <= species1_idx <= species2_idx < n.
        """
        stoichiometry_vector = np.zeros(self.n, dtype=np.int64)
        if species1_idx > 0:
            stoichiometry_vector[np.int64(species1_idx - 1)] += 1
        if species2_idx > 0:
            stoichiometry_vector[np.int64(species2_idx - 1)] += 1
        return stoichiometry_vector

    def map_stoichiometry_matrix_to_species(self, stoichiometry):
        """
        Maps a stoichiometry matrix to the indices of the two species that form each reaction.
        Arguments:
        - stoichiometry: A numpy array of shape (n, m) representing the stoichiometry matrix of the reactions.
        Returns:
        - species_indices: A numpy array of shape (2, m) where each column contains the indices of the two species that form the reaction.
        The species indices are in the range [0, n-1] and are sorted in ascending order.
        The first row contains the index of the first species, and the second row contains the index of the second species.
        If a reaction has only one species, the first row will contain 0 and the second row will contain the index of that species.
        If a reaction has no species, both rows will contain 0.
        """
        n_species, n_reactions = stoichiometry.shape
        species_indices = np.zeros((2, n_reactions), dtype=np.int64)
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
    
    def map_stoichiometry_to_complexes(self, stoichiometry):
        """
        Maps a stoichiometry matrix to the indices of the complexes that form each reaction.
        Arguments:
        - stoichiometry: A numpy array of shape (n, m) representing the stoichiometry matrix of reactants or products.
        Returns:
        - complexes_indices: A numpy array of shape (m,) where each element is the index of the complex that forms the reactants or products.
        """
        species_indices = self.map_stoichiometry_matrix_to_species(stoichiometry)
        complexes_indices = np.zeros(self.m, dtype=np.int64)
        for i in range(self.m):
            complexes_indices[i] = self.map_species_to_complex(species_indices[0, i], species_indices[1, i])
        return complexes_indices
    
    def map_stoichiometry_to_reactions(self, S_R, S_P):
        """
        Maps the stoichiometry matrices of reactants and products to the indices of the reactions they form.
        Arguments:
        - S_R: A numpy array of shape (n, m) representing the stoichiometry matrix of reactants.
        - S_P: A numpy array of shape (n, m) representing the stoichiometry matrix of products.
        Returns:
        - reactions_indices: A numpy array of shape (m,) where each element is the index of the reaction formed by the reactants and products.
        """
        reactants_indices = self.map_stoichiometry_to_complexes(S_R)
        products_indices = self.map_stoichiometry_to_complexes(S_P)
        reactions_indices = np.zeros(self.m, dtype=np.int64)
        for i in range(self.m):
            reactions_indices[i] = self.map_complexes_to_reaction(reactants_indices[i], products_indices[i])
        return reactions_indices
    
    def map_input_influence_matrix_to_reactions(self, reactions_indices, input_influence_matrix):
        """
        Maps the input influence matrix to the indices of the reactions that are influenced by each input.
        Arguments:
        - reactions_indices: A numpy array of shape (m,) where each element is the index of the reaction.
        - input_influence_matrix: A numpy array of shape (p, m) representing the input influence matrix.
        Returns:
        - split_arrays: A list of numpy arrays, where each array contains the indices of the reactions influenced by the corresponding input.
        If an input does not influence any reactions, the corresponding array is empty.
        """
        num_inputs = input_influence_matrix.shape[0]
        u_idx, r_idx = np.nonzero(input_influence_matrix)
        categories_idx = reactions_indices[r_idx]
        if u_idx.size == 0:
            return [np.array([], dtype=np.int64) for _ in range(num_inputs)]
        sort_order = np.argsort(u_idx)
        u_sorted = u_idx[sort_order]
        cats_sorted = categories_idx[sort_order]
        counts = np.bincount(u_sorted, minlength=num_inputs)
        split_arrays = np.split(cats_sorted, np.cumsum(counts[:-1]))
        return split_arrays