import numpy as np
import sympy as sp
from itertools import product
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import copy
from scipy.integrate import solve_ivp

class IOCRN:
    """ A class representing a general Input-Output Chemical Reaction Network (IOCRN). """

    # ------------------------ Construction Methods ------------------------
    def __init__(self, reactions, output_labels):
        """ Initialize an IOCRN with the given reactions and output labels.
        Arguments:
        - reactions: List of Reaction objects.
        - outputs_labels: List of strings representing the labels of the outputs in the IOCRN. 
        Compile must be called after initialization to set up the internal representations of the IOCRN.
        """
        self.compiled = False                   # flag indicating whether the IOCRN has been compiled
        
        # Record the reactions, output labels, and number of outputs
        self.reactions = reactions              # list of Reaction objects
        self.output_labels = output_labels      # list of strings of output species
        self.num_outputs = len(output_labels)   # number of outputs

        # Get the number of unknown parameteres in the IOCRN
        self.num_unknown_params = sum([reaction.num_unknown_params for reaction in self.reactions])

        # Initialize a dictionary to store the last task information
        self.last_task_info = {}
        self.last_task_info['type'] = None

    def clone(self):
        return copy.deepcopy(self)
    
    def reset(self):
        """ Resets the IOCRN to its initial state by clearing the last task information. 
        This method does not modify the reactions or their parameters. 
        """
        self.last_task_info = {}
        self.last_task_info['type'] = None

    def add_reaction(self, reaction):
        """ Add a reaction to the IOCRN. It does not update the internal representations of the IOCRN. 
        Call compile to update its internal representations.    
        Arguments:
        - reaction: A Reaction object to be added to the IOCRN. 
        """
        # Flush the last task information
        self.last_task_info = {}
        self.last_task_info['type'] = None

        # Add the reaction to the list of reactions
        self.reactions.append(reaction)
        self.compiled = False

    def retrieve_species(self):
        """ Retrieve and store the labels of the species present in the IOCRN and their corresponding indices in the IOCRN. 
        """
        # Get labels of all the species involved in the IOCRN, and sort them alphanumerically
        self.species_labels = list(set(sum([reaction.get_involved_species() for reaction in self.reactions], []))) # list of strings of all species
        self.species_labels.sort() 

        # Create a mapping from species labels to their indices
        self.species_idx_dict = {species_label: idx for idx, species_label in enumerate(self.species_labels)} # dictionary mapping species labels to indices

    def retrieve_input(self):
        """ Retrieve and store the input labels and their corresponding indices in the IOCRN. 
        """
        # Get labels of all the species involved in the IOCRN, and sort them alphanumerically
        self.input_labels = list(set(sum([r.get_involved_inputs() for r in self.reactions], []))) # list of strings of all inputs
        self.input_labels.sort()
        self.num_inputs = len(self.input_labels) # number of inputs

        # Create a mapping from input labels to their indices
        self.input_idx_dict = {input_label: idx for idx, input_label in enumerate(self.input_labels)} # dictionary mapping input labels to indices

    def set_unknown_parameters(self, params): #TODO: test this method
        params = np.array(params).flatten()
        for reaction in self.reactions:
            params = reaction.set_unknown_parameters(params)
        
        if len(params) > 0:
            raise Exception(f"Error: {len(params)} parameters were not set - they are still unknown.")
        
        self.compiled = False                

    def species_label_to_idx(self, labels):
        """ Map species labels to their corresponding indices.
        Arguments:
        - labels: A string or a list of strings representing species labels.
        Returns:
        - A single index if a string is provided, or a list of indices if a list of strings is provided. 
        """
        if isinstance(labels, str):
            return self.species_idx_dict[labels] # single index
        return [self.species_idx_dict[label] for label in labels] # list of indices
    
    def input_label_to_idx(self, labels):
        """ Map input labels to their corresponding indices.
        Arguments:
        - labels: A string or a list of strings representing input labels.
        Returns:
        - A single index if a string is provided, or a list of indices if a list of strings is provided. 
        """
        if isinstance(labels, str):
            return self.input_idx_dict[labels] # single index
        return [self.input_idx_dict[l] if l is not None else None for l in labels] # list of indices
    
    def set_library_context(self, reaction_library):
        for reaction in self.reactions:
            reaction.set_library_context(reaction_library)

    def get_stoichiometry_matrix(self):
        """ Construct and return the stoichiometry matrix S of the IOCRN.
        Returns:
        - S: A numpy array of shape (number of species, number of reactions) representing the stoichiometry matrix.
        """
        self.num_species = len(self.species_labels)                 # number of species
        self.num_reactions = len(self.reactions)                    # number of reactions
        S = np.zeros((self.num_species, self.num_reactions))        # numpy array of shape (number of species, number of reactions)
        for j, r in enumerate(self.reactions):
            stoich_dict = r.get_stoichiometry_dict()
            for species, value in stoich_dict.items():
                i = self.species_label_to_idx(species)
                S[i, j] = value
        return S
    
    def gather_reaction_IDs(self):
        """
        Gather and return the IDs of all reactions in the IOCRN.
        Returns:
        - reaction_IDs: A list of integers representing the IDs of the reactions.
        """
        reaction_IDs = [reaction.ID for reaction in self.reactions]
        return reaction_IDs
    
    def gather_reaction_params(self):
        """
        Gather and return the parameters of all reactions in the IOCRN.
        Returns:
        - reaction_params: A list of lists of parameters for all reactions.
        """
        reaction_params = []
        for reaction in self.reactions:
            reaction_params.append(reaction.params)
        return reaction_params

    def compile(self):
        """
        Compile the IOCRN by setting up species, inputs, stoichiometry matrix, and reaction contexts.   
        This method should be called after adding all reactions and before simulating the IOCRN.
        """
        # Compile the species and input labels and indices
        self.retrieve_species()
        self.retrieve_input()

        # Compile the output indices
        self.output_idx = np.array(self.species_label_to_idx(self.output_labels)) # np array of indices for the output species

        # Compile the stoichiometry matrix and the number of species and reactions
        self.S = self.get_stoichiometry_matrix()

        # Compile the number of reactions
        self.num_reactions = len(self.reactions)       

        # Set the context for each reaction by mapping reactant, product, and input labels to their indices in the context of the IOCRN
        for reaction in self.reactions:
            reaction.set_crn_context(self)

        # Set the compiled flag to True
        self.compiled = True
    
    # ------------------------ Printing Methods ------------------------
    def __str__(self):
        """ Return a string representation of the IOCRN, including inputs, species, output species, and reactions. When print is called, this method is invoked.
        """
        out = f'Inputs: {self.input_labels} \n'
        out += f'Species: {self.species_labels} \n'
        out += f'Output Species: {self.output_labels} \n'
        out += '\n'.join([str(r) for r in self.reactions])
        return out
    
    # ------------------------ Computation Methods ------------------------
    def propensity_function(self, x, u):
        """ Compute the propensity vector for the current state x and input u.
        Arguments:
        - x: numpy array of species counts.
        - u: numpy array of input values.
        Returns:
        - propensities: numpy array of propensities for all the reactions.
        """
        propensities = np.array([r.propensity(x, u) for r in self.reactions])
        return propensities
    
    def rate_function(self, t, x, u):
        """ Computes the rate of change of concentrations for the IOCRN given time t, concentrations x, and inputs u.
        Arguments:
        - t: float representing the current time.
        - x: numpy array of shape (n,) representing the concentrations of the species.
        - u: numpy array of shape (p,) representing the inputs to the IOCRN.
        Returns:
        - A numpy array of shape (n,) representing the rate of change of concentrations.
        """
        return np.matmul(self.S, self.propensity_function(x, u))
    
    def transient_response(self, u_list, x0_list, time_horizon, LARGE_NUMBER=1e4):
        """ Computes the transient response of the IOCRN given a list of inputs, a list of initial conditions, and a time horizon. 
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
        if self.num_unknown_params > 0:
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
                x = np.full((self.num_species, time_horizon.shape[0]), LARGE_NUMBER) # numpy array of shape (n, steps)
            else:
                x = solution.y # numpy array of shape (n, steps)
                if solution.status == 1: # if the integration was stopped due to an event, fill the remaining time points after the event with large numbers
                    x = np.concatenate([x, np.full((self.num_species, time_horizon.shape[0] - x.shape[1]), LARGE_NUMBER)], axis=1)
            y = x[self.output_idx, :] # select the output species from the state trajectory

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
    def plot_transient_response(self, fig=None, axes=None, alpha=0.1):
        """ Plots the transient response of the IOCRN for each output species.
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
                axes[i].set_title(f"Transient Response of Output Species {self.species_labels[self.output_idx[i]]}")
                axes[i].set_xlabel("Time")
                axes[i].set_ylabel("Concentration")
        plt.tight_layout()
        return fig, axes