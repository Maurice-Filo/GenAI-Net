import numpy as np
import sympy as sp
from itertools import product
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import copy
from scipy.integrate import solve_ivp
import pandas as pd

from RL4CRN.utils.stochastic import quick_measurement_SSA

try:
    from StochasticSimulationsNew.ReactionNetworkLanguage import make_parser
except ImportError:
    print("Warning: StochasticSimulationsNew package not found. SSA functionality will be unavailable.")

class IOCRN:
    """ A class representing a general Input-Output Chemical Reaction Network (IOCRN). """

    # ------------------------ Construction Methods ------------------------
    def __init__(self, reactions, output_labels, solver='CVODE', atol=1e-6, rtol=1e-3): 
        """ Initialize an IOCRN with the given reactions and output labels.
        Arguments:
        - reactions: List of Reaction objects.
        - outputs_labels: List of strings representing the labels of the outputs in the IOCRN. 
        - solver: String representing the ODE solver to be used for simulations. Default is 'CVODE'. alternative is 'LSODA'.
        - atol: Float representing the absolute tolerance for the ODE solver. Default is 1e-6. (as in scipy.solve_ivp)
        - rtol: Float representing the relative tolerance for the ODE solver. Default is 1e-3. (as in scipy.solve_ivp)
        Compile must be called after initialization to set up the internal representations of the IOCRN.
        """
        
        # Record the reactions, output labels, and number of outputs
        self.reactions = reactions              # list of Reaction objects
        self.output_labels = output_labels      # list of strings of output species
        self.num_outputs = len(output_labels)   # number of outputs
        self.reaction_library = None

        # Get the number of unknown parameteres in the IOCRN
        self.num_unknown_params = sum([reaction.num_unknown_params for reaction in self.reactions])

        # Initialize a dictionary to store the last task information
        self.last_task_info = {}
        self.last_task_info['type'] = None

        self.solver = solver
        self.atol = atol
        self.rtol = rtol

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
        self.reset()

        # Add the reaction to the list of reactions
        self.reactions.append(reaction)
        self.compile()

    def retrieve_species(self):
        """ Retrieve and store the labels of the species present in the IOCRN and their corresponding indices in the IOCRN. """
        # Get labels of all the species involved in the IOCRN, and sort them alphanumerically
        self.species_labels = list(set(sum([reaction.get_involved_species() for reaction in self.reactions], []))) # list of strings of all species
        for output_label in self.output_labels:
            if output_label not in self.species_labels:
                self.species_labels.append(output_label)
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
        """ Set the context for each reaction in the IOCRN using the provided reaction library. """
        for reaction in self.reactions:
            reaction.set_library_context(reaction_library)
        self.reaction_library = reaction_library

    def get_bool_signature(self):
        """ Get the boolean signature of the IOCRN with respect to the provided reaction library. """
        IDs = self.gather_reaction_IDs()
        M = len(self.reaction_library)
        signature = np.zeros(M, dtype=bool)
        signature[IDs] = True
        return signature
    
    def is_topologically_equal(self, other_iocrn):
        """ Compare the topology of this IOCRN with another IOCRN. Assumes both IOCRNs have the same reaction library set.
        Arguments:
        - other_iocrn: Another IOCRN object to compare with.
        Returns:
        - True if the topologies are the same, False otherwise. """
        
        return np.array_equal(self.get_bool_signature(), other_iocrn.get_bool_signature())

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
        """ Gather and return the IDs of all reactions in the IOCRN.
        Returns:
        - reaction_IDs: A list of integers representing the IDs of the reactions. """
        reaction_IDs = [reaction.ID for reaction in self.reactions]
        return reaction_IDs
    
    def gather_reaction_params(self):
        """ Gather and return the parameters of all reactions in the IOCRN.
        Returns:
        - reaction_params: A list of lists of parameters for all reactions. """

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
    
    # ------------------------ Printing Methods ------------------------
    def __str__(self):
        """ Return a string representation of the IOCRN, including inputs, species, output species, and reactions. When print is called, this method is invoked.
        """
        try:
            reaction_signatures = [r.ID for r in self.reactions]
            # sort reactions by their signatures
            ordered_reactions = [r for _, r in sorted(zip(reaction_signatures, self.reactions))]
        except:
            print("Warning: no reaction IDs found, printing reactions in original order.")
            ordered_reactions = self.reactions

        out = f'Inputs: {self.input_labels} \n'
        out += f'Species: {self.species_labels} \n'
        out += f'Output Species: {self.output_labels} \n'
        out += '\n'.join([str(r) for r in ordered_reactions])
        return out
    
    def to_reaction_file(self):
        """ 
        Converts the current IOCRN into a string format compatible with the custom DSL. 
        Returns:
            str: The text content of the reaction file.
        """
        lines = []
        
        # 1. Inputs Section
        if hasattr(self, 'input_labels') and self.input_labels:
            lines.append("// --- Inputs (External Signals) ---")
            for inp in self.input_labels:
                lines.append(f"input {inp};")
            lines.append("")

        # 2. Species Section
        lines.append("// --- Species Definition ---")
        
        # Ensure species list is populated
        if not hasattr(self, 'species_labels') or self.species_labels is None:
            self.retrieve_species()

        # Exclude 'emptyset' or '∅' from definitions
        real_species = [s for s in self.species_labels if s not in ['emptyset', '∅']]
        if real_species:
            lines.append(f"species {', '.join(real_species)} = 0;")
        lines.append("")

        # 3. Reactions Section
        lines.append("// --- Reactions ---")
        
        for r in self.reactions:
            # Delegate the formatting logic to the specific Reaction class
            lines.append(r.to_reaction_format())

        return "\n".join(lines)
    
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
    

    def transient_response_SSA(self, u_list, x0_list, time_horizon, n_trajectories=100, max_threads=10000, max_value=1e6):
        """ 
        Computes the stochastic transient response of the IOCRN using SSA.
        Equivalent to transient_response but returns Mean and Std Dev of stochastic trajectories.
        
        Arguments:
        - u_list: List of input vectors (numpy arrays).
        - x0_list: List of initial condition vectors (numpy arrays).
        - time_horizon: Numpy array of time points.
        - n_trajectories: Number of stochastic runs per configuration (to compute mean/std).
        
        Returns:
            time_horizon, x_mean_list, y_mean_list, x_std_list, y_std_list, last_task_info
        """
        
        # 1. Generate DSL and Parse CRN
        # We need to convert the current object state to the DSL format required by the SSA engine
        crn_text = self.to_reaction_file()
        
        # Assuming make_parser() is available globally or imported
        parser, lexer = make_parser() 
        ssa_crn = parser.parse(crn_text)

        # 2. Setup Simulation Parameters
        t_fin = time_horizon[-1]
        
        # We need to map the time_horizon steps to the SSA 't_step'
        # The SSA engine usually takes a fixed step for recording. 
        # We calculate the average step size from the horizon.
        if len(time_horizon) > 1:
            t_step = float(time_horizon[1] - time_horizon[0])
        else:
            t_step = t_fin / 100.0

        # 3. Prepare Parameter Sets (Cartesian Product of u_list and x0_list)
        # The SSA backend likely expects parameters as a flat list or dictionary.
        # We need to check how your SSA backend handles initial conditions.
        # If your SSA backend 'spread_parameter_sets_among_gpus' only handles reaction rates/inputs,
        # we might need to handle x0 separately. 
        
        # However, typically SSA wrappers allow setting initial species counts.
        # If your SSA implementation (which I don't fully see here) doesn't support 
        # varying x0 per thread block easily, we iterate over x0_list in the outer loop 
        # and batch u_list. But for efficiency, let's assume we can batch inputs.
        
        # Let's map u_list to the 'parameters' expected by the DSL (e.g. u_1, u_2...)
        # Note: This assumes the DSL input order matches u_list indexing.
        
        configurations = list(product(u_list, x0_list))
        
        # We will store results here
        x_mean_list = []
        x_std_list = []
        y_mean_list = []
        y_std_list = []

        # 4. Run Simulation
        # Since the SSA backend might handle data differently, we call the helper 
        # 'quick_measurement_SSA' we defined earlier, but we need to adapt it 
        # because we are varying Initial Conditions (x0) as well.
        
        # If the SSA backend doesn't support varying x0 explicitly in the parameter list, 
        # we might need to run separate batches for each x0. 
        # Assuming 'quick_measurement_SSA' or 'SSA' takes (u1, u2...) but defaults x0 to 0.
        
        # Strategy: Run a loop for each unique Initial Condition set x0
        # and run the batch of all Inputs u for that x0.
        
        # print(f"Running SSA for {len(x0_list)} initial conditions and {len(u_list)} input profiles...")

        for x0_idx, x0 in enumerate(x0_list):
            
            # create species dictionary
            ic_dict = {}
            for s_idx, s_label in enumerate(self.species_labels):
                ssa_crn.species[s_label].value = x0[s_idx]

            # Prepare input parameters for this batch (just the inputs u)
            # The backend expects tuples of parameter values corresponding to 'input ...;' lines
            param_batch = [tuple(u) for u in u_list]
            
            # Use the helper function we made (or call SSA directly)
            # We explicitly ask for ALL species to compute full state trajectories
            summary_df = quick_measurement_SSA(
                ssa_crn, 
                param_batch, 
                t_fin=t_fin, 
                n_trajectories=n_trajectories, 
                max_threads=max_threads,
                t_step=t_step,
                species_to_measure=self.species_labels, # Measure everything
                max_value=max_value
            )
            
            # 5. Extract and Reshape Data
            # The summary_df has columns: [time, u_1, u_2, ..., (Species, mean), (Species, std)]
            # We need to sort it to ensure we match the order of u_list
            
            input_names = [f"u_{k+1}" for k in range(len(u_list[0]))] # Guessing input naming convention from parser
            # If the parser uses specific names, we should match them.
            # Assuming `quick_measurement_SSA` handles the column mapping.
            
            # Iterate through the inputs in the *same order* as u_list to populate the output lists
            for u_vec in u_list:
                # Filter DF for this specific input combination
                # logic to match u_vec to columns 'u_1', 'u_2' etc.
                mask = pd.Series(True, index=summary_df.index)
                
                # We assume the columns in DF are named based on input definitions.
                # We need to map index of u_vec to column name. 
                # self.input_labels should hold ['u_1', 'u_2'] sorted.
                for k, val in enumerate(u_vec):
                    col_name = self.input_labels[k]
                    # Handle float matching tolerance if needed, or exact match
                    mask &= (np.isclose(summary_df[col_name], val))
                
                subset = summary_df[mask].sort_values('time')
                
                # Interpolate to match exact 'time_horizon' requested?
                # The SSA returns data at 't_step'. If 'time_horizon' doesn't match perfectly,
                # we should interpolate.
                
                # Helper to interpolate a species column
                def get_interp_traj(stat_type):
                    # shape (n_species, n_time_points)
                    traj = np.zeros((self.num_species, len(time_horizon)))
                    for s_idx, s_label in enumerate(self.species_labels):
                        # s_label might be complex in DF keys
                        # summary_df keys are often tuples (Label, 'mean')
                        if (s_label, stat_type) in subset.columns:
                            vals = subset[(s_label, stat_type)].values
                            t_sim = subset['time'].values
                            traj[s_idx, :] = np.interp(time_horizon, t_sim, vals)
                    return traj

                x_mean = get_interp_traj('mean')
                x_std = get_interp_traj('std')
                
                # Extract outputs
                y_mean = x_mean[self.output_idx, :]
                y_std = x_std[self.output_idx, :]
                
                x_mean_list.append(x_mean)
                x_std_list.append(x_std)
                y_mean_list.append(y_mean)
                y_std_list.append(y_std)

        # 6. Store and Return Results
        self.last_task_info = {
            'type': 'transient response SSA',
            'inputs': u_list,
            'initial conditions': x0_list,
            'time_horizon': time_horizon,
            'trajectories': x_mean_list,
            'trajectories_std': x_std_list,
            'outputs': y_mean_list,
            'outputs_std': y_std_list
        }
        
        return time_horizon, x_mean_list, y_mean_list, x_std_list, y_std_list, self.last_task_info
    
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
            - last_task_info: A dictionary containing information about the last task performed, including inputs, initial conditions, time horizon, trajectories, and outputs.
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

        
        if self.solver == 'LSODA':
            for u, x0 in product(u_list, x0_list):
                solution = solve_ivp(lambda t, x: self.rate_function(t, x, u), (time_horizon[0], time_horizon[-1]), x0, t_eval=time_horizon, method="LSODA", events=stop_if_unstable, atol=self.atol, rtol=self.rtol)
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

        elif self.solver == 'CVODE':

            for u, x0 in product(u_list, x0_list):

                solution = self.solve_with_cvode(
                    x0,
                    time_horizon,
                    u,
                    nonneg_idx=np.arange(len(x0)),
                    stop_fn=stop_if_unstable,
                )

                T = time_horizon.shape[0]
                if solution.status < 0 or not solution.raw.success:
                    # integration failed → fill with LARGE_NUMBER
                    x = np.full((self.num_species, T), LARGE_NUMBER)
                else:
                    t_sol = np.asarray(solution.t, dtype=float)           # (n_t,)
                    y_sol = np.asarray(solution.y, dtype=float)           # (n_species, n_t)

                    x = np.empty((self.num_species, T), dtype=float)

                    # last time actually reached by CVODE
                    t_last = t_sol[-1]
                    mask = time_horizon <= t_last
                    mask_rest = ~mask

                    # interpolate each species onto the grid up to t_last
                    for i in range(self.num_species):
                        x[i, mask] = np.interp(time_horizon[mask], t_sol, y_sol[i, :])
                        # fill remainder (beyond last CVODE time) with LARGE_NUMBER
                        x[i, mask_rest] = LARGE_NUMBER

                y = x[self.output_idx, :]

                x_list.append(x)
                y_list.append(y)
        else:
            raise ValueError(f"Unknown solver '{self.solver}'. Supported solvers are 'LSODA' and 'CVODE'.")

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
        - axes: matplotlib axes object containing the plots. """

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
    
    def plot_phase_portrait(self, fig=None, axis=None, alpha=0.1):
        """ Plots the phase portrait of the IOCRN.
        Arguments:
        - fig: matplotlib figure object to plot on. If None, a new figure is created.
        - axes: matplotlib axis object to plot on. If None, a new axis is created.
        - alpha: float, transparency level for the plot lines.
        Returns:
        - fig: matplotlib figure object containing the plots.
        - axes: matplotlib axes object containing the plots. """

        # Check if transient response data is available
        if self.last_task_info.get('type') != 'transient response':
            raise ValueError("No transient response data available. Run transient_response() first.")
        
        # If no figure or axes are provided, create a new figure and axes
        if fig is None and axis is None:
            if self.num_species == 2:
                fig, axis = plt.subplots(figsize=(10, 10))
            elif self.num_species == 3:
                fig = plt.figure(figsize=(10, 10))
                axis = fig.add_subplot(111, projection='3d')
            else:
                raise ValueError("Phase portrait can only be plotted for 2 or 3 species.")
        
        # Plot the phase portrait and return the figure and axes
        if self.num_species == 2:
            for j in range(len(self.last_task_info['trajectories'])):
                axis.plot(self.last_task_info['trajectories'][j][0,:], self.last_task_info['trajectories'][j][1,:], alpha=alpha)
            axis.set_xlabel(f"Species {self.species_labels[0]}")
            axis.set_ylabel(f"Species {self.species_labels[1]}")
            axis.set_title("Phase Portrait")
        elif self.num_species == 3:
            for j in range(len(self.last_task_info['trajectories'])):
                axis.plot(self.last_task_info['trajectories'][j][0,:], self.last_task_info['trajectories'][j][1,:], self.last_task_info['trajectories'][j][2,:], alpha=alpha)
            axis.set_xlabel(f"Species {self.species_labels[0]}")
            axis.set_ylabel(f"Species {self.species_labels[1]}")
            axis.set_zlabel(f"Species {self.species_labels[2]}")
            axis.set_title("Phase Portrait")
        plt.tight_layout()
        return fig, axis
    
    def plot_dose_response(self, fig=None, axes=None, alpha=0.5):
        """ Plots the dose response of the IOCRN for each output species. The dose response for each output species, for each input scenario is plotted versus the input dose.
        Arguments:
        - fig: matplotlib figure object to plot on. If None, a new figure is created.
        - axes: matplotlib axes object to plot on. If None, a new set of axes is created.
        - alpha: float, transparency level for the plot lines.
        Returns:
        - fig: matplotlib figure object containing the plots.
        - axes: matplotlib axes object containing the plots. """

        # Check if dose response data is available
        if self.last_task_info.get('type') != 'dose response' and self.last_task_info.get('type') != 'transient response':
            raise ValueError("No dose response data available. Run dose_response() or transient_response() first.")
        
        # If no figure or axes are provided, create a new figure and axes
        if fig is None and axes is None:
            fig, axes = plt.subplots(self.num_outputs, 1, figsize=(10, 5 * self.num_outputs))
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
        
        # Plot the dose responses for each output species and return the figure and axes #TODO: generalize for multiple inputs and implement dose response algebraically
        if self.last_task_info['type'] == 'dose response':
            u_dose = self.last_task_info['input doses']
            for i in range(self.num_outputs):
                for j in range(len(self.last_task_info['input scenarios'])):
                    axes[i].plot(u_dose, self.last_task_info['outputs'][j][i,:], alpha=alpha)
                    axes[i].set_title(f"Dose Response of Output Species {self.species_labels[self.o[i]-1]}")
                    axes[i].set_xlabel("Input Dose")
                    axes[i].set_ylabel("Concentration")
            plt.tight_layout()

        # elif self.last_task_info['type'] == 'transient response': # TODO: generalize for multiple inputs
        #     u_list = self.last_task_info['inputs']
        #     for i in range(self.num_outputs):
        #         u_dose = np.array([u[0] for u in u_list])
        #         y_dose = np.array([y[i,-1] for y in self.last_task_info['outputs']])
        #         axes[i].plot(u_dose, y_dose, alpha=alpha)
        #         axes[i].set_title(f"Dose Response of Output Species {self.species_labels[self.output_idx[i]]}")
        #         axes[i].set_xlabel("Input Dose")
        #         axes[i].set_ylabel("Concentration")
        #     plt.tight_layout()

        elif self.last_task_info['type'] == 'transient response': # TODO: Now generalized
            u_list = self.last_task_info['inputs']
            x0_list = self.last_task_info['initial_conditions']

            step = len(self.last_task_info['outputs']) // len(x0_list)

            for i in range(self.num_outputs):
                for k in enumerate(x0_list):
                    u_dose = np.array([u[0] for u in u_list])
                    y_dose = np.array([y[i,-1] for y in self.last_task_info['outputs'][step*(k):step*(k+1)]])
                    axes[i].plot(u_dose, y_dose, alpha=alpha)
                    axes[i].set_title(f"Dose Response of Output Species {self.species_labels[self.output_idx[i]]}")
                    axes[i].set_xlabel("Input Dose")
                    axes[i].set_ylabel("Concentration")
            plt.tight_layout()

        return fig, axes
    
    def plot_frequency_content(self, fig=None, axes=None, alpha=0.1, t0=0.0):
        """Plots the frequency content (Fourier magnitude spectra) of each output species.
        Arguments:
        - fig: matplotlib figure object to plot on. If None, a new figure is created.
        - axes: matplotlib axes object to plot on. If None, a new set of axes is created.
        - alpha: float, transparency level for the plot lines.
        - t0: float, time threshold; only data with time >= t0 are used for the Fourier transform.
        Returns:
        - fig: matplotlib figure object containing the plots.
        - axes: matplotlib axes object containing the plots. """

        # Check if transient response data is available
        if self.last_task_info.get('type') != 'transient response':
            raise ValueError("No transient response data available. Run transient_response() first.")
        
        # If no figure or axes are provided, create a new figure and axes
        if fig is None and axes is None:
            fig, axes = plt.subplots(self.num_outputs, 1, figsize=(10, 5 * self.num_outputs))
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]

        # Truncate time vector and determine valid indices for t >= t0
        time = np.asarray(self.last_task_info['time_horizon'])
        mask = time >= t0
        if mask.sum() < 2:
            raise ValueError("Not enough data points after t0 to compute a Fourier transform.")

        # Infer (assumed uniform) sampling interval from the truncated time vector
        dt = float(np.mean(np.diff(time[mask])))
        if dt <= 0:
            raise ValueError("Non-positive sampling interval inferred from time_horizon.")  

        # Plot Fourier magnitude spectra for each output species
        N = int(mask.sum())
        freqs = np.fft.rfftfreq(N, d=dt)

        for i in range(self.num_outputs):
            ax = axes[i]
            for j in range(len(self.last_task_info['outputs'])):
                # Extract the i-th output trace from the j-th run, truncated at t0
                y = np.asarray(self.last_task_info['outputs'][j][i, :])[mask]

                # Remove mean to emphasize oscillatory content
                y = y - np.mean(y)

                # Compute one-sided FFT magnitude
                Y = np.fft.rfft(y)
                mag = np.abs(Y) 
                mag = mag / (np.max(mag) + 1e-12) # simple magnitude normalization

                ax.plot(freqs, mag, alpha=alpha)

            ax.set_title(
                f"Frequency Content of Output Species {self.species_labels[self.output_idx[i]]} (t ≥ {t0})"
            )
            ax.set_xlabel("Frequency (1 / time unit)")
            ax.set_ylabel("Magnitude")

        plt.tight_layout()
        return fig, axes
    
    # ------------------------ stochastic simulation methods ------------------------

    def plot_SSA_transient_response(self, fig=None, axes=None, alpha=0.2):
        """ 
        Plots the stochastic transient response (Mean ± Std) of the IOCRN for each output species.
        
        Arguments:
        - fig: matplotlib figure object. If None, a new figure is created.
        - axes: matplotlib axes object. If None, a new set of axes is created.
        - alpha: float, transparency level for the standard deviation shading.
        
        Returns:
        - fig, axes
        """

        # 1. Validation
        if self.last_task_info.get('type') != 'transient response SSA':
            raise ValueError("No stochastic transient response data available. Run transient_response_SSA() first.")
        
        # 2. Setup Figure/Axes
        if fig is None and axes is None:
            fig, axes = plt.subplots(self.num_outputs, 1, figsize=(10, 5 * self.num_outputs))
            # Ensure axes is iterable even if there's only one output
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
        elif not isinstance(axes, (list, np.ndarray)):
             axes = [axes]
        
        # 3. Retrieve Data
        time = self.last_task_info['time_horizon']
        mean_data = self.last_task_info['outputs']      # List of (n_outputs, n_time)
        std_data = self.last_task_info['outputs_std']   # List of (n_outputs, n_time)
        inputs = self.last_task_info.get('inputs', [])

        # 4. Plotting Loop
        for i in range(self.num_outputs):
            ax = axes[i]
            species_idx = self.output_idx[i]
            species_name = self.species_labels[species_idx]

            # Iterate through each input/initial condition scenario
            for j in range(len(mean_data)):
                
                # Extract mean and std for the i-th output species in the j-th scenario
                y_mean = mean_data[j][i, :]
                y_std = std_data[j][i, :]
                
                # Create label based on input if available
                label = f"Scenario {j}"
                if inputs and j < len(inputs):
                    # concise string representation of input
                    label = f"u={np.array2string(np.array(inputs[j]), precision=2, separator=',')}"

                # Plot Mean Line
                line, = ax.plot(time, y_mean, label=label, linewidth=2)
                
                # Plot Standard Deviation Shading
                # We use the color of the line to match the shading
                ax.fill_between(time, 
                                y_mean - y_std, 
                                y_mean + y_std, 
                                color=line.get_color(), 
                                alpha=alpha)

            ax.set_title(f"Stochastic Response: {species_name} (Mean $\pm$ Std)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Count / Concentration")
            ax.grid(True, alpha=0.3)
            
            # Only add legend if there aren't too many scenarios to avoid clutter
            if len(mean_data) <= 10:
                ax.legend(fontsize='small')

        if fig:
            plt.tight_layout()
            
        return fig, axes
        

    
    # ------------------------ CVODE Solver Method ------------------------
    
    def solve_with_cvode(self, x0, time_horizon, u, nonneg_idx, stop_fn):
        t0 = float(time_horizon[0])
        tf = float(time_horizon[-1])
        x0 = np.asarray(x0, dtype=float)
        time_horizon = np.asarray(time_horizon, dtype=float)

        rhsfn = _make_rhs(self.rate_function, u)

        # wrap your stop_if_unstable
        eventsfn = make_eventsfn(stop_fn)

        options = dict(
            rtol=self.rtol,
            atol=self.atol,
            eventsfn=eventsfn,
            num_events=1,
        )

        # CVODE inequality constraints y[i] >= 0
        if nonneg_idx is not None and len(nonneg_idx) > 0:
            nonneg_idx = np.asarray(nonneg_idx, dtype=int)
            options["constraints_idx"] = nonneg_idx
            options["constraints_type"] = np.ones_like(nonneg_idx, dtype=int)  # 1 → y[i] >= 0 :contentReference[oaicite:1]{index=1}

        solver = CVODE(rhsfn, **options)

        # ask for output exactly at your time grid, like t_eval
        soln = solver.solve(time_horizon, x0)

        # adapt to solve_ivp-like shape: y → (n_states, n_times)
        class Solution:
            pass

        solution = Solution()
        solution.t = soln.t
        solution.y = soln.y.T
        solution.message = soln.message
        solution.status = soln.status
        solution.raw = soln
        return solution


import numpy as np
from sksundae.cvode import CVODE

def make_eventsfn(stop_if_unstable):
    def eventsfn(t, y, events):
        # single event → use slot 0
        events[0] = stop_if_unstable(t, y)

    # carry over your SciPy-style attributes, but as 1-element lists
    term = getattr(stop_if_unstable, "terminal", True)
    direction = getattr(stop_if_unstable, "direction", 0)

    eventsfn.terminal = [term]       # list length = num_events
    eventsfn.direction = [direction] # same as SciPy’s direction
    return eventsfn


def _make_rhs(rate_function, u):
    # CVODE rhs: rhs(t, y, yp) — fill yp[:] in place
    def rhsfn(t, y, yp):
        yp[:] = rate_function(t, y, u)
    return rhsfn


