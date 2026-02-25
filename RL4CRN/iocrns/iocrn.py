"""
Input-Output Chemical Reaction Network (IOCRN) representation.

This module defines `IOCRN`, a representation of an input-output chemical
reaction network with utilities for:

- building and compiling a CRN from a list of reactions,
- mapping between symbolic labels (species/inputs) and numeric indices,
- simulating transient responses (ODE solvers and SSA),
- plotting simulation results,
- exporting the network to a DSL format used by external simulators,
- (optionally) solving dynamics using CVODE via `sksundae`.

This modules provides utilities for deterministic simulations of CRNs, 
while it relies on an external package for stochastic simulation (SSA).

Caching:
    Simulation results are cached in `last_task_info` to avoid recomputation
    if the same task is requested repeatedly. Adding reactions or calling
    `reset` clears cached task info.

Solver support:
    - `'LSODA'` uses `scipy.integrate.solve_ivp`.
    - `'CVODE'` uses `sksundae.cvode.CVODE`.

SSA support:
    Stochastic simulation is supported via an external package
    `StochasticSimulationsNew` plus project utilities in :mod:`RL4CRN.utils.stochastic`.
    If the external package is missing, SSA methods raise an ImportError.
"""

import numpy as np
import sympy as sp
from itertools import product
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import copy
from scipy.integrate import solve_ivp
import pandas as pd

from RL4CRN.utils.plotting_style import _merge_cfg, _apply_axes_style, _maybe_legend, _maybe_save, _ensure_axes_list, _DEFAULT_PLOT_CFG, paper_rc_context

class IOCRN:
    """Input-Output Chemical Reaction Network.

    An IOCRN consists of:
        - a list of reactions,
        - a set of output species labels,
        - an implied set of species and inputs gathered from reactions,
        - dynamics defined by stoichiometry and propensities.

    The network must be `compile`d after construction or modification to
    build internal indices and matrices used for simulation.

    Args:
        reactions: List of `Reaction` objects. Each reaction is expected to
            provide methods such as:

            - `get_involved_species()`
            - `get_involved_inputs()`
            - `get_stoichiometry_dict()`
            - `propensity(x, u)`
            - `set_crn_context(iocrn)`

            and attributes such as:
            
            - `ID`
            - `params`
            - `num_parameters`
            - `num_unknown_params`
        
        output_labels: List of species labels treated as outputs.
        solver: ODE solver identifier. Supported values:
        
            - `'CVODE'` (requires `sksundae`)
            - `'LSODA'` (SciPy solve_ivp)
        
        atol: Absolute tolerance for ODE integration.
        rtol: Relative tolerance for ODE integration.

    Attributes:
        reactions: List of reaction objects.
        output_labels: List of output species labels.
        num_outputs: Number of outputs.
        num_unknown_params: Total number of unknown parameters across reactions.
        last_task_info: Dictionary caching results/metadata of the last evaluated
            task (e.g., transient response).
        solver: Selected solver backend name.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
    """

    # ------------------------ Construction Methods ------------------------
    def __init__(self, reactions, output_labels, solver='CVODE', atol=1e-6, rtol=1e-3): 
        """Initialize the IOCRN.

        Important:
            After initialization, call `compile` to create internal
            representations (species/input indices, stoichiometry matrix, etc.).

        Args:
            reactions: List of reaction objects.
            output_labels: List of labels for output species.
            solver: ODE solver backend (`'CVODE'` or `'LSODA'`).
            atol: Absolute tolerance passed to the solver.
            rtol: Relative tolerance passed to the solver.
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
        """Return a deep copy of the IOCRN."""
        return copy.deepcopy(self)
    
    def reset(self):
        """Clear cached task information.

        This does not modify reactions or parameters; it only clears the cached
        `last_task_info`.
        """
        self.last_task_info = {}
        self.last_task_info['type'] = None

    def add_reaction(self, reaction):
        """Add a reaction and recompile.

        Adding a reaction clears cached task info and updates internal
        representations by calling `compile`.

        Args:
            reaction: Reaction object to append to the network.
        """
        # Flush the last task information
        self.reset()

        # Add the reaction to the list of reactions
        self.reactions.append(reaction)
        self.compile()

    def retrieve_species(self):
        """Collect species labels from reactions and create index mappings.

        Side Effects:
            Sets:

            - `species_labels`: sorted list of all species labels
            - `species_idx_dict`: mapping label -> index
        """
        # Get labels of all the species involved in the IOCRN, and sort them alphanumerically
        self.species_labels = list(set(sum([reaction.get_involved_species() for reaction in self.reactions], []))) # list of strings of all species
        for output_label in self.output_labels:
            if output_label not in self.species_labels:
                self.species_labels.append(output_label)
        self.species_labels.sort() 

        # Create a mapping from species labels to their indices
        self.species_idx_dict = {species_label: idx for idx, species_label in enumerate(self.species_labels)} # dictionary mapping species labels to indices

    def retrieve_input(self):
        """Collect input labels from reactions and create index mappings.

        Side Effects:
            Sets:

            - `input_labels`: sorted list of all input labels
            - `num_inputs`: number of inputs
            - `input_idx_dict`: mapping label -> index
        """
        # Get labels of all the species involved in the IOCRN, and sort them alphanumerically
        self.input_labels = list(set(sum([r.get_involved_inputs() for r in self.reactions], []))) # list of strings of all inputs
        self.input_labels.sort()
        self.num_inputs = len(self.input_labels) # number of inputs

        # Create a mapping from input labels to their indices
        self.input_idx_dict = {input_label: idx for idx, input_label in enumerate(self.input_labels)} # dictionary mapping input labels to indices

    # def set_unknown_parameters(self, params): 
    #     """Set all unknown parameters in all reactions.

    #     The provided parameter vector is flattened and consumed in sequence by
    #     each reaction via `reaction.set_unknown_parameters(params)`.

    #     Args:
    #         params: Array-like of unknown parameters.

    #     Raises:
    #         Exception: If not all provided parameters are consumed, indicating a
    #             mismatch between `params` and the unknown parameter slots.
    #     """
    #     params = np.array(params).flatten()
    #     for reaction in self.reactions:
    #         params = reaction.set_unknown_parameters(params)
        
    #     if len(params) > 0:
    #         raise Exception(f"Error: {len(params)} parameters were not set - they are still unknown.")
        
    def species_label_to_idx(self, labels):
        """Map species labels to indices.

        Args:
            labels: Species label (str) or list of species labels.

        Returns:
            If `labels` is a string, returns an integer index.
                If `labels` is a list, returns a list of integer indices.
        """
        if isinstance(labels, str):
            return self.species_idx_dict[labels] # single index
        return [self.species_idx_dict[label] for label in labels] # list of indices
    
    def input_label_to_idx(self, labels):
        """Map input labels to indices.

        Args:
            labels: Input label (str) or list of input labels. Elements may be
                `None`, which are preserved.

        Returns:
            If `labels` is a string, returns an integer index.
                If `labels` is a list, returns a list of indices (with `None` preserved).
        """
        if isinstance(labels, str):
            return self.input_idx_dict[labels] # single index
        return [self.input_idx_dict[l] if l is not None else None for l in labels] # list of indices
    
    def set_library_context(self, reaction_library):
        """Set a reaction-library context for all reactions.

        This is required for some topology comparisons and signatures.

        Args:
            reaction_library: Library object passed to `reaction.set_library_context(...)`.

        Side Effects:
            Sets `self.reaction_library`.
        """
        for reaction in self.reactions:
            reaction.set_library_context(reaction_library)
        self.reaction_library = reaction_library

    def get_bool_signature(self):
        """Return a boolean signature of present reactions relative to a library.

        Returns:
            Boolean numpy array of length `len(self.reaction_library)` where True
                indicates the corresponding library reaction ID is present.

        Raises:
            AttributeError: If `reaction_library` has not been set.
        """
        IDs = self.gather_reaction_IDs()
        M = len(self.reaction_library)
        signature = np.zeros(M, dtype=bool)
        signature[IDs] = True
        return signature
    
    def is_topologically_equal(self, other_iocrn):
        """Check topology equality against another IOCRN.

        Assumes both IOCRNs share the same reaction library context.

        Args:
            other_iocrn: Another `IOCRN`.

        Returns:
            True if their boolean signatures match, else False.
        """
        
        return np.array_equal(self.get_bool_signature(), other_iocrn.get_bool_signature())

    def get_stoichiometry_matrix(self):
        """Construct the stoichiometry matrix $S$.

        Returns:
            Numpy array `S` of shape `(num_species, num_reactions)` where
                `S[i, j]` is the net stoichiometric coefficient of species `i` in
                reaction `j`.
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
        """Return reaction IDs for all reactions in the network."""
        reaction_IDs = [reaction.ID for reaction in self.reactions]
        return reaction_IDs
    
    def gather_reaction_params(self):
        """Return a list of parameter vectors for all reactions."""

        reaction_params = []
        for reaction in self.reactions:
            reaction_params.append(reaction.params)
        return reaction_params

    def compile(self):
        """Compile internal IOCRN representations.

        This method:
            - retrieves and indexes species and inputs,
            - computes `output_idx` for output species,
            - builds the stoichiometry matrix `S`,
            - calls `reaction.set_crn_context(self)` on each reaction.

        Call this after:
            - initialization,
            - adding/removing reactions,
            - changing reaction definitions that affect involved species/inputs.
        
        Important:
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
        """Return a readable string representation of the IOCRN.

        Includes:
            - inputs
            - species
            - output species
            - reactions (sorted by reaction ID if available)
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
        """Export the IOCRN to a custom DSL reaction-file format.

        The exported text contains:
            1. An `input ...;` section (if inputs are present),
            2. A `species ... = 0;` section (excluding emptyset/∅),
            3. A reactions section where each reaction contributes one or more
               lines via `reaction.to_reaction_format()`.

        Returns:
            The DSL text as a single string.
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
        """Compute propensities $a(x,u)$ for all reactions.

        Args:
            x: Species state vector (array-like of shape `(n,)`).
            u: Input vector (array-like of shape `(p,)`).

        Returns:
            Numpy array of propensities of shape `(num_reactions,)`.
        """
        propensities = np.array([r.propensity(x, u) for r in self.reactions])
        return propensities
    
    def rate_function(self, t, x, u):
        """Compute time-derivative $\dot{x} = S a(x,u)$.

        Args:
            t: Time (unused for autonomous dynamics, but accepted by solver APIs).
            x: Species state vector of shape `(n,)`.
            u: Input vector of shape `(p,)`.

        Returns:
            Numpy array of shape `(n,)` giving $\dot{x}$.
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
        # Do the simulation for each input scenario and store the results in lists
        x_list = []
        y_list = []
        num_doses = len(u_dose)
        for u_e in u_list:
            x = np.zeros((self.num_species, num_doses), dtype=np.float64) # numpy array of shape (n, num_doses)
            y = np.zeros((self.num_outputs, num_doses), dtype=np.float64) # numpy array of shape (q, num_doses)
            x_0 = initial_guess
            for i in range(num_doses):
                u = np.concatenate(([u_dose[i]], u_e))
                x[:,i] = fsolve(lambda x, u: self.rate_function(0, x, u), x_0, args=(u,))
                y[:,i] = x[self.num_outputs - 1, i] 
                x_0 = x[:,i]  
            # Append the states and outputs to the lists
            x_list.append(x) 
            y_list.append(y) 

        # Store and return the last task information
        return x_list, y_list
    

    def transient_response_SSA(self, u_list, x0_list, time_horizon, n_trajectories=100, max_threads=10000, max_value=1e6):
        """Compute stochastic transient responses via SSA (mean ± std).

        This method runs a stochastic simulation algorithm (SSA) backend for the
        current IOCRN by first exporting the network to the project DSL
        (`to_reaction_file`) and then invoking the external SSA engine.

        For each pair `(u, x0)` in the Cartesian product `u_list × x0_list`, the
        method performs `n_trajectories` independent SSA runs and returns the
        empirical mean and standard deviation trajectories for all species and
        outputs. Results are interpolated onto the requested `time_horizon`.

        The results are cached in `last_task_info` with
        `type == 'transient response SSA'`.

        Args:
            u_list: List of input vectors. Each element is array-like of shape
                `(p,)`, where `p = num_inputs`. Inputs are treated as constant
                over time for each scenario.
            x0_list: List of initial condition vectors. Each element is array-like
                of shape `(n,)`, where `n = num_species`.
            time_horizon: 1D array-like of time points at which to return
                statistics. The SSA backend is sampled at a step `t_step`
                inferred from `time_horizon` and then interpolated to match it.
            n_trajectories: Number of SSA trajectories per `(u, x0)` scenario used
                to estimate mean and standard deviation.
            max_threads: Maximum threads/parallelism hint for the SSA backend.
                (Currently not directly used in this method; kept for API
                compatibility / future backend control.)
            max_value: Divergence threshold passed to the SSA measurement helper.
                Trajectories exceeding this value may be flagged as diverged.

        Returns:
            Tuple `(time_horizon, x_mean_list, y_mean_list, x_std_list, y_std_list, last_task_info)`:

                - `time_horizon`: The same array of time points passed in.
                - `x_mean_list`: List of mean state trajectories, one per scenario.
                Each entry has shape `(n, T)`.
                - `y_mean_list`: List of mean output trajectories, one per scenario.
                Each entry has shape `(q, T)`.
                - `x_std_list`: List of std-dev state trajectories, one per scenario.
                Each entry has shape `(n, T)`.
                - `y_std_list`: List of std-dev output trajectories, one per scenario.
                Each entry has shape `(q, T)`.
                - `last_task_info`: Dictionary caching inputs, initial conditions,
                trajectories and metadata.

        Raises:
            ImportError: If the external SSA package `StochasticSimulationsNew`
                is not available.
            ValueError: If the network contains unknown parameters
                (`num_unknown_params > 0`) and the backend requires fully specified
                propensities (backend-dependent).

        Notes:
            - This method assumes the SSA helper `RL4CRN.utils.stochastic.quick_measurement_SSA`
              returns a summary dataframe with columns compatible with the parsing
              logic in this implementation.
            - `last_task_info` uses keys:
              `'inputs'`, `'initial conditions'`, `'time_horizon'`,
              `'trajectories'`, `'trajectories_std'`, `'outputs'`, `'outputs_std'`,
              `'has_diverged'`.
        """
        
        # 1. Generate DSL and Parse CRN
        # We need to convert the current object state to the DSL format required by the SSA engine
        crn_text = self.to_reaction_file()
        
        try:
            from StochasticSimulationsNew.ReactionNetworkLanguage import make_parser
        except ImportError:
            raise ImportError("StochasticSimulationsNew package not found. SSA functionality will be unavailable.")    
        
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
            from RL4CRN.utils.stochastic import quick_measurement_SSA # import on demand, only if needed (otherwise this creates a context for CUDA even if not used)
            summary_df, has_diverged = quick_measurement_SSA(
                ssa_crn, 
                param_batch, 
                parameter_names=self.input_labels,
                t_fin=t_fin, 
                n_trajectories=n_trajectories, 
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

            # print(summary_df.head())

            for u_vec in u_list:
                # Filter DF for this specific input combination
                # logic to match u_vec to columns 'u_1', 'u_2' etc.
                mask = pd.Series(True, index=summary_df.index)
                
                # We assume the columns in DF are named based on input definitions.
                # We need to map index of u_vec to column name. 
                # self.input_labels should hold ['u_1', 'u_2'] sorted.
                for k, val in enumerate(u_vec):
                    col_name_str = self.input_labels[k] # 'u_1'
                    
                    # Try to find the matching tuple column
                    # We look for a column that starts with the label 'u_1'
                    # This handles cases where the column might be ('u_1',) or ('u_1', '')

                    matching_col = [c for c in summary_df.columns if c[0] == col_name_str][0]
                    
                    mask &= (np.isclose(summary_df[matching_col], val))

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
                            # print 
                            # print(f"s_label: {s_label}")
                            # print(f"time_horizon: {time_horizon}")
                            # print(f"t_sim: {t_sim}")
                            # print(f"vals: {vals}")
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
            'outputs_std': y_std_list,
            'has_diverged': has_diverged
        }
        
        return time_horizon, x_mean_list, y_mean_list, x_std_list, y_std_list, self.last_task_info
    
    def transient_response(self, u_list, x0_list, time_horizon, LARGE_NUMBER=1e4):
        r"""Compute deterministic transient responses (ODE integration).

        This method integrates the ODE dynamics:

        $$\dot{x}(t) = S\,a(x(t), u),$$

        for each pair `(u, x0)` in the Cartesian product `u_list × x0_list`,
        where inputs `u` are treated as constant in time for each scenario.

        Results are cached in `last_task_info` with
        `type == 'transient response'`. If the cache already contains a transient
        response, the stored results are returned immediately.

        Numerical stability:
            If the integrator fails or the solution becomes unstable (detected by
            an event when `max(|x|)` exceeds `LARGE_NUMBER` or becomes non-finite),
            the remaining portion of the trajectory is filled with `LARGE_NUMBER`.

        Args:
            u_list: List of constant input vectors, each array-like of shape `(p,)`.
            x0_list: List of initial condition vectors, each array-like of shape `(n,)`.
            time_horizon: 1D array-like of time points of shape `(T,)` at which to
                evaluate/interpolate the solution.
            LARGE_NUMBER: Threshold used both as an instability detection bound
                and as a fill value when integration fails.

        Returns:
            Tuple `(time_horizon, x_list, y_list, last_task_info)`:

                - `time_horizon`: 1D numpy array of shape `(T,)`.
                - `x_list`: List of full state trajectories, one per scenario, each
                of shape `(n, T)`.
                - `y_list`: List of output trajectories, one per scenario, each of
                shape `(q, T)`, extracted via `output_idx`.
                - `last_task_info`: Dictionary caching the results. Keys include
                `'inputs'`, `'initial conditions'`, `'time_horizon'`,
                `'trajectories'`, `'outputs'`.

        Raises:
            ValueError: If `num_unknown_params > 0`.
            ValueError: If `solver` is not one of `'LSODA'` or `'CVODE'`.

        Notes:
            - For `'LSODA'`, integration is performed using SciPy's `solve_ivp`.
            - For `'CVODE'`, the method delegates to `solve_with_cvode` and
              interpolates the solver output onto `time_horizon`.
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


    def transient_response_piecewise(self, u_nested_list, x0_list, nested_time_horizon, LARGE_NUMBER=1e4 ,force=False):
        """Compute deterministic transient responses with piecewise-constant inputs.

        This method generalizes `transient_response` to *input sequences*.
        Each element of `u_nested_list` is a sequence of input vectors
        `[u_0, ..., u_{K-1}]` applied consecutively over corresponding time
        segments `nested_time_horizon = [t_0, ..., t_{K-1}]`.

        For each scenario `(u_sequence, x0)` in the Cartesian product
        `u_nested_list × x0_list`, the method integrates the dynamics on each time
        segment in order, using the final state of segment `k` as the initial
        condition for segment `k+1`.

        Instability handling:
            If any segment fails or terminates early due to instability, the
            remainder of the trajectory (for that scenario) is filled with
            `LARGE_NUMBER`.

        Caching:
            Results are cached in `last_task_info` with
            `type == 'transient response'` and returned if a matching cache is
            detected (currently a heuristic check based on lengths).

        Args:
            u_nested_list: List of input sequences. Each entry is a list of input
                vectors `[u_0, ..., u_{K-1}]`, each vector array-like of shape `(p,)`.
                The sequence length must match `len(nested_time_horizon)`.
            x0_list: List of initial conditions, each array-like of shape `(n,)`.
            nested_time_horizon: List of 1D time grids `[t_0, ..., t_{K-1}]`.
                Each `t_k` is a 1D array of times for segment `k`. The segments
                are concatenated (with offsets) into a single `full_time_horizon`
                stored in the cache and returned.
            LARGE_NUMBER: Instability bound and fill value for failed segments.

        Returns:
            Tuple `(full_time_horizon, x_list, y_list, last_task_info)`:

            - `full_time_horizon`: 1D array formed by concatenating shifted segment
              time grids, shape `(T_total,)`.
            - `x_list`: List of full state trajectories, one per scenario, each
              of shape `(n, T_total)`.
            - `y_list`: List of output trajectories, one per scenario, each of
              shape `(q, T_total)`.
            - `last_task_info`: Cache dictionary storing inputs, initial conditions,
              time horizon, trajectories, and outputs.

        Raises:
            ValueError: If any input sequence length does not match the number of
                time segments.
            ValueError: If `num_unknown_params > 0`.
            ValueError: If `solver` is not one of `'LSODA'` or `'CVODE'`.
        """

        # 2. Calculate durations using the correctly identified sequence
        durations = [t[-1] for t in nested_time_horizon]

        # 3. Calculate cumulative offsets
        offsets = np.cumsum([0] + durations[:-1])

        # 4. Apply offsets and concatenate
        shifted_horizons = [t_chunk + offset for t_chunk, offset in zip(nested_time_horizon, offsets)]
        full_time_horizon = np.concatenate(shifted_horizons)

        # 2. Check Cache (Simplified for brevity, assumes inputs match if lengths match)
        if (self.last_task_info.get('type') == 'transient response' and 
            len(self.last_task_info.get('trajectories', [])) == len(u_nested_list) * len(x0_list)) and not force:
            return self.last_task_info['time_horizon'], self.last_task_info['trajectories'], self.last_task_info['outputs'], self.last_task_info
        
        if self.num_unknown_params > 0:
            raise ValueError("The IOCRN has unknown rate constants.")

        # Event function
        def stop_if_unstable(t, x):
            max_val = np.max(np.abs(x))
            if not np.isfinite(max_val): return 0 
            return LARGE_NUMBER - max_val 
        stop_if_unstable.terminal = True
        stop_if_unstable.direction = 0

        x_list = []
        y_list = []

        # 3. Cartesian Product Loop (Preserving your original structure)
        for u_sequence, x0_start in product(u_nested_list, x0_list):
            
            # Validation: The input sequence length must match the time horizon segments
            if len(u_sequence) != len(nested_time_horizon):
                raise ValueError(f"Mismatch: Input sequence has {len(u_sequence)} steps, time_horizon has {len(nested_time_horizon)} segments.")

            current_x0 = x0_start
            
            # Temporary lists to hold the pieces of this specific scenario
            scenario_x_parts = []
            
            simulation_failed = False
            
            # 4. Sequential Loop: Iterate through the steps of this specific scenario
            for u_step, t_segment in zip(u_sequence, nested_time_horizon):
                
                n_steps = len(t_segment)
                
                # If a previous step failed, just fill remaining steps with LARGE_NUMBER
                if simulation_failed:
                    x_part = np.full((self.num_species, n_steps), LARGE_NUMBER)
                    scenario_x_parts.append(x_part)
                    continue

                if self.solver == 'LSODA':
                    # Note: t_eval is relative to the segment, assume t_segment contains absolute times or correct relative times
                    solution = solve_ivp(
                        lambda t, x: self.rate_function(t, x, u_step), 
                        (t_segment[0], t_segment[-1]), 
                        current_x0, 
                        t_eval=t_segment, 
                        method="LSODA", 
                        events=stop_if_unstable, 
                        atol=self.atol, 
                        rtol=self.rtol
                    )
                    
                    if solution.status == -1: 
                        x_part = np.full((self.num_species, n_steps), LARGE_NUMBER)
                        simulation_failed = True
                    else:
                        x_part = solution.y
                        if solution.status == 1: # Unstable event
                            simulation_failed = True
                            current_len = x_part.shape[1]
                            pad = n_steps - current_len
                            if pad > 0:
                                x_part = np.concatenate([x_part, np.full((self.num_species, pad), LARGE_NUMBER)], axis=1)
                
                elif self.solver == 'CVODE':
                    solution = self.solve_with_cvode(
                        current_x0, t_segment, u_step,
                        nonneg_idx=np.arange(len(current_x0)), stop_fn=stop_if_unstable
                    )
                    
                    if solution.status < 0 or not solution.raw.success:
                        x_part = np.full((self.num_species, n_steps), LARGE_NUMBER)
                        simulation_failed = True
                    else:
                        # Interpolate CVODE results to t_segment grid
                        t_sol = np.asarray(solution.t, dtype=float)
                        y_sol = np.asarray(solution.y, dtype=float)
                        x_part = np.empty((self.num_species, n_steps), dtype=float)
                        
                        t_last = t_sol[-1]
                        mask = t_segment <= t_last
                        mask_rest = ~mask
                        
                        for i in range(self.num_species):
                            x_part[i, mask] = np.interp(t_segment[mask], t_sol, y_sol[i, :])
                            x_part[i, mask_rest] = LARGE_NUMBER
                        
                        if np.any(mask_rest): simulation_failed = True

                else:
                    raise ValueError(f"Unknown solver '{self.solver}'")

                # Append the result of this step
                scenario_x_parts.append(x_part)
                
                # Update initial condition for the NEXT step (end of this step)
                if not simulation_failed:
                    current_x0 = x_part[:, -1]

            # 5. Stitch the scenario together
            # Concatenate along time axis (axis 1)
            full_x_scenario = np.concatenate(scenario_x_parts, axis=1)
            full_y_scenario = full_x_scenario[self.output_idx, :]

            x_list.append(full_x_scenario)
            y_list.append(full_y_scenario)

        # Store info
        self.last_task_info = {}
        self.last_task_info['type'] = 'transient response'
        self.last_task_info['inputs'] = u_nested_list
        self.last_task_info['initial conditions'] = x0_list
        self.last_task_info['time_horizon'] = full_time_horizon
        self.last_task_info['trajectories'] = x_list
        self.last_task_info['outputs'] = y_list
        
        return full_time_horizon, x_list, y_list, self.last_task_info

    # ------------------------ Plotting Methods ------------------------
    # def plot_transient_response(self, fig=None, axes=None, alpha=0.1):
    #     """Plot cached transient response trajectories for each output.

    #     This method visualizes the output trajectories stored by
    #     `transient_response` or `transient_response_piecewise`.

    #     Args:
    #         fig: Optional matplotlib figure to plot into. If None, a new figure is
    #             created.
    #         axes: Optional axes list/array. If None, new axes are created with one
    #             subplot per output species.
    #         alpha: Line transparency for overlaid trajectories.

    #     Returns:
    #         Tuple `(fig, axes)` where `axes` is a list-like of length `num_outputs`.

    #     Raises:
    #         ValueError: If no deterministic transient response is cached in
    #             `last_task_info` (i.e., `type != 'transient response'`).
    #     """

    #     # Check if transient response data is available
    #     if self.last_task_info.get('type') != 'transient response':
    #         raise ValueError("No transient response data available. Run transient_response() first.")
        
    #     # If no figure or axes are provided, create a new figure and axes
    #     if fig is None and axes is None:
    #         fig, axes = plt.subplots(self.num_outputs, 1, figsize=(10, 5 * self.num_outputs))
    #         if not isinstance(axes, (list, np.ndarray)):
    #             axes = [axes]
        
    #     # Plot the transient response for each output species and return the figure and axes
    #     for i in range(self.num_outputs):
    #         for j in range(len(self.last_task_info['outputs'])):
    #             axes[i].plot(self.last_task_info['time_horizon'], self.last_task_info['outputs'][j][i,:], alpha=alpha)
    #             axes[i].set_title(f"Transient Response of Output Species {self.species_labels[self.output_idx[i]]}")
    #             axes[i].set_xlabel("Time")
    #             axes[i].set_ylabel("Concentration")
    #     plt.tight_layout()
    #     return fig, axes

    # def plot_transient_response_piecewise(self, fig=None, axes=None, alpha=1.0,
    #                                     input_color="red", shade_on=True,
    #                                     pulse_lw=1.0, pulse_ls="--", gap=False):
    #     """
    #     Multi-frequency aware plotting:
    #     - If self.last_task_info contains 'freq_runs' (list of runs), plot one block per frequency,
    #         stacked vertically, and for each block plot all outputs (one row per output).
    #     - Otherwise fallback to single-run behavior.

    #     Expects either:
    #     A) Single-run (legacy):
    #         last_task_info['time_horizon'], last_task_info['outputs'], last_task_info['input_intervals'], last_task_info['input_pulse']
    #     B) Multi-frequency:
    #         last_task_info['freq_runs'] = [
    #             {'pulse_shape': (t_on,t_off),
    #             'time_horizon': t,
    #             'outputs': outputs,
    #             'input_intervals': seg_intervals,
    #             'input_pulse': u_pulse}, ...
    #         ]
    #     """

    #     if self.last_task_info.get("type") != "transient response":
    #         raise ValueError("No transient response data available. Run transient_response_piecewise() first.")

    #     freq_runs = self.last_task_info.get("freq_runs", None)

    #     # ---------- MULTI-FREQUENCY MODE ----------
    #     if isinstance(freq_runs, list) and len(freq_runs) > 0:
    #         n_freq = len(freq_runs)
    #         n_rows = self.num_outputs * n_freq

    #         if fig is None and axes is None:
    #             fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.5 * n_rows), sharex=False)
    #         if not isinstance(axes, (list, np.ndarray)):
    #             axes = [axes]
    #         axes = list(np.ravel(axes))
    #         if len(axes) != n_rows:
    #             raise ValueError(f"Expected {n_rows} axes for {n_freq} frequencies x {self.num_outputs} outputs, got {len(axes)}")

    #         ax_idx = 0
    #         for f_idx, run in enumerate(freq_runs):
    #             t = np.asarray(run["time_horizon"], dtype=float)
    #             outputs = run["outputs"]

    #             seg_intervals = run.get("input_intervals", None)
    #             u_pulse = run.get("input_pulse", None)
    #             have_input = (seg_intervals is not None) and (u_pulse is not None)

    #             boundaries = step_values = seg_values = gap_bounds = None
    #             if have_input:
    #                 boundaries, step_values, seg_values, gap_bounds = _pulse_step_and_shading_from_intervals(
    #                     seg_intervals, u_pulse, gap=gap
    #                 )

    #             pulse_shape = run.get("pulse_shape", None)
    #             if pulse_shape is not None:
    #                 t_on, t_off = float(pulse_shape[0]), float(pulse_shape[1])
    #                 period = t_on + t_off
    #                 freq_title = f"Pulse shape (t_on={t_on:g}, t_off={t_off:g})  period={period:g}"
    #             else:
    #                 freq_title = f"Frequency run {f_idx+1}"

    #             for out_i in range(self.num_outputs):
    #                 ax = axes[ax_idx]
    #                 ax_idx += 1

    #                 species_name = self.species_labels[self.output_idx[out_i]]
    #                 title = f"{freq_title}  |  Output: {species_name} (Δy from y₀)"

    #                 _plot_one_frequency(
    #                     ax=ax,
    #                     t=t,
    #                     outputs=outputs,
    #                     out_i=out_i,
    #                     title=title,
    #                     alpha=alpha,
    #                     have_input=have_input,
    #                     boundaries=boundaries,
    #                     step_values=step_values,
    #                     seg_values=seg_values,
    #                     gap_bounds=gap_bounds,
    #                     input_color=input_color,
    #                     shade_on=shade_on,
    #                     pulse_lw=pulse_lw,
    #                     pulse_ls=pulse_ls,
    #                 )

    #         axes[-1].set_xlabel("Time")
    #         plt.tight_layout()
    #         return fig, axes

    #     # ---------- SINGLE-RUN FALLBACK ----------
    #     t = np.asarray(self.last_task_info["time_horizon"], dtype=float)
    #     outputs = self.last_task_info["outputs"]

    #     if fig is None and axes is None:
    #         fig, axes = plt.subplots(self.num_outputs, 1, figsize=(10, 4 * self.num_outputs), sharex=True)
    #     if not isinstance(axes, (list, np.ndarray)):
    #         axes = [axes]

    #     seg_intervals = self.last_task_info.get("input_intervals", None)
    #     u_pulse = self.last_task_info.get("input_pulse", None)
    #     have_input = (seg_intervals is not None) and (u_pulse is not None)

    #     boundaries = step_values = seg_values = gap_bounds = None
    #     if have_input:
    #         boundaries, step_values, seg_values, gap_bounds = _pulse_step_and_shading_from_intervals(
    #             seg_intervals, u_pulse, gap=gap
    #         )

    #     for out_i in range(self.num_outputs):
    #         ax = axes[out_i]
    #         species_name = self.species_labels[self.output_idx[out_i]]
    #         title = f"Transient Response of {species_name} (Δy from y₀)"

    #         _plot_one_frequency(
    #             ax=ax,
    #             t=t,
    #             outputs=outputs,
    #             out_i=out_i,
    #             title=title,
    #             alpha=alpha,
    #             have_input=have_input,
    #             boundaries=boundaries,
    #             step_values=step_values,
    #             seg_values=seg_values,
    #             gap_bounds=gap_bounds,
    #             input_color=input_color,
    #             shade_on=shade_on,
    #             pulse_lw=pulse_lw,
    #             pulse_ls=pulse_ls,
    #         )

    #     axes[-1].set_xlabel("Time")
    #     plt.tight_layout()
    #     return fig, axes

        
    # def plot_phase_portrait(self, fig=None, axis=None, alpha=0.1):
    #     """Plot a phase portrait from cached transient response trajectories.

    #     For two species, plots $x_1(t)$ vs $x_2(t)$.
    #     For three species, plots a 3D trajectory.

    #     Args:
    #         fig: Optional matplotlib figure. If None, a new figure is created.
    #         axis: Optional matplotlib axis (2D or 3D). If None, a new axis is
    #             created depending on the number of species.
    #         alpha: Line transparency for overlaid trajectories.

    #     Returns:
    #         Tuple `(fig, axis)`.

    #     Raises:
    #         ValueError: If no deterministic transient response is cached
    #             (`type != 'transient response'`).
    #         ValueError: If `num_species` is not 2 or 3.
    #     """

    #     # Check if transient response data is available
    #     if self.last_task_info.get('type') != 'transient response':
    #         raise ValueError("No transient response data available. Run transient_response() first.")
        
    #     # If no figure or axes are provided, create a new figure and axes
    #     if fig is None and axis is None:
    #         if self.num_species == 2:
    #             fig, axis = plt.subplots(figsize=(10, 10))
    #         elif self.num_species == 3:
    #             fig = plt.figure(figsize=(10, 10))
    #             axis = fig.add_subplot(111, projection='3d')
    #         else:
    #             raise ValueError("Phase portrait can only be plotted for 2 or 3 species.")
        
    #     # Plot the phase portrait and return the figure and axes
    #     if self.num_species == 2:
    #         for j in range(len(self.last_task_info['trajectories'])):
    #             axis.plot(self.last_task_info['trajectories'][j][0,:], self.last_task_info['trajectories'][j][1,:], alpha=alpha)
    #         axis.set_xlabel(f"Species {self.species_labels[0]}")
    #         axis.set_ylabel(f"Species {self.species_labels[1]}")
    #         axis.set_title("Phase Portrait")
    #     elif self.num_species == 3:
    #         for j in range(len(self.last_task_info['trajectories'])):
    #             axis.plot(self.last_task_info['trajectories'][j][0,:], self.last_task_info['trajectories'][j][1,:], self.last_task_info['trajectories'][j][2,:], alpha=alpha)
    #         axis.set_xlabel(f"Species {self.species_labels[0]}")
    #         axis.set_ylabel(f"Species {self.species_labels[1]}")
    #         axis.set_zlabel(f"Species {self.species_labels[2]}")
    #         axis.set_title("Phase Portrait")
    #     plt.tight_layout()
    #     return fig, axis



    
    def plot_logic_response(
        self,
        *,
        u_list=None,
        target_fn=None,
        title="Truth table",
        figsize=(2.2, 1.2),
        silent=False,
    ):
        """
        Plot a truth-table style view of the CRN's logic behavior.

        This expects the CRN to have cached a deterministic multi-scenario simulation in
        `self.last_task_info` (e.g., produced by a reward function that calls
        `transient_response(...)` and stores its results).

        Args:
            u_list: Optional list of input vectors used for the scenarios. If None, tries
                to read `self.last_task_info['u_list']`.
            target_fn: Optional callable mapping u (np.ndarray) -> {0,1} (or bool).
                If None, tries to read `self.last_task_info['logic_fn']`.
            title: Plot title.
            figsize: Matplotlib figure size.
            silent: If True, does not call plt.show().

        Returns:
            (fig, ax) from plot_truth_table_transposed_nature.

        Raises:
            ValueError: If required cached information is missing.
        """
        # 1) Make sure we have cached outputs
        if "outputs" not in self.last_task_info:
            raise ValueError(
                "No cached outputs found in last_task_info. "
                "Run a reward/transient_response that stores last_task_info['outputs'] first."
            )

        # 2) Get u_list
        if u_list is None:
            u_list = self.last_task_info.get("u_list", None)
            if u_list is None:
                raise ValueError(
                    "u_list not provided and not found in last_task_info['u_list'].\n"
                    "Fix: pass u_list=task.u_list, or store it into last_task_info when computing rewards."
                )

        # 3) Decide the target / logic function
        logic_fn = target_fn if target_fn is not None else self.last_task_info.get("logic_fn", None)
        if logic_fn is None:
            raise ValueError(
                "No target_fn provided and no logic_fn found in last_task_info['logic_fn'].\n"
                "Fix: call plot_logic_response(target_fn=...) or store the logic function in last_task_info."
            )

        # 4) Extract actual outputs: one scalar per scenario
        # Uses output species 0, last timepoint by default.
        try:
            actual_outputs = [o[0][-1] for o in self.last_task_info["outputs"]]
        except Exception as e:
            raise ValueError(
                "Could not parse last_task_info['outputs'] for truth table plotting. "
                "Expected a list where each element has shape (num_outputs, n_t)."
            ) from e

        # 5) Import plotting helper (avoid top-level import cycles)
        from RL4CRN.utils.visualizations import plot_truth_table_transposed_nature

        fig, ax = plot_truth_table_transposed_nature(
            u_list=u_list,
            actual_outputs=actual_outputs,
            logic_function=logic_fn,
            title=title,
            figsize=figsize,
            silent=silent,
        )
        return fig, ax

    
    # def plot_dose_response(self, fig=None, axes=None, alpha=0.5):
    #     """Plot dose-response curves from cached simulation data.

    #     Two modes are supported depending on the cached task:
    #         - If `last_task_info['type'] == 'dose response'`, uses precomputed
    #           dose-response data (if present in the cache).
    #         - If `last_task_info['type'] == 'transient response'`, constructs a
    #           dose-response by taking the final-time output value for each input
    #           scenario and plotting it versus the input dose (currently assumes
    #           a single scalar dose per scenario, typically `u[0]`).

    #     Args:
    #         fig: Optional matplotlib figure. If None, a new figure is created.
    #         axes: Optional axes list. If None, new axes are created (one per output).
    #         alpha: Line transparency for plotted curves.

    #     Returns:
    #         Tuple `(fig, axes)`.

    #     Raises:
    #         ValueError: If neither dose-response nor transient-response data is
    #             present in `last_task_info`.
    #     """

    #     # Check if dose response data is available
    #     if self.last_task_info.get('type') != 'dose response' and self.last_task_info.get('type') != 'transient response':
    #         raise ValueError("No dose response data available. Run dose_response() or transient_response() first.")
        
    #     # If no figure or axes are provided, create a new figure and axes
    #     if fig is None and axes is None:
    #         fig, axes = plt.subplots(self.num_outputs, 1, figsize=(10, 5 * self.num_outputs))
    #         if not isinstance(axes, (list, np.ndarray)):
    #             axes = [axes]
        
    #     # Plot the dose responses for each output species and return the figure and axes #TODO: generalize for multiple inputs and implement dose response algebraically
    #     if self.last_task_info['type'] == 'dose response':
    #         u_dose = self.last_task_info['input doses']
    #         for i in range(self.num_outputs):
    #             for j in range(len(self.last_task_info['input scenarios'])):
    #                 axes[i].plot(u_dose, self.last_task_info['outputs'][j][i,:], alpha=alpha)
    #                 axes[i].set_title(f"Dose Response of Output Species {self.species_labels[self.o[i]-1]}")
    #                 axes[i].set_xlabel("Input Dose")
    #                 axes[i].set_ylabel("Concentration")
    #         plt.tight_layout()

    #     # elif self.last_task_info['type'] == 'transient response': # TODO: generalize for multiple inputs
    #     #     u_list = self.last_task_info['inputs']
    #     #     for i in range(self.num_outputs):
    #     #         u_dose = np.array([u[0] for u in u_list])
    #     #         y_dose = np.array([y[i,-1] for y in self.last_task_info['outputs']])
    #     #         axes[i].plot(u_dose, y_dose, alpha=alpha)
    #     #         axes[i].set_title(f"Dose Response of Output Species {self.species_labels[self.output_idx[i]]}")
    #     #         axes[i].set_xlabel("Input Dose")
    #     #         axes[i].set_ylabel("Concentration")
    #     #     plt.tight_layout()

    #     elif self.last_task_info['type'] == 'transient response': # TODO: Now generalized
    #         u_list = self.last_task_info['inputs']
    #         x0_list = self.last_task_info['initial_conditions']

    #         step = len(self.last_task_info['outputs']) // len(x0_list)

    #         for i in range(self.num_outputs):
    #             for k,_ in enumerate(x0_list):
    #                 u_dose = np.array([u[0] for u in u_list])
    #                 y_dose = np.array([y[i,-1] for y in self.last_task_info['outputs'][step*(k):step*(k+1)]])
    #                 axes[i].plot(u_dose, y_dose, alpha=alpha)
    #                 axes[i].set_title(f"Dose Response of Output Species {self.species_labels[self.output_idx[i]]}")
    #                 axes[i].set_xlabel("Input Dose")
    #                 axes[i].set_ylabel("Concentration")
    #         plt.tight_layout()

    #     return fig, axes
    
    # def plot_frequency_content(self, fig=None, axes=None, alpha=0.1, t0=0.0):
    #     """Plot Fourier magnitude spectra of outputs from cached transient responses.

    #     For each cached transient response trajectory, this method:
    #         1) truncates the signal to times `t >= t0`,
    #         2) subtracts the mean,
    #         3) computes a one-sided FFT magnitude spectrum,
    #         4) normalizes magnitudes by their maximum (per trajectory),
    #         5) overlays spectra for all scenarios.

    #     Args:
    #         fig: Optional matplotlib figure. If None, a new figure is created.
    #         axes: Optional axes list. If None, new axes are created (one per output).
    #         alpha: Line transparency for overlaid spectra.
    #         t0: Time threshold; only samples with time >= t0 are used.

    #     Returns:
    #         Tuple `(fig, axes)`.

    #     Raises:
    #         ValueError: If no deterministic transient response is cached.
    #         ValueError: If fewer than two samples exist after `t0`.
    #         ValueError: If a non-positive sampling interval is inferred.
    #     """

    #     # Check if transient response data is available
    #     if self.last_task_info.get('type') != 'transient response':
    #         raise ValueError("No transient response data available. Run transient_response() first.")
        
    #     # If no figure or axes are provided, create a new figure and axes
    #     if fig is None and axes is None:
    #         fig, axes = plt.subplots(self.num_outputs, 1, figsize=(10, 5 * self.num_outputs))
    #         if not isinstance(axes, (list, np.ndarray)):
    #             axes = [axes]

    #     # Truncate time vector and determine valid indices for t >= t0
    #     time = np.asarray(self.last_task_info['time_horizon'])
    #     mask = time >= t0
    #     if mask.sum() < 2:
    #         raise ValueError("Not enough data points after t0 to compute a Fourier transform.")

    #     # Infer (assumed uniform) sampling interval from the truncated time vector
    #     dt = float(np.mean(np.diff(time[mask])))
    #     if dt <= 0:
    #         raise ValueError("Non-positive sampling interval inferred from time_horizon.")  

    #     # Plot Fourier magnitude spectra for each output species
    #     N = int(mask.sum())
    #     freqs = np.fft.rfftfreq(N, d=dt)

    #     for i in range(self.num_outputs):
    #         ax = axes[i]
    #         for j in range(len(self.last_task_info['outputs'])):
    #             # Extract the i-th output trace from the j-th run, truncated at t0
    #             y = np.asarray(self.last_task_info['outputs'][j][i, :])[mask]

    #             # Remove mean to emphasize oscillatory content
    #             y = y - np.mean(y)

    #             # Compute one-sided FFT magnitude
    #             Y = np.fft.rfft(y)
    #             mag = np.abs(Y) 
    #             mag = mag / (np.max(mag) + 1e-12) # simple magnitude normalization

    #             ax.plot(freqs, mag, alpha=alpha)

    #         ax.set_title(
    #             f"Frequency Content of Output Species {self.species_labels[self.output_idx[i]]} (t ≥ {t0})"
    #         )
    #         ax.set_xlabel("Frequency (1 / time unit)")
    #         ax.set_ylabel("Magnitude")

    #     plt.tight_layout()
    #     return fig, axes
    
    # # ------------------------ stochastic simulation methods ------------------------

    # def plot_SSA_transient_response(self, fig=None, axes=None, alpha=0.2):
    #     r"""Plot cached SSA transient responses (mean ± std) for each output.

    #     This method visualizes the stochastic results produced by
    #     `transient_response_SSA`, plotting the mean trajectory and a shaded
    #     band corresponding to $\pm 1$ standard deviation:

    #     $$ y(t) \pm \sigma(t). $$
        
    #     Args:
    #         fig: Optional matplotlib figure. If None, a new figure is created.
    #         axes: Optional axes list. If None, new axes are created (one per output).
    #         alpha: Transparency for the standard deviation shading.

    #     Returns:
    #         Tuple `(fig, axes)`.

    #     Raises:
    #         ValueError: If no SSA transient response is cached in `last_task_info`
    #             (i.e., `type != 'transient response SSA'`).
    #     """

    #     # 1. Validation
    #     if self.last_task_info.get('type') != 'transient response SSA':
    #         raise ValueError("No stochastic transient response data available. Run transient_response_SSA() first.")
        
    #     # 2. Setup Figure/Axes
    #     if fig is None and axes is None:
    #         fig, axes = plt.subplots(self.num_outputs, 1, figsize=(10, 5 * self.num_outputs))
    #         # Ensure axes is iterable even if there's only one output
    #         if not isinstance(axes, (list, np.ndarray)):
    #             axes = [axes]
    #     elif not isinstance(axes, (list, np.ndarray)):
    #          axes = [axes]
        
    #     # 3. Retrieve Data
    #     time = self.last_task_info['time_horizon']
    #     mean_data = self.last_task_info['outputs']      # List of (n_outputs, n_time)
    #     std_data = self.last_task_info['outputs_std']   # List of (n_outputs, n_time)
    #     inputs = self.last_task_info.get('inputs', [])

    #     # 4. Plotting Loop
    #     for i in range(self.num_outputs):
    #         ax = axes[i]
    #         species_idx = self.output_idx[i]
    #         species_name = self.species_labels[species_idx]

    #         # Iterate through each input/initial condition scenario
    #         for j in range(len(mean_data)):
                
    #             # Extract mean and std for the i-th output species in the j-th scenario
    #             y_mean = mean_data[j][i, :]
    #             y_std = std_data[j][i, :]
                
    #             # Create label based on input if available
    #             label = f"Scenario {j}"
    #             if inputs and j < len(inputs):
    #                 # concise string representation of input
    #                 label = f"u={np.array2string(np.array(inputs[j]), precision=2, separator=',')}"

    #             # Plot Mean Line
    #             line, = ax.plot(time, y_mean, label=label, linewidth=2)
                
    #             # Plot Standard Deviation Shading
    #             # We use the color of the line to match the shading
    #             ax.fill_between(time, 
    #                             y_mean - y_std, 
    #                             y_mean + y_std, 
    #                             color=line.get_color(), 
    #                             alpha=alpha)

    #         ax.set_title(f"Stochastic Response: {species_name} (Mean $\pm$ Std)")
    #         ax.set_xlabel("Time")
    #         ax.set_ylabel("Count / Concentration")
    #         ax.grid(True, alpha=0.3)
            
    #         # Only add legend if there aren't too many scenarios to avoid clutter
    #         if len(mean_data) <= 10:
    #             ax.legend(fontsize='small')

    #     # if fig:
    #     #     plt.tight_layout()
            
    #     return fig, axes
        
    # ---- New plotting methods ----

    # ---------------------------------------------------------------------------
    # Methods below are intended to live as instance methods on your IOCRN class.
    # Keep their original signatures, only adding `plot_cfg=None` as trailing kwarg.
    # ---------------------------------------------------------------------------

    def plot_transient_response(self, fig=None, axes=None, alpha=0.1, plot_cfg=None):
        """Plot cached deterministic transient response trajectories.

        This method visualizes trajectories stored by a call that cached
        `last_task_info['type'] == 'transient response'`, typically produced by
        `transient_response(...)` or `transient_response_piecewise(...)`.

        Paper-ready styling can be controlled through `plot_cfg`, which may override
        default rcParams and common cosmetics (grid/spines/legend) while preserving
        the original method API.

        Args:
            fig: Optional matplotlib Figure to draw into. If None and `axes` is None,
                a new Figure is created.
            axes: Optional axes list/array. If None and `fig` is None, new axes are
                created with one subplot per output.
            alpha: Line transparency for overlaid trajectories (default: 0.1).
            plot_cfg: Optional dict controlling paper-ready styling. Supported keys:
                - rc: dict of matplotlib rcParams overrides
                - figsize: tuple (w,h) for figure size
                - constrained_layout: bool
                - tight_layout: bool
                - grid: bool
                - grid_kwargs: dict passed to ax.grid(...)
                - despine: bool (hide top/right spines)
                - spine_lw: float
                - alpha: override alpha passed in args
                - lw: line width override
                - title/xlabel/ylabel: override strings (title can be per-axis via None)
                - legend: "auto" | True | False
                - legend_kwargs: dict passed to ax.legend(...)
                - save: dict like {"path": "...pdf", "dpi": 600}

        Returns:
            Tuple[Figure, List[Axes]]: (fig, axes) where axes has length num_outputs.

        Raises:
            ValueError: If no cached deterministic transient response is present.
        """
        if self.last_task_info.get("type") != "transient response":
            raise ValueError("No transient response data available. Run transient_response() first.")

        cfg = _merge_cfg(_DEFAULT_PLOT_CFG, plot_cfg)

        with paper_rc_context(cfg.get("rc")):
            if fig is None and axes is None:
                figsize = cfg.get("figsize") or (2.8, 1.6 * self.num_outputs)
                fig, axes = plt.subplots(
                    self.num_outputs, 1, figsize=figsize, sharex=True,
                    constrained_layout=bool(cfg.get("constrained_layout", False)),
                )
            axes = _ensure_axes_list(axes, self.num_outputs)

            t = np.asarray(self.last_task_info["time_horizon"], dtype=float)
            outs = self.last_task_info["outputs"]  # list of (q,T)

            a = cfg["alpha"] if cfg["alpha"] is not None else alpha
            lw = cfg["lw"]

            for i in range(self.num_outputs):
                ax = axes[i]
                n_series = 0
                for j in range(len(outs)):
                    y = np.asarray(outs[j], dtype=float)
                    yi = y[i, :] if y.ndim == 2 else y
                    ax.plot(t, yi, alpha=a, linewidth=lw)
                    n_series += 1

                species = self.species_labels[self.output_idx[i]]
                ax.set_title(cfg.get("title") or f"{species}", pad=cfg.get("title_pad", 3.0))
                ax.set_ylabel(cfg.get("ylabel") or "Concentration")
                _apply_axes_style(ax, cfg)
                _maybe_legend(ax, n_series, cfg)

            axes[-1].set_xlabel(cfg.get("xlabel") or "Time")

            if cfg.get("tight_layout", True) and not cfg.get("constrained_layout", False):
                fig.tight_layout()

            _maybe_save(fig, cfg)
            return fig, axes


    def plot_transient_response_piecewise(
        self, fig=None, axes=None, alpha=1.0,
        input_color="red", shade_on=True, trj_color="green",
        pulse_lw=1.0, pulse_ls="--", gap=False,
        plot_cfg=None,
        is_main=True,
        normalize=False,   # <- NEW
        y_label=None,
        legend_label = None,
    ):
        """Plot cached deterministic piecewise transient responses.

        If `is_main=False`, this acts as a follower plotter:
        - plots only trajectories
        - uses the provided `alpha`
        - skips pulse overlays, shading, titles, and styling extras

        If `normalize=True`, all trajectories are divided by a single global peak
        (max value across all cached conditions/scenarios/outputs in this object).
        """
        if self.last_task_info.get("type") != "transient response":
            raise ValueError("No transient response data available. Run transient_response_piecewise() first.")

        cfg = _merge_cfg(_DEFAULT_PLOT_CFG, plot_cfg)
        follower_alpha = alpha

        # ------------------------------------------------------------------
        # Compute ONE normalization factor across all cached conditions
        # ------------------------------------------------------------------
        norm_factor = 1.0
        if normalize:
            freq_runs = self.last_task_info.get("freq_runs", None)

            peaks = []
            if isinstance(freq_runs, list) and len(freq_runs) > 0:
                # multi-frequency cache
                for run in freq_runs:
                    for y in run.get("outputs", []):
                        arr = np.asarray(y, dtype=float)
                        if arr.size > 0:
                            peaks.append(np.nanmax(arr))
            else:
                # single-run cache
                for y in self.last_task_info.get("outputs", []):
                    arr = np.asarray(y, dtype=float)
                    if arr.size > 0:
                        peaks.append(np.nanmax(arr))

            if len(peaks) == 0:
                norm_factor = 1.0
            else:
                norm_factor = float(np.nanmax(peaks))
                # avoid division by zero / weird degenerate cases
                if (not np.isfinite(norm_factor)) or (norm_factor == 0.0):
                    norm_factor = 1.0

        with paper_rc_context(cfg.get("rc")):
            freq_runs = self.last_task_info.get("freq_runs", None)

            # ---------- MULTI-FREQUENCY MODE ----------
            if isinstance(freq_runs, list) and len(freq_runs) > 0:
                n_freq = len(freq_runs)
                n_rows = self.num_outputs * n_freq

                if fig is None and axes is None:
                    figsize = cfg.get("figsize") or (3.3, 1.25 * n_rows)
                    fig, axes = plt.subplots(
                        n_rows, 1, figsize=figsize, sharex=False,
                        constrained_layout=bool(cfg.get("constrained_layout", False)),
                    )
                axes = _ensure_axes_list(axes, n_rows)

                a = (cfg["alpha"] if (cfg["alpha"] is not None and is_main) else follower_alpha)

                ax_idx = 0
                for f_idx, run in enumerate(freq_runs):
                    t = np.asarray(run["time_horizon"], dtype=float)

                    # Normalize trajectories with ONE shared factor
                    outputs = [
                        np.asarray(y, dtype=float) / norm_factor if normalize else np.asarray(y, dtype=float)
                        for y in run["outputs"]
                    ]

                    # Only main figures get input pulse/shading info
                    boundaries = step_values = seg_values = gap_bounds = None
                    have_input = False
                    if is_main:
                        seg_intervals = run.get("input_intervals", None)
                        u_pulse = run.get("input_pulse", None)
                        have_input = (seg_intervals is not None) and (u_pulse is not None)
                        if have_input:
                            boundaries, step_values, seg_values, gap_bounds = _pulse_step_and_shading_from_intervals(
                                seg_intervals, u_pulse, gap=gap
                            )

                    pulse_shape = run.get("pulse_shape", None)
                    if pulse_shape is not None:
                        t_on, t_off = float(pulse_shape[0]), float(pulse_shape[1])
                        period = t_on + t_off
                        freq_title = cfg.get("title") or f"t_on={t_on:g}, t_off={t_off:g} (T={period:g})"
                    else:
                        freq_title = cfg.get("title") or f"Run {f_idx+1}"

                    for out_i in range(self.num_outputs):
                        ax = axes[ax_idx]
                        ax_idx += 1

                        species_name = self.species_labels[self.output_idx[out_i]]
                        title = f"{freq_title} | {species_name}" if is_main else None

                        _plot_one_frequency(
                            ax=ax,
                            t=t,
                            outputs=outputs,                   # <- normalized if requested
                            out_i=out_i,
                            title=title,
                            alpha=a,
                            have_input=have_input,
                            boundaries=boundaries,
                            step_values=step_values,
                            seg_values=seg_values,
                            gap_bounds=gap_bounds,
                            input_color=input_color,
                            trj_color=trj_color,
                            shade_on=(shade_on and is_main),
                            pulse_lw=pulse_lw,
                            pulse_ls=pulse_ls,
                            y_label = "Δ Concentration" if y_label is None else y_label,
                            legend_label = legend_label,
                        )

                        if is_main:
                            _apply_axes_style(ax, cfg)
                            ax.set_ylabel(cfg.get("ylabel") or ("Δy / peak" if normalize else "Δy"))

                if is_main:
                    axes[-1].set_xlabel(cfg.get("xlabel") or "Time")
                    if cfg.get("tight_layout", True) and not cfg.get("constrained_layout", False):
                        fig.tight_layout()
                    _maybe_save(fig, cfg)

                return fig, axes

            # ---------- SINGLE-RUN FALLBACK ----------
            t = np.asarray(self.last_task_info["time_horizon"], dtype=float)
            outputs = [
                np.asarray(y, dtype=float) / norm_factor if normalize else np.asarray(y, dtype=float)
                for y in self.last_task_info["outputs"]
            ]

            if fig is None and axes is None:
                figsize = cfg.get("figsize") or (3.3, 1.55 * self.num_outputs)
                fig, axes = plt.subplots(
                    self.num_outputs, 1, figsize=figsize, sharex=True,
                    constrained_layout=bool(cfg.get("constrained_layout", False)),
                )
            axes = _ensure_axes_list(axes, self.num_outputs)

            boundaries = step_values = seg_values = gap_bounds = None
            have_input = False
            if is_main:
                seg_intervals = self.last_task_info.get("input_intervals", None)
                u_pulse = self.last_task_info.get("input_pulse", None)
                have_input = (seg_intervals is not None) and (u_pulse is not None)

                if have_input:
                    boundaries, step_values, seg_values, gap_bounds = _pulse_step_and_shading_from_intervals(
                        seg_intervals, u_pulse, gap=gap
                    )

            a = (cfg["alpha"] if (cfg["alpha"] is not None and is_main) else follower_alpha)

            for out_i in range(self.num_outputs):
                ax = axes[out_i]
                species_name = self.species_labels[self.output_idx[out_i]]
                title = (cfg.get("title") or f"{species_name}") if is_main else None

                _plot_one_frequency(
                    ax=ax,
                    t=t,
                    outputs=outputs,                        # <- normalized if requested
                    out_i=out_i,
                    title=title,
                    alpha=a,
                    have_input=have_input,
                    boundaries=boundaries,
                    step_values=step_values,
                    seg_values=seg_values,
                    gap_bounds=gap_bounds,
                    input_color=input_color,
                    trj_color=trj_color,
                    shade_on=(shade_on and is_main),
                    pulse_lw=pulse_lw,
                    pulse_ls=pulse_ls,
                    legend_label = legend_label,
                )

                if is_main:
                    _apply_axes_style(ax, cfg)
                    ax.set_ylabel(cfg.get("ylabel") or ("Δy / peak" if normalize else "Δy"))

            if is_main:
                axes[-1].set_xlabel(cfg.get("xlabel") or "Time")
                if cfg.get("tight_layout", True) and not cfg.get("constrained_layout", False):
                    fig.tight_layout()
                _maybe_save(fig, cfg)

            return fig, axes


    def plot_phase_portrait(self, fig=None, axis=None, alpha=0.1, plot_cfg=None):
        """Plot a phase portrait from cached deterministic transient trajectories.

        For two species, plots x1(t) vs x2(t).
        For three species, plots a 3D trajectory.

        Args:
            fig: Optional matplotlib Figure.
            axis: Optional matplotlib Axis (2D or 3D).
            alpha: Line transparency for overlaid trajectories.
            plot_cfg: Optional dict controlling paper-ready styling. Supported keys
                are the same as in plot_transient_response.

        Returns:
            Tuple[Figure, Axes]: (fig, axis).

        Raises:
            ValueError: If no cached deterministic transient response is present.
            ValueError: If num_species is not 2 or 3.
        """
        if self.last_task_info.get("type") != "transient response":
            raise ValueError("No transient response data available. Run transient_response() first.")

        cfg = _merge_cfg(_DEFAULT_PLOT_CFG, plot_cfg)

        with paper_rc_context(cfg.get("rc")):
            if fig is None and axis is None:
                if self.num_species == 2:
                    figsize = cfg.get("figsize") or (2.2, 2.2)
                    fig, axis = plt.subplots(
                        figsize=figsize,
                        constrained_layout=bool(cfg.get("constrained_layout", False)),
                    )
                elif self.num_species == 3:
                    figsize = cfg.get("figsize") or (2.6, 2.6)
                    fig = plt.figure(figsize=figsize)
                    axis = fig.add_subplot(111, projection="3d")
                else:
                    raise ValueError("Phase portrait can only be plotted for 2 or 3 species.")

            trajs = self.last_task_info["trajectories"]
            a = cfg["alpha"] if cfg["alpha"] is not None else alpha
            lw = cfg["lw"]

            if self.num_species == 2:
                for j in range(len(trajs)):
                    tr = np.asarray(trajs[j], dtype=float)
                    axis.plot(tr[0, :], tr[1, :], alpha=a, linewidth=lw)

                axis.set_xlabel(cfg.get("xlabel") or self.species_labels[0])
                axis.set_ylabel(cfg.get("ylabel") or self.species_labels[1])
                axis.set_title(cfg.get("title") or "Phase portrait", pad=cfg.get("title_pad", 3.0))
                axis.set_aspect("equal", adjustable="box")
                _apply_axes_style(axis, cfg)
            else:
                for j in range(len(trajs)):
                    tr = np.asarray(trajs[j], dtype=float)
                    axis.plot(tr[0, :], tr[1, :], tr[2, :], alpha=a, linewidth=lw)

                axis.set_xlabel(self.species_labels[0])
                axis.set_ylabel(self.species_labels[1])
                axis.set_zlabel(self.species_labels[2])
                axis.set_title(cfg.get("title") or "Phase portrait", pad=cfg.get("title_pad", 3.0))

            if cfg.get("tight_layout", True) and not cfg.get("constrained_layout", False):
                fig.tight_layout()
            _maybe_save(fig, cfg)
            return fig, axis


    def plot_dose_response(self, fig=None, axes=None, alpha=0.5, plot_cfg=None):
        """Plot dose-response curves from cached simulation data.

        Two cache modes are supported:

        1) Dose-response cache:
        - last_task_info['type'] == 'dose response'
        - last_task_info['input doses'] : array-like
        - last_task_info['outputs'] : list of arrays shaped (q, n_doses) or similar

        2) Transient-response-derived dose response:
        - last_task_info['type'] == 'transient response'
        - last_task_info['inputs'] : list of u vectors (assumes scalar dose is u[0])
        - last_task_info['initial_conditions'] : list of x0 vectors
        - last_task_info['outputs'] : list of arrays shaped (q, T) for each (ic,u) scenario
        The method groups outputs by IC block and plots final-time output vs u[0].

        Paper-ready styling can be controlled via plot_cfg.

        Args:
            fig: Optional matplotlib Figure.
            axes: Optional axes list/array. If None, new axes are created (one per output).
            alpha: Line transparency for curves (default: 0.5).
            plot_cfg: Optional dict controlling paper-ready styling (see plot_transient_response).

        Returns:
            Tuple[Figure, List[Axes]]: (fig, axes).

        Raises:
            ValueError: If neither dose-response nor transient-response cache is present.
            KeyError: If required cache keys are missing.
        """
        ttype = self.last_task_info.get("type")
        if ttype not in ("dose response", "transient response"):
            raise ValueError("No dose response data available. Run dose_response() or transient_response() first.")

        cfg = _merge_cfg(_DEFAULT_PLOT_CFG, plot_cfg)

        with paper_rc_context(cfg.get("rc")):
            if fig is None and axes is None:
                figsize = cfg.get("figsize") or (2.8, 1.6 * self.num_outputs)
                fig, axes = plt.subplots(
                    self.num_outputs, 1, figsize=figsize, sharex=False,
                    constrained_layout=bool(cfg.get("constrained_layout", False)),
                )
            axes = _ensure_axes_list(axes, self.num_outputs)

            a = cfg["alpha"] if cfg["alpha"] is not None else alpha
            lw = cfg["lw"]

            if ttype == "dose response":
                u_dose = np.asarray(self.last_task_info["input doses"], dtype=float)
                outs = self.last_task_info["outputs"]

                for i in range(self.num_outputs):
                    ax = axes[i]
                    n_series = 0
                    for j in range(len(outs)):
                        y = np.asarray(outs[j], dtype=float)
                        yi = y[i, :] if y.ndim == 2 else y
                        ax.plot(u_dose, yi, alpha=a, linewidth=lw)
                        n_series += 1

                    species = self.species_labels[self.output_idx[i]]
                    ax.set_title(cfg.get("title") or f"{species}", pad=cfg.get("title_pad", 3.0))
                    ax.set_xlabel(cfg.get("xlabel") or "Dose")
                    ax.set_ylabel(cfg.get("ylabel") or "Response")
                    _apply_axes_style(ax, cfg)
                    _maybe_legend(ax, n_series, cfg)

            else:
                u_list = self.last_task_info["inputs"]
                x0_list = self.last_task_info["initial_conditions"]
                outs = self.last_task_info["outputs"]

                # In your current code: outputs are ordered such that for each IC block
                # you have a sweep over u. Infer step size.
                if len(x0_list) == 0:
                    raise ValueError("initial_conditions is empty; cannot build dose-response grouping.")
                step = len(outs) // len(x0_list)
                if step <= 0:
                    raise ValueError("Could not infer IC grouping step for dose-response.")

                u_dose = np.array([np.asarray(u).reshape(-1)[0] for u in u_list], dtype=float)

                for i in range(self.num_outputs):
                    ax = axes[i]
                    n_series = 0
                    for k in range(len(x0_list)):
                        block = outs[step * k : step * (k + 1)]
                        y_dose = np.array([np.asarray(y)[i, -1] for y in block], dtype=float)
                        ax.plot(u_dose, y_dose, alpha=a, linewidth=lw, label=(f"IC {k}" if len(x0_list) <= 10 else None))
                        n_series += 1

                    species = self.species_labels[self.output_idx[i]]
                    ax.set_title(cfg.get("title") or f"{species}", pad=cfg.get("title_pad", 3.0))
                    ax.set_xlabel(cfg.get("xlabel") or "Dose (u₁)")
                    ax.set_ylabel(cfg.get("ylabel") or "Final output")
                    _apply_axes_style(ax, cfg)
                    _maybe_legend(ax, n_series, cfg)

            if cfg.get("tight_layout", True) and not cfg.get("constrained_layout", False):
                fig.tight_layout()
            _maybe_save(fig, cfg)
            return fig, axes


    def plot_frequency_content(self, fig=None, axes=None, alpha=0.1, t0=0.0, plot_cfg=None):
        """Plot normalized one-sided FFT magnitude spectra from cached transients.

        For each cached trajectory:
        1) keep samples with time >= t0,
        2) subtract mean to remove DC component,
        3) compute one-sided FFT magnitudes,
        4) normalize each spectrum by its maximum,
        5) overlay spectra across scenarios.

        Args:
            fig: Optional matplotlib Figure.
            axes: Optional axes list/array. If None, new axes are created (one per output).
            alpha: Line transparency for overlaid spectra.
            t0: Time cutoff; only samples with time >= t0 are used.
            plot_cfg: Optional dict controlling paper-ready styling (see plot_transient_response).

        Returns:
            Tuple[Figure, List[Axes]]: (fig, axes).

        Raises:
            ValueError: If no cached deterministic transient response is present.
            ValueError: If insufficient points exist after t0.
            ValueError: If inferred sampling interval is non-positive.
        """
        if self.last_task_info.get("type") != "transient response":
            raise ValueError("No transient response data available. Run transient_response() first.")

        cfg = _merge_cfg(_DEFAULT_PLOT_CFG, plot_cfg)

        with paper_rc_context(cfg.get("rc")):
            if fig is None and axes is None:
                figsize = cfg.get("figsize") or (2.8, 1.6 * self.num_outputs)
                fig, axes = plt.subplots(
                    self.num_outputs, 1, figsize=figsize, sharex=False,
                    constrained_layout=bool(cfg.get("constrained_layout", False)),
                )
            axes = _ensure_axes_list(axes, self.num_outputs)

            time = np.asarray(self.last_task_info["time_horizon"], dtype=float)
            mask = time >= float(t0)
            if int(mask.sum()) < 2:
                raise ValueError("Not enough data points after t0 to compute Fourier transform.")

            dt = float(np.mean(np.diff(time[mask])))
            if dt <= 0:
                raise ValueError("Non-positive sampling interval inferred from time_horizon.")

            N = int(mask.sum())
            freqs = np.fft.rfftfreq(N, d=dt)

            outs = self.last_task_info["outputs"]
            a = cfg["alpha"] if cfg["alpha"] is not None else alpha
            lw = cfg["lw"]

            for i in range(self.num_outputs):
                ax = axes[i]
                n_series = 0
                for j in range(len(outs)):
                    y = np.asarray(outs[j], dtype=float)
                    yi = y[i, :] if y.ndim == 2 else y
                    yi = yi[mask]
                    yi = yi - np.mean(yi)
                    Y = np.fft.rfft(yi)
                    mag = np.abs(Y)
                    mag = mag / (np.max(mag) + 1e-12)
                    ax.plot(freqs, mag, alpha=a, linewidth=lw)
                    n_series += 1

                species = self.species_labels[self.output_idx[i]]
                ax.set_title(cfg.get("title") or f"{species}", pad=cfg.get("title_pad", 3.0))
                ax.set_xlabel(cfg.get("xlabel") or "Frequency")
                ax.set_ylabel(cfg.get("ylabel") or "Normalized |FFT|")
                _apply_axes_style(ax, cfg)
                _maybe_legend(ax, n_series, cfg)

            if cfg.get("tight_layout", True) and not cfg.get("constrained_layout", False):
                fig.tight_layout()
            _maybe_save(fig, cfg)
            return fig, axes


    def plot_SSA_transient_response(self, fig=None, axes=None, alpha=0.2, plot_cfg=None):
        r"""Plot cached SSA transient responses as mean ± std bands.

        Expects stochastic cache produced by `transient_response_SSA(...)`:
        - last_task_info['type'] == 'transient response SSA'
        - last_task_info['time_horizon'] : np.ndarray
        - last_task_info['outputs'] : List[np.ndarray] mean trajectories (q, T)
        - last_task_info['outputs_std'] : List[np.ndarray] std trajectories (q, T)
        - optionally last_task_info['inputs'] : List[u] for labeling

        Args:
            fig: Optional matplotlib Figure.
            axes: Optional axes list/array. If None, new axes are created (one per output).
            alpha: Transparency for std shading (default: 0.2).
            plot_cfg: Optional dict controlling paper-ready styling (see plot_transient_response).
                Note: plot_cfg["alpha"] (if set) overrides the *line* alpha, while
                the std-band alpha defaults to the method argument `alpha` unless
                plot_cfg includes "band_alpha".

                Additional supported key:
                - band_alpha: float to override std-band transparency

        Returns:
            Tuple[Figure, List[Axes]]: (fig, axes).

        Raises:
            ValueError: If no SSA cache is present.
            KeyError: If required cache keys are missing.
        """
        if self.last_task_info.get("type") != "transient response SSA":
            raise ValueError("No stochastic transient response data available. Run transient_response_SSA() first.")

        cfg = _merge_cfg(_DEFAULT_PLOT_CFG, plot_cfg)

        with paper_rc_context(cfg.get("rc")):
            if fig is None and axes is None:
                figsize = cfg.get("figsize") or (2.8, 1.6 * self.num_outputs)
                fig, axes = plt.subplots(
                    self.num_outputs, 1, figsize=figsize, sharex=True,
                    constrained_layout=bool(cfg.get("constrained_layout", False)),
                )
            axes = _ensure_axes_list(axes, self.num_outputs)

            time = np.asarray(self.last_task_info["time_horizon"], dtype=float)
            mean_data = self.last_task_info["outputs"]
            std_data = self.last_task_info["outputs_std"]
            inputs = self.last_task_info.get("inputs", [])

            line_alpha = cfg["alpha"] if cfg["alpha"] is not None else 1.0
            lw = cfg["lw"]
            band_alpha = cfg.get("band_alpha", alpha)

            for i in range(self.num_outputs):
                ax = axes[i]
                species_name = self.species_labels[self.output_idx[i]]

                n_series = 0
                for j in range(len(mean_data)):
                    y_mean = np.asarray(mean_data[j], dtype=float)[i, :]
                    y_std = np.asarray(std_data[j], dtype=float)[i, :]

                    label = None
                    if inputs and j < len(inputs) and len(mean_data) <= 10:
                        label = f"u={np.array2string(np.asarray(inputs[j]), precision=2, separator=',')}"

                    line, = ax.plot(time, y_mean, alpha=line_alpha, linewidth=lw, label=label)
                    ax.fill_between(
                        time,
                        y_mean - y_std,
                        y_mean + y_std,
                        color=line.get_color(),
                        alpha=band_alpha,
                        linewidth=0.0,
                    )
                    n_series += 1

                ax.set_title(cfg.get("title") or f"{species_name}", pad=cfg.get("title_pad", 3.0))
                ax.set_xlabel(cfg.get("xlabel") or "Time")
                ax.set_ylabel(cfg.get("ylabel") or "Count / Concentration")
                _apply_axes_style(ax, cfg)
                _maybe_legend(ax, n_series, cfg)

            if cfg.get("tight_layout", True) and not cfg.get("constrained_layout", False):
                fig.tight_layout()
            _maybe_save(fig, cfg)
            return fig, axes


    
    # ------------------------ CVODE Solver Method ------------------------
    
    def solve_with_cvode(self, x0, time_horizon, u, nonneg_idx, stop_fn):
        """Integrate the IOCRN ODE using CVODE and return a solve_ivp-like solution.

        This is an internal helper used when `self.solver == 'CVODE'`. It wraps the
        `sksundae.cvode.CVODE` interface into an object that mimics SciPy's
        `solve_ivp` result structure (fields `.t`, `.y`, `.status`, `.message`,
        and `.raw`).

        Args:
            x0: Initial condition vector of shape `(n,)`.
            time_horizon: 1D array of requested times at which the solver should
                return values (similar to `t_eval` in SciPy).
            u: Constant input vector of shape `(p,)` used by the rate function.
            nonneg_idx: Indices of state variables constrained to be nonnegative.
                If provided, CVODE inequality constraints are applied.
            stop_fn: Event-like function `stop_fn(t, x)` returning a scalar whose
                sign determines whether integration should stop (similar to SciPy
                event functions). Its attributes `terminal` and `direction` (if
                present) are forwarded to the CVODE events wrapper.

        Returns:
            A lightweight solution object with:

                - `t`: 1D array of solver times
                - `y`: 2D array of shape `(n, len(t))`
                - `status`: integer status code
                - `message`: solver message
                - `raw`: the underlying CVODE solution object

        Raises:
            ImportError: If `sksundae`/CVODE is not installed.
        """

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


# TODO check 'initial_conditions' and 'initial conditions' in last_task_info consistency

import numpy as np
from sksundae.cvode import CVODE

# code to convert a SciPy-style event function to a CVODE-style event function

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


def _pulse_step_and_shading_from_intervals(
    seg_intervals,
    u_pulse,
    *,
    gap: bool,
):
    """
    Build (boundaries, step_values, seg_values, gap_bounds) from legacy seg_intervals and u_pulse.

    seg_intervals: [(0,T0),(0,T1),...]
    u_pulse: array-like, first element used as ON amplitude
    gap: if True, longest segment is treated as a GAP (forced LOW) and pulse phase restarts after it.

    Returns:
    boundaries: (K+1,)
    step_values: (K+1,) suitable for ax.step(..., where="post")
    seg_values: (K,) per-segment values (useful for shading)
    gap_bounds: None or (t_start, t_end) for the gap segment
    """
    durations = np.array([float(iv[1]) for iv in seg_intervals], dtype=float)
    boundaries = np.cumsum(np.concatenate([[0.0], durations]))  # length K+1
    K = int(durations.size)

    on_value = float(np.asarray(u_pulse).reshape(-1)[0])

    gap_bounds = None

    if gap and K > 0:
        gap_idx = int(np.argmax(durations))
        gap_bounds = (float(boundaries[gap_idx]), float(boundaries[gap_idx + 1]))

        seg_values = np.zeros(K, dtype=float)

        # before gap: ON on even indices (original convention)
        for k in range(0, gap_idx):
            seg_values[k] = on_value if (k % 2 == 0) else 0.0

        # gap: forced low
        seg_values[gap_idx] = 0.0

        # after gap: restart phase (first segment after gap is ON)
        phase = 0
        for k in range(gap_idx + 1, K):
            seg_values[k] = on_value if (phase % 2 == 0) else 0.0
            phase += 1
    else:
        seg_values = np.array([on_value if (k % 2 == 0) else 0.0 for k in range(K)], dtype=float)

    # Drop at end without adding extra segment duration
    step_values = np.concatenate([seg_values, [0.0]])  # length K+1 for where="post"
    return boundaries, step_values, seg_values, gap_bounds


def _plot_one_frequency(
    *,
    ax,
    t,
    outputs,
    out_i: int,
    title: str,
    alpha: float,
    have_input: bool,
    boundaries,
    step_values,
    seg_values,
    gap_bounds,
    input_color: str,
    trj_color: str,
    shade_on: bool,
    pulse_lw: float,
    pulse_ls: str,
    y_label: str,
    legend_label: str,
):
    """Plot a single output channel for one frequency on a provided axis."""
    # trajectories
    for scen in range(len(outputs)):
        y = np.asarray(outputs[scen], dtype=float)
        y_i = y[out_i, :] if y.ndim == 2 else y
        if legend_label:
            ax.plot(t, y_i - y_i[0], alpha=alpha, color=trj_color, label=legend_label)
        else:
            ax.plot(t, y_i - y_i[0], alpha=alpha, color=trj_color)

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.2)

    if have_input and boundaries is not None:
        ax2 = ax.twinx()
        ax2.step(boundaries, step_values, where="post",
                color=input_color, linewidth=pulse_lw, linestyle=pulse_ls, alpha=0.)
        ax2.set_ylabel("Input u₁", color=input_color)
        ax2.tick_params(axis="y", labelcolor=input_color)

        if shade_on:
            for k in range(len(seg_values)):
                if seg_values[k] > 0:
                    ax.axvspan(boundaries[k], boundaries[k + 1], color=input_color, alpha=0.15)

        if gap_bounds is not None:
            ax.axvline(gap_bounds[0], alpha=0.25)
            ax.axvline(gap_bounds[1], alpha=0.25)