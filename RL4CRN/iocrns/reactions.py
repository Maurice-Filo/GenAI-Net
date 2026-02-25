"""
Reaction primitives for IOCRNs.

This module defines the core reaction objects used throughout the package:

- `Reaction` is the abstract base class that stores reaction *structure*
  (reactants/products), parameter slots (possibly unknown), optional input-channel
  associations for parameters, and context hooks used when embedding reactions in
  an IOCRN or registering them in a reaction library.

- `MassAction` implements standard mass-action kinetics with an optional
  scalar input modulation of the rate constant.

- `HillProduction` implements regulated production from the empty complex
  using Hill-type activation/repression.

Reactions are designed to be:

- **library-friendly**: each reaction can be registered and retrieved by ID,
  while equality is typically based on a structure-only ``signature``.
- **CRN-friendly**: once a reaction is placed in an IOCRN, it can precompute
  index mappings (species/input labels → integer indices) for fast simulation.

Conventions:

- Stoichiometry is represented implicitly by repeated labels in
  ``reactant_labels`` / ``product_labels`` (e.g. ``['A','A']`` means 2A).
- Parameter vectors are aligned with ``input_channels`` and
  ``params_controllability``; unknown parameters are represented by ``None``.
"""


import numpy as np

class Reaction:
    # Note: whenever adding new input components for the constructor (e.g. catalysts), ensure they are sorted as well
    def __init__(self, reactant_labels, product_labels, input_channels=[None], params=[None], params_controllability=[False], ID = None, signature=None):
        """Base class for reactions used in IOCRNs.

        A `Reaction` defines:

        - a *structure* (reactants/products, optional "inputs" that modulate parameters),
        - a parameter vector (with optional unknown entries),
        - bookkeeping fields used when the reaction is placed inside a CRN
            (species/input indices) and inside a library (reaction ID).

        The class is intended to be subclassed by specific kinetic laws (e.g.
        `MassAction`, `HillProduction`). Subclasses must implement
        `propensity` and `to_reaction_format`.

        Parameters are stored in parallel lists:

        - ``params[j]``: numerical value (float-like) or ``None`` if unknown.
        - ``params_controllability[j]``: whether the parameter is controllable.
        - ``input_channels[j]``: name of the input signal multiplying/modulating
        that parameter, or ``None`` if the parameter is not input-dependent.

        These lists must have identical length.

        Notes:
            - The constructor sorts reactants/products and de-duplicates + sorts
            ``input_labels`` (excluding ``None``) for deterministic behavior.
            - Equality (``==``) is defined via the reaction ``signature`` only.
            A signature is meant to identify the *topology/structure* of a reaction,
            ignoring parameter values.

        Args:
            reactant_labels: Labels of reactant species (can be empty for ∅).
            product_labels: Labels of product species (can be empty for ∅).
            input_channels: Input channel per parameter slot. Can contain ``None``.
            params: Parameter values per slot. ``None`` entries denote unknown parameters.
            params_controllability: Boolean flag per slot indicating controllability.
            ID: Optional integer identifier used when the reaction is registered in a
                `RL4CRN.iocrns.reaction_library.ReactionLibrary`.
            signature: Optional string uniquely identifying the *structure* of the reaction.

        Attributes:
            reactant_labels: Sorted list of reactant labels.
            product_labels: Sorted list of product labels.
            input_channels: Input channel list, aligned with `params`.
            params: Parameter list (may contain ``None`` values).
            params_controllability: Controllability list, aligned with `params`.
            num_parameters: Number of parameters (length of `params`).
            input_labels: Sorted list of unique non-``None`` input channel labels.
            ID: Reaction ID in a library context (or ``None``).
            signature: Structural signature (or ``None``).
            crn: Set by `set_crn_context` when placed in an IOCRN.
        """
        # Assert that input_channels, params, and params_controllability have the same length
        assert len(input_channels) == len(params) == len(params_controllability), "Input channels, parameters, and parameter controllability lists must have the same length."

        # Record the reaction details
        self.reactant_labels = reactant_labels                  # list of strings, can be empty
        self.product_labels = product_labels                    # list of strings, can be empty
        self.input_channels = input_channels                    # list of strings or None, cannot be empty
        self.params = params                                    # List of floats, where None indicates unknown parameters, cannot be empty
        self.params_controllability = params_controllability    # List of booleans, cannot be empty
        self.num_parameters = len(params)                       # Total number of parameters in the reaction
        self.ID = ID                                            # Integer, can be None
        self.signature = signature                              # String, can be None

        # Extract the unique input labels (excluding None)
        self.input_labels = [channel for channel in input_channels if channel is not None]  
        self.input_labels = list(set(self.input_labels))        # List of strings, can be empty

        # Sort the labels alphanumerically
        self.reactant_labels.sort()
        self.product_labels.sort()
        self.input_labels.sort()               

    # def set_unknown_parameters(self, params, initial_idx=0): 
    #     """Set unknown parameters from a provided list. """
    #     i = initial_idx
    #     for j in range(self.num_parameters):
    #         if self.params[j] is None:
    #             self.params[j] = params[i]
    #             i += 1

    def get_ID(self):
        """Return the reaction's library ID (or ``None`` if unset)."""
        return self.ID
    
    def set_ID(self, ID):
        """Set the reaction's library ID."""
        self.ID = ID

    def get_num_controllable_parameters(self):
        """Return the number of controllable parameters."""
        return sum(self.params_controllability)
    
    def get_num_unknown_params(self):
        """Return the number of unknown parameters (``None`` entries)."""
        return sum(1 for param in self.params if param is None)
        
    def propensity(self, x, u):
        """Compute the propensity/rate contribution of this reaction.

        Subclasses must implement this.

        Args:
            x: Full CRN state vector (species concentrations/counts).
            u: Full input vector (input signal values).

        Returns:
            float: Propensity of the reaction under the provided state and inputs.
        """
        pass

    def __call__(self, x, u):
        """Alias for `propensity`."""
        return self.propensity(x, u)
    
    def set_crn_context(self, crn):
        """Attach the reaction to an IOCRN context.

        Subclasses typically override this to precompute index arrays for fast
        propensity evaluation.

        Args:
            crn: IOCRN instance providing label->index maps.
        """
        self.crn = crn

    def __eq__(self, other):
        """Check structural equivalence via ``signature``.

        Two reactions are considered equal if their signatures are equal.
        """
        return self.signature == other.signature

    def set_library_context(self, reaction_library):
        """Resolve and set this reaction's ID from a reaction library.

        Finds the first reaction in `reaction_library` with the same signature
        (via `__eq__`) and sets `ID` accordingly.

        Args:
            reaction_library: A `RL4CRN.iocrns.reaction_library.ReactionLibrary`.

        Raises:
            ValueError: If no matching reaction is found.
        """
        for reaction in reaction_library.reactions:
            if self == reaction:
                self.ID = reaction.ID
                return
        raise ValueError("Reaction not found in the provided reaction library.")
    
    def to_reaction_format(self):
        """Serialize the reaction to the project's DSL/CRN text format.

        Subclasses must implement this.

        Returns:
            str: A line of text such as `'A:1 -- mak(k) -> B:1;'`.
        """
        raise NotImplementedError("Subclasses must implement to_reaction_format")

    def _format_species_list(self, labels):
        """Format a multiset of species labels into DSL syntax.

        Examples:
            - ``['A', 'A', 'B']`` -> ``'A:2, B:1'``
            - ``[]`` -> ``'emptyset'``

        Args:
            labels: List of species labels, possibly with repeats.

        Returns:
            str: DSL-formatted species multiset.
        """
        if not labels:
            return "emptyset"
        
        counts = {s: labels.count(s) for s in set(labels)}
        # Sort by keys to ensure deterministic output
        formatted_parts = [f"{name}:{counts[name]}" for name in sorted(counts.keys())]
        return ", ".join(formatted_parts)

class MassAction(Reaction):
    def __init__(self, reactant_labels, product_labels, input_channels=[None], params=[None], params_controllability=[True]):
        r"""Mass-action reaction with optional input modulation.

        A mass-action reaction has the form:

        $$\sum_i \nu_i X_i \;\longrightarrow\; \sum_i \nu'_i X_i$$

        with propensity

        $$a(x, u) = k \;\prod_{X_i \in \text{reactants}} x_i \; g(u),$$

        where:
        
        - $k$ is the (possibly unknown) rate constant,
        - $x_i$ are the reactant concentrations/counts,
        - $g(u)$ is an optional scalar input modulation.
            In the current implementation:
            - if the single ``input_channel`` is ``None`` then $g(u)=1$,
            - otherwise $g(u)=u_{c}$ for the corresponding input channel/index.

        The parameter layout is fixed to a single scalar parameter:

        - ``params = [k]

        Args:
            reactant_labels: Reactant species labels. Empty means a 0th-order reaction (∅ → ...).
            product_labels: Product species labels. Empty means degradation (... → ∅).
            input_channels: A single-element list `[channel]` or `[None]`.
            params: A single-element list `[k]` where `k` may be ``None``.
            params_controllability: A single-element list indicating controllability of `k`.

        Attributes:
            rate_constant: The mass-action rate constant `k` (float or ``None``).
            signature: Structural signature (depends only on reactants/products).
            num_continuous_parameters: Always 1.
            num_discrete_parameters: Always 0.
            num_unknown_params: Number of unknown parameters (0 or 1).
        """
        
        # Call the parent constructor
        super().__init__(reactant_labels, product_labels, input_channels, params, params_controllability)
        
        # Ensure params has exactly one element (the rate constant)
        assert len(params) == 1, "MassAction reaction must have exactly one parameter (the rate constant)."

        # Create the reaction signature: depends on the reaction structure only, not on the parameters or inputs
        self.signature = str(('MAK', self.reactant_labels, self.product_labels))

        # Record the rate constant
        self.rate_constant = params[0]                              # float or None
        self.num_continuous_parameters = 1                          # Mass action has one continuous parameter (the rate constant)
        self.num_discrete_parameters = 0                            # Mass action has no discrete parameters
        self.num_unknown_params = self.get_num_unknown_params()     # Number of unknown parameters (0 or 1)

    def set_parameters(self, params):
        """Set the rate constant.

        Args:
            params: Single-element list `[k]`.

        Raises:
            AssertionError: If `params` does not have length 1.
        """
        
        # Ensure params has exactly one element (the rate constant)
        assert len(params) == 1, "MassAction reaction must have exactly one parameter (the rate constant)."

        # Set the rate constant
        self.rate_constant = params[0]

        # Update the params list
        self.params = params
 
    def get_involved_species(self):
        """Return sorted unique species involved in the reaction (reactants ∪ products)."""
        species = list(set(self.reactant_labels + self.product_labels))
        species.sort()
        return species
    
    def get_involved_inputs(self):
        """Return input channel labels used by the reaction (excluding ``None``)."""
        return self.input_labels 
    
    def get_stoichiometry_dict(self):
        """Return stoichiometry coefficients as a dictionary.

        Reactants contribute -1 per occurrence, products +1 per occurrence.

        Returns:
            dict[str, int]: Mapping species label -> net stoichiometric coefficient.
        """
        
        stoich = {}
        for reactant_label in self.reactant_labels:
            if reactant_label in stoich:
                stoich[reactant_label] -= 1
            else:
                stoich[reactant_label] = -1
        for product_label in self.product_labels:
            if product_label in stoich:
                stoich[product_label] += 1
            else:
                stoich[product_label] = 1
        return stoich
    
    def set_crn_context(self, crn):
        """Precompute indices for fast propensity evaluation.

        Args:
            crn: IOCRN instance providing label->index maps.
        """
        
        super().set_crn_context(crn)
        self.reactant_idx = crn.species_label_to_idx(self.reactant_labels) # single index or list of indices
        self.product_idx = crn.species_label_to_idx(self.product_labels) # single index or list of indices
        self.input_idx = crn.input_label_to_idx(self.input_channels) # single index or list of indices

    def propensity(self, x, u):
        """Compute the mass-action propensity.

        Args:
            x: Full species state vector of the parent IOCRN.
            u: Full input vector of the parent IOCRN.

        Returns:
            float: Reaction propensity.

        Notes:
            This implementation multiplies all reactant entries via ``np.prod``.
            For repeated reactants, the state is indexed with repeated indices,
            so the product includes appropriate powers.
        """
        
        # Extract relevant species and inputs
        x = x[self.reactant_idx]
        u = u[self.input_idx[0]] if self.input_idx[0] is not None else 1.0

        # Compute the propensity using mass action kinetics
        return self.rate_constant * np.prod(x) * u
    
    def __str__(self):
        """Human-readable string representation."""
        try:
            species_str = f"{self.reactant_labels} : {self.reactant_idx}, {self.product_labels} : {self.product_idx}, {self.input_channels} : {self.input_idx}"
        except:
            species_str = "unset"
            
        reactants_str = ' + '.join(self.reactant_labels) if self.reactant_labels else '∅'
        products_str = ' + '.join(self.product_labels) if self.product_labels else '∅'
        inputs_str = self.input_channels[0] if self.input_channels[0] is not None else ''
        if inputs_str == '':
            return f"{reactants_str} ----> {products_str};  [MAK({self.rate_constant})]" 
        return f"{reactants_str} ----> {products_str};  [MAK({self.rate_constant}, {inputs_str})]" 
    
    def to_reaction_format(self):
        """Serialize to the CRN DSL.

        Returns:
            str: DSL line, e.g. ``'A:1 -- mak(k*u1) -> B:1;'``.

        Notes:
            The DSL formatting here distinguishes the "zero-order input generation"
            special-case when there are no reactants but an input channel exists.
        """
        lhs = self._format_species_list(self.reactant_labels)
        rhs = self._format_species_list(self.product_labels)
        
        inp = self.input_channels[0]
        k = self.rate_constant

        # Case 1: Zero-order input generation: 'emptyset -- u -> X:1'
        if not self.reactant_labels and inp is not None:
            arrow = f"-- {k}*{inp} ->"
        # Case 2: Standard or Input-Modulated Mass Action
        else:
            if inp is not None:
                arrow = f"-- mak({k}*{inp}) ->"
            else:
                arrow = f"-- mak({k}) ->"
        
        return f"{lhs} {arrow} {rhs};"

class HillProduction(Reaction):
    def __init__(self, product_labels, activator_labels, repressor_labels, input_channels=[None], params=[None], params_controllability=[True]):
        r"""Hill-regulated production reaction (Hill coefficients fixed to 1).

        This reaction represents regulated production from the empty complex:

        $$\emptyset \rightarrow \text{products}$$

        with a propensity of the form:

        $$
            a(x) = b + V_{\max} \;\Bigg(\prod_{i \in A}
            \frac{x_i}{K_{a,i} + x_i}\Bigg)
            \Bigg(\prod_{j \in R}
            \frac{K_{r,j}}{K_{r,j} + x_j}\Bigg),$$

        where:
        - $b$ is the basal production rate,
        - $V_{\max}$ is the maximal regulated production rate,
        - $A$ is the set of activators, $R$ the set of repressors,
        - Hill coefficients are fixed to 1 in this implementation.

        Parameter layout (after sorting activator/repressor labels):
            ``params = [b, Vmax, Ka_1, ..., Ka_|A|, Kr_1, ..., Kr_|R|]``

        Note:
            Each parameter slot may be associated with an ``input_channel`` label.
            In the current implementation, the input vector `u` is collected but not
            applied inside `propensity`. This is a placeholder for future
            extensions where parameters may be modulated by inputs.

        Args:
            product_labels: Product species labels (non-empty).
            activator_labels: Activator species labels (may be empty).
            repressor_labels: Repressor species labels (may be empty).
            input_channels: Input channel list aligned with `params` (may contain None).
            params: Parameter list aligned with `input_channels` (may contain None).
            params_controllability: Controllability flags aligned with `params`.

        Attributes:
            basal_rate: Basal rate `b`.
            maximal_rate: Maximal rate `Vmax`.
            activator_dissociation_constants: List of `Ka` values, aligned with sorted activators.
            repressor_dissociation_constants: List of `Kr` values, aligned with sorted repressors.
            signature: Structural signature independent of parameter values.
            num_continuous_parameters: `2 + |A| + |R|`.
            num_discrete_parameters: 0 (Hill coefficients fixed to 1 here).
            num_unknown_params: Number of unknown parameters (`None` entries).
        """
        
        # Ensure the number of parameters is correct (2 + number of activators + number of repressors)
        assert len(params) ==  2 + len(activator_labels) + len(repressor_labels), "HillProduction reaction must have exactly 2 + number of activators + number of repressors parameters (the basal rate, the maximal rate, and the dissociation constants for each activator and repressor)."

        # Sort and Record the activator and repressor labels
        if len(activator_labels) > 1:
            sorted_activator_indices = sorted(range(len(activator_labels)), key=lambda i: activator_labels[i])
            self.activator_labels = [activator_labels[i] for i in sorted_activator_indices]
        else:
            self.activator_labels = activator_labels

        if len(repressor_labels) > 1:
            sorted_repressor_indices = sorted(range(len(repressor_labels)), key=lambda i: repressor_labels[i])
            self.repressor_labels = [repressor_labels[i] for i in sorted_repressor_indices]
        else:
            self.repressor_labels = repressor_labels

        # Sort the parameters
        # Activators
        self.num_activators = len(self.activator_labels)
        params_activators = [params[2 + i : 3 + i] for i in range(self.num_activators)]
        params_activators = sum([params_activators[i] for i in sorted_activator_indices] if self.num_activators > 1 else params_activators, [])
        # Repressors
        self.num_repressors = len(self.repressor_labels)
        params_repressors = [params[2 + self.num_activators + i : 3 + self.num_activators + i] for i in range(self.num_repressors)]
        params_repressors = sum([params_repressors[i] for i in sorted_repressor_indices] if self.num_repressors > 1 else params_repressors, [])
        # Combine sorted parameters
        params = params[0:2] + params_activators + params_repressors

        # Sort the input channels
        # Activators
        input_channels_activators = [input_channels[2 + i : 3 + i] for i in range(self.num_activators)]
        input_channels_activators = sum([input_channels_activators[i] for i in sorted_activator_indices] if self.num_activators > 1 else input_channels_activators, [])
        # Repressors
        input_channels_repressors = [input_channels[2 + self.num_activators + i : 3 + self.num_activators + i] for i in range(self.num_repressors)]
        input_channels_repressors = sum([input_channels_repressors[i] for i in sorted_repressor_indices] if self.num_repressors > 1 else input_channels_repressors, [])
        # Combine sorted input channels
        input_channels = input_channels[0:2] + input_channels_activators + input_channels_repressors

        # Sort the parameter controllability
        # Activators
        params_controllability_activators = [params_controllability[2 + i : 3 + i] for i in range(self.num_activators)]
        params_controllability_activators = sum([params_controllability_activators[i] for i in sorted_activator_indices] if self.num_activators > 1 else params_controllability_activators, [])
        # Repressors
        params_controllability_repressors = [params_controllability[2 + self.num_activators + i : 3 + self.num_activators + i] for i in range(self.num_repressors)]
        params_controllability_repressors = sum([params_controllability_repressors[i] for i in sorted_repressor_indices] if self.num_repressors > 1 else params_controllability_repressors, [])
        # Combine sorted parameter controllability
        params_controllability = params_controllability[0:2] + params_controllability_activators + params_controllability_repressors

        # Call the parent constructor
        super().__init__([], product_labels, input_channels, params, params_controllability)

        # Create the reaction signature: depends on the reaction structure only, not on the parameters or inputs
        # Signature format: ('HILLProd', product_labels, activator_labels, repressor_labels) all sorted alphanumerically
        self.signature = str(('HILLProd', self.product_labels, self.activator_labels, self.repressor_labels))

        # Record the basal and maximal rates and the dissociation constants
        self.basal_rate = self.params[0]                                                 # float or None
        self.maximal_rate = self.params[1]                                               # float or None
        self.activator_dissociation_constants = self.params[2:2+self.num_activators:1] if self.activator_labels else []  # list of floats or None
        self.repressor_dissociation_constants = self.params[2+self.num_activators:2+self.num_activators+self.num_repressors:1] if self.repressor_labels else []  # list of floats or None

        self.num_continuous_parameters = 2 + len(self.activator_labels) + len(self.repressor_labels)                            # basal and maximal rates, and dissociation constants
        self.num_discrete_parameters = 0                                  
        self.num_unknown_params = self.get_num_unknown_params()   

    def set_parameters(self, params):
        """Set the full parameter vector.

        Layout:
            ``[b, Vmax, Ka_1, ..., Ka_|A|, Kr_1, ..., Kr_|R|]``

        Args:
            params: Parameter vector matching the layout above.

        Raises:
            AssertionError: If the provided vector has the wrong length.
        """

        # Ensure the number of parameters is correct (2 + number of activators + number of repressors)
        assert len(params) ==  2 + len(self.activator_labels) + len(self.repressor_labels), "HillProduction reaction must have exactly 2 + number of activators + number of repressors parameters (the basal rate, the maximal rate, and the dissociation constants for each activator and repressor)."

        # Set the basal and maximal rates, the dissociation constants, and the Hill coefficients
        self.basal_rate = params[0]                                                 # float or None
        self.maximal_rate = params[1]                                               # float or None
        self.activator_dissociation_constants = params[2:2+len(self.activator_labels)] if self.activator_labels else []  # list of floats or None
        self.repressor_dissociation_constants = params[2+len(self.activator_labels):2+len(self.activator_labels)+len(self.repressor_labels)] if self.repressor_labels else []  # list of floats or None

        # Update the params list
        self.params = params

    def get_involved_species(self):
        """Return sorted unique species involved (products ∪ activators ∪ repressors)."""
        species = list(set(self.product_labels + self.activator_labels + self.repressor_labels))
        species.sort()
        return species
    
    def get_involved_inputs(self):
        """ Returns a list of all inputs involved in the reaction. 
        Returns:
        - input_labels: list of strings representing the labels of the involved inputs. """

        return self.input_labels
    
    def get_stoichiometry_dict(self):
        """Return stoichiometry coefficients as a dictionary (products only).

        Returns:
            dict[str, int]: Mapping product label -> stoichiometric coefficient.
        """
        
        stoich = {}
        for product_label in self.product_labels:
            if product_label in stoich:
                stoich[product_label] += 1
            else:
                stoich[product_label] = 1
        return stoich
    
    def set_crn_context(self, crn):
        """Precompute indices for fast propensity evaluation.

        Args:
            crn: IOCRN instance providing label->index maps.
        """

        super().set_crn_context(crn)
        self.product_idx = crn.species_label_to_idx(self.product_labels) # single index or list of indices
        self.activator_idx = crn.species_label_to_idx(self.activator_labels) if self.activator_labels else [] # list of indices
        self.repressor_idx = crn.species_label_to_idx(self.repressor_labels) if self.repressor_labels else [] # list of indices
        self.input_idx = crn.input_label_to_idx(self.input_channels) # single index or list of indices

    def propensity(self, x, u):
        """Compute Hill-regulated production propensity.

        Args:
            x: Full species state vector of the parent IOCRN.
            u: Full input vector of the parent IOCRN.

        Returns:
            float: Reaction propensity.

        Notes:
            Input modulations of Hill coefficients are fixed to 1. This method currently does **not**
            apply input modulation to parameters even if `input_channels` are set.
        """
    
        # Extract relevant species and inputs        
        x_activators = x[self.activator_idx] if self.activator_idx else np.array([])
        x_repressors = x[self.repressor_idx] if self.repressor_idx else np.array([])
        u = np.array([u[i] if i is not None else 1 for i in self.input_idx])

        # Compute the activation term
        activation_term = 1.0
        for i in range(self.num_activators):
            Ka = self.activator_dissociation_constants[i]
            na = 1
            activation_term *= (x_activators[i]**na) / ((Ka*u[2 + i])**na + x_activators[i]**na) if Ka is not None and na is not None else 1.0

        # Compute the repression term
        repression_term = 1.0
        for i in range(self.num_repressors):
            Kr = self.repressor_dissociation_constants[i]
            nr = 1
            repression_term *= (Kr*u[2 + self.num_activators + i])**nr / ((Kr*u[2 + self.num_activators + i])**nr + x_repressors[i]**nr) if Kr is not None and nr is not None else 1.0

        # Compute the propensity using Hill kinetics
        # return self.basal_rate + (self.maximal_rate - self.basal_rate) * activation_term * repression_term 
        return self.basal_rate*u[0] + self.maximal_rate*u[1] * activation_term * repression_term 
    
    def __str__(self):
        """Human-readable string representation."""
        
        try:
            species_str = f"{self.product_labels} : {self.product_idx}, {self.activator_labels} : {self.activator_idx}, {self.repressor_labels} : {self.repressor_idx}, {self.input_channels} : {self.input_idx}"
        except:
            species_str = "unset"
            
        reactants_str = '∅'
        products_str = ' + '.join(self.product_labels) if self.product_labels else '∅'
        
        # Construct parameters string
        # Basal rate
        if self.input_channels[0] is None:
            params_str = f"b = {self.basal_rate}, "
        else:
            params_str = f"b = {self.basal_rate}{self.input_channels[0]}, "

        # Maximal rate
        if self.input_channels[1] is None:
            params_str += f"Vm = {self.maximal_rate},    "
        else:
            params_str += f"Vm = {self.maximal_rate}{self.input_channels[1]},    "

        # Activators
        params_str += f"(Ka, na) = "
        for i in range(self.num_activators):
            if self.input_channels[2 + i] is None:
                params_str += f"{self.activator_labels[i]}({self.activator_dissociation_constants[i]}, {1})"
            else:
                params_str += f"{self.activator_labels[i]}({self.activator_dissociation_constants[i]}{self.input_channels[2 + i]}, {1})"
            params_str += ", " if i < self.num_activators - 1 else ""

        # Repressors
        params_str += f";    (Kr, nr) = "
        for i in range(self.num_repressors):
            if self.input_channels[2 + self.num_activators + i] is None:
                params_str += f"{self.repressor_labels[i]}({self.repressor_dissociation_constants[i]}, {1})"
            else:
                params_str += f"{self.repressor_labels[i]}({self.repressor_dissociation_constants[i]}{self.input_channels[2 + self.num_activators + i]}, {1})"
            params_str += ", " if i < self.num_repressors - 1 else ""
        
        return f"{reactants_str} ----> {products_str};  [HILLProd({params_str})]"
    

    def to_reaction_format(self): # TODO: verify this method
        """Serialize to the CRN DSL.

        Returns:
            str: DSL line, e.g. ``'emptyset -- hill(b=..., Vm=..., A(Ka,1), R(Kr,1)) -> X:1;'``.

        Notes:
            This method emits a `hill(...)` construct assumed to be understood by
            your SSA/DSL backend. If the backend expects a different argument
            order or syntax, adapt accordingly.
        """
        # Reusing the string construction logic from __str__ but adapting the output format
        rhs = self._format_species_list(self.product_labels)
        
        # Helper to fmt param
        def fmt_p(val, inp):
            return f"{val}" if inp is None else f"{val}*{inp}"

        # 1. Basal
        b_str = fmt_p(self.basal_rate, self.input_channels[0])
        # 2. Vmax
        vm_str = fmt_p(self.maximal_rate, self.input_channels[1])
        
        # 3. Activators
        act_parts = []
        for i in range(self.num_activators):
            ka = fmt_p(self.activator_dissociation_constants[i], self.input_channels[2 + i])
            # DSL format for activation: species(Ka, na)
            act_parts.append(f"{self.activator_labels[i]}({ka}, 1)")
            
        # 4. Repressors
        rep_parts = []
        for i in range(self.num_repressors):
            kr = fmt_p(self.repressor_dissociation_constants[i], self.input_channels[2 + self.num_activators + i])
            rep_parts.append(f"{self.repressor_labels[i]}({kr}, 1)")

        # Assemble inside hill(...)
        # Assuming args order: b, Vm, [activators...], [repressors...]
        args = [f"b={b_str}", f"Vm={vm_str}"]
        if act_parts: args.extend(act_parts)
        if rep_parts: args.extend(rep_parts)
        
        hill_def = f"hill({', '.join(args)})"
        
        return f"emptyset -- {hill_def} -> {rhs};"
    
class ActiveDegradation(Reaction):
    def __init__(self, substrate_label, enzyme_label, input_channels=[None], params=[None], params_controllability=[True]):
        r"""Active degradation reaction (Hill coefficients fixed to 1).

        This reaction represents regulated degradation of a substrate species by an enzyme:

        $$\text{substrate} \rightarrow \emptyset$$

        with a propensity of the form:

        $$
            a(x) = a D\;\frac{x}{K_D + x},$$

        where:
        - $a$ is the maximal degradation rate per degradation enzyme,
        - $D$ is the degradation enzyme concentration (picked from the species),
        - $K$ is the Michaelis constant for degradation,
        - Hill coefficients are fixed to 1 in this implementation.

        Parameter layout (after sorting activator/repressor labels):
            ``params = [a, K]``

        Note:
            Each parameter slot may be associated with an ``input_channel`` label.
            In the current implementation, the input vector `u` is collected but not
            applied inside `propensity`. This is a placeholder for future
            extensions where parameters may be modulated by inputs.

        Args:
            substrate_label: Substrate species label (non-empty).
            enzyme_label: Enzyme species label (non-empty).
            input_channels: Input channel list aligned with `params` (may contain None).
            params: Parameter list aligned with `input_channels` (may contain None).
            params_controllability: Controllability flags aligned with `params`.

        Attributes:
            maximal_rate: Maximal rate `a`.
            michaelis_constant: Michaelis constant `K`.
            signature: Structural signature independent of parameter values.
            num_continuous_parameters: `2`.
            num_discrete_parameters: 0 (Hill coefficients fixed to 1 here).
            num_unknown_params: Number of unknown parameters (`None` entries).
        """
        
        # Ensure the number of parameters is correct (2)
        assert len(params) == 2, "ActiveDegradation reaction must have exactly 2 parameters (the maximal rate and the Michaelis constant)."

        # Record the enzyme and substrate labels
        assert len(enzyme_label) == 1, "ActiveDegradation reaction must have exactly one enzyme species label."
        self.enzyme_label = enzyme_label
        assert len(substrate_label) == 1, "ActiveDegradation reaction must have exactly one substrate species label."
        self.substrate_label = substrate_label

        # Call the parent constructor
        super().__init__(substrate_label, [], input_channels, params, params_controllability)

        # Create the reaction signature: depends on the reaction structure only, not on the parameters or inputs
        # Signature format: ('ActiveDeg', substrate_label, enzyme_label)
        self.signature = str(('ActiveDeg', self.substrate_label, self.enzyme_label))

        # Record the maximal rate and the Michaelis constant
        self.maximal_rate = self.params[0]                                                 # float or None
        self.michaelis_constant = self.params[1]                                           # float or None
        self.num_continuous_parameters = 2                            # maximal rate and Michaelis constant
        self.num_discrete_parameters = 0                                  
        self.num_unknown_params = self.get_num_unknown_params()   

    def set_parameters(self, params):
        """Set the full parameter vector.

        Layout:
            ``[a, K]``

        Args:
            params: Parameter vector matching the layout above.

        Raises:
            AssertionError: If the provided vector has the wrong length.
        """

        # Ensure the number of parameters is correct (2)
        assert len(params) ==  2 , "ActiveDegradation reaction must have exactly 2 parameters (the maximal rate and the Michaelis constant)."

        # Set the maximal rate and the Michaelis constant
        self.maximal_rate = params[0]                                                 # float or None
        self.michaelis_constant = params[1]                                           # float or None

        # Update the params list
        self.params = params

    def get_involved_species(self):
        """Return sorted unique species involved (substrate ∪ enzyme)."""
        species = list(set(self.substrate_label + self.enzyme_label))
        species.sort()
        return species
    
    def get_involved_inputs(self):
        """ Returns a list of all inputs involved in the reaction. 
        Returns:
        - input_labels: list of strings representing the labels of the involved inputs. """

        return self.input_labels
    
    def get_stoichiometry_dict(self):
        """Return stoichiometry coefficients as a dictionary (reactant only).

        Returns:
            dict[str, int]: Mapping reactant label -> stoichiometric coefficient.
        """

        stoich = {self.reactant_labels[0]: -1} 
        return stoich
    
    def set_crn_context(self, crn):
        """Precompute indices for fast propensity evaluation.

        Args:
            crn: IOCRN instance providing label->index maps.
        """

        super().set_crn_context(crn)
        self.substrate_idx = crn.species_label_to_idx(self.substrate_label) # single index or list of indices
        self.enzyme_idx = crn.species_label_to_idx(self.enzyme_label) # single index or list of indices
        self.input_idx = crn.input_label_to_idx(self.input_channels) # single index or list of indices

    def propensity(self, x, u):
        """Compute Active Degradation propensity.

        Args:
            x: Full species state vector of the parent IOCRN.
            u: Full input vector of the parent IOCRN.

        Returns:
            float: Reaction propensity.
        """
    
        # Extract relevant species and inputs
        x_enzyme = x[self.enzyme_idx] if self.enzyme_idx else np.array([]) 
        x_substrate = x[self.substrate_idx] if self.substrate_idx else np.array([])       
        u = np.array([u[i] if i is not None else 1 for i in self.input_idx])

        # Compute the propensity
        a = self.maximal_rate
        K = self.michaelis_constant
        D = x_enzyme[0] 
        S = x_substrate[0]
        return a*u[0] * D * S / (K*u[1] + S)
    
    def __str__(self):
        """Human-readable string representation."""
        
        try:
            species_str = f"{self.substrate_label} : {self.substrate_idx}, {self.enzyme_label} : {self.enzyme_idx}, {self.input_channels} : {self.input_idx}"
        except:
            species_str = "unset"
            
        reactants_str = ' + '.join(self.substrate_label) if self.substrate_label else '∅'
        products_str = '∅'
        
        # Construct parameters string
        # Enzyme
        params_str = f"E = {self.enzyme_label},    "
        # Maximal rate
        if self.input_channels[0] is None:
            params_str += f"a = {self.maximal_rate},    "
        else:
            params_str += f"a = {self.maximal_rate}{self.input_channels[0]},    "

        # Enzyme
        if self.input_channels[1] is None:
            params_str += f"K = {self.michaelis_constant}"
        else:
            params_str += f"K = {self.michaelis_constant}{self.input_channels[1]}"
    
        return f"{reactants_str} ----> {products_str};  [ActiveDeg({params_str})]"


