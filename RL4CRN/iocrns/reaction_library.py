"""
Reaction library and construction utilities.

This module defines `ReactionLibrary`, a container for reaction *templates*
used throughout RL4CRN. The library:

- stores a list of reaction objects and assigns each a unique integer `ID`,
- provides fast lookup tables mapping reaction IDs to slices in flattened
  parameter vectors,
- exposes masks useful for parameterized policies (continuous/discrete
  parameter slots and discrete-combination validity in logit space),
- offers helper constructors for common reaction families (mass-action,
  degradation, Hill-regulated production).

The library is designed to be used in:

- actuators (e.g., selecting a reaction by ID),
- observers/tensorizers (e.g., building multi-hot encodings),
- policies that need fixed-size parameter layouts.
"""

from RL4CRN.iocrns.reactions import ActiveDegradation, Reaction, MassAction, HillProduction
from itertools import combinations_with_replacement, combinations, product, accumulate
import numpy as np
from RL4CRN.utils.utils import cartesian_prod
import copy

class ReactionLibrary:
    """Container managing a list of registered reaction templates.

    Each reaction is assigned a unique integer ID at registration time.
    The library supports:

    - retrieval by ID (returns a deep copy),
    - computation of flattened parameter layouts,
    - masks describing which parameter slots exist per reaction.

    Parameters are often treated in a *flattened* layout across the entire library.
    Two related layouts are maintained:

    - a flattened vector of **all parameters** across reactions
        (size `get_num_parameters`),
    - a flattened vector of **controllable parameters** across reactions
        (size `get_num_controllable_parameters`).

    Attributes:
        reactions: List of registered reaction objects.
        last_ID: Next ID to assign (also equals `len(reactions)` after reindexing).
        parameter_lookup_table: List mapping reaction ID -> starting offset in the
            flattened *all-parameters* vector.
        controllable_parameter_lookup_table: List mapping reaction ID -> starting
            offset in the flattened *controllable-parameters* vector.
        continuous_parameter_mask: Optional 2D float mask describing continuous
            parameter slots per reaction.
        discrete_parameter_mask: Optional 2D float mask describing discrete
            parameter slots per reaction.
        logit_mask: Optional 2D boolean mask describing valid combinations of
            discrete categories in a global logit space.
        categories_per_discrete_parameter: Optional list with the number of
            categories for each discrete parameter position (global layout).
            Required by `get_logit_mask`.
    """
    
    def __init__(self, reactions=None):
        """Initialize the library and optionally register initial reactions.

        Args:
            reactions: Optional reaction or list of reactions to register.
                Each entry must be an instance of `RL4CRN.iocrns.reactions.Reaction`.
        """
        
        self.reactions = []
        self.last_ID = 0                                # ID counter for assigning unique IDs to reactions
        self.add_reactions(reactions)                   # Add initial reactions if provided
        self.prepare_lookup_tables()                    # Prepare lookup tables
        self.continuous_parameter_mask = None           # Mask for continuous parameters. 2D numpy array of shape (number of reactions, maximum number of continuous parameters across all reactions). Entries are 1 if the parameter exists for that reaction, and 0 otherwise.
        self.discrete_parameter_mask = None             # Mask for discrete parameters. 2D numpy array of shape (number of reactions, maximum number of discrete parameters across all reactions). Entries are 1 if the parameter exists for that reaction, and 0 otherwise.
        self.logit_mask = None                          # Mask for discrete parameters in logit space. 2D numpy array of shape (number of reactions, number of discrete parameter combinations across all reactions). Entries are True if the parameter combination exists for that reaction, and False otherwise.
        self.categories_per_discrete_parameter = None   # List of values representing the number of categories for each discrete parameter across all reactions. Used for constructing the logit mask.

    def get_reaction(self, ID):
        """Retrieve a reaction by its ID (returns a deep copy).

        Args:
            ID: Integer reaction ID.

        Returns:
            A deep copy of the reaction with the specified ID, or None if the ID
                is out of range.
        """
        
        if ID < len(self.reactions):
            return copy.deepcopy(self.reactions[ID]) # Return a deep copy to prevent external modifications of the library's reactions
        return None

    def add_reactions(self, reactions): #TODO: Check if reactions are already in the library before adding them
        """Register one or more reactions.

        Args:
            reactions: A single `Reaction` instance, or an iterable of
                `Reaction` instances. If None, this method is a no-op.
        """
        
        if reactions is None:
            return
        if isinstance(reactions, Reaction):
            self.register_reaction(reactions)
        else: 
            for reaction in reactions:
                self.register_reaction(reaction)

    def register_reaction(self, reaction):
        """Register a single reaction and assign it a unique ID.

        Args:
            reaction: Reaction instance to add.

        Side Effects:
            - Appends `reaction` to `reactions`.
            - Calls `reaction.set_ID(...)`.
            - Increments `last_ID`.
        """
        self.reactions.append(reaction)
        reaction.set_ID(self.last_ID)
        self.last_ID += 1

    def __len__(self):
        """Returns the number of reactions in the library."""
        return len(self.reactions)
    
    def get_num_parameters(self):
        """Returns the total number of parameters across all reactions in the library."""
        return sum(reaction.num_parameters for reaction in self.reactions)
    
    def get_num_controllable_parameters(self):
        """ Returns the total number of controllable parameters across all reactions in the library. """
        return sum(reaction.get_num_controllable_parameters() for reaction in self.reactions)
    
    def prepare_lookup_tables(self):
        """Prepare lookup tables for flattened parameter layouts.

        This method creates:

        - `parameter_lookup_table`: reaction ID -> starting offset into a
            flattened vector of *all* parameters.
        - `controllable_parameter_lookup_table`: reaction ID -> starting
            offset into a flattened vector of *controllable* parameters.

        Notes:
            Offsets are computed with cumulative sums; for reaction `j`, the slice
            of all-parameters is:

            - `start = parameter_lookup_table[j]`
            - `end   = start + reactions[j].num_parameters`

            Similarly for controllable parameters.
        """
        
        self.parameter_lookup_table = list(accumulate([reaction.num_parameters for reaction in self.reactions], initial=0))[:-1]   
        self.controllable_parameter_lookup_table = list(accumulate([reaction.get_num_controllable_parameters() for reaction in self.reactions], initial=0))[:-1]  
    
    def __str__(self):
        """Return a readable representation listing all registered reactions."""
        out = f'Number of reactions: {len(self)}\n'
        out += '\n'.join(f'R{reaction.ID}: {str(reaction)}' for reaction in self.reactions)
        return out
    
    def print_reactions(self, ID_list=None):
        """Print a subset (or all) reactions in the library.

        Args:
            ID_list: Optional iterable of reaction IDs to print. If None, prints
                all reactions.
        """
        if ID_list is None:
            ID_list = range(len(self))
        out = f'Number of reactions: {len(ID_list)}\n'
        out += '\n'.join(f'R{self.reactions[ID].ID}: {str(self.reactions[ID])}' for ID in ID_list)
        print(out)

    def get_parameter_mask(self, mode='continuous', force=False): #TODO: Test this function
        """Return a per-reaction mask for parameter-slot existence.

        The returned mask is useful when representing reaction parameters in a
        fixed-size tensor where different reactions have different numbers of
        parameters.

        Args:
            mode: Which parameter type to consider:

                - `'continuous'`: uses `reaction.num_continuous_parameters`
                  and caches to `self.continuous_parameter_mask`.
                - `'discrete'`: uses `reaction.num_discrete_parameters`
                  and caches to `self.discrete_parameter_mask`.
            force: If True, recompute even if a cached mask exists.

        Returns:
            A float32 numpy array of shape `(len(self), Pmax)`, where `Pmax` is
                the maximum number of parameters of the selected type across
                reactions. Entry `(i, j)` is 1.0 if reaction `i` has that parameter
                slot, otherwise 0.0.

                If no reactions have parameters of the specified type (`Pmax == 0`),
                returns None.

        Raises:
            ValueError: If `mode` is not `'continuous'` or `'discrete'`.
        """

        # Determine which attributes to use based on the mode
        match mode:
            case 'continuous':
                mask_attr = "continuous_parameter_mask"
                reaction_attr = "num_continuous_parameters"
            case 'discrete':
                mask_attr = "discrete_parameter_mask"
                reaction_attr = "num_discrete_parameters"
            case _:
                raise ValueError(f"Unknown mode: {mode}. Supported modes are 'continuous' and 'discrete'.")
            
        # Return the existing mask if it exists and force is not set
        if getattr(self, mask_attr) is not None and not force:   
            return getattr(self, mask_attr)  
        
        # Compute the maximum number of parameters across all reactions, return None if zero
        max_num_params = max([getattr(reaction, reaction_attr) for reaction in self.reactions], default=0)
        if max_num_params == 0:
            return None
        
        # Create the mask
        mask = np.zeros((len(self), max_num_params), dtype=np.float32) # shape (number of reactions in the library, maximum number of parameters across all reactions)
        for i, reaction in enumerate(self.reactions):
            mask[i, :getattr(reaction, reaction_attr)] = 1.

        # Record and return the mask
        setattr(self, mask_attr, mask)
        return mask
    
    def get_logit_mask(self, force=False): #TODO: Test this function
        r"""Return a mask of valid discrete-category combinations in logit space.

        Discrete parameters are modeled as categorical variables. Let the global
        discrete parameter layout have `D` positions, with
        `categories_per_discrete_parameter = [K1, ..., KD]`. Consider the full
        Cartesian grid of category assignments:


        $$\{0,\dots,K_1-1\} \times \cdots \times \{0,\dots,K_D-1\}.$$

        Not every reaction uses every discrete parameter position. This method
        constructs a boolean mask indicating which global combinations are valid
        for each reaction, by enforcing that *unused* discrete positions take a
        default category (here: category 0) for that reaction.

        Args:
            force: If True, recompute even if `self.logit_mask` is already set.

        Returns:
            Boolean numpy array of shape `(len(self), Ncomb)` where `Ncomb` is the
            total number of combinations in the Cartesian grid. Entry `(r, c)` is
            True if combination `c` is valid for reaction `r`.

            Returns None if the library has no discrete parameters
            (`get_parameter_mask(mode='discrete')` returns None).

        Raises:
            ValueError: If `categories_per_discrete_parameter` is not set but
                discrete parameters exist.

        Notes:
            This method assumes that unused discrete slots must be fixed to
            category 0. If your semantics differ (e.g., "don't-care" instead of
            fixed), you should modify the mask logic accordingly.
        """

        # Determine the number of categories for each discrete parameter across all reactions
        dimensions = self.categories_per_discrete_parameter # List of values representing the number of categories for each discrete parameter across all reactions

        # Get the discrete parameter mask to identify which reactions have discrete parameters and how many
        discrete_parameter_mask = self.get_parameter_mask(mode='discrete') # shape: (number of reactions in the library, maximum number of discrete parameters across all reactions)
        
        # Return the existing logit mask if it exists and force is not set
        if self.logit_mask is not None and not force:
            return self.logit_mask

        # If there are no discrete parameters, return None
        if discrete_parameter_mask is None:
            return None
        
        # Construct a grid of all possible combinations of discrete parameter categories
        grid = cartesian_prod([np.arange(d) for d in dimensions]) # shape: (total number of discrete parameter combinations across all reactions, total number of discrete parameters across all reactions)
        
        # Create the logit mask #TODO: Something seems wrong here
        logit_mask = np.ones((len(self), grid.shape[0]), dtype=bool) # shape: (number of reactions in the library, total number of discrete parameter combinations across all reactions)
        for j in range(len(self)):
            for i in range(len(discrete_parameter_mask[j])):
                if discrete_parameter_mask[j,i] == 0:
                    logit_mask[j] = logit_mask[j] & (grid[:,i] == 0)

        # Record and return the logit mask
        self.logit_mask = logit_mask
        return logit_mask
    
    def clone(self):
        """Return a deep copy of the reaction library."""
        return copy.deepcopy(self)
    
    def merge(self, other_library): # TODO: merging of non mutually exclusive libraries still to be implemented
        """Merge another reaction library into this one.

        Args:
            other_library: Another `ReactionLibrary` whose reactions are
                registered into this library.

        Notes:
            Merging non-mutually-exclusive libraries may require deduplication.
            This method currently appends all reactions and re-prepares lookup
            tables.
        """
        self.add_reactions(other_library.reactions)
        self.prepare_lookup_tables()

    def find_ID(self, reactions):
        """
        Find reaction IDs by matching reaction instances.

        Args:
            reactions: A single `Reaction` instance or a list of `Reaction`
                instances to match by equality (`==`) against library entries.

        Returns:
            A list of integer reaction IDs corresponding to the provided reaction
                instances.  If a reaction instance is not found in the library,
                `None` is returned for that instance.
        """
        
        if isinstance(reactions, Reaction):
            reactions = [reactions]
        
        ids = []
        for reaction in reactions:
            found_id = None
            for r in self.reactions:
                if r == reaction:
                    found_id = r.ID
                    break
            ids.append(found_id)
        return ids

    def remove_reactions(self, reactions, remove_by='ID'):
        """Remove reactions from the library.

        Args:
            reactions: Depending on `remove_by`:

                - `'ID'`: list/iterable of integer reaction IDs.
                - `'instance'`: reaction instance or list of instances to match
                  by equality (`==`) against library entries.
                - `'reactant'`: not implemented.
                - `'product'`: not implemented.
            remove_by: Removal mode. Supported:
                `'ID'`, `'instance'`.

        Side Effects:
            - Removes matching reactions from `reactions`.
            - Reassigns reaction IDs to be contiguous starting from 0.
            - Updates `last_ID`.
            - Recomputes lookup tables.

        Raises:
            NotImplementedError: For `'reactant'` and `'product'`.
            ValueError: If `remove_by` is not a supported option.
        """
        
        # Normalize input to a list of IDs, Reaction instances, or list of reactant/product labels
        if isinstance(reactions, Reaction):
            reactions = [reactions]
        
        # Determine the set of IDs to remove based on the specified method
        match remove_by:
            case 'ID':
                ids_to_remove = set(reactions)
            case 'instance':
                ids_to_remove = set(self.find_ID(reactions))
            case 'reactant':
                raise NotImplementedError("Removal by reactant labels is not implemented yet.")
            case 'product':
                raise NotImplementedError("Removal by product labels is not implemented yet.")
            case _:
                raise ValueError(f"Unknown remove_by option: {remove_by}. Supported options are 'ID', 'instance', 'reactant', 'product'.")

        # Remove reactions with the specified IDs   
        self.reactions = [reaction for reaction in self.reactions if reaction.ID not in ids_to_remove]

        # Reset IDs
        for new_id, reaction in enumerate(self.reactions):
            reaction.set_ID(new_id)
        self.last_ID = len(self.reactions)

        # Prepare lookup tables for efficient parameter access
        self.prepare_lookup_tables()

    def find_zero_reaction(self):
        r"""Find the ID of the zero reaction (∅ → ∅) if it exists.

        Returns:
            The integer ID of the zero reaction if found, otherwise None.
        """
        for reaction in self.reactions:
            if reaction.reactant_labels == [] and reaction.product_labels == []:
                return reaction.ID
        return None
    

def construct_mass_action_library(species_labels, order = 2, order_reactants=None, order_products=None):
    r"""Construct a library of mass-action reactions over a species set.

    Generates all reactions of the form:

    $$\text{reactant complex} \rightarrow \text{product complex}$$

    where reactant complexes have size up to `order_reactants` and product
    complexes have size up to `order_products` (multiset combinations with
    replacement). The pair `([] , [])` (∅ → ∅) is included explicitly.

    Args:
        species_labels: List of species labels usable in complexes.
        order: If `order_reactants` and `order_products` are not provided, both
            are set to this value.
        order_reactants: Maximum stoichiometric order of reactant complexes.
        order_products: Maximum stoichiometric order of product complexes.

    Returns:
        A `ReactionLibrary` populated with `MassAction` reactions.

    Raises:
        AssertionError: If neither `order` nor both `order_reactants` and
            `order_products` are provided.

    Notes:
        The number of distinct complexes of size up to `O` from `n` species is:

        $$\sum_{k=0}^{O} \binom{n + k - 1}{k} = \binom{n + O}{O}.$$

        This function generates all reactant/product pairs except identical
        complexes, plus the explicit empty-to-empty reaction.
    """

    assert order_reactants is not None and order_products is not None or order is not None, "Either order_reactants and order_products or order must be specified."
    
    if order_reactants is None and order_products is None:
        order_reactants = order
        order_products = order

    # Generate all possible complexes up to the given order
    complex_list_reactants = []
    for i in range(order_reactants+1):
        complex_list_reactants += [list(complex) for complex in combinations_with_replacement(species_labels, i)]

    complex_list_products = []
    for i in range(order_products+1):
        complex_list_products += [list(complex) for complex in combinations_with_replacement(species_labels, i)]

    # Combine complexes to form reactions
    reaction_list = []
    for reactants, products in product(complex_list_reactants, complex_list_products):
        if reactants != products:
            reaction_list.append((reactants, products))

    # Add the empty complex as a reactant or product
    reaction_list = [([], [])] + reaction_list  # ∅ → ∅

    # Create a reaction library
    reaction_library = ReactionLibrary()
    for reactants, products in reaction_list:
        reaction = MassAction(reactant_labels=reactants, product_labels=products, input_channels=[None], params=[None], params_controllability=[True])
        reaction_library.add_reactions(reaction)

    # Prepare lookup tables for efficient parameter access
    reaction_library.prepare_lookup_tables()

    return reaction_library

def construct_first_order_degradation_library(species_labels): 
    r"""Construct a library of first-order degradation reactions.

    For each species `X` in `species_labels`, constructs a mass-action reaction:

    $$X \rightarrow \emptyset.$$

    Args:
        species_labels: List of species labels.

    Returns:
        A `ReactionLibrary` containing one degradation reaction per species.
    """

    # Create a reaction library
    reaction_library = ReactionLibrary()
    for reactants in species_labels:
        reaction = MassAction(reactant_labels=[reactants], product_labels=[], input_channels=[None], params=[None], params_controllability=[True])
        reaction_library.add_reactions(reaction)

    # Prepare lookup tables for efficient parameter access
    reaction_library.prepare_lookup_tables()

    return reaction_library

def construct_hill_production_library(species_labels, max_product_order=2, max_num_regulators=2):
    r"""Construct a library of Hill-regulated production reactions.

    Generates reactions of the form:

    $$\emptyset \rightarrow \text{product complex},$$

    regulated by a set of activators and repressors (disjoint), with total number
    of regulators between 1 and `max_num_regulators`.

    For each product complex (multiset of size 1..`max_product_order`) and each
    split into `A` activators and `R` repressors with `A+R = t`, `t` ranging over
    `1..max_num_regulators`, this function constructs one `HillProduction`
    reaction per distinct choice of activator and repressor sets.

    Args:
        species_labels: List of species labels usable as products or regulators.
        max_product_order: Maximum stoichiometric order of the product complex (≥ 1).
        max_num_regulators: Maximum total number of regulators
            (`#activators + #repressors`).

    Returns:
        A `ReactionLibrary` populated with `HillProduction` reactions.

    Notes:
        This constructor uses a parameter layout consistent with the current
        implementation:

        - parameter vector length: `2 + len(activators) + len(repressors)`
        - parameters are initialized to `None`
        - all parameters are marked controllable (`params_controllability=True`)

        If you switch to a richer Hill parameterization (e.g., separate K and n
        per regulator), update both the constructor and this docstring.

        A rough count of generated reactions is:

        $$(\binom{n + P}{P} - 1) \sum_{t=1}^{\min(R, n)} 2^t \binom{n}{t},$$

        where `n = len(species_labels)`, `P = max_product_order`,
        `R = max_num_regulators`.
    """

    # Build all product complexes up to the requested order (exclude empty)
    product_complexes = []
    for i in range(1, max_product_order + 1):
        product_complexes += [list(c) for c in combinations_with_replacement(species_labels, i)]

    reaction_library = ReactionLibrary()

    # Loop over each product complex
    for products in product_complexes:
        # Loop over all possible splits of total regulators into activators/repressors
        for total_regs in range(1, max_num_regulators + 1):
            for num_activators in range(0, total_regs + 1):
                num_repressors = total_regs - num_activators
                # Choose activators
                if num_activators > 0:
                    activator_sets = [list(c) for c in combinations(species_labels, num_activators)]
                else:
                    activator_sets = [[]]
                # For each activator combination, choose repressors from remaining species
                for activators in activator_sets:
                    remaining_species = [s for s in species_labels if s not in activators]
                    if num_repressors > 0:
                        repressor_sets = [list(c) for c in combinations(remaining_species, num_repressors)]
                    else:
                        repressor_sets = [[]]
                    # Build HillProduction for all valid (activator, repressor) combinations
                    for repressors in repressor_sets:
                        # Compute parameter vector size:
                        # [b, Vmax, (Ka,na per activator), (Kr,nr per repressor)]
                        num_params = 2 + len(activators) + len(repressors)
                        params = [None] * num_params
                        input_channels = [None] * num_params
                        params_controllability = [True] * num_params
                        reaction = HillProduction(product_labels=products, activator_labels=activators, repressor_labels=repressors,input_channels=input_channels, params=params, params_controllability=params_controllability)
                        reaction_library.add_reactions(reaction)

    # Prepare lookup tables for efficient parameter access
    reaction_library.prepare_lookup_tables()

    return reaction_library

def construct_active_degradation_library(species_labels):
    r"""Construct a library of active degradation reactions.

    Generates reactions of the form:

    $$\text{species} \rightarrow \emptyset,$$

    regulated by a degradation enzyme.

    For each reactant species (substrate), this function constructs one `ActiveDegradation`
    reaction per distinct choice of degradation enzyme.

    Args:
        species_labels: List of species labels usable as substrates or degradation enzymes.

    Returns:
        A `ReactionLibrary` populated with `ActiveDegradation` reactions.

    Notes:
        This constructor uses a parameter layout consistent with the current
        implementation:

        - parameter vector length: `2`
        - parameters are initialized to `None`
        - all parameters are marked controllable (`params_controllability=True`)

        A rough count of generated reactions is:

        $$n \cdot (n-1),$$

        where `n = len(species_labels)`.
    """

    reaction_library = ReactionLibrary()

    # Loop over each species as a substrate
    for substrate in species_labels:
        # Loop over each species as a potential degradation enzyme
        for enzyme in species_labels:
            if enzyme != substrate:
                params = [None, None]
                input_channels = [None, None]
                params_controllability = [True, True]
                reaction = ActiveDegradation([substrate], [enzyme], input_channels, params, params_controllability)
                reaction_library.add_reactions(reaction)

    # Prepare lookup tables for efficient parameter access
    reaction_library.prepare_lookup_tables()

    return reaction_library

