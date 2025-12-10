from RL4CRN.iocrns.reactions import Reaction, MassAction, HillProduction
from itertools import combinations_with_replacement, combinations, product, accumulate
import numpy as np
from RL4CRN.utils.utils import cartesian_prod
import copy

class ReactionLibrary:
    """ A library to manage a list of reactions.
    Each reaction is assigned a unique ID upon registration. """
    
    def __init__(self, reactions=None):
        """ Initializes the reaction library.
        Arguments:
        - reactions: a Reaction instance or a list of Reaction instances to initialize the library with. """
        
        self.reactions = []
        self.last_ID = 0                                # ID counter for assigning unique IDs to reactions
        self.add_reactions(reactions)                   # Add initial reactions if provided
        self.prepare_lookup_tables()                    # Prepare lookup tables
        self.continuous_parameter_mask = None           # Mask for continuous parameters. 2D numpy array of shape (number of reactions, maximum number of continuous parameters across all reactions). Entries are 1 if the parameter exists for that reaction, and 0 otherwise.
        self.discrete_parameter_mask = None             # Mask for discrete parameters. 2D numpy array of shape (number of reactions, maximum number of discrete parameters across all reactions). Entries are 1 if the parameter exists for that reaction, and 0 otherwise.
        self.logit_mask = None                          # Mask for discrete parameters in logit space. 2D numpy array of shape (number of reactions, number of discrete parameter combinations across all reactions). Entries are True if the parameter combination exists for that reaction, and False otherwise.
        self.categories_per_discrete_parameter = None   # List of values representing the number of categories for each discrete parameter across all reactions. Used for constructing the logit mask.

    def get_reaction(self, ID):
        """ Retrieves a reaction from the library by its ID.
        Arguments:
        - ID: the unique identifier of the reaction.
        Returns:
        - The Reaction instance with the specified ID, or None if not found. (a copy) """
        
        if ID < len(self.reactions):
            return copy.deepcopy(self.reactions[ID]) # Return a deep copy to prevent external modifications of the library's reactions
        return None

    def add_reactions(self, reactions): #TODO: Check if reactions are already in the library before adding them
        """ Adds one or more reactions to the library and register them by assigning them a unique ID.
        Arguments:
        - reactions: a Reaction instance or a list of Reaction instances. """
        
        if reactions is None:
            return
        if isinstance(reactions, Reaction):
            self.register_reaction(reactions)
        else: 
            for reaction in reactions:
                self.register_reaction(reaction)

    def register_reaction(self, reaction):
        """ Registers a reaction in the library, assigning it a unique ID.
        Arguments:
        - reaction: an instance of Reaction. """
        self.reactions.append(reaction)
        reaction.set_ID(self.last_ID)
        self.last_ID += 1

    def __len__(self):
        """ Returns the number of reactions in the library. """
        
        return len(self.reactions)
    
    def get_num_parameters(self):
        """ Returns the total number of parameters across all reactions in the library. """
        
        return sum(reaction.num_parameters for reaction in self.reactions)
    
    def get_num_controllable_parameters(self):
        """ Returns the total number of controllable parameters across all reactions in the library. """
        
        return sum(reaction.get_num_controllable_parameters() for reaction in self.reactions)
    
    def prepare_lookup_tables(self):
        """ Prepares lookup tables for efficient access to reaction parameters.
        Keeps track of the starting index of parameters for each reaction in a flat parameter array.
        Creates two lookup tables:
        - parameter_lookup_table: maps reaction IDs to the starting index of their parameters in a flat parameter array.
        - controllable_parameter_lookup_table: maps reaction IDs to the starting index of their controllable parameters in a flat controllable parameter array. """
        
        self.parameter_lookup_table = list(accumulate([reaction.num_parameters for reaction in self.reactions], initial=0))[:-1]   
        self.controllable_parameter_lookup_table = list(accumulate([reaction.get_num_controllable_parameters() for reaction in self.reactions], initial=0))[:-1]  
    
    def __str__(self):
        """ Returns a string representation of the reaction library. """
        
        out = f'Number of reactions: {len(self)}\n'
        out += '\n'.join(f'R{reaction.ID}: {str(reaction)}' for reaction in self.reactions)
        return out
    
    def print_reactions(self, ID_list=None):
        """ Prints the reactions in the library having IDs in ID_list.
        If ID_list is None, prints all reactions. """
        
        if ID_list is None:
            ID_list = range(len(self))
        out = f'Number of reactions: {len(ID_list)}\n'
        out += '\n'.join(f'R{self.reactions[ID].ID}: {str(self.reactions[ID])}' for ID in ID_list)
        print(out)

    def get_parameter_mask(self, mode='continuous', force=False): #TODO: Test this function
        """ Creates a mask indicating the number of parameters (continuous or discrete) for each reaction.
        Arguments:
        - mode: 'continuous' or 'discrete', indicating which type of parameters to consider
        - force: if True, forces recomputation of the mask even if it already exists.
        Returns:
        - mask: a 2D numpy array where each row corresponds to a reaction and each column corresponds
          to a parameter. The entries are 1 if the parameter exists for that reaction, and 0 otherwise.
        If no reactions have parameters of the specified type, returns None. """

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
        """ Creates a mask for discrete parameters in logit space.
        Each discrete parameter can take on a certain number of categories.
        The logit mask indicates which combinations of discrete parameter categories are valid for each reaction.
        Arguments:
        - force: if True, forces recomputation of the logit mask even if it already exists.
        Returns:
        - logit_mask: a 2D numpy array where each row corresponds to a reaction and each column corresponds
          to a combination of discrete parameter categories. The entries are True if the combination exists for that reaction, and False otherwise.
        If no reactions have discrete parameters, returns None. """ 

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
        """ Creates a deep copy of the reaction library.
        Returns:
        - A new instance of ReactionLibrary that is a deep copy of this library. """

        return copy.deepcopy(self)
    
    def merge(self, other_library): # TODO: merging of non mutually exclusive libraries still to be implemented
        """ Merges another reaction library into this one.
        Arguments:
        - other_library: an instance of ReactionLibrary to merge into this library. """

        self.add_reactions(other_library.reactions)
        self.prepare_lookup_tables()
    
     
# def construct_mass_action_library(species_labels, order=2):
#     """ Constructs a reaction library containing all possible mass-action reactions
#     up to a given order for a set of species.
#     Arguments:
#     - species_labels: list of strings representing the species labels.
#     - order: maximum order of the reactions.
#     Returns:
#     - reaction_library: an instance of ReactionLibrary containing the generated reactions. 
#     The total number of reactions generated is equal to:
#     C (n+O, O) * (C (n+O, O) - 1) + 1,
#     where n is the number of species and O is the order. """

#     # Generate all possible complexes up to the given order
#     complex_list = []
#     for i in range(order+1):
#         complex_list += [list(complex) for complex in combinations_with_replacement(species_labels, i)]

#     # Combine complexes to form reactions
#     reaction_list = []
#     for reactants, products in product(complex_list, repeat=2):
#         if reactants != products:
#             reaction_list.append((reactants, products))

#     # Add the empty complex as a reactant or product
#     reaction_list = [([], [])] + reaction_list  # ∅ → ∅

#     # Create a reaction library
#     reaction_library = ReactionLibrary()
#     for reactants, products in reaction_list:
#         reaction = MassAction(reactant_labels=reactants, product_labels=products, input_channels=[None], params=[None], params_controllability=[True])
#         reaction_library.add_reactions(reaction)

#     # Prepare lookup tables for efficient parameter access
#     reaction_library.prepare_lookup_tables()

#     return reaction_library

def construct_mass_action_library(species_labels, order = 2, order_reactants=None, order_products=None):
    """ Constructs a reaction library containing all possible mass-action reactions
    up to a given order for a set of species.
    Arguments:
    - species_labels: list of strings representing the species labels.
    - order: maximum order of the reactions.
    - order_reactants: maximum order of the reactant complexes.
    - order_products: maximum order of the product complexes.
    Returns:
    - reaction_library: an instance of ReactionLibrary containing the generated reactions. 
    The total number of reactions generated is equal to:
    C (n+O, O) * (C (n+O, O) - 1) + 1,
    where n is the number of species and O is the order. """

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
    """ Constructs a reaction library containing first-order degradation reactions
    for a set of species.
    Arguments:
    - species_labels: list of strings representing the species labels.
    Returns:
    - reaction_library: an instance of ReactionLibrary containing the generated reactions. 
    The total number of reactions generated is equal to the number of species. """

    # Create a reaction library
    reaction_library = ReactionLibrary()
    for reactants in species_labels:
        reaction = MassAction(reactant_labels=[reactants], product_labels=[], input_channels=[None], params=[None], params_controllability=[True])
        reaction_library.add_reactions(reaction)

    # Prepare lookup tables for efficient parameter access
    reaction_library.prepare_lookup_tables()

    return reaction_library

def construct_hill_production_library(species_labels, max_product_order=2, max_num_regulators=2):
    """ Constructs a library of Hill-production reactions:
        ∅ -> (product complex)
    regulated by up to `max_num_regulators` total regulators
    (activators + repressors combined, disjoint sets, at least one regulator).
    Args:
    - species_labels (list[str]): Species usable as products or regulators.
    - max_product_order (int): Maximum stoichiometric order of the product complex (≥1).
    - max_num_regulators (int): Maximum total number of regulators (activators + repressors).
    Returns:
    - ReactionLibrary 
    The total number of reactions generated is:
    (C(n + P, P) - 1) * sum_{t=1..min(R,n)} [ 2^t * C(n, t) ], 
    where n = number of species, P = max_product_order, R = max_num_regulators. """

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







# def construct_hill_production_library(species_labels, max_product_order=2, max_num_regulators=2):
#     """ Constructs a library of Hill-production reactions:
#         ∅ -> (product complex)
#     regulated by up to `max_num_regulators` total regulators
#     (activators + repressors combined, disjoint sets, at least one regulator).
#     Args:
#     - species_labels (list[str]): Species usable as products or regulators.
#     - max_product_order (int): Maximum stoichiometric order of the product complex (≥1).
#     - max_num_regulators (int): Maximum total number of regulators (activators + repressors).
#     Returns:
#     - ReactionLibrary 
#     The total number of reactions generated is:
#     (C(n + P, P) - 1) * sum_{t=1..min(R,n)} [ 2^t * C(n, t) ], 
#     where n = number of species, P = max_product_order, R = max_num_regulators. """

#     # Build all product complexes up to the requested order (exclude empty)
#     product_complexes = []
#     for i in range(1, max_product_order + 1):
#         product_complexes += [list(c) for c in combinations_with_replacement(species_labels, i)]

#     reaction_library = ReactionLibrary()

#     # Loop over each product complex
#     for products in product_complexes:
#         # Loop over all possible splits of total regulators into activators/repressors
#         for total_regs in range(1, max_num_regulators + 1):
#             for num_activators in range(0, total_regs + 1):
#                 num_repressors = total_regs - num_activators
#                 # Choose activators
#                 if num_activators > 0:
#                     activator_sets = [list(c) for c in combinations(species_labels, num_activators)]
#                 else:
#                     activator_sets = [[]]
#                 # For each activator combination, choose repressors from remaining species
#                 for activators in activator_sets:
#                     remaining_species = [s for s in species_labels if s not in activators]
#                     if num_repressors > 0:
#                         repressor_sets = [list(c) for c in combinations(remaining_species, num_repressors)]
#                     else:
#                         repressor_sets = [[]]
#                     # Build HillProduction for all valid (activator, repressor) combinations
#                     for repressors in repressor_sets:
#                         # Compute parameter vector size:
#                         # [b, Vmax, (Ka,na per activator), (Kr,nr per repressor)]
#                         num_params = 2 + 2*len(activators) + 2*len(repressors)
#                         params = [None] * num_params
#                         input_channels = [None] * num_params
#                         params_controllability = [True] * num_params
#                         params_controllability[3::2] = [False] * (len(activators) + len(repressors)) 
#                         params[3::2] = [1.0] * (len(activators) + len(repressors))  # Set Hill coefficients to 1.0 #TODO: Make this customizable 
#                         reaction = HillProduction(product_labels=products, activator_labels=activators, repressor_labels=repressors,input_channels=input_channels, params=params, params_controllability=params_controllability)
#                         reaction_library.add_reactions(reaction)

#     # Prepare lookup tables for efficient parameter access
#     reaction_library.prepare_lookup_tables()

#     return reaction_library