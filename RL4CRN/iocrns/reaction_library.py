from RL4CRN.iocrns.reactions import Reaction, MassAction
from itertools import combinations_with_replacement, product, accumulate
import numpy as np
from RL4CRN.utils.utils import cartesian_prod
import copy

class ReactionLibrary:
    """ A library to manage a list of reactions.
    Each reaction is assigned a unique ID upon registration. """
    def __init__(self, reactions=None):
        """ Initializes the reaction library.
        Arguments:
        - reactions: a Reaction instance or a list of Reaction instances to initialize the library with.
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
        """ Retrieves a reaction from the library by its ID.
        Arguments:
        - ID: the unique identifier of the reaction.
        Returns:
        - The Reaction instance with the specified ID, or None if not found.
        """
        if ID < len(self.reactions):
            return copy.deepcopy(self.reactions[ID])
        return None

    def add_reactions(self, reactions): #TODO: Check if reactions are already in the library before adding them
        """ Adds one or more reactions to the library and register them by assigning them a unique ID.
        Arguments:
        - reactions: a Reaction instance or a list of Reaction instances.
        """
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
        - reaction: an instance of Reaction.
        """
        self.reactions.append(reaction)
        reaction.set_ID(self.last_ID)
        self.last_ID += 1

    def __len__(self):
        """ Returns the number of reactions in the library.
        """
        return len(self.reactions)
    
    def get_num_parameters(self):
        """ Returns the total number of parameters across all reactions in the library.
        """
        return sum(reaction.num_parameters for reaction in self.reactions)
    
    def get_num_controllable_parameters(self):
        """ Returns the total number of controllable parameters across all reactions in the library.
        """
        return sum(reaction.get_num_controllable_parameters() for reaction in self.reactions)
    
    def prepare_lookup_tables(self):
        """ Prepares lookup tables for efficient access to reaction parameters.
        Keeps track of the starting index of parameters for each reaction in a flat parameter array.
        Creates two lookup tables:
        - parameter_lookup_table: maps reaction IDs to the starting index of their parameters in a flat parameter array.
        - controllable_parameter_lookup_table: maps reaction IDs to the starting index of their controllable parameters in a flat controllable parameter array.
        """
        self.parameter_lookup_table = list(accumulate([reaction.num_parameters for reaction in self.reactions], initial=0))[:-1]   
        self.controllable_parameter_lookup_table = list(accumulate([reaction.get_num_controllable_parameters() for reaction in self.reactions], initial=0))[:-1]  
    
    def __str__(self):
        """ Returns a string representation of the reaction library. 
        """
        out = f'Number of reactions: {len(self)}\n'
        out += '\n'.join(f'R{reaction.ID}: {str(reaction)}' for reaction in self.reactions)
        return out
    
    def print_reactions(self, ID_list=None):
        """ Prints the reactions in the library having IDs in ID_list.
        If ID_list is None, prints all reactions. 
        """
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
     
def construct_mass_action_library(species_labels, order=2):
    """ Constructs a reaction library containing all possible mass-action reactions
    up to a given order for a set of species.
    Arguments:
    - species_labels: list of strings representing the species labels.
    - order: maximum order of the reactions.
    Returns:
    - reaction_library: an instance of ReactionLibrary containing the generated reactions. """

    # Generate all possible complexes up to the given order
    complex_list = []
    for i in range(order+1):
        complex_list += [list(complex) for complex in combinations_with_replacement(species_labels, i)]

    # Combine complexes to form reactions
    reaction_list = []
    for reactants, products in product(complex_list, repeat=2):
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