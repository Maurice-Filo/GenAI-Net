import numpy as np

class Reaction:
    # Note: whenever adding new input components for the constructor (e.g. catalysts), ensure they are sorted as well
    def __init__(self, reactant_labels, product_labels, input_channels=[None], params=[None], params_controllability=[False], ID = None, signature=None):
        """ Initializes a reaction.
        Arguments:
        - reactant_labels: list of strings representing the labels of the reactants, can be empty
        - product_labels: list of strings representing the labels of the products, can be empty
        - input_channels: list of strings representing the channels of the inputs, cannot be empty, can contain None. None entries indicate that the corresponding parameter does not depend on any input
        - params: list of floats representing the parameters of the reaction, cannot be empty, can contain None. If None, it indicates that the parameter is unknown and needs to be set later 
        - params_controllability: list of booleans representing whether each parameter is controllable by an input, cannot be empty
        - ID: integer representing the unique identifier of the reaction within the context of a reaction library, can be None
        - signature: string representing the unique signature of the reaction, can be None
        The lengths of input_channels, params, and params_controllability must be the same. 
        Each parameter in params corresponds to the parameter controllability in params_controllability and the input channel in input_channels at the same index. 
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

    def set_unknown_parameters(self, params, initial_idx=0): #TODO: test this method
        i = initial_idx
        for j in range(self.num_parameters):
            if self.params[j] is None:
                self.params[j] = params[i]
                i += 1

    def get_ID(self):
        return self.ID
    
    def set_ID(self, ID):
        self.ID = ID

    def get_num_controllable_parameters(self):
        return sum(self.params_controllability)
    
    def get_num_unknown_params(self):
        return sum(1 for param in self.params if param is None)
        
    def propensity(self, x, u):
        pass

    def __call__(self, x, u):
        return self.propensity(x, u)
    
    def set_crn_context(self, crn):
        self.crn = crn

    def __eq__(self, other):
        """ Checks if two reactions are equivalent based on their signatures. 
        Arguments:
        - other: another Reaction instance to compare with.
        Returns:
        - True if the reactions have the same signature, False otherwise.
        """
        return self.signature == other.signature

    def set_library_context(self, reaction_library):
        """ Sets the context of the reaction within a given reaction library by assigning its ID if it exists in the library based on its signature. """
        for reaction in reaction_library.reactions:
            if self == reaction:
                self.ID = reaction.ID
                return

class MassAction(Reaction):
    def __init__(self, reactant_labels, product_labels, input_channels=[None], params=[None], params_controllability=[True]):
        """ Initializes a mass action reaction.
        Arguments:
        - reactant_labels: list of strings representing the labels of the reactants, can be empty
        - product_labels: list of strings representing the labels of the products, can be empty
        - input_channels: list of one string representing the channel of the input, cannot be empty, can contain None. If None, it indicates that the reaction does not depend on any input
        - params: list of one float representing the rate constant of the reaction, cannot be empty, can contain None. If None, it indicates that the rate constant is unknown and needs to be set later
        - params_controllability: list of one boolean representing whether the rate constant is controllable by an input, cannot be empty
        The lengths of input_channels, params, and params_controllability must be one. 
        """
        # Call the parent constructor
        super().__init__(reactant_labels, product_labels, input_channels, params, params_controllability)
        
        # Ensure params has exactly one element (the rate constant)
        assert len(params) == 1, "MassAction reaction must have exactly one parameter (the rate constant)."

        # Create the reaction signature: depends on the reaction structure only, not on the parameters or inputs
        self.signature = str(('MAK', reactant_labels, product_labels))

        # Record the rate constant
        self.rate_constant = params[0]                              # float or None
        self.num_continuous_parameters = 1                          # Mass action has one continuous parameter (the rate constant)
        self.num_discrete_parameters = 0                            # Mass action has no discrete parameters
        self.num_unknown_params = self.get_num_unknown_params()     # Number of unknown parameters (0 or 1)

    def set_parameters(self, params):
        """ Sets the parameters of the mass action reaction.
        Arguments:
        - params: list of one float representing the rate constant of the reaction. 
        """
        # Ensure params has exactly one element (the rate constant)
        assert len(params) == 1, "MassAction reaction must have exactly one parameter (the rate constant)."

        # Set the rate constant
        self.rate_constant = params[0]

        # Update the params list
        self.params = params
 
    def get_involved_species(self):
        """ Returns a list of all species involved in the reaction (reactants and products). 
        Returns:
        - species: list of strings representing the labels of the involved species. 
        """
        species = list(set(self.reactant_labels + self.product_labels))
        species.sort()
        return species
    
    def get_involved_inputs(self):
        """ Returns a list of all inputs involved in the reaction. 
        Returns:
        - input_labels: list of strings representing the labels of the involved inputs.
        """
        return self.input_labels 
    
    def get_stoichiometry_dict(self):
        """ Constructs and returns the stoichiometry dictionary for the reaction.
        Returns:
        - stoich: A dictionary mapping labels of species involved in the reaction to their stoichiometric coefficients. 
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
        """ Sets the context of the reaction within a given IOCRN by mapping species and input labels to their respective indices.
        Arguments:
        - crn: An instance of IOCRN containing species and input labels. 
        """
        super().set_crn_context(crn)
        self.reactant_idx = crn.species_label_to_idx(self.reactant_labels) # single index or list of indices
        self.product_idx = crn.species_label_to_idx(self.product_labels) # single index or list of indices
        self.input_idx = crn.input_label_to_idx(self.input_channels) # single index or list of indices

    def propensity(self, x, u):
        """ Computes the propensity of the reaction given species counts x and inputs u.
        Arguments:
        - x: numpy array of shape (num_species,) representing the concentrations of the species of the whole IOCRN.
        - u: numpy array of shape (num_inputs,) representing the inputs to the whole IOCRN.
        Returns:
        - propensity: float representing the propensity of the reaction. 
        """
        # Extract relevant species and inputs
        x = x[self.reactant_idx]
        u = u[self.input_idx[0]] if self.input_idx[0] is not None else 1.0

        # Compute the propensity using mass action kinetics
        return self.rate_constant * np.prod(x) * u
    
    def __str__(self): #TODO: remove indices from the string representation
        """ Returns a string representation of the reaction in the format:
        Reactants ----> Products;  [MAK(rate_constant, input)]
        If there are no reactants, it uses '∅' to denote the empty set. If there are no inputs, it omits the input part. 
        Returns:
        - reaction_str: string representing the reaction. 
        """
        try:
            species_str = f"{self.reactant_labels} : {self.reactant_idx}, {self.product_labels} : {self.product_idx}, {self.input_channels} : {self.input_idx}"
        except:
            species_str = "unset"
            
        reactants_str = ' + '.join(self.reactant_labels) if self.reactant_labels else '∅'
        products_str = ' + '.join(self.product_labels) if self.product_labels else '∅'
        inputs_str = self.input_channels[0] if self.input_channels[0] is not None else ''
        if inputs_str == '':
            return f"{reactants_str} ----> {products_str};  [MAK({self.rate_constant})]" #\n {species_str}"
        return f"{reactants_str} ----> {products_str};  [MAK({self.rate_constant}, {inputs_str})]" # \n {species_str}"