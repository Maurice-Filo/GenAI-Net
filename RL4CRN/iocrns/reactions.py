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
        - True if the reactions have the same signature, False otherwise. """
        
        return self.signature == other.signature

    def set_library_context(self, reaction_library):
        """ Sets the context of the reaction within a given reaction library by assigning its ID if it exists in the library based on its signature. """
        
        for reaction in reaction_library.reactions:
            if self == reaction:
                self.ID = reaction.ID
                return
        raise ValueError("Reaction not found in the provided reaction library.")

class MassAction(Reaction):
    def __init__(self, reactant_labels, product_labels, input_channels=[None], params=[None], params_controllability=[True]):
        """ Initializes a mass action reaction.
        Arguments:
        - reactant_labels: list of strings representing the labels of the reactants, can be empty. If empty, it indicates a zeroth-order reaction (e.g., ∅ -> A).
        - product_labels: list of strings representing the labels of the products, can be empty. If empty, it indicates a degradation reaction (e.g., A -> ∅).
        - input_channels: list of one string representing the channel of the input, cannot be empty, can contain None. If None, it indicates that the reaction does not depend on any input.
        - params: list of one float representing the rate constant of the reaction, cannot be empty, can contain None. If None, it indicates that the rate constant is unknown and needs to be set later.
        - params_controllability: list of one boolean representing whether the rate constant is controllable by an input, cannot be empty.
        The lengths of input_channels, params, and params_controllability must be one. """
        
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
        """ Sets the parameters of the mass action reaction.
        Arguments:
        - params: list of one float representing the rate constant of the reaction. """
        
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
        - input_labels: list of strings representing the labels of the involved inputs. """
        return self.input_labels 
    
    def get_stoichiometry_dict(self):
        """ Constructs and returns the stoichiometry dictionary for the reaction.
        Returns:
        - stoich: A dictionary mapping labels of species involved in the reaction to their stoichiometric coefficients. """
        
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
        - crn: An instance of IOCRN containing species and input labels. """
        
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
        - propensity: float representing the propensity of the reaction. """
        
        # Extract relevant species and inputs
        x = x[self.reactant_idx]
        u = u[self.input_idx[0]] if self.input_idx[0] is not None else 1.0

        # Compute the propensity using mass action kinetics
        return self.rate_constant * np.prod(x) * u
    
    def __str__(self):
        """ Returns a string representation of the reaction in the format:
        Reactants ----> Products;  [MAK(rate_constant, input)]
        If there are no reactants, it uses '∅' to denote the empty set. If there are no inputs, it omits the input part. 
        Returns:
        - reaction_str: string representing the reaction. """
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

class HillProduction(Reaction):
    def __init__(self, product_labels, activator_labels, repressor_labels, input_channels=[None], params=[None], params_controllability=[True]):
        """ Initializes a Hill production reaction.
        Arguments:
        - product_labels: list of strings representing the labels of the products, cannot be empty
        - activator_labels: list of strings representing the labels of the activators, can be empty
        - repressor_labels: list of strings representing the labels of the repressors, can be empty
        - input_channels: list of strings representing the channels of the inputs, cannot be empty, can contain None. None entries indicate that the corresponding parameter does not depend on any input
        - params: list of floats representing the parameters of the reaction, cannot be empty, can contain None. If None, it indicates that the parameter is unknown and needs to be set later 
        - params_controllability: list of booleans representing whether each parameter is controllable by an input, cannot be empty
        The lengths of input_channels, params, and params_controllability must be the same. 
        Each parameter in params corresponds to the parameter controllability in params_controllability and the input channel in input_channels at the same index. 
        Layout of params: [b, Vmax, (Ka for each activator in sorted order), (Kr for each repressor in sorted order)]
        Hill coefficients are fixed to one.
        Where:
        - b: basal production rate
        - Vmax: maximal production rate
        - Ka: dissociation constant for each activator
        - Kr: dissociation constant for each repressor """
        
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
        """ Sets the whole parameter vector.
        Layout: [b, Vmax, (Ka for each activator in sorted order), (Kr for each repressor in sorted order)]. """

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
        """ Returns a list of all species involved in the reaction (products, activators, and repressors). 
        Returns:
        - species: list of strings representing the labels of the involved species. """
        
        species = list(set(self.product_labels + self.activator_labels + self.repressor_labels))
        species.sort()
        return species
    
    def get_involved_inputs(self):
        """ Returns a list of all inputs involved in the reaction. 
        Returns:
        - input_labels: list of strings representing the labels of the involved inputs. """

        return self.input_labels
    
    def get_stoichiometry_dict(self):
        """ Constructs and returns the stoichiometry dictionary for the reaction.
        Returns:
        - stoich: A dictionary mapping labels of species involved in the reaction to their stoichiometric coefficients. """
        
        stoich = {}
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
        self.product_idx = crn.species_label_to_idx(self.product_labels) # single index or list of indices
        self.activator_idx = crn.species_label_to_idx(self.activator_labels) if self.activator_labels else [] # list of indices
        self.repressor_idx = crn.species_label_to_idx(self.repressor_labels) if self.repressor_labels else [] # list of indices
        self.input_idx = crn.input_label_to_idx(self.input_channels) # single index or list of indices

    def propensity(self, x, u):
        """ Computes the propensity of the reaction given species counts x and inputs u.
        Arguments:
        - x: numpy array of shape (num_species,) representing the concentrations of the species of the whole IOCRN.
        - u: numpy array of shape (num_inputs,) representing the inputs to the whole IOCRN.
        Returns:
        - propensity: float representing the propensity of the reaction. """
    
        # Extract relevant species and inputs        
        x_activators = x[self.activator_idx] if self.activator_idx else np.array([])
        x_repressors = x[self.repressor_idx] if self.repressor_idx else np.array([])
        u = np.array([u[i] if i is not None else 1 for i in self.input_idx])

        # Compute the activation term
        activation_term = 1.0
        for i in range(self.num_activators):
            Ka = self.activator_dissociation_constants[i]
            na = 1
            activation_term *= (x_activators[i]**na) / (Ka**na + x_activators[i]**na) if Ka is not None and na is not None else 1.0

        # Compute the repression term
        repression_term = 1.0
        for i in range(self.num_repressors):
            Kr = self.repressor_dissociation_constants[i]
            nr = 1
            repression_term *= Kr**nr / (Kr**nr + x_repressors[i]**nr) if Kr is not None and nr is not None else 1.0

        # Compute the propensity using Hill kinetics
        # return self.basal_rate + (self.maximal_rate - self.basal_rate) * activation_term * repression_term 
        return self.basal_rate + self.maximal_rate * activation_term * repression_term 
    
    def __str__(self):
        """ Returns a string representation of the reaction in the format:
        ∅ ----> Products;  [HILLProd(basal_rate, maximal_rate, (Ka for each activator), (Kr for each repressor), inputs)]
        If there are no inputs, it omits the input part. 
        Returns:
        - reaction_str: string representing the reaction. """
        
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











# class HillProduction(Reaction):
#     def __init__(self, product_labels, activator_labels, repressor_labels, input_channels=[None], params=[None], params_controllability=[True]):
#         """ Initializes a Hill production reaction.
#         Arguments:
#         - product_labels: list of strings representing the labels of the products, cannot be empty
#         - activator_labels: list of strings representing the labels of the activators, can be empty
#         - repressor_labels: list of strings representing the labels of the repressors, can be empty
#         - input_channels: list of strings representing the channels of the inputs, cannot be empty, can contain None. None entries indicate that the corresponding parameter does not depend on any input
#         - params: list of floats representing the parameters of the reaction, cannot be empty, can contain None. If None, it indicates that the parameter is unknown and needs to be set later 
#         - params_controllability: list of booleans representing whether each parameter is controllable by an input, cannot be empty
#         The lengths of input_channels, params, and params_controllability must be the same. 
#         Each parameter in params corresponds to the parameter controllability in params_controllability and the input channel in input_channels at the same index. 
#         Layout of params: [b, Vmax, (Ka,na for each activator in sorted order), (Kr,nr for each repressor in sorted order)]
#         Where:
#         - b: basal production rate
#         - Vmax: maximal production rate
#         - Ka: dissociation constant for each activator
#         - na: Hill coefficient for each activator
#         - Kr: dissociation constant for each repressor
#         - nr: Hill coefficient for each repressor """
        
#         # Ensure the number of parameters is correct (2 + 2*number of activators + 2*number of repressors)
#         assert len(params) ==  2 + 2*len(activator_labels) + 2*len(repressor_labels), "HillProduction reaction must have exactly 2 + 2*number of activators + 2*number of repressors parameters (the basal rate, the maximal rate, and the Hill coefficients and dissociation constants for each activator and repressor)."

#         # Sort and Record the activator and repressor labels
#         if len(activator_labels) > 1:
#             sorted_activator_indices = sorted(range(len(activator_labels)), key=lambda i: activator_labels[i])
#             self.activator_labels = [activator_labels[i] for i in sorted_activator_indices]
#         else:
#             self.activator_labels = activator_labels

#         if len(repressor_labels) > 1:
#             sorted_repressor_indices = sorted(range(len(repressor_labels)), key=lambda i: repressor_labels[i])
#             self.repressor_labels = [repressor_labels[i] for i in sorted_repressor_indices]
#         else:
#             self.repressor_labels = repressor_labels

#         # Sort the parameters
#         # Activators
#         self.num_activators = len(self.activator_labels)
#         params_activators = [params[2 + 2*i : 4 + 2*i] for i in range(self.num_activators)]
#         params_activators = sum([params_activators[i] for i in sorted_activator_indices] if self.num_activators > 1 else params_activators, [])
#         # Repressors
#         self.num_repressors = len(self.repressor_labels)
#         params_repressors = [params[2 + 2*self.num_activators + 2*i : 4 + 2*self.num_activators + 2*i] for i in range(self.num_repressors)]
#         params_repressors = sum([params_repressors[i] for i in sorted_repressor_indices] if self.num_repressors > 1 else params_repressors, [])
#         # Combine sorted parameters
#         params = params[0:2] + params_activators + params_repressors

#         # Sort the input channels
#         # Activators
#         input_channels_activators = [input_channels[2 + 2*i : 4 + 2*i] for i in range(self.num_activators)]
#         input_channels_activators = sum([input_channels_activators[i] for i in sorted_activator_indices] if self.num_activators > 1 else input_channels_activators, [])
#         # Repressors
#         input_channels_repressors = [input_channels[2 + 2*self.num_activators + 2*i : 4 + 2*self.num_activators + 2*i] for i in range(self.num_repressors)]
#         input_channels_repressors = sum([input_channels_repressors[i] for i in sorted_repressor_indices] if self.num_repressors > 1 else input_channels_repressors, [])
#         # Combine sorted input channels
#         input_channels = input_channels[0:2] + input_channels_activators + input_channels_repressors

#         # Sort the parameter controllability
#         # Activators
#         params_controllability_activators = [params_controllability[2 + 2*i : 4 + 2*i] for i in range(self.num_activators)]
#         params_controllability_activators = sum([params_controllability_activators[i] for i in sorted_activator_indices] if self.num_activators > 1 else params_controllability_activators, [])
#         # Repressors
#         params_controllability_repressors = [params_controllability[2 + 2*self.num_activators + 2*i : 4 + 2*self.num_activators + 2*i] for i in range(self.num_repressors)]
#         params_controllability_repressors = sum([params_controllability_repressors[i] for i in sorted_repressor_indices] if self.num_repressors > 1 else params_controllability_repressors, [])
#         # Combine sorted parameter controllability
#         params_controllability = params_controllability[0:2] + params_controllability_activators + params_controllability_repressors

#         # Call the parent constructor
#         super().__init__([], product_labels, input_channels, params, params_controllability)

#         # Create the reaction signature: depends on the reaction structure only, not on the parameters or inputs
#         # Signature format: ('HILLProd', product_labels, activator_labels, repressor_labels) all sorted alphanumerically
#         self.signature = str(('HILLProd', self.product_labels, self.activator_labels, self.repressor_labels))

#         # Record the basal and maximal rates, the dissociation constants, and the Hill coefficients
#         self.basal_rate = self.params[0]                                                 # float or None
#         self.maximal_rate = self.params[1]                                               # float or None
#         self.activator_dissociation_constants = self.params[2:2+2*self.num_activators:2] if self.activator_labels else []  # list of floats or None
#         self.activator_hill_coefficients = self.params[3:2+2*self.num_activators:2] if self.activator_labels else []  # list of floats or None
#         self.repressor_dissociation_constants = self.params[2+2*self.num_activators:2+2*self.num_activators+2*self.num_repressors:2] if self.repressor_labels else []  # list of floats or None
#         self.repressor_hill_coefficients = self.params[3+2*self.num_activators:2+2*self.num_activators+2*self.num_repressors:2] if self.repressor_labels else []  # list of floats or None

#         self.num_continuous_parameters = 2 + len(self.activator_labels) + len(self.repressor_labels)                            # basal and maximal rates, and dissociation constants
#         self.num_discrete_parameters = len(self.activator_labels) + len(self.repressor_labels)                                  # Hill coefficients
#         self.num_unknown_params = self.get_num_unknown_params()   

#     def set_parameters(self, params):
#         """ Sets the whole parameter vector.
#         Layout: [b, Vmax, (Ka,na for each activator in sorted order), (Kr,nr for each repressor in sorted order)]. """

#         # Ensure the number of parameters is correct (2 + 2*number of activators + 2*number of repressors)
#         assert len(params) ==  2 + 2*len(self.activator_labels) + 2*len(self.repressor_labels), "HillProduction reaction must have exactly 2 + 2*number of activators + 2*number of repressors parameters (the basal rate, the maximal rate, and the Hill coefficients and dissociation constants for each activator and repressor)."

#         # Set the basal and maximal rates, the dissociation constants, and the Hill coefficients
#         self.basal_rate = params[0]                                                 # float or None
#         self.maximal_rate = params[1]                                               # float or None
#         self.activator_dissociation_constants = params[2:2+len(self.activator_labels)] if self.activator_labels else []  # list of floats or None
#         self.activator_hill_coefficients = params[2+len(self.activator_labels):2+2*len(self.activator_labels)] if self.activator_labels else []  # list of floats or None
#         self.repressor_dissociation_constants = params[2+2*len(self.activator_labels):2+2*len(self.activator_labels)+len(self.repressor_labels)] if self.repressor_labels else []  # list of floats or None
#         self.repressor_hill_coefficients = params[2+2*len(self.activator_labels)+len(self.repressor_labels):] if self.repressor_labels else []  # list of floats or None

#         # Update the params list
#         self.params = params

#     def get_involved_species(self):
#         """ Returns a list of all species involved in the reaction (products, activators, and repressors). 
#         Returns:
#         - species: list of strings representing the labels of the involved species. """
        
#         species = list(set(self.product_labels + self.activator_labels + self.repressor_labels))
#         species.sort()
#         return species
    
#     def get_involved_inputs(self):
#         """ Returns a list of all inputs involved in the reaction. 
#         Returns:
#         - input_labels: list of strings representing the labels of the involved inputs. """
        
#         return self.input_labels
    
#     def get_stoichiometry_dict(self):
#         """ Constructs and returns the stoichiometry dictionary for the reaction.
#         Returns:
#         - stoich: A dictionary mapping labels of species involved in the reaction to their stoichiometric coefficients. """
        
#         stoich = {}
#         for product_label in self.product_labels:
#             if product_label in stoich:
#                 stoich[product_label] += 1
#             else:
#                 stoich[product_label] = 1
#         return stoich
    
#     def set_crn_context(self, crn):
#         """ Sets the context of the reaction within a given IOCRN by mapping species and input labels to their respective indices. 
#         Arguments:
#         - crn: An instance of IOCRN containing species and input labels. 
#         """
        
#         super().set_crn_context(crn)
#         self.product_idx = crn.species_label_to_idx(self.product_labels) # single index or list of indices
#         self.activator_idx = crn.species_label_to_idx(self.activator_labels) if self.activator_labels else [] # list of indices
#         self.repressor_idx = crn.species_label_to_idx(self.repressor_labels) if self.repressor_labels else [] # list of indices
#         self.input_idx = crn.input_label_to_idx(self.input_channels) # single index or list of indices

#     def propensity(self, x, u):
#         """ Computes the propensity of the reaction given species counts x and inputs u.
#         Arguments:
#         - x: numpy array of shape (num_species,) representing the concentrations of the species of the whole IOCRN.
#         - u: numpy array of shape (num_inputs,) representing the inputs to the whole IOCRN.
#         Returns:
#         - propensity: float representing the propensity of the reaction. """
        
#         # x = x[self.reactant_idx]
#         # u = u[self.input_idx[0]] if self.input_idx[0] is not None else 1.0
#         # return self.rate_constant * np.prod(x) * u
    
#         # Extract relevant species and inputs        
#         x_activators = x[self.activator_idx] if self.activator_idx else np.array([])
#         x_repressors = x[self.repressor_idx] if self.repressor_idx else np.array([])
#         u = np.array([u[i] if i is not None else 1 for i in self.input_idx])

#         # Compute the activation term
#         activation_term = 1.0
#         for i in range(self.num_activators):
#             Ka = self.activator_dissociation_constants[i]
#             na = self.activator_hill_coefficients[i]
#             activation_term *= (x_activators[i]**na) / (Ka**na + x_activators[i]**na) if Ka is not None and na is not None else 1.0

#         # Compute the repression term
#         repression_term = 1.0
#         for i in range(self.num_repressors):
#             Kr = self.repressor_dissociation_constants[i]
#             nr = self.repressor_hill_coefficients[i]
#             repression_term *= Kr**nr / (Kr**nr + x_repressors[i]**nr) if Kr is not None and nr is not None else 1.0

#         # Compute the propensity using Hill kinetics
#         return self.basal_rate + (self.maximal_rate - self.basal_rate) * activation_term * repression_term 
    
#     def __str__(self):
#         """ Returns a string representation of the reaction in the format:
#         ∅ ----> Products;  [HILLProd(basal_rate, maximal_rate, (Ka,na for each activator), (Kr,nr for each repressor), inputs)]
#         If there are no inputs, it omits the input part. 
#         Returns:
#         - reaction_str: string representing the reaction. """
        
#         try:
#             species_str = f"{self.product_labels} : {self.product_idx}, {self.activator_labels} : {self.activator_idx}, {self.repressor_labels} : {self.repressor_idx}, {self.input_channels} : {self.input_idx}"
#         except:
#             species_str = "unset"
            
#         reactants_str = '∅'
#         products_str = ' + '.join(self.product_labels) if self.product_labels else '∅'
        
#         # Construct parameters string
#         # Basal rate
#         if self.input_channels[0] is None:
#             params_str = f"b = {self.basal_rate}, "
#         else:
#             params_str = f"b = {self.basal_rate}{self.input_channels[0]}, "

#         # Maximal rate
#         if self.input_channels[1] is None:
#             params_str += f"Vm = {self.maximal_rate},    "
#         else:
#             params_str += f"Vm = {self.maximal_rate}{self.input_channels[1]},    "

#         # Activators
#         params_str += f"(Ka, na) = "
#         for i in range(self.num_activators):
#             if self.input_channels[2 + 2*i] is None:
#                 params_str += f"{self.activator_labels[i]}({self.activator_dissociation_constants[i]}, {self.activator_hill_coefficients[i]})"
#             else:
#                 params_str += f"{self.activator_labels[i]}({self.activator_dissociation_constants[i]}{self.input_channels[2 + 2*i]}, {self.activator_hill_coefficients[i]})"
#             params_str += ", " if i < self.num_activators - 1 else ""

#         # Repressors
#         params_str += f";    (Kr, nr) = "
#         for i in range(self.num_repressors):
#             if self.input_channels[2 + 2*self.num_activators + 2*i] is None:
#                 params_str += f"{self.repressor_labels[i]}({self.repressor_dissociation_constants[i]}, {self.repressor_hill_coefficients[i]})"
#             else:
#                 params_str += f"{self.repressor_labels[i]}({self.repressor_dissociation_constants[i]}{self.input_channels[2 + 2*self.num_activators + 2*i]}, {self.repressor_hill_coefficients[i]})"
#             params_str += ", " if i < self.num_repressors - 1 else ""
        
#         return f"{reactants_str} ----> {products_str};  [HILLProd({params_str})]"