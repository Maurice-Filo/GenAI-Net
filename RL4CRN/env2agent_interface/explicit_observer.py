"""
Explicit observer.

This module defines `ExplicitObserver`, an observer that constructs an
explicit, vector-based representation of an IOCRN state. The representation is
based on:

- which reactions are currently present (multi-hot over reaction IDs),
- the current reaction parameters placed into a fixed parameter vector,
- optionally, a multi-hot encoding of which input channels control which
  controllable parameters.

The observer is intended for use in the env2agent interface, where an
environment state (IOCRN) is mapped to an observation suitable for subsequent
tensorization and policy evaluation.
"""

import numpy as np
from RL4CRN.env2agent_interface.abstract_observer import AbstractObserver

class ExplicitObserver(AbstractObserver):
    """Observer producing an explicit IOCRN representation as numpy arrays.

    Args:
        reaction_library: Reaction library used to define the observation space.
            The observer assumes the library provides:

            - `__len__()` giving the number of reactions in the library
            - `get_num_parameters()` returning the size of the global parameter vector
            - `parameter_lookup_table` mapping `reaction.ID` to the parameter offset
            - `get_num_controllable_parameters()` returning the size of the global
              controllable-parameter vector
            - `controllable_parameter_lookup_table` mapping `reaction.ID` to the
              controllable-parameter offset
        allow_input_observation: If True, the observation additionally includes an
            input-control multi-hot encoding (see `inputs_to_multihot`).

    Notes:
        This observer stores the most recent IOCRN passed to `observe` in
            `self.iocrn` and uses it internally when constructing the encodings.
    """

    def __init__(self, reaction_library, allow_input_observation=False):
        super().__init__()
        self.reaction_library = reaction_library
        self.allow_input_observation = allow_input_observation
        self.iocrn = None

    def observe(self, iocrn):
        """Construct an explicit observation for the given IOCRN.

        The returned observation is a tuple of numpy arrays:

        - `reaction_multihot`: multi-hot encoding indicating which reactions from
          the library are present in the IOCRN. Shape `(M,)`, where
          `M = len(reaction_library)`.
        - `params_cross_multihot`: parameter vector containing the current reaction
          parameters placed into a fixed global layout defined by
          `reaction_library.parameter_lookup_table`. Shape `(P,)`, where
          `P = reaction_library.get_num_parameters()`.
        - optionally, `inputs_multihot` (only if `allow_input_observation=True`):
          concatenated multi-hot vectors describing which controllable parameters
          are controlled by each input channel. Shape `(num_inputs * C,)`, where
          `C = reaction_library.get_num_controllable_parameters()`.

        Args:
            iocrn: IOCRN-like object providing at least:

                - `gather_reaction_IDs()` returning reaction IDs present in the IOCRN
                - `reactions`: iterable of reaction objects with fields:

                    - `ID`
                    - `num_parameters`
                    - `params`
                    - `get_num_controllable_parameters()`
                    - `input_channels`
                - `num_inputs`
                - `input_labels`

        Returns:
            Tuple of numpy arrays:
            
                - `(reaction_multihot, params_cross_multihot)` if input observation
                  is disabled.
                - `(reaction_multihot, params_cross_multihot, inputs_multihot)` if
                  input observation is enabled.
        """
        self.iocrn = iocrn
        reaction_multihot = self.reactions_to_multihot()                  # shape (M,)
        params_cross_multihot = self.params_cross_multihot()              # shape (P,)
        if self.allow_input_observation:
            inputs_multihot = self.inputs_to_multihot()                   # shape (p, C)
            explicit_state = (reaction_multihot, params_cross_multihot, inputs_multihot)
        else:
            explicit_state = (reaction_multihot, params_cross_multihot)
        return explicit_state
        
    def reactions_to_multihot(self):
        """Encode present reactions as a multi-hot vector.

        Uses `self.iocrn.gather_reaction_IDs()` to obtain the set of reaction IDs
        present in the IOCRN and sets the corresponding entries to 1.

        Returns:
            Numpy array of shape `(len(reaction_library),)` with entries in `{0, 1}`.
        """
        idx = np.array(self.iocrn.gather_reaction_IDs(), dtype=np.long) 
        multihot = np.zeros(len(self.reaction_library)) 
        multihot[idx] = 1.
        return multihot
    
    def params_cross_multihot(self):
        """Place reaction parameters into a fixed global parameter vector.

        For each reaction in `self.iocrn.reactions`, parameters are copied into a
        global vector at offsets determined by
        `reaction_library.parameter_lookup_table[reaction.ID]`.

        Returns:
            Numpy array of shape `(reaction_library.get_num_parameters(),)` where
                entries corresponding to active reaction parameters contain their
                numeric values and all other entries are zero.
        """
        multihot = np.zeros(self.reaction_library.get_num_parameters())
        for reaction in self.iocrn.reactions:
            idx = self.reaction_library.parameter_lookup_table[reaction.ID]
            for j in range(reaction.num_parameters):
                multihot[idx + j] = reaction.params[j]
        return multihot
    
    def inputs_to_multihot(self):
        """Encode which inputs control which controllable reaction parameters.

        For each input channel `i` in the IOCRN, this method creates a multi-hot
        vector over the global controllable-parameter layout of the reaction
        library. A position is set to 1 if the corresponding controllable
        parameter is controlled by input `i`, based on matching
        `reaction.input_channels[j]` to `self.iocrn.input_labels[i]`.

        The per-input vectors are concatenated into a single vector.

        Returns:
            Numpy array of shape
                `(self.iocrn.num_inputs * reaction_library.get_num_controllable_parameters(),)`
                with entries in `{0, 1}`.
        """
        multihots = []
        for i in range(self.iocrn.num_inputs):
            multihot = np.zeros(self.reaction_library.get_num_controllable_parameters())
            for reaction in self.iocrn.reactions:
                idx = self.reaction_library.controllable_parameter_lookup_table[reaction.ID]
                for j in range(reaction.get_num_controllable_parameters()):
                    multihot[idx + j] = 1 if reaction.input_channels[j] == self.iocrn.input_labels[i] else 0
            multihots.append(multihot)
        multihots = np.concatenate(multihots)
        return multihots