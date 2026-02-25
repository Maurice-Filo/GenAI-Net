"""
Concrete stepper for IO-CRN environments.

This stepper applies an action by adding a reaction to the environment `state`
via `state.add_reaction(action)` and then calling `state.compile()`.

Limitations:
    Parameter-setting is not implemented. If the state reports unknown parameters
    (`state.num_unknown_params > 0`), `step` raises `NotImplementedError`.

Assumptions about `state` :

- has attribute `num_unknown_params: int`
- provides methods `add_reaction(action)` and `compile()`

"""


from RL4CRN.agent2env_interface.abstract_stepper import AbstractStepper

class IOCRNStepper(AbstractStepper):
    """Stepper that adds a reaction to the CRN state and recompiles it.

    The `action` is applied through `state.add_reaction(action)`. Afterward,
    `state.compile()` is called to finalize the updated state.

    Raises:
        NotImplementedError: If `state.num_unknown_params > 0`.
    """

    def step(self, state, action):
        """Apply `action` to `state` by adding a reaction and compiling.

        Args:
            state: Environment state to be modified (see module docstring for the
                required interface).
            action: Reaction/action object passed to `state.add_reaction(action)`.

        Raises:
            NotImplementedError: If the state contains unknown parameters and would
                require setting them (`state.num_unknown_params > 0`).
        """
        if state.num_unknown_params > 0:
            raise NotImplementedError("Setting parameters directly is not implemented yet.")
        else:
            state.add_reaction(action)
            state.compile()