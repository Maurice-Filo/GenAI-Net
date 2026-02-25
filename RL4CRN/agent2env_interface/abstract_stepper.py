"""
Abstract stepper interface.

Defines the minimal interface for a *stepper*, i.e., a component responsible for
updating an environment state given an action. In this project, steppers are used
to interface reinforcement learning agents with CRN environments.

A concrete implementation should implement `step` and encode the rule that
transforms a `(state, action)` pair into an updated state.
"""

from typing import Protocol

class AbstractStepper(Protocol):
    """Protocol for steppers that advance an environment state given an action."""

    def step(self, state, action):
        """Advance the environment state by applying an action.

        Args:
            state: Environment state to be updated.
            action: Action to apply to the state.

        Returns:
            Implementations may either mutate `state` in-place and return `None`, or return an updated state object. Concrete steppers should document their chosen behavior.
        """
        pass