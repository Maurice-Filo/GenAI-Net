from typing import Protocol

class AbstractStepper(Protocol):

    def step(self, state, action):
        pass