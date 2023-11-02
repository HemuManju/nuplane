import numpy as np

from abc import ABC, abstractmethod


class BaseController(ABC):
    @abstractmethod
    def __init__(self, dt, input_constraints):
        self.dt = dt
        self.input_constraints = input_constraints
        self.state_reference = None
        self.input_reference = None

    @abstractmethod
    def reset(self):
        raise (NotImplementedError)

    @abstractmethod
    def get_control(self, *args, **kwargs):
        raise (NotImplementedError)

    def get_input(self, state, estop=False):
        input = self.get_control(state, estop=estop)
        return np.clip(input, self.input_constraints[0], self.input_constraints[1])

    def set_reference(self, state_reference, input_reference):
        self.state_reference = state_reference
        self.input_reference = input_reference
