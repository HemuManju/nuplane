from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass

import math

import time


class BaseActor(with_metaclass(ABCMeta, object)):
    def __init__(self, client, config=None) -> None:
        self.client = client
        self.config = config
        self.map = None

    @abstractmethod
    def reset(self, *args, **kwargs):
        """Reset the actor

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation(self, *args, **kwargs):
        """Get the observation from the actor.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, *args, **kwargs):
        """Apply the action to the actor

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

