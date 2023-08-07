"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 14.07.2023
"""
from abc import ABC, abstractmethod

from game import Kamisado


class Model(ABC):
    """An agent to play Kamisado.

    Props:
        player: The side the agent plays for.
        game: The game the agent plays.
    """

    def __init__(self, env: Kamisado):
        """Initialize the class."""
        self.env = env

    @abstractmethod
    def predict(self, obs) -> tuple[tuple[int, int], None]:
        """Choose an action based on a observation."""
        pass


from .simple import LookForWinAgent, RandomAgent
