"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 14.07.2023
"""
from abc import ABC, abstractmethod
from collections.abc import Sequence

from game import Kamisado, Tower


class Agent(ABC):
    """An agent to play Kamisado.

    Props:
        player: The side the agent plays for.
        game: The game the agent plays.
    """

    def __init__(self, player: str, game: Kamisado):
        """Initialize the class."""
        self.player = player
        self.game = game

    @abstractmethod
    def choose_tower(self, towers: Sequence[Tower]) -> Tower:
        """Chooses the tower to start the game with."""
        pass

    @abstractmethod
    def choose_action(self, actions: Sequence[tuple[int, int]]) -> tuple[int, int]:
        """Chooses an action out of a list of possible actions."""
        pass


from .simple import LookForWinAgent, RandomAgent
