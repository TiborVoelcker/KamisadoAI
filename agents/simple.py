"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 14.07.2023
"""
import random
from collections.abc import Sequence

from agents import Agent
from game import Tower


class RandomAgent(Agent):
    """An agent that plays randomly."""

    def choose_tower(self, towers: Sequence[Tower]) -> Tower:
        if self.player != "black":
            raise RuntimeError("White was asked to begin the game!")
        return random.choice(towers)

    def choose_action(self, actions: Sequence[tuple[int, int]]) -> tuple[int, int]:
        return random.choice(actions)


class LookForWinAgent(Agent):
    """An agent that plays randomly, but will make the winning move if it can."""

    def choose_tower(self, towers: Sequence[Tower]) -> Tower:
        if self.player != "black":
            raise RuntimeError("White was asked to begin the game!")
        return random.choice(towers)

    def choose_action(self, actions: Sequence[tuple[int, int]]) -> tuple[int, int]:
        goal = 7 if self.player == "black" else 0
        for x, y in actions:
            if y == goal:
                return (x, y)
        return random.choice(actions)
