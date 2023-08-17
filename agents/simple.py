"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 14.07.2023
"""
from agents import Model


class RandomAgent(Model):
    """An agent that plays randomly."""

    def predict(self, obs):
        tower = self.env.current_tower
        if tower is None:
            tower = self.env.np_random.integers(1, 9)
        actions = self.env.valid_actions(tower)
        return {"target": self.env.np_random.choice(actions), "tower": tower}, None


class LookForWinAgent(Model):
    """An agent that plays randomly, but will make the winning move if it can."""

    def predict(self, obs):
        tower = self.env.current_tower
        if tower is None:
            tower = self.env.np_random.integers(1, 9)
        actions = self.env.valid_actions(tower)

        winning = actions[actions[:, 0] == 0]
        if len(winning) != 0:
            return {"tower": tower, "target": winning[0]}, None

        return {"target": self.env.np_random.choice(actions), "tower": tower}, None
