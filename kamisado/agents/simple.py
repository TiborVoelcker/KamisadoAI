"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 14.07.2023
"""
import numpy as np

from kamisado.agents import Model


class RandomAgent(Model):
    """An agent that plays randomly."""

    def predict(self, obs, **kwargs):
        tower = obs[-1]
        if tower == 0:
            tower = self.env.np_random.integers(1, 9)
        mask = self.env.target_mask(tower)
        target = self.env.np_random.choice(mask.nonzero()[0])
        return np.append(tower - 1, target), None


class LookForWinAgent(Model):
    """An agent that plays randomly, but will make the winning move if it can."""

    def predict(self, obs, **kwargs):
        tower = obs[-1]
        if tower == 0:
            tower = self.env.np_random.integers(1, 9)
        mask = self.env.target_mask(tower)
        tower_coords = self.env.get_tower_coords(tower)
        targets = self.env.relative_actions + tower_coords
        winning = (targets[:, 0] == 0) & mask
        if winning.any():
            target = targets[winning][0]
        else:
            target = self.env.np_random.choice(mask.nonzero()[0])
        return np.append(tower - 1, target), None
