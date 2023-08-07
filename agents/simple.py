"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 14.07.2023
"""
from agents import Model


class RandomAgent(Model):
    """An agent that plays randomly."""

    def predict(self, obs):
        actions = self.env.valid_actions(self.env.next_tower)
        return self.env.np_random.choice(actions), None


class LookForWinAgent(Model):
    """An agent that plays randomly, but will make the winning move if it can."""

    def predict(self, obs):
        actions = self.env.valid_actions(self.env.next_tower)
        if self.env.next_tower is not None:
            goal = 7 if self.env.next_tower < 0 else 0
            winning = actions[actions[:, 0] == goal]
            if len(winning) != 0:
                return winning[0], None

        return self.env.np_random.choice(actions), None
