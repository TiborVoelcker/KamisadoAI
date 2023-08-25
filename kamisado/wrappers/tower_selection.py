"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 18.08.2023
"""
from gymnasium import ActionWrapper


class NoTowerSelection(ActionWrapper):
    """Wrapper that automatically selects the correct tower.

    Changing the tower must happen before resolving the relative actions
    (thus `NoTowerSelection` must wrap the `RelativeAction` wrapper).
    """

    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        if self.env.current_tower:
            action[1] = self.env.current_tower

        return action
