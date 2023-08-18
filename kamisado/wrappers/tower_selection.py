"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 18.08.2023
"""
from gymnasium import ActionWrapper, spaces


class NoTowerSelection(ActionWrapper):
    """Wrapper that automatically selects the correct tower.

    Changing the tower must happen before resolving the relative actions
    (thus `NoTowerSelection` must wrap the `RelativeAction` wrapper).
    """

    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        tower = self.env.current_tower
        if tower is None:
            tower = action["tower"]
        return {"target": action["target"], "tower": tower}
