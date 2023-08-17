"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 09.08.2023
"""
import numpy as np
from gymnasium import ActionWrapper, spaces

from kamisado.envs import Game


class FlattenAction(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(0, 7, shape=(3,), dtype=np.int8)

    def action(self, action):
        return {"target": action[0:2], "tower": action[2] + 1}


class NoTowerSelection(ActionWrapper):
    def __init__(self, env: Game):
        super().__init__(env)
        # we use the same action space as for the `FlattenAction` wrapper,
        # so we can reuse the model
        self.action_space = spaces.Box(0, 7, shape=(3,), dtype=np.int8)

    def action(self, action):
        tower = self.env.current_tower
        if tower is None:
            tower = self.env.np_random.integers(1, 9)
        return {"target": action[0:2], "tower": tower}
