"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 18.08.2023
"""
import numpy as np
from gymnasium import ActionWrapper, spaces


class RelativeAction(ActionWrapper):
    """Wrapper to use relative actions for tower movement.

    The relative action is selected as an discrete index referencing the action.
    """

    int_to_relative = np.array(
        [
            [0, 0],
            [-1, 0],
            [-2, 0],
            [-3, 0],
            [-4, 0],
            [-5, 0],
            [-6, 0],
            [-7, 0],
            [-1, -1],
            [-2, -2],
            [-3, -3],
            [-4, -4],
            [-5, -5],
            [-6, -6],
            [-7, -7],
            [-1, 1],
            [-2, 2],
            [-3, 3],
            [-4, 4],
            [-5, 5],
            [-6, 6],
            [-7, 7],
        ]
    )

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Dict(
            {
                "tower": spaces.Discrete(8, start=1),
                "target": spaces.Discrete(21),
            }
        )

    def action(self, action):
        target = self.env.get_tower_coords(action["tower"])
        target = target + self.int_to_relative[action["target"]]
        return {"target": target, "tower": action["tower"]}
