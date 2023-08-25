"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 18.08.2023
"""
import numpy as np
from gymnasium import ActionWrapper, spaces

from kamisado.envs.game import Game


class RelativeAction(ActionWrapper):
    """Wrapper to use relative actions for tower movement.

    The relative action is selected as an discrete index referencing the action.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Dict(
            {
                "tower": spaces.Discrete(8, start=1),
                "target": spaces.Discrete(21),
            }
        )

    def action(self, action):
        target = Game.relative_actions[action["target"]]
        return {"target": target, "tower": action["tower"]}
