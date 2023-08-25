"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 18.08.2023
"""
import numpy as np

from kamisado.envs.game import Game


def __relative_to_int(action):
    return (Game.relative_actions == action).all(1).nonzero()[0][0]


def mask_fn(env):
    valid_tower = env.current_tower
    if valid_tower is None:
        # for the first move, the valid moves depend on the chosen tower
        # we just allow all and hope the model figures out the rest
        return np.ones(22 + 8, dtype=bool)

    target_mask = env.target_mask(valid_tower)
    tower_mask = np.zeros(8, dtype=bool)
    tower_mask[valid_tower - 1] = True
    return np.append(target_mask, tower_mask)
