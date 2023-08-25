"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 18.08.2023
"""
import numpy as np

from .relative import RelativeAction


def __relative_to_int(action):
    return (RelativeAction.int_to_relative == action).all(1).nonzero()[0][0]


def mask_fn(env):
    valid_tower = env.current_tower
    if valid_tower is None:
        # for the first move, the valid moves depend on the chosen tower
        # we just allow all and hope the model figures out the rest
        return np.ones(22 + 8, dtype=bool)

    valid_actions = env.valid_actions(valid_tower)
    # get indexes for actions
    valid_actions = np.array([__relative_to_int(action) for action in valid_actions])

    actions_mask = np.zeros(22, dtype=bool)
    actions_mask[valid_actions] = True
    tower_mask = np.zeros(8, dtype=bool)
    tower_mask[valid_tower - 1] = True
    return np.append(actions_mask, tower_mask)
