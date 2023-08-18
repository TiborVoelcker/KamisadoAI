"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 09.08.2023
"""
from gymnasium import RewardWrapper
from sb3_contrib.common.wrappers import ActionMasker

from .flatten import FlattenAction, FlattenObservation
from .mask import mask_fn
from .relative import RelativeAction
from .tower_selection import NoTowerSelection


def wrap(env, tower_selection=True, mask=True, reward_action=False):
    if reward_action:
        env = ActionReward(env)
    env = RelativeAction(env)
    if not tower_selection:
        env = NoTowerSelection(env)
    env = FlattenAction(FlattenObservation(env))
    if mask:
        env = ActionMasker(env, mask_fn)

    return env


class ActionReward(RewardWrapper):
    """Wrapper to reward taken any valid action."""

    def reward(self, reward):
        if reward == 0:
            return 1
        return reward
