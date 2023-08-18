"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 09.08.2023
"""
from gymnasium import make, register
from stable_baselines3.common.monitor import Monitor

from kamisado.wrappers import wrap

register(
    id="kamisado/Game-v0",
    entry_point="kamisado.envs:Game",
    autoreset=True,
    order_enforce=True,
    max_episode_steps=100,
    reward_threshold=1,
)


def make_env(tower_selection=True, mask=True, reward_action=False, **kwargs):
    env = make("kamisado/Game-v0", **kwargs)
    return Monitor(
        wrap(env, tower_selection=tower_selection, mask=mask, reward_action=reward_action)
    )
