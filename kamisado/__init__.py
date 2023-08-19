"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 09.08.2023
"""
from gymnasium import register

register(
    id="kamisado/Game-v0",
    entry_point="kamisado.envs:Game",
    autoreset=True,
    order_enforce=True,
    max_episode_steps=100,
    reward_threshold=1,
)
