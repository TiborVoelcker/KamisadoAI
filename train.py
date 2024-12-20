"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 30.08.2023
"""
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.policy.policy import PolicySpec

from kamisado.envs.game import Game

ray.init(local_mode=True)

config = PPOConfig().environment(Game).multi_agent(policies={"main": PolicySpec()})

algo = PPO(config)

while True:
    print(algo.train())
