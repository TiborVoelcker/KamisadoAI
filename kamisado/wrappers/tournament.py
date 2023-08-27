"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 18.08.2023
"""
from gymnasium import Wrapper
from gymnasium.core import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from kamisado.agents import Model


class TournamentWrapper(Wrapper):
    def __init__(
        self, env: Env, opponent: type[BaseAlgorithm] | type[Model], enforce_side=False, **kwargs
    ):
        self.enforce_side = enforce_side
        super().__init__(env)
        self.opponent = opponent(env=env, **kwargs)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        if self.enforce_side is False:
            self.opponent_color = self.np_random.choice([0, 1])
        else:
            self.opponent_color = 1 if self.enforce_side == 0 else 0
        if self.opponent_color == 0:
            action, _ = self.opponent.predict(obs, deterministic=True)
            obs, reward, truncated, terminated, info = super().step(action)
            if truncated or terminated:
                raise UserWarning("The opponent player's starting move was invalid!")
        return obs, info

    def step(self, action):
        obs, reward, truncated, terminated, info = super().step(action)
        if truncated or terminated:
            return obs, reward, truncated, terminated, info

        action, _ = self.opponent.predict(obs, deterministic=True)
        obs, reward_opp, truncated, terminated, info = super().step(action)
        if truncated:
            return obs, reward, truncated, terminated, info

        return obs, reward - reward_opp, truncated, terminated, info
