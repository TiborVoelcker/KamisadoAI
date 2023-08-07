"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 14.07.2023
"""


from agents import Model
from game import Kamisado


class Tournament(Kamisado):
    def __init__(self, opponent: type[Model], *args, **kwargs):
        self.opponent = opponent(self)
        super().__init__(*args, **kwargs)

        self.reset()

    def reset(self, seed=None):
        obs, info = super().reset(seed)
        self.opponent_color = self.np_random.choice([-1, 1])
        if self.opponent_color == -1:
            action, _ = self.opponent.predict(obs)
            obs, reward, truncated, terminated, info = super().step(action)
            if not truncated or not terminated:
                action, _ = self.opponent.predict(obs)
                obs, reward, truncated, terminated, info = super().step(action)
                return obs, info
            else:
                return self.reset(seed=seed)
        return obs, info

    def step(self, action: tuple[int, int]):
        obs, reward, truncated, terminated, info = super().step(action)
        if not truncated and not terminated:
            action, _ = self.opponent.predict(obs)
            obs, opponent_reward, truncated, terminated, info = super().step(action)
            reward = reward - opponent_reward
        return obs, reward, truncated, terminated, info
