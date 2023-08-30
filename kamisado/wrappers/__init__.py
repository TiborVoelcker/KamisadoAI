"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 09.08.2023
"""
from gymnasium import RewardWrapper
from stable_baselines3.common.monitor import Monitor

from kamisado.agents.simple import LookForWinAgent

from .tournament import TournamentWrapper


def wrap(
    env,
    tournament_opponent=LookForWinAgent,
    reward_action=False,
):
    if reward_action:
        env = ActionReward(env)
    if tournament_opponent:
        env = TournamentWrapper(env, tournament_opponent)

    return Monitor(env)


class ActionReward(RewardWrapper):
    """Wrapper to reward taken any valid action."""

    def reward(self, reward):
        if reward == 0:
            return 1
        return reward
