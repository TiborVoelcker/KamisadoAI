"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 09.08.2023
"""
from gymnasium import RewardWrapper
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from kamisado.agents.simple import LookForWinAgent

from .flatten import FlattenAction, FlattenObservation
from .tournament import TournamentWrapper
from .tower_selection import NoTowerSelection


def wrap(
    env,
    tournament=True,
    tournament_opponent=LookForWinAgent,
    tower_selection=True,
    reward_action=False,
):
    if reward_action:
        env = ActionReward(env)
    if not tower_selection:
        env = NoTowerSelection(env)
    env = FlattenAction(FlattenObservation(env))
    if tournament and tournament_opponent:
        env = TournamentWrapper(env, tournament_opponent)

    return Monitor(env)


class ActionReward(RewardWrapper):
    """Wrapper to reward taken any valid action."""

    def reward(self, reward):
        if reward == 0:
            return 1
        return reward
