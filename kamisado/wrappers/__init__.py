"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 09.08.2023
"""
from stable_baselines3.common.monitor import Monitor

from kamisado.agents.simple import LookForWinAgent

from .tournament import TournamentWrapper


def wrap(env, tournament_opponent=LookForWinAgent):
    if tournament_opponent:
        env = TournamentWrapper(env, tournament_opponent)

    return Monitor(env)
