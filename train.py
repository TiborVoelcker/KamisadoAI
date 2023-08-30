"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 30.08.2023
"""
from kamisado.agents.ppo import train
from kamisado.agents.simple import LookForWinAgent, RandomAgent

if __name__ == "__main__":
    train(1000000, tournament_opponent=RandomAgent)
