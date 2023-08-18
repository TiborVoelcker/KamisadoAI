"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 18.08.2023
"""
"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 18.08.2023
"""
import time
from pathlib import Path

from gymnasium import make
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.policies import MlpPolicy

from kamisado.wrappers import wrap


def make_env(**kwargs):
    env = make("kamisado/Game-v0", **kwargs)
    return Monitor(wrap(env, tower_selection=False))


def train(timesteps, filename="kamisado/agents/ppo/model"):
    env = make_env()
    check_env(env)

    if Path(filename + ".zip").is_file():
        model = PPO.load(filename, env=env)
    else:
        model = PPO(MlpPolicy, env=env)

    # Train the agent
    start_time = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=EvalCallback(env, eval_freq=timesteps // 50, n_eval_episodes=10),
        reset_num_timesteps=False,
    )
    print(f"Training took {time.time() - start_time:.2f}s")

    model.save(filename)

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
