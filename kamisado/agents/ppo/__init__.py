"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 18.08.2023
"""
"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 18.08.2023
"""
import time

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from kamisado import make_env


def train(timesteps, tower_selection=False, mask=True, reward_action=False):
    env = make_env(tower_selection=tower_selection, mask=mask, reward_action=reward_action)

    if mask:
        from sb3_contrib.ppo_mask import MaskablePPO as PPO

        filename = "kamisado/agents/ppo/model_masked"

        model = PPO.load(filename, env=env)

    else:
        from stable_baselines3.ppo import PPO

        filename = "kamisado/agents/ppo/model"

        model = PPO.load(filename, env=env)

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
