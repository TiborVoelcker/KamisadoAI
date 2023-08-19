"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 18.08.2023
"""
"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 18.08.2023
"""
import time

from gymnasium import make
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from ...wrappers import wrap


def train(timesteps, mask=False, **kwargs):
    env = make("kamisado/Game-v0")
    env = wrap(env, **kwargs)
    eval_env = wrap(make("kamisado/Game-v0"), **kwargs)

    if mask:
        from sb3_contrib.ppo_mask import MaskablePPO as PPO

        filename = "kamisado/agents/ppo/model_masked"

        model = PPO.load(filename + "/best_model", env=env)

    else:
        from stable_baselines3.ppo import PPO

        filename = "kamisado/agents/ppo/model"

        model = PPO.load(filename + "/best_model", env=env)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=50, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        best_model_save_path=filename,
        eval_freq=10000,
        n_eval_episodes=50,
        verbose=1,
    )

    # Train the agent
    start_time = time.time()
    model.learn(
        total_timesteps=timesteps,
        reset_num_timesteps=False,
        callback=eval_callback,
        progress_bar=True,
    )
    print(f"Training took {time.time() - start_time:.2f}s")
