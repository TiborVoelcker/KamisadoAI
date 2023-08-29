"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 18.08.2023
"""
import time
from functools import partial
from pathlib import Path

from gymnasium import make
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.ppo import PPO

from kamisado.wrappers import wrap

path = Path("kamisado/agents/ppo/model/best_model.zip")


def train(timesteps, **kwargs):
    if kwargs.get("tournament_opponent") == "self":
        if not path.exists():
            raise UserWarning(
                "Cannot use `tournament_opponent` 'self', if model does not exist yet!"
            )
        kwargs["tournament_opponent"] = partial(PPO.load, path.parent / path.stem)

    env = wrap(make("kamisado/Game-v0"), **kwargs)
    eval_env = wrap(make("kamisado/Game-v0"), **kwargs)

    if path.exists():
        model = PPO.load(path.parent / path.stem, env=env)
        print("Loaded model")
    else:
        from stable_baselines3.ppo import MlpPolicy

        model = PPO(MlpPolicy, env=env)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=50, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        best_model_save_path=path.parent,
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
