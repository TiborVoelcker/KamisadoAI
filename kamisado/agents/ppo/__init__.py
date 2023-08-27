"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 18.08.2023
"""
import time
from functools import partial
from pathlib import Path

from gymnasium import make
from sb3_contrib.ppo_mask import MaskablePPO as PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from kamisado.wrappers import wrap

path = Path("kamisado/agents/ppo/model")
file = path / "best_mode"


def train(timesteps, **kwargs):
    if kwargs.get("tournament_opponent") == "self":
        if not file.exists():
            raise UserWarning(
                "Cannot use `tournament_opponent` 'self', if model does not exist yet!"
            )
        kwargs["tournament_opponent"] = partial(PPO.load, file)

    env = wrap(make("kamisado/Game-v0"), **kwargs)
    eval_env = wrap(make("kamisado/Game-v0"), **kwargs)

    if file.exists():
        model = PPO.load(file, env=env)
    else:
        from sb3_contrib.ppo_mask import MlpPolicy

        model = PPO(MlpPolicy, env=env)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=50, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        best_model_save_path=path,
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
