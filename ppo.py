import time

from gymnasium import make
from gymnasium.wrappers.flatten_observation import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.policies import MlpPolicy

import kamisado
from kamisado.wrappers import FlattenAction, NoTowerSelection

# Start with a model
env = make("kamisado/Game-v0")
env = Monitor(FlattenObservation(NoTowerSelection(env)))
check_env(env)
model = PPO.load("model", env=env)

# Train the agent
timesteps = 1000000
start_time = time.time()
model.learn(
    total_timesteps=timesteps,
    callback=EvalCallback(env, eval_freq=timesteps // 50, n_eval_episodes=10),
    reset_num_timesteps=False,
)
print(f"Training took {time.time() - start_time:.2f}s")

model.save("model")

# Evaluate the trained agent
env = make("kamisado/Game-v0")
env = FlattenObservation(FlattenAction(env))

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

env = make("kamisado/Game-v0", render_mode="human")
env = FlattenObservation(FlattenAction(env))

# watch the trained agent
truncated, terminated = False, False
obs, info = env.reset()
cum_reward = 0
while not truncated and not terminated:
    action, _ = model.predict(obs)
    obs, reward, truncated, terminated, info = env.step(action)
    cum_reward += reward
print(f"Cumulative reward for the episode:{cum_reward:.2f}")
