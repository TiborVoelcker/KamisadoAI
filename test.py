import os
import time
import uuid

from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.ppo.policies import MlpPolicy

from agents import LookForWinAgent
from tournament import Tournament

# Start with a model
env = Tournament(LookForWinAgent)
check_env(env)
model = PPO(MlpPolicy, env, verbose=0, gamma=1)

# Create log dir for monitoring the results during training and prepare monitoring and logging the training results
# log_dir = "tmp/" + str(uuid.uuid4())
# os.makedirs(log_dir, exist_ok=True)
# env = Monitor(env, log_dir)

# Train the agent
timesteps = 10000
start_time = time.time()
model.learn(
    total_timesteps=timesteps,
    callback=EvalCallback(env, eval_freq=100, n_eval_episodes=10),
    reset_num_timesteps=False,
)
print(f"Training took {time.time() - start_time:.2f}s")

model.save("model")

# Plot the training curve
# not showing!
# plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "Test")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# watch the trained agent
env.render_mode = "human"
truncated, terminated = False, False
obs, info = env.reset()
cum_reward = 0
while not truncated and not terminated:
    action, _ = model.predict(obs)
    obs, reward, truncated, terminated, info = env.step(action)
    cum_reward += reward
print(f"Cumulative reward for the episode:{cum_reward:.2f}")
