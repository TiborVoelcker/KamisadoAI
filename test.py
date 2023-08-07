import os
import time
import uuid

from sb3_contrib.ppo_mask.policies import MlpPolicy
from sb3_contrib.ppo_mask.ppo_mask import MaskablePPO as PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results

from game import Kamisado

# Start with a model
env = Kamisado()
check_env(env)
# model = PPO(
#     MlpPolicy,
#     env,
#     learning_rate=0.0003,
#     n_steps=2048,
#     batch_size=64,
#     n_epochs=10,
#     gamma=0.99,
#     verbose=0,
#     policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
# )
model = PPO(MlpPolicy, env, verbose=0)

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

model.save("masked_model")

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
