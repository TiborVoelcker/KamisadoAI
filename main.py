from stable_baselines3 import PPO

from kamisado.agents.ppo import make_env

env = make_env(render_mode="human")
model = PPO.load("kamisado/agents/ppo/model", env=env)

# watch the trained agent
truncated, terminated = False, False
obs, info = env.reset()
cum_reward = 0
while not truncated and not terminated:
    action, _ = model.predict(obs)
    obs, reward, truncated, terminated, info = env.step(action)
    cum_reward += reward
print(f"Cumulative reward for the episode:{cum_reward:.2f}")
