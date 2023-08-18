from stable_baselines3 import PPO

from kamisado.agents.ppo import make_env

env = make_env(render_mode="human")
model = PPO.load("kamisado/agents/ppo/model", env=env)

# watch the trained agent
truncated, terminated = False, False
obs, info = env.reset()
while not truncated and not terminated:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, truncated, terminated, info = env.step(action)
