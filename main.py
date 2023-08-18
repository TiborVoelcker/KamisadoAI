# from sb3_contrib.ppo_mask import MaskablePPO as PPO

# from kamisado.agents.ppo import make_env

# env = make_env(render_mode="human")
# model = PPO.load("kamisado/agents/ppo/model_masked", env=env)

# # watch the trained agent
# truncated, terminated = False, False
# obs, info = env.reset()
# while not truncated and not terminated:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, truncated, terminated, info = env.step(action)

from kamisado.agents.ppo import train

train(100000, mask=False, reward_action=False)
