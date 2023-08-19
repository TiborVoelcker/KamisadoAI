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

from functools import partial

from stable_baselines3.ppo import PPO

from kamisado.agents.ppo import train
from kamisado.agents.simple import LookForWinAgent

model = partial(PPO.load, "kamisado/agents/ppo/model")

train(
    1000,
    mask=False,
    tournament=True,
    tournament_opponent=LookForWinAgent,
    tower_selection=False,
    reward_action=False,
)
