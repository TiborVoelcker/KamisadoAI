# from gymnasium import make
# from stable_baselines3.ppo import PPO

# from kamisado.agents.ppo import path
# from kamisado.agents.simple import LookForWinAgent
# from kamisado.wrappers import wrap

# env = wrap(
#     make("kamisado/Game-v0", render_mode="human"),
#     tournament=True,
#     tournament_opponent=LookForWinAgent,
#     tower_selection=True,
#     reward_action=False,
# )

# model = PPO.load(path.parent / path.stem, env=env)

# # watch the trained agent
# truncated, terminated = False, False
# obs, info = env.reset()
# print("Model's color: ", "black" if info["agent_color"] == 0 else "white")

# while not truncated and not terminated:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, truncated, terminated, info = env.step(action)
#     print("Reward: ", reward)

from kamisado.agents.ppo import train
from kamisado.agents.simple import LookForWinAgent

if __name__ == "__main__":
    train(
        1000000,
        tournament=True,
        tournament_opponent=LookForWinAgent,
        tower_selection=True,
        reward_action=False,
    )
