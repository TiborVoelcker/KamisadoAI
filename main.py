# from functools import partial

# from gymnasium import make
# from sb3_contrib.ppo_mask import MaskablePPO as PPO

# from kamisado.wrappers import wrap

# model = partial(PPO.load, "kamisado/agents/ppo/model/best_model")
# env = make("kamisado/Game-v0", render_mode="human")
# env = wrap(
#     env, tournament=True, tournament_opponent=model, tower_selection=False, reward_action=False
# )
# model = PPO.load("kamisado/agents/ppo/model/best_model", env=env)

# # watch the trained agent
# truncated, terminated = False, False
# obs, info = env.reset()
# while not truncated and not terminated:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, truncated, terminated, info = env.step(action)

from functools import partial

from stable_baselines3.ppo import PPO

from kamisado.agents.ppo import train

model = partial(PPO.load, "kamisado/agents/ppo/model/best_model")


if __name__ == "__main__":
    train(
        100000,
        mask=False,
        tournament=True,
        tournament_opponent=model,
        tower_selection=False,
        reward_action=False,
    )
