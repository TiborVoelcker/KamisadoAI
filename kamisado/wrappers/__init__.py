"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 09.08.2023
"""
import numpy as np
from gymnasium import ActionWrapper, ObservationWrapper, spaces


def wrap(env, tower_selection=True):
    env = RelativeAction(env)
    if not tower_selection:
        env = NoTowerSelection(env)
    return FlattenAction(FlattenObservation(env))


class FlattenObservation(ObservationWrapper):
    """Wrapper to flatten the action space.

    It will only flatten the dictionary, and not make a one-hot vector out of the
    discrete spaces.
    This wrapper only works if the environment is also wrapped in the
    `RelativeAction` wrapper.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.MultiDiscrete([17] * 64 + [9])

    def observation(self, obs):
        return np.append(obs["board"].flatten() + 8, obs["tower"])


class FlattenAction(ActionWrapper):
    """Wrapper to flatten the action space.

    It will only flatten the dictionary, and not make a one-hot vector out of the
    discrete spaces.
    This wrapper only works if the environment is also wrapped in the
    `RelativeAction` wrapper.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete([22, 8])

    def action(self, action):
        return {
            "target": action[0],
            "tower": action[1] + 1,
        }


class RelativeAction(ActionWrapper):
    """Wrapper to use relative actions for tower movement.

    The relative action is selected as an discrete index referencing the action.
    """

    def __init__(self, env):
        super().__init__(env)
        self.int_to_relative = np.array(
            [
                [0, 0],
                [-1, 0],
                [-2, 0],
                [-3, 0],
                [-4, 0],
                [-5, 0],
                [-6, 0],
                [-7, 0],
                [-1, -1],
                [-2, -2],
                [-3, -3],
                [-4, -4],
                [-5, -5],
                [-6, -6],
                [-7, -7],
                [-1, 1],
                [-2, 2],
                [-3, 3],
                [-4, 4],
                [-5, 5],
                [-6, 6],
                [-7, 7],
            ]
        )
        self.action_space = spaces.Dict(
            {
                "tower": spaces.Discrete(8, start=1),
                "target": spaces.Discrete(21),
            }
        )

    def action(self, action):
        target = self.env.get_tower_coords(action["tower"])
        target = target + self.int_to_relative[action["target"]]
        return {"target": target, "tower": action["tower"]}


class NoTowerSelection(ActionWrapper):
    """Wrapper that automatically selects the correct tower.

    Changing the tower must happen before resolving the relative actions
    (thus `NoTowerSelection` must wrap the `RelativeAction` wrapper).
    """

    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        tower = self.env.current_tower
        if tower is None:
            tower = self.env.np_random.integers(1, 9)
        return {"target": action["target"], "tower": tower}
