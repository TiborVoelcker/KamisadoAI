"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 18.08.2023
"""
import numpy as np
from gymnasium import ActionWrapper, ObservationWrapper, spaces


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
