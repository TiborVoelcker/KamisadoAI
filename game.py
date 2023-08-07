"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 13.07.2023
"""
from functools import partial

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, RegularPolygon
from stable_baselines3.common.env_checker import check_env

Hexagon = partial(RegularPolygon, numVertices=6, orientation=np.pi / 2)
COLORS = ["orange", "blue", "purple", "pink", "yellow", "red", "green", "brown"]


class Kamisado(gym.Env):
    """The game Kamisado."""

    INVALID_ACTION_REWARD = -100
    board_colors = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [6, 1, 4, 7, 2, 5, 8, 3],
            [7, 4, 1, 6, 3, 8, 5, 2],
            [4, 3, 2, 1, 8, 7, 6, 5],
            [5, 6, 7, 8, 1, 2, 3, 4],
            [2, 5, 8, 3, 6, 1, 4, 7],
            [3, 8, 5, 2, 7, 4, 1, 6],
            [8, 7, 6, 5, 4, 3, 2, 1],
        ]
    )
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None):
        """Initialize the class."""
        self.observation_space = spaces.MultiDiscrete([2] + [17] * 64)
        self.action_space = spaces.Box(0, 7, shape=(2,), dtype=int)

        self.render_mode = render_mode
        self.reset()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset the game state."""
        super().reset(seed=seed)
        self.board = np.array(
            [
                [-1, -2, -3, -4, -5, -6, -7, -8],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [8, 7, 6, 5, 4, 3, 2, 1],
            ]
        )
        self.next_tower = None
        self.winner = ""
        self.stopped = False

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def close(self):
        plt.close("all")

    def _get_obs(self) -> np.ndarray:
        started = 0 if self.next_tower is None else 1
        board = self.board.copy()
        board[board < 0] += 17
        return np.concatenate(([started], board.flatten()))

    def _get_info(self) -> dict:
        return {}

    def render(self):
        """Render its (new) position."""
        ax = plt.gca()
        ax.clear()
        plt.imshow(
            self.board_colors, cmap=ListedColormap(COLORS), interpolation="nearest", origin="upper"
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for tower_coords in np.ndindex(self.board.shape):
            tower = self.board[tower_coords]
            if tower == 0:
                continue
            player = "black" if tower < 0 else "white"
            color = COLORS[abs(tower) - 1]
            ax.add_patch(Hexagon(np.flip(tower_coords), radius=0.4, color=player))
            ax.add_patch(Circle(np.flip(tower_coords), radius=0.2, color=color))

        plt.show(block=False)

        plt.pause(1 / self.metadata["render_fps"])

    def get_tower_coords(self, tower: int) -> np.ndarray:
        return np.array(np.where(self.board == tower)).reshape(1, 2)[0]

    def __check_actions(self, step, initial):
        actions = []
        y, x = step(*initial)
        # check if leaving board
        while 0 <= y <= 7 and 0 <= x <= 7:
            # other tower in the way
            if self.board[y, x] != 0:
                break

            actions.append(np.append(y, x))

            y, x = step(y, x)

        return actions

    def valid_actions(self, tower: int | None) -> np.ndarray:
        """Get all possible actions for one tower."""
        if tower is None:
            return np.dstack((np.zeros(8), np.arange(8)))[0]

        tower_coords = self.get_tower_coords(tower)

        if self.tower_is_blocked(tower):
            return tower_coords.reshape(1, 2)

        actions = []
        dy = +1 if tower < 0 else -1

        # going straight
        actions += self.__check_actions(lambda y, x: (y + dy, x), tower_coords)
        # going left
        actions += self.__check_actions(lambda y, x: (y + dy, x - 1), tower_coords)
        # going right
        actions += self.__check_actions(lambda y, x: (y + dy, x + 1), tower_coords)

        return np.array(actions)

    def tower_is_blocked(self, tower: int) -> bool:
        """Check if one tower is blocked and cannot move."""
        y, x = self.get_tower_coords(tower)
        dy = +1 if tower < 0 else -1
        # nothing in front?
        if self.board[y + dy, x] == 0:
            return False
        # nothing to the right?
        if x + 1 <= 7 and self.board[y + dy, x + 1] == 0:
            return False
        # nothing to the left?
        if x - 1 >= 0 and self.board[y + dy, x - 1] == 0:
            return False
        return True

    def do_action(self, tower: int, action: np.ndarray):
        """Perform an action with a tower.

        The action is not checked and could break the game.
        """
        self.board[*self.get_tower_coords(tower)] = 0
        self.board[*action.astype(int)] = tower

    def color_below_tower(self, tower_coords: np.ndarray) -> int:
        """Get the color the tower is standing on."""
        return self.board_colors[*tower_coords.astype(int)]

    def action_is_valid(self, tower: int | None, action: np.ndarray) -> bool:
        return any(np.equal(self.valid_actions(tower), action).all(1))

    def tower_is_winning(self, tower: int) -> bool:
        """True, if the tower is on the starting line of its opponent."""
        y = self.get_tower_coords(tower)[0]
        return (tower < 0 and y == 7) or (tower > 0 and y == 0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, int, bool, bool, dict]:
        """Play one step in the game.

        Args:
            tower (Tower): The tower to move.
            action (tuple[int, int]): The coordinate to move to.

        Returns:
            Tower: The next tower to move.
        """
        # check if game already stopped
        if self.stopped:
            raise RuntimeError("Game already stopped. Use `Kamisado.reset()` to reset the game.")

        tower = self.next_tower

        # validate action
        if not self.action_is_valid(tower, action):
            return (
                self._get_obs(),
                self.INVALID_ACTION_REWARD,
                False,
                True,
                self._get_info(),
            )

        if tower is None:
            self.next_tower = self.color_below_tower(action) * -1
            return (
                self._get_obs(),
                0,
                False,
                False,
                self._get_info(),
            )

        # move tower
        self.do_action(tower, action)

        if self.render_mode == "human":
            self.render()

        # check if game was won
        if self.tower_is_winning(tower):
            self.winner = "black" if tower < 0 else "white"
            self.stopped = "reaching baseline"
            return (
                self._get_obs(),
                1,
                False,
                True,
                self._get_info(),
            )

        # get next tower
        next_player = 1 if tower < 0 else -1
        self.next_tower = self.color_below_tower(action) * next_player

        # check for deadlock
        if self.tower_is_blocked(tower) and self.tower_is_blocked(self.next_tower):
            self.winner = "white" if tower < 0 else "black"
            self.stopped = "deadlock"
            return (
                self._get_obs(),
                -1,
                False,
                True,
                self._get_info(),
            )

        return (
            self._get_obs(),
            0,
            False,
            False,
            self._get_info(),
        )


if __name__ == "__main__":
    env = Kamisado()
    check_env(env, skip_render_check=False)

    env.render_mode = "human"
    env.reset()
    env.step(np.array((1, 0)))  # invalid
    env.step(np.array((0, 7)))
    env.step(np.array((0, 0)))  # invalid
    env.step(np.array((1, 7)))
    env.step(np.array((5, 5)))
