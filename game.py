"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 13.07.2023
"""
from functools import partial
from itertools import product
from typing import overload

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, RegularPolygon
from stable_baselines3.common.env_checker import check_env

Hexagon = partial(RegularPolygon, numVertices=6, orientation=np.pi / 2)
COLORS = ["orange", "blue", "purple", "pink", "yellow", "red", "green", "brown"]


class Tower:
    """A piece in Kamisado.

    Props:
        player: The player the tower belongs to. Either "black" or "white".
        color: The color of the tower.
        game: An instance of 'Kamisado' the tower belongs to.
        xy: The coordinates where to tower currently is on the board.
    """

    def __init__(self, player: str, color: str, game: "Kamisado"):
        """Initializes the class."""
        self.player = player
        self.color = color
        self.game = game

        self.__xy = (0, 0)
        self._patches = []
        self.reset()

    def __repr__(self) -> str:
        return f"<Tower {self.player}|{self.color} {self.xy}>"

    def reset(self):
        """Resets the tower to its starting position."""
        if self.player == "black":
            y = 0
            x = COLORS.index(self.color)
        else:
            y = 7
            x = 7 - COLORS.index(self.color)

        self.xy = (x, y)

    @property
    def xy(self) -> tuple[int, int]:
        """The coordinates where to tower currently is on the board."""
        return self.__xy

    @xy.setter
    def xy(self, value: tuple[int, int]):
        """Set the tower's coordinates. Also renders its new position."""
        self.__xy = value
        if self.game.render_mode == "human":
            self.render()

    @property
    def is_winning(self):
        """True, if the tower is on the starting linen of its opponent."""
        return (self.xy[1] == 7 and self.player == "black") or (
            self.xy[1] == 0 and self.player == "white"
        )

    def render(self):
        """Render its (new) position."""
        if self._patches:
            self._patches[0].remove()
            self._patches[1].remove()
        ax = plt.gca()
        hex = ax.add_patch(Hexagon(self.xy, radius=0.4, color=self.player))
        circ = ax.add_patch(Circle(self.xy, radius=0.2, color=self.color))
        self._patches = [hex, circ]
        plt.draw()


class TowerTuple(tuple[Tower, ...]):
    """An tuple of towers.

    The tuple can be used as-is, but can also be indexed by the player, the color or both.

    Example:
        To get the towers of the black player, use `tower_tuple["black"]`.
        To get both orange towers, use `tower_tuple["orange"]`.
        To get the orange tower of the black player, use `tower_tuple["black", "orange"]`.
    """

    def __new__(cls, game: "Kamisado"):
        """Instantiate the class."""
        towers = []
        for color in COLORS:
            for player in ["black", "white"]:
                t = Tower(player=player, color=color, game=game)
                towers.append(t)

        cls.game = game
        return super().__new__(TowerTuple, towers)

    @overload
    def __getitem__(self, key: tuple[str, str]) -> Tower:
        ...

    @overload
    def __getitem__(self, key: str) -> tuple[Tower, ...]:
        ...

    @overload
    def __getitem__(self, key: tuple[int, int]) -> Tower | None:
        ...

    def __getitem__(
        self, key: tuple[str, str] | tuple[int, int] | str
    ) -> Tower | tuple[Tower, ...] | None:
        """Get the towers of one player, of one color, or one specific tower."""
        if isinstance(key, tuple | np.ndarray):
            # select specific tower by player and color
            if key[0] in ["black", "white"] and key[1] in COLORS:
                for t in self:
                    if t.player == key[0] and t.color == key[1]:
                        return t
            # select specific tower by coordinate
            for t in self:
                if np.array_equal(t.xy, key):
                    return t
            return None
        else:
            if key in ["black", "white"]:
                return tuple(t for t in self if t.player == key)
            if key in COLORS:
                return tuple(t for t in self if t.color == key)

        raise KeyError(f"Key must be a player or color or both, not '{key}'")

    def render_all(self):
        if self.game.render_mode == "human":
            for t in self:
                t.render()

        board = np.zeros((8, 8))
        for tower in self:
            num = COLORS.index(tower.color) + 1
            num += 8 if tower.player == "white" else 0
            board[*np.flip(tower.xy)] = num

        return board


class Kamisado(gym.Env):
    """The game Kamisado.

    Props:
        board_colors: The colors of the board.
        render: Whether to render the board and the towers or not.
        winner: The winner of the game, or "".
        stopped: The reason why the game stopped, or `False`.
    """

    INVALID_ACTION_REWARD = -100
    board_colors = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [5, 0, 3, 6, 1, 4, 7, 2],
        [6, 3, 0, 5, 2, 7, 4, 1],
        [3, 2, 1, 0, 7, 6, 5, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [1, 4, 7, 2, 5, 0, 3, 6],
        [2, 7, 4, 1, 6, 3, 0, 5],
        [7, 6, 5, 4, 3, 2, 1, 0],
    ]
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        """Initialize the class.

        Args:
            render (bool, optional): Whether to render the board and the
                towers or not. Defaults to True.
        """
        self.observation_space = spaces.MultiDiscrete([2] + [17] * 64)
        # self.action_space = spaces.Box(0, 7, shape=(2,), dtype=int)
        self.action_space = spaces.MultiDiscrete([2] * 64)
        self.possible_actions = list(product(range(8), range(8)))

        self.__render_mode = None
        self.towers = TowerTuple(self)

        self.render_mode = render_mode
        self.reset()

    def close(self):
        plt.close("all")

    @property
    def render_mode(self):
        return self.__render_mode

    @render_mode.setter
    def render_mode(self, mode):
        if mode is None or mode in self.metadata["render_modes"]:
            self.__render_mode = mode
            if mode == "human":
                self.render_board()
                self.towers.render_all()
        else:
            raise ValueError(
                f"Render mode must be one of the following: '{self.metadata['render_modes']}'\nNot {mode}"
            )

    def _get_obs(self):
        started = 0 if self.next_tower is None else 1
        board = self.board
        board[board < 0] += 17
        return np.concatenate(([started], board.flatten()))

    def _get_info(self):
        return {}

    def reset(self, seed=None):
        """Reset the game state."""
        for tower in self.towers:
            tower.reset()
        self.next_tower = None
        self.winner = ""
        self.stopped = False

        return self._get_obs(), self._get_info()

    def render_board(self):
        """Render the board."""
        plt.matshow(self.board_colors, cmap=ListedColormap(COLORS))
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.show(block=False)

    def __check_actions(self, step, initial):
        actions = []
        x, y = step(*initial)
        # check if leaving board
        while 0 <= x <= 7 and 0 <= y <= 7:
            # other tower in the way
            if self.towers[x, y] is not None:
                break
            actions.append((x, y))

            x, y = step(x, y)

        return actions

    @property
    def board(self):
        board = np.zeros((8, 8), dtype=int)
        for tower in self.towers:
            num = COLORS.index(tower.color) + 1
            player = -1 if tower.player == "black" else 1
            board[*np.flip(tower.xy)] = num * player

        return board

    def valid_actions(self, tower: Tower | None) -> list[tuple[int, int]]:
        """Get all possible actions for one tower."""
        if tower is None:
            return np.dstack((np.arange(8), np.zeros(8)))[0]

        if self.tower_is_blocked(tower):
            return [tower.xy]

        actions = []
        dy = +1 if tower.player == "black" else -1

        # going straight
        actions += self.__check_actions(lambda x, y: (x, y + dy), tower.xy)
        # going left
        actions += self.__check_actions(lambda x, y: (x - 1, y + dy), tower.xy)
        # going right
        actions += self.__check_actions(lambda x, y: (x + 1, y + dy), tower.xy)

        return actions

    def tower_is_blocked(self, tower: Tower) -> bool:
        """Check if one tower is blocked and cannot move."""
        x, y = tower.xy
        dy = +1 if tower.player == "black" else -1
        # nothing in front?
        if self.towers[x, y + dy] is None:
            return False
        # nothing to the right?
        if self.towers[x + 1, y + dy] is None and x + 1 <= 7:
            return False
        # nothing to the left?
        if self.towers[x - 1, y + dy] is None and x - 1 >= 0:
            return False
        return True

    def color_below_tower(self, tower: Tower) -> str:
        """Get the color the tower is standing on."""
        x, y = tower.xy
        color_idx = self.board_colors[y][x]
        return COLORS[color_idx]

    def action_is_valid(self, tower, action):
        return any(np.equal(self.valid_actions(tower), action).all(1))

    def action_masks(self):
        mask = np.zeros((8, 8, 2), dtype=bool)
        mask[:, :, 0] = True
        for action in self.valid_actions(self.next_tower):
            mask[int(action[1]), int(action[0]), :] = [False, True]

        return mask.flatten()

    def convert_action(self, action):
        return np.flip(np.argwhere(action.reshape((8, 8)))[0])

    def step(self, action: tuple[int, int]):
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

        action = self.convert_action(action)

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
            self.next_tower = self.towers[action]
            return (
                self._get_obs(),
                0,
                False,
                False,
                self._get_info(),
            )

        # move tower
        tower.xy = action

        if self.render_mode == "human":
            plt.pause(1 / self.metadata["render_fps"])

        # check if game was won
        if tower.is_winning:
            self.winner = tower.player
            self.stopped = "reaching baseline"
            return (
                self._get_obs(),
                1,
                False,
                True,
                self._get_info(),
            )

        # get next tower
        next_player = "black" if tower.player == "white" else "white"
        self.next_tower = self.towers[next_player, self.color_below_tower(tower)]

        # check for deadlock
        if self.tower_is_blocked(tower) and self.tower_is_blocked(self.next_tower):
            self.winner = "black" if tower.player == "white" else "white"
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
    action = np.zeros((8, 8), dtype=int)
    action[0, 7] = 1
    env.step(action.flatten())
    action = np.zeros((8, 8), dtype=int)
    action[1, 7] = 1
    env.step(action.flatten())
    action = np.zeros((8, 8), dtype=int)
    action[7, 7] = 1
    env.step(action.flatten())
    action = np.zeros((8, 8), dtype=int)
    action[0, 0] = 1
    env.step(action.flatten())
