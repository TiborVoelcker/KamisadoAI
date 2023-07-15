"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 13.07.2023
"""
from functools import partial
from typing import overload

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, RegularPolygon

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
        if self.game.render:
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
        return super().__new__(TowerTuple, towers)

    @overload
    def __getitem__(self, key: tuple[str, str]) -> Tower:
        ...

    @overload
    def __getitem__(self, key: str) -> tuple[Tower, ...]:
        ...

    def __getitem__(self, key: tuple[str, str] | str) -> Tower | tuple[Tower, ...]:
        """Get the towers of one player, of one color, or one specific tower."""
        if isinstance(key, tuple) and key[0] in ["black", "white"] and key[1] in COLORS:
            for t in self:
                if t.player == key[0] and t.color == key[1]:
                    return t
        if key in ["black", "white"]:
            return tuple(t for t in self if t.player == key)
        if key in COLORS:
            return tuple(t for t in self if t.color == key)

        raise KeyError(f"Key must be a player or color or both, not '{key}'")


class Kamisado:
    """The game Kamisado.

    Props:
        board_colors: The colors of the board.
        render: Whether to render the board and the towers or not.
        winner: The winner of the game, or "".
        stopped: The reason why the game stopped, or `False`.
    """

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

    def __init__(self, render=True):
        """Initialize the class.

        Args:
            render (bool, optional): Whether to render the board and the
                towers or not. Defaults to True.
        """
        self.render = render
        self.winner = ""
        self.stopped = False
        if self.render:
            self.render_board()

        self.towers = TowerTuple(self)

    def reset(self):
        """Reset the game state."""
        for tower in self.towers:
            tower.reset()
        self.winner = ""
        self.stopped = False

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
            if any(t.xy == (x, y) for t in self.towers):
                break
            actions.append((x, y))

            x, y = step(x, y)

        return actions

    def get_actions(self, tower: Tower) -> list[tuple[int, int]]:
        """Get all possible actions for one tower."""
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
        if all(t.xy != (x, y + dy) for t in self.towers):
            return False
        # nothing to the right?
        if x + 1 <= 7 and all(t.xy != (x + 1, y + dy) for t in self.towers):
            return False
        # nothing to the left?
        if x - 1 >= 0 and all(t.xy != (x - 1, y + dy) for t in self.towers):
            return False
        return True

    def color_below_tower(self, tower: Tower) -> str:
        """Get the color the tower is standing on."""
        x, y = tower.xy
        color_idx = self.board_colors[y][x]
        return COLORS[color_idx]

    def step(self, tower: Tower, action: tuple[int, int]) -> Tower:
        """Play one step in the game.

        Args:
            tower (Tower): The tower to move.
            action (tuple[int, int]): The coordinate to move to.

        Returns:
            Tower: The next tower to move.
        """
        # move tower
        tower.xy = action
        # check if game was won
        if tower.is_winning:
            self.winner = tower.player
            self.stopped = "reaching baseline"
        # get next tower
        next_player = "black" if tower.player == "white" else "white"
        next_tower = self.towers[next_player, self.color_below_tower(tower)]
        # check for deadlock
        if self.tower_is_blocked(tower) and self.tower_is_blocked(next_tower):
            self.winner = "black" if tower.player == "white" else "white"
            self.stopped = "deadlock"

        return next_tower


if __name__ == "__main__":
    game = Kamisado()

    plt.pause(0)
