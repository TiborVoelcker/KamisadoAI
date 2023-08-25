"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 09.08.2023
"""
from typing import TypedDict

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

ORANGE = (255, 153, 0)
BLUE = (0, 0, 255)
PURPLE = (153, 0, 204)
PINK = (255, 0, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BROWN = (102, 51, 0)

COLORS = [ORANGE, BLUE, PURPLE, PINK, YELLOW, RED, GREEN, BROWN]


def draw_tower(canvas, tower, coords, radius):
    player = (0, 0, 0) if tower > 0 else (255, 255, 255)
    tower = COLORS[abs(tower) - 1]
    points = np.array([[np.cos(a), np.sin(a)] for a in np.arange(0, 2 * np.pi, np.pi / 3)])
    points = points * radius + coords
    pygame.draw.polygon(canvas, player, points)
    pygame.draw.circle(canvas, tower, coords, radius * 0.6)


class Game(gym.Env):
    INVALID_ACTION_REWARD = -1000
    WINNING_REWARD = 50
    LOOSING_REWARD = -50
    ACTION_REWARD = 0

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
    relative_actions = np.concatenate(
        (
            # not moving
            np.array([[0, 0]]),
            # going left
            np.dstack((np.arange(-1, -8, -1), np.arange(-1, -8, -1)))[0],
            # going up
            np.dstack((np.arange(-1, -8, -1), np.zeros(7, dtype=int)))[0],
            # going right
            np.dstack((np.arange(-1, -8, -1), np.arange(1, 8)))[0],
        )
    )
    relative_actions.setflags(write=False)

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 1}

    def __init__(self, render_mode=None, size=5):
        self.window_size = 512  # The size of the PyGame window

        # Observations are the current board state and the next tower to move.
        # The board state is always seen from the side of the next player.
        # Own towers are numbers 1 to 8 corresponding colors Orange to Brown
        # (see COLORS), opponent's towers are number -8 to -1, empty squares
        # are 0. The tower to move is 0 if it's the start of the game.
        self.observation_space = spaces.MultiDiscrete([17] * 64 + [9])

        # Actions are dictionaries with the tower to move (1 to 8) and its target location.
        self.action_space = spaces.MultiDiscrete([22, 8])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.font = None

    def _get_obs(self):
        return np.append(self.board.flatten() + 8, self.current_tower if self.current_tower else 0)

    def _get_info(self):
        return {"current_player": self.current_player, "board": self.board}

    @property
    def board(self):
        """Current board state.

        The property flips the internal board so that the current player's
        starting line is always at the bottom, the player's towers are
        positive and the opponent's towers are negative.
        """
        # default board state is from black's perspective
        if self.current_player == 1:
            return np.flip(self._board * -1)
        # copy to not have unwanted mutation to internal state
        return self._board.copy()

    @board.setter
    def board(self, new_board: np.ndarray):
        if self.current_player == 1:
            self._board = np.flip(new_board * -1)
        else:
            # copy to not have unwanted mutation to internal state
            self._board = new_board.copy()

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initialize the board for
        self._board = np.array(
            [
                [-1, -2, -3, -4, -5, -6, -7, -8],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [8, 7, 6, 5, 4, 3, 2, 1],
            ],
            dtype=np.int64,
        )

        self.current_tower = None
        # black corresponds to 0, white corresponds to 1
        self.current_player = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def get_tower_coords(self, tower: int) -> np.ndarray:
        """Get the coordinates of a tower."""
        y, x = np.where(self.board == tower)
        return np.array([y[0], x[0]], dtype=np.int64)

    def tower_is_blocked(self, tower: int) -> bool:
        """Check if one tower is blocked and cannot move."""
        tower_coords = self.get_tower_coords(tower)
        test = np.array([[-1, -1], [-1, 0], [-1, 1]])
        if tower < 0:
            test *= -1
        test = test + tower_coords

        test = test[((test >= 0) & (test <= 7)).all(1)]

        return (self.board[tuple(test.T)] != 0).all()

    def action_masks(self):
        valid_tower = self.current_tower
        if valid_tower is None:
            # for the first move, the valid moves depend on the chosen tower
            # we just allow all possible first moves and hope the model
            # figures out the rest
            mask = np.ones(22 + 8, dtype=bool)
            mask[:22:7] = False
            return mask

        target_mask = self.target_mask(valid_tower)
        tower_mask = np.zeros(8, dtype=bool)
        tower_mask[valid_tower - 1] = True
        return np.append(target_mask, tower_mask)

    def target_mask(self, tower: int) -> np.ndarray:
        """Get a mask for valid relative actions."""
        if self.tower_is_blocked(tower):
            return np.append(True, np.zeros(21, dtype=bool))

        tower_coords = self.get_tower_coords(tower)
        if tower > 0:
            abs_actions = self.relative_actions + tower_coords
        else:
            abs_actions = self.relative_actions * -1 + tower_coords

        # set indexes outside of board invalid
        mask = ((abs_actions >= 0) & (abs_actions <= 7)).all(1)
        # set indexes with tower on them invalid
        mask[mask] = self.board[tuple(abs_actions[mask].T)] == 0
        # set all indexes after invalid indexes also invalid
        paths = mask[1:].reshape(3, 7)
        mask = np.invert(np.invert(paths).cumsum(1, dtype=bool))

        return np.append(False, mask.flatten())

    def valid_targets(self, tower: int) -> np.ndarray:
        """Get all possible actions for one tower."""
        return self.relative_actions[self.target_mask(tower)]

    def parse_action(self, action: np.ndarray) -> tuple[bool, int, np.ndarray]:
        """Parse an action.

        Args:
            action (Action): The action to be parsed.

        Returns:
            bool: Wether the action is valid.
            int: The tower to move.
            np.ndarray: The relative target to move to.
        """
        tower = int(action[1] + 1)
        target = self.relative_actions[int(action[0])]

        # check if tower selection is correct
        if not tower == self.current_tower and self.current_tower is not None:
            return False, tower, target
        # check if tower can move to the provided target
        valid_actions = self.valid_targets(tower)
        return (valid_actions == target).all(1).any(), tower, target

    def move_tower(self, tower: int, target: np.ndarray):
        """Move a tower to a relative target."""
        board = self.board
        coords = self.get_tower_coords(tower)
        board[*coords] = 0
        board[*coords + target] = tower
        self.board = board

    def color_at_coords(self, coords: list[int] | np.ndarray):
        """Get the board color at specified coordinates."""
        return self.board_colors[*coords]

    @property
    def is_won(self) -> bool:
        """Check whether the game is won by reaching the goal line.

        This function requires the `self.current_player` to already be pointed
        to the next player.
        """
        return any(self.board[7] < 0)

    @property
    def is_deadlocked(self) -> bool:
        """Check whether the game is deadlocked.

        This function requires the `self.current_tower` to already be pointed
        to the first tower that might be part of the deadlock.
        """
        if self.current_tower is None:
            return False
        pointer = self.current_tower
        while self.tower_is_blocked(pointer):
            pointer_coords = self.get_tower_coords(pointer)
            color = self.color_at_coords(pointer_coords)
            pointer = -color if pointer > 0 else color

            # check if loop is complete
            if pointer == self.current_tower:
                return True
        return False

    def step(self, action: np.ndarray):
        valid, tower, target = self.parse_action(action)
        if not valid:
            return self._get_obs(), self.INVALID_ACTION_REWARD, True, False, self._get_info()

        # move tower
        self.move_tower(tower, target)

        # set next tower and player
        self.current_tower = self.color_at_coords(self.get_tower_coords(tower))
        self.current_player = 1 if self.current_player == 0 else 0

        if self.render_mode == "human":
            self._render_frame()

        # check if game was won
        if self.is_won:
            return (
                self._get_obs(),
                self.WINNING_REWARD,
                False,
                True,
                self._get_info(),
            )

        # check if deadlocked
        if self.is_deadlocked:
            return (
                self._get_obs(),
                self.LOOSING_REWARD,
                False,
                True,
                self._get_info(),
            )

        return (
            self._get_obs(),
            self.ACTION_REWARD,
            False,
            False,
            self._get_info(),
        )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.font = pygame.font.SysFont(None, 72)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))

        # draw board
        SQUARE_SIZE = self.window_size / 8
        for y, x in np.ndindex(self._board.shape):
            color = self.board_colors[y, x]
            pygame.draw.rect(
                canvas,
                COLORS[color - 1],
                (x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
            )

        # draw towers
        for y, x in np.ndindex(self._board.shape):
            tower = self._board[y, x]
            if tower != 0:
                draw_tower(
                    canvas,
                    tower,
                    [(x + 0.5) * SQUARE_SIZE, (y + 0.5) * SQUARE_SIZE],
                    SQUARE_SIZE * 0.4,
                )

        # draw winner
        winner = None
        if self.is_won:
            # attention! current_player is already set to the next player
            # this is required for the `self.is_deadlocked` property
            winner = "White" if self.current_player == 0 else "Black"
        elif self.is_deadlocked:
            # attention! current_player is already set to the next player
            # this is required for the `self.is_deadlocked` property
            winner = "Black" if self.current_player == 0 else "White"

        if winner:
            img = self.font.render(f"{winner} wins", True, GREEN)
            rect = img.get_rect()
            canvas.blit(
                img, (self.window_size / 2 - rect[2] / 2, self.window_size / 2 - rect[3] / 2)
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    env = Game()

    from gymnasium.utils.env_checker import check_env

    check_env(env, skip_render_check=False)

    env = Game(render_mode="human")
    obs, info = env.reset()
    env.step(np.array([13, 3]))
    env.step(np.array([9, 1]))
    env.step(np.array([0, 3]))
    env.step(np.array([4, 1]))
    env.step(np.array([0, 3]))
    env.step(np.array([15, 1]))
    print("Done")
