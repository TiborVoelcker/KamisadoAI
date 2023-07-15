"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 14.07.2023
"""
import logging

import matplotlib.pyplot as plt
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from agents import *
from game import Kamisado


class Tournament:
    """Class to match two agents against each other.

    Props:
        game (Kamisado): The game to be played.
        players (dict[str, Agent]): The two agents that are playing.
    """

    def __init__(self, black: type[Agent], white: type[Agent], render=True):
        """Initializes the class.

        Args:
            black (type[Agent]): The agent to play black.
            white (type[Agent]): The agent to play white.
            render (bool, optional): Whether to visualize the game. Defaults to True.
        """
        self.game = Kamisado(render)

        self.players = {"black": black("black", self.game), "white": white("white", self.game)}

    def play(self, n=1000) -> float:
        """Play a bunch of games.

        Args:
            n (int, optional): The number of games to be played. Defaults to 1000.

        Returns:
            float: The fraction of games won by black.
        """
        black_won = 0
        with logging_redirect_tqdm():
            for _ in trange(n):
                self.play_game()
                if self.game.winner == "black":
                    black_won += 1
                self.game.reset()

        return black_won / n

    def play_game(self, sleep=0.001):
        """Play one game of Kamisado.

        Args:
            sleep (float, optional): How long to wait in between moves.
                Only has an effect if render is True. Defaults to 0.001.
        """
        next_tower = self.players["black"].choose_tower(self.game.towers["black"])

        next_player = "black"
        while not self.game.stopped:
            actions = self.game.get_actions(next_tower)
            action = self.players[next_player].choose_action(actions)
            next_player = "white" if next_player == "black" else "black"
            next_tower = self.game.step(next_tower, action)

            if self.game.render and sleep:
                plt.pause(sleep)

        logging.info("%s won by %s", self.game.winner.capitalize(), self.game.stopped)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO, format="%(message)s")
    test = Tournament(LookForWinAgent, RandomAgent, render=False)
    ratio = test.play()
    print(f"Black won {ratio:.0%}")
