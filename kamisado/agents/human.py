"""
  Copyright (c) Tibor Völcker <tibor.voelcker@hotmail.de>
  Created on 29.08.2023
"""
import numpy as np
import pygame

from kamisado.agents import Model


class HumanAgent(Model):
    """An agent that plays randomly."""

    def get_mouse_click_pos(self):
        click, _, _ = pygame.mouse.get_pressed()
        if click:
            pos = np.array(pygame.mouse.get_pos())
            pos = pos / self.env.window_size * 8
            pos = np.flip(pos).astype(int)

            if self.env.current_player == 1:
                pos = abs(pos - [7, 7])

            return pos

        return None

    def predict(self, obs, **kwargs):
        tower = obs[-1]
        while tower == 0:
            pos = self.get_mouse_click_pos()
            if pos is not None:
                selected = self.env.board[pos[0], pos[1]]
                if selected > 0:
                    tower = selected

            pygame.event.pump()
            self.env.clock.tick(100)

        print("Selected Tower: ", tower)

        target = None
        while target is None:
            pos = self.get_mouse_click_pos()
            if pos is not None:
                relative = pos - self.env.get_tower_coords(tower)
                selected = (self.env.relative_actions == relative).all(1)
                selected = selected & self.env.target_mask(tower)
                if selected.any():
                    target = selected.nonzero()[0][0]

            pygame.event.pump()
            self.env.clock.tick(100)

        print("Target: ", target)

        return np.append(tower - 1, target), None
