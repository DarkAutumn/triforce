
import pygame
from .helpers import draw_text

class RewardButton:
    """A button to display a reward value."""
    def __init__(self, font, count, rewards, action, action_mask, width, on_click = None):
        self.font = font
        self.count = count
        self.rewards = rewards
        self.action = action
        self.width = width
        self.on_click = on_click
        self.action_mask = action_mask

    def draw_reward_button(self, surface, position) -> 'RenderedButton':
        """Draws the button on the surface. Returns a RenderedButton."""
        x = position[0] + 3
        y = position[1] + 2

        start_y = y
        y = draw_text(surface, self.font, self.action, (x, y))

        actions_allowed = ""
        for action, directions in self.action_mask:
            actions_allowed += action.name + ": " + "".join(x.name for x in directions) + ' '
        y = draw_text(surface, self.font, "Allowed: " + actions_allowed, (x, y))
        if self.rewards:
            for reason, value in self.rewards.items():
                color = (255, 0, 0) if value < 0 else (0, 255, 255) if value > 0 else (255, 255, 255)
                next_y = draw_text(surface, self.font, reason, (x, y), color=color)
                draw_text(surface, self.font, f"{'+' if value > 0 else ''}{value:.2f}", (x + 200, y))
                y = next_y

        else:
            text = "none"
            color = (128, 128, 128)
            y = draw_text(surface, self.font, text, (x, y), color)

        if self.count > 1:
            count_text = f"x{self.count}"
            count_text_width, _ = self.font.size(count_text)
            draw_text(surface, self.font, count_text, (x + 275 - count_text_width, start_y))

        height = y - position[1]
        pygame.draw.rect(surface, (255, 255, 255), (position[0], position[1], self.width, height), 1)
        return RenderedButton(self, position, (self.width, height))

class RenderedButton:
    """A rendered button on screen, keeping track of its own dimensions."""
    def __init__(self, button, position, dimensions):
        self.button = button
        self.position = position
        self.dimensions = dimensions

    def is_position_within(self, position):
        """Returns True if the position is within the button."""
        x, y = position
        bx, by = self.position
        bw, bh = self.dimensions
        return bx <= x <= bx + bw and by <= y <= by + bh
