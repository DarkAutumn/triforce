import math
import numpy as np
import pygame

from triforce.zelda_enums import Coordinates, Direction
from .helpers import render_text

class LabeledCircle:
    """A circle with a label."""
    def __init__(self, position, font, label, radius=128, color=(255, 0, 0), width=5):
        assert isinstance(position, Coordinates)
        self.position = position
        self.radius = int(radius)
        self.color = color
        self.width = width
        self.font = font
        self.label = label

    @property
    def size(self):
        """Returns the size of the draw area."""
        return self.radius * 2, self.radius * 2 + 20

    @property
    def centerpoint(self):
        """Returns the centerpoint of the circle."""
        circle_start = self.position + (0, 20)
        centerpoint = circle_start + (self.radius, self.radius)
        return centerpoint

    def draw(self, surface):
        """Draws the labeled circle on the surface."""
        render_text(surface, self.font, self.label, self.position)
        pygame.draw.circle(surface, (255, 255, 255), self.centerpoint, self.radius, 1)

    def _draw_arrow(self, surface, centerpoint, vector, length):
        length = np.clip(length, 0.05, 1)
        arrow_end = np.array(centerpoint) + vector[:2] * self.radius * length
        if vector[0] != 0 or vector[1] != 0:
            pygame.draw.line(surface, self.color, centerpoint, arrow_end, self.width)

            # Arrowhead
            arrowhead_size = 10
            angle = math.atan2(-vector[1], vector[0]) + math.pi

            left = arrow_end + (arrowhead_size * math.cos(angle - math.pi / 6),
                              -arrowhead_size * math.sin(angle - math.pi / 6))
            right = arrow_end + (arrowhead_size * math.cos(angle + math.pi / 6),
                               -arrowhead_size * math.sin(angle + math.pi / 6))

            pygame.draw.polygon(surface, self.color, [arrow_end, left, right])

class DirectionalCircle(LabeledCircle):
    """A vector with a label."""
    def __init__(self, position, font, label, radius=128, color=(255, 0, 0), width=5, center_circle=True, scale_range=(0, 1)):
        super().__init__(position, font, label, radius, color, width)
        self._directions = []
        self._center_circle = center_circle
        self._scale_range = scale_range

    @property
    def directions(self):
        """Returns the directions."""
        return self._directions

    @directions.setter
    def directions(self, value):
        self._directions = value

    def draw(self, surface):
        """Draws the labeled vector on the surface."""
        super().draw(surface)

        for direction in self.directions:
            if isinstance(direction, Direction):
                scale = 1
            else:
                direction, scale = direction

            scale = self._scale_range[0] + scale * (self._scale_range[1] - self._scale_range[0])
            scale = np.clip(scale, 0, 1)

            match direction:
                case Direction.N:
                    self._draw_arrow(surface, self.centerpoint, np.array([0, -1], dtype=np.float32), scale)
                case Direction.S:
                    self._draw_arrow(surface, self.centerpoint, np.array([0, 1], dtype=np.float32), scale)
                case Direction.W:
                    self._draw_arrow(surface, self.centerpoint, np.array([-1, 0], dtype=np.float32), scale)
                case Direction.E:
                    self._draw_arrow(surface, self.centerpoint, np.array([1, 0], dtype=np.float32), scale)
                case _:
                    raise ValueError(f"Unsupported direction {direction}")


        if self.directions and self._center_circle:
            pygame.draw.circle(surface, (0, 0, 0), self.centerpoint, 5)

class LabeledVector(LabeledCircle):
    """A vector with a label."""
    def __init__(self, position, font, label, radius=128, color=(255, 0, 0), width=5):
        super().__init__(position, font, label, radius, color, width)
        self._vector = [0, 0, -1]
        self._scale = 1

    @property
    def vector(self):
        """Returns the vector."""
        return self._vector

    @vector.setter
    def vector(self, value):
        assert len(value) == 2
        self._vector = value

    @property
    def scale(self):
        """Returns the scale of the vector."""
        return self._scale

    @scale.setter
    def scale(self, value):
        assert 0 <= value <= 1
        self._scale = value

    def draw(self, surface):
        """Draws the labeled vector on the surface."""
        super().draw(surface)
        self._draw_arrow(surface, self.centerpoint, self._vector, self._scale)
