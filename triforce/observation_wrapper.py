"""
For this project, we don't neccessarily need to see the whole screen, or even the game in color.
The ZeldaObservationWrapper takes care of this by letting us (optionally) trim off the HUD and
convert the image to grayscale.  We also stack multiple frames together to give the agent a sense
of motion over time.
"""

from typing import Sequence
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
import torch
import torch.nn.functional as F

from .zelda_enums import GAMEPLAY_START_Y, Direction
from .objectives import Objective, ObjectiveKind
from .zelda_objects import ZeldaObject
from .zelda_game import ZeldaGame

GRAYSCALE_WEIGHTS = torch.FloatTensor([0.2989, 0.5870, 0.1140])
OBJECT_KINDS = 3
OBJECTS_PER_KIND = 2
BOOLEAN_FEATURES = 12
DISTANCE_SCALE = 100.0
VIEWPORT_PIXELS = 128

# Image features are defined as a grayscale image of the game screen.
# This can be trimmed to remove the HUD, or resized to be a viewport around Link.

# Object features are defined as a normalized vector (x, y) and a distance scaled to [0, 1]:
#   0: closest enemy tile
#   1: closest projectile tile
#   2: closest item tile

# Objective features:
#   0: move_north
#   1: move_south
#   2: move_east
#   3: move_west
#   4: get_item
#   5: fight_enemies

# Source location:
#   0: from_north
#   1: from_south
#   2: from_east
#   3: from_west

# Boolean features:
#   0 has_beams: whether link has beams available
#   1 has_enemies: whether there are active enemies

class ObservationWrapper(gym.Wrapper):
    """A wrapper that trims the HUD and converts the image to grayscale."""
    def __init__(self, env, kind, normalize):
        super().__init__(env)
        self._prev_loc = None
        self._normalize = normalize

        if kind in ('gameplay', 'viewport'):
            self._trim = GAMEPLAY_START_Y
        else:
            self._trim = 0

        if kind == 'viewport':
            self._viewport_size = VIEWPORT_PIXELS
        else:
            self._viewport_size = None

        self.observation_space = Dict({
            "image": self._get_box_observation_space(),
            "vectors" : Box(low=-1.0, high=1.0, shape=(OBJECT_KINDS, OBJECTS_PER_KIND, 3), dtype=np.float32),
            "information" : gym.spaces.MultiBinary(BOOLEAN_FEATURES)
        })

    def reset(self, **kwargs):
        obs, state = self.env.reset(**kwargs)
        self._prev_loc = state.full_location
        obs = self._get_observation(state, obs)
        return obs, state

    def step(self, action):
        obs, reward, terminated, truncated, state_change = self.env.step(action)

        if state_change.previous.full_location != state_change.state.full_location:
            self._prev_loc = state_change.previous.full_location

        obs = self._get_observation(state_change.state, obs)
        return obs, reward, terminated, truncated, state_change

    def _get_observation(self, state : ZeldaGame, frame):
        return {
            "image" : self._get_image_observation(state, frame),
            "vectors" : self._get_vectors(state),
            "information" : self._get_information(state)
            }

    def _get_box_observation_space(self):
        # modify the observation space to match the new shape
        # we also move the last channel count to be the first dimension to avoid a VecTransposeImage wrapper
        height = self.observation_space.shape[0]
        width = self.observation_space.shape[1]
        dim = 1

        low = 0.0
        high = 1.0 if self._normalize else 255.0
        dtype = np.float32 if self._normalize else np.uint8

        if self._viewport_size:
            height = self._viewport_size
            width = self._viewport_size

        elif self._trim:
            height -= self._trim

        return Box(low=low, high=high, shape=(dim, height, width), dtype=dtype)

    def _get_image_observation(self, state: ZeldaGame, frame: np.ndarray) -> torch.Tensor:
        """
        Convert frame to the expected PyTorch format, possibly trim,
        convert to grayscale, and extract a viewport around Link.
        """

        if self._normalize:
            frame = torch.as_tensor(frame, dtype=torch.float32) / 255.0
        else:
            frame = torch.as_tensor(frame, dtype=torch.float32)

        frame = frame.permute(2, 0, 1)

        if self._trim:
            frame = frame[:, self._trim:, :]

        frame = (frame * GRAYSCALE_WEIGHTS.view(-1, 1, 1)).sum(dim=0, keepdim=True)

        if self._viewport_size:
            x = state.link.position.x
            y = state.link.position.y - self._trim  # we already trimmed
            half_vp = self._viewport_size // 2

            # Current height and width (after trimming)
            h, w = frame.shape[-2], frame.shape[-1]

            # Compute necessary padding
            top_pad = max(0, half_vp - y)
            bottom_pad = max(0, (y + half_vp) - h)
            left_pad = max(0, half_vp - x)
            right_pad = max(0, (x + half_vp) - w)

            # Pad using replicate if Link is near an edge
            if top_pad > 0 or bottom_pad > 0 or left_pad > 0 or right_pad > 0:
                frame = F.pad(frame, (left_pad, right_pad, top_pad, bottom_pad), mode='replicate')

            # Adjust coordinates after padding
            y += top_pad
            x += left_pad

            # Crop out the viewport
            frame = frame[...,  # keep channel as-is
                          y - half_vp : y + half_vp,
                          x - half_vp : x + half_vp]

        # 6) If not normalizing, clamp to [0, 255] and cast to byte
        if not self._normalize:
            frame = frame.clamp_(0, 255).byte()

        if frame.shape[-2] != self._viewport_size or frame.shape[-1] != self._viewport_size:
            frame = self._reshape(frame)

        return frame

    def _reshape(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Occasionally Link can be offscreen due to overscan, which messes up
        our viewport. This ensures the final shape is (1, _viewport_size, _viewport_size).
        """
        pad_h = self._viewport_size - frame.shape[-2]
        pad_w = self._viewport_size - frame.shape[-1]

        if pad_h > 0 or pad_w > 0:
            frame = F.pad(
                frame,
                (0, max(0, pad_w), 0, max(0, pad_h)),
                mode='replicate'
            )

        if (frame.shape[-2] != self._viewport_size or
            frame.shape[-1] != self._viewport_size):
            frame = torch.zeros(
                (1, self._viewport_size, self._viewport_size),
                dtype=frame.dtype,
                device=frame.device
            )

        return frame

    def _get_vectors(self, state : ZeldaGame):
        vectors = torch.zeros(OBJECT_KINDS, OBJECTS_PER_KIND, 3, dtype=torch.float32)

        items = state.items
        if state.treasure_location is not None:
            items = items + [ZeldaObject(state, -1, None, state.treasure_location)]

        kinds = [state.active_enemies, state.projectiles, items]
        for i, kind in enumerate(kinds):
            if kind:
                vectors[i, :OBJECTS_PER_KIND, :] = self._get_object_vectors(kind, OBJECTS_PER_KIND)

        return vectors.clamp(-1, 1)

    def _get_object_vectors(self, objects : Sequence[ZeldaObject], count):
        result = torch.zeros((count, 3), dtype=torch.float32)
        result[:, 2] = -1

        # closest objects first
        objects = sorted(objects, key=lambda obj: obj.distance)
        for i, obj in enumerate(objects[:count]):
            if obj.distance <= 1e-5:
                result[i] = torch.tensor([0, 0, -1], dtype=torch.float32)
            else:
                result[i, :2] = obj.vector[:2]
                result[i, 2] = obj.distance / DISTANCE_SCALE

        return result

    def _get_information(self, state : ZeldaGame):
        objectives = self._get_objectives_vector(state, state.objectives)

        source_direction = torch.zeros(4, dtype=torch.float32)
        self._assign_direction(source_direction, state.full_location.get_direction_of_movement(self._prev_loc))

        features = torch.zeros(2, dtype=torch.float32)
        features[0] = 1.0 if state.active_enemies else 0.0
        features[1] = 1.0 if state.link.are_beams_available else 0.0

        return torch.concatenate([objectives, source_direction, features])

    def _get_objectives_vector(self, state : ZeldaGame, objectives : Objective):
        result = torch.zeros(6, dtype=torch.float32)

        match objectives.kind:
            case ObjectiveKind.MOVE:
                for next_room in objectives.next_rooms:
                    direction = state.full_location.get_direction_of_movement(next_room)
                    self._assign_direction(result, direction)

            case ObjectiveKind.ITEM:
                result[4] = 1.0

            case ObjectiveKind.TREASURE:
                result[4] = 1.0

            case ObjectiveKind.FIGHT:
                result[5] = 1.0

        return result

    def _assign_direction(self, result, direction):
        match direction:
            case Direction.N:
                result[0] = 1.0
            case Direction.S:
                result[1] = 1.0
            case Direction.E:
                result[2] = 1.0
            case Direction.W:
                result[3] = 1.0
            case Direction.NW:
                result[0] = 1.0
                result[3] = 1.0
            case Direction.NE:
                result[0] = 1.0
                result[2] = 1.0
            case Direction.SW:
                result[1] = 1.0
                result[3] = 1.0
            case Direction.SE:
                result[1] = 1.0
                result[2] = 1.0
