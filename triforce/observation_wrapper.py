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

from .zelda_enums import Direction
from .objectives import Objective, ObjectiveKind
from .zelda_objects import ZeldaObject
from .zelda_game import ZeldaGame
from .model_parameters import VIEWPORT_PIXELS, GAMEPLAY_START_Y

GRAYSCALE_WEIGHTS = torch.FloatTensor([0.2989, 0.5870, 0.1140])
OBJECT_KINDS = 3
OBJECTS_PER_KIND = 2
BOOLEAN_FEATURES = 12
DISTANCE_SCALE = 100.0

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

    def _get_image_observation(self, state : ZeldaGame, frame):
        frame = torch.tensor(frame)

        # normalize
        if self._normalize:
            frame = frame / 255.0
        else:
            frame = frame.float()

        # reorder the channels to match the expected input format
        frame = frame.permute(2, 0, 1)

        # trim
        if self._trim:
            frame = frame[:, self._trim:, :]

        # convert to grayscale
        frame = torch.tensordot(frame, GRAYSCALE_WEIGHTS, dims=([0], [0]))
        frame = frame.unsqueeze(0)

        # extract viewport
        if self._viewport_size:
            x = state.link.position.x
            y = state.link.position.y - self._trim
            half_vp = self._viewport_size // 2
            frame_width, frame_height = frame.shape[2], frame.shape[1]

            pad_top = max(0, half_vp - y)
            pad_bottom = max(0, y + half_vp - frame_height)
            pad_left = max(0, half_vp - x)
            pad_right = max(0, x + half_vp - frame_width)

            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                frame = F.pad(frame, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')

            y += pad_top
            x += pad_left
            frame = frame[:, y - half_vp:y + half_vp, x - half_vp:x + half_vp]

        if not self._normalize:
            frame = frame.clamp(0, 255).byte()

        # Check and reshape frame if necessary
        if frame.shape[1] != self._viewport_size or frame.shape[2] != self._viewport_size:
            frame = self.reshape(frame)

        return frame

    def _reshape(self, frame):
        """Occasionally Link can be offscreen due to overscan which messes up our viewport.  This is a quick fix"""
        try:
            pad_height = self._viewport_size - frame.shape[1]
            pad_width = self._viewport_size - frame.shape[2]

            if pad_height > 0 or pad_width > 0:
                frame = F.pad(
                    frame,
                    (0, 0,                                  # Channel padding
                    max(0, pad_width), max(0, pad_width),   # Width padding
                    max(0, pad_height), max(0, pad_height)), # Height padding
                    mode='replicate'
                )

            return frame

        except ValueError:
            # If we fail for any reason, return a blank frame
            return torch.zeros((1, self._viewport_size, self._viewport_size), dtype=frame.dtype, device=frame.device)


    def _get_vectors(self, state : ZeldaGame):
        vectors = np.zeros(shape=(OBJECT_KINDS, OBJECTS_PER_KIND, 3), dtype=np.float32)

        items = state.items
        if state.treasure_location is not None:
            items = items + [ZeldaObject(state, -1, None, state.treasure_location)]

        kinds = [state.active_enemies, state.projectiles, items]
        for i, kind in enumerate(kinds):
            if kind:
                vectors[i, :OBJECTS_PER_KIND, :] = self._get_object_vectors(kind, OBJECTS_PER_KIND)

        return vectors

    def _get_object_vectors(self, objects : Sequence[ZeldaObject], count):
        result = np.zeros((count, 3), dtype=np.float32)
        result[:, 2] = -1

        # closest objects first
        objects = sorted(objects, key=lambda obj: obj.distance)
        for i, obj in enumerate(objects[:count]):
            if np.isclose(obj.distance, 0, atol=1e-5):
                result[i] = [0, 0, -1]

            else:
                dist = np.clip(obj.distance / DISTANCE_SCALE, 0, 1)
                result[i] = [obj.vector[0], obj.vector[1], dist]

        return result

    def _get_information(self, state : ZeldaGame):
        objectives = self._get_objectives_vector(state, state.objectives)

        source_direction = np.zeros(4, dtype=np.float32)
        self._assign_direction(source_direction, state.full_location.get_direction_of_movement(self._prev_loc))

        features = np.zeros(2, dtype=np.float32)
        features[0] = 1.0 if state.active_enemies else 0.0
        features[1] = 1.0 if state.link.are_beams_available else 0.0

        return np.concatenate([objectives, source_direction, features])

    def _get_objectives_vector(self, state : ZeldaGame, objectives : Objective):
        result = np.zeros(6, dtype=np.float32)

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
