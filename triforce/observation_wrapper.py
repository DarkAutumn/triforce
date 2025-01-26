"""
For this project, we don't neccessarily need to see the whole screen, or even the game in color.
The ZeldaObservationWrapper takes care of this by letting us (optionally) trim off the HUD and
convert the image to grayscale.  We also stack multiple frames together to give the agent a sense
of motion over time.
"""

from enum import Enum
from typing import List
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
import numpy as np
import torch
import torch.nn.functional as F

from .zelda_enums import GAMEPLAY_START_Y, Direction
from .objectives import Objective, ObjectiveKind
from .zelda_game import ZeldaGame

GRAYSCALE_WEIGHTS = torch.FloatTensor([0.2989, 0.5870, 0.1140])
BOOLEAN_FEATURES = 14
DISTANCE_SCALE = 100.0
VIEWPORT_PIXELS = 128

ENEMY_COUNT = 4
ENEMY_FEATURES = 6
NUM_ENEMY_TYPES = 49
ITEM_COUNT = 2
ITEM_FEATURES = 4
PROJECTILE_COUNT = 2
PROJECTILE_FEATURES = 5

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
    def __init__(self, env, kind, frame_stack, frame_skip, normalize):
        super().__init__(env)
        self._prev_loc = None
        self._normalize = normalize
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip

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
            "enemy_features": Box(low=-1.0, high=1.0, shape=(4, ENEMY_FEATURES), dtype=np.float32),
            "enemy_id": Discrete(ENEMY_COUNT),
            "item_features": Box(low=-1.0, high=1.0, shape=(2, ITEM_FEATURES), dtype=np.float32),
            "projectile_features": Box(low=-1.0, high=1.0, shape=(2, PROJECTILE_FEATURES), dtype=np.float32),
            "information" : gym.spaces.MultiBinary(BOOLEAN_FEATURES)
        })

        self._id_cache = {}

    def reset(self, **kwargs):
        frames, state = self.env.reset(**kwargs)
        self._prev_loc = state.full_location
        obs = self._get_observation(state, frames)
        return obs, state

    def step(self, action):
        frames, reward, terminated, truncated, state_change = self.env.step(action)
        if state_change.previous.full_location != state_change.state.full_location:
            self._prev_loc = state_change.previous.full_location

        obs = self._get_observation(state_change.state, frames)
        return obs, reward, terminated, truncated, state_change

    def _get_observation(self, state : ZeldaGame, frames : List[np.ndarray]):
        tensor = self._get_stacked_frames(frames, self.frame_stack, self.frame_skip)
        return {
            "image" : self._get_image_observation(state, tensor),
            "enemy_features" : self._get_enemy_features(state),
            "enemy_id" : self._get_enemy_ids(state),
            "item_features" : self._get_item_features(state),
            "projectile_features" : self._get_projectile_features(state),
            "information" : self._get_information(state)
            }

    def _get_box_observation_space(self):
        # modify the observation space to match the new shape
        # we also move the last channel count to be the first dimension to avoid a VecTransposeImage wrapper
        height = self.observation_space.shape[0]
        width = self.observation_space.shape[1]
        channels = self.frame_stack

        low = 0.0
        high = 1.0 if self._normalize else 255.0
        dtype = np.float32 if self._normalize else np.uint8

        if self._viewport_size:
            height = self._viewport_size
            width = self._viewport_size

        elif self._trim:
            height -= self._trim

        return Box(low=low, high=high, shape=(channels, height, width), dtype=dtype)

    def _get_stacked_frames(self, frames, count, skip):
        """
        Stack `count` frames from the `frames` list, skipping `skip` frames between them.
        If there aren't enough frames, duplicate the first frame to pad the stack.

        Args:
            frames (List[np.ndarray]): List of frames as numpy arrays with shape (H, W, C).
            count (int): Number of frames to stack.
            skip (int): Number of frames to skip between stacked frames.

        Returns:
            torch.Tensor: Stacked frames as a tensor with shape (count, C, H, W).
        """
        stacked_frames = []

        for i in range(count):
            frame_index = len(frames) - 1 - i * skip
            if frame_index >= 0:
                stacked_frames.append(frames[frame_index])
            else:
                # Not enough frames, duplicate the first frame
                stacked_frames.append(frames[0])

        # Reverse the order to maintain chronological order: oldest to newest
        stacked_frames.reverse()

        # Convert frames to tensors and permute axes (C, H, W)
        stacked_tensors = [
            torch.as_tensor(frame, dtype=torch.float32).permute(2, 0, 1)
            for frame in stacked_frames
        ]

        # Stack along the first dimension (stack size)
        return torch.stack(stacked_tensors, dim=0)

    def _get_image_observation(self, state: ZeldaGame, frames: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of frames to the expected PyTorch format, possibly trim,
        convert to grayscale, and extract a viewport around Link.

        Args:
            state (ZeldaGame): Current game state containing Link's position.
            frames (torch.Tensor): Batch of frames with shape (N, C, H, W).

        Returns:
            torch.Tensor: Processed batch of frames with shape (N, 1, viewport_size, viewport_size).
        """
        if self._normalize:
            frames = frames / 255.0

        if self._trim:
            frames = frames[:, :, self._trim:, :]

        # Convert to grayscale using the weights
        frames = (frames * GRAYSCALE_WEIGHTS.view(1, -1, 1, 1)).sum(dim=1, keepdim=False)

        if self._viewport_size:
            x = state.link.position.x
            y = state.link.position.y - self._trim  # we already trimmed
            half_vp = self._viewport_size // 2

            # Current height and width (after trimming)
            h, w = frames.shape[-2], frames.shape[-1]

            # Compute necessary padding
            top_pad = max(0, half_vp - y)
            bottom_pad = max(0, (y + half_vp) - h)
            left_pad = max(0, half_vp - x)
            right_pad = max(0, (x + half_vp) - w)

            # Pad using replicate if Link is near an edge
            if top_pad > 0 or bottom_pad > 0 or left_pad > 0 or right_pad > 0:
                frames = F.pad(frames, (left_pad, right_pad, top_pad, bottom_pad), mode='replicate')

            # Adjust coordinates after padding
            y += top_pad
            x += left_pad

            # Crop out the viewport for each frame in the batch
            frames = frames[:,  # keep batch dimension as-is
                            y - half_vp : y + half_vp,
                            x - half_vp : x + half_vp]

        if not self._normalize:
            frames = frames.clamp_(0, 255).byte()

        # Ensure all frames have the expected viewport size
        if frames.shape[-2] != self._viewport_size or frames.shape[-1] != self._viewport_size:
            frames = self._reshape(frames)

        return frames


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

    def _get_enemy_features(self, state: ZeldaGame) -> torch.Tensor:
        # Enemy features:
        #   0: presence (0 or 1 if there is an enemy in this slot)
        #   1: closeness (0 if far away, 1 if right on top of link)
        #   2-3: vector
        #   4-5: direction
        vectors = torch.zeros(ENEMY_COUNT, ENEMY_FEATURES, dtype=torch.float32)
        enemies = state.active_enemies or state.enemies
        if enemies:
            for i, enemy in enumerate(enemies):
                if i >= ENEMY_COUNT:
                    break
                vectors[i, 0] = 1
                vectors[i, 1] = self._distance_to_proximity(enemy.distance, DISTANCE_SCALE)
                vectors[i, 2:4] = enemy.vector
                vectors[1, 4:6] = enemy.direction.vector

        return vectors.clamp(-1, 1)

    def _get_enemy_ids(self, state: ZeldaGame) -> torch.Tensor:
        ids = torch.zeros(ENEMY_COUNT, dtype=torch.float32)
        enemies = state.active_enemies or state.enemies
        for i, enemy in enumerate(enemies):
            if i >= ENEMY_COUNT:
                break

            if (e_id := self._id_cache.get(enemy.id, None)) is None:
                e_id = enemy.id
                if isinstance(e_id, Enum):
                    e_id = e_id.value
                self._id_cache[enemy.id] = int(e_id)

            ids[i] = e_id

        return ids

    def _get_item_features(self, state: ZeldaGame) -> torch.Tensor:
        # Item features:
        #   0: presence (0 or 1 if there is an item in this slot)
        #   1: closeness (0 if far away, 1 if right on top of link)
        #   2-3: vector
        vectors = torch.zeros(ITEM_COUNT, ITEM_FEATURES, dtype=torch.float32)
        items = state.items
        if state.treasure is not None:
            items = [state.treasure, *items]

        for i, item in enumerate(items):
            if i >= ITEM_COUNT:
                break
            vectors[i, 0] = 1
            vectors[i, 1] = self._distance_to_proximity(item.distance, DISTANCE_SCALE)
            vectors[i, 2:4] = item.vector

        return vectors.clamp(-1, 1)

    def _get_projectile_features(self, state: ZeldaGame) -> torch.Tensor:
        # Projectile features:
        #   0: presence (0 or 1 if there is a projectile in this slot)
        #   1: closeness (0 if far away, 1 if right on top of link)
        #   2-3: vector
        #   4: blockable
        vectors = torch.zeros(PROJECTILE_COUNT, PROJECTILE_FEATURES, dtype=torch.float32)
        for i, proj in enumerate(state.projectiles):
            if i >= PROJECTILE_COUNT:
                break
            vectors[i, 0] = 1
            vectors[i, 1] = self._distance_to_proximity(proj.distance, DISTANCE_SCALE)
            vectors[i, 2:4] = proj.vector
            vectors[i, 4] = 1 if proj.blockable else -1

        return vectors.clamp(-1, 1)

    def _distance_to_proximity(self, distance: float, scale: float, min_closeness: float = 0.1) -> float:
        if distance <= 5:
            return 1.0
        if distance >= scale:
            return min_closeness

        closeness = 1.0 - ((distance - 5) / (scale - 5))
        return max(min_closeness, min_closeness + closeness * (1 - min_closeness))

    def _get_information(self, state : ZeldaGame):
        objectives = self._get_objectives_vector(state, state.objectives)

        source_direction = torch.zeros(4, dtype=torch.float32)
        self._assign_direction(source_direction, state.full_location.get_direction_to(self._prev_loc))

        features = torch.zeros(4, dtype=torch.float32)
        features[0] = 1.0 if state.active_enemies else 0.0
        features[1] = 1.0 if state.link.are_beams_available else 0.0
        features[2] = 1.0 if state.link.heart_halves <= 2 else 0.0
        features[3] = 1.0 if state.link.is_health_full else 0.0

        return torch.concatenate([objectives, source_direction, features])

    def _get_objectives_vector(self, state : ZeldaGame, objectives : Objective):
        result = torch.zeros(6, dtype=torch.float32)

        match objectives.kind:
            case ObjectiveKind.MOVE:
                for next_room in objectives.next_rooms:
                    direction = state.full_location.get_direction_to(next_room)
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
