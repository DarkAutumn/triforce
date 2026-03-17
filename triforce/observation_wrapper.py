"""
For this project, we don't neccessarily need to see the whole screen, or even the game in color.
The ZeldaObservationWrapper takes care of this by letting us (optionally) trim off the HUD and
convert the image to grayscale.  We also stack multiple frames together to give the agent a sense
of motion over time.
"""

from typing import List
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
import torch
import torch.nn.functional as F

from .zelda_enums import (GAMEPLAY_START_Y, HUD_TRIM_FULL, HUD_TRIM_CROPPED,
                          Direction, ZeldaEnemyKind, ZeldaItemKind, ZeldaProjectileId)
from .objectives import ObjectiveKind
from .zelda_game import ZeldaGame

GRAYSCALE_WEIGHTS = torch.FloatTensor([0.2989, 0.5870, 0.1140])
_GRAYSCALE_WEIGHTS_4D = GRAYSCALE_WEIGHTS.view(1, -1, 1, 1)
_GRAYSCALE_NORM_WEIGHTS_4D = _GRAYSCALE_WEIGHTS_4D / 255.0
BOOLEAN_FEATURES = 15
VIEWPORT_PIXELS = 128

# Unified entity observation: 11 NES object slots + 1 treasure slot
ENTITY_SLOTS = 12
ENTITY_FEATURES = 7

# Entity type ID mapping for unified embedding (0 = empty/unknown)
_ENTITY_TYPE_MAP = {}
_next_id = 1

for _member in ZeldaEnemyKind:
    _ENTITY_TYPE_MAP[_member] = _next_id
    _next_id += 1

for _member in ZeldaItemKind:
    _ENTITY_TYPE_MAP[_member] = _next_id
    _next_id += 1

for _member in ZeldaProjectileId:
    _ENTITY_TYPE_MAP[_member] = _next_id
    _next_id += 1

TREASURE_TYPE_ID = _next_id
_next_id += 1
NUM_ENTITY_TYPES = _next_id

# Reverse mapping for debugger display
ENTITY_TYPE_NAMES = {0: "Empty"}
for _key, _val in _ENTITY_TYPE_MAP.items():
    ENTITY_TYPE_NAMES[_val] = _key.name
ENTITY_TYPE_NAMES[TREASURE_TYPE_ID] = "Treasure"


def infer_obs_kind(obs_space):
    """Infer the observation kind from a saved observation space.

    Returns:
        (obs_kind, frame_stack) tuple, e.g. ('viewport', 3) or ('full-rgb', 1).
    """
    image_shape = obs_space["image"].shape  # (channels, H, W)
    channels, h, w = image_shape

    if h == VIEWPORT_PIXELS and w == VIEWPORT_PIXELS:
        return 'viewport', channels

    # full-rgb: channels is a multiple of 3 (RGB per stacked frame)
    if channels % 3 == 0 and channels <= 9:
        return 'full-rgb', channels // 3

    # grayscale gameplay: channels == frame_stack
    return 'gameplay', channels


class ObservationWrapper(gym.Wrapper):
    """A wrapper that trims the HUD and converts the image to grayscale."""
    def __init__(self, env, kind, frame_stack, frame_skip, normalize, full_screen=False):
        super().__init__(env)
        self._prev_loc = None
        self._normalize = normalize
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.full_screen = full_screen
        self._grayscale = kind != 'full-rgb'

        if kind in ('gameplay', 'viewport', 'full-rgb'):
            self._trim = HUD_TRIM_FULL if full_screen else HUD_TRIM_CROPPED
            # Viewport centering always uses GAMEPLAY_START_Y so that the viewport
            # position relative to Link is identical across cropped and full modes.
            # (In cropped mode, _trim==56 happens to equal GAMEPLAY_START_Y; in full
            # mode _trim==64 cuts 8 more pixels but we still subtract 56 from link.y.)
            self._viewport_y_offset = GAMEPLAY_START_Y
        else:
            self._trim = 0
            self._viewport_y_offset = 0

        if kind == 'viewport':
            self._viewport_size = VIEWPORT_PIXELS
        else:
            self._viewport_size = None

        self.observation_space = Dict({
            "image": self._get_box_observation_space(),
            "entities": Box(low=-1.0, high=1.0, shape=(ENTITY_SLOTS, ENTITY_FEATURES), dtype=np.float32),
            "entity_types": gym.spaces.MultiDiscrete([NUM_ENTITY_TYPES] * ENTITY_SLOTS),
            "information" : gym.spaces.MultiBinary(BOOLEAN_FEATURES)
        })

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
        entities, entity_types = self._get_entity_observation(state)
        return {
            "image" : self._get_image_observation(state, tensor),
            "entities" : entities,
            "entity_types" : entity_types,
            "information" : self._get_information(state)
            }

    def _get_box_observation_space(self):
        # modify the observation space to match the new shape
        # we also move the last channel count to be the first dimension to avoid a VecTransposeImage wrapper
        height = self.observation_space.shape[0]
        width = self.observation_space.shape[1]

        # Grayscale: 1 channel per stacked frame. RGB: 3 channels per stacked frame.
        channels = self.frame_stack if self._grayscale else self.frame_stack * 3

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
        last = len(frames) - 1
        indices = [max(0, last - i * skip) for i in range(count)]
        indices.reverse()

        # Stack as numpy (contiguous), then zero-copy to torch, permute, and convert to float32
        stacked = np.stack([frames[i] for i in indices], axis=0)
        return torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous().float()

    def _get_image_observation(self, state: ZeldaGame, frames: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of frames to the expected PyTorch format, possibly trim,
        convert to grayscale, and extract a viewport around Link.

        Args:
            state (ZeldaGame): Current game state containing Link's position.
            frames (torch.Tensor): Batch of frames with shape (N, C, H, W).

        Returns:
            torch.Tensor: Processed frames. Shape depends on mode:
                - Grayscale viewport: (N, H_vp, W_vp)
                - Grayscale gameplay: (N, H_trim, W)
                - full-rgb: (N*3, H_trim, W)
        """
        if self._trim:
            frames = frames[:, :, self._trim:, :]

        if self._grayscale:
            if self._normalize:
                # Combine normalization and grayscale in one multiply+sum
                frames = (frames * _GRAYSCALE_NORM_WEIGHTS_4D).sum(dim=1, keepdim=False)
            else:
                frames = (frames * _GRAYSCALE_WEIGHTS_4D).sum(dim=1, keepdim=False)
        else:
            # full-rgb: keep RGB channels, flatten frame_stack × 3 into channel dim
            if self._normalize:
                frames = frames / 255.0
            # frames shape: (N, 3, H, W) → reshape to (N*3, H, W) for channel stacking
            frames = frames.reshape(-1, frames.shape[-2], frames.shape[-1])

        if self._viewport_size:
            x = state.link.position.x
            y = state.link.position.y - self._viewport_y_offset
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
        if self._viewport_size and (frames.shape[-2] != self._viewport_size or frames.shape[-1] != self._viewport_size):
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
                (frame.shape[0], self._viewport_size, self._viewport_size),
                dtype=frame.dtype,
                device=frame.device
            )

        return frame

    def _get_entity_observation(self, state: ZeldaGame):
        """Build unified entity features and type IDs for all 12 slots.

        Entity features per slot (7 dims):
            0: presence (0 or 1)
            1: dir_x  entity movement direction x
            2: dir_y  entity movement direction y
            3: health (enemy HP / 15, 0 for non-enemies)
            4: stun   (enemy stun_timer / 255, 1.0 when clock active)
            5: hurts_on_touch (1 for enemies/projectiles, 0 when clock active)
            6: killable (1 for enemies only)

        Positions are intentionally omitted — the full-screen visual encoder
        learns spatial relationships from pixels via CoordConv.
        """
        features = torch.zeros(ENTITY_SLOTS, ENTITY_FEATURES, dtype=torch.float32)
        types = torch.zeros(ENTITY_SLOTS, dtype=torch.int64)

        has_clock = bool(state.link.clock)

        # NES object slots 1-11 (indices 0-10)
        for i, entry in enumerate(state.all_entities):
            if entry is None:
                continue
            entity, category = entry
            features[i, 0] = 1.0

            if category == 'enemy':
                features[i, 1:3] = entity.direction.vector
                features[i, 3] = entity.health / 15.0
                features[i, 4] = 1.0 if has_clock else min(entity.stun_timer / 255.0, 1.0)
                features[i, 5] = 0.0 if has_clock else 1.0
                features[i, 6] = 1.0
            elif category == 'projectile':
                features[i, 1:3] = entity.direction.vector
                features[i, 5] = 0.0 if has_clock else 1.0

            types[i] = _ENTITY_TYPE_MAP.get(entity.id, 0)

        # Slot 11 (index 11): treasure
        treasure = state.treasure
        if treasure is not None:
            features[11, 0] = 1.0
            types[11] = TREASURE_TYPE_ID

        return features.clamp(-1, 1), types

    def _get_information(self, state : ZeldaGame):
        result = torch.zeros(BOOLEAN_FEATURES, dtype=torch.float32)

        # Objectives (indices 0-5)
        objectives = state.objectives
        match objectives.kind:
            case ObjectiveKind.MOVE:
                for next_room in objectives.next_rooms:
                    direction = state.full_location.get_direction_to(next_room)
                    self._assign_direction(result, direction)
            case ObjectiveKind.ITEM | ObjectiveKind.TREASURE:
                result[4] = 1.0
            case ObjectiveKind.FIGHT:
                result[5] = 1.0

        # Source direction (indices 6-9)
        full_loc = state.full_location
        direction = full_loc.get_direction_to(self._prev_loc)
        self._assign_direction_offset(result, 6, direction)

        # Features (indices 10-14)
        result[10] = 1.0 if state.active_enemies else 0.0
        result[11] = 1.0 if state.link.are_beams_available else 0.0
        result[12] = 1.0 if state.link.health <= 1 else 0.0
        result[13] = 1.0 if state.link.is_health_full else 0.0
        result[14] = 1.0 if state.link.clock else 0.0

        return result

    def _assign_direction_offset(self, result, offset, direction):
        """Assign direction flags at a specific offset in the result tensor."""
        match direction:
            case Direction.N:
                result[offset] = 1.0
            case Direction.S:
                result[offset + 1] = 1.0
            case Direction.E:
                result[offset + 2] = 1.0
            case Direction.W:
                result[offset + 3] = 1.0
            case Direction.NW:
                result[offset] = 1.0
                result[offset + 3] = 1.0
            case Direction.NE:
                result[offset] = 1.0
                result[offset + 2] = 1.0
            case Direction.SW:
                result[offset + 1] = 1.0
                result[offset + 3] = 1.0
            case Direction.SE:
                result[offset + 1] = 1.0
                result[offset + 2] = 1.0

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
