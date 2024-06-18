"""
For this project, we don't neccessarily need to see the whole screen, or even the game in color.
The ZeldaObservationWrapper takes care of this by letting us (optionally) trim off the HUD and
convert the image to grayscale.  We also stack multiple frames together to give the agent a sense
of motion over time.
"""

from collections import deque
import pickle
from typing import Any
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from .model_parameters import VIEWPORT_PIXELS, GAMEPLAY_START_Y

class FrameCaptureWrapper(gym.Wrapper):
    """A wrapper that captures the last 30 frames of the environment."""
    def __init__(self, env, rgb_render : bool | deque):
        super().__init__(env)
        self.env = env
        self.observation_space = self.env.observation_space
        self.frames = deque(maxlen=30)
        self.rgb_deque = None
        if isinstance(rgb_render, deque):
            self.rgb_deque = rgb_render
        elif rgb_render:
            self.rgb_deque = deque(maxlen=120)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        for _ in range(self.frames.maxlen):
            self.frames.append(observation)

        if self.rgb_deque is not None:
            self.rgb_deque.append(self.env.render())

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)

        if self.rgb_deque is not None:
            self.rgb_deque.append(self.env.render())

        return observation, reward, terminated, truncated, info

class ZeldaObservationWrapper(gym.Wrapper):
    """A wrapper that trims the HUD and converts the image to grayscale."""
    def __init__(self, env, frames, grayscale, kind, framestack):
        super().__init__(env)
        self.env = env
        self.frames = frames
        self.observation_space = self.env.observation_space
        self.grayscale = grayscale
        self.framestack = framestack

        if kind in ('gameplay', 'viewport'):
            self.trim = GAMEPLAY_START_Y
        else:
            self.trim = 0

        if kind == 'viewport':
            self.viewport_size = VIEWPORT_PIXELS

        # modify the observation space to match the new shape
        # we also move the last channel count to be the first dimension to avoid a VecTransposeImage wrapper
        if grayscale:
            shape = self.observation_space.shape
            new_shape = (shape[0], shape[1], 1)
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

        if self.viewport_size:
            shape = self.observation_space.shape
            new_shape = (self.viewport_size, self.viewport_size, shape[2])
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

        elif self.trim:
            shape = self.observation_space.shape
            new_shape = (shape[0] - self.trim, shape[1], shape[2])
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

        if framestack > 1:
            shape = self.observation_space.shape
            new_shape = (shape[0] * framestack, shape[1], shape[2])
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._get_observation(info), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._get_observation(info), reward, terminated, truncated, info

    def _get_observation(self, info):
        if self.framestack > 1:
            frames = []
            for i in range(self.framestack):
                frame = self.frames[-i * 2 - 1]
                frame = self.trim_normalize_grayscale(info, frame)
                frames.append(frame)
            result = np.concatenate(frames, axis=0)
            return result

        frame = self.frames[-1]
        frame = self.trim_normalize_grayscale(info, frame)
        return frame

    def trim_normalize_grayscale(self, info, frame):
        """Trim the HUD, normalize the frame, and convert it to grayscale."""
        if self.trim:
            frame = frame[self.trim:, :, :]

        if self.viewport_size:
            if 'link_pos' in info:
                x, y = info.get('link_pos')
                y -= self.trim
            else:
                x, y = 0, 0

            frame = self.extract_viewport(info, frame, x, y)

        if self.grayscale:
            frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            frame = np.expand_dims(frame, axis=-1)

        return frame

    def extract_viewport(self, info, frame, x, y):
        """Extract the viewport around link.  If link is offscreen, pad the frame with the edge color."""
        half_vp = self.viewport_size // 2
        padded_frame = np.pad(frame, ((half_vp, half_vp), (half_vp, half_vp), (0, 0)), mode='edge')

        center_x, center_y = y + half_vp, x + half_vp
        frame = padded_frame[center_x - half_vp:center_x + half_vp, center_y - half_vp:center_y + half_vp]

        # link can sometimes be offscreen due to overscan
        if frame.shape[0] != self.viewport_size or frame.shape[1] != self.viewport_size:
            frame = self.reshape(info, frame, x, y)

        return frame

    def reshape(self, info, frame, x, y):
        """
        Occasionally link can be offscreen due to overscan, so we pad the frame with the edge color.
        This shouldn't really happen, so we are saving a 'reshape_error.pkl' file to debug this issue.
        """
        try:
            return np.pad(frame, ((0, self.viewport_size - frame.shape[0]), (0, self.viewport_size - frame.shape[1]),
                                  (0, 0)), mode='edge')
        except ValueError:
            result = { 'info': info, 'frame' : self.frames[-1], 'reshape_input': frame, 'pos' : (x, y) }

            with open('reshape_error.pkl', 'wb') as f:
                pickle.dump(result, f)

            return np.zeros((self.viewport_size, self.viewport_size, 3), dtype=np.uint8)

# The MultiHead model takes in a distance in addition to vectors.  We want 0 to represent "no enemies",
# so instead of giving a raw distance to targets, we give a "closeness" factor from 0.1 to 1.0 which
# I'm calling a 'threat temperature'.
TEMP_MAX = 1.0
TEMP_MIN = 0.1
MIN_DIST = 8
MAX_DIST = 175
SLOPE =  (TEMP_MAX - TEMP_MIN) / (MAX_DIST - MIN_DIST)
INTERCEPT = TEMP_MAX - SLOPE * MAX_DIST

# Shape:
# 6x3 - 6 enemies with a vector and a threat temperature
# 3x3 - 3 projectiles with a vector and a threat temperature
# 3x3 - 3 items with a vector and a temperature
# 1x3 - objective vector (unused temp)
# = 6x3 + 3x3 + 3x3 + 1x2 = 18 + 9 + 9 + 2 = 39

class MultiHeadObservationWrapper(gym.Wrapper):
    """An observation wrapper used for the torch based PPO/multi-head model.  This is similar to
    ZeldaObservationWrapper and ZeldaVectorFeatures, but sufficiently different that it's in a new class.
    """
    def __init__(self, env, viewport_size, device):
        super().__init__(env)
        self.viewport_size = viewport_size
        self.half_vp = viewport_size // 2
        self.device = device
        self.grayscale_weights = torch.tensor([0.2989, 0.5870, 0.1140]).to(device)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0.0, high=1.0, shape=(1, viewport_size, viewport_size), dtype=np.float32),
            gym.spaces.Box(low=0.0, high=1.0, shape=(13, 3), dtype=np.float32)
        ))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset()
        return self._get_observation(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._get_observation(obs, info), reward, terminated, truncated, info

    def _get_observation(self, obs, info):
        return self._get_image(obs, info), self._get_features(info)

    def _get_image(self, obs, info):
        frame = torch.tensor(obs).to(self.device)

        # normalize
        frame = frame / 255.0

        # reorder the channels to match the expected input format
        frame = frame.permute(2, 0, 1)

        # trim
        frame = frame[:, GAMEPLAY_START_Y:, :]

        # convert to grayscale
        frame = torch.tensordot(frame, self.grayscale_weights, dims=([0], [0]))
        frame = frame.unsqueeze(0)

        # extract viewport
        x = info['link_x']
        y = info['link_y'] - GAMEPLAY_START_Y
        half_vp = self.half_vp
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
        return frame

    def _get_features(self, info):
        values = []
        values.extend(self._get_values(info['active_enemies'], 6))
        values.extend(self._get_values(info['projectiles'], 3))
        values.extend(self._get_values(info['items'], 3))
        obj_vect = info['objective_vector']
        values.append((obj_vect[0], obj_vect[1], 0.0))
        values = [float(item) for sublist in values for item in sublist]

        return torch.tensor(values, device=self.device).unsqueeze(0)

    def _get_values(self, objects, length):
        result = [(0, 0, 0)] * length
        for i, obj in enumerate(objects):
            vect = torch.from_numpy(obj.vector)
            result[i] = (vect[0], vect[1], np.clip(SLOPE * obj.distance + INTERCEPT, 0.1, 1.0))

        return result

__all__ = ['ZeldaObservationWrapper', 'FrameCaptureWrapper', 'MultiHeadObservationWrapper']
