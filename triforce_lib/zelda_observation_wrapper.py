# For this project, we don't neccessarily need to see the whole screen, or even the game in color.
# The ZeldaObservationWrapper takes care of this by letting us (optionally) trim off the HUD and
# convert the image to grayscale.  We also stack multiple frames together to give the agent a sense
# of motion over time.

import gymnasium as gym
import numpy as np
from collections import deque

from .zelda_game_data import zelda_game_data
from .model_parameters import viewport_pixels, gameplay_start_y
 
class FrameCaptureWrapper(gym.Wrapper):
    def __init__(self, env, rgb_render):
        super().__init__(env)
        self.env = env
        self.observation_space = self.env.observation_space
        self.frames = deque(maxlen=30)
        if rgb_render:
            self.rgb_deque = deque(maxlen=120)
        else:
            self.rgb_deque = None

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
    def __init__(self, env, frames, grayscale, kind):
        super().__init__(env)
        self.env = env
        self.frames = frames
        self.observation_space = self.env.observation_space
        self.grayscale = grayscale

        if kind == 'gameplay' or kind == 'viewport':
            self.trim = gameplay_start_y
        else:
            self.trim = 0

        if kind == 'viewport':
            self.viewport_size = viewport_pixels

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

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._get_observation(info), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._get_observation(info), reward, terminated, truncated, info

    def _get_observation(self, info):
        frame = self.frames[-1]
        frame = self.trim_normalize_grayscale(info, frame)
        return frame

    def trim_normalize_grayscale(self, info, frame):
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
        half_vp = self.viewport_size // 2
        padded_frame = np.pad(frame, ((half_vp, half_vp), (half_vp, half_vp), (0, 0)), mode='edge')

        center_x, center_y = y + half_vp, x + half_vp
        frame = padded_frame[center_x - half_vp:center_x + half_vp, center_y - half_vp:center_y + half_vp]

        # link can sometimes be offscreen due to overscan
        if frame.shape[0] != self.viewport_size or frame.shape[1] != self.viewport_size:
            frame = self.reshape(info, frame, x, y)

        return frame

    def reshape(self, info, frame, x, y):
        try:
            return np.pad(frame, ((0, self.viewport_size - frame.shape[0]), (0, self.viewport_size - frame.shape[1]), (0, 0)), mode='edge')
        except ValueError:
            import pickle
            result = { 'info': info, 'frame': frame, 'pos' : (x, y) }

            with open('reshape_error.pkl', 'wb') as f:
                pickle.dump(result, f)

            return np.zeros((self.viewport_size, self.viewport_size, 3), dtype=np.uint8)
