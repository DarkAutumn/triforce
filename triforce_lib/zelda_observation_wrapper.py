# For this project, we don't neccessarily need to see the whole screen, or even the game in color.
# The ZeldaObservationWrapper takes care of this by letting us (optionally) trim off the HUD and
# convert the image to grayscale.  We also stack multiple frames together to give the agent a sense
# of motion over time.

import gymnasium as gym
import numpy as np
from collections import deque
from .model_parameters import viewport_pixels

# the y coordinate where the game viewport starts (above which is the HUD)
gameplay_start_y = 55

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
    def __init__(self, env, frames, n_frames, grayscale, kind):
        super().__init__(env)
        self.env = env
        self.frames = frames
        self.n_frames = n_frames
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
        
        if n_frames > 1:
            shape = self.observation_space.shape
            # add a dimension for the number of frames we stack, so that the new shape is (h, w, ch, frames)
            new_shape = (shape[0], shape[1], shape[2], n_frames)
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._get_observation(info), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._get_observation(info), reward, terminated, truncated, info

    def _get_observation(self, info):
        # Some Zelda enemies flash in and out every other frame, so we need to capture a sequence
        # of frames in a way that will capture the enemy in both states.  We also don't really want
        # to just use the last three frames, as that doesn't give a good sense of motion.  So we will
        # use some from the past, being sure to not always pick odd or even frames.

        if self.n_frames > 1:
            stacked = []
            sequence = [1, 6, 15]
            for i in range(self.n_frames):
                position = sequence[i]
                frame = self.frames[-position]
                frame = self.trim_normalize_grayscale(info, frame)
                stacked.append(frame)

            if self.n_frames > 1:
                stacked_images = np.stack(stacked, axis=-1)
            return stacked_images
        else:
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

            half_vp = self.viewport_size // 2
            padded_frame = np.pad(frame, ((half_vp, half_vp), (half_vp, half_vp), (0, 0)), mode='edge')
            center_x, center_y = y + half_vp, x + half_vp
            frame = padded_frame[center_x - half_vp:center_x + half_vp, center_y - half_vp:center_y + half_vp]

        if self.grayscale:
            frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            frame = np.expand_dims(frame, axis=-1)
        return frame
