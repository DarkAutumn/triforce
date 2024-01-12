import gymnasium as gym
import numpy as np
from collections import deque

# the y coordinate where the game viewport starts (above which is the HUD)
gameplay_start_y = 55

class FrameCaptureWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = self.env.observation_space
        self.frames = deque(maxlen=30)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        for _ in range(self.frames.maxlen):
            self.frames.append(observation[0])
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation[0])
        return observation, reward, done, info

class ZeldaViewWrapper(gym.Wrapper):
    def __init__(self, env, frames, count, grayscale, gameplay_only):
        super().__init__(env)
        self.env = env
        self.frames = frames
        self.n_stack = count
        self.observation_space = self.env.observation_space
        self.grayscale = grayscale
        if gameplay_only:
            self.trim = gameplay_start_y
        else:
            self.trim = 0

        # modify the observation space to match the new shape
        # we also move the last channel count to be the first dimension to avoid a VecTransposeImage wrapper
        if grayscale:
            shape = self.observation_space.shape
            new_shape = (1, shape[0], shape[1])
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)
        else:
            # move the last dimension to the first
            shape = self.observation_space.shape
            new_shape = (shape[2], shape[0], shape[1])
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

        if self.trim:
            shape = self.observation_space.shape
            new_shape = (shape[0], shape[1] - self.trim, shape[2])
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self._get_observation()

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        stacked = []

        # Some Zelda enemies flash in and out every other frame, so we need to capture a sequence
        # of frames in a way that will capture the enemy in both states.

        sequence = [1, 6, 15]
        for i in range(self.n_stack):
            position = sequence[i]
            frame = self.frames[-position]

            if self.trim:
                frame = frame[self.trim:, :, :]

            if self.grayscale:
                frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                frame = frame[np.newaxis, ...]  # Add the channel to the image, put it first to avoid VecTransposeImage
            else:
                # move the last dimension to the first
                frame = np.moveaxis(frame, -1, 0)

            stacked.append(frame)
            print(stacked[-1])

        print(self.observation_space.shape)
        stacked_images = np.stack(stacked, axis=-1)
        print(stacked_images.shape)
        return stacked_images
