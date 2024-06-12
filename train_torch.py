#!/usr/bin/python

import retro
import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from typing import Any

from torch_ppo import MultiHeadPPO, ZeldaMultiHeadNetwork
from triforce.model_parameters import GAMEPLAY_START_Y
from triforce.objective_selector import ObjectiveSelector
from triforce.zelda_wrapper import ZeldaGameWrapper

# Transform object distance into a temperature, which is higher when the enemy is closer.
# Temperature in range 1.0 - 0.1, with 1.0 being 8 pixels or closer.
# A temperature of 0 means no enemy is present.
TEMP_MAX = 1.0
TEMP_MIN = 0.1
MIN_DIST = 8
MAX_DIST = 175
SLOPE =  (TEMP_MAX - TEMP_MIN) / (MAX_DIST - MIN_DIST)
INTERCEPT = TEMP_MAX - SLOPE * MAX_DIST

class ObservationWrapper(gym.Wrapper):
    def __init__(self, env, viewport_size, trim, device):
        super().__init__(env)
        self.viewport_size = viewport_size
        self.half_vp = viewport_size // 2
        self.trim = trim
        self.device = device
        self.grayscale_weights = torch.tensor([0.2989, 0.5870, 0.1140]).to(device)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0.0, high=1.0, shape=(1, viewport_size, viewport_size), dtype=np.float32),
            gym.spaces.Box(low=0.0, high=1.0, shape=(3, 6, 3), dtype=np.float32)
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
        frame = frame[:, self.trim:, :]

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
        features = torch.zeros(3, 6, 3, dtype=torch.float32, device=self.device)
        self._assign_vector(features, 0, info['active_enemies'])
        self._assign_vector(features, 1, info['projectiles'])
        self._assign_vector(features, 2, info['items'])
        return features

    def _assign_vector(self, features, obj_type, objects):
        if objects:
            object_tensor = torch.zeros(6, 3, dtype=torch.float32).to(self.device)
            for i, obj in enumerate(objects):
                vect = torch.from_numpy(obj.vector)
                object_tensor[i, 0] = vect[0]
                object_tensor[i, 1] = vect[1]
                object_tensor[i, 2] = np.clip(SLOPE * obj.distance + INTERCEPT, 0.1, 1.0)

            features[obj_type, :] = object_tensor

class InputWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.MultiDiscrete([5, 3, 3])
        self.a_button = self.unwrapped.buttons.index('A')
        self.up_button = self.unwrapped.buttons.index('UP')
        self.down_button = self.unwrapped.buttons.index('DOWN')
        self.left_button = self.unwrapped.buttons.index('LEFT')
        self.right_button = self.unwrapped.buttons.index('RIGHT')
        self.button_len = len(self.unwrapped.buttons)

    def step(self, action):
        action = self.translate_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def _assign_button_fron_direction(self, action,  buttons):
        match action:
            case 0: buttons[self.up_button] = True
            case 1: buttons[self.down_button] = True
            case 2: buttons[self.left_button] = True
            case 3: buttons[self.right_button] = True
            case 4: buttons[self.up_button] = True
            case _: raise ValueError(f"Invalid dpad action: {action}")

    def translate_action(self, actions):
        pathfinding, danger, decision = actions
        buttons = [False] * self.button_len

        match decision:
            case 0:
                self._assign_button_fron_direction(pathfinding, buttons)

            case 1:
                self._assign_button_fron_direction(danger, buttons)
                buttons[self.a_button] = True

            case 2:
                self._assign_button_fron_direction(danger, buttons)
                buttons[self.a_button] = True

            case _: raise ValueError(f"Invalid button action: {decision}")

        return buttons

def make_zelda_env(save_state, *, render_mode = None, device = 'cpu'):
    env = retro.make(game='Zelda-NES', state=save_state, inttype=retro.data.Integrations.CUSTOM_ONLY,
                     render_mode=render_mode)
    #env = FrameCaptureWrapper(env, render_mode == 'rgb_array')
    env = ZeldaGameWrapper(env)
    env = ObjectiveSelector(env)
    env = ObservationWrapper(env, 128, GAMEPLAY_START_Y, device=device)
    env = InputWrapper(env)

    return env

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = make_zelda_env('0_67s.state', device=device)
    try:
        network = ZeldaMultiHeadNetwork(128, 54, device)
        ppo = MultiHeadPPO(network, device)

        ppo.train(env, 10_000)

    finally:
        env.close()

if __name__ == '__main__':
    main()
