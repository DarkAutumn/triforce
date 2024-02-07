# For this project, we don't neccessarily need to see the whole screen, or even the game in color.
# The ZeldaObservationWrapper takes care of this by letting us (optionally) trim off the HUD and
# convert the image to grayscale.  We also stack multiple frames together to give the agent a sense
# of motion over time.

import math
import gymnasium as gym
import numpy as np
from collections import deque

from .zelda_game_data import zelda_game_data
from .model_parameters import viewport_pixels, gameplay_start_y

def overscan_reshape(tile_layout):
    reshaped_array = tile_layout.reshape((32, 22))

    trimmed_array = reshaped_array[1:-1:]

    return trimmed_array

def overlay_grid_and_text(image_array, tile_numbers, output_filename, pos):
    from PIL import Image, ImageDraw, ImageFont
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(image_array)

    scale = 4

    new_size = (image.width * scale, image.height * scale)
    image = image.resize(new_size, Image.NEAREST)

    draw = ImageDraw.Draw(image)

    # Define the new size of the grid
    grid_width = 30  # Adjusted grid width
    grid_height = 21  # Adjusted grid height
    tile_width = 8  * scale
    tile_height = 8 * scale

    # Optionally, define a font for the text
    font_size = max(min(tile_width, tile_height) // 2, 10)  # Half of the smaller tile dimension or 10, whichever is larger
    font = ImageFont.load_default()  # Default font


    # Iterate over the grid and draw lines and text
    for i in range(grid_width):
        for j in range(grid_height):
            # Calculate the top left corner of the tile, adjusted by start_y
            x = i * tile_width
            y = j * tile_height

            # Draw the grid lines
            draw.rectangle([x, y, x + tile_width - 1, y + tile_height - 1], outline="blue", width=1)

            # Get the tile number and calculate its position
            tile_number = tile_numbers[i, j]  # Adjusted indexing for the new shape
            text = f"{tile_number:02X}"  # Format as two hex digits
            font = ImageFont.load_default()
            text_width, text_height = draw.textsize(text, font=font)
            text_x = x + (tile_width - text_width) // 2
            text_y = y + (tile_height - text_height) // 2

            # Draw the tile number
            draw.text((text_x, text_y), text, fill="white", font=font)

    x, y = pos
    y -= gameplay_start_y
    x = x * scale
    y = y * scale

    draw.rectangle([x - 16, y - 16, x + 16, y + 16], outline="black", width=2)

    # Save the image
    image.save(output_filename)
    pass

    
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

        #ram = self.unwrapped.get_ram()
        #map_offset, map_len = zelda_game_data.tables['tile_layout']
        #tiles = ram[map_offset:map_offset+map_len]
        #overlay_grid_and_text(frame, overscan_reshape(tiles), "overlay.png", info['link_pos'])

        if self.viewport_size:
            if 'link_pos' in info:
                x, y = info.get('link_pos')
                y -= self.trim
            else:
                x, y = 0, 0

            padded_frame = np.pad(frame, ((256, 256), (256, 256), (0, 0)), mode='edge')
            frame = self.extract_viewport(padded_frame, x + 256, y + 256, self.viewport_size)

        if self.grayscale:
            frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            frame = np.expand_dims(frame, axis=-1)

        return frame
    
    def extract_viewport(self, padded_frame, x_padded, y_padded, viewport_size):
        half_vp = viewport_size // 2

        x_start = x_padded - half_vp
        y_start = y_padded - half_vp

        viewport = padded_frame[y_start:y_start+viewport_size, x_start:x_start+viewport_size]
        return viewport
