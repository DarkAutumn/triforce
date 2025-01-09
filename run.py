#! /usr/bin/python
"""Run the ZeldaML agent to play The Legend of Zelda (NES)."""

# While we want to keep this file relatively clean, it's fine to have a bit of a large render function.

# pylint: disable=too-few-public-methods,too-many-locals,too-many-branches,too-many-statements
# pylint: disable=too-many-nested-blocks,duplicate-code

import argparse
import os
import sys
import math
from collections import deque
import pygame
import numpy as np
import cv2
import tqdm

from triforce import ModelSelector, ZeldaScenario, ZeldaModelDefinition, simulate_critique, make_zelda_env, ZeldaAI, \
                     TRAINING_SCENARIOS
from triforce.game_state_change import ZeldaStateChange
from triforce.zelda_game import ZeldaGame
from triforce.zelda_observation_wrapper import FrameCaptureWrapper

class Recording:
    """Used to track and save a recording of the game."""
    # pylint: disable=no-member
    def __init__(self, dimensions, buffer_size):
        self.dimensions = dimensions
        self.recording = None
        self.buffer_size = buffer_size
        self.buffer = []

    def _get_recording(self):
        if self.recording is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.recording = cv2.VideoWriter(self.__get_filename(), fourcc, 60.1, self.dimensions)

        return self.recording

    def write(self, surface):
        """Adds a frame to the recording."""
        surface = surface.copy()

        if self.buffer_size <= 1:
            self._write_surface(surface.copy())

        else:
            self.buffer.append(surface)
            if len(self.buffer) >= self.buffer_size:
                self.flush()

    def _write_surface(self, surface):
        result_frame = pygame.surfarray.array3d(surface)
        result_frame = result_frame.transpose([1, 0, 2])
        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)

        recording = self._get_recording()
        recording.write(result_frame)

    def flush(self):
        """Writes the buffer to the recording."""
        if len(self.buffer) < 1000:
            for frame in self.buffer:
                self._write_surface(frame)
        else:
            for frame in tqdm.tqdm(self.buffer):
                self._write_surface(frame)

        self.buffer.clear()

    def close(self):
        """Stops the recording."""
        self.flush()
        if self.recording:
            self.recording.release()
            self.recording = None

    def __get_filename(self):
        directory = os.path.join(os.getcwd(), "recording")
        if not os.path.exists(directory):
            os.makedirs(directory)

        i = 0
        while True:
            filename = os.path.join(directory, f"gameplay_{i:03d}.avi")
            if not os.path.exists(filename):
                return filename
            i += 1

class DisplayWindow:
    """A window to display the game and the AI model."""
    def __init__(self, scenario : ZeldaScenario, model_path : str):
        self.scenario = scenario
        self.orchestrator = ModelSelector()

        pygame.init()

        self.font = pygame.font.Font(None, 24)

        game_w, game_h = 240, 224
        self.scale = 4

        self.game_width = game_w * self.scale
        self.game_height = game_h * self.scale

        self.obs_width = 128
        self.obs_height = self.game_height
        self.obs_x = 0
        self.obs_y = 0

        self.game_x = self.obs_width
        self.game_y = 0

        self.details_height = 150
        self.details_width = self.game_width + self.obs_width
        self.details_x = 0
        self.details_y = self.game_height

        self.text_x = self.obs_width + self.game_width
        self.text_y = 0
        self.text_height = self.game_height + self.details_height
        self.text_width = 300

        self.total_width = self.obs_width + self.game_width + self.text_width
        self.total_height = max(self.game_height + self.details_height, self.text_height)
        self.dimensions = (self.total_width, self.total_height)

        self.total_rewards = 0.0
        self._last_location = None
        self.start_time = None

        self._available_models = {}
        self._loaded_models = {}
        self.model_path = model_path

    def show(self, headless_recording=False):
        """Shows the game and the AI model."""
        env = make_zelda_env(self.scenario, 'full', render_mode='rgb_array')
        rgb_deque = self._get_rgb_deque(env)

        clock = pygame.time.Clock()

        endings = {}
        reward_map = {}
        buttons = deque(maxlen=100)
        next_action = None

        model_requested = 0
        model_name = None

        show_endings = False
        force_save = False
        recording = None
        cap_fps = True
        overlay = 0

        if headless_recording:
            recording = Recording(self.dimensions, 1)
            force_save = True
            cap_fps = False
            surface = pygame.Surface(self.dimensions)
            print("Headless recording started")
        else:
            surface = pygame.display.set_mode(self.dimensions)

        terminated = True
        truncated = False

        last_info = {}
        info = {}
        state_change : ZeldaStateChange = None

        # modes: c - continue, n - next, r - reset, p - pause, q - quit
        mode = 'c'
        while mode != 'q':
            if terminated or truncated:
                if 'end' in info:
                    endings[info['end']] = endings.get(info['end'], 0) + 1

                last_info = info
                obs, info = env.reset()
                self.total_rewards = 0.0
                reward_map.clear()

                # we use buffer_size to check if we only want to record on a win
                if recording:
                    if not force_save and recording.buffer_size > 1 and not last_info['triforce']:
                        recording.buffer.clear()

                    recording.close()

                force_save = False

            # Perform a step in the environment
            if mode in ('c', 'n'):
                if next_action:
                    action = next_action
                    model_name = "keyboard input"
                    next_action = None
                else:
                    model_name, action = self._get_action_from_model(model_requested, info['state'], obs)

                last_info = info
                obs, _, terminated, truncated, info = env.step(action)
                state_change : ZeldaStateChange = info['state_change']

                if mode == 'n':
                    mode = 'p'

            # update rewards for display
            self._update_rewards(env, reward_map, buttons, last_info, info)

            while True:
                if rgb_deque:
                    rgb_array = rgb_deque.popleft()
                elif mode != 'p':
                    break

                surface.fill((0, 0, 0))

                self._show_observation(surface, obs)

                # render the gameplay
                self._render_game_view(surface, rgb_array, (self.game_x, self.game_y), self.game_width,
                                       self.game_height)
                if overlay:
                    color = "black" if info['level'] == 0 and not state_change.current.in_cave else "white"
                    self._overlay_grid_and_text(surface, overlay, (self.game_x, self.game_y), color, \
                                                self.scale, state_change.current)

                render_text(surface, self.font, f"Model: {model_name}", (self.game_x, self.game_y))
                if "location" in info:
                    render_text(surface, self.font, f"Location: {hex(info['location'])}",
                                (self.game_x + self.game_width - 120, self.game_y))

                # render rewards graph and values
                ending_render = endings if show_endings else None
                self._draw_details(surface, reward_map, ending_render)
                rendered_buttons = self._draw_reward_buttons(surface, buttons, (self.text_x, self.text_y),
                                                            (self.text_width, self.text_height))

                if recording:
                    recording.write(surface)

                # Display the scaled frame
                if not headless_recording:
                    pygame.display.flip()

                else:
                    self._print_location_info(info)

                if cap_fps:
                    clock.tick(60.1)

                # Check for Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        mode = 'q'
                        break

                    if event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:
                            for rendered_button in rendered_buttons:
                                if rendered_button.is_position_within(event.pos):
                                    rendered_button.button.on_click()
                                    break

                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            mode = 'q'
                            break

                        if event.key == pygame.K_r:
                            terminated = truncated = True
                            break

                        if event.key == pygame.K_p:
                            mode = 'p'

                        elif event.key == pygame.K_n:
                            mode = 'n'

                        elif event.key == pygame.K_c:
                            mode = 'c'

                        elif event.key == pygame.K_o:
                            overlay = (overlay + 1) % 5

                        elif event.key == pygame.K_e:
                            show_endings = not show_endings

                        elif event.key == pygame.K_m:
                            model_requested += 1

                        elif event.key == pygame.K_u:
                            cap_fps = not cap_fps

                        elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
                            keys = pygame.key.get_pressed()
                            next_action = self._get_action_from_keys(keys)
                            mode = 'n'

                        elif event.key == pygame.K_s:
                            if not force_save:
                                force_save = True
                                print("Saving this video")

                        elif event.key == pygame.K_F4:
                            if recording is None:
                                recording = Recording(self.dimensions, 0)
                                print("Live recording started")

                            else:
                                print("Live recording stopped")
                                recording.close()
                                recording = None

                        elif event.key == pygame.K_F10:
                            if recording is None:
                                recording = Recording(self.dimensions, 1_000_000_000)
                                print("Frame recording started")
                            else:
                                # don't close the recording here, we don't want to save the buffer if we didn't
                                # win the scenario
                                print("Frame recording stopped")
                                recording = None


        if recording and recording.buffer_size <= 1:
            recording.close()

        env.close()
        pygame.quit()

    def _get_action_from_model(self, model_requested, state, obs):
        zelda_model = self._select_model(state)
        ai, loaded_name = self._select_available_model(zelda_model, model_requested)
        model_name = f"{zelda_model.name} ({loaded_name}) {ai.num_timesteps:,} timesteps"

        action = ai.predict(obs, deterministic=False)
        return model_name, action

    def _get_action_from_keys(self, keys):
        a = ["A"] if keys[pygame.K_a] else []
        b = ["B"] if keys[pygame.K_s] else []
        if keys[pygame.K_LEFT]:
            return ['LEFT'] + a + b
        if keys[pygame.K_RIGHT]:
            return ['RIGHT'] + a + b
        if keys[pygame.K_UP]:
            return ['UP'] + a + b
        if keys[pygame.K_DOWN]:
            return ['DOWN'] + a + b

        return None


    def _select_model(self, state) -> ZeldaModelDefinition:
        acceptable_models = self.orchestrator.find_acceptable_models(state)
        for model in acceptable_models:
            if (models_available := self._available_models.get(model.name)) is None:
                models_available = model.find_available_models(self.model_path)
                self._available_models[model.name] = models_available

            if models_available:
                return model

        return None

    def _select_available_model(self, model : ZeldaModelDefinition, index : int) -> ZeldaAI:
        models_available = self._available_models[model.name]
        names = sorted(models_available.keys(), key=lambda x: int(x) if isinstance(x, int) else -1)

        name = names[index % len(names)]
        path = models_available[name]
        if (result := self._loaded_models.get(path, None)) is None:
            result = ZeldaAI(model)
            result.load(path)
            self._loaded_models[path] = result

        return result, name

    def _get_rgb_deque(self, env):
        while env is not None:
            if isinstance(env, FrameCaptureWrapper):
                return env.rgb_deque

            env = env.env

        return None

    def _print_location_info(self, info):
        if self._last_location is not None:
            last_level, last_location = self._last_location
            if last_level != info['level']:
                if info['level'] == 0:
                    print("Overworld")
                else:
                    print(f"Dungeon {info['level']}")

            if last_location != info['location']:
                print(f"Location: {hex(last_location)} -> {hex(info['location'])}")
        else:
            print("Overworld" if info['level'] == 0 else f"Dungeon {info['level']}")
            print(f"Location: {hex(info['location'])}")

        self._last_location = (info['level'], info['location'])

    def _print_end_info(self, info, terminated):
        total_time = (pygame.time.get_ticks() - self.start_time) / 1000
        term = "terminated" if terminated else "truncated"
        result = f"Episode {term} with {self.total_rewards:.2f} rewards"
        result += f", ending: {info.get('end', '???')} in {total_time:.2f} seconds"
        print(result)

    def _show_observation(self, surface, obs):
        x_pos = self.obs_x
        y_pos = self.obs_y
        y_pos = self._render_observation_view(surface, x_pos, y_pos, obs["image"])
        y_pos = self._draw_arrow(surface, "Objective", (x_pos + self.obs_width // 4, y_pos), obs["vectors"][0],
                                radius=self.obs_width // 4, color=(255, 255, 255), width=3)

        y_pos = self._draw_arrow(surface, "Enemy", (x_pos + self.obs_width // 4, y_pos), obs["vectors"][1],
                                radius=self.obs_width // 4, color=(255, 255, 255), width=3)

        y_pos = self._draw_arrow(surface, "Aligned", (x_pos + self.obs_width // 4, y_pos), obs["vectors"][4],
                                radius=self.obs_width // 4, color=(255, 255, 255), width=3)

        y_pos = self._draw_arrow(surface, "Projectile", (x_pos + self.obs_width // 4, y_pos), obs["vectors"][2],
                                radius=self.obs_width // 4, color=(255, 0, 0), width=3)

        y_pos = self._draw_arrow(surface, "Item", (x_pos + self.obs_width // 4, y_pos), obs["vectors"][3],
                                radius=self.obs_width // 4, color=(255, 255, 255), width=3)

        y_pos = self._write_key_val_aligned(surface, "Enemies", f"{obs['features'][0]:.1f}", x_pos, y_pos,
                                           self.obs_width)
        y_pos = self._write_key_val_aligned(surface, "Beams", f"{obs['features'][1]:.1f}", x_pos, y_pos, self.obs_width)

    def _update_rewards(self, env, reward_map, buttons, last_info, info):
        curr_rewards = {}
        if 'rewards' in info:
            for k, v in info['rewards'].items():
                if k not in reward_map:
                    reward_map[k] = 0
                reward_map[k] += v
                curr_rewards[k] = round(v, 2)
                self.total_rewards += v

        prev = buttons[0] if buttons else None
        action = "+".join(info['buttons'])
        if prev is not None and prev.rewards == curr_rewards and prev.action == action:
            prev.count += 1
        else:
            on_press = DebugReward(env, self.scenario, last_info, info)
            buttons.appendleft(RewardButton(self.font, 1, curr_rewards, action, self.text_width, on_press))

    def _draw_arrow(self, surface, label, start_pos, direction, radius=128, color=(255, 0, 0), width=5):
        render_text(surface, self.font, label, (start_pos[0], start_pos[1]))
        circle_start = (start_pos[0], start_pos[1] + 20)
        centerpoint = (circle_start[0] + radius, circle_start[1] + radius)
        end_pos = (centerpoint[0] + direction[0] * radius, centerpoint[1] + direction[1] * radius)

        pygame.draw.circle(surface, (255, 255, 255), centerpoint, radius, 1)

        if direction[0] != 0 or direction[1] != 0:
            pygame.draw.line(surface, color, centerpoint, end_pos, width)

            # Arrowhead
            arrowhead_size = 10
            angle = math.atan2(-direction[1], direction[0]) + math.pi

            left = (end_pos[0] + arrowhead_size * math.cos(angle - math.pi / 6),
                    end_pos[1] - arrowhead_size * math.sin(angle - math.pi / 6))
            right = (end_pos[0] + arrowhead_size * math.cos(angle + math.pi / 6),
                    end_pos[1] - arrowhead_size * math.sin(angle + math.pi / 6))

            pygame.draw.polygon(surface, color, [end_pos, left, right])

        return circle_start[1] + radius * 2

    def _render_observation_view(self, surface, x, y, img):
        render_text(surface, self.font, "Observation", (x, y))
        y += 20

        if len(img.shape) == 4:
            for i in range(img.shape[0]):
                y = self._render_one_observation(surface, x, y, img[i])

            return y

        return self._render_one_observation(surface, x, y, img)

    def _render_one_observation(self, surface, x, y, img):
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        observation_surface = pygame.surfarray.make_surface(np.swapaxes(img, 0, 1))
        observation_surface = pygame.transform.scale(observation_surface, (img.shape[1], img.shape[0]))
        surface.blit(observation_surface, (x, y))

        y += img.shape[0]
        return y

    def _render_game_view(self, surface, rgb_array, pos, game_width, game_height):
        frame = pygame.surfarray.make_surface(np.swapaxes(rgb_array, 0, 1))
        scaled_frame = pygame.transform.scale(frame, (game_width, game_height))
        surface.blit(scaled_frame, pos)

    def _write_key_val_aligned(self, surface, text, value, x, y, total_width, color=(255, 255, 255)):
        new_y = render_text(surface, self.font, text, (x, y), color)
        value_width, _ = self.font.size(value)
        render_text(surface, self.font, value, (x + total_width - value_width, y), color)
        return new_y

    def _draw_details(self, surface, rewards, endings):
        col = 0
        row = 1
        col_width = self.details_width // 3 - 3

        x = self.details_x
        y = self._write_key_val_aligned(surface, "Total Rewards:", f"{self.total_rewards:.2f}", x,
                                        self.details_y, col_width)
        row_height = y - self.details_y
        row_max = self.details_height // row_height
        items = list(rewards.items())
        items.sort(key=lambda x: x[1], reverse=True)
        for k, v in items:
            color = (255, 0, 0) if v < 0 else (0, 255, 255) if v > 0 else (255, 255, 255)
            self._write_key_val_aligned(surface, f"{k}:", f"{round(v, 2)}", x + col * col_width,
                                        self.details_y + row * row_height, col_width, color)
            row, col = self.__increment(row, col, row_max)

        if endings:
            row, col = self.__increment(row, col, row_max)
            self._write_key_val_aligned(surface, "Episodes:", f"{sum(endings.values())}", x + col * col_width,
                                        self.details_y + row * row_height, col_width)

            items = list(endings.items())
            items.sort(key=lambda x: x[1], reverse=True)
            for k, v in items:
                row, col = self.__increment(row, col, row_max)
                self._write_key_val_aligned(surface, f"{k}:", f"{v}", x + col * col_width,
                                            self.details_y + row * row_height, col_width)

    def __increment(self, row, col, row_max):
        row += 1
        if row >= row_max:
            row = 0
            col += 1
        return row, col

    def _draw_reward_buttons(self, surface, buttons : deque, position, dimensions):
        result = []

        x, y = position
        height = dimensions[1]

        i = 0
        while i < len(buttons) and y < position[1] + height:
            button = buttons[i]
            rendered_button = button.draw_reward_button(surface, (x, y))
            result.append(rendered_button)
            y += rendered_button.dimensions[1] + 1
            i += 1

        # remove unrendered buttons
        while len(buttons) > i:
            buttons.pop()

        return result

    def _overlay_grid_and_text(self, surface, kind, offset, text_color, scale, state : ZeldaGame):
        if not kind:
            return

        tiles = state.room.tiles
        wavefront = state.wavefront

        grid_width = 32
        grid_height = 22
        tile_width = 8 * scale
        tile_height = 8 * scale

        # Pygame font setup
        font_size = int(min(tile_width, tile_height) // 2)
        font = pygame.font.Font(None, font_size)

        for tile_x in range(grid_width):
            for tile_y in range(grid_height):
                x = offset[0] + tile_x * tile_width - 8 * scale
                y = 56 * scale + offset[1] + tile_y * tile_height

                color = (0, 0, 0)

                pygame.draw.rect(surface, color, (x, y, tile_width, tile_height), 1)

                if kind == 1:
                    tile_number = wavefront.get((tile_x, tile_y), None)
                    text = f"{tile_number:02X}" if tile_number is not None else ""
                elif kind == 2:
                    tile_number = tiles[tile_x, tile_y]
                    text = f"{tile_number:02X}" if tile_number is not None else ""
                elif kind == 3:
                    walkable = state.room.walkable[tile_x, tile_y]
                    text = "X" if walkable else ""
                else:
                    text = f"{tile_x:02X} {tile_y:02X}"

                # Render the text
                text_surface = font.render(text, True, text_color)
                text_rect = text_surface.get_rect(center=(x + tile_width // 2, y + tile_height // 2))

                # Draw the text
                surface.blit(text_surface, text_rect)


class DebugReward:
    """An action to take when a reward button is clicked."""
    def __init__(self, env, scenario : ZeldaScenario, last_info, info):
        self.env = env
        self.scenario = scenario
        self.last_info = last_info
        self.info = info

    def __call__(self):
        result = simulate_critique(self.env, self.scenario, self.last_info, self.info)
        reward_dict, terminated, truncated, reason = result
        print(f"{reward_dict = }")
        print(f"{terminated = }")
        print(f"{truncated = }")
        print(f"{reason = }")

class RewardButton:
    """A button to display a reward value."""
    def __init__(self, font, count, rewards, action, width, on_click):
        self.font = font
        self.count = count
        self.rewards = rewards
        self.action = action
        self.width = width
        self.on_click = on_click

    def draw_reward_button(self, surface, position) -> 'RenderedButton':
        """Draws the button on the surface. Returns a RenderedButton."""
        x = position[0] + 3
        y = position[1] + 2

        start_y = y
        y = render_text(surface, self.font, self.action, (x, y))
        if self.rewards:
            for reason, value in self.rewards.items():
                color = (255, 0, 0) if value < 0 else (0, 255, 255) if value > 0 else (255, 255, 255)
                next_y = render_text(surface, self.font, reason, (x, y), color=color)
                render_text(surface, self.font, f"{'+' if value > 0 else ''}{value:.2f}", (x + 200, y))
                y = next_y

        else:
            text = "none"
            color = (128, 128, 128)
            y = render_text(surface, self.font, text, (x, y), color)

        if self.count > 1:
            count_text = f"x{self.count}"
            count_text_width, _ = self.font.size(count_text)
            render_text(surface, self.font, count_text, (x + 275 - count_text_width, start_y))

        height = y - position[1]
        pygame.draw.rect(surface, (255, 255, 255), (position[0], position[1], self.width, height), 1)

        return RenderedButton(self, position, (self.width, height))

class RenderedButton:
    """A rendered button on screen, keeping track of its own dimensions."""
    def __init__(self, button, position, dimensions):
        self.button = button
        self.position = position
        self.dimensions = dimensions

    def is_position_within(self, position):
        """Returns True if the position is within the button."""
        x, y = position
        bx, by = self.position
        bw, bh = self.dimensions
        return bx <= x <= bx + bw and by <= y <= by + bh

def render_text(surface, font, text, position, color=(255, 255, 255)):
    """Render text on the surface and returns the new y position."""
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)
    return position[1] + text_surface.get_height()

def main():
    """Main function."""
    args = parse_args()
    model_path = args.model_path[0] if args.model_path else os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                         'models')
    if args.scenario is None:
        args.scenario = 'zelda'

    scenario = TRAINING_SCENARIOS.get(args.scenario, None)
    if not scenario:
        print(f'Unknown scenario {args.scenario}')
        return

    display = DisplayWindow(scenario, model_path)
    display.show(args.headless_recording)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Triforce - An ML agent to play The Legned of Zelda (NES).")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--color", action='store_true',
                        help="Give the model a color version of the game (instead of grayscale).")
    parser.add_argument("--obs-kind", choices=['gameplay', 'viewport', 'full'], default='viewport',
                        help="The kind of observation to use.")
    parser.add_argument("--model-path", nargs=1, help="Location to read models from.")
    parser.add_argument("--frame-stack", type=int, default=1, help="Number of frames the model was trained with.")
    parser.add_argument("--headless-recording", action='store_true', help="Record the game without displaying it.")

    parser.add_argument('scenario', nargs='?', help='Scenario name')

    try:
        args = parser.parse_args()
        return args

    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(e)
        parser.print_help()
        sys.exit(0)

if __name__ == '__main__':
    main()
