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

from triforce import ZeldaScenario, simulate_critique, make_zelda_env, TRAINING_SCENARIOS, Network
from triforce.state_change_wrapper import StateChange
from triforce.model_definition import ZELDA_MODELS, ZeldaModelDefinition
from triforce.rewards import StepRewards
from triforce.zelda_enums import ActionKind, Coordinates, Direction
from triforce.zelda_game import ZeldaGame

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
    def __init__(self, scenario : ZeldaScenario, model_path : str, model : str):
        self.scenario = scenario

        pygame.init()

        self.font = pygame.font.Font(None, 24)

        game_w, game_h = 240, 224
        self.scale = 4

        self.game_width = game_w * self.scale
        self.game_height = game_h * self.scale

        self.obs_width = 256
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

        self._loaded_models = {}
        self.model_path = model_path
        self.model_definition : ZeldaModelDefinition = ZELDA_MODELS[model]

        self.move_widgets = {}
        self.vector_widgets = []
        self.objective_widgets = []

    def show(self, headless_recording=False):
        """Shows the game and the AI model."""
        env = make_zelda_env(self.scenario, self.model_definition.action_space, render_mode='rgb_array',
                             translation=False)

        clock = pygame.time.Clock()

        endings = {}
        reward_map = {}
        buttons = deque(maxlen=100)
        next_action = None

        model_requested = 0

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

        state_change : StateChange = None
        model_name = None

        rgb_deque = deque()

        # modes: c - continue, n - next, r - reset, p - pause, q - quit
        mode = 'c'
        while mode != 'q':
            if terminated or truncated:

                if state_change is not None and 'end' in state_change.state.info:
                    self._print_end_info(state_change.state.info, terminated)

                obs, state = env.reset()
                action_mask = state.info['action_mask']
                self.start_time = pygame.time.get_ticks()
                state_change = None
                self.total_rewards = 0.0
                reward_map.clear()

                # we use buffer_size to check if we only want to record on a win
                if recording:
                    if not force_save and recording.buffer_size > 1 and \
                            state_change is not None and not state_change.previous.info.get('triforce', 0):
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
                    model, model_name = self._select_model(env, model_requested)
                    if action_mask is not None:
                        action_mask = action_mask.unsqueeze(0)
                    action = model.get_action(obs, action_mask, deterministic=False)
                    success_rate = model.stats.success_rate * 100 if model.stats else 0
                    model_name = f"{self.model_definition.name} ({model.steps_trained:,} timesteps {success_rate:.1f}%)"

                obs, _, terminated, truncated, state_change = env.step(action)
                action_mask = state_change.state.info['action_mask']
                for frame in state_change.frames:
                    rgb_deque.append(frame)

                if mode == 'n':
                    mode = 'p'

            # update rewards for display
            self._update_rewards(env, action, reward_map, buttons, state_change)

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

                state = state_change.state
                if overlay:
                    color = "black" if state.level == 0 and not state.in_cave else "white"
                    self._overlay_grid_and_text(surface, overlay, (self.game_x, self.game_y), color, \
                                                self.scale, state_change.state)

                render_text(surface, self.font, f"Model: {model_name}", (self.game_x, self.game_y))
                render_text(surface, self.font, f"Location: {hex(state.location)}",
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
                    self._print_location_info(state)

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

                        elif event.key == pygame.K_l:
                            model_requested -= 1

                        elif event.key == pygame.K_u:
                            cap_fps = not cap_fps

                        elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
                            keys = pygame.key.get_pressed()
                            next_action = self._get_action_from_keys(state, keys)
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

    def _get_action_from_keys(self, state : ZeldaGame, keys):
        # pylint: disable=too-many-return-statements
        if keys[pygame.K_LEFT]:
            direction = Direction.W
        elif keys[pygame.K_RIGHT]:
            direction = Direction.E
        elif keys[pygame.K_UP]:
            direction = Direction.N
        elif keys[pygame.K_DOWN]:
            direction = Direction.S
        else:
            return None

        model_actions = ActionKind.get_from_list(self.model_definition.action_space)
        link_actions = state.link.get_available_actions(ActionKind.BEAMS in model_actions)
        available = link_actions & model_actions
        sword_available = ActionKind.SWORD in available or ActionKind.BEAMS in available
        if keys[pygame.K_a]:
            if not sword_available:
                return None

            if ActionKind.BEAMS in self.model_definition.action_space:
                return (ActionKind.BEAMS, direction)

            return (ActionKind.SWORD, direction)

        if keys[pygame.K_s]:
            equipment = ActionKind.from_selected_equipment(state.link.selected_equipment)
            if equipment.is_equipment and equipment in available:
                return (equipment, direction)

            return None

        return (ActionKind.MOVE, direction)

    def _select_model(self, env, index : int) -> Network:
        models_available = self.model_definition.find_available_models(self.model_path)
        names = sorted(models_available.keys(), key=lambda x: int(x) if isinstance(x, int) else -1)
        names.append("untrained")

        name = names[index % len(names)]
        path = models_available.get(name)
        if (network := self._loaded_models.get(path, None)) is None:
            network : Network = self.model_definition.neural_net(env.observation_space, env.action_space)
            if name != "untrained":
                network.load(path)
            self._loaded_models[path] = network

        return network, name

    def _print_location_info(self, state):
        if self._last_location is not None:
            last_level, last_location = self._last_location
            if last_level != state.level:
                if state.level == 0:
                    print("Overworld")
                else:
                    print(f"Dungeon {state.level}")

            if last_location != state.location:
                print(f"Location: {hex(last_location)} -> {hex(state.location)}")
        else:
            print("Overworld" if state.level == 0 else f"Dungeon {state.level}")
            print(f"Location: {hex(state.location)}")

        self._last_location = (state.level, state.location)

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

        radius = self.obs_width // 4
        if not self.vector_widgets:
            pos = Coordinates(x_pos, y_pos)
            labels = ["Enemy1", "Enemy 2", "Projectile", "Projectile 2", "Item 1", "Item 2"]
            for label in labels:
                self.vector_widgets.append(LabeledVector(pos, self.font, label, radius))
                size = self.vector_widgets[-1].size
                if len(self.vector_widgets) % 2 == 0:
                    pos = pos + (-size[0], size[1])
                else:
                    pos = pos + (size[0], 0)

        for i in range(obs["vectors"].shape[0]):
            for j in range(obs["vectors"].shape[1]):
                v = obs["vectors"][i, j]
                widget = self.vector_widgets[i * obs["vectors"].shape[1] + j]
                widget.vector = v.cpu().numpy()
                widget.draw(surface)


        if not self.move_widgets:
            pos = Coordinates(0, pos.y)
            self.move_widgets['move-objective'] = DirectionalCircle(pos, self.font, "Objective", radius)
            pos = pos + (self.move_widgets['move-objective'].size[0], 0)
            self.move_widgets['came-from'] = DirectionalCircle(pos, self.font, "Source", radius)

        self.move_widgets['move-objective'].directions = self._get_directions_for_vectors(obs["information"][:6])
        self.move_widgets['move-objective'].draw(surface)

        self.move_widgets['came-from'].directions = self._get_directions_for_vectors(obs["information"][6:10])
        self.move_widgets['came-from'].draw(surface)

        return Coordinates(0, y_pos)



    def _get_directions_for_vectors(self, vectors):
        directions = []
        if vectors[0] > 0:
            directions.append(Direction.N)
        if vectors[1] > 0:
            directions.append(Direction.S)
        if vectors[2] > 0:
            directions.append(Direction.E)
        if vectors[3] > 0:
            directions.append(Direction.W)

        return directions


    def _update_rewards(self, env, action, reward_map, buttons, state_change : StateChange):
        curr_rewards = {}
        last_info = state_change.previous.info
        info = state_change.state.info
        rewards : StepRewards = info.get('rewards', None)
        if rewards is not None:
            self.total_rewards += rewards.value
            for outcome in rewards:
                if outcome.name not in reward_map:
                    reward_map[outcome.name] = 0

                reward_map[outcome.name] += outcome.value
                curr_rewards[outcome.name] = outcome.value

        prev = buttons[0] if buttons else None
        action = f"{state_change.action.direction.name} {state_change.action.kind.name}"
        if prev is not None and prev.rewards == curr_rewards and prev.action == action:
            prev.count += 1
        else:
            on_press = DebugReward(env, action, self.scenario, last_info, info)
            buttons.appendleft(RewardButton(self.font, 1, curr_rewards, action, self.text_width, on_press))

    def _render_observation_view(self, surface, x, y, img):
        render_text(surface, self.font, "Observation", (x, y))
        y += 20

        if len(img.shape) == 4:
            for i in range(img.shape[0]):
                y = self._render_one_observation(surface, x, y, img[i])

            return y

        return self._render_one_observation(surface, x, y, img)

    def _render_one_observation(self, surface, x, y, img):
        img = img.squeeze(0).cpu().numpy()

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        img = (img * 255).astype(np.uint8)

        observation_surface = pygame.surfarray.make_surface(np.swapaxes(img, 0, 1))
        observation_surface = pygame.transform.scale(observation_surface, (img.shape[1] * 2, img.shape[0] * 2))
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



class LabeledCircle:
    """A circle with a label."""
    def __init__(self, position, font, label, radius=128, color=(255, 0, 0), width=5):
        assert isinstance(position, Coordinates)
        self.position = position
        self.radius = int(radius)
        self.color = color
        self.width = width
        self.font = font
        self.label = label

    @property
    def size(self):
        """Returns the size of the draw area."""
        return self.radius * 2, self.radius * 2 + 20

    @property
    def centerpoint(self):
        """Returns the centerpoint of the circle."""
        circle_start = self.position + (0, 20)
        centerpoint = circle_start + (self.radius, self.radius)
        return centerpoint

    def draw(self, surface):
        """Draws the labeled circle on the surface."""
        render_text(surface, self.font, self.label, self.position)
        pygame.draw.circle(surface, (255, 255, 255), self.centerpoint, self.radius, 1)

    def _draw_arrow(self, surface, centerpoint, vector, length):
        length = np.clip(length, 0.05, 1)
        arrow_end = np.array(centerpoint) + vector[:2] * self.radius * length
        if vector[0] != 0 or vector[1] != 0:
            pygame.draw.line(surface, self.color, centerpoint, arrow_end, self.width)

            # Arrowhead
            arrowhead_size = 10
            angle = math.atan2(-vector[1], vector[0]) + math.pi

            left = arrow_end + (arrowhead_size * math.cos(angle - math.pi / 6),
                              -arrowhead_size * math.sin(angle - math.pi / 6))
            right = arrow_end + (arrowhead_size * math.cos(angle + math.pi / 6),
                               -arrowhead_size * math.sin(angle + math.pi / 6))

            pygame.draw.polygon(surface, self.color, [arrow_end, left, right])

class DirectionalCircle(LabeledCircle):
    """A vector with a label."""
    def __init__(self, position, font, label, radius=128, color=(255, 0, 0), width=5):
        super().__init__(position, font, label, radius, color, width)
        self._directions = []

    @property
    def directions(self):
        """Returns the directions."""
        return self._directions

    @directions.setter
    def directions(self, value):
        self._directions = value

    def draw(self, surface):
        """Draws the labeled vector on the surface."""
        super().draw(surface)

        for direction in self.directions:
            match direction:
                case Direction.N:
                    self._draw_arrow(surface, self.centerpoint, np.array([0, -1], dtype=np.float32), 1)
                case Direction.S:
                    self._draw_arrow(surface, self.centerpoint, np.array([0, 1], dtype=np.float32), 1)
                case Direction.W:
                    self._draw_arrow(surface, self.centerpoint, np.array([-1, 0], dtype=np.float32), 1)
                case Direction.E:
                    self._draw_arrow(surface, self.centerpoint, np.array([1, 0], dtype=np.float32), 1)
                case _:
                    raise ValueError(f"Unsupported direction {direction}")


        if self.directions:
            pygame.draw.circle(surface, (0, 0, 0), self.centerpoint, 5)

class LabeledVector(LabeledCircle):
    """A vector with a label."""
    def __init__(self, position, font, label, radius=128, color=(255, 0, 0), width=5):
        super().__init__(position, font, label, radius, color, width)
        self._vector = [0, 0, -1]

    @property
    def vector(self):
        """Returns the vector."""
        return self._vector

    @vector.setter
    def vector(self, value):
        assert len(value) in (2, 3)
        self._vector = value

    def draw(self, surface):
        """Draws the labeled vector on the surface."""
        super().draw(surface)
        dist = self._vector[2] if len(self._vector) == 3 else 1
        self._draw_arrow(surface, self.centerpoint, self._vector, dist)

class DebugReward:
    """An action to take when a reward button is clicked."""
    def __init__(self, env, action, scenario : ZeldaScenario, last_info, info):
        self.env = env
        self.scenario = scenario
        self.last_info = last_info
        self.info = info
        self.action = action

    def __call__(self):
        result = simulate_critique(self.env, self.action, self.scenario, self.last_info, self.info)
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
    scenario = TRAINING_SCENARIOS.get(args.scenario, None)
    if not scenario:
        print(f'Unknown scenario {args.scenario}')
        return

    display = DisplayWindow(scenario, model_path, args.model)
    display.show(args.headless_recording)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Triforce - An ML agent to play The Legned of Zelda (NES).")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--obs-kind", choices=['gameplay', 'viewport', 'full'], default='viewport',
                        help="The kind of observation to use.")
    parser.add_argument("--model-path", nargs=1, help="Location to read models from.")
    parser.add_argument("--headless-recording", action='store_true', help="Record the game without displaying it.")

    parser.add_argument('scenario', type=str, help='Scenario name')
    parser.add_argument('model', type=str, help='Model name')

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
