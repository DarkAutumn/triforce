#! /usr/bin/python
"""Run the ZeldaML agent to play The Legend of Zelda (NES)."""

# While we want to keep this file relatively clean, it's fine to have a bit of a large render function.

# pylint: disable=too-few-public-methods,too-many-locals,too-many-branches,too-many-statements
# pylint: disable=too-many-nested-blocks,duplicate-code

from collections import deque
import pygame
import numpy as np
import torch

from triforce import TrainingScenarioDefinition,ModelDefinition
from triforce.rewards import StepRewards
from triforce.zelda_enums import ActionKind, Coordinates, Direction
from triforce.zelda_game import ZeldaGame

from .environment_wrapper import EnvironmentWrapper, StepResult
from .helpers import draw_text
from .labeled_circle import DirectionalCircle, LabeledVector
from .recording import Recording
from .reward_button import RewardButton


class RewardDebugger:
    """A window to display the game and the AI model."""
    def __init__(self, scenario : TrainingScenarioDefinition, model_path : str, model : str, frame_stack):
        self.scenario = scenario

        pygame.init()

        self.frame_stack = frame_stack

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
        self.details_width = self.game_width
        self.details_x = self.obs_width
        self.details_y = self.game_height

        self.text_x = self.obs_width + self.game_width
        self.text_height = self.game_height + self.details_height
        self.text_width = 300

        self.action_x = self.text_x - self.obs_width
        self.action_y = 20
        self.action_height = 100
        self.action_width = self.action_height * 4

        self._move_circle = DirectionalCircle(Coordinates(self.action_x, self.action_y), self.font, "Move",
                                              self.action_height / 2, center_circle=False, scale_range=(0.15, 1))
        self._attack_circle = DirectionalCircle(Coordinates(self.action_x + self.action_height, self.action_y),
                                                self.font, "Attack", self.action_height / 2, center_circle=False,
                                                scale_range=(0.15, 1))

        self.probs_x = self.text_x
        self.probs_y = 0
        self.probs_height = 0
        self.probs_width = self.text_width

        self.total_width = self.obs_width + self.game_width + self.text_width
        self.total_height = max(self.game_height + self.details_height, self.text_height)
        self.dimensions = (self.total_width, self.total_height)

        self._last_location = None
        self.start_time = None

        self.model_path = model_path
        self.model_definition : ModelDefinition = ModelDefinition.get(model)
        if not self.model_definition:
            raise ValueError(f"Unknown model {model}")

        self.move_widgets = {}
        self.vector_widgets = {}
        self.objective_widgets = []

        self.mode = 'c'
        self.restart_requested = True
        self.cap_fps = True
        self.next_action = None
        self.overlay = 0
        self.buttons = deque(maxlen=100)
        self.endings = {}
        self.total_rewards = 0.0
        self.running_rewards = {}

        self.recording = None
        self.force_save = False

    @property
    def text_y(self):
        """Returns the y position for the text."""
        return self.probs_height

    def show(self):
        """Shows the game and the AI model."""
        clock = pygame.time.Clock()

        self.overlay = 0

        surface = pygame.display.set_mode(self.dimensions)
        env = EnvironmentWrapper(self.model_path, self.model_definition, self.scenario, self.frame_stack)

        # modes: c - continue, n - next, r - reset, p - pause, q - quit
        frame = None
        step = None
        frames = []
        while self.mode != 'q':
            if self.mode != 'p' and not frames:
                if self.restart_requested:
                    if self.recording:
                        self._save_recording((step and step.rewards.ending) or "")

                    step : StepResult = env.restart()
                    action_mask = step.action_mask_desc
                    self.restart_requested = False
                    self.total_rewards = 0.0
                    self.running_rewards.clear()

                else:
                    action_mask = step.action_mask_desc
                    step = env.step(self.next_action)
                    self.next_action = None
                    self.restart_requested = step.terminated or step.truncated

                frames = step.frames

                if self.mode == 'n':
                    self.mode = 'p'

                # update rewards for display
                self._update_rewards(step, action_mask)
                if step.terminated or step.truncated:
                    self.endings[step.rewards.ending] = self.endings.get(step.rewards.ending, 0) + 1


            probs = env.selector.get_probabilities(step.observation, step.action_mask.unsqueeze(0))
            if frames:
                # if we have frames, render them and save them
                frame = frames.pop(0)
                self._render_to_surface(surface, step, frame, env.model_details, probs)
                if self.recording:
                    self.recording.write(surface)

            else:
                # just render the frame in case we change models/probabilities
                self._render_to_surface(surface, step, frame, env.model_details, probs)

            # draw the surface to screen
            pygame.display.flip()
            self._check_input(env, step)

            if self.cap_fps:
                clock.tick(60.1)


        if self.recording:
            self.recording.close(False)

        env.close()
        pygame.quit()

    def _save_recording(self, ending):
        if ending or self.force_save:
            if self.recording.is_buffered:
                if self.force_save or ending.startswith("success"):
                    self.recording.close(True)
                else:
                    self.recording.close(False)
            else:
                self.recording.close(True)
        else:
            self.recording.clear()

        self.force_save = False

    def _render_to_surface(self, surface, step : StepResult, frame : torch.Tensor, model_details, probs):
        surface.fill((0, 0, 0))
        self._show_observation(surface, step.observation)

        # render the gameplay
        self._draw_game_view(surface, frame, (self.game_x, self.game_y), self.game_width, self.game_height)

        if self.overlay:
            color = "black" if step.state.level == 0 and not step.state.in_cave else "white"
            self._overlay_grid_and_text(surface, self.overlay, (self.game_x, self.game_y), color, \
                                        self.scale, step.state)

        draw_text(surface, self.font, f"Model: {model_details}", (self.game_x, self.game_y))
        draw_text(surface, self.font, f"Location: {hex(step.state.location)}",
                    (self.game_x + self.game_width - 120, self.game_y))

        # render rewards graph and values
        self._draw_details(surface)
        if probs:
            self._show_action(surface, probs)
        self._draw_probabilities(surface, probs)
        self._draw_reward_buttons(surface)
        #self._draw_location_info(step.state)

    def _check_input(self, env, step):
        # Check for Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.mode = 'q'
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.mode = 'q'
                    break

                if event.key == pygame.K_r:
                    self.restart_requested = True
                    break

                if event.key == pygame.K_p:
                    self.mode = 'p'

                elif event.key == pygame.K_n:
                    self.mode = 'n'

                elif event.key == pygame.K_c:
                    self.mode = 'c'

                elif event.key == pygame.K_o:
                    self.overlay = (self.overlay + 1) % 5

                elif event.key == pygame.K_e:
                    show_endings = not show_endings

                elif event.key == pygame.K_m:
                    env.selector.next()

                elif event.key == pygame.K_l:
                    env.selector.previous()

                elif event.key == pygame.K_u:
                    self.cap_fps = not self.cap_fps

                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
                    keys = pygame.key.get_pressed()
                    self.next_action = self._get_action_from_keys(step.state.link, keys)
                    if env.is_valid_action(self.next_action):
                        self.mode = 'n'
                    else:
                        print(f"Invalid action: {self.next_action}")
                        self.next_action = None

                elif event.key == pygame.K_s:
                    if self.recording and self.recording.is_buffered and not self.force_save:
                        self.force_save = True
                        print("Saving this video")

                elif event.key == pygame.K_F4:
                    if self.recording is None:
                        self.recording = Recording(self.dimensions, 0)
                        print("Live recording started")

                    else:
                        print("Live recording stopped")
                        self.recording.close(True)
                        self.recording = None

                elif event.key == pygame.K_F10:
                    if self.recording is None:
                        self.recording = Recording(self.dimensions, 1_000_000_000)
                        print("Frame recording started")
                    else:
                        # don't close the recording here, we don't want to save the buffer if we didn't
                        # win the scenario
                        print("Frame recording stopped")
                        self.recording = None


    def _get_action_from_keys(self, link, keys):
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

        if keys[pygame.K_a]:
            beams_separated = ActionKind.BEAMS in self.model_definition.action_space
            action = ActionKind.BEAMS if beams_separated and link.has_beams else ActionKind.SWORD
            return action, direction

        return ActionKind.MOVE, direction

    def _draw_location_info(self, state):
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

    def _show_action(self, surface, probs):
        if self._move_circle is None:
            pos = Coordinates(self.action_x, self.action_y)
            self._move_circle = DirectionalCircle(pos, self.font, "Move", 25)
            self._attack_circle = DirectionalCircle(pos + (0, 50), self.font, "Attack", 25)

        self._move_circle.directions.clear()
        self._attack_circle.directions.clear()

        for action, l in probs.items():
            if action == 'value':
                continue

            for direction, prob in l:
                if prob < 0.01:
                    continue

                if action == ActionKind.MOVE:
                    self._move_circle.directions.append((direction, prob))
                else:
                    self._attack_circle.directions.append((direction, prob))

        # clear background
        pygame.draw.rect(surface, (0, 0, 0), (self.action_x, self.action_y, self.action_width, self.action_height))

        self._move_circle.draw(surface)
        self._attack_circle.draw(surface)

    def _show_observation(self, surface, obs):
        x_pos = self.obs_x
        y_pos = self.obs_y
        y_pos = self._render_observation_view(surface, x_pos, y_pos, obs["image"][-1])

        radius = self.obs_width // 4
        if not self.vector_widgets:
            pos = Coordinates(x_pos, y_pos)
            labels = ["Enemy 1", "Enemy 2", "Enemy 3", "Enemy 4", "Projectile 1", "Projectile 2", "Item 1", "Item 2"]
            for label in labels:
                widget = LabeledVector(pos, self.font, label, radius)
                self.vector_widgets[label] = widget
                size = widget.size
                if len(self.vector_widgets) % 2 == 0:
                    pos = pos + (-size[0], size[1])
                else:
                    pos = pos + (size[0], 0)

        self._update_vector_widget("Enemy 1", "enemy_features", 0, obs)
        self._update_vector_widget("Enemy 2", "enemy_features", 1, obs)
        self._update_vector_widget("Enemy 3", "enemy_features", 2, obs)
        self._update_vector_widget("Enemy 4", "enemy_features", 3, obs)
        self._update_vector_widget("Projectile 1", "projectile_features", 0, obs)
        self._update_vector_widget("Projectile 2", "projectile_features", 1, obs)
        self._update_vector_widget("Item 1", "item_features", 0, obs)
        self._update_vector_widget("Item 2", "item_features", 1, obs)

        for widget in self.vector_widgets.values():
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

    def _update_vector_widget(self, widget_name, observation_name, index, obs):
        widget = self.vector_widgets[widget_name]
        widget.vector = obs[observation_name][index, 2:4].cpu().numpy()
        widget.scale = 1 - obs[observation_name][index, 1].cpu().numpy()

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

    def _update_rewards(self, step : StepResult, prev_action_mask):
        state_change = step.state_change
        if not state_change:
            return

        curr_rewards = {}
        rewards : StepRewards = step.rewards
        if rewards is not None:
            self.total_rewards += rewards.value
            for outcome in rewards:
                if outcome.name not in self.running_rewards:
                    self.running_rewards[outcome.name] = 0

                self.running_rewards[outcome.name] += outcome.value
                curr_rewards[outcome.name] = outcome.value

        buttons = self.buttons
        prev = buttons[0] if buttons else None
        action = f"{state_change.action.direction.name} {state_change.action.kind.name}"
        if prev is not None and prev.rewards == curr_rewards and prev.action == action:
            prev.count += 1
        else:
            buttons.appendleft(RewardButton(self.font, 1, curr_rewards, action, prev_action_mask,
                                            self.text_width))

    def _render_observation_view(self, surface, x, y, img):
        draw_text(surface, self.font, "Observation", (x, y))
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

        y += img.shape[0] * 2
        return y

    def _draw_game_view(self, surface, rgb_array, pos, game_width, game_height):
        frame = pygame.surfarray.make_surface(np.swapaxes(rgb_array, 0, 1))
        scaled_frame = pygame.transform.scale(frame, (game_width, game_height))
        surface.blit(scaled_frame, pos)

    def _write_key_val_aligned(self, surface, text, value, x, y, total_width, color=(255, 255, 255)):
        new_y = draw_text(surface, self.font, text, (x, y), color)
        value_width, _ = self.font.size(value)
        draw_text(surface, self.font, value, (x + total_width - value_width, y), color)
        return new_y

    def _draw_details(self, surface):
        col = 0
        row = 1
        col_width = self.details_width // 3 - 3

        x = self.details_x
        y = self._write_key_val_aligned(surface, "Total Rewards:", f"{self.total_rewards:.2f}", x,
                                        self.details_y, col_width)
        row_height = y - self.details_y
        row_max = self.details_height // row_height
        items = list(self.running_rewards.items())
        items.sort(key=lambda x: x[1], reverse=True)
        for k, v in items:
            color = (255, 0, 0) if v < 0 else (0, 255, 255) if v > 0 else (255, 255, 255)
            self._write_key_val_aligned(surface, f"{k}:", f"{round(v, 2)}", x + col * col_width,
                                        self.details_y + row * row_height, col_width, color)
            row, col = self.__increment(row, col, row_max)

        if self.endings:
            row, col = self.__increment(row, col, row_max)
            self._write_key_val_aligned(surface, "Episodes:", f"{sum(self.endings.values())}", x + col * col_width,
                                        self.details_y + row * row_height, col_width)

            items = list(self.endings.items())
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

    def _draw_probabilities(self, surface, probs):
        y = self.probs_y
        for action, l in probs.items():
            if action == 'value':
                draw_text(surface, self.font, f"Value: {l.item():.2f}", (self.probs_x, y))
                y += 20
                continue

            text = f"{action.name:>5}: "
            for direction, prob in l:
                prob = round(prob*100)
                if prob == 0:
                    continue
                if prob == 100:
                    prob = 99
                if prob > 0:
                    prob = f"{prob:>2}%"

                text += f"{direction.name}: {prob} "

            draw_text(surface, self.font, text, (self.probs_x, y))
            y += 20

        self.probs_height = max(y, self.probs_height)

    def _draw_reward_buttons(self, surface):
        result = []
        buttons = self.buttons

        i = 0
        y = self.text_y
        while i < len(buttons) and y < self.text_y + self.text_height:
            button = buttons[i]
            rendered_button = button.draw_reward_button(surface, (self.text_x, y))
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
