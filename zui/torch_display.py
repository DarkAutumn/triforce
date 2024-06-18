from collections import deque
import math
import numpy as np
import pygame
import torch

from triforce import make_multihead_zelda_env, ZeldaScenario, ZeldaMultiHeadNetwork
from triforce.zelda_game import is_in_cave
from triforce.ml_torch import SelectedAction, SelectedDirection

def render_text(surface, font, text, position, color=(255, 255, 255)):
    """Render text on the surface and returns the new y position."""
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)
    return position[1] + text_surface.get_height()

class DisplayWindowTorch:
    """A window to display the game and the AI model."""
    def __init__(self, scenario : ZeldaScenario, model_path : str):
        self.scenario = scenario

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

        self._last_location = None
        self.start_time = None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = ZeldaMultiHeadNetwork(128, 39, device)
        if model_path:
            self._model.load(model_path)

    def show(self, headless_recording=False):
        """Shows the game and the AI model."""
        rgb_deque = deque(maxlen=120)
        env = make_multihead_zelda_env(self.scenario.start[0], rgb_deque=rgb_deque)

        clock = pygame.time.Clock()

        endings = {}
        next_action = None

        model_requested = 0
        model_name = None

        show_endings = False
        cap_fps = True
        overlay = 0

        surface = pygame.display.set_mode(self.dimensions)

        terminated = True
        truncated = False

        info = {}
        sense_key = torch.zeros(5, dtype=torch.float32)

        rendered_buttons = []

        act_rew = deque(maxlen=16)
        sense_key = SelectedDirection.NONE
        pathfinding_key = SelectedDirection.N

        # modes: c - continue, n - next, r - reset, p - pause, q - quit
        mode = 'c'
        while mode != 'q':
            if terminated or truncated:
                if 'end' in info:
                    endings[info['end']] = endings.get(info['end'], 0) + 1

                obs, info = env.reset()
                act_rew.clear()


            # Perform a step in the environment
            if mode in ('c', 'n'):
                if next_action is not None:
                    action = next_action
                    model_name = "keyboard input"
                    next_action = None
                else:
                    action = self._get_action_from_model(obs, info)

                obs, rewards, terminated, truncated, info = env.step(action)
                act_rew.append((action, rewards))

                if mode == 'n':
                    mode = 'p'

            while True:
                if rgb_deque:
                    rgb_array = rgb_deque.popleft()
                elif mode != 'p':
                    break

                surface.fill((0, 0, 0))

                self._show_observation(surface, obs)
                self._show_act_rew(surface, act_rew, (self.text_x, self.text_y))

                # render the gameplay
                self._render_game_view(surface, rgb_array, (self.game_x, self.game_y), self.game_width,
                                       self.game_height)

                if overlay:
                    color = "black" if info['level'] == 0 and not is_in_cave(info) else "white"
                    self._overlay_grid_and_text(surface, overlay, (self.game_x, self.game_y), color, self.scale, info)
                render_text(surface, self.font, f"Model: {model_name}", (self.game_x, self.game_y))
                if "location" in info:
                    render_text(surface, self.font, f"Location: {hex(info['location'])}",
                                (self.game_x + self.game_width - 120, self.game_y))

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
                            overlay = (overlay + 1) % 2

                        elif event.key == pygame.K_e:
                            show_endings = not show_endings

                        elif event.key == pygame.K_m:
                            model_requested += 1

                        elif event.key == pygame.K_u:
                            cap_fps = not cap_fps

                        elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
                            pathfinding_key = self._get_pathfinding(pygame.key.get_pressed())
                            if pathfinding_key is not None:
                                items = [pathfinding_key, sense_key, SelectedAction.MOVEMENT]
                                items = [x.value for x in items]
                                next_action = torch.tensor(items, dtype=torch.int64).unsqueeze(0)
                                mode = 'n'

                        elif event.key == pygame.K_a:
                            action = SelectedAction.BEAMS if info['beams_available'] else SelectedAction.ATTACK
                            sense = sense_key if sense_key != SelectedDirection.NONE else pathfinding_key
                            items = [pathfinding_key, sense, action]
                            items = [x.value for x in items]
                            next_action = torch.tensor(items, dtype=torch.int64).unsqueeze(0)
                            mode = 'n'

                        elif event.key in (pygame.K_KP_8, pygame.K_KP_2, pygame.K_KP_4, pygame.K_KP_6, pygame.K_KP_5):
                            key_temp = self._map_sense_key(pygame.key.get_pressed())
                            if key_temp is not None:
                                sense_key = key_temp

        env.close()
        pygame.quit()

    def _direction_to_arrow_vector(self, direction):
        if direction == SelectedDirection.N:
            return 0, -1
        if direction == SelectedDirection.S:
            return 0, 1
        if direction == SelectedDirection.W:
            return -1, 0
        if direction == SelectedDirection.E:
            return 1, 0

        return 0, 0

    def _show_act_rew(self, surface, act_rew, pos):
        y = render_text(surface, self.font, "Action/Reward", pos)
        radius = 20
        for action, rewards in reversed(act_rew):
            pathfinding, sense, decision = action.squeeze(0).tolist()
            pathfinding = self._direction_to_arrow_vector(SelectedDirection(pathfinding))
            sense = self._direction_to_arrow_vector(SelectedDirection(sense))
            decision = SelectedAction(decision)

            _ = self._draw_arrow(surface, f"{rewards[0]:.2f}", (pos[0], y), pathfinding,
                                 radius=radius, color=(255, 255, 255), width=3)
            x = pos[0] + radius * 2 + 10
            new_y = self._draw_arrow(surface, f"{rewards[1]:.2f}", (x, y), sense, radius=20,
                                     color=(255, 255, 255), width=3)

            x += radius * 2 + 10
            render_text(surface, self.font, f"{decision.name} {rewards[2]:.2f}", (x, y))

            y = new_y

    def _render_game_view(self, surface, rgb_array, pos, game_width, game_height):
        frame = pygame.surfarray.make_surface(np.swapaxes(rgb_array, 0, 1))
        scaled_frame = pygame.transform.scale(frame, (game_width, game_height))
        surface.blit(scaled_frame, pos)

    def _show_observation(self, surface, obs):
        img, features = obs

        x_pos = self.obs_x
        y_pos = self.obs_y
        y_pos = self._render_observation_view(surface, x_pos, y_pos, img)

        features = features.reshape(13, 3)

        for i in range(features.shape[0]):
            x, y, temp = features[i].cpu().tolist()

            y_pos = self._draw_arrow(surface, f"{temp:.1f}", (x_pos + self.obs_width // 4, y_pos), np.array([x, y]),
                                radius=self.obs_width // 4, color=(255, 255, 255), width=3)

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

    def _overlay_grid_and_text(self, surface, kind, offset, text_color, scale, info):
        if kind == 0:
            return

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
                pygame.draw.rect(surface, (0, 0, 0), (x, y, tile_width, tile_height), 1)

                tile_number = info['wavefront'][tile_y, tile_x]
                if tile_number > 256:
                    continue

                text = f"{tile_number:02X}"

                # Render the text
                text_surface = font.render(text, True, text_color)
                text_rect = text_surface.get_rect(center=(x + tile_width // 2, y + tile_height // 2))

                # Draw the text
                surface.blit(text_surface, text_rect)

    def _render_observation_view(self, surface, x, y, img):
        render_text(surface, self.font, "Observation", (x, y))
        y += 20

        if len(img.shape) == 4:
            for i in range(img.shape[0]):
                y = self._render_one_observation(surface, x, y, img[i])

            return y

        return self._render_one_observation(surface, x, y, img)

    def _write_key_val_aligned(self, surface, text, value, x, y, total_width, color=(255, 255, 255)):
        new_y = render_text(surface, self.font, text, (x, y), color)
        value_width, _ = self.font.size(value)
        render_text(surface, self.font, value, (x + total_width - value_width, y), color)
        return new_y

    def _render_one_observation(self, surface, x, y, img):
        if img.shape[2] == 1:
            img = img.repeat(1, 1, 3)

        np_array = img.squeeze(0).cpu().numpy()
        observation_surface = pygame.surfarray.make_surface(np_array) # .permute(1, 0, 2)
        observation_surface = pygame.transform.scale(observation_surface, (img.shape[1], img.shape[0]))
        surface.blit(observation_surface, (x, y))

        y += img.shape[0]
        return y

    def _get_pathfinding(self, keys):
        if keys[pygame.K_LEFT]:
            return SelectedDirection.W
        if keys[pygame.K_RIGHT]:
            return SelectedDirection.E
        if keys[pygame.K_UP]:
            return SelectedDirection.N
        if keys[pygame.K_DOWN]:
            return SelectedDirection.S

        return None

    def _map_sense_key(self, keys):
        if keys[pygame.K_KP_8]:
            return SelectedDirection.N
        if keys[pygame.K_KP_2]:
            return SelectedDirection.S
        if keys[pygame.K_KP_4]:
            return SelectedDirection.W
        if keys[pygame.K_KP_6]:
            return SelectedDirection.E

        return SelectedDirection.NONE

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

    def _get_action_from_model(self, obs, info):
        with torch.no_grad():
            masks = info.get('masks', None)
            image = obs[0].unsqueeze(0)
            features = obs[1].flatten().unsqueeze(0)
            return self._model.get_action(image, features, masks)
