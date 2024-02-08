#! /usr/bin/python

import argparse
import os

import math
import os
import pygame
import numpy as np
from collections import deque

from triforce_lib import ZeldaAIOrchestrator, ZeldaScenario, ZeldaML, is_in_cave

class Recording:
    def __init__(self, dimensions):
        import cv2

        self.dimensions = dimensions
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.recording = cv2.VideoWriter(self.__get_filename(), fourcc, 60.1, dimensions)

    def append(self, surface):
        import cv2

        result_frame = surface
        result_frame = result_frame.transpose([1, 0, 2])  # Transpose it to the correct format
        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
        self.recording.write(result_frame)


    def stop(self):
        self.recording.release()

    def __get_filename(self):
        directory = os.path.join(os.getcwd(), "recording")
        if not os.path.exists(directory):
            os.makedirs(directory)

        i = 0
        while True:
            filename = os.path.join(directory, f"recording_{i:03d}.avi")
            if not os.path.exists(filename):
                return filename
            i += 1

class Display:
    def __init__(self, zelda_ml : ZeldaML, scenario_name : str):
        scenario = ZeldaScenario.get(scenario_name)
        if not scenario:
            raise Exception(f'Unknown scenario {scenario_name}')
        
        orchestrator = ZeldaAIOrchestrator()
        if not orchestrator.has_any_model:
            raise Exception(f'No models loaded')
        
        self.zelda_ml = zelda_ml
        self.scenario = scenario
        self.orchestrator = orchestrator
                
        pygame.init()

        self.font = pygame.font.Font(None, 24)

        game_x, game_y = 240, 224
        self.scale = 4

        self.game_width = game_x * self.scale
        self.game_height = game_y * self.scale

        self.obs_width = 128
        self.obs_height = self.game_height
        self.obs_x = 0
        self.obs_y = 0

        self.graph_height = 150
        self.game_x = self.obs_width
        self.game_y = 0
        self.graph_width = self.game_width + self.obs_width

        self.text_x = self.obs_width + self.game_width
        self.text_y = 0
        self.text_height = self.game_height + self.graph_height
        self.text_width = 300

        self.total_width = self.obs_width + self.game_width + self.text_width
        self.total_height = max(self.game_height + self.graph_height, self.text_height)
        self.dimensions = (self.total_width, self.total_height)

        self.total_rewards = 0.0

    def show(self):
        env = self.zelda_ml.make_env(self.scenario)

        surface = pygame.display.set_mode(self.dimensions)
        clock = pygame.time.Clock()
        
        block_width = 10
        center_line = self.game_height + self.graph_height // 2
        reward_values = deque(maxlen=self.graph_width // block_width)
        reward_details = deque(maxlen=100)

        model_requested = 0
        model_name = None
        model_kind = None

        recording = None
        cap_fps = True
        overlay = 0

        terminated = True
        truncated = False

        # modes: c - continue, n - next, r - reset, p - pause, q - quit
        mode = 'c'
        while mode != 'q':
            if terminated or truncated:
                obs, info = env.reset()
                self.total_rewards = 0.0
                reward_details.clear()
                reward_values.clear()
                for _ in range(reward_values.maxlen):
                    reward_values.append(0)

                if recording is not None:
                    recording.stop()
                    recording = Recording(self.dimensions)

            # Perform a step in the environment
            if mode == 'c' or mode == 'n':
                acceptable_models = self.orchestrator.select_model(info)
                selected_model = acceptable_models[0]

                model_requested %= len(selected_model.models)
                model = selected_model.models[model_requested]
                model_kind = selected_model.model_kinds[model_requested]
                model_name = selected_model.name if not model_kind else f"{selected_model.name} ({model_kind})"

                action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)

                if mode == 'n':
                    mode = 'p'
            
            # update rewards for display
            self.update_rewards(reward_values, reward_details, info, reward)
            curr_score = info.get('score', None)

            while True:
                if self.zelda_ml.rgb_deque:
                    rgb_array = self.zelda_ml.rgb_deque.popleft()
                elif mode != 'p':
                    break

                surface.fill((0, 0, 0))

                # Show observation values
                x_pos = self.obs_x
                y_pos = self.obs_y
                y_pos = self.render_observation_view(surface, x_pos, y_pos, self.obs_width, obs["image"])
                y_pos = self.draw_arrow(surface, "Objective", (x_pos + self.obs_width // 4, y_pos), obs["vectors"][0], radius=self.obs_width // 4, color=(255, 255, 255), width=3)
                y_pos = self.draw_arrow(surface, "Enemy", (x_pos + self.obs_width // 4, y_pos), obs["vectors"][1], radius=self.obs_width // 4, color=(255, 255, 255), width=3)
                y_pos = self.draw_arrow(surface, "Projectile", (x_pos + self.obs_width // 4, y_pos), obs["vectors"][2], radius=self.obs_width // 4, color=(255, 0, 0), width=3)
                y_pos = self.draw_arrow(surface, "Item", (x_pos + self.obs_width // 4, y_pos), obs["vectors"][3], radius=self.obs_width // 4, color=(255, 255, 255), width=3)
                y_pos = self.render_text(surface, f"Enemies: {obs['features'][0]}", (x_pos, y_pos))
                y_pos = self.render_text(surface, f"Beams: {obs['features'][1]}", (x_pos, y_pos))
                y_pos = self.render_text(surface, f"Rewards: {round(self.total_rewards, 2)}", (x_pos, y_pos))
                if curr_score is not None:
                    y_pos = self.render_text(surface, f"Score: {round(curr_score, 2)}", (x_pos, y_pos))

                # render the gameplay
                self.render_game_view(surface, rgb_array, (self.game_x, self.game_y), self.game_width, self.game_height)
                if overlay:
                    color = "black" if info['level'] == 0 and not is_in_cave(info) else "white"
                    self.overlay_grid_and_text(surface, overlay, (self.game_x, self.game_y), info['tiles'], color, self.scale, self.get_optimal_path(info))
                self.render_text(surface, f"Model: {model_name}", (self.game_x, self.game_y))
                if "location" in info:
                    self.render_text(surface, f"Location: {hex(info['location'])}", (self.game_x + self.game_width - 120, self.game_y))

                # render rewards graph and values
                self.draw_rewards_graph(surface, self.graph_height, block_width, center_line, reward_values)
                self.draw_description_text(surface, reward_details, self.text_x, self.text_y, 20, self.text_height)

                if recording:
                    recording.append(pygame.surfarray.array3d(pygame.display.get_surface()))

                # Display the scaled frame
                pygame.display.flip()
                if cap_fps:
                    clock.tick(60.1)

                # Check for Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        mode = 'q'
                        break

                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            mode = 'q'
                            break

                        elif event.key == pygame.K_r:
                            terminated = truncated = True
                            break

                        elif event.key == pygame.K_p:
                            mode = 'p'

                        elif event.key == pygame.K_n:
                            mode = 'n'

                        elif event.key == pygame.K_c:
                            mode = 'c'

                        elif event.key == pygame.K_o:
                            overlay = (overlay + 1) % 3

                        elif event.key == pygame.K_m:
                            model_requested += 1

                        elif event.key == pygame.K_u:
                            cap_fps = not cap_fps

                        elif event.key == pygame.K_F5:
                            if recording is not None:
                                recording.stop()
                                recording = None
                            else:
                                recording = Recording(self.dimensions)

        if recording:
            recording.stop()

        env.close()
        pygame.quit()

    def update_rewards(self, reward_values, reward_details, info, reward):
        reward_values.append(reward)


        if 'rewards' in info:
            reward_dict = {k: round(v, 2) for k, v in info['rewards'].items()}

            for rew in info.get('rewards', {}).values():
                self.total_rewards += rew

        else:
            reward_dict = {}

        prev = reward_details[0] if reward_details else None
        action = "+".join(info['buttons'])
        if prev is not None and prev['rewards'] == reward_dict and prev['action'] == action:
            prev['count'] += 1
        else:
            reward_details.appendleft({'count': 1, 'rewards': reward_dict, 'action' : action})

    def draw_arrow(self, surface, label, start_pos, direction, radius=128, color=(255, 0, 0), width=5):
        self.render_text(surface, label, (start_pos[0], start_pos[1]))
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

    def render_observation_view(self, surface, x, y, dim, img):
        self.render_text(surface, "Observation", (x, y))
        y += 20

        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
            
        observation_surface = pygame.surfarray.make_surface(np.swapaxes(img, 0, 1))
        observation_surface = pygame.transform.scale(observation_surface, (dim, dim))
        surface.blit(observation_surface, (x, y))

        y += dim
        return y

    def render_game_view(self, surface, rgb_array, pos, game_width, game_height):
        frame = pygame.surfarray.make_surface(np.swapaxes(rgb_array, 0, 1))
        scaled_frame = pygame.transform.scale(frame, (game_width, game_height))
        surface.blit(scaled_frame, pos)

    def draw_rewards_graph(self, surface, graph_height, block_width, center_line, reward_values):
        for i, r in enumerate(reward_values):
            x_position = i * block_width

            if r == 0:
                pygame.draw.line(surface, (255, 255, 255), (x_position, center_line), (x_position + block_width, center_line))
            else:
                color = (0, 0, 255) if r > 0 else (255, 0, 0)  # Blue for positive, red for negative
                block_height = int(abs(r) * (graph_height // 2))
                block_height = max(block_height, 10)
                y_position = center_line - block_height if r > 0 else center_line
                pygame.draw.rect(surface, color, (x_position, y_position, block_width, block_height))

    def render_text(self, surface, text, position, color=(255, 255, 255)):
        text_surface = self.font.render(text, True, color)
        surface.blit(text_surface, position)
        return position[1] + text_surface.get_height()

    def draw_description_text(self, surface, rewards_deque, start_x, start_y, line_height, max_height):
        y = start_y
        for i in range(len(rewards_deque)):
            entry = rewards_deque[i]
            step_count, rewards, action = entry['count'], entry['rewards'], entry['action']

            self.render_text(surface, f"Action: {action}", (start_x, y))
            y += line_height

            if rewards:
                for reason, value in rewards.items():
                    # color=red if negative, light blue if positive, white if zero
                    color = (255, 0, 0) if value < 0 else (0, 255, 255) if value > 0 else (255, 255, 255)
                    self.render_text(surface, reason, (start_x, y), color=color)
                    self.render_text(surface, f"{'+' if value > 0 else ''}{value:.2f}", (start_x + 200, y))
                    y += line_height
                    if y + line_height > max_height:
                        while len(rewards_deque) > i:
                            rewards_deque.pop()

                        return  # Stop rendering when we run out of vertical space
            else:
                text = "none"
                color = (128, 128, 128)
                self.render_text(surface, text, (start_x, y), color)
                y += line_height
                
            # Render the step count (e.g., x3) aligned to the right
            if step_count > 1:
                count_text = f"x{step_count}"
                count_text_width, _ = self.font.size(count_text)
                self.render_text(surface, count_text, (start_x + 275 - count_text_width, y - line_height))

            # Draw dividing line
            pygame.draw.line(surface, (255, 255, 255), (start_x, y), (start_x + 300, y))
            y += 3

    def get_optimal_path(self, info):
        if 'a*_path' in info:
            return info['a*_path'][-1]

    def overlay_grid_and_text(self, surface, kind, offset, tiles, text_color, scale, path = None):
        grid_width = 32
        grid_height = 22
        tile_width = 8 * scale
        tile_height = 8 * scale

        # Pygame font setup
        font_size = int(min(tile_width, tile_height) // 2)
        font = pygame.font.Font(None, font_size)

        for tile_x in range(grid_width):
            for tile_y in range(grid_height):
                if kind == 1 and path and (tile_y, tile_x) not in path:
                    continue

                x = offset[0] + tile_x * tile_width - 8 * scale
                y = 56 * scale + offset[1] + tile_y * tile_height

                pygame.draw.rect(surface, (0, 0, 255), (x, y, tile_width, tile_height), 1)

                tile_number = tiles[tile_y, tile_x] # 1 for overscan
                text = f"{tile_number:02X}"

                # Render the text
                text_surface = font.render(text, True, text_color)
                text_rect = text_surface.get_rect(center=(x + tile_width // 2, y + tile_height // 2))

                # Draw the text
                surface.blit(text_surface, text_rect)

def main(args):
    render_mode = 'rgb_array'
    model_path = args.model_path[0] if args.model_path else os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

    zelda_ml = ZeldaML(args.color, render_mode=render_mode, verbose=args.verbose, ent_coef=args.ent_coef, device="cuda", obs_kind=args.obs_kind)
    zelda_ml.load_models(model_path)

    if args.scenario is None:
        args.scenario = 'zelda'

    display = Display(zelda_ml, args.scenario)
    display.show()

def parse_args():
    parser = argparse.ArgumentParser(description="ZeldaML - An ML agent to play The Legned of Zelda (NES).")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--color", action='store_true', help="Give the model a color version of the game (instead of grayscale).")
    parser.add_argument("--obs-kind", choices=['gameplay', 'viewport', 'full'], default='viewport', help="The kind of observation to use.")
    parser.add_argument("--model-path", nargs=1, help="Location to read models from.")

    parser.add_argument('scenario', nargs='?', help='Scenario name')

    try:
        args = parser.parse_args()
        return args
    except Exception as e:
        print(e)
        parser.print_help()
        exit(0)

if __name__ == '__main__':
    args = parse_args()
    main(args)
