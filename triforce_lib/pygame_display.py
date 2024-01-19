import pygame
import numpy as np
from collections import deque


def pygame_render(zelda_ml):
    env = zelda_ml.env

    pygame.init()
    global font
    font = pygame.font.Font(None, 24)

    game_width, game_height = 640, 480
    graph_height = 150
    graph_width = game_width
    text_height = game_height + graph_height
    text_width = 300
    screen = pygame.display.set_mode((game_width + text_width, max(game_height + graph_height, text_height)))
    clock = pygame.time.Clock()

    block_width = 10
    center_line = game_height + graph_height // 2
    reward_values = deque(maxlen=graph_width // block_width)
    reward_details = deque(maxlen=100)

    terminated = True
    truncated = False
    continue_rendering = True
    while continue_rendering:
        if terminated or truncated:
            obs, info = env.reset()
            reward_values.clear()
            for _ in range(reward_values.maxlen):
                reward_values.append(0)

        # Perform a step in the environment
        action, _states = zelda_ml.model.predict(obs, deterministic=False)  # Replace this with your action logic
        obs, reward, terminated, truncated, info = env.step(action)
        
        # update rewards for display
        update_rewards(reward_values, reward_details, info, reward)

        while zelda_ml.rgb_deque:
            screen.fill((0, 0, 0))  # Black background

            render_game_view(zelda_ml, game_width, game_height, screen)
            draw_rewards_graph(graph_height, screen, block_width, center_line, reward_values)
            render_sidebar(screen, reward_details, game_width, 0, 20, text_height)

            # Display the scaled frame
            pygame.display.flip()
            clock.tick(60.1)

            # Check for Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    continue_rendering = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        terminated = truncated = True
                        break


    env.close()
    pygame.quit()

def render_game_view(zelda_ml, game_width, game_height, screen):
    rgb_array = zelda_ml.rgb_deque.popleft()
    frame = pygame.surfarray.make_surface(np.swapaxes(rgb_array, 0, 1))
    scaled_frame = pygame.transform.scale(frame, (game_width, game_height))
    screen.blit(scaled_frame, (0, 0))

def draw_rewards_graph(graph_height, screen, block_width, center_line, reward_values):
    for i, r in enumerate(reward_values):
        x_position = i * block_width

        if r == 0:
            pygame.draw.line(screen, (255, 255, 255), (x_position, center_line), (x_position + block_width, center_line))
        else:
            color = (0, 0, 255) if r > 0 else (255, 0, 0)  # Blue for positive, red for negative
            block_height = int(abs(r) * (graph_height // 2))
            block_height = max(block_height, 10)
            y_position = center_line - block_height if r > 0 else center_line
            pygame.draw.rect(screen, color, (x_position, y_position, block_width, block_height))

def update_rewards(reward_values, reward_details, info, reward):
    reward_values.append(reward)

    if 'rewards' in info:
        reward_dict = {k: round(v, 2) for k, v in info['rewards'].items()}

    else:
        reward_dict = {}

    prev = reward_details[0] if reward_details else None
    if prev is not None and prev['rewards'] == reward_dict:
        prev['count'] += 1
    else:
        reward_details.appendleft({'count': 1, 'rewards': reward_dict})


def render_text(surface, text, position, color=(255, 255, 255)):
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)

def render_sidebar(surface, rewards_deque, start_x, start_y, line_height, max_height):
    y = start_y
    for i in range(len(rewards_deque)):
        entry = rewards_deque[i]
        step_count, rewards = entry['count'], entry['rewards']
        if rewards:
            for reason, value in rewards.items():
                # color=red if negative, light blue if positive, white if zero
                color = (255, 0, 0) if value < 0 else (0, 255, 255) if value > 0 else (255, 255, 255)
                render_text(surface, reason, (start_x, y), color=color)
                render_text(surface, f"{'+' if value > 0 else ''}{value:.2f}", (start_x + 200, y))
                y += line_height
                if y + line_height > max_height:
                    while len(rewards_deque) > i:
                        rewards_deque.pop()

                    return  # Stop rendering when we run out of vertical space
        else:
            text = "none"
            color = (128, 128, 128)
            render_text(surface, text, (start_x, y), color)
            y += line_height
            
        # Render the step count (e.g., x3) aligned to the right
        if step_count > 1:
            count_text = f"x{step_count}"
            count_text_width, _ = font.size(count_text)
            render_text(surface, count_text, (start_x + 275 - count_text_width, y - line_height))

        # Draw dividing line
        pygame.draw.line(surface, (255, 255, 255), (start_x, y), (start_x + 300, y))
        y += 3