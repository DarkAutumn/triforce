import math
import pygame
import numpy as np
from collections import deque


def pygame_render(zelda_ml):
    env = zelda_ml.env

    pygame.init()
    global font
    font = pygame.font.Font(None, 24)

    obs_width = 128
    obs_height = 640
    obs_x = 0
    obs_y = 0

    game_width, game_height = 640, 480
    graph_height = 150
    game_x = obs_width
    game_y = 0
    graph_width = game_width + obs_width

    text_x = obs_width + game_width
    text_y = 0
    text_height = game_height + graph_height
    text_width = 300
    screen = pygame.display.set_mode((obs_width + game_width + text_width, max(game_height + graph_height, text_height)))
    clock = pygame.time.Clock()

    block_width = 10
    center_line = game_height + graph_height // 2
    reward_values = deque(maxlen=graph_width // block_width)
    reward_details = deque(maxlen=100)

    terminated = True
    truncated = False

    # modes: c - continue, n - next, r - reset, p - pause, q - quit
    mode = 'c'
    while mode != 'q':
        if terminated or truncated:
            obs, info = env.reset()
            reward_details.clear()
            reward_values.clear()
            for _ in range(reward_values.maxlen):
                reward_values.append(0)

        # Perform a step in the environment
        if mode == 'c' or mode == 'n':
            action, _states = zelda_ml.model.predict(obs, deterministic=False)  # Replace this with your action logic
            obs, reward, terminated, truncated, info = env.step(action)

            if mode == 'n':
                mode = 'p'
        
        # update rewards for display
        update_rewards(reward_values, reward_details, info, reward)


        while True:
            if zelda_ml.rgb_deque:
                rgb_array = zelda_ml.rgb_deque.popleft()
            elif mode != 'p':
                break


            screen.fill((0, 0, 0))  # Black background

            # Show observation values
            y_pos = render_observation_view(screen, obs_x, obs_y, obs_width, obs["image"])
            y_pos = draw_arrow(screen, "Objective", (obs_x + obs_width // 4, y_pos), obs["vectors"][0], radius=obs_width // 4, color=(255, 255, 255), width=3)
            y_pos = draw_arrow(screen, "Enemy", (obs_x + obs_width // 4, y_pos), obs["vectors"][1], radius=obs_width // 4, color=(255, 255, 255), width=3)
            y_pos = draw_arrow(screen, "Projectile", (obs_x + obs_width // 4, y_pos), obs["vectors"][2], radius=obs_width // 4, color=(255, 0, 0), width=3)
            y_pos = draw_arrow(screen, "Item", (obs_x + obs_width // 4, y_pos), obs["vectors"][3], radius=obs_width // 4, color=(255, 255, 255), width=3)

            # render the gameplay
            render_game_view(rgb_array, (game_x, game_y), game_width, game_height, screen)

            # render rewards graph and values
            draw_rewards_graph(graph_height, screen, block_width, center_line, reward_values)
            draw_description_text(screen, reward_details, text_x, text_y, 20, text_height)

            # Display the scaled frame
            pygame.display.flip()
            clock.tick(60.1)

            # Check for Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    mode = 'q'
                    break

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        terminated = truncated = True
                        break

                    elif event.key == pygame.K_p:
                        mode = 'p'

                    elif event.key == pygame.K_n:
                        mode = 'n'

                    elif event.key == pygame.K_c:
                        mode = 'c'

    env.close()
    pygame.quit()

def draw_arrow(screen, label, start_pos, direction, radius=128, color=(255, 0, 0), width=5):
    render_text(screen, label, (start_pos[0], start_pos[1]))
    circle_start = (start_pos[0], start_pos[1] + 20)
    centerpoint = (circle_start[0] + radius, circle_start[1] + radius)
    end_pos = (centerpoint[0] + direction[0] * radius, centerpoint[1] + direction[1] * radius)

    pygame.draw.circle(screen, (255, 255, 255), centerpoint, radius, 1)

    if direction[0] != 0 or direction[1] != 0:
        pygame.draw.line(screen, color, centerpoint, end_pos, width)

        # Arrowhead
        arrowhead_size = 10
        angle = math.atan2(-direction[1], direction[0]) + math.pi

        left = (end_pos[0] + arrowhead_size * math.cos(angle - math.pi / 6),
                end_pos[1] - arrowhead_size * math.sin(angle - math.pi / 6))
        right = (end_pos[0] + arrowhead_size * math.cos(angle + math.pi / 6),
                end_pos[1] - arrowhead_size * math.sin(angle + math.pi / 6))

        pygame.draw.polygon(screen, color, [end_pos, left, right])
        

    return circle_start[1] + radius * 2

def render_observation_view(screen, x, y, dim, img):
    render_text(screen, "Observation", (x, y))
    y += 20

    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
        
    observation_surface = pygame.surfarray.make_surface(np.swapaxes(img, 0, 1))
    observation_surface = pygame.transform.scale(observation_surface, (dim, dim))
    screen.blit(observation_surface, (x, y))

    y += dim
    return y

def render_game_view(rgb_array, pos, game_width, game_height, screen):
    frame = pygame.surfarray.make_surface(np.swapaxes(rgb_array, 0, 1))
    scaled_frame = pygame.transform.scale(frame, (game_width, game_height))
    screen.blit(scaled_frame, pos)

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
    action = "+".join(info['buttons'])
    if prev is not None and prev['rewards'] == reward_dict and prev['action'] == action:
        prev['count'] += 1
    else:
        reward_details.appendleft({'count': 1, 'rewards': reward_dict, 'action' : action})


def render_text(surface, text, position, color=(255, 255, 255)):
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)

def draw_description_text(surface, rewards_deque, start_x, start_y, line_height, max_height):
    y = start_y
    for i in range(len(rewards_deque)):
        entry = rewards_deque[i]
        step_count, rewards, action = entry['count'], entry['rewards'], entry['action']

        render_text(surface, f"Action: {action}", (start_x, y))
        y += line_height

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