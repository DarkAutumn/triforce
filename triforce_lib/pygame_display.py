from collections import deque


def pygame_render(zelda_ml):
    import pygame
    import numpy as np

    env = zelda_ml.env

    pygame.init()
    game_width, game_height = 640, 480
    graph_height = 150
    screen = pygame.display.set_mode((game_width, game_height + graph_height))
    clock = pygame.time.Clock()

    block_width = 10
    center_line = game_height + graph_height // 2
    rewards = deque(maxlen=game_width // block_width)

    terminated = True
    truncated = False
    continue_rendering = True
    while continue_rendering:
        if terminated or truncated:
            obs, info = env.reset()
            rewards.clear()
            for _ in range(rewards.maxlen):
                rewards.append(0)

        # Perform a step in the environment
        action, _states = zelda_ml.model.predict(obs, deterministic=False)  # Replace this with your action logic
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        while zelda_ml.rgb_deque:
            screen.fill((0, 0, 0))  # Black background

            # render the game
            rgb_array = zelda_ml.rgb_deque.popleft()
            frame = pygame.surfarray.make_surface(np.swapaxes(rgb_array, 0, 1))
            scaled_frame = pygame.transform.scale(frame, (game_width, game_height))
            screen.blit(scaled_frame, (0, 0))

            # Draw rewards graph
            for i, r in enumerate(rewards):
                x_position = i * block_width

                if r == 0:
                    # Draw a white line for zero reward
                    pygame.draw.line(screen, (255, 255, 255), (x_position, center_line), (x_position + block_width, center_line))
                else:
                    color = (0, 0, 255) if r > 0 else (255, 0, 0)  # Blue for positive, red for negative
                    block_height = int(abs(r) * (graph_height // 2))
                    y_position = center_line - block_height if r > 0 else center_line
                    pygame.draw.rect(screen, color, (x_position, y_position, block_width, block_height))

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