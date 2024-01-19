def pygame_render(zelda_ml):
    import pygame
    import numpy as np

    env = zelda_ml.env

    pygame.init()
    width, height = 640, 480
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    terminated = True
    truncated = False
    continue_rendering = True
    while continue_rendering:
        if terminated or truncated:
            obs, info = env.reset()

        # Perform a step in the environment
        action, _states = zelda_ml.model.predict(obs, deterministic=False)  # Replace this with your action logic
        obs, reward, terminated, truncated, info = env.step(action)

        while zelda_ml.rgb_deque:
            rgb_array = zelda_ml.rgb_deque.popleft()

            # Convert the observation to a Pygame surface
            frame = pygame.surfarray.make_surface(np.swapaxes(rgb_array, 0, 1))

            # Scale the frame
            scaled_frame = pygame.transform.scale(frame, (width, height))

            # Display the scaled frame
            screen.blit(scaled_frame, (0, 0))
            pygame.display.flip()

            # Check for Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    continue_rendering = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        obs, info = env.reset()

            clock.tick(60.1)                        

    env.close()
    pygame.quit()