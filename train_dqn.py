from collections import deque
import random

import numpy as np
import gym_zelda_nomenu
from models import ZeldaBaseModel
from gym_zelda_nomenu import ZeldaScoreDungeon

def scoreNone(self):
    return 0.0

env = gym_zelda_nomenu.ZeldaSmartItemEnv()
env.set_score_function(scoreNone)

# move to the first dungeon
env.reset_to_first_dungeon()
env.zelda_memory.sword = 1
env.zelda_memory.bombs = 2
env.save_state()

gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
replay_buffer = deque(maxlen=2000)
batch_size = 32
train_every = 100

# eventually these are command line options:
model = ZeldaBaseModel()
weights = "zelda_dqn_weights.h5"
level = 1
episodes = 1000
max_seconds = 600
max_steps = 600 * 60.1  # nes runs at 60.1 fps


env.set_score_function(ZeldaScoreDungeon().score)

frames = deque(maxlen=model.frame_count)

def observation_to_state(obs):
    screen = obs["screen"]
    frames.append(screen.copy())
    return model.get_model_input(frames)

since_last_train = 0
for episode in range(1, episodes):
    state = observation_to_state(env.reset())
    
    steps = 0
    while steps < max_steps:
        # ensure we don't try to take action while we aren't in control
        if env.is_scrolling:
            env.skip_screen_scroll()

            # Start frames fresh if we change rooms
            frames.clear()
            for x in range(frames.maxlen):
                frames.append(env.screen.copy())

            state = model.get_model_input(frames)

        if np.random.rand() <= epsilon:
            action = random.randint(0, len(env.actions) - 1)
        else:
            predicted = model.model.predict(state, verbose=0)
            action = np.argmax(predicted[0])
        

        observation, reward, done, _ = env.step(action)

        # copy the screen buffer into frames, get the model input from the observed state
        next_state = observation_to_state(observation)
        
        # Store in replay buffer
        replay_buffer.append((state, action, reward, next_state, done))
        since_last_train += 1

        state = next_state
        
        if since_last_train >= train_every and len(replay_buffer) > batch_size:
            since_last_train = 0
            
            minibatch = random.sample(replay_buffer, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = reward + gamma * np.amax(model.model.predict(next_state, verbose = 0)[0])

                target_f = model.model.predict(state, verbose=0)
                target_f[0][action] = target
                model.model.fit(state, target_f, epochs=1, verbose=0)

        if done:
            break

        env.render()

    epsilon = max(epsilon_min, epsilon_decay * epsilon)