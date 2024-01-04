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
rendering = True
verbose = True
nondeterministic = True
train_model = False
reset_options = { "nondeterministic": nondeterministic }

# this is the length of playable minutes for each episode (doesn't count scrolling)
episode_length_minutes = 1
max_frames = episode_length_minutes * 60 * 60.1  # nes runs at 60.1 fps

# We shoot for 4 decisions per second, so 60.1 / 4 ~= 15.  We also want to introduce
# some randomness in the game, which comes from delays between each action.  Zelda
# runs on frame rules for randomness.
action_cooldown_frame_min = 10
action_cooldown_frame_max = 20

# how long to hold down the buttons for each action (for just a and b, not movement)
button_hold_length = 4

def get_button_hold(action):
    # Keep holding movement buttons for the whole duration. Note that we don't want
    # to hold down the direction buttons when associated with an attack or item, since
    # that will cause link to move in that direction when those are intended to be simply
    # an attack in a particular direction and not a move.
    if action == "MoveUp" or action == "MoveDown" or action == "MoveLeft" or action == "MoveRight":
        return action
    
    # otherwise perform no action
    return 0

env.set_score_function(ZeldaScoreDungeon().score)

frames = deque(maxlen=model.frame_count)

def observation_to_state(obs):
    screen = obs["screen"]
    frames.append(screen.copy())
    return list(frames)

def skip_screen_scroll(state):
    if not env.is_scrolling:
        return state

    while env.is_scrolling:
        env.skip_screen_scroll()
        frames.clear()
        for x in range(frames.maxlen):
            frames.append(env.screen.copy())

        return list(frames)

def random_delay(min_delay, max_delay, action):
    if max_delay < frames.maxlen:
        return 0

    if not nondeterministic:
        return 0
    
    # wait a short amount of time before starting the episode to introduce randomness
    delay = random.randint(min_delay, max_delay)
    for _ in range(delay):
        env.skip_frame(action)

    # ensure the screen buffer is up to date
    frames.clear()
    for _ in range(frames.maxlen):
        frames.append(env.screen.copy())

    return delay

show_image = False
since_last_train = 0
for episode in range(1, episodes):
    state = observation_to_state(env.reset(options=reset_options))
    state = skip_screen_scroll(state)

    frame_count = random_delay(0, action_cooldown_frame_max, 0)
    while frame_count < max_frames:
        # ensure we don't try to take action while we aren't in control
        state = skip_screen_scroll(state)
        
        predicted = False
        if np.random.rand() <= epsilon:
            action = random.randint(0, len(env.actions) - 1)
        else:
            predicted = model.predict(state, verbose=0)
            action = np.argmax(predicted[0])
        
        if verbose:
            print(f"Episode: {episode}, Frame: {frame_count}, Action: {env.actions[action]}, Predicted: {predicted}")

        observation, reward, done, _ = env.step(action)
        frame_count += 1

        # hold buttons for a bit
        frame_count += button_hold_length
        for _ in range(button_hold_length):
            env.skip_frame(action)

        frame_count += random_delay(action_cooldown_frame_min, action_cooldown_frame_max, get_button_hold(action))

        # copy the screen buffer into frames, get the model input from the observed state
        next_state = observation_to_state(observation)
        
        # Store in replay buffer
        replay_buffer.append((state, action, reward, next_state, done))
        since_last_train += 1

        state = next_state
        
        if train_model and since_last_train >= train_every and len(replay_buffer) > batch_size:
            if verbose:
                print(f"Training model - samples:{len(replay_buffer)} - epsilon:{epsilon} - time since last train:{since_last_train}")

            since_last_train = 0
            
            minibatch = random.sample(replay_buffer, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = reward + gamma * np.amax(model.predict(next_state, verbose = 0)[0])

                target_f = model.predict(state, verbose=0)
                target_f[0][action] = target
                model.fit(state, target_f, epochs=1, verbose=0)

        if done:
            break

        env.render()

    epsilon = max(epsilon_min, epsilon_decay * epsilon)