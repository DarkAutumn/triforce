import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress TensorFlow INFO and WARNING messages

from collections import deque
import random
import tqdm

import numpy as np
import gym_zelda_nomenu
from models import NoHitModel
from gym_zelda_nomenu import ZeldaScoreNoHit



env = gym_zelda_nomenu.ZeldaNoHitEnv()

gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
replay_buffer = deque(maxlen=2000)
batch_size = 32


# eventually these are command line options:
output_path = "/models/"
episodes = 10000
rendering = False
verbose = False
nondeterministic = True
train_model = True
reset_options = { "nondeterministic": nondeterministic }
episode_length_minutes = 2
decisions_per_second = 2

model = NoHitModel()
model.load(output_path)

# this is the length of playable minutes for each episode (doesn't count scrolling)
max_frames = episode_length_minutes * 60 * 60.1  # nes runs at 60.1 fps

action_cooldown_frame_min = max(1, int(60.1 / decisions_per_second) - 5)
action_cooldown_frame_max = min(60, int(60.1 / decisions_per_second) + 5)

# how long to hold down the buttons for each action (for just a and b, not movement)
button_hold_length = min(4, action_cooldown_frame_min)

def get_button_hold(action):
    # Keep holding movement buttons for the whole duration. Note that we don't want
    # to hold down the direction buttons when associated with an attack or item, since
    # that will cause link to move in that direction when those are intended to be simply
    # an attack in a particular direction and not a move.
    if action == "MoveUp" or action == "MoveDown" or action == "MoveLeft" or action == "MoveRight":
        return action
    
    # otherwise perform no action
    return 0


frames = deque(maxlen=model.frame_count)

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
        env.skip_frame(None)

    # ensure the screen buffer is up to date
    frames.clear()
    for _ in range(frames.maxlen):
        frames.append(env.screen.copy())

    return delay

def reset_state():
    state = env.reset(options=reset_options)

    # introduce randomnesss
    if nondeterministic:
        delay = random.randint(0, 31)
        for _ in range(delay):
            env.skip_frame(None)

    env.move_until_next_screen("MoveRight")

    # skip the scrolling screen
    env.skip_screen_scroll()
    frames.clear()
    for _ in range(frames.maxlen):
        frames.append(env.screen.copy())

    return state


# Get the name of the current script
script_name = os.path.basename(__file__)

# Remove the '.py' extension and add the datetime and '.log'
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_name = f"/output/{script_name[:-3]}_{current_time}.log"

# Open the log file for writing
with open(log_file_name, 'w') as log_file:
    def print_log(message):
        if verbose:
            print(message)
        log_file.write(message + '\n')
        log_file.flush()


    for episode in tqdm.tqdm(range(0, episodes)):
        state = reset_state()
        state = list(frames)


        total_score = 0.0

        scorer = ZeldaScoreNoHit(verbose=verbose)
        env.set_score_function(scorer.score)

        frame_count = 0
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
                print_log(f"Episode: {episode}, Frame: {frame_count}, Action: {env.actions[action]}, Predicted: {predicted}")

            next_state, reward, done, _ = env.step(action)
            frame_count += 1

            total_score += reward

            frames.append(next_state.copy())
            next_state = list(frames)

            # hold buttons for a bit
            frame_count += button_hold_length
            for _ in range(button_hold_length):
                env.skip_frame(action)

            frame_count += random_delay(action_cooldown_frame_min, action_cooldown_frame_max, get_button_hold(action))
            
            # Store in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            since_last_train += 1

            state = next_state

            if done:
                break

            if rendering:
                env.render()
            
            if episode and episode % 100 == 0:
                model.save(output_path, episode)

        if batch_size < len(replay_buffer):
            if verbose:
                print_log(f"Training model - samples:{len(replay_buffer)} - epsilon:{epsilon}")

            since_last_train = 0
            
            minibatch = random.sample(replay_buffer, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = reward + gamma * np.amax(model.predict(next_state, verbose = 0)[0])

                target_f = model.predict(state, verbose=0)
                target_f[0][action] = target
                model.fit(state, target_f, epochs=1, verbose=0)
        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        print_log(f"Episode: {episode}, Score: {total_score}, Epsilon: {epsilon}")

model.save(output_path, "complete")
