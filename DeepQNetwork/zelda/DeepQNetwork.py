import random
import numpy as np
import tensorflow as tf
from collections import deque
import os

# model parameters
image_height = 240
image_width = 256
image_channels = 3
num_output = 8                # number of actions the agent can take

# Deep Q-learning Agent parameters
gamma = 0.95  # discount factor
epsilon = 1.0  # exploration rate
epsilon_min = 0.10
epsilon_decay = 0.998
learning_rate = 0.001
batch_size = 512

max_memory_len = 20_000            # one per frame we made a decision on
max_decisions_per_game = 1_000

num_episodes = 1001
output_dir = 'd:/model_output/zelda'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class DqnAgent:
    """Deep Q-learning Agent based on Jon Krohn's video series"""
    def __init__(self, frame_count, feature_count):
        self.memory = deque(maxlen=2000)  # double-ended queue; acts like list, but elements can be added/removed from either end
        
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        
        self.epsilon_min = epsilon_min  # minimum exploration probability
        self.epsilon_decay = epsilon_decay  # exploration probability decay factor
        
        self.learning_rate = learning_rate
        
        self.model = self._build_model(frame_count, feature_count)

    def _build_model(self, frame_count, feature_count):
        # Image processing pathway with LSTM
        image_input = tf.keras.Input(shape=(frame_count, image_height, image_width, image_channels))
        conv_layers = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))(image_input)
        pool_layers = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))(conv_layers)
        conv2_layers = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))(pool_layers)
        flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(conv2_layers)
        lstm = tf.keras.layers.LSTM(64)(flatten)

        # Game state pathway
        state_input = tf.keras.Input(shape=(feature_count,))
        dense_state = tf.keras.layers.Dense(64, activation='relu')(state_input)

        # Combining pathway
        combined = tf.keras.layers.concatenate([lstm, dense_state])
        combined_dense = tf.keras.layers.Dense(512, activation='relu')(combined)

        # Output layer
        output = tf.keras.layers.Dense(num_output, activation='sigmoid')(combined_dense)

        # Create the model
        model = tf.keras.Model(inputs=[image_input, state_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        return model

    def remember(self, state, action, reward, next_state, done):
        # state: current state
        # action: action taken
        # reward: reward from action
        # next_state: next state
        # done: whether game is done or not
        self.memory.append((state, action, reward, next_state, done))  # append tuple to memory

    def act(self, state):
        # state: current state
        if np.random.rand() <= self.epsilon:  # if random number from 0 to 1 is less than exploration rate
            return np.random.beta(0.5, 0.5, num_output)
        
        act_values = self.model.predict(state)  # predict reward value based on current state
        return np.argmax(act_values[0])         # return action with highest reward

    def replay(self, batch_size):
        # batch_size: size of random sample from memory
        minibatch = random.sample(self.memory, batch_size)  # random sample from memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target_f = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
                
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
    
    def load(self, name):
        self.model.load_weights(name)
        
    def save(self, name):
        self.model.save_weights(name)

class DqnAgentRunner:
    def __init__(self, iterations, score_function, frames, parameters):
        # per run variables
        self.curr_iteration = 0
        self.total_iterations = iterations
        self.score_function = score_function
        
        self.agent = DqnAgent(frames, parameters)
        if os.path.exists(output_dir + '/weights.h5'):
            self.agent.load(output_dir + '/weights.h5')
        
        # per iteration variables
        self.prev_action = None
        self.prev_state = None
        self.prev_state_for_score = None
        self.score = 0.0
    
    def start_iteration(self):
        if self.curr_iteration < self.total_iterations:
            self.curr_iteration += 1
            return True
        
        self.prev_action = None
        self.prev_state = None
        self.prev_state_for_score = None
        self.score = 0.0
        
        return False
    
    def end_iteration(self):
        if self.prev_action is not None:
            print(f"episode: {self.curr_iteration} score: {self.score}")
            self.prev_action = None
            
            if len(self.agent.memory) > batch_size:
                self.agent.replay(batch_size)
                
            if self.iterations_remaining % 100 == 0:
                self.agent.save(output_dir + f"/weights_{self.curr_iteration:04d}.h5")

    def next_action(self, state_for_model, state_for_score, done):
        if self.prev_action is not None:
            reward = self.score_function(self.prev_state_for_score, state_for_score)
            self.agent.remember(self.prev_state, self.prev_action, reward, state_for_model, done)
            
        if done:
            self.end_iteration()
            return None
        
        # calculate next action
        action = self.agent.act(state_for_model)
        
        # update variables for next iteration
        self.prev_action = action
        
        self.prev_state = state_for_model
        self.prev_state_for_score = state_for_score
        
        return action
