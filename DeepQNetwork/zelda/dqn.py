from collections.abc import Sequence
import random
import numpy as np
import tensorflow as tf
from collections import deque
import os


# Deep Q-learning Agent parameters
gamma = 0.95  # discount factor
epsilon = 1.0  # exploration rate
epsilon_min = 0.10
epsilon_decay = 0.998
learning_rate = 0.001

class DqnAgent:
    """Deep Q-learning Agent based on Jon Krohn's video series"""
    def __init__(self, model, get_random_action):
        self.model = model
        self.get_random_action = get_random_action

        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        
        self.epsilon_min = epsilon_min  # minimum exploration probability
        self.epsilon_decay = epsilon_decay  # exploration probability decay factor
        
        self.learning_rate = learning_rate

    def act(self, model_input) -> (bool, np.ndarray[float]):
        """Returns the action the agent should take based on the current state and whether the action was predicted or random"""
        if np.random.rand() <= self.epsilon:  # if random number from 0 to 1 is less than exploration rate
            return (False, self.get_random_action())
        
        act_values = self.model.predict(model_input, verbose=0) # predict reward value based on current state
        result = np.argmax(act_values[0])                           # return action with highest reward

        return (True, result)

    def learn(self, memory, batch_size):
        minibatch = random.sample(memory, batch_size)

        # Initialize arrays for batch update
        image_input = []
        feature_input = []
        target_output = []

        for state, action, reward, next_state, done in minibatch:
            # Predict the Q-values for the current state
            target_f = self.model.predict(state, verbose=0)

            # Compute the target Q-value
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            # Update the Q-value for the taken action
            target_f[0][action] = target

            # Accumulate the states and targets
            image_input.append(state[0])
            feature_input.append(state[1])
            target_output.append(target_f)

        # Fit the model on the batch
        image_input = np.array(image_input)
        feature_input = np.array(feature_input)
        target_output = np.array(target_output)
        
        image_input = np.squeeze(image_input, axis=1)
        feature_input = np.squeeze(feature_input, axis=1)

        self.model.fit([image_input, feature_input], target_output, batch_size=batch_size, epochs=1)

        # Update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


class DqnAgentRunner:
    def __init__(self, model, score, get_random_action, max_memory):
        self.memory = deque(maxlen=max_memory)
        self.agent = DqnAgent(model, get_random_action)
        self.score = score
        self.prev_score_state = None
        self.reset()
        
    def reset(self):
        self.prev_action = None
        self.prev_state = None
    
    def learn(self, batch_size):
        self.agent.learn(self.memory, batch_size)

    def done(self, state, reward):
        self.memory.append((self.prev_state, self.prev_action, reward, state, True))
        self.reset()

    def act(self, state, score_state, done = False):
        if self.prev_action is not None:
            reward = self.score(self.prev_score_state)
            self.memory.append((self.prev_state, self.prev_action, reward, state, done))
            
        if done:
            return None
        
        # calculate next action
        action = self.agent.act(state)
        
        # update variables for next iteration
        self.prev_action = action
        self.prev_state = state
        self.prev_score_state = score_state
        
        return action
    