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
        """Returns the action the agent should take based on the current state"""
        if np.random.rand() <= self.epsilon:  # if random number from 0 to 1 is less than exploration rate
            return (False, self.get_random_action())
        
        act_values = self.model.predict(model_input)  # predict reward value based on current state
        result = np.argmax(act_values[0])         # return action with highest reward

        return (True, result)

    def learn(self, memory, batch_size):
        # batch_size: size of random sample from memory
        minibatch = random.sample(memory, batch_size)  # random sample from memory
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
    
class DqnAgentRunner:
    def __init__(self, model, get_random_action, max_memory):
        self.memory = deque(maxlen=max_memory)
        self.agent = DqnAgent(model, get_random_action)
        self.reset()
        
    def reset(self):
        self.prev_action = None
        self.prev_state = None
    
    def learn(self, batch_size):
        self.agent.learn(self.memory, batch_size)

    def done(self, state, reward):
        self.memory.append((self.prev_state, self.prev_action, reward, state, True))
        self.reset()

    def act(self, state, reward, done = False):
        if self.prev_action is not None:
            self.memory.append((self.prev_state, self.prev_action, reward, state, done))
            
        if done:
            return None
        
        # calculate next action
        action = self.agent.act(state)
        
        # update variables for next iteration
        self.prev_action = action
        self.prev_state = state
        
        return action
    