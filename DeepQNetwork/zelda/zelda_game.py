from collections.abc import Sequence
import ctypes
import numpy as np
from collections import deque

from .dqn import DqnAgentRunner
from .rewards import ZeldaScoreBasic, ZeldaScoreDungeon
from .zelda_frame import ZeldaFrame
from .models import ZeldaModelXL
from .zelda_constants import zelda_game_modes

# number of frames to give the model
model_parameter_count = 4
model_frame_count = 5
frames_saved = 60

default_max_memory = 500   # number of steps max to keep in memory, at 4 per second, this is 83 minutes of gameplay
default_batch_size = 32

class LegendOfZeldaAgent:
    def __init__(self, model = "default", scorer = "default", max_memory = default_max_memory):
        model = self.get_model_by_name(model)
        self.model = model
        self.dqn_agent = DqnAgentRunner(model.model, model.get_random_action, max_memory)
        self.prev = None
        self.scorer = self.get_scorer_by_name(scorer)

    def get_model_by_name(self, name):
        if name == "default" or name == "xl":
            return ZeldaModelXL()
        else:
            raise ValueError("Unknown model name: " + name)
        
    def get_scorer_by_name(self, name):
        if name == "default" or name == "basic":
            return ZeldaScoreBasic()
        elif name == "dungeon":
            return ZeldaScoreDungeon()
        else:
            raise ValueError("Unknown scorer name: " + name)

    def begin_game(self):
        self.dqn_agent.reset()
        self.scorer.reset()


    def act(self, frames : list[ZeldaFrame]) -> [bool]:
        """Returns either a controller state, or None if the game is over."""
        # todo: handle game over, make sure game over works

        # Frames contain a sequnce of the last N frames.  The last frame is the current frame.
        # We score a reward based on the last frame, but the model needs to be fed a sequence
        # of frames in case it needs to use RNNs or similar.
        game_state = frames[-1].game_state
        if game_state.mode != zelda_game_modes.gameplay and game_state.mode != game_state.mode == zelda_game_modes.game_over:
            return None
        
        curr_frame = frames[-1]
        reward = self.scorer.score(curr_frame.game_state)

        model_state = self.model.get_model_input(frames)
        predicted, action = self.dqn_agent.act(model_state, reward)

        # store the action into the current frame
        curr_frame.predicted = predicted
        curr_frame.action = action

        return action
    
    def end_game(self, frames : Sequence[ZeldaFrame]):
        """Ends the game and provides the final reward for the game."""
        # Similar to act, but the game is over and there is no next state to predict
        model_state = self.model.get_model_input(frames)
        reward = self.scorer.score(frames[-1].game_state)
        self.dqn_agent.done(model_state, reward)

    def learn(self, batch_size = default_batch_size):
        self.dqn_agent.learn(batch_size)

    def save(self, path):
        self.model.save(path)