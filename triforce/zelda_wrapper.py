"""
Responsible for interpreting complex game state and producing an object model in the 'info' dictionary.
Zelda has a very complicated combat system.  This class is responsible for detecting when the
agent has killed or injured an enemy.
"""

from collections import deque
from random import randint
import gymnasium as gym

from .game_state_change import ZeldaStateChange
from .zelda_game_state import ZeldaGameState
from .zelda_cooldown_handler import ZeldaCooldownHandler, ActionTranslator

class ZeldaGameWrapper(gym.Wrapper):
    """Interprets the game state and produces more information in the 'info' dictionary."""
    def __init__(self, env, deterministic=False, action_translator=None, states_to_track=16):
        super().__init__(env)

        self.deterministic = deterministic

        action_translator = action_translator or ActionTranslator(env)
        self.action_translator = action_translator
        self.cooldown_handler = ZeldaCooldownHandler(env, action_translator)

        # per-reset state
        self._total_frames = 0
        self.states = deque(maxlen=states_to_track)
        self._discounts = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Per-reset state
        self.states.clear()
        self._discounts.clear()
        self.cooldown_handler.reset()

        # Randomize the RNG if requested
        if not self.deterministic:
            for i in range(12):
                self.unwrapped.data.set_value(f'rng_{i}', randint(1, 255))

        # Move forward to the first frame where the agent can control Link
        _, _, _, info = self.cooldown_handler.skip(1)
        obs, info, frames_skipped = self.cooldown_handler.skip_uncontrollable_states(info)
        self._total_frames = frames_skipped + 1

        # Initial game state
        state = ZeldaGameState(None, self, info, self._total_frames)
        self.states.append(state)

        # Update info dictionary
        info['state'] = state
        info['total_frames'] = self._total_frames

        return obs, info

    def step(self, action):
        # get link position for movement actions
        prev = self.states[-1] if self.states else None
        link_position = prev.link.position if prev else None

        # Take action
        obs, terminated, truncated, info, frames = self.cooldown_handler.act_and_wait(action, link_position)
        self._total_frames += frames

        # Save the current state
        curr = ZeldaGameState(prev, self, info, self._total_frames)
        self.states.append(curr)

        # Update info dictionary
        info['action'] = self.action_translator.get_action_type(action)
        info['buttons'] = self._get_button_names(action, self.env.unwrapped.buttons)
        info['state_change'] = ZeldaStateChange(self, prev, curr, self._discounts)
        info['state'] = curr
        info['total_frames'] = self._total_frames

        return obs, 0, terminated, truncated, info

    def _get_button_names(self, act, buttons):
        result = []
        for i, b in enumerate(buttons):
            if act[i]:
                result.append(b)
        return result

__all__ = [ZeldaGameWrapper.__name__]
