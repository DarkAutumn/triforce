"""
Responsible for interpreting complex game state and producing an object model in the 'info' dictionary.
Zelda has a very complicated combat system.  This class is responsible for detecting when the
agent has killed or injured an enemy.
"""

from random import randint
from typing import Union
import gymnasium as gym

from .models_and_scenarios import ZeldaScenario
from .objectives import Objectives
from .game_state_change import ZeldaStateChange
from .zelda_game import ZeldaGame
from .zelda_cooldown_handler import ZeldaCooldownHandler, ActionTranslator

class ZeldaGameWrapper(gym.Wrapper):
    """Interprets the game state and produces more information in the 'info' dictionary."""
    def __init__(self, env, scenario : ZeldaScenario, deterministic=False, action_translator=None):
        super().__init__(env)

        self.deterministic = deterministic

        action_translator = action_translator or ActionTranslator(env)
        self.action_translator = action_translator
        self.cooldown_handler = ZeldaCooldownHandler(env, action_translator)

        # per-reset state
        self._total_frames = 0
        self._state_change : Union[ZeldaGame | ZeldaStateChange] = None
        self._discounts = {}
        self._objectives : Objectives = None

        self.per_frame = []
        for key, value in scenario.per_frame.items():
            self.per_frame.append((key, value))

        self.per_reset = []
        for key, value in scenario.per_reset.items():
            self.per_reset.append((key, value))

        self.per_room = []
        for key, value in scenario.per_room.items():
            self.per_room.append((key, value))

    def __getattr__(self, name):
        if name == 'state':
            if isinstance(self._state_change, ZeldaStateChange):
                return self._state_change.current

            return self._state_change

        if name == 'state_change':
            return self._state_change if isinstance(self._state_change, ZeldaStateChange) else None

        return super().__getattr__(name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Per-reset state
        self._state_change = None
        self._discounts.clear()
        self.cooldown_handler.reset()
        self._objectives = Objectives()

        # Randomize the RNG if requested
        if not self.deterministic:
            for i in range(12):
                self.unwrapped.data.set_value(f'rng_{i}', randint(1, 255))

        # Move forward to the first frame where the agent can control Link
        _, _, _, info = self.cooldown_handler.skip(1)
        obs, info, frames_skipped = self.cooldown_handler.skip_uncontrollable_states(info)
        self._total_frames = frames_skipped + 1

        self._update_dictionary(None, info)
        return obs, info

    def step(self, action):
        # get link position for movement actions
        link_position = self.state.link.position

        # Take action
        obs, terminated, truncated, info, frames = self.cooldown_handler.act_and_wait(action, link_position)
        self._total_frames += frames

        self._update_dictionary(action, info)
        return obs, 0, terminated, truncated, info

    def _update_dictionary(self, action, info):
        if action is not None:
            info['action'] = self.action_translator.get_action_type(action)
            info['buttons'] = self._get_button_names(action, self.env.unwrapped.buttons)

        prev = self.state
        state = ZeldaGame(prev, self, info, self._total_frames)
        health_changed = self._apply_modifications(prev, state)

        if prev is not None:
            self._state_change = ZeldaStateChange(self, prev, state, self._discounts, health_changed)
        else:
            self._state_change = state

        objectives = self._objectives.get_current_objectives(prev, state)
        state.objectives = objectives
        state.wavefront = state.room.calculate_wavefront_for_link(objectives.targets)
        state.total_frames = self._total_frames

    def _get_button_names(self, act, buttons):
        result = []
        for i, b in enumerate(buttons):
            if act[i]:
                result.append(b)
        return result

    def _apply_modifications(self, prev : ZeldaGame, curr : ZeldaGame) -> float:
        health = curr.link.health

        if prev is None:
            for name, value in self.per_reset:
                self._set_value(curr, name, value)

        elif prev.full_location != curr.full_location:
            for name, value in self.per_room:
                self._set_value(curr, name, value)

        for name, value in self.per_frame:
            self._set_value(curr, name, value)

        return curr.link.health - health

    def _set_value(self, obj, name, value):
        if not hasattr(obj, name):
            obj = obj.link

        if isinstance(value, str):
            value = getattr(obj, value)

        setattr(obj, name, value)


__all__ = [ZeldaGameWrapper.__name__]
