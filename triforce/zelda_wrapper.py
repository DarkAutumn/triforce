"""
Responsible for interpreting complex game state and producing an object model in the 'info' dictionary.
Zelda has a very complicated combat system.  This class is responsible for detecting when the
agent has killed or injured an enemy.
"""

from random import randint
from typing import Tuple
import gymnasium as gym

from .rewards import StepRewards
from .model_definition import ZeldaScenario
from .objectives import Objectives
from .game_state_change import ZeldaStateChange
from .zelda_game import ZeldaGame
from .zelda_cooldown_handler import ZeldaCooldownHandler

class ZeldaGameWrapper(gym.Wrapper):
    """Interprets the game state and produces more information in the 'info' dictionary."""
    def __init__(self, env, scenario : ZeldaScenario = None, deterministic=False):
        super().__init__(env)

        self.deterministic = deterministic

        self.cooldown_handler = ZeldaCooldownHandler(env)

        # per-reset state
        self._total_frames = 0
        self._steps = 0
        self._prev_state = None
        self._discounts = {}
        self._objectives : Objectives = None

        self.per_reset = []
        self.per_room = []
        self.per_frame = []

        if scenario is not None:
            for key, value in scenario.per_reset.items():
                self.per_reset.append((key, value))

            for key, value in scenario.per_room.items():
                self.per_room.append((key, value))

            for key, value in scenario.per_frame.items():
                self.per_frame.append((key, value))

    def reset(self, **kwargs) -> Tuple[gym.spaces.Box, ZeldaGame]:
        obs, info = self.env.reset(**kwargs)

        # Per-reset state
        self._discounts.clear()
        self.cooldown_handler.reset()
        self._objectives = Objectives()
        self._prev_state = None

        # Randomize the RNG if requested
        if not self.deterministic:
            for i in range(12):
                self.unwrapped.data.set_value(f'rng_{i}', randint(1, 255))

        # Move forward to the first frame where the agent can control Link
        frames, terminated, truncated, info = self.cooldown_handler.gain_control_of_link()
        assert not terminated and not truncated

        # don't count these frames as part of the total since we weren't in control of Link
        self._total_frames = 0
        self._steps = -1

        state = self._update_state(None, None, info)
        return frames[-1] if frames else obs, state

    def step(self, action) -> Tuple[gym.spaces.Box, float, bool, bool, ZeldaStateChange]:
        # get link position for movement actions
        link_position = self._prev_state.link.position

        # Take action
        frames, terminated, truncated, info = self.cooldown_handler.act_and_wait(action, link_position)
        change = self._update_state(action, frames, info)
        return frames[-1], StepRewards(), terminated, truncated, change

    def _update_state(self, action, frames, info):
        prev, state = self._create_and_set_state(info)
        health_changed = self._apply_modifications(prev, state)

        objectives = self._objectives.get_current_objectives(prev, state)
        state.objectives = objectives
        state.wavefront = state.room.calculate_wavefront_for_link(objectives.targets)

        if frames:
            self._total_frames += len(frames)

        state.total_frames = self._total_frames
        info['total_frames'] = self._total_frames
        info['steps'] = self._steps = self._steps + 1

        if prev:
            return ZeldaStateChange(self, prev, state, action, frames, self._discounts, health_changed)

        return state

    def _create_and_set_state(self, info):
        prev = self._prev_state
        state = ZeldaGame(self, info, self._total_frames)
        self._prev_state = state
        return prev, state

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

    def _set_value(self, state, name, value):
        order = [state, state.link]
        if hasattr(state.link, name):
            order = [state.link, state]

        obj = order.pop(0)
        if not hasattr(obj, name):
            obj = order.pop(0)

        if isinstance(value, str):
            value = getattr(obj, value)

        setattr(obj, name, value)


__all__ = [ZeldaGameWrapper.__name__]
