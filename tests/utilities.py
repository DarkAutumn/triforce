# pylint: disable=all
import os
import sys

from triforce.rewards import StepRewards

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import gymnasium as gym
import retro
from triforce.zelda_enums import Direction
from triforce.action_space import ActionKind, ActionTaken, ZeldaActionSpace
from triforce.state_change_wrapper import StateChange, StateChangeWrapper
from triforce.frame_skip_wrapper import FrameSkipWrapper

class CriticWrapper(gym.Wrapper):
    """Wraps the environment to actually call our critics and end conditions."""
    def __init__(self, env, critics=None, end_conditions=None):
        super().__init__(env)

        assert critics or end_conditions

        self.critics = critics or []
        self.end_conditions = end_conditions or []
        self._discounts = {}

    def reset(self, **kwargs):
        obs, state = super().reset(**kwargs)

        for c in self.critics:
            c.clear()

        for ec in self.end_conditions:
            ec.clear()

        self._discounts.clear()
        return obs, state

    def step(self, act):
        obs, _, terminated, truncated, change = self.env.step(act)
        rewards = StepRewards()

        for c in self.critics:
            c.critique_gameplay(change, rewards)

        end = [x.is_scenario_ended(change) for x in self.end_conditions]
        terminated = terminated or any((x[0] for x in end))
        truncated = truncated or any((x[1] for x in end))

        return obs, rewards, terminated, truncated, change

class ZeldaActionReplay:
    def __init__(self, savestate, wrapper=None, render_mode=None):
        env = retro.make(game='Zelda-NES', state=savestate, inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode=render_mode)
        self.data = env.data
        env = FrameSkipWrapper(env, deterministic=True)
        env = StateChangeWrapper(env, None)
        env = ZeldaActionSpace(env, 'all')
        self.action_space = env
        if wrapper:
            env = wrapper(env)

        self._prev = None

        _, state = env.reset()
        self._set_prev(state)

        self.actions_taken = []
        self.env = env

    def deactivate(self):
        if self._prev:
            self._prev.deactivate()
            self._prev = None

    def _set_prev(self, state):
        self.deactivate()
        if isinstance(state, StateChange):
            state = state.state

        self._prev = state

    def __delattr__(self, __name: str) -> None:
        self.deactivate()
        self.env.close()

    def reset(self):
        self.actions_taken = []
        result = self.env.reset()
        self.data.set_value('hearts_and_containers', 0xff)
        return result


    def run_steps(self, commands):
        for x in self.iterate_steps(commands):
            pass

    def iterate_steps(self, commands):
        i = 0
        while i < len(commands):
            a = commands[i]
            count = 0
            idx = i + 1
            while idx < len(commands) and '0' <= commands[idx] <= '9':
                count = count * 10 + int(commands[idx])
                idx += 1

            for i in range(max(count, 1)):
                yield self.move(a)

            i = idx

    def move(self, direction):
        if direction in ['u', 'd', 'l', 'r']:
            if direction == 'u':
                direction = Direction.N
            elif direction == 'd':
                direction = Direction.S
            elif direction == 'l':
                direction = Direction.W
            elif direction == 'r':
                direction = Direction.E

        return self.act(ActionKind.MOVE, direction)

    def act(self, action : ActionKind, direction : Direction):
        assert action in self._prev.link.get_available_actions(True)
        return self.step((action, direction))


    def step(self, action):
        self.actions_taken.append(action)
        result = self.env.step(action)
        self.env.render()

        state_change : StateChange = result[-1]
        self._set_prev(state_change)
        return result
