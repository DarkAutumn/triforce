# pylint: disable=all

import gymnasium as gym
import retro
from triforce.action_space import ZeldaActionSpace
from triforce.objective_selector import ObjectiveSelector
from triforce.zelda_hit_detect import ZeldaHitDetect
from triforce.zelda_wrapper import ZeldaGameWrapper

class CriticWrapper(gym.Wrapper):
    """Wraps the environment to actually call our critics and end conditions."""
    def __init__(self, env, critics=None, end_conditions=None):
        super().__init__(env)

        assert critics or end_conditions

        self.critics = critics or []
        self.end_conditions = end_conditions or []
        self._last = None

    def reset(self, **kwargs):
        state = super().reset(**kwargs)

        for c in self.critics:
            c.clear()

        for ec in self.end_conditions:
            ec.clear()

        self._last = state[1]
        return state

    def step(self, act):
        obs, rewards, terminated, truncated, state = self.env.step(act)
        reward_dict = {}

        for c in self.critics:
            c.critique_gameplay(self._last, state, reward_dict)

        end = [x.is_scenario_ended(self._last, state) for x in self.end_conditions]
        terminated = terminated or any((x[0] for x in end))
        truncated = truncated or any((x[1] for x in end))

        state['rewards'] = reward_dict

        self._last = state
        return obs, rewards, terminated, truncated, state

class ZeldaActionReplay:
    def __init__(self, savestate, wrapper=None, render_mode=None):
        env = retro.make(game='Zelda-NES', state=savestate, inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode=render_mode)
        self.data = env.data
        env = ZeldaGameWrapper(env, deterministic=True)
        env = ZeldaHitDetect(env)
        env = ObjectiveSelector(env)
        env = ZeldaActionSpace(env, 'all')
        if wrapper:
            env = wrapper(env)

        self.actions = env.actions

        self.buttons = {
            'u': 'UP',
            'd': 'DOWN',
            'l': 'LEFT',
            'r': 'RIGHT',
            'a': 'A',
            'b': 'B',
        }

        env.reset()
        self.actions_taken = ""
        self.env = env

    def __delattr__(self, __name: str) -> None:
        self.env.close()

    def reset(self):
        self.actions_taken = ""
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
                yield self.step(a)

            i = idx

    def step(self, button):
        if button == 'x':
            self.reset()
            self.actions_taken = self.actions_taken[:-1]

            for button in self.actions_taken:
                self.step(button)

        elif button == 'c':
            self.reset()
            self.actions_taken = ""

        else:
            self.actions_taken += button

            action = self.buttons[button]
            if action in ['A', 'B']:
                action = [self.get_prev_direction(), action]

            act = self.get_real_action(action)

            result =  self.env.step(act)
            self.env.render()

            return result

    def get_real_action(self, action):
        if not isinstance(action, list):
            action = [action]

        return self.actions.index(action)

    def get_prev_direction(self):
        for x in reversed(self.actions_taken):
            if x in ['u', 'd', 'l', 'r']:
                return self.buttons[x]
