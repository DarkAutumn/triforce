# a helper class for testing

import retro

from .action_space import ZeldaAttackOnlyActionSpace
from .zelda_wrapper import ZeldaGameWrapper

class ZeldaActionReplay:
    def __init__(self, savestate, wrapper=None, render_mode=None):
        env = retro.make(game='Zelda-NES', state=savestate, inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode=render_mode)
        env = ZeldaGameWrapper(env, deterministic=True)
        env = ZeldaAttackOnlyActionSpace(env)
        if wrapper:
            env = wrapper(env)

        self.map = {}

        for i in range(len(env.actions)):
            if env.actions[i][0] == 'UP':
                self.map['u'] = i

            if env.actions[i][0] == 'DOWN':
                self.map['d'] = i

            if env.actions[i][0] == 'LEFT':
                self.map['l'] = i

            if env.actions[i][0] == 'RIGHT':
                self.map['r'] = i

            if env.actions[i][0] == 'A':
                self.map['a'] = i

            if env.actions[i][0] == 'B':
                self.map['b'] = i

        env.reset()
        self.actions_taken = ""
        self.env = env

    def __delattr__(self, __name: str) -> None:
        self.env.close()

    def reset(self):
        self.actions_taken = ""
        return self.env.reset()
    
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
            action = self.map[button]
            
            result =  self.env.step(action)
            self.env.render()
            
            return result