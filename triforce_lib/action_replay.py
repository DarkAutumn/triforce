# a helper class for testing

import retro

from .action_space import ZeldaActionSpace
from .zelda_wrapper import ZeldaGameWrapper

class ZeldaActionReplay:
    def __init__(self, savestate):
        env = retro.make(game='Zelda-NES', state=savestate, inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode="human")
        env = ZeldaGameWrapper(env, deterministic=True)
        env = ZeldaActionSpace(env)

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

        self.env = env

    def reset(self):
        self.actions_taken = ""
        return self.env.reset()
    
    def step(self, button):
        if button == 'x':
            self.reset()
            self.actions_taken = self.actions_taken[:-1]

            for button in self.actions_taken:
                self.step(button)

        else:
            self.actions_taken += button
            action = self.map[button]
            
            result =  self.env.step(action)
            self.env.render()
            
            return result