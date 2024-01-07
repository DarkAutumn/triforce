import os
import retro

from stable_baselines3 import PPO

from .rewards_base import ZeldaRewardBase
from .frameskip import Frameskip

class ZeldaScenario:
    _scenarios = {}
    _is_initialized = False

    def __init__(self, name, description, algorithm, policy, state, rewards = None):
        self.name = name
        self.description = description
        self.algorithm = algorithm
        self.policy = policy
        self.state = state
        self.rewards = rewards

        ZeldaScenario._scenarios[name] = self

    def __str__(self):
        return f'{self.name} - {self.description}'

    def __repr__(self):
        return f'{self.name} - {self.description}'
    
    @classmethod
    def initialize(cls):
        if not cls._is_initialized:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            retro.data.Integrations.add_custom_path(os.path.join(script_dir, 'custom_integrations'))
            cls._is_initialized = True

    @classmethod
    def get(cls, name):
        cls.initialize()
        return cls._scenarios[name]
    
    def load_model(self, path):
        if self.algorithm != 'ppo':
            raise Exception(f'Unsupported algorithm: {self.algorithm}')
        
        return PPO.load(path)
    
    def create_model(self, env, **kwargs):
        if self.algorithm != 'ppo':
            raise Exception(f'Unsupported algorithm: {self.algorithm}')
        
        return PPO('CnnPolicy', env, **kwargs)

    def create_env(self, **kwargs):
        env = retro.make(game='Zelda-NES', state=self.state, inttype=retro.data.Integrations.CUSTOM_ONLY, **kwargs)

        # We only take action every so many frames, not every single frame.
        env = Frameskip(env, 10, 20)
        env = self.rewards(env, verbose=True)

        return env
    
    def get_model_name(self, iterations):
        return f'{self.algorithm}_{self.policy}_{self.name}_{iterations}.zip'
    
ZeldaScenario('gauntlet', 'Run link from the starting screen to the furthest right screen without dying.', 'ppo', 'cnn', '120w.state', rewards = ZeldaRewardBase)

def load_scenario(name) -> ZeldaScenario:
    return ZeldaScenario.get(name)

__all__ = ['load_scenario']