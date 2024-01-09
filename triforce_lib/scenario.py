import gymnasium as gym

from .wrappers import Frameskip
from .critic import ZeldaCritic
from .end_condition import ZeldaEndCondition

# average 15 frames per second, or 4 decisions per second
frameskip_min = 10
frameskip_max = 20

class ScenarioGymWrapper(gym.Wrapper):
    def __init__(self, env, critics : [ZeldaCritic], end_conditions : [ZeldaEndCondition]):
        super().__init__(env)

        self._critics = critics
        self._conditions = end_conditions

        self._last_state = None

    def reset(self, **kwargs):
        state = super().reset(**kwargs)

        self._last_state = None
        for c in self._critics:
            c.clear()

        return state
    
    def step(self, act):
        obs, rewards, terminated, truncated, state = self.env.step(act)

        if self._last_state is not None:
            for c in self._critics:
                rewards += c.get_rewards(self._last_state, state)

            terminated = terminated or any((x.is_terminated(state) for x in self._conditions))
            truncated = truncated or any((x.is_truncated(state) for x in self._conditions))

        self._last_state = state
        return obs, rewards, terminated, truncated, state

class ZeldaScenario:
    _scenarios = {}

    def __init__(self, name, description, start_state, critics : [ZeldaCritic], end_conditions : [ZeldaEndCondition]):
        self.name = name
        self.description = description
        self.start_state = start_state
        self.critics = critics
        self.end_conditions = end_conditions

    def __str__(self):
        return f'{self.name} - {self.description}'
    
    def activate(self, env):
        env = Frameskip(env, frameskip_min, frameskip_max)
        env = ScenarioGymWrapper(env, self.critics, self.end_conditions)
        return env
    
    def debug(self, debug):
        verbose = 2 if debug else 0
        
        self.verbose = verbose

        for c in self.critics:
            c.verbose = verbose

        for ec in self.end_conditions:
            ec.verbose = verbose


    @classmethod
    def get(cls, name):
        return ZeldaScenario._scenarios.get(name, None)
    
    @classmethod
    def register(cls, scenario):
        if scenario.name in ZeldaScenario._scenarios:
            raise Exception(f'Scenario {scenario.name} already registered')

        ZeldaScenario._scenarios[scenario.name] = scenario

    @classmethod
    def get_all_scenarios(cls):
        return ZeldaScenario._scenarios.keys()

__all__ = ['ZeldaScenario', 'ZeldaEndCondition', 'ZeldaGameplayEndCondition']