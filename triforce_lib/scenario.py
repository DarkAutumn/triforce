from random import randint
import gymnasium as gym

from .critic import ZeldaCritic
from .end_condition import ZeldaEndCondition
from .zelda_modes import is_mode_scrolling

actions_per_second = 4

# Frame skip values based on actions per second
frameskip_ranges = {
    1: (58, 62),      # one action every ~60 frames
    2: (30, 50),      # one action every ~40 frames
    3: (20, 30),      # one action every ~20 frames
    4: (10, 20),      # one action every ~15 frames
    5: (9, 15),       # one action every ~12 frames
}

class ScenarioGymWrapper(gym.Wrapper):
    """Wraps the environment to actually call our critics and end conditions."""
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
                rewards += c.critique_gameplay(self._last_state, state)

            terminated = terminated or any((x.is_terminated(state) for x in self._conditions))
            truncated = truncated or any((x.is_truncated(state) for x in self._conditions))

        self._last_state = state
        return obs, rewards, terminated, truncated, state


class Frameskip(gym.Wrapper):
    """Skip every min-max frames.  This ensures that we do not take too many actions per second."""
    def __init__(self, env, skip_min, skip_max):
        super().__init__(env)
        self._skip_min = skip_min
        self._skip_max = skip_max

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, act):
        total_rew = 0.0
        terminated = False
        truncated = False
        for i in range(randint(self._skip_min, self._skip_max)):
            obs, rew, terminated, truncated, info = self.env.step(act)
            total_rew += rew
            if terminated or truncated:
                break

        mode = info["mode"]
        while is_mode_scrolling(mode):
            obs, rew, terminated, truncated, info = self.env.step(act)
            total_rew += rew
            if terminated or truncated:
                break
            
            mode = info["mode"]

        return obs, total_rew, terminated, truncated, info


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
        frameskip_min, frameskip_max = frameskip_ranges[actions_per_second]
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