from random import randint
import gymnasium as gym

from .critic import ZeldaCritic
from .end_condition import ZeldaEndCondition
from .zelda_game import is_mode_scrolling
from .damage_detector import DamageDetector

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
    def __init__(self, env, critics : [ZeldaCritic], end_conditions : [ZeldaEndCondition], verbose):
        super().__init__(env)

        self._critics = critics
        self._conditions = end_conditions
        self.verbose = verbose

        self._last_state = None
        self._report_interval = 10000
        self._curr_step = 0
        self._reward_summary = {}

    def reset(self, **kwargs):
        state = super().reset(**kwargs)

        self._last_state = None
        for c in self._critics:
            rewards = c.reward_history
            for key, value in rewards.items():
                self._reward_summary[key] = self._reward_summary.get(key, 0) + value

            c.clear()

        for ec in self._conditions:
            ec.clear()

        return state
    
    def step(self, act):
        obs, rewards, terminated, truncated, state = self.env.step(act)
        self._curr_step += 1

        if self._last_state is not None:
            for c in self._critics:
                rewards += c.critique_gameplay(self._last_state, state)

            terminated = terminated or any((x.is_terminated(state) for x in self._conditions))
            truncated = truncated or any((x.is_truncated(state) for x in self._conditions))

        # verbose==1 means we print every _report_interval steps
        # verbose==2 means we print every time the run ended

        if self.verbose:
            if self.verbose == 1 and self._curr_step % self._report_interval == 0:
                self.print_sorted_summary("Reward summary:")
                self._reward_summary.clear()

            if self.verbose == 2 and (terminated or truncated):
                print(f"Run ended in {self._curr_step} steps:")
                self.print_sorted_summary(f"Run ended in {self._curr_step} steps:")
                self._curr_step = 0
                self._reward_summary.clear()
        

        self._last_state = state
        return obs, rewards, terminated, truncated, state

    def print_sorted_summary(self, message):
        if not self._reward_summary:
            print("No rewards to report.")
        else:
            print(message)
            sorted_items = sorted(self._reward_summary.items(), key=lambda x: x[1], reverse=True)
            max_key_length = max(len(key) for key in self._reward_summary)

            for key, value in sorted_items:
                print(f"{round(value, 2):<{max_key_length + 3}.2f}{key}")

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
    
    def activate(self, env, verbose):
        self.verbose = verbose

        for c in self.critics:
            c.verbose = verbose

        for ec in self.end_conditions:
            ec.verbose = verbose

        frameskip_min, frameskip_max = frameskip_ranges[actions_per_second]
        env = DamageDetector(env)
        env = Frameskip(env, frameskip_min, frameskip_max)
        env = ScenarioGymWrapper(env, self.critics, self.end_conditions, verbose)
        return env
    
    def debug(self, debug):
        verbose = 2 if debug else 0


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