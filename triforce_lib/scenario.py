from random import randint
import gymnasium as gym

from .critic import ZeldaCritic
from .end_condition import ZeldaEndCondition

class ScenarioGymWrapper(gym.Wrapper):
    """Wraps the environment to actually call our critics and end conditions."""
    def __init__(self, env, critics : [ZeldaCritic], end_conditions : [ZeldaEndCondition], verbose):
        super().__init__(env)

        self._critics = [c(verbose=verbose) for c in critics]
        self._conditions = [ec(verbose=verbose) for ec in end_conditions]
        self.verbose = verbose

        self._last_state = None
        self._report_interval = 10000
        self._curr_step = 0
        self._reward_summary = {}
        self._end_summary = {}

    def reset(self, **kwargs):
        state = super().reset(**kwargs)

        self._last_state = None
        for c in self._critics:
            rewards = c.reward_history
            for key, value in rewards.items():
                self._reward_summary[key] = self._reward_summary.get(key, 0) + value

            c.clear()

        for ec in self._conditions:
            rewards = ec.end_causes
            for key, value in rewards.items():
                self._end_summary[key] = self._end_summary.get(key, 0) + value

            ec.clear()

        return state
    
    def step(self, act):
        obs, rewards, terminated, truncated, state = self.env.step(act)
        self._curr_step += 1

        if self._last_state is not None:
            for c in self._critics:
                rewards += c.critique_gameplay(self._last_state, state)

            end = [x.is_scenario_ended(self._last_state, state) for x in self._conditions]
            terminated = terminated or any((x[0] for x in end))
            truncated = truncated or any((x[1] for x in end))

        # verbose==1 means we print every _report_interval steps
        # verbose==2 means we print every time the run ended

        if self.verbose:
            if self.verbose == 1 and self._curr_step % self._report_interval == 0:
                self.print_sorted_summary("Reward summary:")
                self.print_end_summary()
                self._reward_summary.clear()

            if self.verbose == 2 and (terminated or truncated):
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

    def print_end_summary(self):
        if self.verbose:
            self.print_sorted_summary("Scenario end reason:")
            for key, value in self._end_summary.items():
                print(f"    {value}: {key}")

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