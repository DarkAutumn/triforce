import json
import os
import gymnasium as gym
import retro


from .zelda_game_data import zelda_game_data
from .scenario_dungeon import DungeonEndCondition, ZeldaDungeonCritic
from .scenario_gauntlet import GauntletEndCondition, ZeldaGuantletRewards
from .scenario_dungeon_combat import ZeldaDungeonCombatCritic, ZeldaDungeonCombatEndCondition

class ScenarioGymWrapper(gym.Wrapper):
    """Wraps the environment to actually call our critics and end conditions."""
    def __init__(self, env, scenario, verbose):
        super().__init__(env)

        self._scenario = scenario
        self._critics = [c(verbose=verbose) for c in scenario.critics]
        self._conditions = [ec(verbose=verbose) for ec in scenario.end_conditions]
        
        self._curr_room = -1
        self.game_data = zelda_game_data
        self.verbose = verbose

        self._last_state = None
        self._report_interval = 10000
        self._curr_step = 0
        self._reward_summary = {}
        self._end_summary = {}

    def reset(self, **kwargs):
        self._curr_room = (self._curr_room + 1) % len(self._scenario.all_start_states)
        save_state = self._scenario.all_start_states[self._curr_room]
        print(f"Starting room: {save_state}")
        
        env_unwrapped = self.unwrapped
        env_unwrapped.load_state(save_state, retro.data.Integrations.CUSTOM_ONLY)

        state = super().reset(**kwargs)

        # assign data for the scenario
        if self._scenario.data:
            data = env_unwrapped.data
            for key, value in self._scenario.data.items():
                data.set_value(key, value)

        # update history
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

    def __init__(self, name, description, critics : [str], end_conditions : [str], level, start, data):
        self.name = name
        self.description = description
        self.critics = [ZeldaScenario.resolve_critic(x) for x in critics]
        self.end_conditions = [ZeldaScenario.resolve_end_condition(x) for x in end_conditions]
        self.level = level
        self.start = start
        self.data = data
        
        self.all_start_states = []
        for x in start:
            i = len(self.all_start_states)
            self.all_start_states.extend(zelda_game_data.get_savestates_by_name(x))
            if len(self.all_start_states) == i:
                raise Exception(f'Could not find save state for {x}')
            
        if not self.all_start_states:
            raise Exception(f'Could not find any save states for {name}')
        

    def __str__(self):
        return f'{self.name} - {self.description}'
    
    def activate(self, env, verbose):
        self.verbose = verbose

        for c in self.critics:
            c.verbose = verbose

        for ec in self.end_conditions:
            ec.verbose = verbose

        env = ScenarioGymWrapper(env, self, verbose)
        return env
    
    def debug(self, debug):
        verbose = 2 if debug else 0


    @classmethod
    def get(cls, name):
        return ZeldaScenario._scenarios.get(name, None)
    
    @classmethod
    def get_all_scenarios(cls):
        if not ZeldaScenario._scenarios:
            # load scenarios.json
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            scenarios_file = os.path.join(curr_dir, 'scenarios.json')
            with open(scenarios_file, 'r') as f:
                data = json.load(f)
            
            for json_scenario in data['scenarios']:
                scenario = ZeldaScenario(**json_scenario)
                ZeldaScenario._scenarios[scenario.name] = scenario

        return ZeldaScenario._scenarios.keys()
    
    @classmethod
    def resolve_critic(cls, name):
        if name == 'ZeldaGuantletRewards':
            return ZeldaGuantletRewards
        elif name == 'ZeldaDungeonCritic':
            return ZeldaDungeonCritic
        elif name == 'ZeldaDungeonCombatCritic':
            return ZeldaDungeonCombatCritic
        
        raise Exception(f'Unknown critic {name}')
    
    @classmethod
    def resolve_end_condition(cls, name):
        if name == 'GauntletEndCondition':
            return GauntletEndCondition
        elif name == 'DungeonEndCondition':
            return DungeonEndCondition
        elif name == 'ZeldaDungeonCombatEndCondition':
            return ZeldaDungeonCombatEndCondition
        
        raise Exception(f'Unknown end condition {name}')

__all__ = ['ZeldaScenario']