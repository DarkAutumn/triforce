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
    def __init__(self, env, scenario):
        super().__init__(env)

        self._scenario = scenario
        self._critics = [c() for c in scenario.critics]
        self._conditions = [ec() for ec in scenario.end_conditions]
        
        self._curr_room = -1
        self.game_data = zelda_game_data

        self._last_state = None

    def reset(self, **kwargs):
        if len(self._scenario.all_start_states) > 1:
            self._curr_room = (self._curr_room + 1) % len(self._scenario.all_start_states)
            save_state = self._scenario.all_start_states[self._curr_room]
            
            env_unwrapped = self.unwrapped
            env_unwrapped.load_state(save_state, retro.data.Integrations.CUSTOM_ONLY)
        else:
            env_unwrapped = self.unwrapped

        state = super().reset(**kwargs)

        # assign data for the scenario
        if self._scenario.data:
            data = env_unwrapped.data
            for key, value in self._scenario.data.items():
                data.set_value(key, value)

        self._last_state = None
        for c in self._critics:
            c.clear()

        for ec in self._conditions:
            ec.clear()

        return state
    
    def step(self, act):
        obs, rewards, terminated, truncated, info = self.env.step(act)

        if self._last_state is not None:
            reward_dict = {}
            for c in self._critics:
                c.critique_gameplay(self._last_state, info, reward_dict)

            if reward_dict:
                for value in reward_dict.values():
                    rewards += value

            info['rewards'] = reward_dict

            end = [x.is_scenario_ended(self._last_state, info) for x in self._conditions]
            terminated = terminated or any((x[0] for x in end))
            truncated = truncated or any((x[1] for x in end))
            reason = [x[2] for x in end if x[2]]

            if reason:
                # I guess we could have more than one reason, but I'm not going to cover that corner case
                info['end'] = reason[0]

        self._last_state = info
        return obs, rewards, terminated, truncated, info


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
    
    def activate(self, env):
        env = ScenarioGymWrapper(env, self)
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