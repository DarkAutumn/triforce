import gymnasium as gym
import retro

from .zelda_game_data import zelda_game_data

from . import end_conditions
from . import critics

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

        obs, info = super().reset(**kwargs)

        # assign data for the scenario
        self.set_data(env_unwrapped, self._scenario.data, info)
        self.set_data(self.unwrapped, self._scenario.fixed, info)

        self._last_state = None
        for c in self._critics:
            c.clear()

        for ec in self._conditions:
            ec.clear()

        self._last_state = info

        return obs, info

    def set_data(self, env_unwrapped, data, info):
        if data:
            game_data = env_unwrapped.data
            for key, value in data.items():
                game_data.set_value(key, value)
                assert key in info
                old = info[key]
                info[key] = value
    
    def step(self, act):
        obs, rewards, terminated, truncated, info = self.env.step(act)

        if self._last_state is not None:
            reward_dict = {}
            for c in self._critics:
                c.critique_gameplay(self._last_state, info, reward_dict)
                c.set_score(self._last_state, info)

            if reward_dict:
                for value in reward_dict.values():
                    rewards += value

            info['rewards'] = reward_dict

            end = (x.is_scenario_ended(self._last_state, info) for x in self._conditions)
            end = [x for x in end if x is not None]
            terminated = terminated or any((x[0] for x in end))
            truncated = truncated or any((x[1] for x in end))
            reason = [x[2] for x in end if x[2]]

            if reason:
                # I guess we could have more than one reason, but I'm not going to cover that corner case
                info['end'] = reason[0]

            if truncated or terminated:
                if 'score' in info:
                    info['final-score'] = info['score']
                else:
                    info['final-score'] = 0

        self.set_data(self.unwrapped, self._scenario.fixed, info)
        self._last_state = info
        return obs, rewards, terminated, truncated, info

class ZeldaScenario:
    _scenarios = {}

    def __init__(self, name, description, critics : [str], end_conditions : [str], level, start, data, fixed):
        self.name = name
        self.description = description
        self.critics = [ZeldaScenario.resolve_critic(x) for x in critics]
        self.end_conditions = [ZeldaScenario.resolve_end_condition(x) for x in end_conditions]
        self.level = level
        self.start = start
        self.data = data
        self.fixed = fixed
        
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
    
    def simulate_step(self, last_info, next_info):
        critics = [c() for c in self.critics]
        for c in critics:
            c.clear()

        end_conditions = [ec() for ec in self.end_conditions]
        for ec in end_conditions:
            ec.clear()

        reward_dict = {}
        for c in critics:
            c.critique_gameplay(last_info, next_info, reward_dict)

        end = (x.is_scenario_ended(last_info, next_info) for x in end_conditions)
        end = [x for x in end if x is not None]
        terminated = any((x[0] for x in end))
        truncated = any((x[1] for x in end))
        reason = [x[2] for x in end if x[2]]

        return reward_dict, terminated, truncated, reason
    
    def activate(self, env):
        env = ScenarioGymWrapper(env, self)
        return env
    
    @classmethod
    def get(cls, name):
        return cls._scenarios.get(name, None)

    @classmethod
    def initialize(cls, scenarios):
        if not ZeldaScenario._scenarios:
            
            for json_scenario in scenarios:
                if 'fixed' not in json_scenario:
                    json_scenario['fixed'] = {}

                scenario = ZeldaScenario(**json_scenario)
                ZeldaScenario._scenarios[scenario.name] = scenario

        return ZeldaScenario._scenarios
    
    @classmethod
    def resolve_critic(cls, name):
        critic = getattr(critics, name, None)
        if not critic:
            raise Exception(f'Unknown critic {name}')
        
        return critic
    
    @classmethod
    def resolve_end_condition(cls, name):
        end_condition = getattr(end_conditions, name, None)
        if not end_condition:
            raise Exception(f'Unknown end condition {name}')
        
        return end_condition

__all__ = ['ZeldaScenario']