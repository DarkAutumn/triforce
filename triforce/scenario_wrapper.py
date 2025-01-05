"""Wraps the environment to call our critic and end conditions."""

import gymnasium as gym
import retro

from .zelda_game_data import zelda_game_data
from .models_and_scenarios import ZeldaScenario
from . import critics
from . import end_conditions

class ScenarioWrapper(gym.Wrapper):
    """Wraps the environment to call our critic and end conditions."""
    def __init__(self, env, scenario : ZeldaScenario):
        super().__init__(env)

        self._scenario = scenario
        self._critic = getattr(critics, scenario.critic)()
        for k, v in scenario.reward_overrides.items():
            assert hasattr(self._critic, k)
            setattr(self._critic, k, v)

        self._conditions = [getattr(end_conditions, ec)() for ec in scenario.end_conditions]

        self._curr_room = -1
        self.game_data = zelda_game_data

    def reset(self, **kwargs):
        if len(self._scenario.start) > 1:
            self._curr_room = (self._curr_room + 1) % len(self._scenario.start)
            save_state = self._scenario.start[self._curr_room]

            env_unwrapped = self.unwrapped
            env_unwrapped.load_state(save_state, retro.data.Integrations.CUSTOM_ONLY)
        else:
            env_unwrapped = self.unwrapped

        obs, info = super().reset(**kwargs)

        # assign data for the scenario
        self.__set_data(env_unwrapped, self._scenario.data, info)
        self.__set_data(self.unwrapped, self._scenario.fixed, info)

        self._critic.clear()
        for ec in self._conditions:
            ec.clear()

        return obs, info

    def __set_data(self, env_unwrapped, data, info):
        if data:
            game_data = env_unwrapped.data
            for key, value in data.items():
                game_data.set_value(key, value)
                assert key in info
                info[key] = value

    def step(self, action):
        obs, rewards, terminated, truncated, info = self.env.step(action)

        reward_dict = {}
        state_change = info['state_change']

        self._critic.critique_gameplay(state_change, reward_dict)
        info['score'] = self._critic.get_score(state_change)

        if reward_dict:
            for value in reward_dict.values():
                rewards += value

        info['rewards'] = reward_dict

        end = (x.is_scenario_ended(state_change) for x in self._conditions)
        end = [x for x in end if x is not None]
        terminated = terminated or any((x[0] for x in end))
        truncated = truncated or any((x[1] for x in end))
        reason = [x[2] for x in end if x[2]]

        success = any(end.startswith("success") for end in reason)
        if reason:
            # I guess we could have more than one reason, but I'm not going to cover that corner case
            info['end'] = reason[0]

        if truncated or terminated:
            if 'score' in info and success:
                info['final-score'] = info['score']

        self.__set_data(self.unwrapped, self._scenario.fixed, info)
        return obs, rewards, terminated, truncated, info

__all__ = [ScenarioWrapper.__name__]
