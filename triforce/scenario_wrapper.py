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

        obs, state = super().reset(**kwargs)

        self._critic.clear()
        for ec in self._conditions:
            ec.clear()

        return obs, state

    def step(self, action):
        obs, rewards, terminated, truncated, state_change = self.env.step(action)
        state = state_change.state

        self._critic.critique_gameplay(state_change, rewards)
        state.info['score'] = self._critic.get_score(state_change)
        state.info['rewards'] = rewards

        end = (x.is_scenario_ended(state_change) for x in self._conditions)
        end = [x for x in end if x is not None]
        terminated = terminated or any((x[0] for x in end))
        truncated = truncated or any((x[1] for x in end))
        reason = [x[2] for x in end if x[2]]

        if reason:
            # I guess we could have more than one reason, but I'm not going to cover that corner case
            state.info['end'] = reason[0]

        if truncated or terminated:
            if hasattr(state, 'score'):
                state.info['final-score'] = state.info['score']

        return obs, rewards, terminated, truncated, state_change

__all__ = [ScenarioWrapper.__name__]
