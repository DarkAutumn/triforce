# A wrapper to create a Zelda environment.

import retro

from .state_change_wrapper import StateChangeWrapper
from .gym_translation_wrapper import GymTranslationWrapper
from .frame_skip_wrapper import FrameSkipWrapper
from .action_space import ZeldaActionSpace
from .observation_wrapper import ObservationWrapper
from .scenario_wrapper import ScenarioWrapper, TrainingScenarioDefinition
from .rewards import EpisodeRewardTracker

def make_zelda_env(scenario : TrainingScenarioDefinition, action_space : str, *,
                   obs_kind = 'viewport', render_mode = None, translation=True):
    """
    Creates a Zelda retro environment for the given scenario.
    Args:
        scenario:     The scenario to use.
        action_space: The action space to use, typically ZeldaModelDefinition.action_space.
        grayscale:    Whether to convert the observation to grayscale.
        framestack:   The number of frames to stack in the observation.
        obs_kind:     The kind of observation to use (viewport, gameplay).
        render_mode:  The render mode to use.
    """
    env = retro.make(game='Zelda-NES', state=scenario.start[0], inttype=retro.data.Integrations.CUSTOM_ONLY,
                     render_mode=render_mode)

    # Skip frames where Link is not controllable.  Returns all frames skipped as its observation.
    env = FrameSkipWrapper(env)

    # Convert a standard 'info' dictionary into an object model to interact with the environment.
    # Instead of 'info', reset returns a ZeldaGame and step returns a StateChange.  The info dictionary
    # is still available at ZeldaGame.info and StateChange.current.info.
    env = StateChangeWrapper(env, scenario)

    # Reduces the action space to only the actions we want the model to take, and what is actually possible in game.
    env = ZeldaActionSpace(env, action_space)

    # Converts our list of frames into a standard observation space.
    env = ObservationWrapper(env, obs_kind, normalize=True)

    # Process the scenario. This is where we define the end conditions and rewards for the scenario.
    # Replaces the float reward with a StepRewards object.
    env = ScenarioWrapper(env, scenario)

    # Calculate the total reward for the episode.
    env = EpisodeRewardTracker(env)

    # Translates our state/state_change objects back into the info dictionary, and our StepRewards back into
    # a float.
    if translation:
        env = GymTranslationWrapper(env)

    return env

__all__ = ['make_zelda_env']
