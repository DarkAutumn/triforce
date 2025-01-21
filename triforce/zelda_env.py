# A wrapper to create a Zelda environment.

import random
import retro

from .training_hints import TrainingHintWrapper
from .state_change_wrapper import StateChangeWrapper
from .gym_translation_wrapper import GymTranslationWrapper
from .frame_skip_wrapper import FrameSkipWrapper
from .action_space import ZeldaActionSpace
from .observation_wrapper import ObservationWrapper
from .scenario_wrapper import ScenarioWrapper, TrainingScenarioDefinition

def make_zelda_env(scenario : TrainingScenarioDefinition, action_space : str, **kwargs):
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
    render_mode = kwargs.get('render_mode', None)
    translation = kwargs.get('translation', True)
    frame_stack = kwargs.get('frame_stack', 1)
    obs_kind = kwargs.get('obs_kind', 'viewport')

    state = random.choice(scenario.start)
    env = retro.make(game='Zelda-NES', state=state, inttype=retro.data.Integrations.CUSTOM_ONLY,
                     render_mode=render_mode)

    # Skip frames where Link is not controllable.  Returns all frames skipped as its observation.
    env = FrameSkipWrapper(env)

    # Convert a standard 'info' dictionary into an object model to interact with the environment.
    # Instead of 'info', reset returns a ZeldaGame and step returns a StateChange.  The info dictionary
    # is still available at ZeldaGame.info and StateChange.current.info.
    env = StateChangeWrapper(env, scenario)

    # Whether to use optional training hints.
    if scenario.use_hints:
        env = TrainingHintWrapper(env)

    # Reduces the action space to only the actions we want the model to take, and what is actually possible in game.
    env = ZeldaActionSpace(env, action_space)

    # Converts our list of frames into a standard observation space.
    env = ObservationWrapper(env, obs_kind, frame_stack, frame_skip=2, normalize=True)

    # Process the scenario. This is where we define the end conditions and rewards for the scenario.
    # Replaces the float reward with a StepRewards object.
    env = ScenarioWrapper(env, scenario)

    # Translates our state/state_change objects back into the info dictionary, and our StepRewards back into
    # a float.
    if translation:
        env = GymTranslationWrapper(env)

    return env

__all__ = ['make_zelda_env']
