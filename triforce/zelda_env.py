# A wrapper to create a Zelda environment.

import retro

from .gym_translation_wrapper import GymTranslationWrapper
from .zelda_wrapper import ZeldaGameWrapper
from .action_space import ZeldaActionSpace
from .observation_wrapper import ObservationWrapper
from .scenario_wrapper import ScenarioWrapper
from .models_and_scenarios import ZeldaScenario

def make_zelda_env(scenario : ZeldaScenario, action_space : str, *,
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

    # Wrap the game to produce new info about game state and to hold the button down after the action is
    # taken to achieve the desired number of actions per second.
    env = ZeldaGameWrapper(env, scenario)

    # Reduces the action space to only the actions we want the model to take, and what is actually possible in game.
    env = ZeldaActionSpace(env, action_space)

    # Frame stack and convert to grayscale if requested
    env = ObservationWrapper(env, obs_kind, normalize=False)

    # Activate the scenario.  This is where rewards and end conditions are checked, using some of the new
    # info state provded by ZeldaGameWrapper above.
    env = ScenarioWrapper(env, scenario)

    # Translate our object-oriented environment into a gym environment.
    if translation:
        env = GymTranslationWrapper(env)

    return env

__all__ = ['make_zelda_env']
