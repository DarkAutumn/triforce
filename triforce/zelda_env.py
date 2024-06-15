# A wrapper to create a Zelda environment.

import retro

from .critics import MultiHeadCritic
from .objective_selector import ObjectiveSelector
from .zelda_wrapper import ZeldaGameWrapper
from .action_space import MultiHeadInputWrapper, ZeldaActionSpace
from .zelda_observation_wrapper import FrameCaptureWrapper, MultiHeadObservationWrapper, ZeldaObservationWrapper
from .zelda_vector_features import ZeldaVectorFeatures
from .scenario_wrapper import ScenarioWrapper
from .models_and_scenarios import ZeldaScenario

def make_zelda_env(scenario : ZeldaScenario, action_space : str, *, grayscale = True, framestack = 1,
                   obs_kind = 'viewport', render_mode = None):
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

    # Capture the raw observation frames into a deque.
    env = FrameCaptureWrapper(env, render_mode == 'rgb_array')
    captured_frames = env.frames

    # Wrap the game to produce new info about game state and to hold the button down after the action is
    # taken to achieve the desired number of actions per second.
    env = ZeldaGameWrapper(env)

    # The AI orchestration piece.  This is responsible for selecting the model to use and the target
    # objective.
    env = ObjectiveSelector(env)

    # Frame stack and convert to grayscale if requested
    env = ZeldaObservationWrapper(env, captured_frames, grayscale, kind=obs_kind, framestack=framestack)

    # Reduce the action space to only the actions we want the model to take (no need for A+B for example,
    # since that doesn't make any sense in Zelda)
    env = ZeldaActionSpace(env, action_space)

    # Extract features from the game for the model, like whether link has beams or has keys and expose
    # these as observations.
    env = ZeldaVectorFeatures(env)

    # Activate the scenario.  This is where rewards and end conditions are checked, using some of the new
    # info state provded by ZeldaGameWrapper above.
    env = ScenarioWrapper(env, scenario)

    return env

def make_multihead_zelda_env(save_state, *, render_mode = None, device = 'cpu', rgb_deque=None):
    """Creates a Zelda retro environment for use with the multi-headed model."""
    if rgb_deque is not None:
        render_mode = 'rgb_array'

    env = retro.make(game='Zelda-NES', state=save_state, inttype=retro.data.Integrations.CUSTOM_ONLY,
                     render_mode=render_mode)

    if rgb_deque is not None:
        env = FrameCaptureWrapper(env, rgb_render=rgb_deque)

    env = ZeldaGameWrapper(env)
    env = ObjectiveSelector(env, produce_astar=False)
    env = MultiHeadObservationWrapper(env, 128, device)
    env = MultiHeadInputWrapper(env)
    env = MultiHeadCritic(env)

    return env

__all__ = ['make_zelda_env', 'make_multihead_zelda_env']
