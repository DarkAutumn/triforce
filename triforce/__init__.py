"""
The triforce library is a collection of tools for training and running AI models for the NES game The Legend of Zelda.
"""

import os
import retro

from .zelda_env import make_zelda_env
from .model_definition import ZeldaModelDefinition, ZeldaScenario, ZELDA_MODELS, TRAINING_SCENARIOS
from .simulate_critic import simulate_critique
from .models import Network, get_neural_network, register_neural_network

# add custom integrations to retro
script_dir = os.path.dirname(os.path.realpath(__file__))
retro.data.Integrations.add_custom_path(os.path.join(script_dir, 'custom_integrations'))

LOADED_MODELS = {}

# define the model surface area
__all__ = [
    make_zelda_env.__name__,
    ZeldaModelDefinition.__name__,
    ZeldaScenario.__name__,
    simulate_critique.__name__,
    Network.__name__,
    get_neural_network.__name__,
    register_neural_network.__name__,
    'ZELDA_MODELS',
    'TRAINING_SCENARIOS'
    ]
