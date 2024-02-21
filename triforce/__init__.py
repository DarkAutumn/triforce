"""
The triforce library is a collection of tools for training and running AI models for the NES game The Legend of Zelda.
"""

import os
import json
import retro

from .model_selector import ModelSelector
from .zelda_env_factory import ZeldaEnvFactory
from .models_and_scenarios import ZeldaAIModel, ZeldaScenario
from .zelda_game import is_in_cave
from .simulate_critic import simulate_critique

# add custom integrations to retro
script_dir = os.path.dirname(os.path.realpath(__file__))
retro.data.Integrations.add_custom_path(os.path.join(script_dir, 'custom_integrations'))

# define the model surface area
__all__ = [
    ZeldaEnvFactory.__name__,
    ZeldaAIModel.__name__,
    ZeldaScenario.__name__,
    ModelSelector.__name__,
    simulate_critique.__name__
    ]
