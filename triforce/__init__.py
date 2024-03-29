"""
The triforce library is a collection of tools for training and running AI models for the NES game The Legend of Zelda.
"""

import os
import json
import retro

from .model_selector import ModelSelector
from .zelda_env import make_zelda_env
from .models_and_scenarios import ZeldaModelDefinition, ZeldaScenario, ZELDA_MODELS, TRAINING_SCENARIOS
from .machine_learning import ZeldaAI
from .simulate_critic import simulate_critique

# add custom integrations to retro
script_dir = os.path.dirname(os.path.realpath(__file__))
retro.data.Integrations.add_custom_path(os.path.join(script_dir, 'custom_integrations'))

LOADED_MODELS = {}

def load_model_version(model_def : ZeldaModelDefinition, version):
    """Loads the specified version of the model."""
    assert version in model_def.available_models, f"Model kind {version} is not available"

    if model_def.name not in LOADED_MODELS:
        LOADED_MODELS[model_def.name] = {}

    if version not in LOADED_MODELS[model_def.name]:
        ml_model = ZeldaAI(model_def)
        LOADED_MODELS[model_def.name][version] = ml_model

    return LOADED_MODELS[model_def.name][version]

# define the model surface area
__all__ = [
    make_zelda_env.__name__,
    load_model_version.__name__,
    ZeldaAI.__name__,
    ZeldaModelDefinition.__name__,
    ZeldaScenario.__name__,
    ModelSelector.__name__,
    simulate_critique.__name__,
    'ZELDA_MODELS',
    'TRAINING_SCENARIOS'
    ]
