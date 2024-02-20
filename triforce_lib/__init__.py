"""
The triforce library is a collection of tools for training and running AI models for the NES game The Legend of Zelda.
"""

import os
import json
import retro

from .zelda_orchestrator import ZeldaAIOrchestrator
from .zeldaml import ZeldaML
from .models_and_scenarios import ZeldaAIModel, ZeldaScenario
from .zelda_game import is_in_cave

# add custom integrations to retro
script_dir = os.path.dirname(os.path.realpath(__file__))
retro.data.Integrations.add_custom_path(os.path.join(script_dir, 'custom_integrations'))

# define the model surface area
__all__ = [
    ZeldaML.__name__,
    ZeldaAIModel.__name__,
    ZeldaScenario.__name__,
    ZeldaAIOrchestrator.__name__,
    is_in_cave.__name__,
    ]
