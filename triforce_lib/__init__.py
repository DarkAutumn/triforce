import os
import retro

from .action_replay import ZeldaActionReplay
from .end_condition import ZeldaEndCondition, ZeldaEndCondition
from .critic import ZeldaCritic, ZeldaGameplayCritic
from .zeldaml import ZeldaML
from .scenario import ZeldaScenario
from .models import ZeldaModel
from .pygame_display import pygame_render

# add custom integrations to retro
script_dir = os.path.dirname(os.path.realpath(__file__))
retro.data.Integrations.add_custom_path(os.path.join(script_dir, 'custom_integrations'))

# parse json
import json
with open(os.path.join(script_dir, 'triforce.json')) as f:
    all_settings = json.load(f)
    ZeldaScenario.initialize(all_settings['scenarios'])
    ZeldaModel.initialize(all_settings['models'])

# define the model surface area
__all__ = [
    ZeldaML.__name__,
    ZeldaScenario.__name__,
    ZeldaCritic.__name__,
    ZeldaEndCondition.__name__,
    ZeldaGameplayCritic.__name__,
    ZeldaEndCondition.__name__,
    ZeldaActionReplay.__name__,
    pygame_render.__name__,
    ]
