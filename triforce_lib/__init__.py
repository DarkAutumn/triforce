import os
import retro

from .zelda_orchestrator import ZeldaAIOrchestrator
from .action_replay import ZeldaActionReplay
from .zeldaml import ZeldaML
from .scenario import ZeldaScenario
from .models import ZeldaModel

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
    ZeldaActionReplay.__name__,
    ZeldaAIOrchestrator.__name__,
    ]
