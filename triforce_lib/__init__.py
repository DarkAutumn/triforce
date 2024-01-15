import os
import retro

from .action_replay import ZeldaActionReplay
from.end_condition import ZeldaEndCondition, ZeldaEndCondition
from .critic import ZeldaCritic, ZeldaGameplayCritic
from .zeldaml import ZeldaML
from .scenario import ZeldaScenario

# add custom integrations to retro
script_dir = os.path.dirname(os.path.realpath(__file__))
retro.data.Integrations.add_custom_path(os.path.join(script_dir, 'custom_integrations'))

# define the model surface area
__all__ = [
    ZeldaML.__name__,
    ZeldaScenario.__name__,
    ZeldaCritic.__name__,
    ZeldaEndCondition.__name__,
    ZeldaGameplayCritic.__name__,
    ZeldaEndCondition.__name__,
    ZeldaActionReplay.__name__
    ]
