import os
import retro

from.end_condition import ZeldaEndCondition, ZeldaGameplayEndCondition
from .critic import ZeldaCritic, ZeldaGameplayCritic
from .zeldaml import ZeldaML
from .scenario import ZeldaScenario
from .scenario_gauntlet import GauntletScenario
from .scenario_dungeon import DungeonScenario

# add custom integrations to retro
script_dir = os.path.dirname(os.path.realpath(__file__))
retro.data.Integrations.add_custom_path(os.path.join(script_dir, 'custom_integrations'))

# register all scenarios
ZeldaScenario.register(GauntletScenario())

# register all dungeons (only have the first completed so far)
for i in range(1):
    ZeldaScenario.register(DungeonScenario(i+1))


# define the model surface area
__all__ = [ ZeldaML.__name__, ZeldaScenario.__name__, ZeldaCritic.__name__, ZeldaEndCondition.__name__, ZeldaGameplayCritic.__name__, ZeldaGameplayEndCondition.__name__]
