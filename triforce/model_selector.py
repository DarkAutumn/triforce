"""Model selection based on the current game state."""

from typing import List

from .models_and_scenarios import ZeldaModelDefinition, ZELDA_MODELS

class ModelSelector:
    """Selects the best model for the current game state."""
    def __init__(self):
        self.models_by_priority = sorted((model for model in ZELDA_MODELS.values()),
                                         key=lambda x: x.priority, reverse=True)

    def find_acceptable_models(self) -> List[ZeldaModelDefinition]:
        """Selects a model based on the current game state."""
        return self.models_by_priority
