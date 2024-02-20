"""Model selection based on the current game state."""

from typing import Sequence
from .models_and_scenarios import ZeldaAIModel
from .zelda_game import has_beams

class ModelSelector:
    """Selects the best model for the current game state."""
    def __init__(self, loaded_models : Sequence[ZeldaAIModel]):
        # we only select usable models
        self.models_by_priority = [x for x in loaded_models if x.available_models]
        self.models_by_priority.sort(key=lambda x: x.priority, reverse=True)

    def select_model(self, info) -> ZeldaAIModel:
        """Selects a model based on the current game state."""
        acceptable_models = [model for model in self.models_by_priority if self.__is_model_acceptable(model, info)]
        return acceptable_models or self.models_by_priority

    def __is_model_acceptable(self, model : ZeldaAIModel, info):
        location = info['location']
        level = info['level']

        matches_level = level in model.levels
        matches_room = model.rooms is None or location in model.rooms
        matches_enemy_requirements = not model.requires_enemies or info['enemies']
        matches_equipment = self.__matches_equipment(model, info)

        return matches_level and matches_room and matches_enemy_requirements and matches_equipment

    def __matches_equipment(self, model : ZeldaAIModel, info):
        for equipment in model.equipment_required:
            if equipment == "beams":
                if not has_beams(info):
                    return False

            elif equipment == "bombs":
                if info['bombs'] == 0:
                    return False

            elif equipment == "nosword":
                if info['sword']:
                    return False

            else:
                raise NotImplementedError("Unknown equipment requirement: " + equipment)

        return True
