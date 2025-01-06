"""Model selection based on the current game state."""

from typing import List
from .models_and_scenarios import ZeldaModelDefinition, ZELDA_MODELS

class ModelSelector:
    """Selects the best model for the current game state."""
    def __init__(self):
        self.models_by_priority = sorted((model for model in ZELDA_MODELS.values()),
                                         key=lambda x: x.priority, reverse=True)

    def find_acceptable_models(self, state) -> List[ZeldaModelDefinition]:
        """Selects a model based on the current game state."""
        acceptable_models = [model for model in self.models_by_priority if self.__is_model_acceptable(model, info)]
        return acceptable_models or self.models_by_priority

    def __is_model_acceptable(self, model : ZeldaModelDefinition, state):
        location = info['location']
        level = info['level']

        matches_level = level in model.levels
        matches_room = model.rooms is None or location in model.rooms
        matches_enemy_requirements = not model.requires_enemies or info['enemies']
        matches_equipment = self.__matches_equipment(model, info)

        return matches_level and matches_room and matches_enemy_requirements and matches_equipment

    def __matches_equipment(self, model : ZeldaModelDefinition, state):
        if model.requires_triforce is not None:
            if get_num_triforce_pieces(info) < model.requires_triforce:
                return False

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
