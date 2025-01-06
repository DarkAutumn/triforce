"""Model selection based on the current game state."""

from typing import List

from triforce.link import Link
from triforce.zelda_enums import SwordKind
from triforce.zelda_game import ZeldaGame
from .models_and_scenarios import ZeldaModelDefinition, ZELDA_MODELS

class ModelSelector:
    """Selects the best model for the current game state."""
    def __init__(self):
        self.models_by_priority = sorted((model for model in ZELDA_MODELS.values()),
                                         key=lambda x: x.priority, reverse=True)

    def find_acceptable_models(self, state : ZeldaGame) -> List[ZeldaModelDefinition]:
        """Selects a model based on the current game state."""
        acceptable_models = [model for model in self.models_by_priority if self.__is_model_acceptable(model, state)]
        return acceptable_models or self.models_by_priority

    def __is_model_acceptable(self, model : ZeldaModelDefinition, state : ZeldaGame) -> bool:
        matches_level = state.level in model.levels
        matches_room = model.rooms is None or state.location in model.rooms
        matches_enemy_requirements = not model.requires_enemies or state.enemies
        matches_equipment = self.__matches_equipment(model, state.link)

        return matches_level and matches_room and matches_enemy_requirements and matches_equipment

    def __matches_equipment(self, model : ZeldaModelDefinition, link : Link):
        if model.requires_triforce is not None:
            if link.triforce_pieces < model.requires_triforce:
                return False

        for equipment in model.equipment_required:
            if equipment == "beams":
                if not link.has_beams:
                    return False

            elif equipment == "bombs":
                if link.bombs == 0:
                    return False

            elif equipment == "nosword":
                if link.sword != SwordKind.NONE:
                    return False

            else:
                raise NotImplementedError("Unknown equipment requirement: " + equipment)

        return True
