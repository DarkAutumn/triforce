from .models import ZeldaModel
from .zelda_game import has_beams

class ZeldaAIOrchestrator:
    def __init__(self):
        # we only select usable models
        self.models_by_priority = [x for x in ZeldaModel.get_all() if x.model]
        self.models_by_priority.sort(key=lambda x: x.priority, reverse=True)
    
    @property
    def has_any_model(self):
        return len(self.models_by_priority) > 0

    def select_model(self, info):
        acceptable_models = [model for model in self.models_by_priority if self.is_model_acceptable(model, info)]
        return acceptable_models if acceptable_models else self.models_by_priority

    def is_model_acceptable(self, model, info):
        location = info['location']
        level = info['level']

        matches_level = level in model.levels
        matches_room = model.rooms is None or location in model.rooms
        matches_enemy_requirements = not model.requires_enemies or info['objects'].enemy_count
        matches_equipment = self.matches_equipment(model, info)

        return matches_level and matches_room and matches_enemy_requirements and matches_equipment

    def matches_equipment(self, model, info):
        for equipment in model.equipment_required:
            if equipment == "beams":
                if not has_beams(info):
                    return False
                
            elif equipment == "bombs":
                if info['bombs'] == 0:
                    return False
            else:
                raise Exception("Unknown equipment requirement: " + equipment)
            
        return True
    