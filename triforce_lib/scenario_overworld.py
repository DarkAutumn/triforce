from typing import Dict

from .zelda_game import get_heart_halves
from .end_condition import ZeldaEndCondition
from .critic import ZeldaGameplayCritic

overworld_dungeon1_walk_rooms = set([0x77, 0x78, 0x67, 0x68, 0x58, 0x48, 0x38, 0x37])

class Overworld1Critic(ZeldaGameplayCritic):
    def clear(self):
        super().clear()
        self.seen = set()
        self.allowed_rooms = overworld_dungeon1_walk_rooms

        self.left_allowed_area_penalty = -self.reward_large
        self.left_without_sword_penalty = -self.reward_large

    def critique_location_discovery(self, old, new, rewards):
        level = new['level']
        location = new['location']

        if level == 0:
            if location not in self.allowed_rooms:
                rewards['penalty-left-allowed-area'] = self.left_allowed_area_penalty

            elif old['location'] == 0x77 and location != 0x77 and not new['sword']:
                rewards['penalty-no-sword'] = self.left_without_sword_penalty
                
            else:
                return super().critique_location_discovery(old, new, rewards)
            
        elif level == 1:
            # don't forget to reward for reaching level 1 dungeon
            return super().critique_location_discovery(old, new, rewards)

    def set_score(self, old : Dict[str, int], new : Dict[str, int]):
        new_location = new['location']
        self.seen.add(new_location)
        new['score'] = new['sword'] + len(self.seen) - 1 + get_heart_halves(new) * 0.5

class Overworld1EndCondition(ZeldaEndCondition):
    def __init__(self):
        super().__init__()
        self.allowed_rooms = overworld_dungeon1_walk_rooms

    def is_scenario_ended(self, old: Dict[str, int], new: Dict[str, int]) -> (bool, bool, str):
        terminated, truncated, reason = super().is_scenario_ended(old, new)

        if not terminated and not truncated:
            location = new['location']

            if new['level'] == 1:
                reason = "reached-dungeon1"
                terminated = True

            elif location not in self.allowed_rooms:
                reason = "left-scenario"
                terminated = True

            elif new['sword'] == 0 and location != 0x77:
                reason = "no-sword"
                terminated = True

        return terminated, truncated, reason