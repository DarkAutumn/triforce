from typing import Dict

from .zelda_game import get_heart_halves, mode_gameplay, mode_cave
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
        self.leave_early_penalty = -self.reward_maximum
        self.entered_cave_penalty = -self.reward_large
        self.move_perpendicular_penalty = 0.0
        self.equipment_reward = 0.0
        
    def critique_location_discovery(self, old, new, rewards):
        if old['location'] != new['location'] and old['location_objective']:
            if old['location_objective'] == new['location']:
                rewards['reward-new-location'] = self.new_location_reward
            else:
                rewards['penalty-left-early'] = self.leave_early_penalty

        level = new['level']
        location = new['location']

        if old['mode'] == mode_gameplay and location == 0x77 and new['mode'] == mode_cave:
            rewards['penalty-entered-cave'] = self.entered_cave_penalty
            
        elif level == 0:
            if location not in self.allowed_rooms:
                rewards['penalty-left-allowed-area'] = self.left_allowed_area_penalty

            elif old['location'] == 0x77 and location != 0x77 and not new['sword']:
                rewards['penalty-no-sword'] = self.left_without_sword_penalty
                
            else:
                super().critique_location_discovery(old, new, rewards)
            
        elif level == 1:
            # don't forget to reward for reaching level 1 dungeon
            super().critique_location_discovery(old, new, rewards)

    def set_score(self, old : Dict[str, int], new : Dict[str, int]):
        new_location = new['location']
        self.seen.add(new_location)
        new['score'] = new['sword'] + len(self.seen) - 1 + get_heart_halves(new) * 0.5

class OverworldSwordCritic(ZeldaGameplayCritic):
    pass

class OverworldSwordEndCondition(ZeldaEndCondition):
    pass

class Overworld1EndCondition(ZeldaEndCondition):
    def __init__(self):
        super().__init__()
        self.allowed_rooms = overworld_dungeon1_walk_rooms
        self._seen = set()
        self._last_discovery = 0
        self.location_timeout = 1200 # 5 minutes to find a new room

    def clear(self):
        super().clear()
        self._seen.clear()
        self._last_discovery = 0

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

        if not truncated:
            old_location = old['location']
            new_location = new['location']
            if old_location != new_location and new_location not in self._seen:
                self._seen.add(new_location)
                self._last_discovery = 0

            else:
                self._last_discovery += 1

            if self._last_discovery > self.location_timeout:
                reason = "truncated-no-discovery"
                truncated = True

        return terminated, truncated, reason