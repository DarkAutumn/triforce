from typing import Dict

from .zelda_game import get_heart_halves, is_in_cave, mode_gameplay, mode_cave
from .end_conditions import ZeldaEndCondition
from .critic import ZeldaGameplayCritic

overworld_dungeon1_walk_rooms = set([0x78, 0x67, 0x68, 0x58, 0x48, 0x38, 0x37])

class Overworld1Critic(ZeldaGameplayCritic):
    def clear(self):
        super().clear()
        self.seen = set()
        self.allowed_rooms = overworld_dungeon1_walk_rooms

        self.left_allowed_area_penalty = -self.reward_large
        self.left_without_sword_penalty = -self.reward_large
        self.leave_early_penalty = -self.reward_maximum
        self.entered_cave_penalty = -self.reward_large
        self.equipment_reward = None
        
    def critique_location_discovery(self, old, new, rewards):
        if old['location'] != new['location'] and old['location_objective']:
            if old['location_objective'] != new['location']:
                rewards['penalty-left-early'] = self.leave_early_penalty
                return

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
        new['score'] = new['sword'] + len(self.seen) - 1

class OverworldSwordCritic(ZeldaGameplayCritic):
    def __init__(self):
        super().__init__()
        self.entered_cave = False

        self.entered_cave_reward = self.reward_large
        self.left_cave_penalty = -self.reward_large

    def clear(self):
        self.entered_cave = False

    def critique_location_discovery(self, old, new, rewards):
        if not self.entered_cave and is_in_cave(new):
            self.entered_cave = True
            rewards['reward-entered-cave'] = self.entered_cave_reward

        if is_in_cave(old) and not is_in_cave(new) and not new['sword']:
            rewards['penalty-left-cave'] = self.left_cave_penalty

    
    def set_score(self, old : Dict[str, int], new : Dict[str, int]):
        score = 0
        if self.entered_cave:
            score += 1

        if new['sword']:
            score += 1

        if new['sword'] and not is_in_cave(new):
            score += 1

        new['score'] = score