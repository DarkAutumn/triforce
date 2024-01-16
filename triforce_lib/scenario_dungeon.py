import typing
from typing import Dict

from .zelda_game import get_heart_halves
from .end_condition import *
from .critic import ZeldaGameplayCritic

class ZeldaDungeonCritic(ZeldaGameplayCritic):
    def __init__(self):
        super().__init__()

        # reward finding new locations and kills high in dungeons
        self.kill_reward = self.reward_large
        self.health_change_reward = self.reward_large
        self.leave_dungeon_penalty = -self.reward_large

    def critique_location_discovery(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int], rewards : typing.Dict[str, float]):
        # Override.  Only allow rewards for new locations in dungeons
        if new_state['level'] != 0:
            super().critique_location_discovery(old_state, new_state, rewards)
        else:
            if old_state['location'] != new_state['location']:
                rewards['leave-room-penalty'] = self.leave_dungeon_penalty

class DungeonEndCondition(ZeldaEndCondition):
    def __init__(self):
        super().__init__()

        self._seen = set()

        self._last_discovery = 0        
        self.location_timeout = 1200 # 5 minutes to find a new room

    def clear(self):
        super().clear()
        self._seen.clear()
        self._last_discovery = 0

    def is_scenario_ended(self, old: Dict[str, int], new: Dict[str, int]) -> (bool, bool):
        terminated, truncated, reason = super().is_scenario_ended(old, new)

        level = new['level']
        if not terminated:
            if level == 0:
                reason = "terminated-left-dungeon"
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

    def clear(self):
        self._last_discovery = 0
