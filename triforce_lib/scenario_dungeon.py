import typing
from typing import Dict

from .zelda_game import get_heart_halves
from .end_condition import *
from .critic import ZeldaGameplayCritic

class ZeldaDungeonCritic(ZeldaGameplayCritic):
    def __init__(self, reporter=None):
        super().__init__(reporter)

        # reward finding new locations and kills high in dungeons
        self.kill_reward = self.reward_large
        self.health_change_reward = self.reward_large
        self.leave_dungeon_penalty = -self.reward_large

    def critique_location_discovery(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int]):
        # Override.  Only allow rewards for new locations in dungeons
        reward = 0
        if new_state['level'] != 0:
            reward += super().critique_location_discovery(old_state, new_state)
        else:
            reward += self.leave_dungeon_penalty
            self.report(reward, f"Penalty for leaving the dungeon! {reward}", "penalty-leave-dungeon")

        return reward

class DungeonEndCondition(ZeldaEndCondition):
    def __init__(self, reporter):
        super().__init__(reporter)

        self._seen = set()

        self._last_discovery = 0        
        self.location_timeout = 1200 # 5 minutes to find a new room

    def clear(self):
        super().clear()
        self._seen.clear()
        self._last_discovery = 0

    def is_scenario_ended(self, old: Dict[str, int], new: Dict[str, int]) -> (bool, bool):
        terminated, truncated = super().is_scenario_ended(old, new)

        level = new['level']
        if not terminated:
            if level == 0:
                self.report("terminated-left-dungeon", "Left dungeon")
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
                self.report("truncated-no-discovery", "Truncated - No new rooms discovered in 5 minutes")
                truncated = True
        
        return terminated, truncated

    def clear(self):
        self._last_discovery = 0
