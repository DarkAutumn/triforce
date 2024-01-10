import typing
from typing import Dict
from .end_condition import *
from .scenario import ZeldaScenario
from .critic import ZeldaCritic, ZeldaGameplayCritic

class ZeldaDungeonCritic(ZeldaGameplayCritic):
    def __init__(self, verbose=False):
        super().__init__(verbose)

        # reward finding new locations and kills high in dungeons
        self.new_tile_reward = self.reward_small
        self.kill_reward = self.reward_large
        self.health_change_reward = self.reward_large
        self.leave_dungeon_penalty = -self.reward_large
        
        self.tile_sizing_x = 80
        self.tile_sizing_y = 60

        self.clear()

    def clear(self):
        super().clear()

        # do not reward for any new locations in the overworld
        self._visted_locations[0] = [True] * 256
        
        # what tiles have been visited on each individual dungeon room
        self._squares_visited = [set()] * 256

    def critique_gameplay(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int]):
        rewards = super().critique_gameplay(old_state, new_state)

        rewards += self.critique_tile_discovery(old_state, new_state)

        return rewards
    
    def critique_location_discovery(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int]):
        # Override.  Only allow rewards for new locations in dungeons
        reward = 0
        if new_state['level'] != 0:
            reward += super().critique_location_discovery(old_state, new_state)
        else:
            reward += self.leave_dungeon_penalty
            self.report(reward, f"Penalty for leaving the dungeon! {reward}")

        return reward
    
    def critique_tile_discovery(self, old_state : typing.Dict[str, int], new_state : typing.Dict[str, int]):
        """reward reaching new parts of each dungeon room"""

        # only score in dungeons, not if the player made it back to the map
        if new_state['level'] == 0:
            return 0.0
        
        reward = 0.0

        room = new_state['location']
        positions_seen = self._squares_visited[room]

        # only consider tiles of size 8x8 instead of every unique location
        position = (int(new_state['link_x'] / self.tile_sizing_x), int(new_state['link_y'] / self.tile_sizing_y))
        if position not in positions_seen:
            positions_seen.add(position)

            reward += self.new_tile_reward
            self.report(reward, f"Reward for moving to new section of room {room:x} ({position}): {reward}")

        return reward

class DungeonEndCondition(ZeldaEndCondition):
    def __init__(self, verbose=0):
        super().__init__(verbose)

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
                self.print_verbose('End Scenario - Left Dungeon')
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
                self.print_verbose('Truncated - No new rooms discovered in 5 minutes')
                truncated = True
        
        return terminated, truncated

    def mark_visited(self, location):
        self._seen.add(location)

    def is_terminated(self, info):
        level = info['level']
        if level == 0:
            self.print_verbose('End Scenario - Left Dungeon')
            return True
        
        return super().is_terminated(info)
    
    def is_truncated(self, info):
        # todo
        location = info['location']

        if self._last_discovery > self._dicovery_requirement:
            self.print_verbose('Truncated - No new rooms discovered in 5 minutes')
            return True
        
        return super().is_truncated(info)
    
    def clear(self):
        self._free.clear()
        self._last_discovery = 0


class DungeonScenario(ZeldaScenario):
    def __init__(self, dungeon, verbose=False):
        self.dungeon = dungeon
        super().__init__(f'dungeon{dungeon}', f"The Dungeon Scenario - Try to complete dungeon {dungeon}", f'dungeon{dungeon}.state', [ZeldaDungeonCritic], [DungeonEndCondition])

    def __str__(self):
        return f'Dungeon {self.dungeon} Scenario'
    