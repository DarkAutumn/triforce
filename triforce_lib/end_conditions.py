from typing import Dict
from .zelda_game import is_mode_death

class ZeldaEndCondition:
    def clear(self):
        """Called to clear the state of the end condition.  Called at the start of each scenario."""
        pass

    def is_scenario_ended(self, old : Dict[str, int], new : Dict[str, int]) -> tuple[bool, bool, str]:
        """Called to determine if the scenario has ended, returns (terminated, truncated, reason) or None"""
        pass

class GameOver(ZeldaEndCondition):
    def is_scenario_ended(self, old : Dict[str, int], new : Dict[str, int]) -> tuple[bool, bool, str]:
        if is_mode_death(new['mode']):
            return True, False, "failure-terminated-death"

class Timeout(ZeldaEndCondition):
    def __init__(self):
        super().__init__()
        self.__seen = set()

        # the number of timesteps the agent can be in the same position before we truncate
        self.position_timeout = 50
        self.no_discovery_timeout = 1200

    def clear(self):
        self.__position_duration = 0
        self.__seen.clear()
        self.__last_discovery = 0

    def is_scenario_ended(self, old : Dict[str, int], new : Dict[str, int]) -> tuple[bool, bool, str]:
        # Check if link is stuck in one position on the screen
        if self.position_timeout:
            if old['link_pos'] == new['link_pos'] and not new['step_hits']:
                if self.__position_duration >= self.position_timeout:
                    return False, True, "failure-stuck"

                self.__position_duration += 1
            else:
                self.__position_duration = 0

        # Check if link never found a new room
        if self.no_discovery_timeout:
            location = new['location']
            if old['location'] != location and location not in self.__seen:
                self.__seen.add(location)
                self.__last_discovery = 0

            else:
                self.__last_discovery += 1

            if self.__last_discovery > self.no_discovery_timeout:
                return False, True, "failure-no-discovery"

class GainedTriforce(ZeldaEndCondition):
    def __init__(self):
        super().__init__()

    def is_scenario_ended(self, old : Dict[str, int], new : Dict[str, int]) -> tuple[bool, bool, str]:
        if new['triforce'] != 0 and old['triforce'] != new['triforce']:
            return True, False, "success-gained-triforce"

class LeftDungeon(ZeldaEndCondition):
    def is_scenario_ended(self, old : Dict[str, int], new : Dict[str, int]) -> tuple[bool, bool, str]:
        if new['level'] == 0:
            return True, False, "failure-left-dungeon"

class EnteredDungeon(ZeldaEndCondition):
    def is_scenario_ended(self, old : Dict[str, int], new : Dict[str, int]) -> tuple[bool, bool, str]:
        if new['level'] == 1:
            return True, False, "success-entered-dungeon"

class LeftOverworld1Area(ZeldaEndCondition):
    overworld_dungeon1_walk_rooms = set([0x77, 0x78, 0x67, 0x68, 0x58, 0x48, 0x38, 0x37])

    def is_scenario_ended(self, old : Dict[str, int], new : Dict[str, int]) -> tuple[bool, bool, str]:
        if new['level'] == 0 and new['location'] not in self.overworld_dungeon1_walk_rooms:
            return True, False, "failure-left-play-area"

class StartingRoomConditions(ZeldaEndCondition):
    def is_scenario_ended(self, old : Dict[str, int], new : Dict[str, int]) -> tuple[bool, bool, str]:
        if new['location'] != 0x77:
            if new['sword']:
                return True, False, "success-found-sword"
            else:
                return True, False, "failure-no-sword"

class DefeatedBoss(ZeldaEndCondition):
    def is_scenario_ended(self, old : Dict[str, int], new : Dict[str, int]) -> tuple[bool, bool, str]:
        if not new['enemies']:
            return True, False, "success-killed-boss"

        if new['location'] != 0x35:
            return True, False, "failure-left-boss-room"
