"""All end conditions for training."""

from typing import Dict
from .zelda_game import get_num_triforce_pieces, is_mode_death

class ZeldaEndCondition:
    """
    Base class for all end conditions.  End conditions are used to determine if a scenario was terminated or
    truncated, and why.
    """

    def clear(self):
        """Called to clear the state of the end condition.  Called at the start of each scenario."""

    def is_scenario_ended(self, old : Dict[str, int], new : Dict[str, int]) -> tuple[bool, bool, str]:
        """Called to determine if the scenario has ended, returns (terminated, truncated, reason) or None"""

class GameOver(ZeldaEndCondition):
    """Whether the game mode is death."""
    def is_scenario_ended(self, old : Dict[str, int], new : Dict[str, int]) -> tuple[bool, bool, str]:
        if is_mode_death(new['mode']):
            return True, False, "failure-terminated-death"

        return False, False, None

class Timeout(ZeldaEndCondition):
    """End the scenario if the agent is in the same screen position, or fails to discover a new room."""
    def __init__(self):
        super().__init__()
        self.__position_duration = 0
        self.__last_discovery = 0
        self.__seen = set()

        # the number of timesteps the agent can be in the same position before we truncate
        self.position_timeout = 50
        self.no_discovery_timeout = 1200

    def clear(self):
        self.__position_duration = 0
        self.__last_discovery = 0
        self.__seen.clear()

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

        return False, False, None

class GainedTriforce(ZeldaEndCondition):
    """End the scenario if the agent gains a piece of triforce."""
    def is_scenario_ended(self, old : Dict[str, int], new : Dict[str, int]) -> tuple[bool, bool, str]:
        if new['triforce'] != 0 and old['triforce'] != new['triforce']:
            return True, False, "success-gained-triforce"

        return False, False, None

class AllowedArea(ZeldaEndCondition):
    """End the scenario if the agent leaves the defined play area for this scenario."""
    def is_scenario_ended(self, old : Dict[str, int], new : Dict[str, int]) -> tuple[bool, bool, str]:
        objective = old['objective']
        old_walk = objective.walk

        if old_walk != new['objective'].walk:
            location = new['level'], new['location']
            if location == old_walk[-1]:
                return True, False, "success-completed-scenario"

            return True, False, "failure-left-play-area"

        return False, False, None
