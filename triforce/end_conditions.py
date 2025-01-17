"""All end conditions for training."""

from .state_change_wrapper import StateChange
from .zelda_enums import SwordKind

class ZeldaEndCondition:
    """
    Base class for all end conditions.  End conditions are used to determine if a scenario was terminated or
    truncated, and why.
    """

    def clear(self):
        """Called to clear the state of the end condition.  Called at the start of each scenario."""

    def is_scenario_ended(self, state_change : StateChange) -> tuple[bool, bool, str]:
        """Called to determine if the scenario has ended, returns (terminated, truncated, reason) or None"""

class GameOver(ZeldaEndCondition):
    """Whether the game mode is death."""
    def is_scenario_ended(self, state_change : StateChange) -> tuple[bool, bool, str]:
        if state_change.state.game_over:
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

    def is_scenario_ended(self, state_change : StateChange) -> tuple[bool, bool, str]:
        # Check if link is stuck in one position on the screen
        prev, curr = state_change.previous, state_change.state

        if self.position_timeout:
            if prev.link.position == curr.link.position and not state_change.hits:
                if self.__position_duration >= self.position_timeout:
                    return False, True, "failure-stuck"

                self.__position_duration += 1
            else:
                self.__position_duration = 0

        # Check if link never found a new room
        if self.no_discovery_timeout:
            if prev.location != curr.location and curr.location not in self.__seen:
                self.__seen.add(curr.location)
                self.__last_discovery = 0

            else:
                self.__last_discovery += 1

            if self.__last_discovery > self.no_discovery_timeout:
                return False, True, "failure-no-discovery"

        return False, False, None

class StartingSwordCondition(ZeldaEndCondition):
    """End conditions for 'pick up the sword' scenario."""
    def is_scenario_ended(self, state_change : StateChange) -> tuple[bool, bool, str]:
        state = state_change.state
        if state.location != 0x77 and state.link.sword == SwordKind.NONE:
            return True, False, "failure-no-sword"

        return False, False, None

class GainedTriforce(ZeldaEndCondition):
    """End the scenario if the agent gains a piece of triforce."""
    def is_scenario_ended(self, state_change : StateChange) -> tuple[bool, bool, str]:
        prev_link, curr_link = state_change.previous.link, state_change.state.link
        if prev_link.triforce_pieces < curr_link.triforce_pieces \
                or prev_link.triforce_of_power < curr_link.triforce_of_power:
            return True, False, "success-gained-triforce"

        return False, False, None

class LeftDungeon(ZeldaEndCondition):
    """End the scenario if the agent leaves the dungeon."""
    def is_scenario_ended(self, state_change : StateChange) -> tuple[bool, bool, str]:
        if state_change.state.level == 0:
            return True, False, "failure-left-dungeon"

        return False, False, None

class EnteredDungeon(ZeldaEndCondition):
    """End the scenario if the agent enters the dungeon."""
    def is_scenario_ended(self, state_change : StateChange) -> tuple[bool, bool, str]:
        state = state_change.state
        if state.level == 0:
            return False, False, None

        if state.level == state.link.triforce_pieces + 1:
            return True, False, "success-entered-dungeon"

        return False, True, "truncated-entered-dungeon"

class LeftOverworld1Area(ZeldaEndCondition):
    """End the scenario if the agent leaves the allowable areas between the start room and dungeon 1."""
    overworld_dungeon1_walk_rooms = set([0x78, 0x67, 0x68, 0x58, 0x48, 0x38, 0x37])

    def is_scenario_ended(self, state_change : StateChange) -> tuple[bool, bool, str]:
        state = state_change.state
        if state.level == 0 and state.location not in self.overworld_dungeon1_walk_rooms:
            return True, False, "failure-left-play-area"

        return False, False, None

class StartingRoomConditions(ZeldaEndCondition):
    """End conditions for 'pick up the sword' scenario."""
    def is_scenario_ended(self, state_change : StateChange) -> tuple[bool, bool, str]:
        state = state_change.state
        if state.location != 0x77:
            if state.link.sword != SwordKind.NONE:
                return True, False, "success-found-sword"

            return True, False, "failure-no-sword"

        return False, False, None

class DefeatedBoss(ZeldaEndCondition):
    """End condition for killing the boss."""
    def is_scenario_ended(self, state_change : StateChange) -> tuple[bool, bool, str]:
        state = state_change.state
        if not state.enemies:
            return True, False, "success-killed-boss"

        if state.location != 0x35:
            return True, False, "failure-left-boss-room"

        return False, False, None

class LeftRoom(ZeldaEndCondition):
    """End condition for leaving the current room."""
    def is_scenario_ended(self, state_change : StateChange) -> tuple[bool, bool, str]:
        prev, curr = state_change.previous, state_change.state

        if prev.location != curr.location:
            if curr.location in prev.objectives.locations:
                return False, True, "truncated-left-room"

            return True, False, "failure-left-room"

        return False, False, None
