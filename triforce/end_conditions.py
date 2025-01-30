"""All end conditions for training."""

from .objectives import ObjectiveKind, ObjectiveSelector
from .state_change_wrapper import StateChange
from .zelda_enums import SwordKind, ZeldaEnemyKind

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
        self.__last_correct_room = 0
        self.__seen = set()
        self.__tile_timeout = {}

        # the number of timesteps the agent can be in the same position before we truncate
        self.position_timeout = 50
        self.no_discovery_timeout = 1200
        self.no_next_room_timeout = 300

    def clear(self):
        self.__position_duration = 0
        self.__last_discovery = 0
        self.__last_correct_room = 0
        self.__seen.clear()
        self.__tile_timeout.clear()

    def is_scenario_ended(self, state_change : StateChange) -> tuple[bool, bool, str]:
        # pylint: disable=too-many-branches

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
            if prev.full_location == curr.full_location:
                self.__last_discovery += 1
                self.__last_correct_room += 1
            else:
                if curr.full_location not in self.__seen:
                    self.__seen.add(curr.location)
                    self.__last_discovery = 0

                if curr.full_location in prev.objectives.next_rooms:
                    self.__last_correct_room = 0

            if self.__last_discovery > self.no_discovery_timeout:
                return False, True, "failure-no-discovery"

            if self.__last_correct_room > self.no_next_room_timeout:
                return False, True, "failure-no-next-room"


        if prev.full_location != curr.full_location or state_change.hits:
            self.__tile_timeout.clear()
        else:
            count = self.__tile_timeout.get(curr.link.tile, 0) + 1
            self.__tile_timeout[curr.link.tile] = count

            if count > 30:
                return False, True, "failure-stuck"

        return False, False, None

class NextRoomTimeout(ZeldaEndCondition):
    """End the scenario if the agent is in the same screen position, or fails to discover a new room."""
    def __init__(self):
        super().__init__()
        self.__position_duration = 0
        self.__last_discovery = 0

        # the number of timesteps the agent can be in the same position before we truncate
        self.position_timeout = 50
        self.no_discovery_timeout = 1200

    def clear(self):
        self.__position_duration = 0
        self.__last_discovery = 0

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
            if prev.location != curr.location and curr.full_location in prev.objectives.next_rooms:
                self.__last_discovery = 0

            else:
                self.__last_discovery += 1

            if self.__last_discovery > self.no_discovery_timeout:
                return False, True, "failure-no-next-room"

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

        if any(x.id == ZeldaEnemyKind.WallMaster for x in state_change.previous.enemies) \
                and state_change.previous.full_location.manhattan_distance(state_change.state.full_location) > 1:
            return True, False, "failure-wallmastered"

        return False, False, None

class LeftWallmasterRoom(ZeldaEndCondition):
    """End the scenario if the agent leaves the wallmaster room in the wrong direction."""
    def is_scenario_ended(self, state_change):
        if state_change.state.location == 0x44:
            return True, False, "failure-left-wallmaster-room"

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
    overworld_dungeon1_walk_rooms = set([0x77, 0x78, 0x67, 0x68, 0x58, 0x48, 0x38, 0x37])

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

class LeftRoute(ZeldaEndCondition):
    """End condition for leaving the current room."""
    def is_scenario_ended(self, state_change : StateChange) -> tuple[bool, bool, str]:
        prev = state_change.previous
        state = state_change.state
        if prev.full_location != state.full_location:
            objectives : ObjectiveSelector = state_change.previous.objectives
            if objectives.kind == ObjectiveKind.MOVE and state.full_location not in objectives.next_rooms:
                return True, False, "failure-left-route"

        return False, False, None

class RoomWalkCondition(ZeldaEndCondition):
    """End condition for leaving the initial room walk scenario."""
    def __init__(self):
        super().__init__()
        self.same_room_timout = 750
        self._current_timeout = 0

    def clear(self):
        self._current_timeout = 0

    def is_scenario_ended(self, state_change : StateChange) -> tuple[bool, bool, str]:
        prev = state_change.previous
        state = state_change.state

        if state.game_over:
            return True, False, "failure-terminated-death"

        if prev.full_location != state.full_location:
            if state.full_location in prev.objectives.next_rooms:
                return True, False, "success-exit"

            return True, False, "failure-wrong-exit"

        self._current_timeout += 1
        if self._current_timeout >= self.same_room_timout:
            return True, False, "failure-stuck"

        return False, False, None

class NowhereToGoCondition(ZeldaEndCondition):
    """End condition for when the objective selector said we have to move to a room but we have nowhere
    to go.  This is often because locked or barred doors are in the way."""
    def is_scenario_ended(self, state_change):
        objectives = state_change.state.objectives
        if objectives.kind == ObjectiveKind.MOVE and not objectives.next_rooms:
            return True, False, "failure-nowhere-to-go"

        return False, False, None

class LeftPlayArea(ZeldaEndCondition):
    """End condition for leaving the initial room walk scenario."""
    def is_scenario_ended(self, state_change):
        location = state_change.state.full_location
        if location.level != 1 or location.value == 0x73:
            return True, False, "failure-left-play-area"

        return False, False, None

class Dungeon1DidntGetKey(ZeldaEndCondition):
    """End condition for leaving the initial room walk scenario."""
    def is_scenario_ended(self, state_change):
        if state_change.state.location == 0x63 and state_change.state.link.keys == 0:
            return True, False, "failure-no-key"

        return False, False, None
