"""End conditions for specialist episodes and their deny-by-default classification (design §3.3, §6).

Each condition looks at a ``StateChange`` and returns an ending name or None.  Ending names:

| ending | default class | meaning |
|---|---|---|
| ``exit-correct`` | success if listed in the specialist's ``success`` | left the family (``one_room``: the room) through an exit the objective wanted |
| ``enter-room:<level>:<room>`` | success if listed | entered that room (e.g. ``enter-room:2:0x0e``) |
| ``triforce-collected`` | success if listed | picked up a triforce piece |
| ``boss-dead-and-heart`` | success if listed | collected the heart container (it only appears once the boss is dead) |
| ``exit-wrong`` | soft failure | left a room through an exit the objective did not want |
| ``death`` / ``wallmastered`` | hard failure | game over / dragged back to the dungeon entrance |
| ``stuck`` / ``no-progress`` | hard failure | Link didn't move for 50 steps / no new correct room for 2000 steps |
| ``max-steps`` | soft failure (truncation) | the env hit ``max_steps`` |

Success kinds are only successes for specialists that list them; any ending that is neither a listed success
nor in the specialist's/central failure table raises ``UnclassifiedEndingError``.
"""

from typing import Dict, Iterable, List, Optional

from .rewards import HARD_FAILURE, OUTCOMES, SOFT_FAILURE, SUCCESS
from .state_change import StateChange
from .zelda_enums import MapLocation, ZeldaEnemyKind

EXIT_CORRECT = 'exit-correct'
EXIT_WRONG = 'exit-wrong'
TRIFORCE_COLLECTED = 'triforce-collected'
BOSS_DEAD_AND_HEART = 'boss-dead-and-heart'
DEATH = 'death'
WALLMASTERED = 'wallmastered'
STUCK = 'stuck'
NO_PROGRESS = 'no-progress'
MAX_STEPS = 'max-steps'
ENTER_ROOM_PREFIX = 'enter-room:'

SUCCESS_KINDS = (EXIT_CORRECT, TRIFORCE_COLLECTED, BOSS_DEAD_AND_HEART)

# Central failure table.  Successes are never listed here: they come from each specialist's `success` list.
DEFAULT_FAILURES: Dict[str, str] = {
    EXIT_WRONG: SOFT_FAILURE,
    MAX_STEPS: SOFT_FAILURE,
    DEATH: HARD_FAILURE,
    WALLMASTERED: HARD_FAILURE,
    STUCK: HARD_FAILURE,
    NO_PROGRESS: HARD_FAILURE,
}


class UnclassifiedEndingError(RuntimeError):
    """An episode ended for a reason the specialist doesn't classify (design §6: deny-by-default)."""


def parse_enter_room(kind: str) -> MapLocation:
    """'enter-room:2:0x0e' (optionally ':cave') -> MapLocation."""
    parts = kind[len(ENTER_ROOM_PREFIX):].split(':')
    if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] != 'cave'):
        raise ValueError(f"Bad success kind {kind!r}; expected enter-room:<level>:<room>[:cave]")
    return MapLocation(int(parts[0]), int(parts[1], 16), len(parts) == 3)


def enter_room_name(location: MapLocation) -> str:
    """Canonical ending name for entering location."""
    suffix = ':cave' if location.in_cave else ''
    return f"{ENTER_ROOM_PREFIX}{location.level}:0x{location.value:02x}{suffix}"


def validate_success_kind(kind: str) -> str:
    """Raises ValueError for an unknown success kind; returns the canonical name."""
    if kind.startswith(ENTER_ROOM_PREFIX):
        return enter_room_name(parse_enter_room(kind))
    if kind not in SUCCESS_KINDS:
        raise ValueError(f"Unknown success kind {kind!r}; expected one of {SUCCESS_KINDS} or enter-room:<l>:<r>")
    return kind


def classify_ending(ending: str, success: Iterable[str], endings: Optional[Dict[str, str]] = None) -> str:
    """success / hard-failure / soft-failure for an ending; raises UnclassifiedEndingError if it isn't listed."""
    if ending in success:
        return SUCCESS
    if endings and ending in endings:
        outcome = endings[ending]
        assert outcome in OUTCOMES, f"Bad outcome {outcome!r} for ending {ending!r}"
        return outcome
    if ending in DEFAULT_FAILURES:
        return DEFAULT_FAILURES[ending]
    raise UnclassifiedEndingError(
        f"Ending {ending!r} is not classified (success={sorted(success)}, overrides={endings or {}}); "
        "list it in the specialist's success/endings")


class ZeldaEndCondition:
    """Base class: ``check`` returns an ending name or None."""
    def clear(self):
        """Called at reset."""

    def check(self, state_change: StateChange) -> Optional[str]:
        """The ending name if the episode ended this step."""
        raise NotImplementedError


class GameOver(ZeldaEndCondition):
    """Link died."""
    def check(self, state_change):
        return DEATH if state_change.state.game_over else None


class Wallmastered(ZeldaEndCondition):
    """A Wallmaster dragged Link out of the room (a jump of more than one room)."""
    def check(self, state_change):
        prev, curr = state_change.previous, state_change.state
        if prev.full_location != curr.full_location and not state_change.gained_triforce \
                and ZeldaEnemyKind.Wallmaster in prev.enemies \
                and prev.full_location.manhattan_distance(curr.full_location) > 1:
            return WALLMASTERED
        return None


class GainedTriforce(ZeldaEndCondition):
    """Picked up a triforce piece."""
    def check(self, state_change):
        return TRIFORCE_COLLECTED if state_change.gained_triforce else None


class BossDeadAndHeart(ZeldaEndCondition):
    """Collected a heart container: it appears only after the boss dies."""
    def check(self, state_change):
        prev, curr = state_change.previous.link, state_change.state.link
        return BOSS_DEAD_AND_HEART if curr.max_health > prev.max_health else None


class RoomExit(ZeldaEndCondition):
    """Room transitions.  Inside the family a correct exit continues the episode (unless ``one_room``: then every
    correct exit ends it); leaving the family through a correct exit is ``exit-correct``; any exit the objective
    didn't want is ``exit-wrong``.  Entering a room named by an ``enter-room`` success kind ends the episode with
    that name, but only when the objective wanted it (0x37 after Dungeon 1 wants E, not the D1 stairs again)."""
    def __init__(self, rooms: Iterable[MapLocation], enter_rooms: Iterable[MapLocation] = (), one_room: bool = False):
        self.rooms = frozenset(rooms)
        self.enter_rooms = frozenset(enter_rooms)
        self.one_room = one_room

    def check(self, state_change):
        prev, curr = state_change.previous, state_change.state
        if prev.full_location == curr.full_location:
            return None

        location = curr.full_location
        correct = location in prev.objectives.next_rooms
        if not correct:
            return EXIT_WRONG
        if location in self.enter_rooms:
            return enter_room_name(location)
        if location in self.rooms and not self.one_room:
            return None
        return EXIT_CORRECT


class Timeout(ZeldaEndCondition):
    """Link is stuck in one position, or fails to reach a new correct room (v1 Timeout)."""
    def __init__(self, position_timeout: int = 50, no_progress_timeout: int = 2000):
        self.position_timeout = position_timeout
        self.no_progress_timeout = no_progress_timeout
        self._position_duration = 0
        self._last_progress = 0
        self._entered_rooms = set()

    def clear(self):
        self._position_duration = 0
        self._last_progress = 0
        self._entered_rooms = set()

    def check(self, state_change):
        prev, curr = state_change.previous, state_change.state

        if prev.link.position == curr.link.position and not state_change.hits:
            self._position_duration += 1
            if self._position_duration >= self.position_timeout:
                return STUCK
        else:
            self._position_duration = 0

        if prev.full_location == curr.full_location:
            self._last_progress += 1
        elif curr.full_location in prev.objectives.next_rooms and curr.full_location not in self._entered_rooms:
            self._entered_rooms.add(curr.full_location)
            self._last_progress = 0

        if self._last_progress > self.no_progress_timeout:
            return NO_PROGRESS

        return None


def build_end_conditions(rooms: Iterable[MapLocation], success: Iterable[str],
                         one_room: bool = False) -> List[ZeldaEndCondition]:
    """The end conditions for a specialist, in priority order.  Triforce / heart-container conditions are only
    checked for specialists that list them as success (otherwise those events are just rewards).  ``one_room``
    ends every episode at its first room exit (``SpecialistConfig.one_room``)."""
    success = list(success)
    enter_rooms = [parse_enter_room(k) for k in success if k.startswith(ENTER_ROOM_PREFIX)]
    conditions: List[ZeldaEndCondition] = [GameOver(), Wallmastered()]
    if TRIFORCE_COLLECTED in success:
        conditions.append(GainedTriforce())
    if BOSS_DEAD_AND_HEART in success:
        conditions.append(BossDeadAndHeart())
    conditions += [RoomExit(rooms, enter_rooms, one_room), Timeout()]
    return conditions
