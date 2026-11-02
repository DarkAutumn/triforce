"""The v2 action schema: factored (kind × direction × duration) timed options, plus masking.

Every specialist shares this schema.  A model emits three integers ``(kind, direction, duration)``.
Direction and duration only apply to some kinds; for the others the index is ignored and the mask
row is all False (the model must treat that head as not applicable).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .zelda_enums import ActionKind, Direction

ACTION_SCHEMA_VERSION = 1

KINDS = tuple(ActionKind)
DIRECTIONS = (Direction.N, Direction.S, Direction.W, Direction.E,
              Direction.NW, Direction.NE, Direction.SW, Direction.SE)
CARDINALS = DIRECTIONS[:4]
DURATIONS = (4, 8, 16, 32)

NUM_KINDS = len(KINDS)
NUM_DIRECTIONS = len(DIRECTIONS)
NUM_DURATIONS = len(DURATIONS)

DIRECTION_INDEX = {d: i for i, d in enumerate(DIRECTIONS)}
DURATION_INDEX = {d: i for i, d in enumerate(DURATIONS)}

# Which heads apply to each kind.
TIMED_KINDS = frozenset({ActionKind.MOVE, ActionKind.WAIT})
DIRECTIONAL_KINDS = frozenset(KINDS) - {ActionKind.WAIT}
DIAGONAL_KINDS = frozenset({ActionKind.BOOMERANG})

# Flat mask layout: [kind (K) | direction (K×D) | duration (K×U)].
MASK_SIZE = NUM_KINDS + NUM_KINDS * NUM_DIRECTIONS + NUM_KINDS * NUM_DURATIONS


@dataclass(frozen=True)
class Action:
    """A timed option chosen by the agent."""
    kind: ActionKind
    direction: Direction = Direction.NONE
    duration: int = 0

    def __post_init__(self):
        if self.kind in DIRECTIONAL_KINDS:
            assert self.direction in DIRECTION_INDEX, f"{self.kind.name} requires a direction"
            assert self.kind in DIAGONAL_KINDS or self.direction in CARDINALS, \
                f"{self.kind.name} does not support diagonal {self.direction.name}"
        if self.kind in TIMED_KINDS:
            assert self.duration in DURATION_INDEX, f"{self.kind.name} requires a duration in {DURATIONS}"

    @staticmethod
    def from_indices(kind: int, direction: int, duration: int) -> 'Action':
        """Builds an action from head outputs, ignoring heads that don't apply to the kind."""
        kind = ActionKind(int(kind))
        dir_value = DIRECTIONS[int(direction)] if kind in DIRECTIONAL_KINDS else Direction.NONE
        dur_value = DURATIONS[int(duration)] if kind in TIMED_KINDS else 0
        return Action(kind, dir_value, dur_value)

    def to_indices(self) -> np.ndarray:
        """(kind, direction, duration) head indices; non-applicable heads are 0."""
        return np.array([int(self.kind),
                         DIRECTION_INDEX.get(self.direction, 0) if self.kind in DIRECTIONAL_KINDS else 0,
                         DURATION_INDEX.get(self.duration, 0) if self.kind in TIMED_KINDS else 0],
                        dtype=np.int64)

    def __str__(self):
        parts = [self.kind.name]
        if self.kind in DIRECTIONAL_KINDS:
            parts.append(self.direction.name)
        if self.kind in TIMED_KINDS:
            parts.append(str(self.duration))
        return ' '.join(parts)

    @staticmethod
    def parse(text: str) -> 'Action':
        """Parses 'MOVE N 16', 'sword e', 'wait 8', 'boomerang nw'."""
        parts = text.upper().split()
        kind = ActionKind[parts[0]]
        rest = parts[1:]
        direction, duration = Direction.NONE, 0
        if kind in DIRECTIONAL_KINDS:
            direction = Direction[rest.pop(0)]
        if kind in TIMED_KINDS:
            duration = int(rest.pop(0)) if rest else 16
        if rest:
            raise ValueError(f"Unexpected tokens in action '{text}': {rest}")
        return Action(kind, direction, duration)


class ActionMask:
    """Legal actions as three boolean arrays: kind[K], direction[K, D], duration[K, U]."""
    def __init__(self):
        self.kind = np.zeros(NUM_KINDS, dtype=bool)
        self.direction = np.zeros((NUM_KINDS, NUM_DIRECTIONS), dtype=bool)
        self.duration = np.zeros((NUM_KINDS, NUM_DURATIONS), dtype=bool)

    def allow(self, kind: ActionKind, direction: Optional[Direction] = None):
        """Allows kind (with direction, if it is directional).  Timed kinds get every duration."""
        self.kind[kind] = True
        if direction is not None:
            self.direction[kind, DIRECTION_INDEX[direction]] = True
        if kind in TIMED_KINDS:
            self.duration[kind, :] = True

    def disallow_kind(self, kind: ActionKind):
        """Removes a kind entirely."""
        self.kind[kind] = False
        self.direction[kind, :] = False
        self.duration[kind, :] = False

    def is_allowed(self, action: Action) -> bool:
        """Whether a concrete action is legal under this mask."""
        if not self.kind[action.kind]:
            return False
        if action.kind in DIRECTIONAL_KINDS and not self.direction[action.kind, DIRECTION_INDEX[action.direction]]:
            return False
        if action.kind in TIMED_KINDS and not self.duration[action.kind, DURATION_INDEX[action.duration]]:
            return False
        return True

    def directions(self, kind: ActionKind):
        """Allowed directions for kind."""
        return [d for i, d in enumerate(DIRECTIONS) if self.direction[kind, i]]

    def to_flat(self) -> np.ndarray:
        """Flat bool[MASK_SIZE] for observation/rollout storage."""
        return np.concatenate([self.kind, self.direction.ravel(), self.duration.ravel()])

    @staticmethod
    def from_flat(flat: np.ndarray) -> 'ActionMask':
        """Inverse of to_flat."""
        mask = ActionMask()
        k, d = NUM_KINDS, NUM_KINDS * NUM_DIRECTIONS
        mask.kind = flat[:k].astype(bool)
        mask.direction = flat[k:k + d].reshape(NUM_KINDS, NUM_DIRECTIONS).astype(bool)
        mask.duration = flat[k + d:].reshape(NUM_KINDS, NUM_DURATIONS).astype(bool)
        return mask

    def __str__(self):
        parts = []
        for kind in KINDS:
            if not self.kind[kind]:
                continue
            text = kind.name
            if kind in DIRECTIONAL_KINDS:
                text += '[' + ','.join(d.name for d in self.directions(kind)) + ']'
            parts.append(text)
        return ' '.join(parts)


# Directions where pressing a d-pad button with a weapon near a dungeon wall would start a room transition.
_EDGE_TILE_MIN = 0x03
_EDGE_TILE_MAX_X = 0x1c
_EDGE_TILE_MAX_Y = 0x12


def compute_action_mask(state) -> ActionMask:
    """Legal actions for a ZeldaGame state.  Ports v1's masking with the v2 changes (design §5.3):

    - MOVE into a not-yet-pushed push block is legal (pushing means holding against it).
    - BOMB is offered whenever bombs > 0 and enemies or a bombable wall are present; Link does not
      need to be lined up with the wall.
    - Empty-mask fallback: unmask all MOVE directions.
    """
    # pylint: disable=too-many-branches
    link = state.link
    mask = ActionMask()
    available = link.get_available_actions()

    for direction in CARDINALS:
        if state.can_link_move(direction) or state.is_push_block_direction(direction):
            mask.allow(ActionKind.MOVE, direction)

    mask.allow(ActionKind.WAIT)

    has_enemies = bool(state.active_enemies)
    has_items = bool(state.items)
    has_bomb_walls = bool(state.bomb_walls)

    # Items with no enemies: only kinds that can pick items up.  Nothing to hit: no weapons at all.
    item_pickup = {ActionKind.SWORD, ActionKind.BEAMS, ActionKind.BOOMERANG, ActionKind.ARROW}
    sword_dirs = link.get_sword_directions_allowed()
    item_dirs = set(link.get_item_directions_allowed())

    # Near dungeon walls, a directional weapon press can trigger a room transition (v1 _update_mask).
    blocked = set()
    if state.level != 0:
        if link.tile.x <= _EDGE_TILE_MIN or link.tile.x >= _EDGE_TILE_MAX_X:
            blocked |= {Direction.N, Direction.S}
        if link.tile.y <= _EDGE_TILE_MIN or link.tile.y >= _EDGE_TILE_MAX_Y:
            blocked |= {Direction.W, Direction.E}

    for kind in (ActionKind.SWORD, ActionKind.BEAMS, ActionKind.BOOMERANG, ActionKind.ARROW, ActionKind.BOMB):
        if kind not in available:
            continue

        if kind == ActionKind.BOMB:
            if not (has_enemies or has_bomb_walls):
                continue
        elif not has_enemies and not (has_items and kind in item_pickup):
            continue

        dirs = sword_dirs if kind in (ActionKind.SWORD, ActionKind.BEAMS) else item_dirs
        for direction in CARDINALS:
            if direction in dirs and direction not in blocked:
                mask.allow(kind, direction)

        if kind in DIAGONAL_KINDS:
            for diag, (a, b) in {Direction.NW: (Direction.N, Direction.W), Direction.NE: (Direction.N, Direction.E),
                                 Direction.SW: (Direction.S, Direction.W),
                                 Direction.SE: (Direction.S, Direction.E)}.items():
                if a in item_dirs and b in item_dirs and a not in blocked and b not in blocked:
                    mask.allow(kind, diag)

    if not mask.direction[ActionKind.MOVE].any():
        # Sword knockback can leave Link where the walkability model thinks he is stuck.
        for direction in CARDINALS:
            mask.allow(ActionKind.MOVE, direction)

    return mask
