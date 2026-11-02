"""Start states for specialist episodes (design §7.2).

- Seed states: savestate globs from the specialist config.  The room is verified from RAM after loading
  (filenames are unreliable, docs/route.md); a state outside the specialist's rooms raises ``StartStateError``.
- ``ProbabilisticSelector``: ported from v1.  Round-robin until it has a full window of results covering every
  state, then samples states weighted by recent results: recency decay 0.9, failures ×2, damage taken ×1.5,
  over the last 128 results.
- Harvested states: ``<harvested dir>/*.state`` (written by running the previous specialist), mixed with seed
  states at ``harvested_fraction`` (50/50 by default).
- ``SavestateCapture``: the first time a room is entered from a side, save ``<level>_<room><side>.state``
  (side = n/s/e/w the room was entered from; ``c`` for stairs/cave transitions between levels).
- ``ram_randomization_values``: reset-time hp/bombs/keys randomization.  Position jitter is not applied: a
  RAM-written position can put Link inside walls or a doorway, and nothing yet checks that it is safe.
"""

import glob
import gzip
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence

from .emulator import GAME_DIR
from .specialist import RamRandomization, SpecialistConfig, format_room
from .state_change import StateChange
from .zelda_enums import Direction, MapLocation

DECAY_FACTOR = 0.9
FAILURE_WEIGHT = 2.0
DAMAGE_WEIGHT = 1.5
MEMORY_SIZE = 128
HARVEST_RESCAN_RESETS = 64


class StartStateError(ValueError):
    """A start state that doesn't belong to the specialist (or no start states at all)."""


@dataclass
class StartResult:
    """One finished episode from a start state."""
    state: str
    success: bool
    health_lost: bool


class ProbabilisticSelector:
    """Picks start states, favouring recent failures and damage (v1 ProbabilisticSelector, keyed by state)."""
    def __init__(self, states: Sequence[str], rng: Optional[random.Random] = None,
                 memory_size: int = MEMORY_SIZE):
        if not states:
            raise StartStateError("ProbabilisticSelector needs at least one state")
        self.states = list(states)
        self.rng = rng or random.Random()
        self.memory: Deque[StartResult] = deque(maxlen=memory_size)
        self._next_round_robin = 0

    def next(self) -> str:
        """The next start state."""
        if len(self.memory) < self.memory.maxlen or not self._all_states_seen():
            state = self.states[self._next_round_robin % len(self.states)]
            self._next_round_robin += 1
            return state
        weights = self.weights()
        names = list(weights)
        return self.rng.choices(names, weights=[weights[n] for n in names], k=1)[0]

    def record(self, state: str, success: bool, health_lost: bool):
        """Records an episode result for a state this selector returned."""
        if state in self.states:
            self.memory.append(StartResult(state, success, health_lost))

    def weights(self) -> Dict[str, float]:
        """Unnormalized sampling weight per state from the result memory (newest first decays least)."""
        weights: Dict[str, float] = {}
        for i, result in enumerate(reversed(self.memory)):
            weight = DECAY_FACTOR ** i * (1.0 if result.success else FAILURE_WEIGHT)
            if result.health_lost:
                weight *= DAMAGE_WEIGHT
            weights[result.state] = weights.get(result.state, 0.0) + weight
        return weights

    def _all_states_seen(self) -> bool:
        seen = {r.state for r in self.memory}
        return all(s in seen for s in self.states)


class StartStatePool:
    """Seed + harvested start states for one specialist."""
    def __init__(self, config: SpecialistConfig, rng: Optional[random.Random] = None):
        self.config = config
        self.rng = rng or random.Random()
        seeds = config.resolve_start_states()
        self.seed = ProbabilisticSelector(seeds, self.rng) if seeds else None
        self.harvested: Optional[ProbabilisticSelector] = None
        self._resets = 0
        self._scan_harvested()
        if self.seed is None and self.harvested is None:
            raise StartStateError(f"Specialist {config.name!r} has no start states "
                                  f"(globs {config.start_states}, harvested dir {config.harvested_dir})")

    def _scan_harvested(self):
        paths = sorted(glob.glob(os.path.join(self.config.harvested_dir, '*.state')))
        if not paths:
            return
        if self.harvested is None:
            self.harvested = ProbabilisticSelector(paths, self.rng)
        else:
            self.harvested.states = paths

    def next(self) -> str:
        """Path of the next start state."""
        self._resets += 1
        if self._resets % HARVEST_RESCAN_RESETS == 0:
            self._scan_harvested()
        if self.harvested is not None and (self.seed is None or self.rng.random() < self.config.harvested_fraction):
            return self.harvested.next()
        return self.seed.next()

    def record(self, state: str, success: bool, health_lost: bool):
        """Feeds an episode result back to whichever selector owns the state."""
        for selector in (self.seed, self.harvested):
            if selector is not None:
                selector.record(state, success, health_lost)


def verify_start_room(name: str, location: MapLocation, rooms: Sequence[MapLocation], specialist: str):
    """Raises StartStateError if a loaded start state is outside the specialist's rooms."""
    if location not in rooms:
        raise StartStateError(
            f"Start state {os.path.basename(name)!r} loads into room {format_room(location)}, which is not one of "
            f"{specialist}'s rooms {[format_room(r) for r in rooms]} (filenames can lie; RAM decides)")


def _hearts_encoding(health_half_hearts: int, max_health: int) -> Dict[str, int]:
    """RAM values for Link's hp (see Link.health): full health is (max-1) filled + partial $FF (beams)."""
    containers = (max_health - 1) & 0x0F
    if health_half_hearts >= 2 * max_health:
        return {'hearts_and_containers': (containers << 4) | containers, 'partial_hearts': 0xFF}
    full, half = divmod(health_half_hearts, 2)
    return {'hearts_and_containers': (containers << 4) | full, 'partial_hearts': 0x7F if half else 0}


def ram_randomization_values(spec: RamRandomization, max_health: int, rng: random.Random) -> Dict[str, int]:
    """RAM variable values for one reset.  hp is drawn in half hearts within [min, max] fraction of max hp
    (at least half a heart)."""
    values: Dict[str, int] = {}
    if spec.hp is not None:
        lo, hi = (1.0 if v == 'max' else float(v) for v in spec.hp)
        total = 2 * max_health
        half_hearts = rng.randint(max(1, round(lo * total)), max(1, round(hi * total)))
        values.update(_hearts_encoding(half_hearts, max_health))
    if spec.bombs is not None:
        values['bombs'] = rng.randint(*spec.bombs)
    if spec.keys is not None:
        values['keys'] = rng.randint(*spec.keys)
    return values


_SIDE_LETTER = {Direction.N: 'n', Direction.S: 's', Direction.E: 'e', Direction.W: 'w'}


def entry_side(prev: MapLocation, curr: MapLocation) -> str:
    """n/s/e/w side of curr that Link entered from, or 'c' for stairs/cave transitions."""
    if prev.level != curr.level or prev.in_cave != curr.in_cave or prev.manhattan_distance(curr) != 1:
        return 'c'
    return _SIDE_LETTER.get(curr.get_direction_to(prev), 'c')


class SavestateCapture:
    """Saves the first entry of each (room, side) as ``<level>_<room><side>.state`` (v1 _try_save_state)."""
    def __init__(self, directory: str = GAME_DIR):
        self.directory = directory
        self.saved: List[str] = []

    def filename(self, prev: MapLocation, curr: MapLocation) -> str:
        """Savestate filename for entering curr from prev."""
        return f"{curr.level}_{curr.value:02x}{entry_side(prev, curr)}.state"

    def on_step(self, change: StateChange, emu) -> Optional[str]:
        """Writes a savestate if this step entered a new (room, side); returns its path."""
        if not change.changed_location or change.state.in_cave or change.state.game_over:
            return None
        path = os.path.join(self.directory, self.filename(change.previous.full_location, change.state.full_location))
        if os.path.exists(path):
            return None
        os.makedirs(self.directory, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with gzip.open(tmp, 'wb') as f:
            f.write(emu.save_state())
        os.replace(tmp, path)      # atomic: parallel workers may race on the same room
        self.saved.append(path)
        return path
