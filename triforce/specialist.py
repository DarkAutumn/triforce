"""Specialist task configs: ``specialists/<name>.yaml`` (design §3.3).

```yaml
name: stalfos-keese-gel
rooms: ['(1, 0x73)', '(0, 0x77, cave)']   # (level, room[, cave]); must be tagged `specialist: <name>` in game.yaml
start_states: ['1_73*', '1_72e']          # savestate globs (integration dir); rooms are verified from RAM at load
success: [exit-correct]                   # exit-correct | boss-dead-and-heart | triforce-collected | enter-room:<l>:<r>
endings: {}                               # optional ending -> outcome overrides of end_conditions.DEFAULT_FAILURES
reward_profile: default
reward: {pbrs: on}
obs: {objective: on}
ram_randomization: {hp: [0.5, max], bombs: [0, 8], keys: [0, 1]}
fresh_rooms: true                         # pre-entry savestates: clear the room's kill history (full spawn)
items: {bombs: off}                       # mask out items even when Link has them (bombs/boomerang/arrows)
curriculum: {enabled: off}               # backward curriculum over trajectories/<name> (triforce.explore)
one_room: off                             # on: every episode ends at its first room exit (else when it leaves rooms)
max_steps: 2000
warm_start: null
```
"""

import glob
import os
import re
from typing import Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .emulator import GAME_DIR
from .end_conditions import validate_success_kind
from .rewards import OUTCOMES
from .zelda_enums import ActionKind, MapLocation

SPECIALISTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'specialists')
HARVESTED_DIR = os.path.join(GAME_DIR, 'harvested')

_ROOM_RE = re.compile(r'^\(?\s*(\d+)\s*,\s*(0x[0-9a-fA-F]+|\d+)\s*(?:,\s*(cave))?\s*\)?$')

HpBound = Union[float, Literal['max']]


def parse_room(text: Union[str, list, tuple]) -> MapLocation:
    """'(1, 0x73)', '(0, 0x77, cave)' or [1, '0x73'] -> MapLocation."""
    if isinstance(text, (list, tuple)):
        text = ', '.join(str(x) for x in text)
    match = _ROOM_RE.match(str(text).strip())
    if not match:
        raise ValueError(f"Bad room {text!r}; expected '(level, 0xRR)' or '(level, 0xRR, cave)'")
    level, room, cave = match.groups()
    return MapLocation(int(level), int(room, 0), cave is not None)


def format_room(location: MapLocation) -> str:
    """MapLocation -> '(1, 0x73)' / '(0, 0x77, cave)'."""
    cave = ', cave' if location.in_cave else ''
    return f"({location.level}, 0x{location.value:02x}{cave})"


class RewardConfig(BaseModel):
    """reward: block."""
    model_config = ConfigDict(extra='forbid')
    pbrs: bool = True


class ObsConfig(BaseModel):
    """obs: block."""
    model_config = ConfigDict(extra='forbid')
    objective: bool = True


class CurriculumConfig(BaseModel):
    """Backward curriculum over Go-Explore trajectories (``curriculum.py``); off unless enabled.  ``trajectories``
    defaults to ``trajectories/<name>`` at the repo root (``python -m triforce.explore --specialist <name>``)."""
    model_config = ConfigDict(extra='forbid')
    enabled: bool = False
    trajectories: Optional[str] = None
    fraction: float = 0.5
    threshold: float = 0.6


class ItemsConfig(BaseModel):
    """Which inventory items the specialist may use (off = masked out even when Link has the item)."""
    model_config = ConfigDict(extra='forbid')
    bombs: bool = True
    boomerang: bool = True
    arrows: bool = True

    def disabled_kinds(self) -> frozenset:
        """ActionKinds masked out for this specialist."""
        pairs = ((self.bombs, ActionKind.BOMB), (self.boomerang, ActionKind.BOOMERANG), (self.arrows, ActionKind.ARROW))
        return frozenset(kind for enabled, kind in pairs if not enabled)


class RamRandomization(BaseModel):
    """Reset-time RAM randomization.  hp bounds are fractions of max hp (or 'max'); bombs/keys are inclusive
    integer ranges.  Omitted fields leave the savestate's value.

    ``full_hp`` (with ``hp``) is the share of resets that start at full health, the only state where the sword fires
    beams; the rest draw hp below full (no beams) within the ``hp`` bounds.  Without it, hp is uniform over the bounds
    (full health only as often as any other value)."""
    model_config = ConfigDict(extra='forbid')
    hp: Optional[Tuple[HpBound, HpBound]] = None
    full_hp: Optional[float] = None
    bombs: Optional[Tuple[int, int]] = None
    keys: Optional[Tuple[int, int]] = None

    @field_validator('full_hp')
    @classmethod
    def _check_full_hp(cls, value):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"full_hp must be a probability in [0, 1], got {value}")
        return value

    @field_validator('hp')
    @classmethod
    def _check_hp(cls, value):
        if value is not None:
            lo, hi = (1.0 if v == 'max' else float(v) for v in value)
            if not 0.0 < lo <= hi <= 1.0:
                raise ValueError(f"hp bounds must satisfy 0 < min <= max <= 1 (or 'max'), got {value}")
        return value

    @field_validator('bombs', 'keys')
    @classmethod
    def _check_range(cls, value):
        if value is not None and not 0 <= value[0] <= value[1] <= 255:
            raise ValueError(f"range must satisfy 0 <= lo <= hi <= 255, got {value}")
        return value


class SpecialistConfig(BaseModel):
    """One specialist task (design §3.3)."""
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)

    name: str
    rooms: List[MapLocation]
    start_states: List[str] = Field(default_factory=list)
    success: List[str]
    endings: Dict[str, str] = Field(default_factory=dict)
    reward_profile: str = 'default'
    reward: RewardConfig = Field(default_factory=RewardConfig)
    obs: ObsConfig = Field(default_factory=ObsConfig)
    ram_randomization: RamRandomization = Field(default_factory=RamRandomization)
    fresh_rooms: bool = True                     # pre-entry start states spawn the room's full monster count
    items: ItemsConfig = Field(default_factory=ItemsConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    one_room: bool = False                       # episodes end at the first correct room exit, even inside `rooms`
    max_steps: int = 2000
    warm_start: Optional[str] = None
    harvested_states: Optional[str] = None       # directory; default <integration dir>/harvested/<name>
    harvested_fraction: float = 0.5              # share of resets drawn from harvested states when any exist
    trainable: bool = True
    notes: Optional[str] = None

    @field_validator('rooms', mode='before')
    @classmethod
    def _parse_rooms(cls, value):
        rooms = [x if isinstance(x, MapLocation) else parse_room(x) for x in value]
        if not rooms:
            raise ValueError("a specialist needs at least one room")
        return rooms

    @field_validator('success', mode='before')
    @classmethod
    def _parse_success(cls, value):
        value = [value] if isinstance(value, str) else list(value)
        if not value:
            raise ValueError("a specialist needs at least one success kind")
        return [validate_success_kind(v) for v in value]

    @field_validator('endings')
    @classmethod
    def _check_endings(cls, value):
        for ending, outcome in value.items():
            if outcome not in OUTCOMES:
                raise ValueError(f"ending {ending!r}: outcome must be one of {OUTCOMES}, got {outcome!r}")
        return value

    @field_validator('max_steps')
    @classmethod
    def _check_max_steps(cls, value):
        if value <= 0:
            raise ValueError("max_steps must be positive")
        return value

    @property
    def harvested_dir(self) -> str:
        """Directory of harvested start states for this specialist."""
        return self.harvested_states or os.path.join(HARVESTED_DIR, self.name)

    def resolve_start_states(self) -> List[str]:
        """Absolute paths of seed savestates matching start_states globs (sorted, de-duplicated)."""
        found = set()
        for pattern in self.start_states:
            path = pattern if os.path.isabs(pattern) else os.path.join(GAME_DIR, pattern)
            if not path.endswith('.state'):
                path += '.state'
            found.update(glob.glob(path))
        return sorted(found)


def specialist_path(name: str) -> str:
    """specialists/<name>.yaml."""
    return os.path.join(SPECIALISTS_DIR, f"{name}.yaml")


def load_specialist(name: str) -> SpecialistConfig:
    """Loads specialists/<name>.yaml (or a path to a yaml file)."""
    path = name if name.endswith('.yaml') and os.path.exists(name) else specialist_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No specialist config {path}; known: {list_specialists()}")
    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    config = SpecialistConfig(**data)
    if not name.endswith('.yaml') and config.name != name:
        raise ValueError(f"{path} declares name {config.name!r}")
    return config


def list_specialists() -> List[str]:
    """Names of every specialists/*.yaml."""
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(SPECIALISTS_DIR, '*.yaml')))
