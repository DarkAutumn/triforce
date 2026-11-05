"""Specialist task configs: ``specialists/<name>.yaml`` (design §3.3).

Parsing and validation live in the native module (``env::config``); this module keeps the Python-facing names.
``SpecialistConfig.rooms`` is a list of ``(level, room, in_cave)`` tuples; nested blocks (``reward``, ``items``,
``ram_randomization``, ``curriculum``, ``obs``) read as attribute namespaces; ``replace(**fields)`` re-validates.
"""

import re
from pathlib import Path
from typing import List, Union

from ._native import ConfigError, SpecialistConfig
from .paths import HARVESTED_DIR, SPECIALISTS_DIR
from .zelda_enums import MapLocation

__all__ = ['ConfigError', 'SpecialistConfig', 'format_room', 'harvested_dir', 'list_specialists', 'load_specialist',
           'parse_room', 'room_locations', 'specialist_path']

_ROOM_RE = re.compile(r'^\(?\s*(\d+)\s*,\s*(0x[0-9a-fA-F]+|\d+)\s*(?:,\s*(cave))?\s*\)?$')


def parse_room(text: Union[str, list, tuple]) -> MapLocation:
    """'(1, 0x73)', '(0, 0x77, cave)' or [1, '0x73'] -> MapLocation."""
    if isinstance(text, (list, tuple)):
        text = ', '.join(str(x) for x in text)
    match = _ROOM_RE.match(text.strip())
    if not match:
        raise ValueError(f"bad room {text!r}: expected '(level, 0xROOM)' or '(level, 0xROOM, cave)'")
    level, room, cave = match.groups()
    return MapLocation(int(level), int(room, 0), cave is not None)


def format_room(location: MapLocation) -> str:
    """MapLocation -> '(1, 0x73)' / '(0, 0x77, cave)'."""
    cave = ', cave' if location.in_cave else ''
    return f"({location.level}, 0x{location.value:02x}{cave})"


def room_locations(config: SpecialistConfig) -> List[MapLocation]:
    """The config's rooms as MapLocations."""
    return [MapLocation(level, room, in_cave) for level, room, in_cave in config.rooms]


def harvested_dir(config: SpecialistConfig) -> Path:
    """Directory of harvested start states for this specialist."""
    return Path(config.harvested_states) if config.harvested_states else HARVESTED_DIR / config.name


def specialist_path(name: str) -> Path:
    """specialists/<name>.yaml."""
    return SPECIALISTS_DIR / f"{name}.yaml"


def load_specialist(name: str) -> SpecialistConfig:
    """Loads specialists/<name>.yaml (or a path to a yaml file); raises ConfigError (a ValueError)."""
    return SpecialistConfig.load(str(name))


def list_specialists() -> List[str]:
    """Names of every specialists/*.yaml."""
    return list(SpecialistConfig.names())
