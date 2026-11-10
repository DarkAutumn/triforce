"""Specialist task configs: ``specialists/<name>.yaml`` (design §3.3).

Parsing and validation live in the native module (``env::config``); this module keeps the Python-facing names.
``SpecialistConfig.rooms`` is a list of ``(level, room, in_cave)`` tuples; nested blocks (``reward``, ``items``,
``ram_randomization``, ``curriculum``, ``obs``) read as attribute namespaces; ``replace(**fields)`` re-validates.
"""

from typing import List

from ._native import ConfigError, SpecialistConfig

__all__ = ['ConfigError', 'SpecialistConfig', 'list_specialists', 'load_specialist']


def load_specialist(name: str) -> SpecialistConfig:
    """Loads specialists/<name>.yaml (or a path to a yaml file); raises ConfigError (a ValueError)."""
    return SpecialistConfig.load(str(name))


def list_specialists() -> List[str]:
    """Names of every specialists/*.yaml."""
    return list(SpecialistConfig.names())
