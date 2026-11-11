"""Specialist PPO agents that play The Legend of Zelda (NES).  ``__version__`` is the repo's ``VERSION`` file
(MAJOR.MINOR.COMPAT.REV, AGENTS.md "Versioning"); a model is compatible when its first three numbers match."""

from pathlib import Path


def _read_version() -> str:
    try:
        return (Path(__file__).resolve().parent.parent / 'VERSION').read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        from importlib.metadata import version  # pylint: disable=import-outside-toplevel
        return version('triforce')


__version__ = _read_version()
