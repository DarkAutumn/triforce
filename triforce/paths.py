"""Repository paths shared by the Python side (the native module resolves the same ones)."""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent
STATES_DIR = Path(os.environ.get('TRIFORCE_STATES_DIR', PACKAGE_DIR / 'states'))
HARVESTED_DIR = STATES_DIR / 'harvested'
TRAJECTORIES_DIR = ROOT / 'trajectories'
SPECIALISTS_DIR = ROOT / 'specialists'
CORPUS_MOVIES_DIR = ROOT / 'corpus' / 'movies'
BASE_MOVIE = CORPUS_MOVIES_DIR / 'power_on_to_start.tfmovie'


def state_path(name: str) -> Path:
    """``STATES_DIR/<name>.state`` (``name`` may already end in ``.state`` or be a path)."""
    path = Path(name)
    if path.suffix == '.state' and (path.is_absolute() or path.parent != Path('.')):
        return path
    return STATES_DIR / (name if name.endswith('.state') else f'{name}.state')
