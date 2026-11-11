"""What a result was produced by: the code version, the game data, the models and the start state.  Stored with
chain results (``Runner.provenance``), evaluations (``eval.json``) and replays (``replay.meta``)."""

import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from . import _native
from .registry import model_version


@lru_cache(maxsize=1)
def code_version() -> str:
    """``git describe --always --dirty`` of the checkout, or 'unknown'."""
    root = _native.paths()['root']
    try:
        out = subprocess.run(['git', '-C', str(root), 'describe', '--always', '--dirty'], capture_output=True,
                             text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
    return out.stdout.strip() or 'unknown'


@lru_cache(maxsize=1)
def data_hash() -> str:
    """The game and route data the run played under (game.yaml, routes.yaml)."""
    digest = hashlib.sha256()
    for name in ('game.yaml', 'routes.yaml'):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()[:16]


def location_text(location) -> Optional[str]:
    """MapLocation → '2:0x0e' / '0:0x77:cave' (``runner.parse_location`` reads it back)."""
    if location is None:
        return None
    return f"{location.level}:0x{location.value:02x}" + (':cave' if location.in_cave else '')


def models_header(names: Iterable[str], policies: Iterable, checkpoints: dict, deterministic: Iterable[bool]) -> list:
    """The ``models`` entry: per policy index, its name, checkpoint, Rust network hash, whether it acted by argmax
    and the triforce version its run dir is stamped with (None without a checkpoint or a stamp)."""
    return [{'name': name, 'checkpoint': checkpoints.get(name), 'hash': f'{policy.hash:016x}',
             'deterministic': bool(argmax),
             'version': None if checkpoints.get(name) is None else model_version(checkpoints[name])}
            for name, policy, argmax in zip(names, policies, deterministic)]


def state_hash(name: str, states_dir=None) -> Optional[str]:
    """Hash of the start state file ``name`` resolves to (as the native store resolves it: an absolute path as is,
    else ``<states_dir>/<name>.state``), None when there is no such file."""
    path = Path(name)
    if not path.is_absolute():
        base = Path(states_dir) if states_dir is not None else Path(_native.paths()['states_dir'])
        path = base / (name if name.endswith('.state') else f'{name}.state')
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None
