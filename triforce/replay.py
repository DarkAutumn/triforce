"""Replay files (docs/rust-actor.md): what an evaluation or chain run played, small enough to keep every play and
exact enough to replay it.

A replay file is JSON lines.  The first line is the header: ``kind`` (``eval`` or ``chain``), ``task`` (what the
pool was built from: the specialist config(s) as JSON, and for chains the route, ``start_from``, ``stop_at``,
``max_steps``, ``stall_steps`` and ``missing``), ``models`` (in policy-index order: ``name``, ``checkpoint`` and
the Rust network's ``hash``), ``backend`` (where the probabilities came from), ``code`` (``git describe --always
--dirty``), ``data`` (a hash of game.yaml and routes.yaml), ``deterministic`` and ``states_dir``.  Every further line
is one played episode: ``job`` (its key: the eval seed or the chain run seed), ``start_state`` (eval),
``actions`` (base64, one byte per head index, three per decision) and ``result`` (the eval record or the run
result).

``replay(header, play)`` rebuilds the task for that one job, feeds the recorded actions to the same models and
returns what the pool reports: every decision's ``log_prob``, ``entropy`` and ``value`` under the policy that acted
(bit for bit the original play's when the code and models match), ``obs`` with ``observations=True``, and the
episode's ``events``.  A replay that runs past its recorded actions has diverged and raises.
"""

import base64
import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from . import _native

BACKEND = 'rust-cpu'
FORMAT = 1


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


def encode_actions(actions) -> str:
    """[n, 3] head indices → base64 of one byte each."""
    return base64.b64encode(np.asarray(actions, dtype=np.uint8).tobytes()).decode('ascii')


def decode_actions(text: str) -> np.ndarray:
    """Inverse of ``encode_actions``: an [n, 3] int64 array."""
    return np.frombuffer(base64.b64decode(text), dtype=np.uint8).reshape(-1, 3).astype(np.int64)


def location_text(location) -> Optional[str]:
    """MapLocation → '2:0x0e' / '0:0x77:cave' (``runner.parse_location`` reads it back)."""
    if location is None:
        return None
    return f"{location.level}:0x{location.value:02x}" + (':cave' if location.in_cave else '')


def models_header(names: Iterable[str], policies: Iterable, checkpoints: dict) -> list:
    """The ``models`` entry: per policy index, its name, checkpoint and Rust network hash."""
    return [{'name': name, 'checkpoint': checkpoints.get(name), 'hash': f'{policy.hash:016x}'}
            for name, policy in zip(names, policies)]


def header(kind: str, task: dict, models: list, deterministic: bool, states_dir=None) -> dict:
    """A replay file's first line."""
    return {'format': FORMAT, 'kind': kind, 'task': task, 'models': models, 'backend': BACKEND,
            'code': code_version(), 'data': data_hash(), 'deterministic': bool(deterministic),
            'states_dir': None if states_dir is None else str(states_dir)}


class ReplayWriter:
    """Writes a replay file: the header, then one line per play as plays finish."""

    def __init__(self, path, head: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, 'w', encoding='utf-8')  # pylint: disable=consider-using-with
        self._file.write(json.dumps(head) + '\n')

    def write(self, play: dict, result: dict, start_state: Optional[str] = None) -> None:
        """One played episode (a ``poll_plays`` dict) and its result."""
        line = {'job': int(play['job']), 'actions': encode_actions(play['actions']), 'result': result}
        if start_state is not None:
            line['start_state'] = start_state
        self._file.write(json.dumps(line) + '\n')
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read(path) -> tuple[dict, list[dict]]:
    """(header, plays) of a replay file; each play's ``actions`` decoded to [n, 3]."""
    with open(path, encoding='utf-8') as f:
        lines = [json.loads(line) for line in f if line.strip()]
    if not lines or lines[0].get('format') != FORMAT:
        raise ValueError(f"{path} is not a replay file (format {FORMAT})")
    return lines[0], [{**line, 'actions': decode_actions(line['actions'])} for line in lines[1:]]


def load_policies(head: dict) -> list:
    """The header's models as Rust networks, checked against the recorded hashes."""
    # pylint: disable=import-outside-toplevel
    from .registry import load_model
    policies = []
    for model in head['models']:
        if model['checkpoint'] is None:
            raise ValueError(f"model {model['name']} was not loaded from a checkpoint; pass policies=")
        network, _ = load_model(model['checkpoint'])
        policy = network.to_native()
        if f'{policy.hash:016x}' != model['hash']:
            raise ValueError(f"{model['checkpoint']} is not the model the play used "
                             f"(hash {policy.hash:016x}, recorded {model['hash']})")
        policies.append(policy)
    return policies


def _task(head: dict, play: dict):
    task, states_dir = head['task'], head['states_dir']
    if head['kind'] == 'eval':
        config = _native.SpecialistConfig.from_json(task['config'])
        return _native.EvalTask(config, [(play['start_state'], int(play['job']))], states_dir=states_dir)
    if head['kind'] == 'chain':
        # pylint: disable=import-outside-toplevel
        from .runner import parse_location
        configs = [_native.SpecialistConfig.from_json(text) for text in task['configs']]
        stop_at = parse_location(task['stop_at']) if task['stop_at'] else None
        return _native.ChainTask(configs, task['route'], [int(play['job'])], stop_at=stop_at,
                                 start_from=task['start_from'], max_steps=task['max_steps'], missing=task['missing'],
                                 states_dir=states_dir, stall_steps=task['stall_steps'])
    raise ValueError(f"unknown replay kind {head['kind']!r}")


def replay(head: dict, play: dict, observations: bool = False, policies: Optional[list] = None) -> dict:
    """Replays one recorded play (see the module doc); ``policies`` (Rust networks in policy-index order) replace
    loading the header's checkpoints."""
    if head['backend'] != BACKEND:
        raise ValueError(f"the play's probabilities came from {head['backend']!r}, this replays on {BACKEND!r}")
    policies = load_policies(head) if policies is None else policies
    job = int(play['job'])
    with _native.EnvPool(_task(head, play), num_envs=1, threads=1) as pool:
        pool.play(policies, record='observations' if observations else 'actions',
                  replay={job: np.asarray(play['actions'])})
        pool.wait_idle()
        plays, events = pool.poll_plays(), pool.poll_events()
    if len(plays) != 1:
        raise RuntimeError(f"replaying job {job} gave {len(plays)} plays")
    if len(plays[0]['actions']) != len(play['actions']):
        raise RuntimeError(f"the replay of job {job} diverged: it ended after {len(plays[0]['actions'])} of "
                           f"{len(play['actions'])} recorded decisions")
    return {**plays[0], 'events': events}
