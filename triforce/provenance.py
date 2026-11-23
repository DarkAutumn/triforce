"""What a result was produced by: the code version, the game data, the models and the start state.  Stored with
chain results (``Runner.provenance``), evaluations (``eval.json``) and replays (``replay.meta``)."""

import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from . import _native
from .registry import CHECKPOINT_NAME, SNAPSHOTS_NAME, checkpoint_sha1, model_version


def _git(*args) -> str:
    """``git <args>`` in the checkout, stripped, or 'unknown'."""
    root = _native.paths()['root']
    try:
        out = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
    return out.stdout.strip() or 'unknown'


@lru_cache(maxsize=1)
def code_version() -> str:
    """``git describe --always --dirty`` of the checkout, or 'unknown'."""
    return _git('describe', '--always', '--dirty')


@lru_cache(maxsize=1)
def code_commit() -> str:
    """The checkout's commit (``git rev-parse HEAD``), or 'unknown'."""
    return _git('rev-parse', 'HEAD')


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


def checkpoint_label(checkpoint) -> str:
    """A checkpoint (a ``ckpt.pt`` or its run dir) as provenance records it: ``<specialist>/<run>/ckpt.pt``, a
    numbered checkpoint (``registry.snapshot_dir``) ``<specialist>/<run>/checkpoints/it<N>/ckpt.pt``; no local
    directory (the ``checkpoint_sha1`` / network ``hash`` next to it identify the weights)."""
    path = Path(checkpoint)
    if path.is_dir():
        path = path / CHECKPOINT_NAME
    numbered = len(path.parts) >= 5 and path.parts[-3] == SNAPSHOTS_NAME
    return '/'.join(path.parts[-5:] if numbered else path.parts[-3:])


def models_header(names: Iterable[str], policies: Iterable, checkpoints: dict, acts: Iterable[str]) -> list:
    """The ``models`` entry of a result's provenance (``Runner.provenance``): per policy index, its name, checkpoint
    (``checkpoint_label``), Rust network hash, the act mode it played by (``act``, canonical spelling) and the
    triforce version its run dir is stamped with (None without a checkpoint or a stamp).  Replays describe their
    networks with ``replay_models`` instead (no path)."""
    return [{'name': name, 'checkpoint': None if checkpoints.get(name) is None else checkpoint_label(checkpoints[name]),
             'hash': f'{policy.hash:016x}', 'act': act,
             'version': None if checkpoints.get(name) is None else model_version(checkpoints[name])}
            for name, policy, act in zip(names, policies, acts)]


# The outputs a replay's DECISIONS keep of a network, as META ``models[i].outputs`` describes them.
DISTRIBUTION_OUTPUT = {
    'name': 'distribution', 'stream': 'DECISIONS dist', 'shape': ['actions.distribution'],
    'labels': "META actions.layout: the kind head (one entry per actions.kinds), then per kind its direction head "
              "(actions.directions), then per kind its duration head (actions.durations)",
    'meaning': "the network's action distribution as the act mode saw it before drawing: log-probabilities of each "
               "head given the kind; -inf: masked (not legal here)",
    'units': 'nats (a byte each: META actions.encoding)'}
VALUE_OUTPUT = {
    'name': 'value', 'stream': 'DECISIONS value', 'shape': [],
    'meaning': "the critic's value of the state: the expected discounted return from here",
    'units': "reward units of this policy's training"}
HIT_OUTPUT = {
    'name': 'hit', 'stream': 'DECISIONS danger', 'shape': ['actions.joint'],
    'labels': "joint actions, (kind * len(actions.directions) + direction) * len(actions.durations) + duration; "
              "only the legal ones are recorded",
    'meaning': "P(Link loses health during this action if it is taken now): the hit head's prediction (the lab's "
               "'hit' target)",
    'units': 'probability (a byte each: P x 255)'}


def network_description(network, checkpoint: Optional[dict] = None) -> dict:
    """What a replay's META says of ``network`` (a ``ZeldaPolicy`` or a ``triforce.nets`` network) beside its index,
    name, hash and act mode (``replay_models``): its family, run id, iterations and steps (from ``checkpoint``, the
    saved dict, when it has one), its architecture (name, parameter count, config without the observation layout),
    whether it is served from Python, and the outputs DECISIONS record of it (``DISTRIBUTION_OUTPUT``,
    ``VALUE_OUTPUT``, and ``HIT_OUTPUT`` when it has a hit head).  No path."""
    from . import nets  # pylint: disable=import-outside-toplevel,cyclic-import
    config = network.config.to_dict()
    arch = nets.arch_of(config)
    hit = any(name.rsplit('.', 1)[-1] == 'hit_head' for name, _ in network.named_modules())
    out = {'architecture': {'name': arch, 'parameters': sum(p.numel() for p in network.parameters()),
                            'config': _without_layout(config)},
           'served': arch != nets.ZELDA,
           'outputs': [DISTRIBUTION_OUTPUT, VALUE_OUTPUT, *([HIT_OUTPUT] if hit else [])]}
    if checkpoint is not None:
        out.update({'family': checkpoint.get('specialist'), 'run': checkpoint.get('run_id'),
                    'iterations': checkpoint.get('iterations'), 'steps': checkpoint.get('steps'),
                    'obs_schema': checkpoint.get('obs_schema_version'),
                    'action_schema': checkpoint.get('action_schema_version')})
    return out


def _without_layout(config):
    """A network config without its observation layouts (``obs_spec``, at any depth): the replay is model-neutral."""
    if isinstance(config, dict):
        return {k: _without_layout(v) for k, v in config.items() if k != 'obs_spec'}
    return config


def checkpoint_description(network, saved: dict, checkpoint) -> dict:
    """``network_description`` of the network loaded from ``checkpoint`` (file or run dir; ``saved``: its dict) with
    the ``ckpt.pt`` SHA-1 and the version its run dir is stamped with: the checkpoint by content, never by path."""
    return {**network_description(network, saved), 'checkpoint_sha1': checkpoint_sha1(checkpoint),
            'version': model_version(checkpoint)}


def replay_models(names: Iterable[str], policies: Iterable, acts: Iterable[str], described: dict) -> list:
    """A replay's META ``models``: per policy index (DECISIONS' and ROOM events' ``policy``) its index, name (the
    policy key: a family, or ``family@route``), Rust network hash, the act mode it played by and its description
    (``described[name]``: ``checkpoint_description`` / ``network_description``).  No filesystem path of any kind
    (the owner, Nov 23: passed around, a path is useless); a network is known by its family, run id and checkpoint
    SHA-1."""
    return [{'index': index, 'name': name, 'hash': f'{policy.hash:016x}', 'act': act, **described.get(name, {})}
            for index, (name, policy, act) in enumerate(zip(names, policies, acts))]


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
