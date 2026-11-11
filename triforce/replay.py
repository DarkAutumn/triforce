"""Replay files (docs/rust-actor.md): what an evaluation or chain run played, small enough to keep every play and
exact enough to replay it.

A replay file is JSON lines.  The first line is the header: ``kind`` (``eval`` or ``chain``), ``task`` (what the
pool was built from: the specialist config(s) as JSON, and for chains the route, ``start_from``, ``stop_at``,
``max_steps``, ``stall_steps`` and ``missing``), ``models`` (in policy-index order: ``name``, ``checkpoint``, the
Rust network's ``hash`` and ``deterministic``, whether it acted by argmax), ``backend`` (where the probabilities came
from), ``code`` (``git describe --always --dirty``), ``data`` (a hash of game.yaml and routes.yaml) and
``states_dir`` (and, for chains, ``start_hash``: the ``start_from`` state file's hash).  Every further line
is one played episode: ``job`` (its key: the eval seed or the chain run seed), ``start_state`` and ``start_hash``
(eval: the start state's name and its file's hash), ``actions`` (base64, one byte per head index, three per
decision) and ``result`` (the eval record or the run result).

``replay(header, play)`` rebuilds the task for that one job, feeds the recorded actions to the same models and
returns what the pool reports: every decision's ``log_prob``, ``entropy`` and ``value`` under the policy that acted
(bit for bit the original play's when the code and models match), ``obs`` with ``observations=True``, and the
episode's ``events``.  A replay that runs past its recorded actions, stops before them, or ends with another result
than the recorded one has diverged and raises.  Before playing, a changed game/route data hash or start state file
raises (``strict=False``: warns); a different code version warns.  ``read`` drops a cut-off last line (a writer
killed mid-write) with a warning.

``meta`` and ``record_options`` are for the binary TFREPLAY recordings (``crates/replay``; ``EnvPool(record=...)``
reports each finished episode or run as a ``replay`` event).
"""

import base64
import json
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from . import __version__, _native
from .provenance import code_version, data_hash, state_hash

BACKEND = 'rust-cpu'
FORMAT = 2

STREAMS = ('rewards', 'decisions', 'keyframes')     # the optional streams of a TFREPLAY recording
DEFAULT_KEYFRAME_FRAMES = 600


def meta(kind: str, models: list, route: Optional[str] = None, stop_at: Optional[str] = None, **extra) -> dict:
    """The META base object of a recording: what played.  ``kind``: 'chain', 'eval' or 'train'; ``models``: per
    policy index its name, checkpoint, hash, act mode and version (``provenance.models_header``)."""
    return {'triforce': __version__, 'code': code_version(), 'data': data_hash(), 'kind': kind, 'route': route,
            'stop_at': stop_at, 'models': models, **extra}


def record_options(meta: dict, streams: Iterable[str] = (), state: str = 'embed',
                   keyframe_frames: int = DEFAULT_KEYFRAME_FRAMES) -> dict:
    """``EnvPool(record=...)``: record every episode or run as a TFREPLAY file.  ``streams``: the optional ones
    to keep (``STREAMS``); ``state``: 'embed' (a replay that starts from a state file carries it) or 'path' (it
    names it); ``keyframe_frames``: with 'keyframes', a state every that many frames."""
    # pylint: disable=redefined-outer-name
    streams = set(streams)
    if streams - set(STREAMS):
        raise ValueError(f"unknown replay stream(s) {sorted(streams - set(STREAMS))}; streams: {STREAMS}")
    if state not in ('embed', 'path'):
        raise ValueError(f"state must be 'embed' or 'path', not {state!r}")
    return {'state': state, 'rewards': 'rewards' in streams, 'decisions': 'decisions' in streams,
            'keyframe_frames': int(keyframe_frames) if 'keyframes' in streams else 0, 'meta': json.dumps(meta)}


def encode_actions(actions) -> str:
    """[n, 3] head indices → base64 of one byte each."""
    return base64.b64encode(np.asarray(actions, dtype=np.uint8).tobytes()).decode('ascii')


def decode_actions(text: str) -> np.ndarray:
    """Inverse of ``encode_actions``: an [n, 3] int64 array."""
    return np.frombuffer(base64.b64decode(text), dtype=np.uint8).reshape(-1, 3).astype(np.int64)


def header(kind: str, task: dict, models: list, states_dir=None) -> dict:
    """A replay file's first line."""
    head = {'format': FORMAT, 'kind': kind, 'task': task, 'models': models, 'backend': BACKEND,
            'code': code_version(), 'data': data_hash(),
            'states_dir': None if states_dir is None else str(states_dir)}
    if kind == 'chain' and task.get('start_from'):
        head['start_hash'] = state_hash(task['start_from'], states_dir)
    return head


class ReplayWriter:
    """Writes a replay file: the header, then one line per play as plays finish."""

    def __init__(self, path, head: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._states_dir = head.get('states_dir')
        self._hashes: dict = {}
        self._file = open(self.path, 'w', encoding='utf-8')  # pylint: disable=consider-using-with
        self._file.write(json.dumps(head) + '\n')

    def write(self, play: dict, result: dict, start_state: Optional[str] = None) -> None:
        """One played episode (a ``poll_plays`` dict) and its result."""
        line = {'job': int(play['job']), 'actions': encode_actions(play['actions']), 'result': result}
        if start_state is not None:
            if start_state not in self._hashes:
                self._hashes[start_state] = state_hash(start_state, self._states_dir)
            line['start_state'] = start_state
            line['start_hash'] = self._hashes[start_state]
        self._file.write(json.dumps(line) + '\n')
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read(path) -> tuple[dict, list[dict]]:
    """(header, plays) of a replay file; each play's ``actions`` decoded to [n, 3].  A cut-off last line (the
    writer died mid-write) is dropped with a warning; a bad line before it raises ValueError naming the line."""
    with open(path, encoding='utf-8') as f:
        numbered = [(number, text) for number, text in enumerate(f, 1) if text.strip()]
    lines = []
    for index, (number, text) in enumerate(numbered):
        try:
            line = json.loads(text)
            if index and not {'job', 'actions', 'result'} <= set(line):
                raise ValueError("a play line needs job, actions and result")
        except ValueError as error:
            if index == len(numbered) - 1 and index > 0:
                warnings.warn(f"{path}: dropped a truncated last line ({number})", stacklevel=2)
                break
            raise ValueError(f"{path}:{number}: not a replay line: {error}") from error
        lines.append(line)
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


def _check_world(head: dict, play: dict, strict: bool) -> None:
    """Raises (``strict``) or warns when the data or start state the play was recorded with changed; warns on
    another code version."""
    problems = []
    if head.get('data') != data_hash():
        problems.append(f"game/route data hash is {data_hash()}, recorded {head.get('data')}")
    if head['kind'] == 'eval' and 'start_hash' in play:
        now = state_hash(play['start_state'], head['states_dir'])
        if now != play['start_hash']:
            problems.append(f"start state {play['start_state']} hash is {now}, recorded {play['start_hash']}")
    if head['kind'] == 'chain' and 'start_hash' in head:
        now = state_hash(head['task']['start_from'], head['states_dir'])
        if now != head['start_hash']:
            problems.append(f"start state {head['task']['start_from']} hash is {now}, recorded {head['start_hash']}")
    if problems:
        message = f"job {play['job']} was recorded in another world: " + '; '.join(problems)
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=3)
    if head.get('code') != code_version():
        warnings.warn(f"replaying code {code_version()}, the play was recorded with {head.get('code')}", stacklevel=3)


def _plain(value):
    return json.loads(json.dumps(value))


def replay(head: dict, play: dict, observations: bool = False, policies: Optional[list] = None,
           strict: bool = True) -> dict:
    """Replays one recorded play (see the module doc); ``policies`` (Rust networks in policy-index order) replace
    loading the header's checkpoints; ``strict=False`` only warns about a changed data hash or start state."""
    if head['backend'] != BACKEND:
        raise ValueError(f"the play's probabilities came from {head['backend']!r}, this replays on {BACKEND!r}")
    _check_world(head, play, strict)
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
    if head['kind'] == 'eval':
        results = [event['record'] for event in events if event['type'] == 'eval']
    else:
        results = [event['result'] for event in events if event['type'] == 'run' and event['result']['steps'] > 0]
    if _plain(results) != [_plain(play['result'])]:
        raise RuntimeError(f"the replay of job {job} diverged: it ended with {results}, recorded {play['result']}")
    return {**plays[0], 'events': events}
