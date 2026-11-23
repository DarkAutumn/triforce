"""Recording runs as TFREPLAY files (``crates/replay``): what an evaluation, a chain run or a played game did,
exact enough to replay bit for bit on the emulator alone.

A replay is one binary file per run or episode: where it started (console power on, a state file by name, or an
embedded state), every controller input, every memory edit, the cheated items spent, the state it ended in, how it
ended, what the agent was steered to and could do (TARGETS: the objective and its role-labelled targets; MASKS:
the legal actions and the rule that blocked a kind; DECISIONS: every model decision with its whole action
distribution, the act mode's draw and the hit head's danger per action; all always recorded) and the optional
streams of ``STREAMS`` (every reward term, a state every ``keyframe_frames`` frames).  ``_native.Replay`` reads one, ``_native.ReplayPlayer``
plays one, ``_native.replay_verify`` checks one against the emulator, ``_native.replay_stitch`` and
``_native.replay_truncate`` join and cut them.

A pool records when it is built with ``record=record_options(meta(...), ...)`` and reports every finished run or
episode as a ``replay`` event (``{'type': 'replay', 'env', 'job', 'data'}``); ``ReplayDir`` keeps those as files.
Command lines share ``add_arguments`` / ``options_from`` (``--replays DIR``, ``--streams``, ``--keyframe-frames``,
``--state``).
"""

import json
import os
from pathlib import Path
from typing import Collection, Iterable, Optional

from . import __version__, _native
from .provenance import code_commit, code_version, data_hash

STREAMS = ('rewards', 'keyframes')     # the optional streams of a recording (decisions, targets, masks: always)
DEFAULT_KEYFRAME_FRAMES = 600
ENDINGS = ('unknown', 'win', 'death', 'stuck', 'other', 'error')
SUFFIX = '.tfreplay'


def meta(kind: str, models: list, route: Optional[str] = None, stop_at: Optional[str] = None, **extra) -> dict:
    """The META base object of a recording: what played.  ``kind``: 'chain', 'eval' or 'train'; ``models``: per
    policy index the network that played it, described without any path (``provenance.replay_models``); ``code``
    and ``commit``: the checkout that played (``git describe``, ``git rev-parse HEAD``)."""
    return {'triforce': __version__, 'code': code_version(), 'commit': code_commit(), 'data': data_hash(),
            'kind': kind, 'route': route, 'stop_at': stop_at, 'models': models, **extra}


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
    if 'keyframes' in streams and int(keyframe_frames) < 1:
        raise ValueError(f"keyframe_frames must be at least 1, not {keyframe_frames}")
    return {'state': state, 'rewards': 'rewards' in streams,
            'keyframe_frames': int(keyframe_frames) if 'keyframes' in streams else 0, 'meta': json.dumps(meta)}


def parse_streams(text: Optional[str]) -> tuple:
    """``--streams``: a comma-separated subset of ``STREAMS`` ('' or None: none), in ``STREAMS`` order; ValueError
    naming an unknown one."""
    given = {name.strip() for name in (text or '').split(',') if name.strip()}
    if given - set(STREAMS):
        raise ValueError(f"unknown replay stream(s) {sorted(given - set(STREAMS))}; streams: {', '.join(STREAMS)}")
    return tuple(name for name in STREAMS if name in given)


def add_arguments(parser, default_state: str = 'embed', replays: bool = True) -> None:
    """``--replays DIR`` (unless ``replays`` is False: the caller has its own output folder), ``--streams``,
    ``--keyframe-frames`` and ``--state`` (``options_from`` reads them)."""
    if replays:
        parser.add_argument('--replays', default=None, metavar='DIR',
                            help=f"record every run as a replay file DIR/<seed>-<ending>{SUFFIX}: its inputs, memory "
                                 "edits and end state, replayable bit for bit on the emulator alone (tf replay "
                                 "info|verify)")
    parser.add_argument('--streams', default='', metavar='LIST',
                        help=f"optional replay streams to record, comma-separated: {', '.join(STREAMS)} (every reward "
                             "term per step; a state every --keyframe-frames frames, to seek to or continue from; "
                             "every decision, target and action mask is always recorded)")
    parser.add_argument('--keyframe-frames', type=int, default=DEFAULT_KEYFRAME_FRAMES, metavar='N',
                        help=f"with the keyframes stream: a state every N frames (default {DEFAULT_KEYFRAME_FRAMES})")
    parser.add_argument('--state', choices=('embed', 'path'), default=default_state,
                        help="a replay that starts from a state file carries the state (embed: self-contained) or "
                             f"only names it (path: smaller; replaying needs the same file); default {default_state}")


def options_from(args, meta: dict, force: bool = False) -> Optional[dict]:
    """The ``EnvPool(record=...)`` dict of parsed ``add_arguments`` flags; None without ``--replays`` unless
    ``force``.  ValueError for an unknown stream, or stream flags without ``--replays``."""
    # pylint: disable=redefined-outer-name
    streams = parse_streams(args.streams)
    if not force and getattr(args, 'replays', None) is None:
        if streams:
            raise ValueError("--streams needs --replays")
        return None
    return record_options(meta, streams, args.state, args.keyframe_frames)


class ReplayDir:
    """A folder of recordings: ``add(seed, data)`` writes ``<path>/<prefix><seed:08d>-<ending>.tfreplay``.
    ``only``: the endings to keep (``ENDINGS`` names; None: all); ``saved`` counts the files written, ``endings``
    every recording seen by ending."""

    def __init__(self, path, only: Optional[Collection[str]] = None, prefix: str = ''):
        if only is not None and set(only) - set(ENDINGS):
            raise ValueError(f"unknown ending(s) {sorted(set(only) - set(ENDINGS))}; endings: {ENDINGS}")
        self.path = Path(path)
        self.only = None if only is None else frozenset(only)
        self.prefix = prefix
        self.saved = 0
        self.endings: dict = {}
        self.paths: list = []
        self.path.mkdir(parents=True, exist_ok=True)

    def add(self, seed: int, data: bytes) -> Optional[Path]:
        """Writes one recording (atomically) if its ending is wanted; returns its path, or None when dropped."""
        ending = _native.Replay(data).ending
        self.endings[ending] = self.endings.get(ending, 0) + 1
        if self.only is not None and ending not in self.only:
            return None
        return self.write(f'{self.prefix}{int(seed):08d}-{ending}{SUFFIX}', data)

    def write(self, name: str, data: bytes) -> Path:
        """Writes ``data`` as ``<path>/<name>`` atomically and counts it."""
        path = self.path / name
        tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
        tmp.write_bytes(data)
        os.replace(tmp, path)
        self.saved += 1
        self.paths.append(path)
        return path


def chain_records(files: Iterable) -> tuple[list, dict]:
    """The chain records and provenance of a chain's runs from their replays alone (version 2; no route data, code or
    models outside them): (the runs sorted by seed, each as ``eval-chain`` writes it to its JSON lines:
    ``RunResult.to_dict`` of META ``run`` with the VISIT events as its rooms; the ``<output>.meta.json`` object:
    ``{'provenance': ..., 'batches': [{'first_seed', 'runs', 'finished'}]}``).  The provenance is META ``chain``
    with the models (a checkpoint as ``<family>/<run>/ckpt.pt``: a numbered snapshot's own folder is not kept), the
    version, code and data; ``missing`` names the families without a network (their reasons named paths: None).
    ``files``: replay paths or bytes of one chain (the same META ``chain``, ``models``); ValueError otherwise."""
    from .runner import RunResult   # pylint: disable=import-outside-toplevel,cyclic-import
    records, header = [], None
    for item in files:
        recording = _native.Replay(item if isinstance(item, (bytes, bytearray)) else Path(item).read_bytes())
        meta = recording.meta
        if meta.get('kind') != 'chain' or 'run' not in meta or 'chain' not in meta:
            raise ValueError(f"{item if not isinstance(item, (bytes, bytearray)) else 'a replay'}: not a chain run "
                             "recorded with its META run and chain (version 2)")
        rooms = [event['record'] for event in recording.events() if event['type'] == 'visit']
        records.append(RunResult.from_dict({**meta['run'], 'rooms': rooms}).to_dict())
        mine = {key: meta.get(key) for key in ('chain', 'models', 'triforce', 'code', 'data')}
        if header is not None and mine != header:
            raise ValueError("the replays are not one chain's: their META chain, models or versions differ")
        header = mine
    if header is None:
        raise ValueError("no replay")
    records.sort(key=lambda r: r['seed'])
    chain = {k: v for k, v in header['chain'].items() if k not in ('route_steps', 'triforces')}
    models = [{'name': m['name'], 'checkpoint': f"{m['family']}/{m['run']}/ckpt.pt" if m.get('run') else None,
               'hash': m['hash'], 'act': m['act'], 'version': m.get('version'),
               'checkpoint_sha1': m.get('checkpoint_sha1'), 'config_sha256': m.get('config_sha256')}
              for m in header['models']]
    provenance = {**chain, 'models': models, 'missing': {name: None for name in chain['missing']},
                  'version': header['triforce'], 'code': header['code'], 'data': header['data']}
    return records, {'provenance': provenance,
                     'batches': [{'first_seed': records[0]['seed'], 'runs': len(records), 'finished': True}]}


def chain_route(files: Iterable) -> dict:
    """The route facts a chain's report needs, from one of its replays: ``{'steps': the route's step count,
    'triforces': {step: dungeon level}}`` (META ``chain``)."""
    item = next(iter(files))
    chain = _native.Replay(item if isinstance(item, (bytes, bytearray)) else Path(item).read_bytes()).meta['chain']
    return {'steps': chain['route_steps'], 'triforces': {step: level for step, level in chain['triforces']}}
