"""Plays a route end to end with trained specialists (design §9).

The chain itself runs natively (``ChainTask`` in an ``EnvPool``): on every room entry the room's game.yaml
``specialist:`` tag picks who acts, specialists switch only on room transitions, and a run ends with a
``RunResult``.  This module owns the models: each pool row names its acting specialist (the ``policy`` column, an
index into the loaded-model list), and the specialists act inside the pool (``EnvPool.play`` with their Rust
networks, docs/rust-actor.md).  Sampled actions use Gumbel noise keyed to (run seed, decision), so a seed plays the
same run whichever env or thread runs it.  A specialist without a promoted model ends the runs that need it as
``error`` with the registry's reason; an untagged room the route needs is an error too, a wrong exit into one a
gameplay failure.  With ``record`` set every run is recorded as a replay (``triforce.replay``).
"""

import difflib
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from . import __version__, _native, replay
from .paths import STATES_DIR
from .provenance import code_version, data_hash, location_text, models_header
from .registry import (CHECKPOINT_NAME, act_mode, checkpoint_sha1, load_model, promoted_checkpoint,
                       require_compatible)
from .replay import ReplayDir
from .specialist import SpecialistConfig, list_specialists, load_specialist
from .zelda_enums import MapLocation

DEFAULT_MAX_STEPS = 50_000

RoomKey = Tuple[int, int, bool]


class RunnerError(RuntimeError):
    """The runner can't start: unknown or malformed route."""


@dataclass
class Route:
    """Where a route starts and what ends it.  The start: ``start_state`` (``'start'``: the game start state), or
    ``power_on`` (``{'name', 'delay', 'max_delay'}``: console power on, ``delay`` None = drawn per run seed from
    ``0..=max_delay``), or ``checkpoint`` (``{'state': bytes, 'step': int, 'granted': [bombs, keys, rupees]}``:
    the state another run ended in, e.g. ``Replay.states()[-1]``).  The end: ``terminal`` (a room), or
    ``stop_step`` (a route step, reached by the route's own exit), or neither: the route's last triforce.
    ``start_step``: the route step the start is on."""
    name: str
    start_state: str
    terminal: Optional[MapLocation] = None
    stop_step: Optional[int] = None
    power_on: Optional[dict] = None
    checkpoint: Optional[dict] = None
    start_step: int = 0

    @staticmethod
    def load(name: str, stop_at: Optional[MapLocation] = None, start_from: Optional[str] = None,
             states_dir=None, *, stop_step: Optional[int] = None, power_on: Optional[dict] = None,
             checkpoint: Optional[dict] = None) -> 'Route':
        """Loads a route from routes.yaml.  ``stop_at`` ends it on entering that room, ``stop_step`` on reaching
        that route step, instead of on its last step's triforce.  One of ``start_from`` (a savestate, e.g. a
        route entry state like ``2_7ds_0``, in ``states_dir``, default ``paths.STATES_DIR``, or a path),
        ``power_on`` and ``checkpoint`` replaces the game start (``Route``).  The start state must exist, the
        stop must be after the start's route step, and a power-on name and delay must be ones the game takes
        (the native ``Route`` checks the same when the chain task is built)."""
        # pylint: disable=too-many-arguments, too-many-branches
        try:
            steps = _native.route_steps(name)
        except _native.ConfigError as err:
            raise RunnerError(str(err)) from err
        if not steps:
            raise RunnerError(f"Unknown route '{name}'")
        if steps[0]['entry'] != 'start':
            raise RunnerError(f"Route '{name}' must begin at the game start (entry: start)")
        if stop_at is None and stop_step is None and steps[-1]['exit'] != 'triforce':
            raise RunnerError(f"Route '{name}' doesn't end on a triforce")
        if stop_at is not None and stop_step is not None:
            raise RunnerError("a stop room and a stop step are exclusive")
        if start_from == 'start':
            start_from = None
        starts = [what for what, given in (('a start state (--from)', start_from), ('a power-on start', power_on),
                                           ('a checkpoint', checkpoint)) if given is not None]
        if len(starts) > 1:
            raise RunnerError(f"{' and '.join(starts)} are exclusive: each replaces the game start")
        start = 0
        if start_from is not None:
            path = _start_state_path(start_from, states_dir)
            if not path.is_file():
                near = difflib.get_close_matches(path.stem, [p.stem for p in path.parent.glob('*.state')], n=5)
                raise RunnerError(f"Unknown start state '{start_from}': {path} not found (states in {path.parent}"
                                  + (f"; close: {near}" if near else '') + ')')
            start = _start_step(steps, path.stem)
        if power_on is not None:
            power_on = _power_on(power_on)
        if checkpoint is not None:
            start = checkpoint.get('step')
            if not isinstance(start, int) or not 0 <= start < len(steps):
                raise RunnerError(f"the checkpoint's step {start!r} is not a step of route '{name}' "
                                  f"(0..={len(steps) - 1})")
            checkpoint = {'state': bytes(checkpoint['state']), 'step': start,
                          'granted': list(checkpoint.get('granted') or (0, 0, 0)),
                          'randomize_rng': bool(checkpoint.get('randomize_rng', False))}
        if stop_at is not None:
            later = [(s['level'], s['room'], s['in_cave']) for s in steps[start + 1:]]
            if (stop_at.level, stop_at.value, stop_at.in_cave) not in later:
                rooms = sorted(set(later), key=later.index)
                raise RunnerError(f"--stop-at {location_text(stop_at)} is not a room of route '{name}' after the start "
                                  f"step; rooms: {[location_text(MapLocation(*room)) for room in rooms]}")
        if stop_step is not None and not start < stop_step < len(steps):
            raise RunnerError(f"--stop-at step:{stop_step} is not a step of route '{name}' after the start step "
                              f"{start}; steps: {start + 1}..={len(steps) - 1}")
        return Route(name, start_from or 'start', stop_at, stop_step, power_on, checkpoint, start)


def _power_on(options: dict) -> dict:
    """A power-on start's ``{'name', 'delay', 'max_delay'}`` checked and filled in (name LINK, a random delay over
    the whole range); ``RunnerError`` for a name the game can't register or a delay out of range."""
    unknown = sorted(set(options) - {'name', 'delay', 'max_delay'})
    if unknown:
        raise RunnerError(f"unknown power-on options {unknown} (name, delay, max_delay)")
    limit = _native.MAX_POWER_ON_DELAY
    try:
        name = _native.power_on_name(options.get('name') or 'LINK')
    except ValueError as err:
        raise RunnerError(str(err)) from err
    delay, max_delay = options.get('delay'), options.get('max_delay')
    max_delay = limit if max_delay is None else max_delay
    for what, value in (('delay', delay), ('max delay', max_delay)):
        if value is not None and (not isinstance(value, int) or not 0 <= value <= limit):
            raise RunnerError(f"the power-on {what} {value!r} must be a frame count in 0..={limit}")
    return {'name': name, 'delay': delay, 'max_delay': max_delay}


def _start_state_path(name: str, states_dir=None) -> Path:
    """Where the chain looks for start state ``name``: a ``.state`` path as is, else ``<states_dir>/<name>.state``."""
    path = Path(name)
    if path.suffix == '.state' and (path.is_absolute() or path.parent != Path('.')):
        return path
    return Path(states_dir or STATES_DIR) / (name if name.endswith('.state') else f'{name}.state')


_STATE_NAME = re.compile(r'^(\d+)_([0-9a-f]{2})(n|s|e|w|c|cave)(?:_h\d+)?_\d+$')
_STATE_SIDES = {'n': ('N', False), 's': ('S', False), 'e': ('E', False), 'w': ('W', False), 'c': ('cave', False),
                'cave': ('S', True)}


def _start_step(steps: List[dict], label: str) -> int:
    """The first route step a start state named ``<level>_<room><side>_<chain>`` (or a harvested
    ``..._h<seed>_<n>``) begins: same room and entry (``start_states::state_entry``); 0 when the name doesn't say."""
    match = _STATE_NAME.match(label)
    if match is None:
        return 0
    entry, in_cave = _STATE_SIDES[match.group(3)]
    room = (int(match.group(1)), int(match.group(2), 16), in_cave)
    return next((s['index'] for s in steps if (s['level'], s['room'], s['in_cave']) == room and s['entry'] == entry),
                0)


@dataclass
class RoomRecord:
    """One visit to one room."""
    index: int
    level: int
    room: str
    in_cave: bool
    specialist: str
    steps: int = 0
    frames: int = 0
    health_in: float = 0.0
    health_out: float = 0.0
    max_health: int = 0
    keys_in: Optional[int] = None   # keys carried on entering / leaving the room (None in pre-Nov-6 records)
    keys_out: Optional[int] = None
    outcome: str = 'in-progress'    # exit-correct, exit-wrong, retreat, retreat-limit, missed (heart waypoint), death,
    #                                 stuck (room or run budgets, wrong-exit or bounce limit), milestone, error
    # Dropped hearts during the visit: healed = real hearts gained, wasted = taken - healed, left = still on the
    # floor when the visit ended.
    hearts_dropped: int = 0
    hearts_taken: int = 0
    hearts_healed: float = 0.0
    hearts_wasted: float = 0.0
    hearts_left: int = 0
    hearts_expired: int = 0
    linger: dict = field(default_factory=dict)
    blocks: int = 0
    no_damage_max: Optional[int] = None     # longest stretch of actions without damage dealt or taken (None: older)
    # Bombs placed during the visit, and cheated items used (a room that needs an item Link lacks grants it: a bomb
    # in requires-bombs rooms, a key at a locked route door, rupees in requires-rupees shops;
    # crates/zelda/src/cheats.rs).  0 in older records.
    bombs_used: int = 0
    cheated_bombs: int = 0
    cheated_keys: int = 0
    cheated_rupees: int = 0
    # Dodongo stuns (a bomb's smoke) and bombs he swallowed during the visit (crates/env/src/critic.rs
    # BossTransitions).  0 in older records.
    boss_stuns: int = 0
    boss_eats: int = 0
    # Bombs in the visit (crates/zelda/src/bombs.rs): placed, enemy hits credited to them, placed without a hit,
    # damage dealt (wood-sword hits: HP lost, a kill's remaining HP), 10th damage-free kills (any weapon), bombs
    # placed while the next kill was the 10th, forced bomb drops armed (a bomb made the 10th kill), bomb pickups.
    # 0 in older records.
    bomb_placed: int = 0
    bomb_hits: int = 0
    bomb_misses: int = 0
    bomb_damage: int = 0
    bomb_tenth_kills: int = 0
    bomb_tenth_tries: int = 0
    bomb_armed: int = 0
    bomb_picked_up: int = 0


@dataclass
class RunResult:
    """Everything one run produced; written as JSON lines by ``write_jsonl``."""
    route: str
    seed: Optional[int]
    result: str = 'in-progress'     # success, failure, error
    reason: str = ''
    steps: int = 0
    frames: int = 0
    retreats: int = 0               # rooms left through a route step's `retreat` exit (4 end the run: retreat-limit)
    wrong_exits: int = 0            # wrong exits into tagged rooms (the 20th ends the run stuck: wrong-exit-limit)
    rooms: List[RoomRecord] = field(default_factory=list)

    @property
    def failed_room(self) -> Optional[RoomRecord]:
        """The room the run ended in, if it didn't succeed."""
        return self.rooms[-1] if self.rooms and self.result != 'success' else None

    @property
    def cheated(self) -> Dict[str, int]:
        """Cheated items used in the run, per item (``_native.CHEAT_ITEMS``): any means it wasn't beaten naturally."""
        return {item: sum(getattr(room, f'cheated_{item}') for room in self.rooms) for item in _native.CHEAT_ITEMS}

    def to_dict(self) -> dict:
        """JSON-friendly dict."""
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> 'RunResult':
        """Inverse of ``to_dict`` (also reads the native pool's ``run`` events)."""
        rooms = [RoomRecord(**room) for room in data.get('rooms', ())]
        return RunResult(**{**data, 'rooms': rooms})


def write_jsonl(path, results: List[RunResult], append: bool = False):
    """Writes results (one per line) to path: a complete new file (temp file + ``os.replace``), or appended to it
    with ``append``."""
    path = Path(path)
    lines = ''.join(json.dumps(result.to_dict()) + '\n' for result in results)
    if append:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(lines)
        return
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(lines)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class Policies:
    """The specialists' Rust networks for a chain pool, in policy-index order, plus why the others are missing.

    ``checkpoints`` (name -> checkpoint file or run dir) overrides the registry's promoted model; ``networks``
    (name -> a ``ZeldaPolicy``) supplies a model directly; ``specialist_configs`` (name -> config) replaces a
    specialist's canonical YAML at runtime.  ``checkpoints`` afterwards maps each loaded name to its file and
    ``deterministic`` holds, per policy, whether its model acts by argmax (the run's act mode, ``registry``; a
    given network samples)."""

    def __init__(self, models_dir=None, checkpoints: Optional[Dict[str, str]] = None,
                 specialist_configs: Optional[Dict[str, SpecialistConfig]] = None,
                 networks: Optional[Dict[str, object]] = None, names: Optional[Iterable[str]] = None):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        overrides = dict(checkpoints or {})
        configs = dict(specialist_configs or {})
        given = dict(networks or {})
        known = set(names if names is not None else list_specialists()) | set(given) | set(configs)
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise RunnerError(f"--checkpoint for unknown specialist(s) {unknown}; known: {sorted(known)}")
        for name, path in overrides.items():
            if not Path(path).is_file() and not (Path(path) / CHECKPOINT_NAME).is_file():
                raise RunnerError(f"--checkpoint {name}={path}: no checkpoint file there")
        self.names: List[str] = []
        self.configs: List[SpecialistConfig] = []
        self.policies: list = []
        self.checkpoints: Dict[str, str] = {}
        self.deterministic: List[bool] = []
        self.missing: Dict[str, str] = {}
        for name in sorted(known):
            network = given.get(name)
            argmax = False
            if network is None:
                if name in overrides:
                    path = overrides[name]
                else:
                    try:
                        path = promoted_checkpoint(name, models_dir)
                    except FileNotFoundError as err:
                        self.missing[name] = str(err)
                        continue
                require_compatible(path)
                with torch.random.fork_rng():   # building the network draws init weights; keep seeds intact
                    network, _ = load_model(path)
                self.checkpoints[name] = str(path)
                argmax = act_mode(path) == 'argmax'
            self.names.append(name)
            self.configs.append(configs.get(name) or load_specialist(name))
            self.policies.append(network.to_native())
            self.deterministic.append(argmax)


class ChainPool:
    """An ``EnvPool`` of chain runs of one route (one run per seed, shared by ``num_envs`` envs) and the policies
    acting in it.  ``run()`` plays every seed; watchers call ``play()`` and follow ``poll_events()`` themselves.
    ``stall_steps`` (the watch player's simulations; None = off): room events carry ``stalled`` after that many
    actions in a room without damage dealt or taken, and the run ends ``stuck`` at twice as many.  ``record``
    (``replay.record_options``; None = off): every run is recorded
    as a TFREPLAY file and reported as a ``replay`` event; ``replays`` (a ``ReplayDir``) keeps those as files.
    ``deterministic``: None = each policy's act mode (``Policies.deterministic``), else argmax (True) or sampling
    (False) for every policy.  ``harvest`` (a directory; None = off): every correct room entry where the acting
    specialist changes (a handoff) saves its pre-entry state as
    ``harvest/<specialist>/<level>_<room><side>_h<seed>_<n>.state`` (``n``: the room's index in the run), a start
    state for that specialist's ``harvested_dir``."""

    def __init__(self, route: Route, seeds: Sequence[int], policies: Policies, num_envs: int = 1,
                 threads: Optional[int] = None, seed: int = 0, fps: float = 0.0,
                 deterministic: Optional[bool] = None,
                 max_steps: int = DEFAULT_MAX_STEPS, states_dir=None, record: Optional[dict] = None,
                 stall_steps: Optional[int] = None, replays: Optional[ReplayDir] = None, harvest=None):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        if replays is not None and record is None:
            raise ValueError("a replay folder needs record options (replay.record_options)")
        self.route = route
        self.policies = policies
        self.deterministic = (list(policies.deterministic) if deterministic is None
                              else [bool(deterministic)] * len(policies.policies))
        self.record = record
        self.replays = replays
        start_from = None if route.start_state == 'start' else route.start_state
        states_dir = None if states_dir is None else str(states_dir)
        task = _native.ChainTask(policies.configs, route.name, [int(s) for s in seeds], stop_at=route.terminal,
                                 start_from=start_from, max_steps=max_steps, missing=policies.missing,
                                 states_dir=states_dir, stall_steps=stall_steps,
                                 harvest=None if harvest is None else str(harvest),
                                 stop_step=route.stop_step, power_on=route.power_on, checkpoint=route.checkpoint)
        num_envs = max(1, min(num_envs, len(seeds)))
        self.pool = _native.EnvPool(task, num_envs=num_envs, threads=threads, seed=seed, fps=fps, record=record)

    def play(self):
        """Every env starts acting on its own (``EnvPool.play``)."""
        self.pool.play(self.policies.policies, deterministic=self.deterministic,
                       distributions=bool(self.record and self.record.get('decisions')))

    def poll_events(self) -> List[dict]:
        """The pool's events since the last call; with ``replays`` set, every finished run's recording (a
        ``replay`` event) is written to the folder first."""
        events = self.pool.poll_events()
        if self.replays is not None:
            for event in events:
                if event['type'] == 'replay':
                    self.replays.add(int(event['job']), event['data'])
        return events

    @staticmethod
    def results(events: Iterable[dict]) -> List[Tuple[int, RunResult]]:
        """(env, RunResult) of every ``run`` event in ``events``."""
        return [(e['env'], RunResult.from_dict(e['result'])) for e in events if e['type'] == 'run']

    def run(self, on_result: Optional[Callable[[RunResult], None]] = None) -> List[RunResult]:
        """Plays every seed; returns the results sorted by seed."""
        results: List[RunResult] = []
        self.play()
        while True:
            idle = self.pool.wait_idle(timeout=0.1)
            for _, result in self.results(self.poll_events()):
                results.append(result)
                if on_result is not None:
                    on_result(result)
            if idle:
                return sorted(results, key=lambda r: (r.seed is None, r.seed))

    def close(self):
        """Stops the pool's worker threads (idempotent)."""
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Runner:
    """Plays a route with the promoted (or given) specialists: ``run(route, seed)`` for one run,
    ``run_many(route, seeds)`` for a batch on ``parallel`` envs."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, models_dir=None, checkpoints: Optional[Dict[str, str]] = None,
                 deterministic: Optional[bool] = None,
                 max_steps: int = DEFAULT_MAX_STEPS,
                 specialist_configs: Optional[Dict[str, SpecialistConfig]] = None,
                 networks: Optional[Dict[str, object]] = None, parallel: int = 1, threads: Optional[int] = None,
                 states_dir=None, harvest=None):
        """
        checkpoints: specialist name → checkpoint path, overriding the registry's promoted model.
        specialist_configs: specialist name → loaded config, replacing its canonical YAML at runtime.
        networks: specialist name → a ``ZeldaPolicy`` to act with instead of a checkpoint.
        parallel: envs in the chain pool; threads: its worker threads (default min(parallel, cores)).
        deterministic: None = each model's act mode (registry), else argmax (True) or sampling (False) for all.
        harvest: save every handoff's pre-entry state under this directory (``ChainPool``).
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.policies = Policies(models_dir, checkpoints, specialist_configs, networks)
        self.deterministic = deterministic
        self.max_steps = max_steps
        self.parallel = parallel
        self.threads = threads
        self.states_dir = states_dir
        self.harvest = harvest

    def pool(self, route: Route, seeds: Sequence[int], **kwargs) -> ChainPool:
        """A chain pool over this runner's policies (``kwargs``: ``num_envs``, ``threads``, ``fps``, ``seed``,
        ``record``, ``replays``, ``stall_steps``)."""
        kwargs.setdefault('num_envs', self.parallel)
        kwargs.setdefault('threads', self.threads)
        return ChainPool(route, seeds, self.policies, deterministic=self.deterministic, max_steps=self.max_steps,
                         states_dir=self.states_dir, harvest=self.harvest, **kwargs)

    def record_meta(self, route: Route, **extra) -> dict:
        """The META object of recordings of ``route`` played by this runner (``chain_meta``)."""
        return chain_meta(self.policies, self.deterministic, route, **extra)

    def provenance(self, route: Route) -> dict:
        """What plays ``route``: per policy its name, checkpoint, ``ckpt.pt`` sha1, Rust network hash, act mode and
        config hash; the specialists without a model; the route, start, end and step cap; the code version and the
        game data hash (``replay``).  Stored with every chain result (``chain_summary``, ``eval-chain``'s
        ``<output>.meta.json``)."""
        policies = self.policies
        deterministic = (list(policies.deterministic) if self.deterministic is None
                         else [bool(self.deterministic)] * len(policies.policies))
        models = models_header(policies.names, policies.policies, policies.checkpoints, deterministic)
        for model, config in zip(models, policies.configs):
            checkpoint = policies.checkpoints.get(model['name'])
            model['checkpoint_sha1'] = None if checkpoint is None else checkpoint_sha1(checkpoint)
            model['act'] = 'argmax' if model['deterministic'] else 'sample'
            model['config_sha256'] = hashlib.sha256(config.to_json().encode()).hexdigest()
        checkpoint = route.checkpoint and {'step': route.checkpoint['step'], 'granted': route.checkpoint['granted'],
                                           'randomize_rng': route.checkpoint['randomize_rng'],
                                           'state_sha1': hashlib.sha1(route.checkpoint['state']).hexdigest()}
        return {'route': route.name, 'start_from': None if route.start_state == 'start' else route.start_state,
                'power_on': route.power_on, 'checkpoint': checkpoint,
                'stop_at': location_text(route.terminal), 'stop_step': route.stop_step,
                'max_steps': self.max_steps, 'models': models,
                'missing': dict(sorted(policies.missing.items())), 'version': __version__, 'code': code_version(),
                'data': data_hash()}

    def run_many(self, route: Route, seeds: Sequence[int],
                 on_result: Optional[Callable[[RunResult], None]] = None, record: Optional[dict] = None,
                 replays: Optional[ReplayDir] = None) -> List[RunResult]:
        """Plays route once per seed; results sorted by seed.  Never raises for gameplay failures.  ``record``
        and ``replays``: ``ChainPool``'s (every run recorded into a folder)."""
        seeds = list(seeds)
        if not seeds:
            return []
        with self.pool(route, seeds, record=record, replays=replays) as pool:
            return pool.run(on_result)

    def run(self, route: Route, seed: Optional[int] = None) -> RunResult:
        """Plays route once (seed None: 0)."""
        return self.run_many(route, [0 if seed is None else seed])[0]


def parse_location(text: str) -> MapLocation:
    """'2:0x0e', '0:0x77:cave' → MapLocation; anything else raises ``RunnerError``."""
    parts = text.split(':')
    try:
        if (not 2 <= len(parts) <= 3 or (len(parts) == 3 and parts[2] != 'cave')
                or not parts[1].lower().startswith('0x')):
            raise ValueError(text)
        return MapLocation(int(parts[0]), int(parts[1], 16), len(parts) == 3)
    except ValueError:
        raise RunnerError(f"'{text}' is not a room (LEVEL:0xROOM[:cave], e.g. 1:0x73)") from None


def parse_stop(text: Optional[str]) -> Tuple[Optional[MapLocation], Optional[int]]:
    """``--stop-at``: (room, None) for a room (``1:0x73``), (None, step) for a route step (``step:24``), (None,
    None) without one; anything else raises ``RunnerError``."""
    if not text:
        return None, None
    if text.startswith('step:'):
        if not text[5:].isdecimal():
            raise RunnerError(f"'{text}' is not a route step (step:N, e.g. step:24)")
        return None, int(text[5:])
    return parse_location(text), None


def stop_text(route: 'Route') -> Optional[str]:
    """What ends ``route`` as ``--stop-at`` takes it (``1:0x73``, ``step:24``); None: the route's own end."""
    return location_text(route.terminal) if route.stop_step is None else f'step:{route.stop_step}'


STOP_AT_HELP = ("end the route on entering this room (e.g. 1:0x73), or on reaching this route step by the route's "
                "own exit (step:N, e.g. step:24: back on the overworld with Dungeon 2's triforce)")


def add_power_on_arguments(parser) -> None:
    """``--power-on``, ``--name`` and ``--delay``: a start from console power on (``power_on_options``)."""
    limit = _native.MAX_POWER_ON_DELAY
    parser.add_argument('--power-on', action='store_true',
                        help="start every run from console power on, by controller inputs alone (title screen, "
                             "name entry, game start): no state file and no memory write; the run's randomness is "
                             "--delay")
    parser.add_argument('--name', default=None, metavar='NAME',
                        help="with --power-on: the name to register (1 to 8 of A-Z, 0-9 and space; default LINK; "
                             "ZELDA is refused, it starts the second quest)")
    parser.add_argument('--delay', default=None, metavar='N|random',
                        help=f"with --power-on: idle frames on the file-select screen before the name is entered, "
                             f"which set the game's random numbers: N in 0..={limit} (every value is a different "
                             f"game), or 'random' (default): drawn from that range by each run's seed")


def power_on_options(args) -> Optional[dict]:
    """The ``Route.load`` ``power_on`` of parsed ``add_power_on_arguments`` flags (None without ``--power-on``);
    ``RunnerError`` for ``--name`` or ``--delay`` without it, or a delay that is no frame count."""
    if not args.power_on:
        given = [flag for flag, value in (('--name', args.name), ('--delay', args.delay)) if value is not None]
        if given:
            raise RunnerError(f"{' and '.join(given)} need{'s' if len(given) == 1 else ''} --power-on")
        return None
    delay = None
    if args.delay not in (None, 'random'):
        if not args.delay.isdecimal():
            raise RunnerError(f"--delay {args.delay!r} must be a frame count in 0..={_native.MAX_POWER_ON_DELAY}, "
                              f"or 'random'")
        delay = int(args.delay)
    return _power_on({'name': args.name, 'delay': delay})


def parse_overrides(items: Iterable[str]) -> Dict[str, str]:
    """``SPECIALIST=PATH`` command-line items → {specialist: path}; a malformed item raises ``RunnerError``."""
    overrides = {}
    for item in items:
        name, sep, path = item.partition('=')
        if not sep or not name or not path:
            raise RunnerError(f"--checkpoint '{item}' is not SPECIALIST=PATH")
        overrides[name] = path
    return overrides


def chain_summary(results: List[RunResult]) -> dict:
    """The milestone metric, keys carried at the milestone and a per-room failure breakdown.

    ``arrival_keys`` counts successful runs by the keys Link carried into the milestone room (a ``stop_at`` room,
    or the room of the route's last triforce), so milestone arrivals and full key carriage stay separate
    counters.  ``cheated_<item>`` totals the cheated items used (bombs, keys, rupees) and ``runs_with_cheated_<item>``
    counts the runs that used any."""
    runs = len(results)
    successes = sum(r.result == 'success' for r in results)
    failures: Dict[str, int] = {}
    arrival_keys: Dict[str, int] = {}
    for r in results:
        room = r.failed_room
        if room is not None:
            key = f"{room.level}:{room.room}{':cave' if room.in_cave else ''} {room.specialist} {room.outcome}"
            failures[key] = failures.get(key, 0) + 1
        elif r.result == 'success' and r.rooms:
            keys = r.rooms[-1].keys_in
            label = 'unknown' if keys is None else str(keys)
            arrival_keys[label] = arrival_keys.get(label, 0) + 1
    cheated = [r.cheated for r in results]
    summary = {'runs': runs, 'successes': successes, 'success_rate': successes / runs if runs else 0.0,
               'arrival_keys': dict(sorted(arrival_keys.items())),
               'failures': dict(sorted(failures.items(), key=lambda kv: -kv[1])),
               'errors': sorted({r.reason for r in results if r.result == 'error'})}
    for item in _native.CHEAT_ITEMS:
        summary[f'cheated_{item}'] = sum(c[item] for c in cheated)
        summary[f'runs_with_cheated_{item}'] = sum(c[item] > 0 for c in cheated)
    return summary


def chain_meta(policies: Policies, deterministic: Optional[bool], route: Route, **extra) -> dict:
    """The META object of recordings of ``route`` played by ``policies`` (``replay.meta``; ``deterministic``:
    ``ChainPool``'s; ``extra``: more entries): what ``replay.record_options`` takes."""
    modes = list(policies.deterministic) if deterministic is None else [bool(deterministic)] * len(policies.policies)
    models = models_header(policies.names, policies.policies, policies.checkpoints, modes)
    return replay.meta('chain', models, route.name, stop_text(route), **extra)
