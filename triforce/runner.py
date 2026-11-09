"""Plays a route end to end with trained specialists (design §9).

The chain itself runs natively (``ChainTask`` in an ``EnvPool``): on every room entry the room's game.yaml
``specialist:`` tag picks who acts, specialists switch only on room transitions, and a run ends with a
``RunResult``.  This module owns the models: each pool row names its acting specialist (the ``policy`` column, an
index into the loaded-model list), and the specialists act inside the pool (``EnvPool.play`` with their Rust
networks, docs/rust-actor.md).  Sampled actions use Gumbel noise keyed to (run seed, decision), so a seed plays the
same run whichever env or thread runs it.  A specialist without a promoted model ends the runs that need it as
``error`` with the registry's reason; an untagged room the route needs is an error too, a wrong exit into one a
gameplay failure.  With ``plays`` set every run is kept in a replay file (``triforce.replay``).
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from . import _native
from .registry import act_mode, load_model, promoted_checkpoint
from .replay import ReplayWriter, header, location_text, models_header
from .specialist import SpecialistConfig, list_specialists, load_specialist
from .zelda_enums import MapLocation

DEFAULT_MAX_STEPS = 50_000

RoomKey = Tuple[int, int, bool]


class RunnerError(RuntimeError):
    """The runner can't start: unknown or malformed route."""


@dataclass
class Route:
    """Where a route starts and which room ends it (``terminal``; None: the route ends on its last triforce)."""
    name: str
    start_state: str
    terminal: Optional[MapLocation] = None

    @staticmethod
    def load(name: str, stop_at: Optional[MapLocation] = None, start_from: Optional[str] = None) -> 'Route':
        """Loads a route from routes.yaml.  ``stop_at`` ends it on entering that room instead of on its last step's
        triforce, and ``start_from`` (a savestate, e.g. a route entry state like ``2_7ds_0``) replaces the game
        start, for partial chain evals."""
        try:
            steps = _native.route_steps(name)
        except _native.ConfigError as err:
            raise RunnerError(str(err)) from err
        if not steps:
            raise RunnerError(f"Unknown route '{name}'")
        if steps[0]['entry'] != 'start':
            raise RunnerError(f"Route '{name}' must begin at the game start (entry: start)")
        if stop_at is None and steps[-1]['exit'] != 'triforce':
            raise RunnerError(f"Route '{name}' doesn't end on a triforce")
        return Route(name, start_from or 'start', stop_at)


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
    #                                 stuck, milestone, error
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


def write_jsonl(path: str, results: List[RunResult]):
    """Appends results (one per line) to path."""
    with open(path, 'a', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result.to_dict()) + '\n')


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
        self.names: List[str] = []
        self.configs: List[SpecialistConfig] = []
        self.policies: list = []
        self.checkpoints: Dict[str, str] = {}
        self.deterministic: List[bool] = []
        self.missing: Dict[str, str] = {}
        for name in sorted(set(names if names is not None else list_specialists()) | set(given) | set(configs)):
            network = given.get(name)
            argmax = False
            if network is None:
                try:
                    path = overrides.get(name) or promoted_checkpoint(name, models_dir)
                    with torch.random.fork_rng():   # building the network draws init weights; keep seeds intact
                        network, _ = load_model(path)
                except FileNotFoundError as err:
                    self.missing[name] = str(err)
                    continue
                self.checkpoints[name] = str(path)
                argmax = act_mode(path) == 'argmax'
            self.names.append(name)
            self.configs.append(configs.get(name) or load_specialist(name))
            self.policies.append(network.to_native())
            self.deterministic.append(argmax)


class ChainPool:
    """An ``EnvPool`` of chain runs of one route (one run per seed, shared by ``num_envs`` envs) and the policies
    acting in it.  ``run()`` plays every seed; watchers call ``play()`` and follow ``poll_events()`` themselves.
    ``stall_steps`` (watch player only; None = off): room events carry ``stalled`` after that many actions in a
    room without damage dealt or taken, and the run ends ``stuck`` at twice as many.  ``hold_finished_frame``
    (watch player): a finished run's last frame stays published, and its env waits, until
    ``pool.set_hold(env, False)``.  ``plays``: write every run to this replay file (``triforce.replay``).
    ``deterministic``: None = each policy's act mode (``Policies.deterministic``), else argmax (True) or sampling
    (False) for every policy."""

    def __init__(self, route: Route, seeds: Sequence[int], policies: Policies, num_envs: int = 1,
                 threads: Optional[int] = None, seed: int = 0, fps: float = 0.0,
                 deterministic: Optional[bool] = None,
                 max_steps: int = DEFAULT_MAX_STEPS, states_dir=None, record: bool = False,
                 stall_steps: Optional[int] = None, hold_finished_frame: bool = False, plays=None):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.route = route
        self.policies = policies
        self.deterministic = (list(policies.deterministic) if deterministic is None
                              else [bool(deterministic)] * len(policies.policies))
        start_from = None if route.start_state == 'start' else route.start_state
        states_dir = None if states_dir is None else str(states_dir)
        task = _native.ChainTask(policies.configs, route.name, [int(s) for s in seeds], stop_at=route.terminal,
                                 start_from=start_from, max_steps=max_steps, missing=policies.missing,
                                 states_dir=states_dir, stall_steps=stall_steps,
                                 hold_finished_frame=hold_finished_frame)
        num_envs = max(1, min(num_envs, len(seeds)))
        self.pool = _native.EnvPool(task, num_envs=num_envs, threads=threads, seed=seed, fps=fps, record=record)
        self.writer = None
        if plays is not None:
            task = {'configs': [config.to_json() for config in policies.configs], 'route': route.name,
                    'start_from': start_from, 'stop_at': location_text(route.terminal), 'max_steps': max_steps,
                    'stall_steps': stall_steps, 'missing': policies.missing}
            models = models_header(policies.names, policies.policies, policies.checkpoints, self.deterministic)
            self.writer = ReplayWriter(plays, header('chain', task, models, states_dir))
        self._plays: Dict[int, dict] = {}
        self._results: Dict[int, dict] = {}

    def play(self):
        """Every env starts acting on its own (``EnvPool.play``)."""
        self.pool.play(self.policies.policies, deterministic=self.deterministic,
                       record=None if self.writer is None else 'actions')

    def poll_events(self) -> List[dict]:
        """The pool's events since the last call; with ``plays`` set, writes each finished run's replay line."""
        events = self.pool.poll_events()
        if self.writer is not None:
            self._plays.update((int(play['job']), play) for play in self.pool.poll_plays())
            for event in events:
                if event['type'] == 'run' and event['result']['steps'] > 0:
                    self._results[int(event['result']['seed'])] = event['result']
            for seed in sorted(self._plays.keys() & self._results.keys()):
                self.writer.write(self._plays.pop(seed), self._results.pop(seed))
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
        """Stops the pool's worker threads and closes the replay file (idempotent)."""
        self.pool.close()
        if self.writer is not None:
            self.writer.close()

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
                 states_dir=None):
        """
        checkpoints: specialist name → checkpoint path, overriding the registry's promoted model.
        specialist_configs: specialist name → loaded config, replacing its canonical YAML at runtime.
        networks: specialist name → a ``ZeldaPolicy`` to act with instead of a checkpoint.
        parallel: envs in the chain pool; threads: its worker threads (default min(parallel, cores)).
        deterministic: None = each model's act mode (registry), else argmax (True) or sampling (False) for all.
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.policies = Policies(models_dir, checkpoints, specialist_configs, networks)
        self.deterministic = deterministic
        self.max_steps = max_steps
        self.parallel = parallel
        self.threads = threads
        self.states_dir = states_dir

    def pool(self, route: Route, seeds: Sequence[int], **kwargs) -> ChainPool:
        """A chain pool over this runner's policies (``kwargs``: ``num_envs``, ``threads``, ``fps``, ``seed``,
        ``record``, ``plays``)."""
        kwargs.setdefault('num_envs', self.parallel)
        kwargs.setdefault('threads', self.threads)
        return ChainPool(route, seeds, self.policies, deterministic=self.deterministic, max_steps=self.max_steps,
                         states_dir=self.states_dir, **kwargs)

    def run_many(self, route: Route, seeds: Sequence[int],
                 on_result: Optional[Callable[[RunResult], None]] = None, plays=None) -> List[RunResult]:
        """Plays route once per seed; results sorted by seed.  Never raises for gameplay failures.  ``plays``: a
        replay file to write."""
        seeds = list(seeds)
        if not seeds:
            return []
        with self.pool(route, seeds, plays=plays) as pool:
            return pool.run(on_result)

    def run(self, route: Route, seed: Optional[int] = None) -> RunResult:
        """Plays route once (seed None: 0)."""
        return self.run_many(route, [0 if seed is None else seed])[0]


def parse_location(text: str) -> MapLocation:
    """'2:0x0e', '0:0x77:cave' → MapLocation."""
    parts = text.split(':')
    return MapLocation(int(parts[0]), int(parts[1], 16), len(parts) > 2 and parts[2] == 'cave')


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
