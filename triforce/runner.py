"""Plays a route end to end with trained specialists (design §9).

The chain itself runs natively (``ChainTask`` in an ``EnvPool``): on every room entry the room's game.yaml
``specialist:`` tag picks who acts, specialists switch only on room transitions, and a run ends with a
``RunResult``.  This module owns the models: each pool row names its acting specialist (the ``policy`` column, an
index into the loaded-model list).  Envs run asynchronously; each specialist's ready rows go through a fixed-shape
forward on ``device`` (``inference.PoolInference``), and sampled actions use Gumbel noise keyed to (run seed,
decision), so a seed plays the same run whichever env or thread runs it.  A specialist without a promoted model
ends the runs that need it as ``error`` with the registry's reason; an untagged room the route needs is an error
too, a wrong exit into one a gameplay failure.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import _native
from .inference import PoolInference
from .registry import load_model, promoted_checkpoint
from .specialist import SpecialistConfig, list_specialists, load_specialist
from .zelda_enums import MapLocation

DEFAULT_MAX_STEPS = 50_000

RoomKey = Tuple[int, int, bool]


class RunnerError(RuntimeError):
    """The runner can't start: unknown or malformed route."""


@dataclass
class Route:
    """Where a route starts and which room ends it."""
    name: str
    start_state: str
    terminal: MapLocation

    @staticmethod
    def load(name: str, stop_at: Optional[MapLocation] = None, start_from: Optional[str] = None) -> 'Route':
        """Loads a route from routes.yaml.  ``stop_at`` replaces the terminal room and ``start_from`` (a savestate,
        e.g. a route entry state like ``0_37c_2``) the game start, for partial chain evals."""
        try:
            steps = _native.route_steps(name)
        except _native.ConfigError as err:
            raise RunnerError(str(err)) from err
        if not steps:
            raise RunnerError(f"Unknown route '{name}'")
        if steps[0]['entry'] != 'start':
            raise RunnerError(f"Route '{name}' must begin at the game start (entry: start)")
        terminals = [s for s in steps if s['exit'] == 'terminal']
        if stop_at is None and not terminals:
            raise RunnerError(f"Route '{name}' has no terminal room")
        last = terminals[-1] if terminals else steps[-1]
        terminal = stop_at or MapLocation(last['level'], last['room'], last['in_cave'])
        return Route(name, start_from or 'start', terminal)


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
    outcome: str = 'in-progress'    # exit-correct, exit-wrong, death, stuck, milestone, error
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


@dataclass
class RunResult:
    """Everything one run produced; written as JSON lines by ``write_jsonl``."""
    route: str
    seed: Optional[int]
    result: str = 'in-progress'     # success, failure, error
    reason: str = ''
    steps: int = 0
    frames: int = 0
    rooms: List[RoomRecord] = field(default_factory=list)

    @property
    def failed_room(self) -> Optional[RoomRecord]:
        """The room the run ended in, if it didn't succeed."""
        return self.rooms[-1] if self.rooms and self.result != 'success' else None

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
    """The specialists' networks for a chain pool, in policy-index order, plus why the others are missing.

    ``checkpoints`` (name -> checkpoint file or run dir) overrides the registry's promoted model; ``networks``
    (name -> anything with ``get_action(obs, deterministic=...)``) supplies a loaded policy directly;
    ``specialist_configs`` (name -> config) replaces a specialist's canonical YAML at runtime; ``device`` is where
    the loaded networks run."""

    def __init__(self, models_dir=None, checkpoints: Optional[Dict[str, str]] = None,
                 specialist_configs: Optional[Dict[str, SpecialistConfig]] = None,
                 networks: Optional[Dict[str, object]] = None, device: str = 'cpu',
                 names: Optional[Iterable[str]] = None):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        overrides = dict(checkpoints or {})
        configs = dict(specialist_configs or {})
        given = dict(networks or {})
        self.device = device
        self.names: List[str] = []
        self.configs: List[SpecialistConfig] = []
        self.networks: list = []
        self.missing: Dict[str, str] = {}
        for name in sorted(set(names if names is not None else list_specialists()) | set(given) | set(configs)):
            network = given.get(name)
            if network is None:
                try:
                    path = overrides.get(name) or promoted_checkpoint(name, models_dir)
                    with torch.random.fork_rng():   # building the network draws init weights; keep seeds intact
                        network, _ = load_model(path)
                except FileNotFoundError as err:
                    self.missing[name] = str(err)
                    continue
                network.eval()
                network.to(device)
            self.names.append(name)
            self.configs.append(configs.get(name) or load_specialist(name))
            self.networks.append(network)


class ChainPool:
    """An ``EnvPool`` of chain runs of one route (one run per seed, shared by ``num_envs`` envs) and the policies
    acting in it.  ``run()`` plays every seed asynchronously; watchers drive ``recv`` / ``actions`` / ``send``
    themselves."""

    def __init__(self, route: Route, seeds: Sequence[int], policies: Policies, num_envs: int = 1,
                 threads: Optional[int] = None, seed: int = 0, fps: float = 0.0, deterministic: bool = False,
                 max_steps: int = DEFAULT_MAX_STEPS, states_dir=None, record: bool = False):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.route = route
        self.policies = policies
        self.deterministic = deterministic
        start_from = None if route.start_state == 'start' else route.start_state
        task = _native.ChainTask(policies.configs, route.name, [int(s) for s in seeds], stop_at=route.terminal,
                                 start_from=start_from, max_steps=max_steps, missing=policies.missing,
                                 states_dir=None if states_dir is None else str(states_dir))
        num_envs = max(1, min(num_envs, len(seeds)))
        self.pool = _native.EnvPool(task, num_envs=num_envs, threads=threads, seed=seed, fps=fps, record=record)
        self.buffers = self.pool.make_buffers()
        self.inference = PoolInference(policies.networks, num_envs, policies.device, deterministic)

    def recv(self, min_ready: Optional[int] = None, timeout: Optional[float] = None) -> int:
        """Rows ready (see ``EnvPool.recv``); they are in ``buffers``."""
        return self.pool.recv(self.buffers, min_ready=min_ready, timeout=timeout)

    def live_rows(self, count: int) -> np.ndarray:
        """Indices of the first ``count`` rows that await an action (not finished)."""
        return np.flatnonzero(~self.buffers['finished'][:count])

    def actions(self, rows: np.ndarray) -> np.ndarray:
        """Actions for buffer rows ``rows`` (see ``inference.PoolInference.actions``)."""
        b = self.buffers
        obs = {key: value[rows] for key, value in b['obs'].items()}
        return self.inference.actions(b['env_ids'][rows], obs, b['policy'][rows], b['job'][rows], b['step'][rows])

    def send(self, env_ids, actions):
        """Queues one action per env."""
        self.pool.send(env_ids, actions)

    def results(self, events: Iterable[dict]) -> List[Tuple[int, RunResult]]:
        """(env, RunResult) of every ``run`` event in ``events``."""
        return [(e['env'], RunResult.from_dict(e['result'])) for e in events if e['type'] == 'run']

    def run(self, on_result: Optional[Callable[[RunResult], None]] = None) -> List[RunResult]:
        """Plays every seed (each ready env acts at once); returns the results sorted by seed."""
        results: List[RunResult] = []
        while self.pool.active:
            count = self.recv(min_ready=1)
            rows = self.live_rows(count)
            if len(rows):
                self.send(self.buffers['env_ids'][rows], self.actions(rows))
            for _, result in self.results(self.pool.poll_events()):
                results.append(result)
                if on_result is not None:
                    on_result(result)
        for _, result in self.results(self.pool.poll_events()):
            results.append(result)
            if on_result is not None:
                on_result(result)
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

    def __init__(self, models_dir=None, checkpoints: Optional[Dict[str, str]] = None, deterministic: bool = False,
                 max_steps: int = DEFAULT_MAX_STEPS, device: str = 'cpu',
                 specialist_configs: Optional[Dict[str, SpecialistConfig]] = None,
                 networks: Optional[Dict[str, object]] = None, parallel: int = 1, threads: Optional[int] = None,
                 states_dir=None):
        """
        checkpoints: specialist name → checkpoint path, overriding the registry's promoted model.
        device: torch device for the specialists' networks ('cpu' or 'cuda').
        specialist_configs: specialist name → loaded config, replacing its canonical YAML at runtime.
        networks: specialist name → a loaded policy (``get_action(obs, deterministic=...)``), e.g. a scripted one.
        parallel: envs in the chain pool; threads: its worker threads (default min(parallel, cores)).
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.policies = Policies(models_dir, checkpoints, specialist_configs, networks, device)
        self.deterministic = deterministic
        self.max_steps = max_steps
        self.parallel = parallel
        self.threads = threads
        self.states_dir = states_dir

    def pool(self, route: Route, seeds: Sequence[int], **kwargs) -> ChainPool:
        """A chain pool over this runner's policies (``kwargs``: ``num_envs``, ``threads``, ``fps``, ``seed``,
        ``record``)."""
        kwargs.setdefault('num_envs', self.parallel)
        kwargs.setdefault('threads', self.threads)
        return ChainPool(route, seeds, self.policies, deterministic=self.deterministic, max_steps=self.max_steps,
                         states_dir=self.states_dir, **kwargs)

    def run_many(self, route: Route, seeds: Sequence[int],
                 on_result: Optional[Callable[[RunResult], None]] = None) -> List[RunResult]:
        """Plays route once per seed; results sorted by seed.  Never raises for gameplay failures."""
        seeds = list(seeds)
        if not seeds:
            return []
        with self.pool(route, seeds) as pool:
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

    ``arrival_keys`` counts successful runs by the keys Link carried into the milestone room (the Dodongo room on
    milestone1: 3 = every key on the route kept), so milestone arrivals and full key carriage stay separate
    counters."""
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
    return {'runs': runs, 'successes': successes, 'success_rate': successes / runs if runs else 0.0,
            'arrival_keys': dict(sorted(arrival_keys.items())),
            'failures': dict(sorted(failures.items(), key=lambda kv: -kv[1])),
            'errors': sorted({r.reason for r in results if r.result == 'error'})}
