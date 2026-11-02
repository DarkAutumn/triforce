"""Go-Explore phase A: find one successful path per seed start state by restoring savestates.

    python -m triforce.explore --specialist NAME [--states 0_38s,0_48s] [--iterations 3000] [--explore-steps 25]
        [--patience 300] [--parallel N] [--seed 0] [--out trajectories]

Plain PPO only learns what a random-ish policy stumbles into.  Here the emulator is a simulator we can rewind
(Ecoffet et al., Go-Explore): keep an archive of *cells*, coarse summaries of the game state, each holding the
exact ``GameSession.snapshot()`` of the shortest path found to it.  Each iteration picks a cell, restores it and
takes up to ``--explore-steps`` random legal actions; every new cell, or a known cell reached in fewer steps, is
archived.  Episode endings come from the specialist's own ``ZeldaEnv`` (no RAM randomization), so "success" is
exactly what training rewards.

Cell: (room, Link's tile // 2, whole hearts, living enemies, keys, boomerang).  Selection weight:
``1/sqrt(1 + times chosen) × (1 + progress) × (1 + closeness)``: progress counts rooms entered, kills and
key/boomerang pickups on the cell's path; closeness = 8 / (8 + PBRS wavefront
distance to the objective).  The search stops ``--patience`` iterations after the first success (or at
``--iterations``).  The best success (most health, then fewest steps) is replayed from the start and written as
``<out>/<specialist>/<state>/NNNN.state`` (gzip savestate before step NNNN; 0000 = room entry) and
``trajectory.json``.  ``train --curriculum <out>/<specialist>`` then runs the backward curriculum on them.
"""

import argparse
import gzip
import json
import math
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .actions import DIRECTIONAL_KINDS, DURATIONS, KINDS, TIMED_KINDS, Action
from .env import ZeldaEnv
from .rewards import SUCCESS
from .session import GameSession
from .specialist import RamRandomization, SpecialistConfig, load_specialist
from .start_states import state_label
from .zelda_enums import ActionKind, Direction

KIND_WEIGHTS = {ActionKind.MOVE: 6.0, ActionKind.WAIT: 1.0}      # every other legal kind: 1.0
REPEAT_PROBABILITY = 0.5


@dataclass
class Cell:
    """One archive entry."""
    snapshot: dict
    actions: List[Tuple[int, int, int]]
    progress: int               # progress events on the path (see progress_events)
    closeness: float
    health: float
    chosen: int = 0

    @property
    def weight(self) -> float:
        """Selection weight (see module doc)."""
        return (1 + self.progress) * (1 + self.closeness) / math.sqrt(1 + self.chosen)


@dataclass
class ExploreResult:
    """The outcome of exploring one start state."""
    start_state: str
    success: bool
    actions: List[Tuple[int, int, int]] = field(default_factory=list)
    health: float = 0.0
    ending: Optional[str] = None
    iterations: int = 0
    env_steps: int = 0
    cells: int = 0
    seconds: float = 0.0


def _int(value) -> int:
    return int(getattr(value, 'value', value))


def cell_key(state) -> tuple:
    """Coarse summary of a state for the archive."""
    link = state.link
    living = sum(1 for enemy in state.enemies if not enemy.is_dying)
    return (state.level, state.location, state.in_cave, link.tile.x // 2, link.tile.y // 2, int(link.health),
            living, link.keys, _int(link.boomerang))


def closeness(state) -> float:
    """8 / (8 + wavefront distance to the objective); 0 without a wavefront."""
    wavefront = getattr(state, 'pbrs_wavefront', None)
    distance = wavefront.get(state.link.tile) if wavefront is not None else None
    return 0.0 if distance is None else 8.0 / (8.0 + distance)


def progress_events(prev, curr) -> int:
    """Room entered, enemy killed, key or boomerang gained in one step: selection favours cells further along."""
    if prev.full_location != curr.full_location:
        return 1
    return (int(curr.room_kills > prev.room_kills) + int(curr.link.keys > prev.link.keys)
            + int(_int(curr.link.boomerang) > _int(prev.link.boomerang)))


def random_action(mask, rng: random.Random, previous: Optional[Action]) -> Action:
    """A random legal action; repeats the previous one with REPEAT_PROBABILITY when it is still legal."""
    if previous is not None and rng.random() < REPEAT_PROBABILITY and mask.is_allowed(previous):
        return previous
    kinds = [kind for kind in KINDS if mask.kind[kind]]
    kind = rng.choices(kinds, weights=[KIND_WEIGHTS.get(kind, 1.0) for kind in kinds], k=1)[0]
    direction = rng.choice(mask.directions(kind)) if kind in DIRECTIONAL_KINDS else Direction.NONE
    duration = rng.choice([d for i, d in enumerate(DURATIONS) if mask.duration[kind, i]]) \
        if kind in TIMED_KINDS else 0
    return Action(kind, direction, duration)


def explore_config(config: SpecialistConfig) -> SpecialistConfig:
    """The specialist's config without RAM randomization: explore the savestate as it is."""
    return config.model_copy(update={'ram_randomization': RamRandomization()})


class Explorer:
    """Go-Explore over one specialist's ZeldaEnv."""

    def __init__(self, config: SpecialistConfig, session: Optional[GameSession] = None, seed: int = 0):
        self.env = ZeldaEnv(explore_config(config), session=session, capture_dir=None)
        self.seed = seed
        self.rng = random.Random(seed)
        self._root: Optional[dict] = None       # snapshot at the start of the last explore() call

    def close(self):
        """Releases the env."""
        self.env.close()

    def explore(self, start_state: str, iterations: int = 3000, explore_steps: int = 25,
                patience: int = 300, accept: Optional[Callable[[object], bool]] = None) -> ExploreResult:
        """Searches for a successful path from start_state; returns the best one found.  ``accept(state)`` can
        reject successes (e.g. a correct exit into the wrong room): a rejected success ends the attempt as a
        failure."""
        # pylint: disable=too-many-locals
        started = time.time()
        env, session = self.env, self.env.session
        env.reset(seed=self.seed, options={'start_state': start_state})
        root = session.snapshot()
        archive: Dict[tuple, Cell] = {cell_key(session.state): Cell(root, [], 0, closeness(session.state),
                                                                    session.state.link.health)}
        best: Optional[Tuple[float, int, list, str]] = None      # (health, -steps, actions, ending)
        first_success = None
        env_steps = 0
        iteration = 0
        for iteration in range(1, iterations + 1):
            if first_success is not None and iteration - first_success > patience:
                break
            cells = list(archive.values())
            cell = self.rng.choices(cells, weights=[c.weight for c in cells], k=1)[0]
            cell.chosen += 1
            env.restore(cell.snapshot, start_state)
            actions, progress, previous = list(cell.actions), cell.progress, None
            for _ in range(explore_steps):
                previous = random_action(session.mask, self.rng, previous)
                before = session.state
                _, _, terminated, truncated, info = env.step(previous.to_indices())
                env_steps += 1
                actions.append(tuple(int(x) for x in previous.to_indices()))
                if terminated or truncated:
                    if info['outcome'] == SUCCESS and (accept is None or accept(session.state)):
                        candidate = (info['health'], -len(actions), actions, info['ending'])
                        if best is None or candidate[:2] > best[:2]:
                            best = candidate
                        first_success = first_success or iteration
                    break
                state = session.state
                progress += progress_events(before, state)
                key = cell_key(state)
                known = archive.get(key)
                if known is None or len(actions) < len(known.actions):
                    archive[key] = Cell(session.snapshot(), list(actions), progress, closeness(state),
                                        state.link.health, known.chosen if known else 0)

        result = ExploreResult(start_state, best is not None, iterations=iteration, env_steps=env_steps,
                               cells=len(archive), seconds=time.time() - started)
        if best is not None:
            result.health, result.actions, result.ending = best[0], best[2], best[3]
        self._root = root
        return result

    def write(self, result: ExploreResult, directory: str) -> bool:
        """Replays a successful result from the start, writing the savestate before every step.  Returns True if
        the replay ends in the same success (it should: restores are exact)."""
        env, session = self.env, self.env.session
        os.makedirs(directory, exist_ok=True)
        for stale in os.listdir(directory):
            if stale.endswith('.state') or stale == 'trajectory.json':
                os.remove(os.path.join(directory, stale))
        env.restore(self._root, result.start_state)
        info = {}
        for index, indices in enumerate(result.actions):
            with gzip.open(os.path.join(directory, f"{index:04d}.state"), 'wb') as file:
                file.write(session.emu.save_state())
            _, _, terminated, truncated, info = env.step(list(indices))
            if (terminated or truncated) and index != len(result.actions) - 1:
                break
        replayed = info.get('outcome') == SUCCESS and info.get('ending') == result.ending
        meta = {'start_state': result.start_state, 'success': replayed, 'ending': result.ending,
                'steps': len(result.actions), 'health': result.health,
                'actions': [str(Action.from_indices(*a)) for a in result.actions],
                'explore': {'iterations': result.iterations, 'env_steps': result.env_steps, 'cells': result.cells,
                            'seconds': round(result.seconds, 1), 'seed': self.seed}}
        with open(os.path.join(directory, 'trajectory.json'), 'w', encoding='utf-8') as file:
            json.dump(meta, file, indent=1)
        return replayed

    def replay(self, result: ExploreResult, capture_pre_entry: bool = False) -> Tuple[dict, Optional[bytes]]:
        """Replays a result from the start of the last explore() (exact restores, so it reproduces).  Returns the
        final step's info and, with capture_pre_entry, the savestate of the last frame before its room entry."""
        env, session = self.env, self.env.session
        env.restore(self._root, result.start_state)
        session.executor.capture_pre_entry = capture_pre_entry
        info = {}
        try:
            for indices in result.actions:
                _, _, terminated, truncated, info = env.step(list(indices))
                if terminated or truncated:
                    break
            return info, session.executor.last_pre_entry
        finally:
            session.executor.capture_pre_entry = False


_WORKER = {}


def _worker_init(config, seed):
    _WORKER['explorer'] = Explorer(config, seed=seed)


def _worker_run(job):
    state, out, iterations, explore_steps, patience = job
    explorer: Explorer = _WORKER['explorer']
    result = explorer.explore(state, iterations, explore_steps, patience)
    replayed = explorer.write(result, os.path.join(out, state)) if result.success else False
    return result, replayed


def explore_specialist(specialist: str, out: str, states: Optional[List[str]] = None, iterations: int = 3000,
                       explore_steps: int = 25, patience: int = 300, parallel: int = 1, seed: int = 0) -> list:
    """Explores every seed start state (or ``states``); returns [(ExploreResult, replayed)] in state order."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    config = load_specialist(specialist)
    states = states or sorted(state_label(p) for p in config.resolve_start_states())
    out = os.path.join(out, config.name)
    jobs = [(state, out, iterations, explore_steps, patience) for state in states]
    if parallel <= 1:
        _worker_init(config, seed)
        return [_worker_run(job) for job in jobs]
    with mp.get_context('spawn').Pool(min(parallel, len(jobs)), _worker_init, (config, seed)) as pool:
        return pool.map(_worker_run, jobs, chunksize=1)


def main(argv=None):
    """Command line."""
    parser = argparse.ArgumentParser(prog='python -m triforce.explore', description=__doc__.split('\n\n')[0])
    parser.add_argument('--specialist', required=True)
    parser.add_argument('--states', default=None, help="comma-separated seed states (default: all)")
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--explore-steps', type=int, default=25)
    parser.add_argument('--patience', type=int, default=300, help="iterations to keep improving after a success")
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='trajectories')
    args = parser.parse_args(argv)

    results = explore_specialist(args.specialist, args.out, args.states.split(',') if args.states else None,
                                 args.iterations, args.explore_steps, args.patience, args.parallel, args.seed)
    for result, replayed in results:
        status = (f"success in {len(result.actions)} steps, health {result.health}"
                  + ("" if replayed else " (REPLAY MISMATCH)")) if result.success else "no success"
        print(f"{result.start_state:8} {status:40} iterations={result.iterations} env_steps={result.env_steps} "
              f"cells={result.cells} {result.seconds:.0f}s")
    total = sum(r.env_steps for r, _ in results)
    print(f"{sum(r.success and ok for r, ok in results)}/{len(results)} states solved; {total} env steps")
    return 0


if __name__ == '__main__':
    main()
