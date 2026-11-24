"""Go-Explore phase A: find one successful path per seed start state by restoring savestates.

    triforce job run --agent A --task T --subtask S --desc '...' --timeout 40m -- \\
        python -m triforce.explore --specialist NAME [--states 0_38s_0,0_48s_0] [--iterations 3000]
        [--explore-steps 25] [--patience 300] [--parallel N] [--seed 0] [--variant] [--room-kills 5:0x05=4]
        [--out trajectories]

A job of the machine's queue (``jobs.require_job``: outside ``triforce job`` and without a terminal it refuses);
the states explored so far of the total are the job's progress (``triforce job list``, the training GUI).

Plain PPO only learns what a random-ish policy stumbles into.  Here the emulator is a simulator we can rewind
(Ecoffet et al., Go-Explore): keep an archive of *cells*, coarse summaries of the game state, each holding the
exact session snapshot of the shortest path found to it.  Each iteration picks a cell, restores it and takes up to
``--explore-steps`` random legal actions; every new cell, or a known cell reached in fewer steps, is archived.
Episode endings come from the specialist's own env (no RAM randomization), so "success" is exactly what training
rewards.

Cell: (room, Link's tile // 2, whole hearts, living enemies, their health, keys, boomerang).  Selection weight:
``1/sqrt(1 + times chosen) × (1 + progress) × (1 + closeness)``: progress counts rooms entered, kills and
key/boomerang pickups on the cell's path; closeness = 8 / (8 + PBRS wavefront distance to the objective).  The
search stops ``--patience`` iterations after the first success (or at ``--iterations``).  The best success (most
health, then fewest steps) is replayed from the start and written as ``<out>/<specialist>/<state>/NNNN.state``
(state before step NNNN; 0000 = room entry) and ``trajectory.json``.  ``train --curriculum <out>/<specialist>``
then runs the backward curriculum on them.

The search itself is native (``triforce._native.explore_states`` / ``Explorer``); each start state gets a fresh
explorer seeded with ``--seed``, so results don't depend on ``--parallel`` (threads).  It reports the states explored
(``progress``), which ``main`` passes on as the job's progress.

``--variant`` writes ``<out>/<specialist>/<state>~s<seed>`` instead: an extra track from the same seed state.
The seed also draws the reset's RNG and spawn, so another seed is another room (for Dodongo, another movement
pattern); the curriculum keeps every track directory as its own track.

``--room-kills LEVEL:0xRR=K`` (round 24) explores a staged start: K kills are banked in that dungeon room before its
entry (``LevelKillCounts``), so the game makes K fewer monsters, and the track is ``<state>~k<K>`` (the Olive
Garden's six Blue Darknuts are beyond the explorer; training stages the same rooms with ``ram_randomization:
room_kills``).  ``--clips off`` explores with the run-time clips switch off: ``<state>~plain`` (5:0x64's fight and
push, a room whose one route step both ways share).
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

from . import jobs
from ._native import AcceptSpec, Explorer, ExploreResult, explore_states
from .paths import TRAJECTORIES_DIR, scratch
from .specialist import load_specialist

__all__ = ['AcceptSpec', 'Explorer', 'ExploreResult', 'explore_specialist', 'main']


def family_name(specialist: str) -> str:
    """The folder a specialist's tracks are written under: its config's name (``specialists/d3first/moldorm.yaml``
    writes ``<out>/moldorm``, as the native explorer does)."""
    return load_specialist(specialist).name


def check_out(family: str, out) -> None:
    """Raises ValueError when ``out`` is a family's track directory instead of the trajectories root (the
    explorer appends ``<family>/``, so ``trajectories/<family>`` would write tracks no training loads)."""
    path = Path(out)
    if path.name == family or any(path.glob('*/trajectory.json')):
        raise ValueError(f"--out {out} is a track directory; pass the trajectories root (e.g. {path.parent}): "
                         f"tracks are written to <out>/{family}/<state>")


def track_dir(out, specialist: str, state: str, seed: int, variant: bool, kills: Optional[int] = None,
              plain: bool = False) -> Path:
    """Where the track of ``state`` is written."""
    name = (state if kills is None else f'{state}~k{kills}') + ('~plain' if plain else '')
    return Path(out).absolute() / specialist / (f'{name}~s{seed}' if variant else name)


def parse_room_kills(text: str) -> tuple:
    """'5:0x05=4' -> (5, 0x05, 4)."""
    room, _, kills = text.partition('=')
    level, _, number = room.partition(':')
    try:
        return int(level), int(number, 16), int(kills)
    except ValueError as error:
        raise ValueError(f"--room-kills {text!r}: expected LEVEL:0xRR=K") from error


def explore_specialist(specialist: str, out: str, states: Optional[List[str]] = None, iterations: int = 3000,
                       explore_steps: int = 25, patience: int = 300, parallel: int = 1, seed: int = 0,
                       states_dir: Optional[str] = None, variant: bool = False,
                       room_kills: Optional[tuple] = None, clips: Optional[bool] = None,
                       progress: Optional[Callable[[int, int], None]] = None) -> list:
    """Explores every seed start state (or ``states``) of ``specialist``, writing successes under
    ``<out>/<specialist>/<state>`` (``<state>~s<seed>`` with ``variant``); returns [(ExploreResult, replayed)] in
    state order.  ``out`` must be the trajectories root (``check_out``).  ``progress(done, total)``: the states
    explored so far, as they finish."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    family = family_name(specialist)
    check_out(family, out)

    def run(root: str) -> list:
        return explore_states(specialist, root, states=states, iterations=iterations, explore_steps=explore_steps,
                              patience=patience, parallel=parallel, seed=seed,
                              states_dir=None if states_dir is None else str(states_dir),
                              room_kills=None if room_kills is None else [room_kills], clips=clips,
                              progress=progress)

    if not variant:
        return run(str(out))
    with tempfile.TemporaryDirectory() as tmp:
        results = run(tmp)
        source = Path(tmp) / family              # the explorer creates it only when a state was solved
        tracks = sorted(source.iterdir()) if source.is_dir() else []
        target_root = Path(out) / family
        for track in tracks:
            target_root.mkdir(parents=True, exist_ok=True)
            target = target_root / f'{track.name}~s{seed}'
            shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(track), target)
    return results


def main(argv=None):
    """Command line; exits 1 unless every state was solved and its replay matched."""
    parser = argparse.ArgumentParser(prog='python -m triforce.explore', description=__doc__.split('\n\n')[0])
    parser.add_argument('--specialist', required=True)
    parser.add_argument('--states', default=None, help="comma-separated seed states (default: all)")
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--explore-steps', type=int, default=25)
    parser.add_argument('--patience', type=int, default=300, help="iterations to keep improving after a success")
    parser.add_argument('--parallel', type=int, default=1, help="explorer threads")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--variant', action='store_true',
                        help="write <state>~s<seed>: an extra track beside the seed state's own")
    parser.add_argument('--clips', choices=('on', 'off'), default=None,
                        help="the run-time clips switch (default: the start state's way, else on); off writes "
                             "<state>~plain, a plain-way track of a room both ways share (5:0x64)")
    parser.add_argument('--room-kills', default=None, metavar='LEVEL:0xRR=K',
                        help="a staged start: K kills banked in that dungeon room before its entry (track <state>~k<K>)")
    # Today's checkouts write into their trajectories/ as before; one without it (the store's snapshots are read-only)
    # writes into scratch, never a new trajectories/ that would then shadow the snapshot for every reader.
    out = TRAJECTORIES_DIR if TRAJECTORIES_DIR.exists() else scratch('trajectories')
    parser.add_argument('--out', default=str(out),
                        help="trajectories root (<out>/<specialist>/<state> is written; not <specialist>'s own dir; "
                             "default this checkout's trajectories/, else target/trajectories)")
    args = parser.parse_args(argv)
    try:
        check_out(family_name(args.specialist), args.out)
    except ValueError as error:
        parser.error(str(error))

    try:
        room_kills = parse_room_kills(args.room_kills) if args.room_kills else None
    except ValueError as error:
        parser.error(str(error))
    jobs.require_job()
    results = explore_specialist(args.specialist, args.out, args.states.split(',') if args.states else None,
                                 args.iterations, args.explore_steps, args.patience, args.parallel, args.seed,
                                 variant=args.variant, room_kills=room_kills,
                                 clips=None if args.clips is None else args.clips == 'on',
                                 progress=lambda done, total: jobs.progress(done, total, 'states'))
    for result, replayed in results:
        status = (f"success in {len(result.actions)} steps, health {result.health}"
                  + ("" if replayed else " (REPLAY MISMATCH)")) if result.success else "no success"
        kills = room_kills[2] if room_kills else None
        track = track_dir(args.out, family_name(args.specialist), result.start_state, args.seed, args.variant,
                          kills, args.clips == 'off')
        written = f"  -> {track}" if result.success and replayed else ''
        print(f"{result.start_state:8} {status:40} iterations={result.iterations} env_steps={result.env_steps} "
              f"cells={result.cells} {result.seconds:.0f}s{written}")
    total = sum(r.env_steps for r, _ in results)
    solved = sum(r.success and ok for r, ok in results)
    print(f"{solved}/{len(results)} states solved; {total} env steps")
    return 0 if solved == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
