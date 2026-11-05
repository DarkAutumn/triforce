"""Go-Explore phase A: find one successful path per seed start state by restoring savestates.

    python -m triforce.explore --specialist NAME [--states 0_38s_0,0_48s_0] [--iterations 3000]
        [--explore-steps 25] [--patience 300] [--parallel N] [--seed 0] [--out trajectories]

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
explorer seeded with ``--seed``, so results don't depend on ``--parallel`` (threads).
"""

import argparse
from typing import List, Optional

from ._native import AcceptSpec, Explorer, ExploreResult, explore_states
from .paths import TRAJECTORIES_DIR

__all__ = ['AcceptSpec', 'Explorer', 'ExploreResult', 'explore_specialist', 'main']


def explore_specialist(specialist: str, out: str, states: Optional[List[str]] = None, iterations: int = 3000,
                       explore_steps: int = 25, patience: int = 300, parallel: int = 1, seed: int = 0,
                       states_dir: Optional[str] = None) -> list:
    """Explores every seed start state (or ``states``) of ``specialist``, writing successes under
    ``<out>/<specialist>/<state>``; returns [(ExploreResult, replayed)] in state order."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    return explore_states(specialist, str(out), states=states, iterations=iterations, explore_steps=explore_steps,
                          patience=patience, parallel=parallel, seed=seed,
                          states_dir=None if states_dir is None else str(states_dir))


def main(argv=None):
    """Command line."""
    parser = argparse.ArgumentParser(prog='python -m triforce.explore', description=__doc__.split('\n\n')[0])
    parser.add_argument('--specialist', required=True)
    parser.add_argument('--states', default=None, help="comma-separated seed states (default: all)")
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--explore-steps', type=int, default=25)
    parser.add_argument('--patience', type=int, default=300, help="iterations to keep improving after a success")
    parser.add_argument('--parallel', type=int, default=1, help="explorer threads")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=str(TRAJECTORIES_DIR),
                        help="trajectories root (<out>/<specialist>/<state> is written)")
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
