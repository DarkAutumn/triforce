"""Round 27 §13 (docs/experiments/round27.md "The planner as a teacher"): the honest planner's decisions in
Dungeon 6's two block rooms as demonstration rows for ``train --demos``.

    python scripts/experiments/r27_plan_demos.py CHECKPOINT OUT.npz [--family F|YAML] [--act MODE] [--runs N]
        [--states 6_38s 6_3aw] [--state-files FILE...] [--route R]

The family (``--family``, wizzrobe by default; round 29 records level 9's rooms of wizzrobe, patra and ganon too)
plays ``--runs`` episodes from every chain of each state stem (``6_38s_0..2``, ``6_3aw_0..2``)
in a native ``EvalTask`` pool, its own hp and bombs draws, full rooms (``room_kills`` is training only), the act mode
``registry.PLAN_ACT`` (``temperature:0.5+relax:16+plan:16:1``: before each decision the emulator tries the action on
redrawn random numbers and takes the policy's next best if Link would be hit; docs/rust-actor.md "Foresight"). The
pool records every decision's observation (with the action mask) and the action it executed, the planner's choice
(``EnvPool.play(record='observations')``, as ``triforce.best_of_n``).  ``--act`` plays another act mode instead
(round 27's ow-lynel: ``registry.ACT_RULE``, a model's own clean runs as demonstrations; nothing is replaced then).
``--state-files`` adds start states played as named, ``--runs`` each (round 29's ow-zora: harvested state files,
counted under their stem).  ``--route`` names the stems' route (round 32: ``--route d3first --states 3_6bs`` plays
``d3first_3_6bs_0..2``, the route's start-state namespace; d2first's names are bare).  ``--family`` may be a yaml path
(round 32: a config copy with ``state_routes: [d3first]`` and ``chain_hp: 1.0``, the hearts the route's chains bring).

Rows: every decision of the episodes that succeed with at most ``--max-loss`` hearts lost; a decision the planner
replaced (``decisions.foresight.replaced`` of the episode's visit: the policy's draw would have been hit) is written
``--repeat`` times, since it is the lesson and a few among a hundred. 0x28's bomb stays ``bomb28.npz``
(``r27_d6_rules.py``), the planner's 0x28 (it walks back out and in) is not recorded.

It plays a model, so it is a job: it waits its turn in the machine's line (``triforce.jobs``) with a log folder
(``events.logged_job``) and lowers its own priority, like ``triforce evaluate``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from triforce import _native, registry
from triforce.best_of_n import play_episodes
from triforce.demos import save_demos
from triforce.events import logged_job
from triforce.jobs import add_job_arguments, begin_job, work
from triforce.model import MASK_KEY
from triforce.specialist import load_specialist

DEFAULT_FAMILY = 'wizzrobe'


def replaced_indices(record: dict) -> set[int]:
    """Decision indices of the episode the planner replaced (a one-room episode: its single visit's)."""
    out = set()
    offset = 0
    for visit in record['visits']:
        foresight = (visit.get('decisions') or {}).get('foresight') or {}
        out.update(offset + int(entry[0]) for entry in foresight.get('replaced', []))
        offset += int((visit.get('decisions') or {}).get('n', 0))
    return out


def state_names(stems: list[str], route: str | None = None) -> list[str]:
    """The start states of ``stems``, chains 0-2, in ``route``'s start-state namespace (``d3first_3_6bs_0``; a route
    without one, d2first, and no route: the bare ``3_6bs_0``)."""
    namespace = _native.route_namespace(route) if route else None
    prefix = f'{namespace}_' if namespace else ''
    return [f'{prefix}{stem}_{chain}' for stem in stems for chain in (0, 1, 2)]


def record_demos(family: str, checkpoint: Path, out: Path, stems: list[str], runs: int, max_loss: float,
                 repeat: int, parallel: int, seed: int, act: str = registry.PLAN_ACT,
                 state_files: list[str] | None = None, route: str | None = None,
                 file_runs: int | None = None) -> int:
    """Plays, filters and writes the rows; returns how many."""
    config = load_specialist(family)
    policy = registry.load_model(checkpoint)[0].to_native()
    file_runs = runs if file_runs is None else file_runs
    starts = [s for s in state_names(stems, route) for _ in range(runs)]
    starts += [s for s in state_files or [] for _ in range(file_runs)]
    jobs = [(state, seed + index, True) for index, state in enumerate(starts)]
    task = _native.EvalTask(config, jobs)
    records, plays = {}, {}
    with _native.EnvPool(task, num_envs=max(1, min(parallel, len(jobs))), seed=seed) as pool:
        pool.play([policy], act=act, record='observations')
        while True:
            idle = pool.wait_idle(timeout=0.5)
            for event in pool.poll_events():
                if event['type'] == 'eval':
                    records[int(event['record']['seed'])] = event['record']
            plays.update((int(play['job']), play) for play in pool.poll_plays())
            if idle:
                break
    observations, actions = [], []
    by_state: dict[str, list] = {}
    for key, record in sorted(records.items()):
        lost = record['start_health'] - record['health']
        play = plays[key]
        replaced = replaced_indices(record)
        n = len(play['actions'])
        if len(replaced) and max(replaced) >= n:
            raise ValueError(f"seed {key}: replaced decision {max(replaced)} past the play's {n} decisions")
        kept = bool(record['success']) and lost <= max_loss
        # a harvested state file (``--state-files``) counts under its stem (``0_44s_h100_0`` -> ``0_44s``)
        label = Path(record['start_state']).name.removesuffix('.state').split('_h')[0]
        stats = by_state.setdefault(label, [0, 0, 0, 0, 0.0, 0])
        stats[0] += 1
        stats[1] += bool(record['success'])
        stats[4] += lost if record['success'] else 0.0
        if not kept:
            continue
        stats[2] += 1
        stats[3] += len(replaced)
        stats[5] += n + (repeat - 1) * len(replaced)
        for index in range(n):
            row = {k: np.asarray(v[index]) for k, v in play['obs'].items()}
            for _ in range(repeat if index in replaced else 1):
                observations.append(row)
                actions.append(tuple(int(a) for a in play['actions'][index]))
    for state, (episodes, wins, kept, swaps, lost, rows) in sorted(by_state.items()):
        print(f"{state}: {wins} of {episodes} won ({lost / max(wins, 1):.2f} hearts lost in the wins); kept {kept} "
              f"(at most {max_loss} hearts lost), {swaps} replaced decisions in them, {rows} rows")
    if not observations or MASK_KEY not in observations[0]:
        raise ValueError("no rows (or no action mask in the recorded observations)")
    save_demos(out, observations, actions)
    print(f"{out}: {len(actions)} rows")
    return len(actions)


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path, help="the family's run dir (or ckpt.pt) the planner plays")
    parser.add_argument('out', type=Path)
    parser.add_argument('--family', default=DEFAULT_FAMILY, help="the family's name or a yaml path (a config copy)")
    parser.add_argument('--act', default=registry.PLAN_ACT, help="the act mode played (default the planner's)")
    parser.add_argument('--states', nargs='*', default=['6_38s', '6_3aw'], help="start state stems (chains 0-2)")
    parser.add_argument('--state-files', nargs='+', default=[],
                        help="more start states played as named (e.g. harvested state files), --file-runs each")
    parser.add_argument('--runs', type=int, default=100, help="episodes a start state")
    parser.add_argument('--file-runs', type=int, default=None, help="episodes a --state-files state (default --runs)")
    parser.add_argument('--route', default=None,
                        help="the stems' route: its start-state namespace prefixes them (default: bare names)")
    parser.add_argument('--max-loss', type=float, default=6.0, help="hearts a kept episode may lose")
    parser.add_argument('--repeat', type=int, default=5, help="rows a replaced decision is written")
    parser.add_argument('--parallel', type=int, default=8)
    parser.add_argument('--seed', type=int, default=5000)
    add_job_arguments(parser)
    args = parser.parse_args(argv)
    nice = begin_job(args)
    file_runs = args.runs if args.file_runs is None else args.file_runs
    episodes = 3 * len(args.states) * args.runs + len(args.state_files) * file_runs
    name = load_specialist(args.family).name
    with logged_job('r27-plan-demos', 'eval', [{'specialist': name, 'run_id': args.checkpoint.name,
                                                 'status': 'evaluating'}], nice=nice,
                    work=work(eval_episodes=2 * episodes)):
        record_demos(args.family, args.checkpoint, args.out, args.states, args.runs, args.max_loss, args.repeat,
                     args.parallel, args.seed, args.act, args.state_files, args.route, file_runs)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
