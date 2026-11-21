"""Round 35 (docs/experiments/round35.md "The data"): the look-ahead's verdict on every candidate it tried, as rows for
a learned guard (``r35_guard_train.py``).

    python scripts/experiments/r35_guard_data.py CHECKPOINT OUT.npz --family YAML --route d3first
        --states 7_69s 7_59s ... --runs N [--model-runs M] [--audit 1] [--seed S] [--parallel 10]

The family plays ``--runs`` episodes from every chain of each stem (``d3first_7_69s_0..2``) in the honest planner's
act mode (``registry.PLAN_ACT``, ``temperature:0.5+relax:16+plan:16:1``: before each decision the emulator tries the
model's draw on redrawn random numbers and the next of the model's ranking while it would be hit), and
``--model-runs`` more in ``temperature:0.5+relax:16+plan:1:1`` (the model alone, decision for decision, its own
draw tried and never replaced: labels on the states the model reaches by itself). Every decision of every episode is
kept, won or lost: its observation (the keys the policy reads and the mask) and the action taken; and every
candidate the look-ahead judged (``EnvPool.poll_plays`` ``tried``): the decision, the joint action, the future and
whether Link lost health in it. ``--audit A`` (``TRIFORCE_PLAN_AUDIT``) tries every judged candidate on A more
redrawn futures that never decide (futures 1..A): the agreement of two redraws of one decision, the ceiling a guard
that sees no random numbers can reach.

Written (``.npz``): ``obs/<key>`` and ``mask`` per decision row, ``row_episode``, ``row_decision``, ``row_action``
``[N, 3]``; ``ep_seed``, ``ep_state``, ``ep_mode`` (0 planner, 1 model), ``ep_success``, ``ep_lost`` (hearts),
``ep_start_health``; ``tried_row`` (the decision row), ``tried_joint`` (``kind * 32 + direction * 4 + duration``),
``tried_future``, ``tried_lost``.

It plays a model, so it is a job: it waits its turn in the machine's line (``triforce.jobs``).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from triforce import _native, registry
from triforce.events import logged_job
from triforce.jobs import add_job_arguments, begin_job, work
from triforce.model import MASK_KEY
from triforce.specialist import load_specialist

MODEL_ACT = 'temperature:0.5+relax:16+plan:1:1'
DIRECTIONS, DURATIONS = 8, 4


def state_names(stems: list[str], route: str | None) -> list[str]:
    namespace = _native.route_namespace(route) if route else None
    prefix = f'{namespace}_' if namespace else ''
    return [f'{prefix}{stem}_{chain}' for stem in stems for chain in (0, 1, 2)]


def play(config, policy, starts: list[str], seed: int, act: str, parallel: int) -> tuple[dict, dict]:
    """Plays every start once with ``act``; (eval records, plays) by seed."""
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
            plays.update((int(p['job']), p) for p in pool.poll_plays())
            if idle:
                break
    return records, plays


def collect(family: str, checkpoint: Path, out: Path, stems: list[str], runs: int, model_runs: int, audit: int,
            parallel: int, seed: int, route: str | None) -> None:
    config = load_specialist(family)
    registry.require_compatible(checkpoint)
    network = registry.load_model(checkpoint)[0]
    keys = sorted(network.config.obs_spec)
    policy = network.to_native()
    os.environ['TRIFORCE_PLAN_AUDIT'] = str(audit)
    names = state_names(stems, route)
    obs = {k: [] for k in keys}
    rows = {'mask': [], 'row_episode': [], 'row_decision': [], 'row_action': []}
    eps = {'ep_seed': [], 'ep_state': [], 'ep_mode': [], 'ep_success': [], 'ep_lost': [], 'ep_start_health': []}
    tried = {'tried_row': [], 'tried_joint': [], 'tried_future': [], 'tried_lost': []}
    base = 0
    for mode, (act, n, offset) in enumerate([(registry.PLAN_ACT, runs, 0), (MODEL_ACT, model_runs, 1_000_000)]):
        if n == 0:
            continue
        starts = [s for s in names for _ in range(n)]
        records, plays = play(config, policy, starts, seed + offset, act, parallel)
        for key in sorted(records):
            record, p = records[key], plays[key]
            steps = len(p['actions'])
            episode = len(eps['ep_seed'])
            eps['ep_seed'].append(key)
            eps['ep_state'].append(Path(record['start_state']).name.removesuffix('.state'))
            eps['ep_mode'].append(mode)
            eps['ep_success'].append(bool(record['success']))
            eps['ep_lost'].append(record['start_health'] - record['health'])
            eps['ep_start_health'].append(record['start_health'])
            for k in keys:
                obs[k].append(np.asarray(p['obs'][k]))
            rows['mask'].append(np.asarray(p['obs'][MASK_KEY]).astype(bool))
            rows['row_episode'].append(np.full(steps, episode, dtype=np.int32))
            rows['row_decision'].append(np.arange(steps, dtype=np.int32))
            rows['row_action'].append(np.asarray(p['actions'], dtype=np.int8).reshape(-1, 3))
            t = p['tried']
            a = np.asarray(t['action'], dtype=np.int64).reshape(-1, 3)
            tried['tried_row'].append(base + np.asarray(t['decision'], dtype=np.int64))
            tried['tried_joint'].append(((a[:, 0] * DIRECTIONS + a[:, 1]) * DURATIONS + a[:, 2]).astype(np.int16))
            tried['tried_future'].append(np.asarray(t['future'], dtype=np.int8))
            tried['tried_lost'].append(np.asarray(t['lost'], dtype=bool))
            base += steps
        print(f"{act}: {len(records)} episodes, {sum(len(p['actions']) for p in plays.values())} decisions")
    arrays = {f'obs/{k}': np.concatenate(v) for k, v in obs.items()}
    if 'obs/grid' in arrays:
        arrays['obs/grid'] = arrays['obs/grid'].astype(np.uint8)
    arrays.update({k: np.concatenate(v) for k, v in rows.items()})
    arrays.update({k: np.concatenate(v) for k, v in tried.items()})
    arrays.update({k: np.asarray(v) for k, v in eps.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as file:
        np.savez_compressed(file, **arrays)
    lost = arrays['tried_lost']
    first = arrays['tried_future'] == 0
    print(f"{out}: {len(arrays['mask'])} decisions in {len(eps['ep_seed'])} episodes; {int(first.sum())} judged "
          f"candidates on future 0, {int(lost[first].sum())} lost ({lost[first].mean():.4f})")


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path, help="the family's run dir (or ckpt.pt)")
    parser.add_argument('out', type=Path)
    parser.add_argument('--family', required=True, help="the family's name or a yaml path (a config copy)")
    parser.add_argument('--states', nargs='+', required=True, help="start state stems (chains 0-2)")
    parser.add_argument('--route', default=None, help="the stems' route (its start-state namespace)")
    parser.add_argument('--runs', type=int, default=100, help="planner episodes a start state")
    parser.add_argument('--model-runs', type=int, default=0, help="model-alone (plan:1:1) episodes a start state")
    parser.add_argument('--audit', type=int, default=1, help="audit futures a judged candidate is also tried on")
    parser.add_argument('--parallel', type=int, default=10)
    parser.add_argument('--seed', type=int, default=35000)
    add_job_arguments(parser)
    args = parser.parse_args(argv)
    nice = begin_job(args)
    episodes = 3 * len(args.states) * (args.runs + args.model_runs)
    name = load_specialist(args.family).name
    with logged_job('r35-guard-data', 'eval', [{'specialist': name, 'run_id': args.checkpoint.name,
                                                 'status': 'evaluating'}], nice=nice,
                    work=work(eval_episodes=3 * episodes)):
        collect(args.family, args.checkpoint, args.out, args.states, args.runs, args.model_runs, args.audit,
                args.parallel, args.seed, args.route)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
