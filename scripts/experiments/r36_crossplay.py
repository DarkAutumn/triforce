"""Round 36's cross-play (docs/experiments/round36.md "Cross-play"): today's networks played in other families' rooms.

    python scripts/experiments/r36_crossplay.py specs OUT_DIR
        Every family's task on `d3first`'s start states, split by the route step's run: OUT_DIR/<family>-pre.json
        (the steps before the magic sword, played by `promoted.d3first` where a family has one) and
        <family>-post.json (the rest, `promoted`).  The family's config with `state_routes: [d3first]` and its own
        states listed by name; nothing else changes (rooms, masks, items, RAM randomization, max_steps).
    python scripts/experiments/r36_crossplay.py run GROUP SPECS_DIR OUT_DIR --pairs A:B[,B...] ... [--episodes 5]
        [--seed 100000] [--parallel 10] [--models-dir DIR] [--states-bare]
        Plays every (network A, task B) of the group in one process that waits its turn in the job line
        (triforce.jobs): A's run (pre: A's `promoted.d3first` else `promoted`; post: `promoted`) on each part of B's
        task, `--episodes` a start state, episode i seeded `seed + i` (so every network meets the same episodes of
        a task), A's own act mode.  Each result goes to OUT_DIR/<A>__<B>-<part>.json (records slimmed); an existing
        file is done (resume).  `A:all` plays A on every task in SPECS_DIR; `all:B` every family's network on B.
    python scripts/experiments/r36_crossplay.py check FAMILY RUN OUT --episodes N [--seed 0]
        The pipeline's check: RUN on FAMILY's own config restricted to the bare states its main eval.json holds.
    python scripts/experiments/r36_crossplay.py report EVALS_DIR OUT_JSON
        The matrix: per (A, B) success, hearts lost a room and hits a room, with B's own network on the same
        episodes (paired difference and its standard error, clustered by start state).
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# pylint: disable=wrong-import-position
from triforce import _native, registry
from triforce.clips import episode_jobs
from triforce.entry_states import route_steps
from triforce.evaluate import evaluate_specialist
from triforce.events import logged_job
from triforce.jobs import work
from triforce.specialist import list_specialists, load_specialist

ROUTE = 'd3first'
PARTS = ('pre', 'post')


def families() -> list[str]:
    return sorted(list_specialists())


def cmd_specs(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    state_steps = defaultdict(list)
    for step in route_steps(ROUTE):
        for chain in range(3):
            state_steps[step.filename(chain)[:-len('.state')]].append(step.index)
    for family in families():
        config = load_specialist(family)
        states = sorted({job[0] for job in episode_jobs(config, 1, 0, None)})
        parts = {part: [] for part in PARTS}
        for state in states:
            if not state.startswith(f'{ROUTE}_'):
                continue
            steps = state_steps.get(state)
            if not steps:
                print(f'{family}: {state} is no step of {ROUTE}; left out')
                continue
            pre = {'@' in _native.step_policy(ROUTE, index, family) for index in steps}
            if len(pre) > 1:
                raise ValueError(f'{state} on both sides of the sword')
            parts['pre' if pre.pop() else 'post'].append(state)
        for part, names in parts.items():
            path = out / f'{family}-{part}.yaml'
            if not names:
                path.unlink(missing_ok=True)
                continue
            spec = config.to_dict()
            spec['state_routes'] = [ROUTE]
            spec['start_states'] = names
            spec['notes'] = f'r36 cross-play: {family} on {ROUTE} ({part} the magic sword), {len(names)} states'
            path.write_text(json.dumps(spec, indent=1))
        print(family, {part: len(names) for part, names in parts.items()})


def network(family: str, part: str, models_dir) -> Path:
    """The run that plays `family`'s steps on that side of the sword (the route's own rule)."""
    return registry.promoted_checkpoint(family, models_dir, ROUTE if part == 'pre' else None)


def slim(record: dict) -> dict:
    visits = record.get('visits') or []
    return {'state': record['start_state'], 'seed': record['seed'], 'success': record['success'],
            'outcome': record['outcome'], 'ending': record['ending'], 'steps': record['steps'],
            'start_health': record['start_health'], 'health': record['health'],
            'clips': record.get('clips_allowed'),
            'lost': sum(v.get('health_lost', 0) for v in visits), 'hits': sum(v.get('hits', 0) for v in visits),
            'dealt': sum(v.get('hits_dealt', 0) for v in visits),
            'exits': [v.get('exit') for v in visits], 'rooms': len(visits),
            'linger': (record.get('linger') or {}).get('max_steps'),
            'cheated': record.get('cheated')}


def pairs_of(specs: list[str], spec_dir: Path) -> list[tuple[str, str, str]]:
    """(A, B, part) of every `A:B[,B]` spec (`all` on either side)."""
    tasks = sorted({p.stem.rsplit('-', 1)[0] for p in spec_dir.glob('*.yaml')})
    result = []
    for spec in specs:
        a, bs = spec.split(':')
        for net in (families() if a == 'all' else [a]):
            for task in (tasks if bs == 'all' else bs.split(',')):
                for part in PARTS:
                    if (spec_dir / f'{task}-{part}.yaml').is_file() and (net, task, part) not in result:
                        result.append((net, task, part))
    return result


def cmd_run(a):
    spec_dir, out = Path(a.specs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = pairs_of(a.pairs, spec_dir)
    todo = [p for p in pairs if not (out / f'{p[0]}__{p[1]}-{p[2]}.json').is_file()]
    episodes = sum(len(episode_jobs(load_specialist(str(spec_dir / f'{b}-{part}.yaml')), a.episodes, a.seed, None))
                   for _, b, part in todo)
    print(f'{a.group}: {len(pairs)} pairs, {len(todo)} to play, {episodes} episodes', flush=True)
    runs = [{'specialist': f'{n}@{b}-{part}', 'status': 'pending'} for n, b, part in todo]
    with logged_job(f'r36x-{a.group}', 'eval', runs, a.logs_dir, nice=10, work=work(eval_episodes=episodes)):
        for net, task, part in todo:
            config = load_specialist(str(spec_dir / f'{task}-{part}.yaml'))
            checkpoint = network(net, part, a.models_dir)
            result = evaluate_specialist(config, checkpoint, a.episodes, a.parallel, a.seed, threads=a.parallel)
            slimmed = {key: result[key] for key in ('specialist', 'checkpoint', 'episodes_per_state', 'seed', 'act',
                                                   'clips', 'checkpoint_sha1', 'version', 'code', 'episodes',
                                                   'success_rate', 'mean_health_change', 'mean_steps', 'outcomes',
                                                   'endings')}
            slimmed.update(network=net, task=task, part=part, records=[slim(r) for r in result['records']])
            path = out / f'{net}__{task}-{part}.json'
            path.with_suffix('.tmp').write_text(json.dumps(slimmed))
            path.with_suffix('.tmp').rename(path)
            print(f'{net:18} on {task:18} {part:4}: {result["success_rate"]:6.1%} of {result["episodes"]:4}  '
                  f'hearts {result["mean_health_change"]:+.2f}  steps {result["mean_steps"]:.0f}', flush=True)


def cmd_check(a):
    main_eval = json.loads((ROOT.parent / 'triforce' / 'models' / a.family / a.run / 'eval.json').read_text())
    states = sorted(main_eval['by_state'])
    spec = load_specialist(a.family).to_dict()
    spec['start_states'] = states
    path = Path(a.out).with_suffix('.spec.yaml')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=1))
    checkpoint = registry.checkpoint_path(a.family, a.run, a.models_dir)
    with logged_job(f'r36x-check-{a.family}', 'eval', [{'specialist': a.family, 'status': 'pending'}], a.logs_dir,
                    nice=10, work=work(eval_episodes=a.episodes * len(states))):
        result = evaluate_specialist(load_specialist(str(path)), checkpoint, a.episodes, a.parallel, a.seed,
                                     threads=a.parallel)
    rows = []
    for state in states:
        mine, theirs = result['by_state'].get(state), main_eval['by_state'][state]
        rows.append((state, mine['episodes'], mine['success_rate'], theirs['episodes'], theirs['success_rate'],
                     mine['health_change'], theirs['health_change']))
        print(*rows[-1])
    print('total', result['episodes'], result['success_rate'], result['mean_health_change'],
          main_eval['episodes'], main_eval['success_rate'], main_eval['mean_health_change'])
    Path(a.out).write_text(json.dumps({'rows': rows, 'mine': [result['success_rate'], result['mean_health_change']],
                                       'main': [main_eval['success_rate'], main_eval['mean_health_change']]}))


def _mean_se(values: list[float], clusters: list[str]) -> tuple[float, float]:
    """Mean and its standard error with episodes clustered by start state (episodes of one state share it)."""
    n = len(values)
    if n == 0:
        return float('nan'), float('nan')
    mean = sum(values) / n
    groups = defaultdict(float)
    for value, cluster in zip(values, clusters):
        groups[cluster] += value - mean
    g = len(groups)
    var = sum(s * s for s in groups.values()) / (n * n) * (g / (g - 1) if g > 1 else 1.0)
    return mean, math.sqrt(var)


def cmd_report(a):
    evals = Path(a.evals)
    cells = defaultdict(dict)   # (A, B) -> part -> records
    for path in evals.rglob('*__*.json'):
        data = json.loads(path.read_text())
        cells[(data['network'], data['task'])][data['part']] = data['records']
    matrix = []
    for (net, task), parts in sorted(cells.items()):
        own = cells.get((task, task))
        if own is None or set(own) != set(parts):
            continue
        mine, base = [], []
        for part in sorted(parts):
            key = lambda r: (r['state'], r['seed'], r['clips'])  # noqa: E731
            theirs = {key(r): r for r in own[part]}
            for record in parts[part]:
                if key(record) in theirs:
                    mine.append(record)
                    base.append(theirs[key(record)])
        clusters = [r['state'] for r in mine]

        def stat(field, records):
            return _mean_se([float(r[field]) for r in records], clusters)

        succ, succ_se = stat('success', mine)
        own_succ, own_se = stat('success', base)
        diff, diff_se = _mean_se([float(m['success']) - float(b['success']) for m, b in zip(mine, base)], clusters)
        lost, lost_se = stat('lost', mine)
        own_lost, _ = stat('lost', base)
        fails = [r for r in mine if not r['success']]
        kinds = defaultdict(int)
        for r in fails:
            kinds[failure_kind(r)] += 1
        matrix.append({'network': net, 'task': task, 'episodes': len(mine), 'states': len(set(clusters)),
                       'success': round(succ, 4), 'success_se': round(succ_se, 4),
                       'own_success': round(own_succ, 4), 'own_success_se': round(own_se, 4),
                       'diff': round(diff, 4), 'diff_se': round(diff_se, 4),
                       'ratio': round(succ / own_succ, 3) if own_succ > 0 else None,
                       'hearts_lost': round(lost, 3), 'hearts_lost_se': round(lost_se, 3),
                       'own_hearts_lost': round(own_lost, 3),
                       'hits': round(stat('hits', mine)[0], 3), 'own_hits': round(stat('hits', base)[0], 3),
                       'failures': dict(sorted(kinds.items()))})
    Path(a.output).write_text(json.dumps({'route': ROUTE, 'cells': matrix}, indent=0))
    print(f'{len(matrix)} cells')


def failure_kind(record: dict) -> str:
    """What a failed episode looked like: died, left by another exit (`wrong-exit`), stalled in the room until a cutoff
    (`stall`), or another ending."""
    ending = record['ending']
    if record['health'] <= 0 or ending in ('death', 'died'):
        return 'died'
    if ending in ('exit-wrong', 'wrong-exit', 'left-room', 'exit-incorrect') or 'exit' in ending:
        return 'wrong-exit'
    if ending in ('max-steps', 'no-damage', 'no-room-progress', 'timeout', 'no-progress'):
        return 'stall'
    return ending


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    specs = sub.add_parser('specs')
    specs.add_argument('out')
    run = sub.add_parser('run')
    run.add_argument('group')
    run.add_argument('specs')
    run.add_argument('out')
    run.add_argument('--pairs', nargs='+', required=True)
    for p in (run,):
        p.add_argument('--episodes', type=int, default=5)
        p.add_argument('--seed', type=int, default=100000)
    check = sub.add_parser('check')
    check.add_argument('family')
    check.add_argument('run')
    check.add_argument('out')
    check.add_argument('--episodes', type=int, default=20)
    check.add_argument('--seed', type=int, default=0)
    for p in (run, check):
        p.add_argument('--parallel', type=int, default=10)
        p.add_argument('--models-dir', default=None)
        p.add_argument('--logs-dir', default=None)
    report = sub.add_parser('report')
    report.add_argument('evals')
    report.add_argument('output')
    args = parser.parse_args()
    {'specs': cmd_specs, 'run': cmd_run, 'check': cmd_check, 'report': cmd_report}[args.cmd](args)


if __name__ == '__main__':
    main()
