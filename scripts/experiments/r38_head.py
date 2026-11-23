"""Round 38: the hit head carried to the families that cost the most runs on `d3first` (docs/experiments/round38.md
"The hit head across families").

    python scripts/experiments/r38_head.py cost [--records GLOB...]          # runs and hearts lost a family
    python scripts/experiments/r38_head.py manifests --families KEY...        # target/r38h/manifest-<family>.json
    python scripts/experiments/r38_head.py combine --head NAME=CKPT --families KEY... --weights 2 4 8
    python scripts/experiments/r38_head.py bench --families KEY... --arm NAME=SPEC... --out DIR
    python scripts/experiments/r38_head.py pr --hit NAME=CKPT... --data DIR... --families KEY... --out JSON
    python scripts/experiments/r38_head.py slices36 --hit CKPT --family darknut|boss-gleeok|goriya-all
    python scripts/experiments/r38_head.py chains BASE.jsonl NEW.jsonl... --target L:0xRR,...  [--out JSON]
    python scripts/experiments/r38_head.py product [--arm SEG=NEW.jsonl...]  [--out JSON]

``cost``: from the 1000-run segment chains of today's 51 networks on the corrected route (``r36base``: every room
record has its family, hits and health in and out), per family and segment the runs lost in its rooms (the run's
last room, for every way a run ends short) and the hearts lost there (health in + hearts picked up - health out),
per 1000 runs of the segment (the 300-run segments scaled).

``combine``: per family, its ``d3first`` policy (``registry.policy_checkpoint``) and the head side by side
(``hitshift``, nothing trained), ``models/r38head/<family>-<NAME>-w<w>``.

``bench``: one queued job; per family its manifest (24 of its ``d3first`` harvested entries x 10 seeds, the
yardstick's choice) and every arm, ``DIR/<family>/bench-<arm>.json`` (``r38_hit.py table DIR/<family>``). An arm
``SPEC``: ``teacher`` (today's network), ``planner`` (today's network with the look-ahead, beside), or a combination
name ``NAME-w<w>`` (``models/r38head/<family>-NAME-w<w>``).

``pr``: per family, each head on the held-out shards of the recordings (``DIR#KEY``): precision = recall on the
actions taken and on every judged candidate (``r38_hit.pr_report``), and the judged hits split by what was in the
air at the decision (from the inputs: an enemy shot present, ``projectile`` and ``hurts_on_touch`` set): recall at
the P = R threshold of the hits with a shot in the air and of those without one (touches, and shots fired after
the decision; the recordings do not name the hitter). ``slices36``: the same on round 36's collections, where the
hitter is known (touch / a shot in the air / a shot fired after the decision), and by the hit's frame in the action.

``chains``: paired on seeds, the arrival rate with its standard error, the rooms that lose runs, and per family
(its rooms summed) hits and hearts lost a visit and runs lost, base against new. ``product``: the predicted rate of
one run from the start (round 37's ``r37_single_run.py``): the product of the segment rates of today's team (main's
1000-run record ``r37fix-1k`` where it has the segment, round 36's ``r37_one_table.ref_path`` otherwise; every rate
by the segment's arrival, ``eval_batch.D3FIRST_TARGETS``), and the same with each ``--arm`` segment's rate instead.
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'experiments'))

import r38_hit as rh  # noqa: E402
from r37_one_table import SEGMENTS, ref_path  # noqa: E402
from triforce import _native, registry  # noqa: E402
from triforce.eval_batch import D3FIRST_MILESTONES, D3FIRST_TARGETS  # noqa: E402
from triforce.events import progress_job, write_json_atomic  # noqa: E402
from triforce.jobs import add_job_arguments, begin_job, work  # noqa: E402
from triforce.lab import bench as yardstick  # noqa: E402

R38H = ROOT / 'target' / 'r38h'
CHAINS = str(ROOT / 'target' / 'r56-chains')
RECORD = 'r37fix-1k'
PLANNER = 'temperature:0.5+relax:16+plan:16:1'
FEATURES = list(_native.ENTITY_FEATURES)
PRESENCE, PROJECTILE, HURTS = (FEATURES.index(k) for k in ('presence', 'projectile', 'hurts_on_touch'))


# ---------------------------------------------------------------------------------------------- the cost table

def cost(paths: list[str]) -> dict:
    """family -> segment -> {runs, lost, hearts, hits, visits} per 1000 runs of the segment (module docstring)."""
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    rates = {}
    for path in paths:
        segment = Path(path).stem.split('-', 2)[2]
        runs = [json.loads(line) for line in open(path, encoding='utf-8')]
        scale = 1000 / len(runs)
        rates[segment] = sum(r['result'] == 'success' for r in runs) / len(runs)
        for run in runs:
            for room in run['rooms']:
                cell = out[room['specialist']][segment]
                cell['visits'] += scale
                cell['hits'] += scale * room['hits']
                cell['hearts'] += scale * max(0.0, room['health_in'] + room['hearts_healed'] - room['health_out'])
            if run['result'] != 'success' and run['rooms']:
                out[run['rooms'][-1]['specialist']][segment]['lost'] += scale
    return {'rates': rates, 'families': {f: {s: dict(c) for s, c in segs.items()} for f, segs in out.items()}}


def cost_table(result: dict, segments: list[str]) -> str:
    lines = ['| family | ' + ' | '.join(f'{s} lost / hearts' for s in segments) + ' | runs lost | hearts lost | '
             'hits a visit |', '|---|' + '---:|' * (len(segments) + 3)]
    rows = []
    for family, segs in result['families'].items():
        cells = [segs.get(s, {}) for s in segments]
        lost = sum(c.get('lost', 0) for c in cells)
        hearts = sum(c.get('hearts', 0) for c in cells)
        visits = sum(c.get('visits', 0) for c in cells)
        if visits:
            rows.append((lost, family, cells, hearts, sum(c.get('hits', 0) for c in cells) / visits))
    for lost, family, cells, hearts, hpv in sorted(rows, reverse=True):
        lines.append(f'| {family} | ' + ' | '.join(f"{c.get('lost', 0):.0f} / {c.get('hearts', 0):.0f}" if c else ''
                                                  for c in cells) + f' | {lost:.0f} | {hearts:.0f} | {hpv:.2f} |')
    lines.append('| segment rate | ' + ' | '.join(f"{result['rates'][s]:.3f}" for s in segments) + ' | | | |')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------------- networks, bench

def family_of(key: str) -> str:
    return registry.split_policy(key)[0]


def write_manifests(keys: list[str], states: int, episodes: int, out: Path) -> None:
    base = yardstick.load_manifest()
    states_dir = Path(_native.paths()['states_dir'])
    out.mkdir(parents=True, exist_ok=True)
    for key in keys:
        names, source = yardstick.choose_states(family_of(key), states, states_dir, 'd3first')
        manifest = {**base, 'episodes': episodes, 'states_per_family': states,
                    'entries': [{'key': key, 'family': family_of(key), 'source': source, 'states': names}]}
        write_json_atomic(out / f'manifest-{family_of(key)}.json', manifest)
        print(f'[manifests] {key}: {len(names)} states ({source}) x {episodes}', flush=True)


def combination(key: str, name: str) -> str:
    return str(ROOT / 'models' / 'r38head' / f'{family_of(key)}-{name}')


def combine(keys: list[str], heads: list[str], weights: list[float]) -> None:
    for key in keys:
        policy = str(registry.policy_checkpoint(key, ROOT / 'models'))
        for head in heads:
            name, checkpoint = head.split('=', 1)
            checkpoint = checkpoint.replace('{family}', family_of(key))
            rh.combine(policy, checkpoint, weights, 'r38head', f'{family_of(key)}-{name}', 'softplus')
            print(f'[combine] {key}: {policy} + {checkpoint} -> r38head/{family_of(key)}-{name}-w*', flush=True)


def bench(keys: list[str], arms: list[str], out: Path, manifests: Path, parallel: int, logs_dir=None) -> None:
    plan = []
    for key in keys:
        manifest = yardstick.load_manifest(manifests / f'manifest-{family_of(key)}.json')
        teacher = str(registry.policy_checkpoint(key, ROOT / 'models'))
        for arm in arms:
            if arm == 'teacher':
                plan.append((key, manifest, arm, teacher, None))
            elif arm == 'planner':
                plan.append((key, manifest, arm, teacher, PLANNER))
            else:
                plan.append((key, manifest, arm, combination(key, arm), None))
    total = sum(len(m['entries'][0]['states']) * m['episodes'] for _, m, *_ in plan)
    with progress_job('lab-bench', 'eval', total, root=logs_dir, work=work(eval_episodes=total)) as progress:
        done = 0
        for key, manifest, arm, checkpoint, act in plan:
            path = out / family_of(key) / f'bench-{arm}.json'
            episodes = len(manifest['entries'][0]['states']) * manifest['episodes']
            if path.exists():
                print(f'[bench] {key} {arm}: kept {path}', flush=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                result = yardstick.run_bench({key: checkpoint}, manifest, parallel, act=act)
                result['arm'] = {'name': arm, 'checkpoint': checkpoint, 'act': act}
                write_json_atomic(path, yardstick.compact(result))
                row = result['keys'][key]
                print(f"[bench] {key} {arm}: hits/visit {row['hits_per_visit']:.3f} hearts {row['hearts_lost']:.3f} "
                      f"success {row['success']:.3f} decisions {row['decisions']:.1f} "
                      f"({result['decisions_per_second']:.0f} decisions/s)", flush=True)
            done += episodes
            progress.update(done, 0.0, key=f'{key} {arm}')


# ---------------------------------------------------------------------------------------------- precision

def shot_in_air(obs: dict, rows: np.ndarray) -> np.ndarray:
    f = obs['entity_features'][rows]
    return ((f[..., PRESENCE] > 0) & (f[..., PROJECTILE] > 0) & (f[..., HURTS] > 0)).any(-1)


def pr_sliced(rows: rh.Rows, logits: torch.Tensor, index: np.ndarray) -> dict:
    out = rh.pr_report(rows, logits, index)
    where = np.full(rows.rows, -1)
    where[index] = np.arange(len(index))
    keep = where[rows.c_row] >= 0
    c_rows, joints, lost = rows.c_row[keep], rows.c_joint[keep], rows.c_lost[keep]
    scores = logits[torch.as_tensor(where[c_rows]), torch.as_tensor(joints)].cpu().numpy()
    k = int(lost.sum())
    if k == 0:
        return out
    threshold = np.sort(scores)[-k]
    shot = shot_in_air(rows.obs, c_rows)
    for name, m in (('shot_in_air', shot), ('no_shot', ~shot)):
        hit = lost & m
        out[f'recall_{name}'] = float((scores[hit] >= threshold).mean()) if hit.any() else float('nan')
        out[f'lost_{name}'] = int(hit.sum())
    out['decisions'] = int(len(index))
    out['taken_lost_rate'] = float(rows.c_lost[rows.c_taken].mean())
    return out


def pr(heads: list[str], sources: list[str], device, everything: bool = False) -> dict:
    results = {}
    loaded = {}
    for source in sources:
        rows = rh.load_rows(source)
        index = np.arange(rows.rows) if everything else np.flatnonzero(rows.held)
        for head in heads:
            name, checkpoint = head.split('=', 1)
            family = source.partition('#')[2]
            checkpoint = checkpoint.replace('{family}', family_of(family) if family else '')
            if checkpoint not in loaded:
                loaded[checkpoint] = rh.load(checkpoint).to(device).eval()
            report = pr_sliced(rows, rh.hit_logits(loaded[checkpoint], rows.obs, index, device), index)
            results[f'{name} on {source}'] = report
            print(name, source, json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                            for k, v in report.items()}), flush=True)
    return results


def slices36(head: str, family: str, device) -> dict:
    """The head on round 36's held-out collection of ``family``, the hitter known: recall at the P = R threshold of
    touches, of shots in the air at the decision, of shots fired after it, and by the hit's frame in the action."""
    import r36_hit_bench as hb  # pylint: disable=import-outside-toplevel
    import r38_lane  # pylint: disable=import-outside-toplevel
    data, _ = hb.load_family(family)
    rows = rh.load_rows(f'r36:{family}')
    network = rh.load(head).to(device).eval()
    index = np.flatnonzero(rows.held)
    logits = rh.hit_logits(network, rows.obs, index, device)
    out = rh.pr_report(rows, logits, index)
    where = np.full(rows.rows, -1)
    where[index] = np.arange(len(index))
    keep = where[data.c_row] >= 0
    scores = logits[torch.as_tensor(where[data.c_row[keep]]), torch.as_tensor(data.c_joint[keep])].cpu().numpy()
    lost = data.c_lost[keep].astype(bool)
    d = data.d
    typ = d['tried_hit_type'][data.c_index][keep].astype(np.int64)
    frame = d['tried_hit_frame'][data.c_index][keep].astype(np.int64)
    post = r38_lane.post_decision_shots(data)[keep]
    shot = lost & (typ >= 0x53)
    groups = {'touch': lost & (typ < 0x53), 'shot_in_air': shot & ~post, 'shot_after': post}
    for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 999)):
        groups[f'frame_{lo}_{hi}'] = lost & (frame >= lo) & (frame < hi)
    # every judged candidate, then the model's own draws alone (mode 1, the taken action: where the heads differ)
    own = data.c_own[keep] & (data.mode_of_row[data.c_row[keep]] == 1)
    for prefix, sel in (('', np.ones(len(lost), bool)), ('taken1_', own)):
        threshold = np.sort(scores[sel])[-int(lost[sel].sum())]
        for name, m in groups.items():
            m = m & sel
            out[f'{prefix}recall_{name}'] = float((scores[m] >= threshold).mean()) if m.any() else float('nan')
            out[f'{prefix}lost_{name}'] = int(m.sum())
    print(family, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}), flush=True)
    return out


# ---------------------------------------------------------------------------------------------- chains

def arrival(run: dict, target: str) -> int:
    """Rooms of ``run`` up to its arrival (the target rooms ``L:0xRR,...`` entered in order: ``eval_batch``'s
    segment arrivals, as the record of a segment start without ``--stop-at`` is scored), or -1 (never arrived)."""
    want = [(int(level), int(room, 16)) for level, room in (t.split(':') for t in target.split(','))]
    k = 0
    for i, room in enumerate(run['rooms']):
        if (room['level'], int(room['room'], 16)) == want[k]:
            k += 1
            if k == len(want):
                return i
    return -1


def _trace(run: dict, target: str) -> list:
    end = arrival(run, target)
    return [(r['level'], r['room'], r['hits'], r['steps']) for r in (run['rooms'] if end < 0 else run['rooms'][:end])]


def chain_compare(base: str, new: str, target: str) -> dict:
    """Paired on seeds: the arrival rate of each (a chain stopped at the target's last room, ``--stop-at``, and a
    record that played on score alike), the difference with its standard error, and per room up to the arrival the
    runs lost (the last room of a run that never arrived) and hits a visit of each."""
    a = {r['seed']: r for r in map(json.loads, open(base, encoding='utf-8'))}
    b = {r['seed']: r for r in map(json.loads, open(new, encoding='utf-8'))}
    seeds = sorted(set(a) & set(b))
    sa = np.array([arrival(a[s], target) >= 0 for s in seeds], float)
    sb = np.array([arrival(b[s], target) >= 0 for s in seeds], float)
    diff = sb - sa
    out = {'runs': len(seeds), 'base': float(sa.mean()), 'new': float(sb.mean()), 'diff': float(diff.mean()),
           'se': float(diff.std(ddof=1) / math.sqrt(len(seeds))),
           'identical_runs': int(sum(_trace(a[s], target) == _trace(b[s], target) for s in seeds))}
    rooms = {}
    for tag, runs in (('base', a), ('new', b)):
        for s in seeds:
            run = runs[s]
            end = arrival(run, target)
            played = run['rooms'] if end < 0 else run['rooms'][:end]
            for room in played:
                cell = rooms.setdefault(f"{room['level']}:{room['room']} {room['specialist']}", defaultdict(float))
                cell[f'{tag}_visits'] += 1
                cell[f'{tag}_hits'] += room['hits']
                cell[f'{tag}_hearts'] += max(0.0, room['health_in'] + room['hearts_healed'] - room['health_out'])
            if end < 0 and run['rooms']:
                last = run['rooms'][-1]
                rooms.setdefault(f"{last['level']}:{last['room']} {last['specialist']}",
                                 defaultdict(float))[f'{tag}_lost'] += 1
    out['rooms'] = {k: dict(v) for k, v in sorted(rooms.items(), key=lambda kv: -kv[1].get('base_lost', 0))}
    return out


def family_rows(rooms: dict) -> dict:
    """``chain_compare``'s rooms summed by family: visits, hits and hearts lost a visit, runs lost, each side."""
    sums: dict = defaultdict(lambda: defaultdict(float))
    for name, cell in rooms.items():
        for k, v in cell.items():
            sums[name.split(' ', 1)[1]][k] += v
    rows = {}
    for family, c in sums.items():
        row = {}
        for tag in ('base', 'new'):
            visits = c[f'{tag}_visits']
            row.update({f'{tag}_visits': visits, f'{tag}_lost': c[f'{tag}_lost'],
                        f'{tag}_hits': c[f'{tag}_hits'] / visits if visits else 0.0,
                        f'{tag}_hearts': c[f'{tag}_hearts'] / visits if visits else 0.0})
        rows[family] = row
    return dict(sorted(rows.items(), key=lambda kv: -kv[1]['base_lost']))


def segment_rate(path, segment: str) -> tuple:
    """(the share of a chain's runs that arrive at ``segment``'s end, runs); 'start' arrives at Level 3's entrance;
    a segment without a target (d9, the last) counts success."""
    runs = [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]
    if segment == 'start':
        ok = [arrival(r, dict(D3FIRST_MILESTONES)['d3']) >= 0 for r in runs]
    elif segment in D3FIRST_TARGETS:
        ok = [arrival(r, D3FIRST_TARGETS[segment]) >= 0 for r in runs]
    else:
        ok = [r['result'] == 'success' for r in runs]
    return sum(ok) / len(runs), len(runs)


def product(arms: dict) -> dict:
    """Per segment the team's rate (its record) and the arm's where given; both products (module docstring)."""
    out: dict = {'segments': {}}
    team = with_arms = 1.0
    for segment in ['start', *SEGMENTS[:-1]]:
        name = 'full' if segment == 'start' else segment
        record = Path(CHAINS) / f'{RECORD}-{name}.jsonl'
        record = record if record.is_file() else ref_path(name)
        p, runs = segment_rate(record, segment)
        row = {'record': str(record), 'p': p, 'runs': runs}
        if segment in arms:
            row['arm_p'], row['arm_runs'] = segment_rate(arms[segment], segment)
            row['arm'] = arms[segment]
        team *= p
        with_arms *= row.get('arm_p', p)
        out['segments'][segment] = row
    out['team'], out['with_arms'] = team, with_arms
    return out


# ---------------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('cost')
    c.add_argument('--records', nargs='+', default=sorted(glob.glob(f'{CHAINS}/r36base-1k-*.jsonl')) + [
        f'{CHAINS}/r36base-300-{s}.jsonl' for s in ('d1', 'd8', 'd9', 'post-d3', 'post-d4', 'post-d6', 'post-d8')])
    c.add_argument('--segments', nargs='+', default=['d4', 'd3', 'd5', 'd8'])
    c.add_argument('--out', type=Path, default=R38H / 'cost.json')
    m = sub.add_parser('manifests')
    m.add_argument('--families', nargs='+', required=True)
    m.add_argument('--states', type=int, default=24)
    m.add_argument('--episodes', type=int, default=10)
    m.add_argument('--out', type=Path, default=R38H)
    k = sub.add_parser('combine')
    k.add_argument('--families', nargs='+', required=True)
    k.add_argument('--head', nargs='+', required=True, metavar='NAME=CKPT', help='CKPT may hold {family}')
    k.add_argument('--weights', type=float, nargs='+', default=[2, 4, 8])
    b = sub.add_parser('bench')
    b.add_argument('--families', nargs='+', required=True)
    b.add_argument('--arm', nargs='+', required=True)
    b.add_argument('--out', type=Path, required=True)
    b.add_argument('--manifests', type=Path, default=R38H)
    b.add_argument('--parallel', type=int, default=10)
    b.add_argument('--logs-dir', default=None)
    add_job_arguments(b)
    p = sub.add_parser('pr')
    p.add_argument('--hit', nargs='+', required=True, metavar='NAME=CKPT', help='CKPT may hold {family}')
    p.add_argument('--data', nargs='+', required=True, help='recording DIRs (each read per family)')
    p.add_argument('--families', nargs='+', required=True)
    p.add_argument('--all', action='store_true')
    p.add_argument('--out', type=Path, default=None)
    s = sub.add_parser('slices36')
    s.add_argument('--hit', required=True)
    s.add_argument('--family', nargs='+', required=True)
    s.add_argument('--out', type=Path, default=None)
    h = sub.add_parser('chains')
    h.add_argument('base')
    h.add_argument('new', nargs='+')
    h.add_argument('--target', required=True, help="the segment's arrival, eval_batch's D3FIRST_TARGETS (d4: "
                   "'4:0x03,0:0x45')")
    h.add_argument('--out', type=Path, default=None, help='JSON: each new chain\'s comparison, rooms and families')
    o = sub.add_parser('product')
    o.add_argument('--arm', nargs='*', default=[], metavar='SEG=JSONL', help="a segment's chain in place of the team's")
    o.add_argument('--out', type=Path, default=None)
    for q in (p, s):
        q.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.command == 'cost':
        result = cost(args.records)
        write_json_atomic(args.out, result)
        print(cost_table(result, args.segments))
    elif args.command == 'manifests':
        write_manifests(args.families, args.states, args.episodes, args.out)
    elif args.command == 'combine':
        combine(args.families, args.head, args.weights)
    elif args.command == 'bench':
        begin_job(args)
        bench(args.families, args.arm, args.out, args.manifests, args.parallel, args.logs_dir)
    elif args.command == 'pr':
        sources = []
        for directory in args.data:
            keys = json.loads((Path(directory) / 'dataset.json').read_text())['families']
            sources += [f'{directory}#{key}' for key in keys if key in args.families]
        results = pr(args.hit, sources, torch.device(args.device), args.all)
        if args.out:
            write_json_atomic(args.out, results)
    elif args.command == 'slices36':
        results = {f: slices36(args.hit.replace('{family}', f), f, torch.device(args.device)) for f in args.family}
        if args.out:
            write_json_atomic(args.out, results)
    elif args.command == 'chains':
        results = {}
        for new in args.new:
            result = chain_compare(args.base, new, args.target)
            result['families'] = family_rows(result['rooms'])
            results[new] = result
            print(new, json.dumps({k: v for k, v in result.items() if k not in ('rooms', 'families')}))
            for room, cell in list(result['rooms'].items())[:8]:
                print(f"  {room}: lost {cell.get('base_lost', 0):.0f} -> {cell.get('new_lost', 0):.0f}; hits a visit "
                      f"{cell.get('base_hits', 0) / max(cell.get('base_visits', 1), 1):.2f} -> "
                      f"{cell.get('new_hits', 0) / max(cell.get('new_visits', 1), 1):.2f}")
            for family, row in result['families'].items():
                if row['base_lost'] or row['new_lost']:
                    print(f"  [{family}] visits {row['base_visits']:.0f} -> {row['new_visits']:.0f}; hits a visit "
                          f"{row['base_hits']:.3f} -> {row['new_hits']:.3f}; hearts lost a visit "
                          f"{row['base_hearts']:.3f} -> {row['new_hearts']:.3f}; runs lost {row['base_lost']:.0f} -> "
                          f"{row['new_lost']:.0f}")
        if args.out:
            write_json_atomic(args.out, results)
    elif args.command == 'product':
        result = product(dict(arm.split('=', 1) for arm in args.arm))
        print(f"team {result['team']:.3g} (one in {1 / result['team']:,.0f}); with the arms {result['with_arms']:.3g} "
              f"(one in {1 / result['with_arms']:,.0f}); the owner's target 1e-4")
        print('| segment | team p | arm p | -ln p team | -ln p arm |\n|---|---:|---:|---:|---:|')
        for segment, row in result['segments'].items():
            arm = row.get('arm_p')
            print(f"| {segment} | {100 * row['p']:.1f}% | {'' if arm is None else f'{100 * arm:.1f}%'} | "
                  f"{-math.log(row['p']):.2f} | {'' if arm is None else f'{-math.log(arm):.2f}'} |")
        if args.out:
            write_json_atomic(args.out, result)
    return 0


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    sys.exit(main())
