#!/usr/bin/env python3
"""Round 39, "Beams at range" (docs/experiments/round39.md): the beam weights swept on one family, and the beam head.

    python scripts/experiments/r39_beam.py bench BASE.json NEW.json...                 # the rooms tier, beams beside
    python scripts/experiments/r39_beam.py chains BASE.jsonl NEW.jsonl... [--target L:0xRR,...] [--family F...]
    python scripts/experiments/r39_beam.py combine --policy CKPT --beam CKPT --weights 2 4 --name FAMILY --run RUN

``bench``: each network's ``triforce.lab.bench`` result against the base, paired by episode (success, hearts lost,
hits a visit; ``bench.compare``), and beside them the sword shots (round 39's ``beam_*`` episode fields): sword kills
made out of melee range (``far``) and their share, the mean sword-kill distance, shots an episode, the share of
decisions lined up for a shot that moved or waited instead.

``chains``: paired on seeds, the segment's rate (arrival at ``--target``, ``eval_batch``'s targets, or the run's
``success`` without one: Dungeon 9 ends the game) with its standard error, and per family (``--family``, default
every one with sword kills) from ``r38_beams.table``: visits, far share, kill distance, unfired share, the visits
entered with the beams (hearts within half a heart of full) and their share, hits and hearts lost a visit, and the
runs that ended in the family's rooms.

``combine``: ``beamshift`` networks (``triforce/nets/beamshift.py``), one a weight: a policy (any run, a
``hitshift`` included) and a beam head (``r38_hit.py retrain`` on ``DIR+beam`` recordings), nothing trained; saved
as ``<models>/<name>/<run>-w<weight>`` with the default act mode, loadable like any run.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'experiments'))

import r38_beams  # noqa: E402
from triforce import registry  # noqa: E402
from triforce.lab import bench as yardstick  # noqa: E402


def beam_cells(episodes: list) -> dict:
    s = {f: sum(e.get(f'beam_{f}', 0) for e in episodes) for f in yardstick.BEAMS}
    sword = s['kills_far'] + s['kills_near']
    return {'far': s['kills_far'], 'sword': sword, 'far_share': s['kills_far'] / sword if sword else 0.0,
            'distance': s['kill_distance'] / sword if sword else 0.0,
            'shots': s['shots'] / len(episodes) if episodes else 0.0,
            'unfired': s['lined_unfired'] / s['lined'] if s['lined'] else 0.0, 'lined': s['lined']}


def bench_report(base: str, news: list[str]) -> str:
    b = json.loads(Path(base).read_text())
    lines = ['| network | episodes | success | hearts lost | hits | far kills | far share | distance | shots | '
             'unfired |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    every = [e for row in b['keys'].values() for e in yardstick.records(row)]
    c = beam_cells(every)
    lines.append(f"| {Path(base).stem} (base) | {len(every)} | {b['total']['success']:.3f} | "
                 f"{b['total']['hearts_lost']:.3f} | {b['total']['hits_per_visit']:.3f} | {c['far']} | "
                 f"{c['far_share']:.1%} | {c['distance']:.1f} | {c['shots']:.2f} | {c['unfired']:.1%} |")
    for path in news:
        n = json.loads(Path(path).read_text())
        cmp = yardstick.compare(b, n)['total']
        mine = [e for row in n['keys'].values() for e in yardstick.records(row)]
        c = beam_cells(mine)

        def cell(metric):
            mean, se = cmp[metric]
            return f"{mean:+.3f} ± {se:.3f}"
        lines.append(f"| {Path(path).stem} | {cmp['pairs']} | {cell('success')} | {cell('hearts_lost')} | "
                     f"{cell('hits')} | {c['far']} | {c['far_share']:.1%} | {c['distance']:.1f} | {c['shots']:.2f} | "
                     f"{c['unfired']:.1%} |")
        losing = yardstick.compare(b, n)['losing']
        if losing:
            lines.append(f"| ↳ losing rows | {'; '.join(losing)} |||||||||")
    return '\n'.join(lines)


def arrived(run: dict, target) -> bool:
    if not target:
        return run.get('result') == 'success'
    want = [(int(level), int(room, 16)) for level, room in (t.split(':') for t in target.split(','))]
    k = 0
    for room in run['rooms']:
        if (room['level'], int(room['room'], 16)) == want[k]:
            k += 1
            if k == len(want):
                return True
    return False


def chain_report(base: str, news: list[str], target, families) -> str:
    runs = {p: {r['seed']: r for r in map(json.loads, open(p, encoding='utf-8')) if r} for p in [base, *news]}
    lines = []
    for path in news:
        seeds = sorted(set(runs[base]) & set(runs[path]))
        a = np.array([arrived(runs[base][s], target) for s in seeds], float)
        b = np.array([arrived(runs[path][s], target) for s in seeds], float)
        d = b - a
        lines.append(f"{Path(path).name}: {len(seeds)} runs, {a.mean():.1%} -> {b.mean():.1%} "
                     f"({d.mean() * 100:+.1f} ± {d.std(ddof=1) * 100 / math.sqrt(len(seeds)):.1f}; base only "
                     f"{int((d < 0).sum())}, new only {int((d > 0).sum())})")
    lines += ['', '| chain | family | visits | full % | far share | distance | unfired | hits/visit | hearts/visit | '
              'runs ended |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for path in [base, *news]:
        table = r38_beams.table([path])
        ended = defaultdict(int)
        for run in runs[path].values():
            if not arrived(run, target) and run['rooms']:
                ended[run['rooms'][-1]['specialist']] += 1
        hits, hearts = defaultdict(float), defaultdict(float)
        for run in runs[path].values():
            for room in run['rooms']:
                hits[room['specialist']] += room['hits']
                hearts[room['specialist']] += room.get('beam_hearts_lost', 0.0)
        keys = families or [k for k in table if table[k]['kills_far'] + table[k]['kills_near'] > 0]
        for key in keys:
            r = table.get(key)
            if not r:
                continue
            fmt = lambda v, s: '-' if v is None or v != v else format(v, s)  # noqa: E731
            lines.append(f"| {Path(path).stem} | {key} | {r['visits']} | {r['full_visits'] / r['visits']:.1%} | "
                         f"{fmt(r['far_share'], '.1%')} | {fmt(r['kill_distance'], '.1f')} | "
                         f"{fmt(r['unfired_share'], '.1%')} | {hits[key] / r['visits']:.3f} | "
                         f"{hearts[key] / r['visits']:.3f} | {ended[key]} |")
    return '\n'.join(lines)


def combine(policy_path: str, beam_path: str, weights: list[float], name: str, run: str, models_dir=None) -> None:
    from triforce.nets.beamshift import BeamShift  # pylint: disable=import-outside-toplevel
    policy = registry.load_model(policy_path, check_schema=False)[0]
    beam = registry.load_model(beam_path, check_schema=False)[0]
    for weight in weights:
        network = BeamShift.of(policy, beam, weight)
        path = registry.save_checkpoint(name, f'{run}-w{weight:g}', network, root=models_dir,
                                        extra={'policy': str(policy_path), 'beam': str(beam_path)})
        registry.set_act_mode(path, registry.DEFAULT_ACT)
        print(f'[combine] {path}', flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('bench')
    b.add_argument('base')
    b.add_argument('new', nargs='+')
    c = sub.add_parser('chains')
    c.add_argument('base')
    c.add_argument('new', nargs='+')
    c.add_argument('--target', default=None)
    c.add_argument('--family', nargs='*', default=None)
    m = sub.add_parser('combine')
    m.add_argument('--policy', required=True)
    m.add_argument('--beam', required=True)
    m.add_argument('--weights', type=float, nargs='+', required=True)
    m.add_argument('--name', required=True)
    m.add_argument('--run', required=True)
    m.add_argument('--models-dir', default=None)
    args = parser.parse_args(argv)
    if args.command == 'bench':
        print(bench_report(args.base, args.new))
    elif args.command == 'chains':
        print(chain_report(args.base, args.new, args.target, args.family))
    else:
        combine(args.policy, args.beam, args.weights, args.name, args.run, args.models_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
