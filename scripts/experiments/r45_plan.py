#!/usr/bin/env python3
"""Round 45 (Tracks45): the explorer's work list for the walked doors, written as a shell script.

    python scripts/experiments/r45_plan.py --done ROOT [--done ROOT...] [--families F,G] [--skip-families F,G]
        --out SCRIPT [--iterations 20000] [--patience 3000] [--parallel 8]

For every member of `triforce/groups.yaml` whose task (`lab.collect.task_yaml`) has its curriculum on, the walked
states it draws (`triforce/states/d3first_walk.json`, chain 0) that it does not declare `curriculum.untracked` and
that no track under the `--done` roots already starts from: one `python -m triforce.explore --layout route` line a
family, the states comma-separated. It prints the counts by family. Nothing is played here.
"""
import argparse
import fnmatch
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def plan(done_roots: list, families: set | None, skip: set) -> dict:
    from triforce import inputs  # pylint: disable=import-outside-toplevel
    from triforce.lab.collect import task_yaml  # pylint: disable=import-outside-toplevel
    from triforce.specialist import load_specialist  # pylint: disable=import-outside-toplevel
    walked = {Path(n).stem for n in json.loads((ROOT / 'triforce/states/d3first_walk.json').read_text())['doors']}
    done = set()
    for root in done_roots:
        for meta in Path(root).glob('*/*/trajectory.json'):
            done.add((meta.parent.parent.name, json.loads(meta.read_text())['start_state']))
    groups = yaml.safe_load((ROOT / 'triforce/groups.yaml').read_text())['teachers']
    out = {}
    for group in groups.values():
        for member in group['members']:
            family = member.split('@')[0]
            if (families and family not in families) or family in skip:
                continue
            config = load_specialist(task_yaml(member))
            if not config.curriculum.enabled:
                continue
            declared = inputs.untracked(config)
            seeds = sorted(Path(p).stem for p in config.resolve_start_states(None))
            todo = [s for s in seeds if s in walked and (config.name, s) not in done
                    and not any(fnmatch.fnmatchcase(s, g) for g in declared)]
            if todo:
                out[member] = (task_yaml(member), todo)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('--done', action='append', default=[])
    parser.add_argument('--families', default=None)
    parser.add_argument('--skip-families', default='')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--explore-out', default='target/r45/explore')
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--patience', type=int, default=3000)
    parser.add_argument('--parallel', type=int, default=8)
    args = parser.parse_args(argv)
    work = plan(args.done, set(args.families.split(',')) if args.families else None,
                set(filter(None, args.skip_families.split(','))))
    lines = ['#!/bin/bash', '# Tracks45: written by scripts/experiments/r45_plan.py', 'cd ' + str(ROOT),
             'export TRIFORCE_STATES_DIR=$PWD/target/r45/states']
    for member, (task, states) in work.items():
        lines.append(f".venv/bin/python -m triforce.explore --specialist {task} --states {','.join(states)} "
                     f"--iterations {args.iterations} --patience {args.patience} --parallel {args.parallel} "
                     f"--layout route --out {args.explore_out}")
        print(f'{member}: {len(states)}')
    lines.append('exit 0')
    args.out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    args.out.chmod(0o755)
    print(f'{sum(len(s) for _, s in work.values())} states in {len(work)} tasks -> {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
