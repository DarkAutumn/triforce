"""Round 36, "The wood-sword pairs": two chain files (``triforce eval-chain --output``) paired run by run.

    python scripts/experiments/r36_merge_chains.py BASE.jsonl NEW.jsonl [--out R.json]

Per file: success (the run reaches the end of the segment), rooms cleared a run; paired by seed: the difference in
success (new - base) with its standard error, runs that changed either way, and where each file's failed runs end
(the record's ``reason``)."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


def load(path) -> dict:
    runs = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            run = json.loads(line)
            runs[run['seed']] = run
    return runs


def summary(runs: list) -> dict:
    n = len(runs)
    return {'runs': n, 'success': sum(r['result'] == 'success' for r in runs) / n if n else 0.0,
            'rooms': sum(len(r.get('rooms') or []) for r in runs) / n if n else 0.0,
            'failures': Counter(str(r.get('reason')) for r in runs if r['result'] != 'success').most_common(8)}


def compare(base: dict, new: dict) -> dict:
    seeds = sorted(set(base) & set(new))
    diffs = [(new[s]['result'] == 'success') - (base[s]['result'] == 'success') for s in seeds]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    return {'pairs': n, 'base': summary([base[s] for s in seeds]), 'new': summary([new[s] for s in seeds]),
            'diff': mean, 'se': math.sqrt(var / n), 'gained': diffs.count(1), 'lost': diffs.count(-1),
            'identical': sum(json.dumps(base[s].get('rooms')) == json.dumps(new[s].get('rooms')) for s in seeds)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('base')
    parser.add_argument('new')
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    result = compare(load(args.base), load(args.new))
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
