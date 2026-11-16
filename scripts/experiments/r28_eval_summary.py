"""One line per start state of `triforce evaluate` result files (the round 28 walks' model runs,
docs/routes/d8-walks.md): episodes, successes, endings, mean heart change, median frames, bombs, arrows and keys.

    python scripts/experiments/r28_eval_summary.py DIR_OR_JSON...
"""

import collections
import json
import statistics
import sys
from pathlib import Path


def summarize(path: Path):
    data = json.loads(path.read_text())
    by = collections.defaultdict(list)
    for r in data['records']:
        by[r['start_state']].append(r)
    print(f"{data['specialist']} ({Path(data['checkpoint']).parent.name}, act {data['act']}, clips {data['clips']}):")
    for state, rows in sorted(by.items()):
        n = len(rows)
        ok = [r for r in rows if r['success']]
        ends = collections.Counter(r['ending'] for r in rows)
        change = statistics.mean(r['health'] - r['start_health'] for r in rows)
        won = statistics.mean(r['health'] - r['start_health'] for r in ok) if ok else None
        frames = int(statistics.median(r['frames'] for r in ok)) if ok else None
        all_frames = int(statistics.median(r['frames'] for r in rows))
        extra = []
        bombs = statistics.mean(r.get('bombs', {}).get('used', 0) for r in rows)
        if bombs:
            extra.append(f'bombs used {bombs:.2f}')
        cheated = collections.Counter()
        for r in rows:
            for key, value in (r.get('cheated') or {}).items():
                if isinstance(value, (int, float)) and value:
                    cheated[key] += value
        for key, value in cheated.items():
            extra.append(f'cheated {key} {value / n:.2f}')
        dropped = statistics.mean(r['hearts']['dropped'] for r in rows)
        taken = statistics.mean(r['hearts']['taken'] for r in rows)
        print(f"  {state}: {len(ok)}/{n}; endings {dict(ends)}; hearts {change:+.2f}"
              + (f' (wins {won:+.2f})' if ok and len(ok) < n else '')
              + f"; frames median {all_frames}" + (f' (wins {frames})' if ok and len(ok) < n else '')
              + f"; hearts dropped {dropped:.2f} taken {taken:.2f}" + ('; ' + ', '.join(extra) if extra else ''))


def main():
    for arg in sys.argv[1:]:
        p = Path(arg)
        for f in (sorted(p.glob('*.json')) if p.is_dir() else [p]):
            summarize(f)


if __name__ == '__main__':
    main()
