"""Round 43 (Perf43): are two fixed-seed lab runs the same run? (round40.md "Where a teacher's time goes"): per
iteration, whether the rollout's counts and sums (episodes, frames, rewards, successes, the losses) are equal, and
the first iteration where they differ.

    .venv/bin/python scripts/experiments/r43_perf_same.py RUN_DIR RUN_DIR
"""
import json
import sys
from pathlib import Path

KEYS = ('episodes', 'rollout/frames', 'reward/total', 'success_rate', 'health_change', 'losses/approx_kl',
        'losses/value_loss', 'charts/aux_hit_labels')


def main() -> int:
    a, b = ([json.loads(line) for line in (Path(p) / 'metrics.jsonl').read_text().splitlines()] for p in sys.argv[1:3])
    first = None
    for x, y in zip(a, b):
        same = all(x.get(k) == y.get(k) for k in KEYS)
        if not same and first is None:
            first = x['iteration']
        print(f"iteration {x['iteration']}: {'same' if same else 'differs'}  "
              + '  '.join(f"{k.split('/')[-1]} {x.get(k)!r:.10}/{y.get(k)!r:.10}" for k in KEYS[:4]))
    print(f"first difference: {first if first is not None else 'none'} of {min(len(a), len(b))} iterations")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
