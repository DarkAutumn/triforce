"""Round 36 (docs/experiments/round36.md "What the observation lacks"): one markdown row per ``r36_hit_oracle.py``
result (its last epoch): training loss, held-out log loss, candidate AUC, and per mode the draws' AUC and precision
= recall at the look-ahead's veto rate with its standard error (resampled episodes).

    python scripts/experiments/r36_hit_table.py OUT1.json OUT2.json ...
"""

import json
import sys
from pathlib import Path


def main(argv) -> int:
    print('| run | train loss | held-out log loss | AUC (all) | planner draws: AUC, P = R | model draws: AUC, P = R |')
    print('|---|---:|---:|---:|---|---|')
    for path in argv:
        r = json.loads(Path(path).read_text())['history'][-1]
        cells = [Path(path).stem, f"{r['train_loss']:.3f}", f"{r['log_loss']:.3f}", f"{r['auc']:.4f}"]
        for mode in ('draw0', 'draw1'):
            m = r.get(mode)
            cells.append(f"{m['auc']:.3f}, **{m['pr']:.3f}** ± {m['pr_se']:.3f}" if m else '')
        print('| ' + ' | '.join(cells) + ' |')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
