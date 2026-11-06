"""Evaluation restricted to post-D1 start states (the ones carrying the D1 triforce), per specialist and arm.

    python scripts/experiments/post_d1_eval.py AUDIT.json LABEL=EVAL_DIR_OR_PATTERN ...

AUDIT.json: `python -m triforce.entry_states audit --states '*' --loads 1 --output AUDIT.json`. Each LABEL maps
to a pattern with `{s}` for the specialist, e.g. `A=target/r37/eval-{s}-on.json`. Prints episode-weighted success
and real health change over the post-D1 states, plus the pooled total across specialists.
"""
import json
import sys

SPECS = "ow-octorok ow-zora ow-leever-peahat ow-moblin ow-start d2-rope goriya".split()


def aggregate(path, post):
    rows = [v for k, v in json.load(open(path))['by_state'].items() if k in post]
    n = sum(v['episodes'] for v in rows)
    if not n:
        return None
    success = sum(v['success_rate'] * v['episodes'] for v in rows)
    health = sum(v['health_change'] * v['episodes'] for v in rows)
    return n, success, health


def main(audit, arms):
    post = {r['state'] for r in json.load(open(audit)) if r['triforce'] >= 1}
    labels = [a.split('=', 1) for a in arms]
    print(f"{'specialist':18} {'eps':>5} " + ' '.join(f"{label:>17}" for label, _ in labels))
    totals = {label: [0, 0.0, 0.0] for label, _ in labels}
    for s in SPECS:
        cells = []
        for label, pattern in labels:
            agg = aggregate(pattern.format(s=s), post)
            if agg is None:
                cells.append(None)
                continue
            n, succ, health = agg
            totals[label][0] += n
            totals[label][1] += succ
            totals[label][2] += health
            cells.append(f"{100 * succ / n:6.1f}% {health / n:+.2f}")
        if all(c is None for c in cells):
            continue
        n = aggregate(labels[0][1].format(s=s), post)[0]
        print(f"{s:18} {n:5d} " + ' '.join(f"{c or '-':>17}" for c in cells))
    print(f"{'pooled':18} {'':5} " + ' '.join(
        f"{100 * t[1] / t[0]:6.1f}% {t[2] / t[0]:+.2f}".rjust(17) for t in totals.values()))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
