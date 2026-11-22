"""Round 37: segment chains of a one-network models root against the team's record, paired by seed.

    python scripts/experiments/r37_one_table.py --tag TAG [--chains DIR] [--json OUT] [--segments ...]

Reads ``DIR/TAG-SEG.jsonl`` (``eval-batch``, ``target/r37/one-chains.sh``) and the team's chains of record on the
corrected route (``REF``: ``r36base-{1k,300}-SEG.jsonl``; ``post-d5`` and ``post-d7`` from the dock guard's
``r36guard-1k`` batch, the code every later chain runs).  Per segment: success of both, the paired difference on the
common seeds with its standard error, and the rooms whose runs lost per 1,000 entering differ most."""
import argparse
import collections
import json
import math
from pathlib import Path

SEGMENTS = ['d3', 'post-d3', 'd4', 'post-d4', 'd1', 'post-d1', 'd2', 'post-d2', 'd5', 'post-d5', 'd7', 'post-d7',
            'd6', 'post-d6', 'd8', 'post-d8', 'd9', 'full']
BASE = Path('/home/leculver/git/triforce/target/r56-chains')
GUARD = Path('/home/leculver/git/triforce/target/worktrees/r36-route/r36-route/chains')


def ref_path(seg: str) -> Path:
    if seg in ('post-d5', 'post-d7'):
        return GUARD / f'r36guard-1k-{seg}.jsonl'
    for size in ('1k', '300'):
        path = BASE / f'r36base-{size}-{seg}.jsonl'
        if path.is_file():
            return path
    raise FileNotFoundError(seg)


def load(path: Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return {r['seed']: r for r in (json.loads(line) for line in f if line.strip())}


def lost_per_room(runs: dict) -> dict:
    """(level, room) -> (runs lost there, runs entering it)."""
    entering, lost = collections.Counter(), collections.Counter()
    for r in runs.values():
        seen = set()
        for room in r['rooms']:
            key = (room['level'], room['room'])
            if key not in seen:
                entering[key] += 1
                seen.add(key)
        if r['result'] != 'success' and r['rooms']:
            last = r['rooms'][-1]
            lost[(last['level'], last['room'])] += 1
    return {k: (lost[k], entering[k]) for k in entering}


def paired(a: dict, b: dict) -> tuple[float, float, int]:
    seeds = sorted(set(a) & set(b))
    d = [(b[s]['result'] == 'success') - (a[s]['result'] == 'success') for s in seeds]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    return 100 * m, 100 * sd / math.sqrt(n), n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--chains', default='/home/leculver/git/triforce-r37-one/target/r37/chains')
    parser.add_argument('--segments', nargs='*', default=SEGMENTS)
    parser.add_argument('--json', default=None)
    args = parser.parse_args()
    rows, out = [], {}
    for seg in args.segments:
        path = Path(args.chains) / f'{args.tag}-{seg}.jsonl'
        if not path.is_file():
            continue
        new, ref = load(path), load(ref_path(seg))
        rate = lambda runs: 100 * sum(r['result'] == 'success' for r in runs.values()) / len(runs)  # noqa: E731
        diff, se, n = paired(ref, new)
        common = set(new) & set(ref)
        a = lost_per_room({s: ref[s] for s in common})
        b = lost_per_room({s: new[s] for s in common})
        moves = []
        for key in set(a) | set(b):
            la, ea = a.get(key, (0, 0))
            lb, eb = b.get(key, (0, 0))
            pa = 1000 * la / ea if ea else 0.0
            pb = 1000 * lb / eb if eb else 0.0
            moves.append((lb - la, key, pa, pb, ea, eb))
        moves.sort(key=lambda m: -abs(m[0]))
        where = '; '.join(f"{lv}:{rm} {pa:.0f} -> {pb:.0f}" + (' (new room)' if ea == 0 else '')
                          for _, (lv, rm), pa, pb, ea, eb in moves[:3] if abs(_) >= 5)
        ends = collections.Counter(r['reason'].split(' in ')[0].split(':')[0] for r in new.values()
                                   if r['result'] != 'success')
        rows.append(f"| {seg} | {rate(ref):.1f}% ({len(ref)}) | {rate(new):.1f}% ({len(new)}) | "
                    f"{diff:+.1f} ± {se:.1f} | {where} |")
        out[seg] = {'team': rate(ref), 'team_runs': len(ref), 'one': rate(new), 'one_runs': len(new),
                    'paired': diff, 'se': se, 'n': n, 'ends': dict(ends.most_common(5)),
                    'rooms': [{'level': lv, 'room': rm, 'team_lost_per_1000': pa, 'one_lost_per_1000': pb,
                               'team_entering': ea, 'one_entering': eb}
                              for _, (lv, rm), pa, pb, ea, eb in moves[:8]],
                    'ref': str(ref_path(seg))}
    print("| segment | 51 networks | one network | paired (points) | rooms that differ most (lost per 1,000 entering: "
          "team -> one) |")
    print("|---|---:|---:|---:|---|")
    print('\n'.join(rows))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
