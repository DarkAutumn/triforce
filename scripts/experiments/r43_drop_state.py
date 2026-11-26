"""Round 43, Carrier43 (round40.md "Why the Darknut teacher walks past 3x5b's bomb carriers", "The drop state"): the
table of forced-drop counters real play holds at a room entry, which a training reset draws from
(`ram_randomization.drop_state: real`, `env::start_states::drop_state_values`).

    python scripts/experiments/r43_drop_state.py build SCAN.jsonl... [--out triforce/drop_state.json]
    python scripts/experiments/r43_drop_state.py report [--table triforce/drop_state.json]

`build` reads Seeds43's scans of power-on replays (`scripts/experiments/r43_seeds.py scan`: one row a run, `entries`
a room entry each with the pre-entry RAM's `world_kill_count` `$627`, `help_drop_count` `$50`, `help_drop_bomb` `$51`)
and keeps every entry the scan found whose memory edits before it are only the harmless ones (`r43_seeds.HARMLESS_EDITS`:
cheated wall bombs and shop rupees, which no counter reads). It counts the triples (`$627`, `$50`, `$51`) by the kind
of door (`dungeon`: level 1-9; `overworld`: level 0, its caves too) and writes them with their counts. The triples are
written as real play held them: 9% of real entries do not fit a streak computed from the kill count (deaths that count
for one and not the other), so a reset draws a real triple and never computes one counter from another. Reads no
emulator: a plain script, seconds.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

HARMLESS_EDITS = {0x658, 0x66D}           # r43_seeds.HARMLESS_EDITS
ROOT = Path(__file__).resolve().parents[2]


def build(scans, out: Path) -> int:
    counts = {'dungeon': collections.Counter(), 'overworld': collections.Counter()}
    runs, versions = 0, collections.Counter()
    for path in scans:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            run = json.loads(line)
            if 'entries' not in run:
                continue
            runs += 1
            versions[run.get('version')] += 1
            for e in run['entries']:
                if not e.get('found') or not set(e.get('edit_addrs', [])) <= HARMLESS_EDITS:
                    continue
                kind = 'overworld' if e['level'] == 0 else 'dungeon'
                counts[kind][(e['world_kill_count'], e['help_drop_count'], e['help_drop_bomb'])] += 1
    table = {
        'what': "room entries of power-on runs: (WorldKillCount $627, HelpDropCount $50, IsHelpDropBomb $51, count)",
        'built_by': 'scripts/experiments/r43_drop_state.py build',
        'scans': [str(p) for p in scans],
        'runs': runs,
        'versions': dict(sorted(versions.items(), key=lambda kv: str(kv[0]))),
        'entries': {kind: sum(c.values()) for kind, c in counts.items()},
    }
    for kind, c in counts.items():
        table[kind] = [[k, h, b, n] for (k, h, b), n in sorted(c.items())]
    # One triple a line: readable, and a rebuild's diff is the triples that changed.
    text = json.dumps({k: v for k, v in table.items() if k not in counts}, indent=1)[:-2]
    for kind in counts:
        rows = ',\n  '.join(json.dumps(row) for row in table[kind])
        text += f',\n "{kind}": [\n  {rows}\n ]'
    out.write_text(text + '\n}\n', encoding='utf-8')
    json.loads(out.read_text(encoding='utf-8'))
    print(f"{out}: {runs} runs, {table['entries']} entries, triples {len(table['dungeon'])} / {len(table['overworld'])}")
    return 0


def report(path: Path) -> int:
    table = json.loads(path.read_text(encoding='utf-8'))
    bands = [('0', 0, 0), ('1-3', 1, 3), ('4-9', 4, 9), ('10-15', 10, 15), ('16+', 16, 255)]
    for kind in ('dungeon', 'overworld'):
        rows = table[kind]
        total = sum(r[3] for r in rows)
        real = {name: sum(r[3] for r in rows if lo <= r[0] <= hi) / total for name, lo, hi in bands}
        uniform = {name: (min(hi, 15) - lo + 1) / 16 if lo <= 15 else 0.0 for name, lo, hi in bands}
        drawn = {name: 2 / 3 * real[name] + 1 / 3 * uniform[name] for name, _, _ in bands}
        print(f"{kind}: {total} entries, {len(rows)} triples; armed as bombs ($50 = 10, $51 > 0): "
              f"{sum(r[3] for r in rows if r[1] >= 10 and r[2])}")
        for name, _, _ in bands:
            print(f"  K {name:>5}: real {real[name]:6.1%}  drawn {drawn[name]:6.1%}")
        missing = [k for k in range(16) if not any(r[0] == k for r in rows)]
        print(f"  K 0-15 without a triple: {missing or 'none'}")
    return 0


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('scans', nargs='+', type=Path)
    b.add_argument('--out', type=Path, default=ROOT / 'triforce' / 'drop_state.json')
    r = sub.add_parser('report')
    r.add_argument('--table', type=Path, default=ROOT / 'triforce' / 'drop_state.json')
    args = parser.parse_args(argv)
    return build(args.scans, args.out) if args.cmd == 'build' else report(args.table)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
