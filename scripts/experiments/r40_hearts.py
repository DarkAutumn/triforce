"""Round 40 (Hearts41, docs/experiments/round40.md "Hearts in the stored states"): the stored states that hold Link's
hearts in the bytes `env::start_states::hearts_encoding` wrote before 2.0.0.16, and a corrected copy of a snapshot.

    python scripts/experiments/r40_hearts.py audit ROOT... --out FILE.jsonl [--only FAMILY,...]
    python scripts/experiments/r40_hearts.py table FILE.jsonl...
    python scripts/experiments/r40_hearts.py repair SRC DST --audit FILE.jsonl...
    python scripts/experiments/r40_hearts.py verify SRC DST

The game holds hearts as `HeartValues` ($66F: containers - 1 in the high nibble, the full hearts BESIDES the partial
one in the low) and `HeartPartial` ($670).  Play leaves a full heart (low nibble >= 1) beside an empty partial only
in two ways (zelda-asm): mid-refill (`asm:World_FillHearts@CompleteHeart` zeroes the partial and adds the heart,
`World_IsFillingHearts` ($63) still set), and a hit whose fraction equals the partial exactly (`asm:Link_BeHarmed`:
the partial is $FF after every refill to full and each borrow takes 1/256 heart more, so 31 borrows with the red
ring, 63 with the blue, 127 without, all since that refill).  Death (`@HandleDied`) and the old woman's heart
container with fewer than four (`asm:Person_LifeOrMoneyState2Choose@PayWithHeartContainer`) leave (0, $00).  The old
writer wrote a whole number H of hearts below full as (H, $00): a state with a full heart, an empty partial and no
refill running is the old writer's.  ``repair`` writes such a state again with ``hearts_encoding`` at the same half
hearts (H - 1, $FF), through the project's own restore, ``set_value`` and ``save_state_file``; every other file of
the copy is a hard link to the source's (``store.link_tree``), a rewritten one a new file.  ``verify`` proves it per
file: identical bytes, or the reader's half hearts equal, the two bytes changed as the encoding says, and the
decompressed machine state equal but for those two bytes."""

import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path

from triforce import jobs
from triforce._native import Game, Snapshot, hearts_encoding
from triforce.store import link_tree

HEARTS, PARTIAL, FILLING, RING, MODE = 0x66F, 0x670, 0x63, 0x662, 0x12
# TFZSTATE: magic 8, version 2, ROM SHA-1 20, flags 1, zstd length 4, then the zstd body (zcore::state_file).
HEADER = 8 + 2 + 20 + 1 + 4


def state_files(root: Path):
    """Every ``*.state`` under ``root`` (relative paths, sorted), ``power_on/`` aside."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != 'power_on')
        out += [Path(dirpath, name).relative_to(root) for name in filenames if name.endswith('.state')]
    return sorted(out)


def read(game: Game, data: bytes) -> dict:
    """The hearts bytes, refill flag, ring, game mode and the reader's half hearts of a state file's bytes."""
    state = game.restore(Snapshot.from_state_file(data))
    return {'hc': state.read(HEARTS), 'partial': state.read(PARTIAL), 'filling': state.read(FILLING),
            'ring': state.read(RING), 'mode': state.read(MODE), 'half_hearts': round(2 * state.health),
            'max_health': state.max_health}


def old_bytes(rec: dict) -> bool:
    """A full heart beside an empty partial with no refill running and Link not dying: the old writer's (H, $00)."""
    return rec['partial'] == 0 and rec['hc'] & 0x0F >= 1 and rec['filling'] == 0 and rec['mode'] != 0x11


def pattern(rec: dict) -> str:
    """(nibble against containers - 1, partial class) of a record."""
    nibble, top = rec['hc'] & 0x0F, rec['hc'] >> 4
    rel = 'N=C-1' if nibble == top else 'N<C-1' if nibble < top else 'N>C-1'
    p = rec['partial']
    cls = {0xFF: '$FF', 0x7F: '$7F', 0x00: '$00'}.get(p, '$80-$FE' if p >= 0x80 else '$01-$7E')
    return f"{rel} {cls}" + (' filling' if rec['filling'] else '')


def audit(roots: list, out: str, only: set) -> int:
    jobs.require_job()
    game = Game(seed=0)
    todo = [(root, rel) for root in roots for rel in state_files(Path(root))
            if not only or (rel.parts[0] if len(rel.parts) > 1 else '') in only]
    with open(out, 'w', encoding='utf-8') as file:
        for done, (root, rel) in enumerate(todo, 1):
            rec = read(game, (Path(root) / rel).read_bytes())
            rec.update(root=str(root), path=str(rel), family=rel.parts[0] if len(rel.parts) > 1 else '')
            file.write(json.dumps(rec) + '\n')
            if done % 500 == 0:
                jobs.progress(done, len(todo), 'states')
    jobs.progress(len(todo), len(todo), 'states', force=True)
    print(f"{out}: {len(todo)} states")
    return 0


def records(paths: list):
    for path in paths:
        with open(path, encoding='utf-8') as file:
            yield from (json.loads(line) for line in file)


def table(paths: list) -> int:
    counts = collections.defaultdict(collections.Counter)
    olds = collections.defaultdict(collections.Counter)
    for rec in records(paths):
        key = (Path(rec['root']).name, rec['family'] or '(entry states)')
        counts[key][pattern(rec)] += 1
        if old_bytes(rec):
            olds[key][rec['path'].split('/')[-1].rsplit('_', 2)[0]] += 1
    names = sorted({p for c in counts.values() for p in c})
    print('| snapshot | family | states | ' + ' | '.join(names) + ' | old bytes |')
    print('|---|---|---:|' + '---:|' * len(names) + '---:|')
    total = collections.Counter()
    for key in sorted(counts):
        total.update(counts[key])
        print(f"| {key[0]} | {key[1]} | {sum(counts[key].values())} | "
              + ' | '.join(str(counts[key][n] or '') for n in names) + f" | {sum(olds[key].values()) or ''} |")
    print(f"| all | | {sum(total.values())} | " + ' | '.join(str(total[n]) for n in names)
          + f" | {sum(sum(c.values()) for c in olds.values())} |")
    for key in sorted(olds):
        print(f"old bytes in {key[0]}/{key[1]}: {dict(olds[key])}")
    return 0


def repair(src: Path, dst: Path, paths: list) -> int:
    jobs.require_job()
    old = [rec for rec in records(paths) if Path(rec['root']).resolve() == src.resolve() and old_bytes(rec)]
    files, _ = link_tree(src, dst, link=True)
    game = Game(seed=0)
    for rec in old:
        source, target = src / rec['path'], dst / rec['path']
        before = read(game, source.read_bytes())
        assert old_bytes(before), rec['path']
        hc, partial = encoded(before)
        target.unlink()             # a hard link to the store's file: never written through
        game.set_value('hearts_and_containers', hc)
        game.set_value('partial_hearts', partial)
        game.save_state_file(str(target))
        os.chmod(target, os.stat(source).st_mode & 0o777)
    print(f"{dst}: {files} files, {len(old)} rewritten")
    return 0


def encoded(rec: dict) -> tuple[int, int]:
    """The game's bytes for the half hearts the reader reads from ``rec`` (``hearts_encoding``)."""
    out = hearts_encoding(rec['half_hearts'], rec['max_health'])
    return out['hearts_and_containers'], out['partial_hearts']


def body(data: bytes) -> bytes:
    """The decompressed machine state (postcard) of a TFZSTATE file."""
    length = int.from_bytes(data[HEADER - 4:HEADER], 'little')
    return subprocess.run(['zstd', '-dcq'], input=data[HEADER:HEADER + length], capture_output=True,
                          check=True).stdout


def verify(src: Path, dst: Path) -> int:
    jobs.require_job()
    game = Game(seed=0)
    ours, theirs = state_files(dst), state_files(src)
    others = sorted({str(p) for root in (src, dst) for p in Path(root).rglob('*') if p.is_file()}
                    - {str(src / p) for p in theirs} - {str(dst / p) for p in ours})
    assert ours == theirs and not others, (len(ours), len(theirs), others[:5])
    same, changed = 0, []
    for done, rel in enumerate(ours, 1):
        a, b = (src / rel).read_bytes(), (dst / rel).read_bytes()
        if a == b:
            same += 1
        else:
            before, after = read(game, a), read(game, b)
            assert old_bytes(before), rel
            assert before['half_hearts'] == after['half_hearts'], rel
            assert (after['hc'], after['partial']) == encoded(before), rel
            assert a[:HEADER - 4] == b[:HEADER - 4], rel        # magic, version, ROM, executor flags
            x, y = body(a), body(b)
            diff = [i for i in range(max(len(x), len(y))) if i >= min(len(x), len(y)) or x[i] != y[i]]
            assert len(x) == len(y) and [(x[i], y[i]) for i in diff] == \
                [(before['hc'], after['hc']), (before['partial'], after['partial'])], (rel, diff)
            changed.append({'path': str(rel), 'half_hearts': before['half_hearts'],
                            'before': [before['hc'], before['partial']], 'after': [after['hc'], after['partial']],
                            'body_bytes': len(x), 'body_diff_at': diff})
        if done % 500 == 0:
            jobs.progress(done, len(ours), 'states')
    for c in changed:
        print(json.dumps(c))
    print(f"{dst}: {len(ours)} files, {same} byte-identical to {src}, {len(changed)} differ (each: same half hearts, "
          f"the two bytes as hearts_encoding writes them, the rest of the machine state equal)")
    return 0


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit')
    a.add_argument('roots', nargs='+', type=Path)
    a.add_argument('--out', required=True)
    a.add_argument('--only', default='', help="families (first folder) to read, comma separated")
    t = sub.add_parser('table')
    t.add_argument('audits', nargs='+')
    r = sub.add_parser('repair')
    r.add_argument('src', type=Path)
    r.add_argument('dst', type=Path)
    r.add_argument('--audit', nargs='+', required=True)
    v = sub.add_parser('verify')
    v.add_argument('src', type=Path)
    v.add_argument('dst', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'audit':
        return audit(args.roots, args.out, {f for f in args.only.split(',') if f})
    if args.command == 'table':
        return table(args.audits)
    if args.command == 'repair':
        return repair(args.src, args.dst, args.audit)
    return verify(args.src, args.dst)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
