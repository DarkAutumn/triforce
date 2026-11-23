"""Round 32's walk of the recorder's four warps on `d3first` (docs/routes/d3first-walks-b.md §1): out of Levels 5,
2, 6 and 8 with the pieces and the whirlwind index this route brings, which facing and how many plays set Link
down at Level 3 (0x74), Level 4 (0x45), Level 2 (0x3c) and Level 1 (0x37).

    python scripts/experiments/r32b_recorder.py [--seeds N]

Each exit is staged by `r32b_walks.out_of` (the probe into the triforce room, the walk onto the piece: the game
ends the level).  The table plays every facing (as he comes out, i.e. down; Right, Up, Left: 2 frames of the
button, then release) with 1, 2 and 3 plays (each later play on the first free frame after the tune's 152-frame
freeze); the chosen sequence is then played on `--seeds` trials with the game's own randomness (monsters live).

It plays the game over many seeded trials: a job of the machine's queue (``jobs.require_job``: ``triforce job run
... --timeout 30m -- .venv/bin/python scripts/experiments/r32b_recorder.py ...``)."""

import argparse
import collections
import statistics
import tempfile
from pathlib import Path

from triforce.jobs import require_job

from r26_walks import OBJ_DIR, OBJ_STATE, OBJ_TYPE, PLAY, TELEPORT_INDEX, WHIRLWIND, WHIRLWIND_SUMMONED
from r27_walks import load
from r32b_walks import L1, L2, L3, L4, L5, L6, L7, L8, ledger_row, out_of

RECORDER = 5
FREEZE = 152
TMP = Path(tempfile.mkdtemp(prefix='r32b-recorder-'))

# (dungeon left, destination level, its landing room, the sequence the route plays: facing button or None, plays)
WARPS = [(5, 3, 0x74, 'RIGHT', 1), (2, 4, 0x45, 'RIGHT', 1), (6, 2, 0x3C, None, 2), (8, 1, 0x37, None, 1)]
LANDING = {1: 0x37, 2: 0x3C, 3: 0x74, 4: 0x45, 5: 0x0B, 6: 0x22, 7: 0x42, 8: 0x6D}


def ride(w, face, plays, limit=1200) -> dict:
    """RECORDER `plays` times (B with slot 5; the first now, each next FREEZE frames later), after 2 frames of
    `face` (None: as he stands).  To the end of the carry: where he is set down."""
    w.select_b(RECORDER)
    w.press('NONE', 1)                      # the first frame after a load reads no input
    turn = 0
    if face:
        w.press(face, 2)
        w.press('NONE', 1)
        turn = 3
    at = [i * FREEZE for i in range(plays)]
    start, steps, h0, carried = w.r(TELEPORT_INDEX), [], w.hearts(), None
    for t in range(limit):
        w.press('B' if t in at else 'NONE', 1)
        if t in at:
            steps.append(w.r(TELEPORT_INDEX))
        if carried is None and w.r(0x522) == 1:
            carried = t
        if t > 300 and w.mode == PLAY and w.r(0x522) == 0 and w.r(OBJ_TYPE + 9) != WHIRLWIND and \
                w.r(OBJ_STATE) == 0 and carried is not None:
            break
    return {'landed': w.room, 'level': w.level, 'index': (start, steps), 'frames': t + 1 + turn,
            'released': (w.x, w.y), 'lost': h0 - w.hearts(), 'carried_at': carried,
            'summoned': w.r(WHIRLWIND_SUMMONED)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--seeds', type=int, default=20)
    args = parser.parse_args()
    require_job()
    for left, dest, landing, face, plays in WARPS:
        w = out_of(left)
        row = ledger_row(w)
        print(f'out of Level {left}: {row}')
        path = TMP / f'out_{left}.state'
        w.game.save_state_file(str(path))
        snap = w.game.snapshot()
        table = []
        for f in (None, 'RIGHT', 'UP', 'LEFT'):
            for n in (1, 2, 3):
                w.game.restore(snap)
                r = ride(w, f, n)
                where = [k for k, v in LANDING.items() if v == r['landed']]
                table.append(f'{f or "down (as out)"} x{n}: 0x{r["landed"]:02x} (L{where[0] if where else "?"}), '
                             f'$523 {r["index"][0]:#04x} -> {[hex(s) for s in r["index"][1]]}, {r["frames"]} frames')
        print('  ' + '\n  '.join(table))
        landed, frames, lost, released = collections.Counter(), [], collections.Counter(), collections.Counter()
        for seed in range(args.seeds):
            t = load(path, seed)
            r = ride(t, face, plays)
            landed[f'0x{r["landed"]:02x}'] += 1
            frames.append(r['frames'])
            lost[r['lost']] += 1
            released[(hex(r['released'][0]), hex(r['released'][1]))] += 1
        print(f'  route: {face or "as out (down)"} x{plays} -> L{dest} 0x{landing:02x}: {args.seeds} seeds landed '
              f'{dict(landed)}; frames median {statistics.median(frames)} (min {min(frames)}, max {max(frames)}); '
              f'hearts lost {dict(lost)}; released at {dict(released)}')


if __name__ == '__main__':
    main()
