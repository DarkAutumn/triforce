"""Phase IX (docs/routes/phase9-walks.md §4): 8:0x4c without its key.  Round 28's walk of the room
(`r28_pols.cross_4c`: out of the block ring W one square, down to row $9D, E to X $B0, up to the top row, W to
($78,$5D), one bomb facing N, through; with the key, the long way round to its corner and back first) played from
the route's own entry state `8_4cc_<chain>` (what the chains bring: 13 hearts, the magic sword, the bow and arrows,
8 bombs, 2 keys) with the game's own randomness for its seed, the eight Pols Voices live.

    python scripts/experiments/r31_d8_walks.py cross [--seeds 20] [--hearts H]

`--hearts H` writes H hearts (of 13) first: the chains enter 0x4c with 4.5 on average (r29land-1k, step 294).

It plays the game over many seeded trials: a job of the machine's queue (``jobs.require_job``: ``triforce job run
... --timeout 30m -- .venv/bin/python scripts/experiments/r31_d8_walks.py ...``)."""

import argparse
import collections

from triforce.jobs import require_job

import r28_pols
from r26_walks import BOMBS, RUPEES, STATES
from r27_walks import hearts_bytes
from r28_walks import PLAY, load, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['cross'])
    p.add_argument('--seeds', type=int, default=20)
    p.add_argument('--hearts', type=float, default=None)
    args = p.parse_args()
    require_job()

    def enter(seed, bombs=None):
        w = load(STATES / f'8_4cc_{seed % 3}.state', seed)
        if args.hearts is not None:
            for k, v in hearts_bytes(args.hearts, 13).items():
                w.game.set_value(k, v)
        w.until(lambda w: w.mode == PLAY, 'NONE', 600)
        return w

    r28_pols.enter = enter
    first = enter(0)
    print(f"8_4cc_0: {first.where()} hearts {first.hearts()} bombs {first.r(BOMBS)} rupees {first.r(RUPEES)}")
    for key in (False, True):
        for mode in ('straight', 'shoot'):
            rows, keys, rupees = [], collections.Counter(), []
            for seed in range(args.seeds):
                row, k, r = r28_pols.cross_4c(seed, key, mode)
                rows.append(row)
                keys[k] += 1
                rupees.append(r)
            print(f"0x4c tunnel -> N wall{', by the corner key' if key else ''}, {mode} (hearts "
                  f"{args.hearts or 'the state'}): {summary(rows)}; keys taken {dict(keys)}; rupees spent mean "
                  f"{sum(rupees) / len(rupees):.1f} max {max(rupees)}")


if __name__ == '__main__':
    main()
