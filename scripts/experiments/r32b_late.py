"""Round 32's crossings of `d3first`'s late legs (docs/routes/d3first-walks-b.md §6): Level 7 to the grave and the
magic sword (12 containers), Level 6's exit by the recorder to Level 2 and on to Level 8's burned tree, Level 8's exit
by the recorder to Level 1, 0x47's burned-tree heart and Level 9's rock wall.  These are d2first's legs; the walks of
rounds 27, 28 and 29 are played again with what `d3first` holds there (`r32b_walks.D3`), monsters live.

    python scripts/experiments/r32b_late.py grave [--seeds N]   # r27: every crossing to Level 6, the wood sword
                                                                # before the grave; the sword at 12 and 11 containers
    python scripts/experiments/r32b_late.py d8leg [--seeds N]   # r28: the whirlwind out of 0x22 to 0x6d's tree
    python scripts/experiments/r32b_late.py d9leg [--seeds N]   # r29: the whirlwind out of 0x6d to 0x05's wall
    python scripts/experiments/r32b_late.py ledger              # d2first states: rupees, bombs, keys at boundaries

What differs from d2first there is written over the earlier walks' inventories: before the grave the wood sword
(round 27's walks hold the white one), and the keys, bombs and rupees of `d3first`'s ledger (§7); the containers,
pieces, candles, arrows, recorder, raft and ladder are the same on both routes from Level 7 on.

It plays the game over many seeded trials: a job of the machine's queue (``jobs.require_job``: ``triforce job run
... --timeout 30m -- .venv/bin/python scripts/experiments/r32b_late.py ...``)."""

import argparse

from triforce.jobs import require_job

import r27_leg_walks
import r27_walks
import r28_leg_walks
import r28_walks
import r29_route_leg
import r29_route_walks
from r32b_walks import D3


def grave(args):
    inv = dict(D3[7]['inv'])                                     # 12 containers, the wood sword, the red candle
    r27_walks.OUT_OF_D7.clear()
    r27_walks.OUT_OF_D7.update(inv)
    r27_leg_walks.CROSSINGS[:] = [(label, state, 1 if sword == 2 else sword, points, button, line)
                                  for label, state, sword, points, button, line in r27_leg_walks.CROSSINGS]
    print('d3first out of Level 7:', {k: inv[k] for k in ('sword', 'candle', 'triforce', 'keys', 'bombs', 'rupees')},
          f'containers {(inv["hearts_and_containers"] >> 4) + 1}')
    r27_leg_walks.rooms(args)
    r27_leg_walks.sword(args)


def d8leg(args):
    inv = dict(D3[6]['inv'])                                     # pieces 1-5 and 7 (the D6 piece not yet)
    r28_walks.OUT_OF_D6.clear()
    r28_walks.OUT_OF_D6.update(inv)
    print('d3first out of Level 6:', {k: inv[k] for k in ('sword', 'candle', 'triforce', 'keys', 'bombs', 'rupees')},
          f'containers {(inv["hearts_and_containers"] >> 4) + 1}')
    r28_leg_walks.rooms(args)


def d9leg(args):
    inv = dict(D3[8]['inv'])                                     # pieces 1-7 (the D8 piece not yet)
    r29_route_walks.OUT_OF_D8.clear()
    r29_route_walks.OUT_OF_D8.update(inv)
    print('d3first out of Level 8:', {k: inv[k] for k in ('sword', 'candle', 'triforce', 'keys', 'bombs', 'rupees')},
          f'containers {(inv["hearts_and_containers"] >> 4) + 1}')
    r29_route_leg.rooms(argparse.Namespace(seeds=args.seeds, guard=False))


# d2first's start states at the boundaries of its segments (chains 0-2): their purses give each stretch's drops.
LEDGER_STATES = ['2_7ds', '0_3cc', '1_73s', '0_37c', '3_7cs', '0_74c', '4_71s', '0_45c', '5_76s', '5_14s', '0_0bc',
                 '7_79s', '7_2bw', '0_42c', '6_79s', '6_0cs', '0_22c', '8_7es', '8_2cs', '0_6dc', '9_76s']


def ledger(args):
    """Rupees, bombs and keys held in d2first's start states (as their chains left them), chains 0-2."""
    from r26_walks import BOMBS, KEYS, RUPEES, Walk
    for name in LEDGER_STATES:
        row = []
        for chain in range(3):
            w = Walk()
            w.game.load(f'{name}_{chain}', randomize_rng=False)
            row.append((w.r(RUPEES), w.r(BOMBS), w.r(KEYS), w.max_hearts()))
        print(f'{name}: (rupees, bombs, keys, containers) by chain {row}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['grave', 'd8leg', 'd9leg', 'ledger'])
    parser.add_argument('--seeds', type=int, default=20)
    args = parser.parse_args()
    require_job()
    globals()[args.what](args)


if __name__ == '__main__':
    main()
