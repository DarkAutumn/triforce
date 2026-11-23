"""Round 26's walks of the leg from the D5 triforce to Dungeon 7 (docs/routes/d7-walks.md items 1-4): the rupees,
bombs and keys of chain arrivals at 0:0x0b, the whirlwind from 0x0b, the shops of 0x44 and 0x34 with 0x34's armos,
and 0x42's pond.

    python scripts/experiments/r26_leg_walks.py ledger HARVEST_DIR... [--chains JSONL...]
    python scripts/experiments/r26_leg_walks.py whirlwind [--states STATE...] [--seeds N]
    python scripts/experiments/r26_leg_walks.py shops [--rupees N]
    python scripts/experiments/r26_leg_walks.py armos [--seeds N]
    python scripts/experiments/r26_leg_walks.py pond

`ledger` reads the 5:0x14 handoff states a D5 chain harvested (`triforce eval-chain --from 5_76s_<c> --harvest
DIR`): the rupees there are the rupees at 0x0b (nothing on the way changes them).  The others start from the route's
own states: 5_14s_<c> walked up to the triforce (the level ends, Link comes out at 0x0b), or the coast states
0_54e_<c> / 0_52e_<c> walked to 0x44 / 0x42 with the monster slots emptied on the way (`Walk.staging`) and the
arrival's inventory written (what is written is printed).  Every trial loads with the game's own randomness for its
seed.  It plays the game over many seeded trials: a job of the machine's queue (``jobs.require_job``: ``triforce
job run ... --timeout 30m -- .venv/bin/python scripts/experiments/r26_leg_walks.py ...``)."""

import argparse
import collections
import glob
import json
import statistics
import tempfile
from pathlib import Path

from triforce.jobs import require_job

from r26_walks import (ARROW, BOMBS, FLUTE_TIMER, FOOD, KEYS, MAGIC_SHIELD, OBJ_DIR, OBJ_STATE, OBJ_TYPE, PLAY,
                       RUPEES, SECRET_COLOR_CYCLE, SUBMODE, TELEPORT_INDEX, TRIFORCE, WHIRLWIND, WHIRLWIND_SUMMONED,
                       Walk)

RECORDER, BOMB_SLOT = 5, 1


def to_0b(state, seed=0) -> Walk:
    """A 5:0x14 state walked Up onto the triforce; returns the Walk on 0x0b's first frame of play."""
    w = Walk(seed=seed)
    w.game.load(state, randomize_rng=True, seed=seed)
    w.until(lambda w: w.r(TRIFORCE) & 0x10, 'UP', 400)
    w.until(lambda w: w.mode == PLAY and w.level == 0, 'NONE', 3000)
    return w


def ledger(args):
    rows = []
    for root in args.dirs:
        for f in sorted(glob.glob(str(Path(root) / '**' / '5_14s_h*.state'), recursive=True)):
            w = Walk()
            w.game.load(f, randomize_rng=False)
            rows.append((w.r(RUPEES), w.r(BOMBS), w.r(KEYS)))
    rupees = [r[0] for r in rows]
    print(f'{len(rows)} arrivals at 5:0x14: rupees min {min(rupees)} median {statistics.median(rupees)} max '
          f'{max(rupees)}; after 80 + 60 in the shops: {min(rupees) - 140}-{max(rupees) - 140}; bombs '
          f'{sorted(collections.Counter(r[1] for r in rows).items())} (mean {statistics.mean(r[1] for r in rows):.2f}); '
          f'keys {dict(collections.Counter(r[2] for r in rows))}')
    for path in args.chains or []:
        runs = [json.loads(line) for line in open(path)]
        ends = [r['rooms'][-1] for r in runs if r['result'] == 'success']
        bombs = [e['bombs_out'] for e in ends]
        print(f'{path}: {len(ends)} of {len(runs)} runs at 0:0x0b; bombs {sorted(collections.Counter(bombs).items())} '
              f'(mean {statistics.mean(bombs):.2f}); keys {dict(collections.Counter(e["keys_out"] for e in ends))}')


def ride(w: Walk, second, face=None, limit=900) -> dict:
    """RECORDER (B with slot 5) now and, if `second`, again `second` frames later; to the end of the carry."""
    w.select_b(RECORDER)
    w.press('NONE', 1)                      # the first frame after a load reads no input
    if face:
        w.press(face, 2)
        w.press('NONE', 1)
    plays, start, h0 = [0] + ([second] if second is not None else []), w.r(TELEPORT_INDEX), w.hearts()
    rooms, steps = [], []
    for t in range(limit):
        w.press('B' if t in plays else 'NONE', 1)
        if t in plays:
            steps.append(w.r(TELEPORT_INDEX))
        key = (w.room, w.mode)
        if not rooms or rooms[-1][1:] != key:
            rooms.append((t, *key))
        if t > 300 and w.mode == PLAY and w.x == 0x80 and w.r(OBJ_STATE) == 0 and w.r(OBJ_TYPE + 9) != WHIRLWIND:
            break
    return {'landed': f'0x{w.room:02x}', 'index': (start, steps), 'frames': t + 1, 'released': (w.x, w.y),
            'facing': w.r(OBJ_DIR), 'lost': h0 - w.hearts(), 'summoned_flag': w.r(WHIRLWIND_SUMMONED),
            'rooms': rooms}


def whirlwind(args):
    states = args.states or [f'5_14s_{c}' for c in range(3)]
    for state in states:
        w = to_0b(state)
        print(f'{state}: out at {w.where()} facing {w.r(OBJ_DIR)}, $523 = {w.r(TELEPORT_INDEX)}')
        snap = w.game.snapshot()
        for second in (None, 140, 151, 152, 180, 216, 217, 230):
            w.game.restore(snap)
            r = ride(w, second)
            print(f'  second play at +{second}: lands {r["landed"]} after {r["frames"]} frames at '
                  f'({r["released"][0]:#04x},{r["released"][1]:#04x}); $523 {r["index"]}')
        for face in ('UP', 'RIGHT', 'LEFT'):
            for second in (None, 160):
                w.game.restore(snap)
                r = ride(w, second, face=face)
                print(f'  facing {face}, {"two plays" if second else "one play"}: lands {r["landed"]}')
        w.game.restore(snap)
        print('  room/mode sequence of two plays (+160):', ride(w, 160)['rooms'])
    landed, lost = collections.Counter(), collections.Counter()
    for seed in range(args.seeds):
        w = to_0b(states[seed % len(states)], seed)
        r = ride(w, 152)
        landed[r['landed']] += 1
        lost[r['lost']] += 1
    print(f'{args.seeds} seeds, plays at +0 and +152: landed {dict(landed)}, hearts lost {dict(lost)}')


def stage_room(state, legs, target, inventory) -> Walk:
    """`state` walked by `legs` (monsters emptied), then into `target` across the edge it faces; `inventory`
    written on arrival."""
    w = Walk()
    w.game.load(state, randomize_rng=False)
    w.staging = True
    w.press('NONE', 1)
    w.path(*legs[:-1])
    w.until(lambda w: w.mode != PLAY, legs[-1], 600)          # monsters emptied until the scroll starts
    w.staging = False
    w.until(lambda w: w.room == target and w.mode == PLAY, legs[-1], 600)
    for name, value in inventory.items():
        w.game.set_value(name, value)
    print(f'staged {w.where()}; written: {inventory}')
    return w


def into_cave(w: Walk, legs) -> int:
    w.staging = True
    w.path(*legs)
    w.staging = False
    w.until(lambda w: w.mode != PLAY, 'UP', 120)
    return w.until(lambda w: w.mode in (0x0B, 0x0C) and w.r(SUBMODE) != 0 and w.y == 0xD5, 'NONE', 600)


def buy(w: Walk, snap, legs, after=220) -> str:
    w.game.restore(snap)
    room = w.room
    r0 = (w.r(RUPEES), w.r(BOMBS), w.r(ARROW), w.r(FOOD), w.r(KEYS), w.r(MAGIC_SHIELD), w.ow_flags(room))
    w.path(*legs, limit=300)
    w.press('NONE', after)
    r1 = (w.r(RUPEES), w.r(BOMBS), w.r(ARROW), w.r(FOOD), w.r(KEYS), w.r(MAGIC_SHIELD), w.ow_flags(room))
    names = ('rupees', 'bombs', 'arrow', 'food', 'keys', 'shield', 'flag')
    return ', '.join(f'{n} {a}->{b}' for n, a, b in zip(names, r0, r1) if a != b) or 'nothing'


SHOP_PATHS = {'right ware (Up to $A5, Right to $98, Up to $9D)': [('y', 0xA5), ('x', 0x98), ('y', 0x9D)],
              'walk-in row (Up to $9D at X $70, Right to $98)': [('y', 0x9D), ('x', 0x98)],
              'left ware (Up to $9D, Left to $58)': [('y', 0x9D), ('x', 0x58)],
              'middle ware (Up to $A5, Right to $78, Up to $9D)': [('y', 0xA5), ('x', 0x78), ('y', 0x9D)]}


def shops(args):
    inv = {'rupees': args.rupees, 'bombs': 3, 'keys': 5, 'hearts_and_containers': 0xAA, 'partial_hearts': 0xFF,
           'triforce': 0x1F, 'whistle': 1}
    w = stage_room('0_54e_0', [('x', 0x78), 'UP'], 0x44, inv)
    print('0x44 cave: in after', into_cave(w, [('y', 0x5D), ('x', 0x40), ('y', 0x4D)]), 'frames,', w.where())
    snap = w.game.snapshot()
    for name, legs in SHOP_PATHS.items():
        print(f'  {name}: {buy(w, snap, legs)}')
    w.game.restore(snap)
    w.path(*SHOP_PATHS['right ware (Up to $A5, Right to $98, Up to $9D)'])
    w.press('NONE', 200)
    w.path(('y', 0xBD), ('x', 0x70))
    w.until(lambda w: w.mode == PLAY, 'DOWN', 400)
    print('  out of the cave at', w.where(), 'flag', hex(w.ow_flags(0x44)))
    out = w.game.snapshot()
    w.staging = True
    w.goto(y=0x5D)
    w.staging = False
    w.until(lambda w: w.mode != PLAY, 'UP', 120)
    w.until(lambda w: w.mode in (0x0B, 0x0C) and w.y == 0xD5, 'NONE', 400)
    w.press('NONE', 160)
    print('  back in the shop after buying: wares', [hex(o[1]) for o in w.objects()], '; right ware again:',
          buy(w, w.game.snapshot(), SHOP_PATHS['right ware (Up to $A5, Right to $98, Up to $9D)']))
    w.game.restore(out)
    w.staging = True
    w.path(('y', 0x5D), ('x', 0x80), ('y', 0x45))
    w.staging = False
    w.until(lambda w: w.room == 0x34 and w.mode == PLAY, 'UP', 300)
    path = Path(tempfile.mkdtemp()) / '0_34s_walk.state'
    w.game.save_state_file(str(path))
    print('0x34 in at', w.where(), '(saved for `armos`:', path, ')')
    return path


ARMOS = (8, 8)      # the secret Armos square (tile column, row): X $40, Y $80


def armos_tiles(w):
    return ' '.join(f'{w.tile(ARMOS[0] + dc, ARMOS[1] + dr):02x}' for dr in (0, 1) for dc in (0, 1))


def armos(args):
    path = shops(argparse.Namespace(rupees=args.rupees))
    sides = {'W': ([('y', 0x6D), ('x', 0x30), ('y', 0x7D)], 'RIGHT'),
             'E': ([('y', 0x6D), ('x', 0x50), ('y', 0x7D)], 'LEFT'),
             'N': ([('y', 0x6D), ('x', 0x40)], 'DOWN'), 'S': ([('y', 0x8D), ('x', 0x40)], 'UP')}
    for side, (legs, push) in sides.items():
        w = Walk()
        w.game.load(str(path), randomize_rng=False, controllable=True)
        w.path(*legs)
        before, objs = armos_tiles(w), {o[0]: o[1] for o in w.objects()}
        woke = stairs = None
        for t in range(150):
            w.press(push if t < 30 else 'NONE', 1)
            new = [o for o in w.objects() if objs.get(o[0]) != o[1]]
            if new and woke is None:
                woke = (t, w.x, w.y, [(hex(o[1]), hex(o[2]), hex(o[3])) for o in new])
            if stairs is None and armos_tiles(w) != before:
                stairs = (t, armos_tiles(w))
        print(f'armos from {side}: tiles {before}; woke {woke}; tiles changed {stairs}')
    for strategy in ('hold', 'back off'):
        entered, lost = 0, collections.Counter()
        for seed in range(args.seeds):
            w = Walk(seed=seed)
            w.game.load(str(path), randomize_rng=True, seed=seed, controllable=True)
            h0 = w.hearts()
            w.path(('y', 0x6D), ('x', 0x30), ('y', 0x7D))
            if strategy == 'hold':
                ok = w.until(lambda w: w.mode != PLAY, 'RIGHT', 400) >= 0
            else:
                w.press('RIGHT', 2)
                w.press('LEFT', 12)
                w.press('NONE', 60)
                w.path(('y', 0x7D))
                ok = w.until(lambda w: w.mode != PLAY, 'RIGHT', 400) >= 0
            entered += ok and w.mode == 0x10
            lost[h0 - w.hearts()] += 1
        print(f'stairs from W, {strategy}: {entered}/{args.seeds} in, hearts lost {sorted(lost.items())}')
    w = Walk()
    w.game.load(str(path), randomize_rng=False, controllable=True)
    w.path(('y', 0x6D), ('x', 0x30), ('y', 0x7D))
    w.press('RIGHT', 2)
    w.press('LEFT', 12)
    w.press('NONE', 60)
    w.path(('y', 0x7D))
    w.until(lambda w: w.mode != PLAY, 'RIGHT', 400)
    into = w.until(lambda w: w.mode in (0x0B, 0x0C) and w.y == 0xD5, 'NONE', 400)
    snap = w.game.snapshot()
    print('0x34 cave in after', into, 'frames')
    for name, legs in SHOP_PATHS.items():
        print(f'  {name}: {buy(w, snap, legs)}')
    w.game.restore(snap)
    w.path(*SHOP_PATHS['right ware (Up to $A5, Right to $98, Up to $9D)'])
    w.press('NONE', 200)
    w.path(('y', 0xBD), ('x', 0x70))
    w.until(lambda w: w.mode == PLAY, 'DOWN', 400)
    print('  out at', w.where(), 'facing', w.r(OBJ_DIR), 'armos tiles', armos_tiles(w), 'flag', hex(w.ow_flags(0x34)))
    out = w.game.snapshot()
    for d in ('NONE', 'DOWN', 'LEFT', 'RIGHT', 'UP'):
        w.game.restore(out)
        objs = {o[0]: o[1] for o in w.objects()}
        w.press(d, 40)
        new = [(hex(o[1]), hex(o[2]), hex(o[3])) for o in w.objects() if objs.get(o[0]) != o[1]]
        print(f'  40 frames of {d} from the exit spot: new objects {new}')


POND = (12, 10)     # the pond's stairs square (tile column, row): X $60, Y $90


def pond_tiles(w):
    return ' '.join(f'{w.tile(POND[0] + dc, POND[1] + dr):02x}' for dr in (0, 1) for dc in (0, 1))


def pond(args):
    inv = {'rupees': 31, 'bombs': 3, 'keys': 5, 'hearts_and_containers': 0xAA, 'partial_hearts': 0xFF,
           'triforce': 0x1F, 'whistle': 1, 'food': 1, 'arrows': 1}
    w = stage_room('0_52e_0', [('x', 0x20), ('y', 0x5D), ('x', 0x70), 'UP'], 0x42, inv)
    w.select_b(RECORDER)
    snap = w.game.snapshot()
    for k in range(0, 9):
        w.game.restore(snap)
        w.press('NONE', 1)
        w.press('UP', k)
        y = w.y
        w.press('B', 1)
        print(f'B after {k} frames of Up (Y {y:#04x}):', 'plays' if w.r(FLUTE_TIMER) == 152 else 'nothing')
    w.game.restore(snap)
    w.press('NONE', 1)
    w.goto(y=0xCD)
    index, log = w.r(TELEPORT_INDEX), []
    w.press('B', 1)
    for t in range(300):
        key = (w.r(SECRET_COLOR_CYCLE), pond_tiles(w), w.ow_flags(0x42))
        if not log or log[-1][1:] != key:
            log.append((t, *key))
        w.press('NONE', 1)
    print('$523', index, '->', w.r(TELEPORT_INDEX), '; (frame after B, cycle, tiles, flag):', log)
    drained = w.game.snapshot()
    n = w.path(('y', 0xBD), ('x', 0x60))
    m = w.until(lambda w: w.mode != PLAY, 'UP', 200)
    k = w.until(lambda w: w.mode == PLAY, 'NONE', 600)
    print('stairs: walk', n, '+', m, 'frames, dungeon play', k, 'frames later at', w.where())
    for leave_at in (152, 200, 225, 300):
        w.game.restore(snap)
        w.press('NONE', 1)
        w.goto(y=0xCD)
        w.press('B', 1)
        w.press('NONE', leave_at - 1)
        cycle = w.r(SECRET_COLOR_CYCLE)
        w.until(lambda w: w.room == 0x52 and w.mode == PLAY, 'DOWN', 300)
        w.until(lambda w: w.room == 0x42 and w.mode == PLAY, 'UP', 300)
        w.press('NONE', 10)
        back = (pond_tiles(w), hex(w.ow_flags(0x42)))
        w.path(('y', 0xBD), ('x', 0x60))
        w.until(lambda w: w.mode != PLAY, 'UP', 120)
        print(f'left S at B+{leave_at} (cycle {cycle:#x}), came back: tiles/flag {back}, Up from X $60 ends at '
              f'{w.where()}')
    w.game.restore(drained)
    w.until(lambda w: w.room == 0x52 and w.mode == PLAY, 'DOWN', 300)
    w.until(lambda w: w.room == 0x42 and w.mode == PLAY, 'UP', 300)
    w.press('UP', 6)
    w.press('B', 1)
    w.press('NONE', 300)
    print('a second play after coming back:', pond_tiles(w), 'flag', hex(w.ow_flags(0x42)), '$523', w.r(TELEPORT_INDEX))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('ledger')
    p.add_argument('dirs', nargs='+')
    p.add_argument('--chains', nargs='*')
    p = sub.add_parser('whirlwind')
    p.add_argument('--states', nargs='*')
    p.add_argument('--seeds', type=int, default=60)
    p = sub.add_parser('shops')
    p.add_argument('--rupees', type=int, default=171)
    p = sub.add_parser('armos')
    p.add_argument('--rupees', type=int, default=171)
    p.add_argument('--seeds', type=int, default=40)
    sub.add_parser('pond')
    args = parser.parse_args()
    require_job()
    {'ledger': ledger, 'whirlwind': whirlwind, 'shops': shops, 'armos': armos, 'pond': pond}[args.cmd](args)


if __name__ == '__main__':
    main()
