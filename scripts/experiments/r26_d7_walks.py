"""Round 26's walks of Dungeon 7 by script (docs/routes/d7-walks.md items 5, 6, 8, 9, 10 and the blind crossings):
Link put into a level 7 room by `triforce.probe` with the route's inventory, doors, keys and bombs
(`r26_walks.d7_state`), every trial loaded with the game's own randomness for its seed.

    python scripts/experiments/r26_d7_walks.py digdogger [--seeds N]   # 7:0x1c: kill one small one, out W, back in
    python scripts/experiments/r26_d7_walks.py dodongos [--seeds N]    # 7:0x0c: the E wall bomb among the Dodongos
    python scripts/experiments/r26_d7_walks.py ladder [--seeds N]      # 7:0x49 / 0x18: the ladder refused
    python scripts/experiments/r26_d7_walks.py clip [--seeds N]        # 7:0x1a: CLIP N by the agent's action
    python scripts/experiments/r26_d7_walks.py food                    # 7:0x28: FOOD by the agent's action
    python scripts/experiments/r26_d7_walks.py crossings [--seeds N]   # 0x69, 0x19, 0x39, 0x59 walked blind

It plays the game over many seeded trials: a job of the machine's queue (``jobs.require_job``: ``triforce job run
... --timeout 30m -- .venv/bin/python scripts/experiments/r26_d7_walks.py ...``)."""

import argparse
import collections
import random
import statistics
import tempfile
from pathlib import Path

from triforce.jobs import require_job

from r26_walks import (BOMBS, CUR_OPENED_DOORS, DIGDOGGER_BIG3, DIGDOGGER_SMALL, DODONGO, FOOD, GRUMBLE, OBJ_META,
                       OBJ_STATE, OBJ_TYPE, OBJ_X, OBJ_Y, PLAY, UW_FLAGS_7TO9, Walk, d7_state)

RECORDER, BOMB_SLOT, LADDER = 5, 1, 0x64        # SelectedItemSlot values; LadderSlot ($64)
TMP = Path(tempfile.mkdtemp())


def state(room, entry, **kwargs):
    return d7_state(TMP / f'7_{room:02x}{entry.lower()}_walk.state', room, entry, **kwargs)


def load(path, seed) -> Walk:
    w = Walk(seed=seed)
    w.game.load(path, randomize_rng=True, seed=seed)
    return w


def nearest(w) -> str:
    best = min(((max(abs(o[2] - w.x), abs(o[3] - w.y)), o[1]) for o in w.objects()), default=None)
    return f'${best[1]:02X}' if best else '?'


class Tally:
    """Presses one frame at a time and books every heart lost to the object nearest to Link."""

    def __init__(self, w):
        self.w, self.by = w, collections.Counter()

    def press(self, buttons):
        h = self.w.hearts()
        self.w.press(buttons, 1)
        if self.w.hearts() < h:
            self.by[nearest(self.w)] += h - self.w.hearts()

    def go(self, axis, value, limit=400):
        w = self.w
        for _ in range(limit):
            cur = w.x if axis == 'x' else w.y
            if cur == value or w.mode != PLAY:
                return cur == value
            self.press({'x': ('RIGHT', 'LEFT'), 'y': ('DOWN', 'UP')}[axis][0 if value > cur else 1])
        return False


def small(w):
    return [o for o in w.objects() if o[1] == DIGDOGGER_SMALL and o[5] == 0]


def digdogger(args):
    path = state(0x1C, 'W')
    # 1. the death state for one, two, none (the Nov 13 walk's method), out W and back
    for kills in (0, 1, 2):
        w = load(path, 0)
        w.select_b(RECORDER)
        w.press('RIGHT', 6)
        legal = w.game.legal
        w.press('B', 1)
        n = w.until(lambda w: len(small(w)) == 3, 'NONE', 400)
        for o in small(w)[:kills]:
            w.game.write(OBJ_META + o[0], 0x10)
        w.press('NONE', 40)
        w.path(('y', 0x8D), ('x', 0x20))
        w.until(lambda w: w.room == 0x1B and w.mode == PLAY, 'LEFT', 300)
        flag = w.r(UW_FLAGS_7TO9 + 0x1C)
        w.press('NONE', 2)
        w.until(lambda w: w.room == 0x1C and w.mode == PLAY, 'RIGHT', 300)
        w.press('NONE', 60)
        left = [f'${o[1]:02X}' for o in w.objects()]
        w.path(('x', 0x78))
        w.goto(y=0x5D, limit=200)
        out = w.until(lambda w: w.room == 0x0C, 'UP', 200) >= 0
        print(f'{kills} put in the death state: mask at X $18 {legal!r}; three small ones {n + 1} frames after B; '
              f"0x1c's flag ${flag:02X} after leaving; back in: objects {left}, doors ${w.r(CUR_OPENED_DOORS):02X}, "
              f'N into 0x0c: {out}')
    # 2. a sword kill (stand inside the W door facing E, swing at a small one in reach), then out and back
    results, lost = collections.Counter(), []
    for seed in range(args.seeds):
        w = load(path, seed)
        w.select_b(RECORDER)
        h0 = w.hearts()
        w.press('RIGHT', 6)
        w.press('B', 1)
        w.until(lambda w: len(small(w)) == 3, 'NONE', 400)
        killed = False
        for _ in range(1500):
            if any(0 <= o[2] - w.x < 0x1C and abs(o[3] - w.y) < 12 for o in small(w)):
                w.press('A', 1)
                w.press('NONE', 8)
            else:
                w.press('NONE', 1)
            if len([o for o in w.objects() if o[1] == DIGDOGGER_SMALL and o[5] < 0x10]) < 3:
                killed = True
                break
            if w.hearts() <= 2:
                break
        if killed:
            w.press('NONE', 30)
            w.until(lambda w: w.room == 0x1B and w.mode == PLAY, 'LEFT', 300)
            w.press('NONE', 2)
            w.until(lambda w: w.room == 0x1C and w.mode == PLAY, 'RIGHT', 300)
            w.press('NONE', 30)
            empty = not any(o[1] in (DIGDOGGER_SMALL, DIGDOGGER_BIG3) for o in w.objects())
            results[(empty, bool(w.r(CUR_OPENED_DOORS) & 8))] += 1
        else:
            results['no kill'] += 1
        lost.append(h0 - w.hearts())
    print(f'sword kill then out W and back ({args.seeds} seeds): (room empty, N open) {dict(results)}; hearts lost '
          f'{sorted(collections.Counter(lost).items())}')


def dodongos(args):
    path = state(0x0C, 'S')
    for label, waits in (('at once', lambda s: 0), ('after 0-239 frames at the spot', lambda s: random.Random(s).randrange(240))):
        rows = []
        for seed in range(args.seeds):
            w = load(path, seed)
            t = Tally(w)
            for axis, value in (('y', 0xBD), ('x', 0xD0), ('y', 0x8D)):
                t.go(axis, value)
            w.select_b(BOMB_SLOT)
            t.press('RIGHT')
            for _ in range(waits(seed)):
                t.press('NONE')
            at = (w.x, w.y)
            b0 = w.r(BOMBS)
            t.press('B')
            eaten = opened = False
            for _ in range(120):
                if any(w.r(OBJ_STATE + o[0]) == 1 for o in w.objects() if o[1] == DODONGO):
                    eaten = True
                t.press('NONE')
                opened = opened or bool(w.r(CUR_OPENED_DOORS) & 1)
            rows.append((at == (0xD0, 0x8D), w.r(BOMBS) < b0, eaten, opened, t.by))
        print(f'0x0c E wall bomb {label}: (at the spot, placed, eaten, opened) '
              f'{dict(collections.Counter(r[:4] for r in rows))}; hearts lost by {dict(sum((r[4] for r in rows), collections.Counter()))}')


def ladder(args):
    def used(w):
        return sum(1 for s in range(1, 0x0C) if w.r(OBJ_TYPE + s))
    for name, path, edge, top in (('0x49', state(0x49, 'S'), 0x85, 0x5D), ('0x18 row 5', state(0x18, 'S'), 0xB5, 0x8D)):
        for wait in (0, 120, 300):
            refused, crossed, types = [], 0, collections.Counter()
            for seed in range(args.seeds):
                w = load(path, seed)
                w.press('NONE', wait)
                n = 0
                for _ in range(600):
                    w.press('UP', 1)
                    if w.y == edge and not w.r(LADDER) and used(w) == 11:
                        n += 1
                        types.update(f'${w.r(OBJ_TYPE + s):02X}' for s in range(1, 0x0C))
                    if w.y <= top:
                        crossed += 1
                        break
                if n:
                    refused.append(n)
            print(f'{name}, Up after {wait} frames: ladder refused (11 slots full at the edge) in {len(refused)}/'
                  f'{args.seeds} runs for {sorted(refused)} frames; crossed {crossed}; slot types then '
                  f'{dict(types.most_common(5))}')


def clip(args):
    path = state(0x1A, 'W')
    w = load(path, 0)
    w.game.set_clips(True)
    w.game.set_clips_allowed(True)
    snap, offered = w.game.snapshot(), []
    w.staging = True
    for y in range(0x5D, 0xBE, 0x10):
        for x in range(0x20, 0xD1, 0x10):
            w.game.restore(snap)
            w.game.apply_ram({'link_x': x, 'link_y': y})
            w.press('NONE', 1)
            if (w.x, w.y) == (x, y):
                dirs = [d for d in 'NSWE' if w.game.skip_available(d)]
                if dirs:
                    offered.append((f'({x:#04x},{y:#04x})', ''.join(dirs)))
    print('CLIP offered at (Link teleported there, monsters emptied):', offered)
    results, lost, frames = collections.Counter(), collections.Counter(), []
    for seed in range(args.seeds):
        w = load(path, seed)
        w.game.set_clips(True)
        w.game.set_clips_allowed(True)
        h0, f0 = w.hearts(), w.game.frame
        if w.path(('y', 0x9D), ('x', 0x60), limit=300) < 0:
            results['knocked off the way'] += 1
        elif not w.game.skip_available('N'):
            results['CLIP N not offered on arrival'] += 1
        else:
            w.game.act('CLIP N')
            on_block = (w.x, w.y) == (0x60, 0x8D)
            for _ in range(6):
                if w.mode != PLAY:
                    break
                w.game.act('MOVE E')
            results[('on the block' if on_block else 'clip ended elsewhere', 'stairs' if w.mode != PLAY else 'no stairs')] += 1
            if w.mode != PLAY:
                frames.append(w.game.frame - f0)
        lost[h0 - w.hearts()] += 1
    print(f'from the W entry: Down to $9D, Right to $60, CLIP N, MOVE E ({args.seeds} seeds): {dict(results)}; '
          f'hearts lost {sorted(lost.items())}; frames to the stairs {sorted(frames)}')


def food(args):
    path = state(0x28, 'S')
    w = load(path, 0)
    snap = w.game.snapshot()
    n = w.until(lambda w: 'FOOD' in w.game.legal, 'NONE', 200)
    print(f'standing in the doorway: FOOD offered after {n} frames (-1: never in 200); legal {w.game.legal}')
    w.game.restore(snap)
    n = w.until(lambda w: 'FOOD' in w.game.legal, 'UP', 200)
    print(f'holding Up: FOOD offered after {n} frames at {w.where()}: {w.game.legal}')
    offered = w.game.snapshot()
    change = w.game.act('FOOD N')
    gone = not any(o[1] == GRUMBLE for o in w.objects())
    print(f"act FOOD N: {change.frames} frames + {change.menu_frames} menu frames; InvFood {w.r(FOOD)}, Grumble gone "
          f"{gone}, 0x28's flag ${w.r(UW_FLAGS_7TO9 + 0x28):02X}")
    n = w.goto(y=0x5D, limit=300)
    print(f'Up to the N door: {n} frames; then {w.until(lambda w: w.room == 0x18, "UP", 200)} frames into 0x18')
    w.game.restore(offered)
    change = w.game.act('BOOMERANG N')
    print(f'BOOMERANG N first ({change.frames} frames): legal {w.game.legal}')
    n = w.until(lambda w: 'FOOD' in w.game.legal, 'NONE', 120)
    w.game.act('FOOD N')
    print(f'FOOD offered again {n} frames later; FOOD N then: InvFood {w.r(FOOD)}')


CROSSINGS = (('0x69 S->N, bomb N', 0x69, 'S', [('y', 0x5D)], 'UP', 0x59),
             ('0x19 W->E, bomb E', 0x19, 'W', [('x', 0x50), ('y', 0x7D), ('x', 0xD0), ('y', 0x8D)], 'RIGHT', 0x1A),
             ('0x39 S->W by row 1', 0x39, 'S', [('y', 0x6D), ('x', 0x20), ('y', 0x8D)], None, 0x38),
             ('0x39 S->W by the door row', 0x39, 'S', [('y', 0x8D)], None, 0x38),
             ('0x59 S->N', 0x59, 'S', [], None, 0x49))
EXIT_BUTTON = {0x59: 'UP', 0x1A: 'RIGHT', 0x38: 'LEFT', 0x49: 'UP'}


def crossings(args):
    for name, room, entry, legs, bomb, target in CROSSINGS:
        path = state(room, entry)
        rows = []
        for seed in range(args.seeds):
            w = load(path, seed)
            h0, f0 = w.hearts(), w.game.frame
            t = Tally(w)
            for axis, value in legs:
                t.go(axis, value)
            if bomb:
                w.select_b(BOMB_SLOT)
                t.press(bomb)
                t.press('NONE')
                t.press('B')
                for _ in range(90):
                    t.press('NONE')
            for _ in range(300):
                if w.room == target or w.game.state.game_over:
                    break
                t.press(EXIT_BUTTON[target])
            rows.append((w.room == target, h0 - w.hearts(), w.game.frame - f0, t.by))
        start = collections.Counter()
        if room == 0x39:
            for seed in range(args.seeds):
                w = load(path, seed)
                start.update((f'({o[2]:#04x},{o[3]:#04x})',) for o in w.objects() if o[1] == DIGDOGGER_BIG3)
        print(f"{name}: through {sum(r[0] for r in rows)}/{args.seeds}; hearts lost mean "
              f"{statistics.mean(r[1] for r in rows):.2f} {sorted(collections.Counter(r[1] for r in rows).items())}; "
              f"frames median {statistics.median(r[2] for r in rows)}; by {dict(sum((r[3] for r in rows), collections.Counter()))}"
              + (f'; big Digdogger at the first frame {dict(start)}' if start else ''))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name in ('digdogger', 'dodongos', 'ladder', 'clip', 'food', 'crossings'):
        p = sub.add_parser(name)
        p.add_argument('--seeds', type=int, default=40)
    args = parser.parse_args()
    require_job()
    globals()[args.cmd](args)


if __name__ == '__main__':
    main()
