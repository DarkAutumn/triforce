"""Phase IX (docs/routes/phase9-walks.md §3): Dungeon 6's 0x78 (the ringleader room) and 0x29's key.  Every trial
loads the route's own entry state (`6_78e_<chain>`, `6_29w_<chain>`: 12 of 12 hearts, the magic sword) with the
game's own randomness for its seed; the Wizzrobes are live.

    python scripts/experiments/r31_d6_walks.py ringleader [--seeds 100] [--hearts H]
    python scripts/experiments/r31_d6_walks.py key29 [--seeds 20] [--hearts H]

`ringleader`: 0x78 E -> N (5 Red Wizzrobes; slot $01 is the ringleader, asm:Secret_Trigger2Ringleader) by today's
path (the walk of d6-walks.md: W to X $D0, up to Y $7D, W to X $78, Up), with no swing, with up to one or two swings
(SWORD only while a Red stands solid or fading in Link's lane within reach ahead: 24 px, or the whole row when his
hearts are full and the magic sword shoots its beam), and with the swings and a detour to a bomb drop that lies
within 64 px of where Link stands when it appears (taken once it stops flashing).  `--hearts H` writes H hearts (of 12)
first: below full there is no beam.

`key29`: 0x29 W -> S (dark; 3 Blue, 2 Red Wizzrobes; the key lies at ($80,$90) inside the water ring) by today's path
(down the W edge, E along the bottom row, S) and two ways by the key: E along the door row over the ring's W side (the
ladder), the key, back to X $78 and down over the ring's S side; or down the W edge, E along the bottom row to X $80,
up over the ring's S side to the key and back down.

It plays the game over many seeded trials: a job of the machine's queue (``jobs.require_job``: ``triforce job run
... --timeout 30m -- .venv/bin/python scripts/experiments/r31_d6_walks.py ...``)."""

import argparse
import collections
import statistics

from triforce.jobs import require_job

from r26_walks import BOMBS, KEYS, OBJ_META, OBJ_STATE, OBJ_TYPE, OBJ_X, OBJ_Y, STATES
from r27_walks import WIZZROBE_RED, Tally, hearts_bytes, load, name

ITEM_ID, DROPPED, BOMB_ITEM = 0xAC, 0x60, 0x00     # asm:Item_ObjItemId (alias of ObjState), dropped item type, id
LIFETIME = 0x3A8                                     # asm:Item_ObjItemLifetime: >= $F0 flashing, not takeable
FACING = {'LEFT': (-1, 0), 'RIGHT': (1, 0), 'UP': (0, -1), 'DOWN': (0, 1)}
OBJ_DIR_LINK = 0x98


def facing(w) -> str:
    return {1: 'RIGHT', 2: 'LEFT', 4: 'DOWN', 8: 'UP'}.get(w.r(OBJ_DIR_LINK), 'LEFT')


def red_in_reach(w, reach: int):
    """A Red Wizzrobe that can be hit now (fading or solid, not hidden: ObjState $40-$FE) in Link's lane ahead."""
    dx, dy = FACING[facing(w)]
    for o in w.objects():
        slot, typ, x, y, state, meta = o[:6]
        if typ != WIZZROBE_RED or meta != 0 or not 0x40 <= state <= 0xFE:
            continue
        ax, ay = x - w.x, y - w.y
        if dx and abs(ay) < 10 and 0 < ax * dx <= reach:
            return slot
        if dy and abs(ax) < 10 and 0 < ay * dy <= reach:
            return slot
    return None


class Crossing(Tally):
    """Today's path, pressing SWORD (A, one frame, then released) at a Red in reach up to `swings` times, only
    outside the doorway (Link cannot swing there) and 12 frames apart; `leader_only`: at the ringleader (slot $01)
    alone.  `pick`: a bomb drop within 64 px is taken (after its flashing) before the path goes on."""

    def __init__(self, w, swings=0, pick=False, leader_only=False):
        super().__init__(w)
        self.swings, self.pick, self.leader_only, self.swung, self.last = swings, pick, leader_only, 0, -99
        self.dead_slots, self.drops, self.leader, self.first = set(), collections.Counter(), None, None
        self.seen_drops = set()

    def watch(self):
        w = self.w
        for o in w.objects():
            slot, typ, x, y, state, meta = o[:6]
            if slot == 1 and typ == WIZZROBE_RED and state == 0xFE and self.first is None:
                self.first = (x, y)
            if typ == WIZZROBE_RED and meta >= 0x10 and slot not in self.dead_slots:
                self.dead_slots.add(slot)
                if slot == 1:
                    self.leader = self.frames
            if typ == DROPPED and slot not in self.seen_drops and w.r(ITEM_ID + slot) != 0xFF:
                self.seen_drops.add(slot)
                self.drops[{0x00: 'bomb', 0x22: 'heart', 0x18: 'rupee', 0x0F: '5 rupees', 0x23: 'fairy',
                            0x21: 'clock'}.get(w.r(ITEM_ID + slot), f'${w.r(ITEM_ID + slot):02X}')] += 1

    def ready(self) -> bool:
        w = self.w
        if self.swung >= self.swings or w.room != 0x78 or self.frames - self.last < 12:
            return False
        if not (0x20 <= w.x <= 0xD0 and 0x5D <= w.y <= 0xBD):
            return False
        slot = red_in_reach(w, 0x100 if w.hearts() >= w.max_hearts() else 24)
        return slot is not None and (slot == 1 or not self.leader_only)

    def press(self, buttons='NONE', frames: int = 1):
        for _ in range(frames):
            if self.ready():
                self.swung += 1
                self.last = self.frames
                super().press('A', 1)
            else:
                super().press(buttons, 1)
            self.watch()

    def take_bombs(self):
        """A bomb drop within 64 px: wait out its flashing, walk onto it, and back."""
        w = self.w
        for slot in range(1, 0x0C):
            if w.r(OBJ_TYPE + slot) == DROPPED and w.r(ITEM_ID + slot) == BOMB_ITEM:
                x, y = w.r(OBJ_X + slot), w.r(OBJ_Y + slot)
                if max(abs(x - w.x), abs(y - w.y)) > 64:
                    continue
                home = (w.x, w.y)
                while w.r(OBJ_TYPE + slot) == DROPPED and w.r(LIFETIME + slot) >= 0xF0:
                    Tally.press(self, 'NONE')
                tx, ty = x & 0xF8, (y & 0xF0) | 0x0D
                self.walk((tx, w.y), (tx, ty), limit=200)
                self.walk((w.x, home[1]), home, limit=300)


def ringleader_trial(path, seed, hearts, swings, pick, leader_only, doorstep):
    w = load(path, seed)
    if hearts is not None:
        for k, v in hearts_bytes(hearts, 12).items():
            w.game.set_value(k, v)
    w.wait_play()
    t = Crossing(w, swings=swings, pick=pick, leader_only=leader_only)
    bombs0 = w.r(BOMBS)
    ok = t.walk((0xD0, 0x8D), limit=400)
    while ok and doorstep and t.frames < doorstep and t.swung < swings:
        t.press('NONE')                                   # facing W on the door row, out of the doorway
    for point in ((0xD0, 0x7D), (0x78, 0x7D)):
        if not ok or not t.walk(point, limit=400):
            ok = False
            break
        if pick:
            t.take_bombs()
    ok = ok and t.leave('UP', 400, line=('x', 0x78)) and w.room == 0x68
    return {'ok': ok, 'lost': t.lost(), 'frames': t.frames, 'swung': t.swung, 'kills': len(t.dead_slots),
            'leader': t.leader is not None, 'drops': t.drops, 'bombs': w.r(BOMBS) - bombs0, 'first': t.first,
            'by': t.by}


RING_VARIANTS = (('no swing (today)', 0, False, False, 0),
                 ('one swing in passing', 1, False, False, 0),
                 ('two swings in passing', 2, False, False, 0),
                 ('doorstep: up to two swings until frame 60', 2, False, False, 60),
                 ('doorstep, the leader only, until frame 60', 2, False, True, 60),
                 ('doorstep two swings + bomb drops taken', 2, True, False, 60))


def cmd_ringleader(args):
    states = [STATES / f'6_78e_{c}.state' for c in range(3)]
    for label, swings, pick, leader_only, doorstep in RING_VARIANTS:
        rows = [ringleader_trial(states[s % 3], s, args.hearts, swings, pick, leader_only, doorstep)
                for s in range(args.seeds)]
        n = len(rows)
        drops = sum((r['drops'] for r in rows), collections.Counter())
        print(f"0x78 E->N, {label} (hearts {args.hearts or 'full'}): {sum(r['ok'] for r in rows)}/{n}; hearts lost "
              f"mean {statistics.mean(r['lost'] for r in rows):.2f} max {max(r['lost'] for r in rows)}; frames median "
              f"{statistics.median(r['frames'] for r in rows):.0f} (max {max(r['frames'] for r in rows)}); swings "
              f"{sum(r['swung'] for r in rows)}; runs with a kill {sum(r['kills'] > 0 for r in rows)}, leader killed "
              f"{sum(r['leader'] for r in rows)} (all five dead {sum(r['kills'] == 5 for r in rows)}); kills "
              f"{sum(r['kills'] for r in rows)}; drops {dict(drops)} (runs with a bomb drop "
              f"{sum(r['drops']['bomb'] > 0 for r in rows)}); bombs taken {sum(r['bombs'] for r in rows)} in "
              f"{sum(r['bombs'] > 0 for r in rows)} runs; by {dict(sum((r['by'] for r in rows), collections.Counter()))}")
        if swings == 0:
            firsts = collections.Counter(f'({r["first"][0]:#04x},{r["first"][1]:#04x})' for r in rows if r['first'])
            print(f'    the ringleader (slot $01) first appears at {dict(firsts.most_common())}')


WAYS29 = {
    'today (W edge, bottom row, S)': [(0x20, 0xBD), (0x78, 0xBD)],
    'key by the door row (ladder over the W side), down at X $78': [(0x80, 0x8D), (0x78, 0x8D), (0x78, 0xBD)],
    'key from below (bottom row to X $80, up over the S side and back)': [(0x20, 0xBD), (0x80, 0xBD), (0x80, 0x8D),
                                                                          (0x80, 0xBD), (0x78, 0xBD)],
}


def cmd_key29(args):
    states = [STATES / f'6_29w_{c}.state' for c in range(3)]
    first = load(states[0], 0)
    first.wait_play()
    items = [(name(o[1]), o[2], o[3], first.r(ITEM_ID + o[0])) for o in first.objects() if o[1] >= 0x60]
    print(f'0x29 on the first frame: room-item objects {items}; keys {first.r(KEYS)}')
    for label, points in WAYS29.items():
        rows, keys = [], collections.Counter()
        for s in range(args.seeds):
            w = load(states[s % 3], s)
            if args.hearts is not None:
                for k, v in hearts_bytes(args.hearts, 12).items():
                    w.game.set_value(k, v)
            w.wait_play()
            t = Tally(w)
            k0 = w.r(KEYS)
            ok = t.walk(*points, limit=500) and t.leave('DOWN', 300, line=('x', 0x78)) and w.room == 0x39
            keys[w.r(KEYS) - k0] += 1
            rows.append((ok, t.lost(), t.frames, t.by))
        lost = [r[1] for r in rows]
        print(f"0x29 W->S, {label} (hearts {args.hearts or 'full'}): {sum(r[0] for r in rows)}/{len(rows)}; hearts "
              f"lost mean {statistics.mean(lost):.2f} max {max(lost)}; frames median "
              f"{statistics.median(r[2] for r in rows):.0f}; keys gained {dict(keys)}; by "
              f"{dict(sum((r[3] for r in rows), collections.Counter()))}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['ringleader', 'key29'])
    p.add_argument('--seeds', type=int, default=None)
    p.add_argument('--hearts', type=float, default=None)
    args = p.parse_args()
    require_job()
    args.seeds = args.seeds or (100 if args.cmd == 'ringleader' else 20)
    {'ringleader': cmd_ringleader, 'key29': cmd_key29}[args.cmd](args)


if __name__ == '__main__':
    main()
