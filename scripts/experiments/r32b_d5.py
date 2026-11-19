"""Round 32's walks of Dungeon 5 with the wood sword (`d3first`: 8 containers on entry; docs/routes/d3first-walks-b.md
§2): what a Blue Darknut takes, the Olive Garden's six by the scripted fight that exists (round 28's
`r28_darknut.Fighter`: stab what stands in reach from the side or behind, step aside from what would touch, else to
the nearest Darknut's flank) with the wood sword and the bombs the route brings, and Digdogger's small one.

    python scripts/experiments/r32b_d5.py measure [--seeds N]    # hits and blasts to kill one Blue Darknut, by side
    python scripts/experiments/r32b_d5.py garden [--seeds N]     # 5:0x05 cleared: wood/white sword, bombs 0-8, hearts
    python scripts/experiments/r32b_d5.py digdogger [--seeds N]  # 5:0x24: the recorder, the small one, wood/white

Every trial loads d2first's own pre-entry state of the room (`5_05e_<c>`, `5_24e_<c>`, c = seed mod 3: natural spawns
and randomness for the seed) and writes what `d3first` holds there: the sword, 8 heart containers (full unless a
line says otherwise), the bombs of the line.  The rest of the inventory is d2first's (the recorder, keys)."""

import argparse
import collections
import statistics

from r26_walks import BOMBS, DIGDOGGER_SMALL, HEARTS, OBJ_DIR, OBJ_HP, OBJ_INVINC, OBJ_META, OBJ_STATE, OBJ_TYPE
from r27_walks import INV_SWORD, hearts_bytes, load, summary
from r28_darknut import BUTTON, OPP, VEC, Fighter, only, strike
from r28_walks import DARKNUT_BLUE, PLAY

BOMB_SLOT, RECORDER_SLOT = 1, 5
BLAST = 0x13                                  # a bomb's ObjState while it blasts (asm:Bomb_CheckBombOrFireVsMonster)


def room(name: str, seed: int, sword: int, hearts: float = 8.0, containers: int = 8, bombs: int = 0):
    w = load(f'{name}_{seed % 3}', seed)
    w.wait_play()
    w.game.write(INV_SWORD, sword)
    for k, v in hearts_bytes(hearts, containers).items():
        w.game.set_value(k, v)
    w.game.write(BOMBS, bombs)
    return w


def measure(args):
    """One Blue Darknut of the Olive Garden alone (the others taken away every frame), Link written beside it and
    one swing at a time: the swings that took HP until it died, by where the blade comes from; then bombs."""
    for sword, label in ((1, 'wood'), (2, 'white')):
        for relation in ('side', 'behind', 'front'):
            hits = collections.Counter()
            for seed in range(args.seeds):
                w = room('5_05e', seed, sword)
                w.press('NONE', 40)
                slots = [o[0] for o in w.objects() if o[1] == DARKNUT_BLUE][:1]
                n = 0
                for _ in range(12):
                    only(w, slots)
                    if w.r(OBJ_TYPE + slots[0]) != DARKNUT_BLUE or w.r(OBJ_META + slots[0]) != 0:
                        break
                    lost, _, _ = strike(w, slots[0], relation)
                    n += 1 if lost else 0
                    for _ in range(40):
                        only(w, slots)
                        h = w.r(HEARTS)
                        w.press('NONE', 1)
                        w.game.write(HEARTS, h)
                dead = w.r(OBJ_TYPE + slots[0]) != DARKNUT_BLUE or w.r(OBJ_META + slots[0]) != 0
                hits[n if dead else f'alive after {n}'] += 1
            print(f'{label} sword from the {relation}: swings that took HP until it died {dict(hits)}')
    # bombs: laid 16 px ahead of where the Darknut will be, Link facing perpendicular or along it (never its front)
    kills = collections.Counter()
    for seed in range(args.seeds):
        w = room('5_05e', seed, 1, bombs=8)
        w.press('NONE', 40)
        slot = [o[0] for o in w.objects() if o[1] == DARKNUT_BLUE][0]
        landed = 0
        for trial in range(8):
            only(w, [slot])
            if w.r(OBJ_TYPE + slot) != DARKNUT_BLUE or w.r(OBJ_META + slot) != 0:
                break
            w.game.write(BOMBS, 8)
            mx, my, md = w.r(0x70 + slot), w.r(0x84 + slot), w.r(OBJ_DIR + slot)
            md = md if md in VEC else 4
            link_dir = md                                       # along its facing: never its front
            bx, by = mx + VEC[md][0] * 0x18, my + VEC[md][1] * 0x18
            lx, ly = bx - VEC[link_dir][0] * 0x10, by - VEC[link_dir][1] * 0x10
            w.game.apply_ram({'link_x': max(0x20, min(0xD0, lx)), 'link_y': max(0x5D, min(0xBD, ly))})
            w.game.write(OBJ_DIR, link_dir)
            w.game.write(0x394, 0)
            w.select_b(BOMB_SLOT)
            hp = w.r(OBJ_HP + slot)
            w.press('NONE', 1)
            w.press('B', 1)
            for _ in range(110):
                only(w, [slot])
                h = w.r(HEARTS)
                w.press('NONE', 1)
                w.game.write(HEARTS, h)
                if w.r(OBJ_TYPE + slot) != DARKNUT_BLUE or w.r(OBJ_META + slot) != 0 or w.r(OBJ_HP + slot) != hp:
                    landed += 1
                    break
            for _ in range(50):
                only(w, [slot])
                h = w.r(HEARTS)
                w.press('NONE', 1)
                w.game.write(HEARTS, h)
        dead = w.r(OBJ_TYPE + slot) != DARKNUT_BLUE or w.r(OBJ_META + slot) != 0
        kills[landed if dead else f'alive after {landed}'] += 1
    print(f'bombs (laid ahead of it, Link facing its way): blasts that took HP until it died {dict(kills)}')


class BombFighter(Fighter):
    """`Fighter` that lays a bomb instead of a swing at a Darknut in the blade's reach while Link holds more than
    `reserve` bombs (the bomb's direction is Link's facing: the same side rule as the blade), then keeps away."""

    def __init__(self, w, reserve=0, **kwargs):
        super().__init__(w, **kwargs)
        self.reserve, self.bombs_laid = reserve, 0
        self.blast_hp = self.sword_hp = 0
        self.hp = {}
        w.select_b(BOMB_SLOT)

    def frame(self):
        w = self.w
        before = {o[0]: o[6] for o in self.darknuts()}
        blasting = any(w.r(OBJ_STATE + s) == BLAST for s in (0x10, 0x11))
        bomb_out = any(0x11 <= w.r(OBJ_STATE + s) <= 0x14 for s in (0x10, 0x11))
        if not self.cool and w.r(BOMBS) > self.reserve and not bomb_out:
            hits = [h for h in self.strikable(False) if h[0] <= 0x1C]
            if hits and w.r(OBJ_DIR) == hits[0][1]:
                self.t.press('B')
                self._book(before, blasting)
                return
        super().frame()
        self._book(before, blasting)

    def _book(self, before, blasting):
        for o in self.w.objects():
            if o[0] in before and o[1] in self.TARGETS and o[6] < before[o[0]]:
                if blasting:
                    self.blast_hp += before[o[0]] - o[6]
                else:
                    self.sword_hp += before[o[0]] - o[6]


def garden(args):
    for sword, label in ((1, 'wood'), (2, 'white')):
        for bombs in args.bombs:
            for hearts in (8.0, 6.0):
                if sword == 2 and (hearts != 8.0 or bombs not in (0, 8)):
                    continue
                rows, ends, laid, frames, left, split = [], collections.Counter(), [], [], [], []
                for seed in range(args.seeds):
                    w = room('5_05e', seed, sword, hearts=hearts, bombs=bombs)
                    f = BombFighter(w)
                    b0 = w.r(BOMBS)
                    end = f.fight(9000)
                    f.bombs_laid = b0 - w.r(BOMBS)
                    ok = end == 'clear'
                    rows.append((ok, f.t.lost(), f.t.frames, f.t.by))
                    ends[end] += 1
                    laid.append(f.bombs_laid)
                    if ok:
                        frames.append(f.t.frames)
                        left.append(w.hearts())
                    split.append((f.sword_hp, f.blast_hp))
                sw = sum(s for s, _ in split)
                bl = sum(b for _, b in split)
                print(f'5:0x05, {label} sword, {bombs} bombs, from {hearts} of 8 hearts: {summary(rows)}; ended '
                      f'{dict(ends)}; bombs used mean {statistics.mean(laid):.1f}; Darknut HP nibbles taken by the '
                      f'sword {sw}, by blasts {bl}; hearts left (cleared) mean '
                      f'{statistics.mean(left) if left else 0:.2f}')


class DigFighter(BombFighter):
    TARGETS = (DIGDOGGER_SMALL,)
    SHIELDED = ()


def digdogger(args):
    for sword, label, bombs in ((1, 'wood', 0), (1, 'wood', 4), (2, 'white', 0)):
        for hearts in (8.0, 5.0):
            rows, ends, swings, killed_at = [], collections.Counter(), [], []
            for seed in range(args.seeds):
                w = room('5_24e', seed, sword, hearts=hearts, bombs=bombs)
                w.select_b(RECORDER_SLOT)
                w.press('LEFT', 6)
                w.press('B', 1)
                w.until(lambda w: any(o[1] == DIGDOGGER_SMALL for o in w.objects()), 'NONE', 400)
                f = DigFighter(w)
                b0 = w.r(BOMBS)
                end = f.fight(6000)
                f.bombs_laid = b0 - w.r(BOMBS)
                rows.append((end == 'clear', f.t.lost(), f.t.frames + 160, f.t.by))
                ends[end] += 1
                swings.append(f.swings)
                killed_at.append(f.bombs_laid)
            print(f'5:0x24, {label} sword, {bombs} bombs, from {hearts} of 8 hearts: recorder, then the small one: '
                  f'{summary(rows)}; ended {dict(ends)}; swings mean {statistics.mean(swings):.1f}; bombs used mean '
                  f'{statistics.mean(killed_at):.1f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['measure', 'garden', 'digdogger'])
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--bombs', type=lambda s: [int(x) for x in s.split(',')], default=[0, 2, 4, 8])
    args = parser.parse_args()
    globals()[args.what](args)


if __name__ == '__main__':
    main()
