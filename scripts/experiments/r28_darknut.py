"""Round 28's walks of Dungeon 8's Blue Darknut rooms, 0x5e (five, four shooting statues) and 0x3e (six, two
statues) (docs/routes/d8-walks.md item 3): what a hit, a touch and a bomb do, and a scripted fight.  Link: 13
hearts, the magic sword, no bomb (the route's ledger) unless a measurement says it writes them.

    python scripts/experiments/r28_darknut.py measure [--seeds N]    # hits by side, the beam, a touch, bombs by facing
    python scripts/experiments/r28_darknut.py clear [--seeds N]      # the scripted fight of both rooms
    python scripts/experiments/r28_darknut.py parts [--seeds N]      # the same fight, left by the entry door when low
    python scripts/experiments/r28_darknut.py mixed [--seeds N]      # 0x4e and 0x3f: fought through, and cleared

The fight (`Fighter`) reads positions and facings only, a frame at a time: it stabs a Darknut that stands in the
sword's reach and does not face the blade, steps aside from what is predicted to touch Link (`r28_walks.Dodger`),
and else walks to the nearest Darknut's flank."""

import argparse
import collections
import statistics

from r28_walks import (BOMBS, BOMB_SLOT, CELLAR, DARKNUT_BLUE, DARKNUT_RED, GIBDO, HEARTS, KEYS, OBJ_DIR, OBJ_HP,
                       OBJ_INVINC, OBJ_META, OBJ_TYPE, OBJ_X, OBJ_Y, PLAY, RUPEES, SWORD_BLOCKED, Dodger, Tally28,
                       d8_state, hearts_bytes, load, summary)

VEC = {1: (1, 0), 2: (-1, 0), 4: (0, 1), 8: (0, -1)}
OPP = {1: 2, 2: 1, 4: 8, 8: 4}
BUTTON = {1: 'RIGHT', 2: 'LEFT', 4: 'DOWN', 8: 'UP'}
SIDE = {1: (4, 8), 2: (4, 8), 4: (1, 2), 8: (1, 2)}


def not_full(w):
    """Measurement only: 12.5 of 13 hearts, so the sword throws no beam."""
    hb = hearts_bytes(12.5, 13)
    w.game.set_value('hearts_and_containers', hb['hearts_and_containers'])
    w.game.set_value('partial_hearts', hb['partial_hearts'])


def strike(w, slot, relation='side', beam=False):
    """Measurement only: Link written beside the monster in `slot` and one swing.  `relation`: where the blade comes
    from against the monster's facing: 'front' (the blade's direction opposite to its facing), 'behind' (the same
    direction) or 'side'.  Returns (the HP nibble lost, its facing, the blade's direction)."""
    mx, my, md = w.r(OBJ_X + slot), w.r(OBJ_Y + slot), w.r(OBJ_DIR + slot)
    md = md if md in VEC else 4
    blade = OPP[md] if relation == 'front' else md if relation == 'behind' else SIDE[md][0]
    dist = 0x30 if beam else 0x10
    lx, ly = mx - VEC[blade][0] * dist, my - VEC[blade][1] * dist
    if not (0x18 <= lx <= 0xD8 and 0x58 <= ly <= 0xC0) and relation == 'side':
        blade = SIDE[md][1]
        lx, ly = mx - VEC[blade][0] * dist, my - VEC[blade][1] * dist
    w.game.apply_ram({'link_x': max(0, min(255, lx)), 'link_y': max(0, min(255, ly))})
    w.game.write(OBJ_DIR, blade)
    w.game.write(0x394, 0)                                        # ObjGridOffset of Link
    w.game.write(OBJ_INVINC, 0)
    if not beam:
        not_full(w)
    hp = w.r(OBJ_HP + slot)
    facing = w.r(OBJ_DIR + slot)
    w.press('A', 1)
    for _ in range(40 if not beam else 60):
        hearts = w.r(HEARTS)
        w.press('NONE', 1)
        w.game.write(HEARTS, hearts)
        if w.r(OBJ_TYPE + slot) != DARKNUT_BLUE or w.r(OBJ_META + slot) != 0:
            return hp >> 4, facing, blade
    return (hp - w.r(OBJ_HP + slot)) >> 4, facing, blade


def lone(seed, room=0x5E, keep=1, bombs=None):
    """The room with all but `keep` Darknuts and every fireball taken away each frame is the caller's; here: the
    state loaded, 40 frames in, the slots of the kept ones."""
    w = load(d8_state(room, 'S'), seed)
    w.wait_play()
    if bombs is not None:
        w.game.write(BOMBS, bombs)
    w.press('NONE', 40)
    slots = [o[0] for o in w.objects() if o[1] == DARKNUT_BLUE][:keep]
    return w, slots


def only(w, slots):
    for o in w.objects():
        if o[0] not in slots:
            w.game.write(OBJ_TYPE + o[0], 0)


def measure(args):
    # hits to kill and what the shield does, by where the blade comes from
    for relation in ('side', 'behind', 'front'):
        hits, landed, total = collections.Counter(), 0, 0
        for seed in range(args.seeds):
            w, slots = lone(seed)
            n = 0
            for _ in range(8):
                only(w, slots)
                if w.r(OBJ_TYPE + slots[0]) != DARKNUT_BLUE or w.r(OBJ_META + slots[0]) != 0:
                    break
                lost, _, _ = strike(w, slots[0], relation)
                total += 1
                if lost:
                    landed += 1
                    n += 1
                for _ in range(40):                               # its invincibility runs out
                    only(w, slots)
                    hearts = w.r(HEARTS)
                    w.press('NONE', 1)
                    w.game.write(HEARTS, hearts)
            dead = w.r(OBJ_TYPE + slots[0]) != DARKNUT_BLUE or w.r(OBJ_META + slots[0]) != 0
            hits[n if dead else f'alive after {n}'] += 1
        print(f'magic sword from the {relation}: {landed} of {total} swings took HP; swings that landed until it '
              f'died {dict(hits)}')
    # the beam at full hearts, 48 px away
    for relation in ('side', 'behind', 'front'):
        landed = total = 0
        for seed in range(args.seeds):
            w, slots = lone(seed)
            only(w, slots)
            lost, _, _ = strike(w, slots[0], relation, beam=True)
            total += 1
            landed += 1 if lost else 0
        print(f'magic beam (13 of 13 hearts, Link written 48 px away) from the {relation}: {landed} of {total} took '
              f'HP')
    # a touch
    lost = collections.Counter()
    for seed in range(args.seeds):
        w, slots = lone(seed)
        only(w, slots)
        h = w.hearts()
        w.game.apply_ram({'link_x': w.r(OBJ_X + slots[0]), 'link_y': w.r(OBJ_Y + slots[0])})
        w.game.write(OBJ_INVINC, 0)
        w.press('NONE', 3)
        lost[h - w.hearts()] += 1
    print(f'a touch at 13 containers, no ring: hearts lost {dict(lost)}')
    # bombs by facing: Link's facing when he lays the bomb against the Darknut's when the blast reaches it
    table = collections.defaultdict(collections.Counter)
    kills = collections.Counter()
    for seed in range(args.seeds):
        w, slots = lone(seed, bombs=8)
        slot = slots[0]
        landed = 0
        for trial in range(12):
            only(w, [slot])
            if w.r(OBJ_TYPE + slot) != DARKNUT_BLUE or w.r(OBJ_META + slot) != 0:
                break
            w.game.write(BOMBS, 8)
            mx, my, md = w.r(OBJ_X + slot), w.r(OBJ_Y + slot), w.r(OBJ_DIR + slot)
            md = md if md in VEC else 4
            link_dir = (1, 2, 4, 8)[(seed + trial) % 4]
            # the bomb lies $10 ahead of Link; it is laid about where the Darknut will be when it blows
            bx, by = mx + VEC[md][0] * 0x18, my + VEC[md][1] * 0x18
            lx, ly = bx - VEC[link_dir][0] * 0x10, by - VEC[link_dir][1] * 0x10
            w.game.apply_ram({'link_x': max(0x10, min(0xE0, lx)), 'link_y': max(0x50, min(0xC8, ly))})
            w.game.write(OBJ_DIR, link_dir)
            w.game.write(0x394, 0)
            w.select_b(BOMB_SLOT)
            hp = w.r(OBJ_HP + slot)
            w.press('NONE', 1)
            w.press('B', 1)
            outcome, facing = 'out of the blast', None
            for _ in range(110):
                only(w, [slot])
                hearts = w.r(HEARTS)
                inv = w.r(OBJ_INVINC + slot)
                w.press('NONE', 1)
                w.game.write(HEARTS, hearts)
                gone = w.r(OBJ_TYPE + slot) != DARKNUT_BLUE or w.r(OBJ_META + slot) != 0
                if gone or w.r(OBJ_HP + slot) != hp:
                    outcome, facing = 'damage', md
                    landed += 1
                    break
                if inv == 0 and w.r(OBJ_INVINC + slot) != 0:
                    outcome, facing = 'parried (flash, no damage)', w.r(OBJ_DIR + slot)
                    break
                md = w.r(OBJ_DIR + slot) if w.r(OBJ_DIR + slot) in VEC else md
            if facing is not None:
                rel = ('opposite' if facing == OPP[link_dir] else 'same' if facing == link_dir else 'perpendicular')
                table[rel][outcome] += 1
            else:
                table['-'][outcome] += 1
            for _ in range(50):
                only(w, [slot])
                hearts = w.r(HEARTS)
                w.press('NONE', 1)
                w.game.write(HEARTS, hearts)
        dead = w.r(OBJ_TYPE + slot) != DARKNUT_BLUE or w.r(OBJ_META + slot) != 0
        kills[landed if dead else f'alive after {landed}'] += 1
    print('bombs (8 written), the Darknut\'s facing as the blast reaches it against Link\'s when he laid the bomb:')
    for rel, c in sorted(table.items()):
        print(f'   {rel}: {dict(c)}')
    print(f'   blasts that took HP until it died {dict(kills)}')


# ---- the scripted fight ---------------------------------------------------------------------------------------------

class Fighter:
    """One frame of the fight at a time (see the module docstring)."""

    REACH = (10, 27)                          # the blade's reach ahead of Link (px between the two positions)
    LATERAL = 9

    TARGETS = (DARKNUT_BLUE, DARKNUT_RED, GIBDO)
    SHIELDED = (DARKNUT_BLUE, DARKNUT_RED)

    def __init__(self, w, through=None, near=0x30):
        self.w = w
        self.t = Tally28(w)
        self.d = Dodger(self.t)
        self.cool = 0
        self.swings = self.turns = 0
        self.through, self.near = through, near   # a point to make for, fighting only what comes within `near` px

    def darknuts(self):
        return [o for o in self.w.objects() if o[1] in self.TARGETS and o[5] == 0]

    def strikable(self, beam):
        """[(distance, direction)] of Darknuts the blade (or, at full hearts, the beam) would hurt now."""
        w, out = self.w, []
        for s, kind, x, y, _st, _m, _hp in self.darknuts():
            if w.r(OBJ_INVINC + s):
                continue
            facing = w.r(OBJ_DIR + s) if kind in self.SHIELDED else 0
            for direction, (dx, dy) in VEC.items():
                along = (x - w.x) * dx + (y - w.y) * dy
                lateral = abs((x - w.x) * dy) + abs((y - w.y) * dx)
                far = 0x70 if beam else self.REACH[1]
                if self.REACH[0] <= along <= far and lateral <= self.LATERAL and facing != OPP[direction]:
                    out.append((along, direction))
        return sorted(out)

    def flank(self):
        """The nearest square beside or behind a Darknut (16 px off it, not ahead of it)."""
        w, best = self.w, None
        for s, _t, x, y, _st, _m, _hp in self.darknuts():
            facing = w.r(OBJ_DIR + s)
            facing = facing if facing in VEC else 4
            for direction in (SIDE[facing][0], SIDE[facing][1], OPP[facing]):
                gx, gy = x + VEC[direction][0] * 0x12, y + VEC[direction][1] * 0x12
                if not (0x20 <= gx <= 0xD0 and 0x5D <= gy <= 0xBD):
                    continue
                dist = abs(gx - w.x) + abs(gy - w.y)
                if best is None or dist < best[0]:
                    best = (dist, (gx, gy))
        return best[1] if best else None

    def frame(self):
        w = self.w
        threats = self.d.threats()
        if self.cool:
            self.cool -= 1
            self.t.press('NONE')
            return
        beam = w.hearts() >= w.max_hearts()
        hits = [] if w.r(SWORD_BLOCKED) else self.strikable(beam)   # a Bubble's touch: no sword
        if hits and self.d.danger(w.x, w.y, 0, 0, threats, 0) < 0.35:
            direction = hits[0][1]
            if w.r(OBJ_DIR) == direction:
                self.t.press('A')
                self.swings += 1
                self.cool = 9
            else:
                self.t.press(BUTTON[direction])                   # a frame of the direction turns him
                self.turns += 1
            return
        drops = [o for o in w.objects() if o[1] == 0x60]
        close = [o for o in self.darknuts() if max(abs(o[2] - w.x), abs(o[3] - w.y)) < self.near]
        if self.through is not None and (not close or w.r(SWORD_BLOCKED)):
            goal = self.through
        elif drops and w.hearts() < w.max_hearts():
            goal = min(((abs(o[2] - w.x) + abs(o[3] - w.y), (o[2], o[3])) for o in drops))[1]
        else:
            goal = self.flank() or (w.x, w.y)
        self.t.press(self.d.choose(goal, threats))

    def fight(self, limit=9000, leave_at=None):
        """Until no Darknut is left ('clear'), Link is dead ('dead'), `limit` frames ('time'), or his hearts are at
        or under `leave_at` ('low')."""
        w = self.w
        for _ in range(limit):
            if w.mode != PLAY:
                return 'left'
            if w.hearts() <= 0:
                return 'dead'
            if not [o for o in w.objects() if o[1] in self.TARGETS]:
                return 'clear'
            if self.through is not None and (w.x, w.y) == tuple(self.through):
                return 'there'
            if leave_at is not None and w.hearts() <= leave_at:
                return 'low'
            self.frame()
        return 'time'

    def walk_out(self, points, button, limit=900):
        w, room = self.w, self.w.room
        for p in points:
            if not self.d.go(p, limit):
                return False
        for _ in range(limit):
            if w.mode != PLAY:
                break
            self.t.press(button)
        w.until(lambda w: w.mode in (PLAY, CELLAR), button, 400)
        return w.room != room and w.hearts() > 0


ROOMS = {0x5E: {'take': (0x78, 0x8D), 'out': [(0x78, 0x5D)], 'button': 'UP', 'next': 0x4E, 'back': 0x6E},
         0x3E: {'take': None, 'out': [(0x80, 0x9D), (0xA0, 0x9D), (0xA0, 0x8D), (0xD0, 0x8D)], 'button': 'RIGHT',
                'next': 0x3F, 'back': 0x4E}}


def fight_room(room, seed, hearts=None, limit=9000):
    w = load(d8_state(room, 'S', hearts=hearts), seed)
    w.wait_play()
    f = Fighter(w)
    keys = w.r(KEYS)
    end = f.fight(limit)
    fought = f.t.frames
    ok = False
    if end == 'clear':
        spec = ROOMS[room]
        if spec['take']:
            f.d.go(spec['take'], 600, done=lambda w: w.r(KEYS) > keys)
        ok = f.walk_out(spec['out'], spec['button']) and w.room == spec['next']
    return {'ok': ok, 'end': end, 'lost': f.t.lost(), 'frames': f.t.frames, 'fought': fought, 'by': f.t.by,
            'swings': f.swings, 'keys': (keys, w.r(KEYS)), 'hearts': w.hearts()}


def clear(args):
    for room in (0x5E, 0x3E):
        for hearts in (None, 8.0):
            rows, ends, swings, keys, fought = [], collections.Counter(), [], collections.Counter(), []
            for seed in range(args.seeds):
                r = fight_room(room, seed, hearts)
                rows.append((r['ok'], r['lost'], r['frames'], r['by']))
                ends[r['end']] += 1
                swings.append(r['swings'])
                keys[f'{r["keys"][0]} -> {r["keys"][1]}'] += 1
                if r['ok']:
                    fought.append(r['fought'])
            print(f'8:0x{room:02x} scripted fight from {hearts or 13.0} hearts: {summary(rows)}; fights ended '
                  f'{dict(ends)}; frames to the last kill (median, wins) '
                  f'{int(statistics.median(fought)) if fought else None}; swings mean {statistics.mean(swings):.1f}; '
                  f'keys {dict(keys)}')


def parts(args):
    """The fight left by the entry door (S) whenever Link has lost `step` hearts since he came in, and entered
    again: the kills are banked (LevelKillCounts), the Darknuts left are made anew."""
    for room in (0x5E, 0x3E):
        spec = ROOMS[room]
        for step in (3.0,):
            rows, visits, ends = [], [], collections.Counter()
            for seed in range(args.seeds):
                w = load(d8_state(room, 'S'), seed)
                w.wait_play()
                f = Fighter(w)
                keys, n, end = w.r(KEYS), 1, None
                while n <= 6:
                    end = f.fight(6000, leave_at=max(0.5, w.hearts() - step) if w.hearts() > step else None)
                    if end != 'low':
                        break
                    # out by the entry door and straight back in
                    if not f.d.go((0x78, 0xBD), 600):
                        end = 'dead' if w.hearts() <= 0 else 'stuck'
                        break
                    for button, target in (('DOWN', spec['back']), ('UP', room)):
                        for _ in range(600):
                            if w.room == target and w.mode == PLAY:
                                break
                            f.t.press(button if w.mode != PLAY or w.x == 0x78 else
                                      ('RIGHT' if w.x < 0x78 else 'LEFT'))
                    if w.room != room or w.hearts() <= 0:
                        end = 'dead' if w.hearts() <= 0 else 'stuck'
                        break
                    n += 1
                    f.d.prev = {}
                ok = False
                if end == 'clear':
                    if spec['take']:
                        f.d.go(spec['take'], 600, done=lambda w: w.r(KEYS) > keys)
                    ok = f.walk_out(spec['out'], spec['button']) and w.room == spec['next']
                rows.append((ok, f.t.lost(), f.t.frames, f.t.by))
                visits.append(n)
                ends[end] += 1
            print(f'8:0x{room:02x} in parts (out S after every {step} hearts lost, back in): {summary(rows)}; fights '
                  f'ended {dict(ends)}; visits {dict(collections.Counter(visits))}')


MIXED = {0x4E: {'entry': 'S', 'through': (0x78, 0x5D), 'out': [(0x78, 0x5D)], 'button': 'UP', 'next': 0x3E,
                'take': (0x78, 0x8D)},
         0x3F: {'entry': 'W', 'through': (0xB0, 0x8D), 'out': [(0xB0, 0x8D)], 'button': 'RIGHT', 'next': 0x2F,
                'take': (0x78, 0x8D)}}


def mixed(args):
    """0x4e and 0x3f (two Bubbles, a Blue and two Red Darknuts, three Gibdos): crossed fighting only what comes
    near, and cleared for the room's item (0x4e 5 rupees, 0x3f 4 bombs)."""
    for room, spec in MIXED.items():
        for mode in ('through', 'clear'):
            for hearts in (None, 6.0):
                rows, ends, gains, blocked = [], collections.Counter(), collections.Counter(), 0
                for seed in range(args.seeds):
                    w = load(d8_state(room, spec['entry'], hearts=hearts), seed)
                    w.wait_play()
                    f = Fighter(w, through=spec['through'] if mode == 'through' else None)
                    b0, r0 = w.r(BOMBS), w.r(RUPEES)
                    end = f.fight(6000)
                    ok = False
                    if end in ('clear', 'there'):
                        if end == 'clear':
                            f.d.go(spec['take'], 400, done=lambda w: w.r(BOMBS) > b0 or w.r(RUPEES) > r0)
                        ok = f.walk_out(spec['out'], spec['button']) and w.room == spec['next']
                    rows.append((ok, f.t.lost(), f.t.frames, f.t.by))
                    ends[end] += 1
                    gains[f'bombs {w.r(BOMBS) - b0:+d}, rupees {w.r(RUPEES) - r0:+d}'] += 1
                    blocked += 1 if w.r(SWORD_BLOCKED) else 0
                print(f'8:0x{room:02x} {"crossed, fighting what comes near" if mode == "through" else "cleared"} from '
                      f'{hearts or 13.0} hearts: {summary(rows)}; ended {dict(ends)}; {dict(gains)}; sword blocked on '
                      f'leaving in {blocked}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['measure', 'clear', 'parts', 'mixed'])
    parser.add_argument('--seeds', type=int, default=20)
    args = parser.parse_args()
    globals()[args.what](args)


if __name__ == '__main__':
    main()
