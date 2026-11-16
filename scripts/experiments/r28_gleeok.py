"""Round 28's walks of 8:0x3c, the Gleeok with four heads (docs/routes/d8-walks.md item 6).  Link comes in by the
bombed S wall with 13 hearts and the magic sword (`r28_walks.d8_state`).

    python scripts/experiments/r28_gleeok.py watch [--seeds N]     # the heads' sweep, the fireballs, the safe rows
    python scripts/experiments/r28_gleeok.py hits [--seeds N]      # hits per head, the cadence, the flying heads
    python scripts/experiments/r28_gleeok.py leave                 # out by the S wall in mid fight and back
    python scripts/experiments/r28_gleeok.py kill [--seeds N] [--hearts H] [--row Y]   # the scripted fight

Addresses: the necks' segment arrays and the shared head byte as `crates/zelda/src/game.rs` reads them
(asm:Gleeok_NeckXAddrsLo, asm:GleeokDeadNeckMask; the attached heads share slot 5's HP and timer)."""

import argparse
import collections
import statistics

from r28_walks import (ALL_DEAD, CUR_OPENED_DOORS, FIREBALL_BOSS, FLYING_HEAD, GLEEOK4, HEARTS, OBJ_DIR, OBJ_HP,
                       OBJ_INVINC, OBJ_META, OBJ_STATE, OBJ_TYPE, OBJ_X, OBJ_Y, PLAY, ROOM_KILL_COUNT, TRIFORCE,
                       UW_FLAGS_7TO9, LEVEL_KILL_COUNTS, Dodger, Tally28, d8_state, hearts_bytes, load, name, summary)

NECK_X, NECK_Y = (0x438, 0x452, 0x46C, 0x395), (0x445, 0x45F, 0x479, 0x3BD)
DEAD_MASK, HEAD = 0x511, 5                    # GleeokDeadNeckMask; the slot whose HP and timer the heads share
BASE = (0x7C, 0x6F)


def heads(w):
    """[(neck, x, y)] of the attached heads."""
    if w.r(OBJ_TYPE + 1) != GLEEOK4 or 0x10 <= w.r(OBJ_META + 1) <= 0x13:    # gone, or in its death
        return []
    dead = w.r(DEAD_MASK)
    return [(n, w.r(NECK_X[n] + 4), w.r(NECK_Y[n] + 4)) for n in range(4) if not dead & (1 << n)]


def flying(w):
    return [o for o in w.objects() if o[1] == FLYING_HEAD]


def fireballs(w):
    return [o for o in w.objects() if o[1] == FIREBALL_BOSS]


def raw_hearts(w) -> int:
    """Link's health in 1/256 hearts (the shown hearts round a 3/8-heart loss away)."""
    return (w.r(HEARTS) & 0x0F) * 256 + w.r(HEARTS + 1)


def enter(seed, hearts=None):
    w = load(d8_state(0x3C, 'S', hearts=hearts), seed)
    w.wait_play()
    return w


def kept(w, buttons='NONE', frames=1):
    """Measurement only: frames with Link's hearts written back."""
    hearts, partial = w.r(HEARTS), w.r(HEARTS + 1)
    for _ in range(frames):
        w.press(buttons, 1)
        w.game.write(HEARTS, hearts)
        w.game.write(HEARTS + 1, partial)


def watch(args):
    xs, ys, born, speeds, first_moves = [], [], [], [], []
    for seed in range(args.seeds):
        w = enter(seed)
        prev, trail, made_at = set(), {}, None
        for f in range(900):
            kept(w)
            for _, x, y in heads(w):
                xs.append(x)
                ys.append(y)
            balls = {o[0]: (o[2], o[3]) for o in fireballs(w)}
            if balls and not prev:
                if made_at is not None:
                    born.append(f - made_at)
                made_at = f
                trail = {'start': f, 'pos': list(balls.values())[0], 'moved': None}
            elif balls and trail:
                pos = list(balls.values())[0]
                if trail['moved'] is None and pos != trail['pos']:
                    trail['moved'] = f
                    first_moves.append(f - trail['start'])
                    trail['from'] = pos
                elif trail['moved'] is not None and f - trail['moved'] == 20:
                    dx, dy = pos[0] - trail['from'][0], pos[1] - trail['from'][1]
                    speeds.append((dx * dx + dy * dy) ** 0.5 / 20)
            prev = set(balls)
    print(f'attached heads over {args.seeds} x 900 frames: X ${min(xs):02X}-${max(xs):02X}, Y ${min(ys):02X}-'
          f'${max(ys):02X} (the neck base stands at (${BASE[0]:02X},${BASE[1]:02X}))')
    print(f'fireballs ($56, one at a time): a new one {min(born)}-{max(born)} frames after the one before was made '
          f'(median {int(statistics.median(born))}, {len(born)} seen); it waits {min(first_moves)}-{max(first_moves)} '
          f'frames, then flies {min(speeds):.2f}-{max(speeds):.2f} px a frame (mean {statistics.mean(speeds):.2f})')
    # what touches Link where: fireballs taken away, Link held on a row under the middle
    for row in (0x8D, 0x95, 0x9D, 0xA5, 0xAD, 0xBD):
        hits, amounts = 0, collections.Counter()
        for seed in range(args.seeds):
            w = enter(seed)
            for f in range(600):
                w.game.write(OBJ_TYPE + 0x0B, 0)                 # no fireball: the heads alone
                w.game.apply_ram({'link_x': 0x78, 'link_y': row})
                w.game.write(OBJ_INVINC, 0)
                w.game.write(0xC0, 0)
                w.game.write(0xD3, 0)
                h = raw_hearts(w)
                w.press('NONE', 1)
                if raw_hearts(w) < h:
                    hits += 1
                    amounts[f'{(h - raw_hearts(w)) / 256:.3f}'] += 1
                hb = hearts_bytes(13, 13)
                w.game.set_value('hearts_and_containers', hb['hearts_and_containers'])
                w.game.set_value('partial_hearts', hb['partial_hearts'])
        print(f'Link held at ($78,${row:02X}), fireballs taken away, {args.seeds} x 600 frames: {hits} touches by '
              f'attached heads or the base, hearts each {dict(amounts)}')
    # a fireball's damage, Link standing still on the entry spot
    amounts = collections.Counter()
    for seed in range(args.seeds):
        w = enter(seed)
        for _ in range(600):
            h = w.hearts()
            near = min(((max(abs(o[2] - w.x), abs(o[3] - w.y)), name(o[1])) for o in w.objects()), default=(0, '?'))
            w.press('NONE', 1)
            if w.hearts() < h:
                amounts[(near[1], h - w.hearts())] += 1
    print(f'Link standing on the entry spot ($78,$BD) 600 frames: hits by (nearest object, hearts) {dict(amounts)}')


def under(w, head, dy=0x10):
    """Measurement only: Link written under a head, facing up."""
    w.game.apply_ram({'link_x': head[1], 'link_y': min(0xBD, head[2] + dy)})
    w.game.write(OBJ_DIR, 8)
    w.game.write(0x394, 0)
    w.game.write(0xC0, 0)
    w.game.write(0xD3, 0)
    w.game.write(OBJ_INVINC, 0)


def swing(w, state={}):
    """Measurement only: one frame; A on it when the heads' timer is 0 and A was not down the frame before (the
    game swings on the press, not while A is held)."""
    press = w.r(OBJ_INVINC + HEAD) == 0 and not state.get('down')
    state['down'] = press
    kept(w, 'A' if press else 'NONE')


def hits(args):
    per_head, totals, gaps, fliers, after = collections.Counter(), collections.Counter(), [], collections.Counter(), []
    hp_seen = collections.Counter()
    for seed in range(args.seeds):
        w = enter(seed)
        hb = hearts_bytes(12.5, 13)                              # no beam: every hit is the blade's
        w.game.set_value('hearts_and_containers', hb['hearts_and_containers'])
        w.game.set_value('partial_hearts', hb['partial_hearts'])
        landed, since_cut, last_hit, max_fly, cuts = 0, 0, None, 0, []
        kept(w, 'NONE', 2)                                       # the heads' byte is set on the room's first frame
        hp, n_heads = w.r(OBJ_HP + HEAD), 4
        dead_at = None
        for f in range(3000):
            hs = heads(w)
            if not hs:
                dead_at = f
                break
            low = max(hs, key=lambda h: h[2])
            under(w, low)
            swing(w)
            now, n_now = w.r(OBJ_HP + HEAD), len(heads(w))
            if now != hp or n_now != n_heads:
                landed += 1
                since_cut += 1
                if last_hit is not None:
                    gaps.append(f - last_hit)
                last_hit = f
                hp_seen[f'${hp:02X} -> ${now:02X}'] += 1
                if n_now != n_heads:
                    cuts.append(since_cut)
                    since_cut = 0
                hp, n_heads = now, n_now
            max_fly = max(max_fly, len(flying(w)))
        per_head[tuple(cuts)] += 1
        totals[landed] += 1
        fliers[max_fly] += 1
        if dead_at is not None:
            log, doors, dead = [], w.r(CUR_OPENED_DOORS), w.r(ALL_DEAD)
            fly_left = len(flying(w))
            for g in range(1, 120):
                kept(w)
                if w.r(ALL_DEAD) and not dead:
                    dead = 1
                    log.append((g, 'room flagged empty'))
                if w.r(CUR_OPENED_DOORS) != doors:
                    doors = w.r(CUR_OPENED_DOORS)
                    log.append((g, f'doors ${doors:02X}'))
            after.append((dead_at, fly_left, len(flying(w)), tuple(log)))
    print(f'magic sword, Link written under the lowest head, A whenever the heads\' timer is 0 ({args.seeds} seeds): '
          f'hits until each head fell {dict(per_head)}; hits in all {dict(totals)}; the shared HP byte went '
          f'{dict(hp_seen)}')
    print(f'   frames between two hits that landed: min {min(gaps)}, median {int(statistics.median(gaps))}; most '
          f'flying heads at once {dict(fliers)}')
    print(f'   the fight took {min(a[0] for a in after)}-{max(a[0] for a in after)} frames; flying heads on the '
          f'frame the last head fell {dict(collections.Counter(a[1] for a in after))}, 120 frames later '
          f'{dict(collections.Counter(a[2] for a in after))}; then {dict(collections.Counter(a[3] for a in after))}')
    # the flying head: can it be hurt, how fast is it
    w = enter(0)
    hb = hearts_bytes(12.5, 13)
    w.game.set_value('hearts_and_containers', hb['hearts_and_containers'])
    w.game.set_value('partial_hearts', hb['partial_hearts'])
    for f in range(3000):
        if flying(w):
            break
        hs = heads(w)
        under(w, max(hs, key=lambda h: h[2]))
        swing(w)
    fl = flying(w)[0]
    slot, steps, swings, touched = fl[0], [], 0, collections.Counter()
    w.game.apply_ram({'link_x': 0x20, 'link_y': 0xBD})
    prev = (w.r(OBJ_X + slot), w.r(OBJ_Y + slot))
    for f in range(600):
        kept(w)
        pos = (w.r(OBJ_X + slot), w.r(OBJ_Y + slot))
        steps.append(((pos[0] - prev[0]) ** 2 + (pos[1] - prev[1]) ** 2) ** 0.5)
        prev = pos
    for i in range(20):
        x, y = w.r(OBJ_X + slot), w.r(OBJ_Y + slot)
        w.game.apply_ram({'link_x': max(0x10, x - 0x10), 'link_y': y})
        w.game.write(OBJ_DIR, 1)
        w.game.write(OBJ_INVINC, 0)
        kept(w, 'A')
        swings += 1
        kept(w, 'NONE', 20)
    alive = w.r(OBJ_TYPE + slot) == FLYING_HEAD and w.r(OBJ_META + slot) == 0
    w.game.write(OBJ_INVINC, 0)
    h = w.hearts()
    w.game.apply_ram({'link_x': w.r(OBJ_X + slot), 'link_y': w.r(OBJ_Y + slot)})
    w.press('NONE', 3)
    print(f'a flying head (type $46, slot {slot}): {statistics.mean(steps):.2f} px a frame over 600 frames (max '
          f'{max(steps):.1f}); after {swings} magic sword swings beside it: {"still there" if alive else "gone"}, HP '
          f'byte ${w.r(OBJ_HP + slot):02X}; Link written onto it: {h - w.hearts()} hearts')


def leave(_args):
    for label, cut in (('before any hit', 0), ('after one head fell and one more hit', 4)):
        w = enter(0)
        hb = hearts_bytes(12.5, 13)
        w.game.set_value('hearts_and_containers', hb['hearts_and_containers'])
        w.game.set_value('partial_hearts', hb['partial_hearts'])
        kept(w, 'NONE', 2)
        landed, hp, n = 0, w.r(OBJ_HP + HEAD), 4
        for _ in range(3000):
            if landed >= cut:
                break
            hs = heads(w)
            under(w, max(hs, key=lambda h: h[2]))
            swing(w)
            if w.r(OBJ_HP + HEAD) != hp or len(heads(w)) != n:
                landed += 1
                hp, n = w.r(OBJ_HP + HEAD), len(heads(w))
        before = (len(heads(w)), w.r(OBJ_HP + HEAD), len(flying(w)), w.r(ROOM_KILL_COUNT))
        w.game.apply_ram({'link_x': 0x78, 'link_y': 0xBD})
        w.game.write(0xC0, 0)
        w.game.write(0xD3, 0)
        out = None
        for f in range(400):
            if w.room == 0x4C and w.mode == PLAY:
                out = f
                break
            kept(w, 'DOWN')
        there = (w.x, w.y, w.r(LEVEL_KILL_COUNTS + 0x3C), w.r(UW_FLAGS_7TO9 + 0x3C))
        for f in range(400):
            if w.room == 0x3C and w.mode == PLAY:
                break
            kept(w, 'UP' if w.x == 0x78 else ('RIGHT' if w.x < 0x78 else 'LEFT'))
        kept(w, 'NONE', 30)
        print(f'{label}: heads {before[0]}, shared HP ${before[1]:02X}, flying heads {before[2]}, RoomKillCount '
              f'{before[3]}; Down from ($78,$BD): in 0x4c after {out} frames at (${there[0]:02X},${there[1]:02X}), '
              f'banked {there[2]}, flag ${there[3]:02X}; back in at ({w.x:#04x},{w.y:#04x}): heads '
              f'{len(heads(w))}, shared HP ${w.r(OBJ_HP + HEAD):02X}, flying heads {len(flying(w))}, dead-neck mask '
              f'${w.r(DEAD_MASK):02X}')


# ---- the scripted fight ---------------------------------------------------------------------------------------------

class GleeokFighter:
    """A frame at a time: stab up at a head that is in the blade's reach above Link (the beam's, at full hearts)
    while the heads' shared timer is 0; else stand on `row` under the head that hangs lowest; step aside from
    fireballs, flying heads and the heads themselves (`r28_walks.Dodger`).  Positions only."""

    def __init__(self, w, row=0x9D):
        self.w, self.row = w, row
        self.t = Tally28(w)
        self.d = Dodger(self.t, harmless=(GLEEOK4,))
        self.cool = self.swings = 0

    def frame(self):
        w = self.w
        hs = heads(w)
        threats = self.d.threats() + [(x, y, 0, 0, 0x0F) for _, x, y in hs] + [(BASE[0], BASE[1], 0, 0, 0x12)]
        if self.cool:
            self.cool -= 1
            self.t.press('NONE')
            return
        beam = w.hearts() >= w.max_hearts()
        ready = w.r(OBJ_INVINC + HEAD) == 0
        reach = [h for h in hs if abs(h[1] - w.x) <= 8 and (10 <= w.y - h[2] <= 27 or (beam and w.y - h[2] >= 10))]
        shots = [th for th in threats if th[4] < 0x0F]
        if ready and reach and self.d.danger(w.x, w.y, 0, 0, shots, 0) < 0.3:
            if w.r(OBJ_DIR) == 8:
                self.t.press('A')
                self.swings += 1
                self.cool = 9
            else:
                self.t.press('UP')
            return
        low = max(hs, key=lambda h: (h[2], -abs(h[1] - w.x)))
        goal = (low[1] & 0xF8, self.row if not beam else max(self.row, 0xAD))
        self.t.press(self.d.choose(goal, threats))

    def fight(self, limit=12000):
        w = self.w
        for _ in range(limit):
            if w.mode != PLAY:
                return 'left'
            if w.hearts() <= 0:
                return 'dead'
            if not heads(w):
                return 'killed'
            self.frame()
        return 'time'


def kill(args):
    rows, ends, swings, fights, got = [], collections.Counter(), [], [], collections.Counter()
    for seed in range(args.seeds):
        w = enter(seed, args.hearts)
        f = GleeokFighter(w, args.row)
        end = f.fight()
        fought, lost_fight = f.t.frames, f.t.lost()
        ok = False
        if end == 'killed':
            containers = w.max_hearts()
            f.d.harmless = set()
            f.d.go((0x28, 0xBD), 900, done=lambda w: w.max_hearts() > containers)
            took = w.max_hearts() > containers
            hearts_then = (w.hearts(), w.max_hearts())
            if f.d.walk((0x78, 0xBD), (0x78, 0x5D), limit=900):
                for _ in range(400):
                    if w.mode != PLAY:
                        break
                    f.t.press('UP')
                w.until(lambda w: w.mode == PLAY, 'UP', 400)
            if w.room == 0x2C:
                w.until(lambda w: w.r(TRIFORCE) == 0xFF, 'UP', 300)
                w.until(lambda w: w.mode == PLAY and w.level == 0, 'NONE', 3000)
                ok = w.level == 0 and w.r(TRIFORCE) == 0xFF
                got[f'container {took}; before the piece {hearts_then[0]}/{hearts_then[1]}'
                    f'; out in 0:0x{w.room:02x} with {w.hearts()}/{w.max_hearts()}' if seed < 3 else
                    f'container {took}'] += 1
            fights.append((fought, lost_fight))
        rows.append((ok, f.t.lost(), f.t.frames, f.t.by))
        ends[end] += 1
        swings.append(f.swings)
    print(f'8:0x3c scripted fight from {args.hearts or 13.0} hearts, standing row ${args.row:02X}: {summary(rows)}; '
          f'fights ended {dict(ends)}; swings mean {statistics.mean(swings):.1f}')
    if fights:
        print(f'   wins: frames to the kill median {int(statistics.median(f[0] for f in fights))}, hearts lost in '
              f'the fight mean {statistics.mean(f[1] for f in fights):.2f}; {dict(got)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['watch', 'hits', 'leave', 'kill'])
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--hearts', type=float, default=None)
    parser.add_argument('--row', type=lambda s: int(s, 0), default=0x9D)
    args = parser.parse_args()
    globals()[args.what](args)


if __name__ == '__main__':
    main()
