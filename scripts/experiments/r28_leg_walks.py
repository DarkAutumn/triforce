"""Round 28's walks of the leg from Dungeon 6's piece to Dungeon 8's entrance (docs/routes/d8-walks.md item 1).
Every trial loads a room's pre-entry state (`r28_walks.leg_states`: the leg walked once from a chain-made state of
the D7 triforce step, so `TeleportingLevelIndex` is what the route's own plays at 0x0b left) with the game's own
randomness for its seed; the monsters are live and the hearts real.

    python scripts/experiments/r28_leg_walks.py whirlwind [--seeds N]  # 0x22: the two plays, their window, a miss
    python scripts/experiments/r28_leg_walks.py rooms [--seeds N]      # every crossing by a scripted straight walk
    python scripts/experiments/r28_leg_walks.py tree [--seeds N]       # 0x6d: where the candle burns the tree

It plays the game over many seeded trials: a job of the machine's queue (``jobs.require_job``: ``triforce job run
... --timeout 30m -- .venv/bin/python scripts/experiments/r28_leg_walks.py ...``).
"""

import argparse
import collections
import statistics

from triforce.jobs import require_job

from r28_walks import (CANDLE_SLOT, LegWalk, OBJ_DIR, OBJ_STATE, OBJ_TYPE, OBJ_X, OBJ_Y, OW_FLAGS, PLAY,
                       RECORDER_SLOT, SECOND_PLAY, STAIRS_TILES, TELEPORT_INDEX, TMP, TREE_TILE, TRIFORCE,
                       WHIRLWIND_SUMMONED, Tally28, leg_states, load, name, ride, summary)

DIRS = {'RIGHT': 1, 'LEFT': 2, 'DOWN': 4, 'UP': 8}


def objs(w, slots=range(1, 0x0C)):
    return [(s, name(w.r(OBJ_TYPE + s)), f'{w.r(OBJ_X + s):02X},{w.r(OBJ_Y + s):02X}') for s in slots
            if w.r(OBJ_TYPE + s)]


# ---- the whirlwind out of 0x22 ---------------------------------------------------------------------------------------

def whirlwind(args):
    states = leg_states(TMP / 'leg')
    w = load(states['0_22c'], 0)
    w.wait_play()
    print(f'0x22 out of Dungeon 6: {w.where()} facing {w.r(OBJ_DIR)} (4 = down), index ${w.r(TELEPORT_INDEX):02X}, '
          f'pieces ${w.r(TRIFORCE):02X}, hearts {w.hearts()}/{w.max_hearts()}, RECORDER in the mask without a route: '
          f'{w.game.is_allowed("RECORDER")}')
    snap = w.game.snapshot()
    # the frames of one ride
    r = ride(w)
    print(f'two plays (B on frames 0 and {SECOND_PLAY}), seed 0: index {[f"${i:02X}" for i in r["index"]]}, the '
          f'whirlwind is made on frame {r["made"]}, picks Link up on frame {r["pickup"]}, he is free in '
          f'0x{r["landed"]:02x} at ({r["at"][0]:#04x},{r["at"][1]:#04x}) facing {r["facing"]} after {r["frames"]} '
          f'frames')
    # the second play's window
    print('the second play on frame N after the first (seed 0; the Lynel live):')
    for second in (None, 100, 150, 151, 152, 153, 154, 180, 200, 215, 216, 217, 218, 219, 220, 222, 225, 230, 260):
        w.game.restore(snap)
        r = ride(w, plays=(0,) if second is None else (0, second), limit=1100)
        print(f'  {second}: index {[f"${i:02X}" for i in r["index"]]}, pickup {r["pickup"]}, lands '
              + (f'0x{r["landed"]:02x} at ({r["at"][0]:#04x},{r["at"][1]:#04x}) after {r["frames"]}' if r['rode']
                 else 'nowhere'))
    # other counts and facings (Link turned by one frame of the direction, on the spot he came out at)
    print('other plays (seed 0):')
    for label, faces in (('three down', 'DDD'), ('one up', 'U'), ('down, up', 'DU'), ('up, up', 'UU'),
                         ('one right', 'R'), ('one left', 'L')):
        w.game.restore(snap)
        w.select_b(RECORDER_SLOT)
        w.press('NONE', 1)
        index, pos = [], (w.x, w.y)
        for i, face in enumerate(faces):
            button = {'D': 'DOWN', 'U': 'UP', 'R': 'RIGHT', 'L': 'LEFT'}[face]
            if w.r(OBJ_DIR) != DIRS[button]:
                w.press(button, 1)
            w.press('B', 1)
            w.press('NONE', 1)
            index.append(w.r(TELEPORT_INDEX))
            if i + 1 < len(faces):
                w.until(lambda w: w.r(0x3C) == 0, 'NONE', 200)   # FluteTimer: the tune's freeze
        moved = (w.x, w.y) != pos
        room = None
        for _ in range(1500):
            w.press('NONE', 1)
            if w.mode == PLAY and w.room != 0x22 and w.r(0x522) == 0 and w.r(OBJ_STATE) == 0:
                room = w.room
                break
            if w.mode == PLAY and w.level != 0:
                room = f'level {w.level}'
                break
        print(f'  {label}: index {[f"${i:02X}" for i in index]}; lands in '
              + (f'0x{room:02x} at ({w.x:#04x},{w.y:#04x})' if isinstance(room, int) else str(room))
              + (f'; the turn moved Link to ({w.x:#04x},{w.y:#04x})' if moved and room is None else ''))
    # many seeds: the Lynel live
    landed, lost, frames, by, misses = collections.Counter(), collections.Counter(), [], collections.Counter(), []
    for seed in range(args.seeds):
        w = load(states['0_22c'], seed)
        w.wait_play()
        t = Tally28(w)
        r = ride(w, press=None)
        # ride() presses through w; book the hearts by re-reading
        landed[f'0x{r["landed"]:02x}' if r['rode'] else 'missed'] += 1
        lost[r['lost']] += 1
        if r['rode']:
            frames.append(r['frames'])
        else:
            misses.append((seed, r))
        del t
    print(f'{args.seeds} seeds, plays on frames 0 and {SECOND_PLAY}, Link not moved: landed {dict(landed)}; hearts '
          f'lost {dict(lost)}; frames to free in 0x3c: median {int(statistics.median(frames)) if frames else None}, '
          f'range {min(frames) if frames else None}-{max(frames) if frames else None}')
    for seed, r in misses:
        print(f'  missed, seed {seed}: {r}')
    # what the Lynel does meanwhile: where it is and whether its shot reaches the row
    hits = collections.Counter()
    for seed in range(args.seeds):
        w = load(states['0_22c'], seed)
        w.wait_play()
        t = Tally28(w)
        t.press('NONE', 700)                                     # no play at all: 700 frames on the exit spot
        hits[t.lost()] += 1
    print(f'{args.seeds} seeds standing 700 frames on the exit spot without playing: hearts lost {dict(hits)}')
    # a missed pickup: what a play in the same visit does, a hit's invincibility, and the recovery
    w = load(states['0_22c'], 0)
    w.wait_play()
    w.__class__ = LegWalk
    w.staging = True
    r = ride(w, limit=520, press=lambda f: 'DOWN' if 310 <= f < 322 else 'NONE')       # off the row before pickup
    i0 = w.r(TELEPORT_INDEX)
    again = ride(w, plays=(0,), limit=400)
    print(f'Link steps off the row (12 frames of Down from frame 310): rode {r["rode"]}, index ${i0:02X}, summoned '
          f'flag {w.r(WHIRLWIND_SUMMONED)}; a third play in the same visit: rode {again["rode"]}, index '
          f'${w.r(TELEPORT_INDEX):02X} (it steps, no whirlwind comes)')
    w = load(states['0_22c'], 0)
    w.wait_play()
    w.__class__ = LegWalk
    w.staging = True

    def flash(f):
        if 340 <= f < 400:
            w.game.write(0x4F0, 0x18)                            # ObjInvincibilityTimer of Link: as after a hit
        return 'NONE'

    r = ride(w, limit=700, press=flash)
    print(f'Link invincible (timer written $18) while the whirlwind passes: rode {r["rode"]}, index '
          f'${w.r(TELEPORT_INDEX):02X}, Link at {w.where()}')
    recovered = collections.Counter()
    for seed in range(min(args.seeds, 20)):
        w = load(states['0_22c'], seed)
        w.wait_play()
        w.__class__ = LegWalk
        w.staging = True                                         # the recovery's arithmetic, not its fight
        r = ride(w, limit=520, press=lambda f: 'DOWN' if 310 <= f < 322 else 'NONE')
        missed_index = w.r(TELEPORT_INDEX)
        # leave S and come back, then net zero: one down, one up
        w.goto(x=0x70)
        w.until(lambda w: w.room == 0x32 and w.mode == PLAY, 'DOWN', 600)
        w.until(lambda w: w.room == 0x22 and w.mode == PLAY, 'UP', 600)
        back = (w.x, w.y, w.r(TELEPORT_INDEX))
        w.goto(y=0xAD)
        w.select_b(RECORDER_SLOT)
        w.press('DOWN', 1)
        w.press('B', 1)
        w.press('NONE', 1)
        i1 = w.r(TELEPORT_INDEX)
        w.until(lambda w: w.r(0x3C) == 0, 'NONE', 200)
        y1 = w.y
        w.press('UP', 1)
        w.press('B', 1)
        w.press('NONE', 1)
        i2 = w.r(TELEPORT_INDEX)
        room = None
        for _ in range(1500):
            w.press('NONE', 1)
            if w.mode == PLAY and w.room != 0x22 and w.r(0x522) == 0 and w.r(OBJ_STATE) == 0:
                room = w.room
                break
        recovered[room] += 1
        if seed == 0:
            print(f'the recovery, seed 0: missed with index ${missed_index:02X}; out S into 0x32 and back in at '
                  f'({back[0]:#04x},{back[1]:#04x}); at Y ${y1:02X} one play facing down ${i1:02X}, a frame of Up '
                  f'one play facing up ${i2:02X}: lands '
                  + (f'0x{room:02x} at ({w.x:#04x},{w.y:#04x})' if room is not None else 'nowhere'))
    print(f'the recovery over {min(args.seeds, 20)} seeds (monsters emptied): lands '
          f'{ {(f"0x{k:02x}" if k is not None else None): v for k, v in recovered.items()} }')
    # two plays facing down after a miss (the wrong recovery)
    w = load(states['0_22c'], 0)
    w.wait_play()
    w.__class__ = LegWalk
    w.staging = True
    ride(w, limit=520, press=lambda f: 'DOWN' if 310 <= f < 322 else 'NONE')
    w.goto(x=0x70)
    w.until(lambda w: w.room == 0x32 and w.mode == PLAY, 'DOWN', 600)
    w.until(lambda w: w.room == 0x22 and w.mode == PLAY, 'UP', 600)
    w.goto(y=0xAD)
    w.press('DOWN', 1)
    r = ride(w, limit=1500)
    print(f'after a miss, two plays facing down again: index {[f"${i:02X}" for i in r["index"]]}, lands '
          f'0x{r["landed"]:02x}')


# ---- every crossing by a scripted straight walk ---------------------------------------------------------------------

# (label, state, the points walked (x first; ('y', x, y): y first), the exit button, the line kept while leaving)
CROSSINGS = [
    ('0x3c whirlwind->S', '0_3cr', [(0x70, 0xAD)], 'DOWN', ('x', 0x70)),
    ('0x4c N->E', '0_4cn', [('y', 0x70, 0x8D)], 'RIGHT', ('y', 0x8D)),
    ('0x4d W->S', '0_4dw', [(0x50, 0x8D), (0x50, 0xBD), (0xC0, 0xBD)], 'DOWN', ('x', 0xC0)),
    ('0x4d W->S by the top row', '0_4dw', [(0x10, 0x8D), (0x10, 0x5D), (0xE0, 0x5D), (0xE0, 0xBD), (0xC0, 0xBD)],
     'DOWN', ('x', 0xC0)),
    ('0x5d N->S (right opening)', '0_5dn', [], 'DOWN', ('x', 0xC0)),
    ('0x5d N->S (right in, left out)', '0_5dn', [('y', 0xC0, 0x8D), (0x30, 0x8D)], 'DOWN', ('x', 0x30)),
]


def crossing(path, seed, points, button, line, limit=900):
    w = load(path, seed)
    w.wait_play()
    w.until(lambda w: w.r(0x522) == 0 and w.r(OBJ_STATE) == 0, 'NONE', 300)   # set down by the whirlwind
    t = Tally28(w)
    start = (w.x, w.y)
    ok = t.walk(*points, limit=limit) if points else True
    ok = ok and t.leave(button, limit, line=line)
    return ok, t.lost(), t.frames, t.by, start, (w.level, w.room, w.x, w.y)


def fires(w):
    """[(slot, x, y, state)] of Link's fires: weapon slots $10/$11 whose ObjState is $21 (moving) or $22
    (standing)."""
    return [(s, w.r(OBJ_X + s), w.r(OBJ_Y + s), w.r(OBJ_STATE + s)) for s in (0x10, 0x11)
            if w.r(OBJ_STATE + s) in (0x21, 0x22)]


def burn_and_enter(path, seed, window=(0xB8, 0xC0), face='LEFT', limit=900):
    """0x6d from N: down to row $5D, into the X `window`, face the tree, the candle, stand until the fire is out,
    onto the stairs and into level 8.  A knockback is walked back; the candle is used again when a fire ended
    off the tree."""
    w = load(path, seed)
    w.wait_play()
    t = Tally28(w)
    uses, burned_at = 0, None
    w.select_b(CANDLE_SLOT)
    for _ in range(limit):
        if w.mode != PLAY or w.hearts() <= 0:
            break
        burned = w.tile(*TREE_TILE) in STAIRS_TILES
        if burned and burned_at is None:
            burned_at = t.frames
        if burned:
            if w.y != 0x5D:
                t.press('DOWN' if w.y < 0x5D else 'UP')
            else:
                t.press('LEFT' if w.x > 0xA0 else 'RIGHT')
        elif w.y != 0x5D:
            t.press('DOWN' if w.y < 0x5D else 'UP')
        elif fires(w):
            t.press('NONE')
        elif w.x > window[1]:
            t.press('LEFT')
        elif w.x < window[0]:
            t.press('RIGHT')
        elif w.r(OBJ_DIR) != DIRS[face]:
            t.press(face)                                        # a frame of the direction: he turns and steps
        else:
            t.press('B')
            uses += 1
            t.press('NONE', 2)
    frames = t.frames
    ok = w.mode != PLAY
    if ok:
        w.until(lambda w: w.mode == PLAY and w.level == 8, 'NONE', 900)
        ok = w.level == 8
    return ok, t.lost(), frames, t.by, uses, burned_at, (w.level, w.room, w.x, w.y)


def rooms(args):
    states = leg_states(TMP / 'leg')
    for label, state, points, button, line in CROSSINGS:
        rows, ends, kinds = [], collections.Counter(), collections.Counter()
        for seed in range(args.seeds):
            ok, lost, frames, by, start, end = crossing(states[state], seed, points, button, line)
            rows.append((ok, lost, frames, by))
            ends[f'{end[0]}:0x{end[1]:02x} ({end[2]:#04x},{end[3]:#04x})'] += 1
            w = load(states[state], seed)
            w.wait_play()
            seen = collections.Counter()
            for _ in range(8):                                   # 240 frames: the edge spawns come late
                w.press('NONE', 30)
                now = collections.Counter(name(o[1]) for o in w.objects() if o[1] < 0x40)
                for k, v in now.items():
                    seen[k] = max(seen[k], v)
            kinds[tuple(sorted(seen.items()))] += 1
        print(f'{label}: in at ({start[0]:#04x},{start[1]:#04x}); {summary(rows)}; arrives {dict(ends)}')
        print(f'   monsters seen in 240 frames standing still (most at once): {dict(kinds)}')
    for label, state, x, face in (('0x6d N->stairs, candle facing W from X $B8-$C0', '0_6dn', (0xB8, 0xC0), 'LEFT'),
                                  ('0x6d N (left opening)->stairs, candle facing E from X $78-$88', '0_6dn_left',
                                   (0x78, 0x88), 'RIGHT')):
        rows, uses, burned, ends = [], collections.Counter(), [], collections.Counter()
        for seed in range(args.seeds):
            ok, lost, frames, by, n, at, end = burn_and_enter(states[state], seed, x, face)
            rows.append((ok, lost, frames, by))
            uses[n] += 1
            if at is not None:
                burned.append(at)
            ends[f'{end[0]}:0x{end[1]:02x} ({end[2]:#04x},{end[3]:#04x})'] += 1
        print(f'{label}: {summary(rows)}; candle uses {dict(uses)}; stairs showing at frame (median) '
              f'{int(statistics.median(burned)) if burned else None}; arrives {dict(ends)}')


# ---- the tree --------------------------------------------------------------------------------------------------------

def tree(args):
    states = leg_states(TMP / 'leg')
    w = load(states['0_6dn'], 0)
    w.wait_play()
    w.__class__ = LegWalk
    w.staging = True
    w.goto(y=0x5D)
    print(f'0x6d from N at X $C0, on row $5D: {w.where()}; tree tile {w.tile(*TREE_TILE):#04x}, room flag '
          f'${w.r(OW_FLAGS + 0x6D):02X}; objects {objs(w, range(1, 0x14))}')
    w.select_b(CANDLE_SLOT)
    snap = w.game.snapshot()

    def use(x, y, direction):
        """Link written to (x, y) facing `direction`, the candle: (frames to the stairs or None, the fire's path)."""
        w.game.restore(snap)
        w.game.apply_ram({'link_x': x, 'link_y': y})
        w.game.write(OBJ_DIR, direction)
        w.press('NONE', 1)
        w.press('B', 1)
        path, shown = [], None
        for f in range(1, 140):
            w.press('NONE', 1)
            lit = fires(w)
            if lit:
                path.append((f, lit[0][1:]))
            if shown is None and w.tile(*TREE_TILE) in STAIRS_TILES:
                shown = f
        return shown, path

    shown, path = use(0xC0, 0x5D, 2)
    print(f'Link at ($C0,$5D) facing W: stairs {shown} frames after B; the fire: first seen frame {path[0][0]} at '
          f'({path[0][1][0]:#04x},{path[0][1][1]:#04x}), standing from frame '
          f'{next(f for f, p in path if p[2] == 0x22)} at ({path[-1][1][0]:#04x},{path[-1][1][1]:#04x}), '
          f'last seen frame {path[-1][0]}; room flag ${w.r(OW_FLAGS + 0x6D):02X}, tiles '
          f'{[hex(w.tile(TREE_TILE[0] + dx, TREE_TILE[1] + dy)) for dx in (0, 1) for dy in (0, 1)]}')
    for label, y, direction, xs in (('facing W on row $5D', 0x5D, 2, range(0xA8, 0xD9)),
                                    ('facing E on row $5D', 0x5D, 1, range(0x68, 0x99))):
        good = [x for x in xs if use(x, y, direction)[0] is not None]
        print(f'{label}: the tree burns from X {[f"${x:02X}" for x in (good[0], good[-1])] if good else None} '
              f'({len(good)} of {len(list(xs))} X written; every X between: '
              f'{good == list(range(good[0], good[-1] + 1)) if good else None})')
    for label, x, direction, ys in (('facing W in the column X $C0 above the row', 0xC0, 2, range(0x3D, 0x7E)),):
        good = [y for y in ys if use(x, y, direction)[0] is not None]
        print(f'{label}: burns from Y {[f"${y:02X}" for y in (good[0], good[-1])] if good else None} ({len(good)} Y)')
    # what the flame does to Link: walking W into it at once, and standing
    for label, button, frames in (('standing', 'NONE', 120), ('walking W at once', 'LEFT', 60),
                                  ('walking W after 40 frames', None, 0)):
        w.game.restore(snap)
        w.staging = False
        w.game.apply_ram({'link_x': 0xC0, 'link_y': 0x5D})
        w.game.write(OBJ_DIR, 2)
        for s in range(1, 0x0C):
            if w.r(OBJ_TYPE + s) < 0x60:
                w.game.write(OBJ_TYPE + s, 0)
        w.press('NONE', 1)
        h = w.hearts()
        w.press('B', 1)
        log = []
        for f in range(1, 200):
            for s in range(1, 0x0C):
                if w.r(OBJ_TYPE + s) < 0x60:
                    w.game.write(OBJ_TYPE + s, 0)
            b = button if button is not None else ('NONE' if f < 40 else 'LEFT')
            hh = w.hearts()
            w.press(b if f <= frames or button is None else 'NONE', 1)
            if w.hearts() < hh:
                log.append((f, hh - w.hearts(), hex(w.x)))
            if w.mode != PLAY:
                break
        print(f'the flame and Link, {label}: hearts lost {h - w.hearts() if w.mode == PLAY else "?"} {log}; '
              f'stairs {w.tile(*TREE_TILE) in STAIRS_TILES or w.mode != PLAY}; Link at {w.where()}')
        w.staging = True
    # walking onto the stairs
    w.game.restore(snap)
    w.game.apply_ram({'link_x': 0xC0, 'link_y': 0x5D})
    w.game.write(OBJ_DIR, 2)
    w.press('NONE', 1)
    w.press('B', 1)
    n = w.until(lambda w: w.tile(*TREE_TILE) in STAIRS_TILES, 'NONE', 200)
    m = w.until(lambda w: w.mode != PLAY, 'LEFT', 200)
    k = w.until(lambda w: w.mode == PLAY and w.level == 8, 'NONE', 900)
    print(f'B, {n} frames to the stairs, {m} frames of Left from X $C0 to the mode change (Link at X '
          f'${w.x:02X}), {k} more until he plays in level 8: {w.where()}; 0x6d flag now ${w.r(OW_FLAGS + 0x6D):02X}')
    # two fires: the red candle again while the first stands
    w.game.restore(snap)
    w.game.apply_ram({'link_x': 0xC8, 'link_y': 0x5D})
    w.game.write(OBJ_DIR, 2)
    w.press('NONE', 1)
    w.press('B', 1)
    w.press('NONE', 20)
    w.press('B', 1)
    w.press('NONE', 5)
    two = fires(w)
    w.press('B', 1)
    w.press('NONE', 5)
    print(f'the red candle twice 20 frames apart: fires (slot, x, y, state) {two}; after a third B: {fires(w)}; '
          f'HasUsedCandle ${w.r(0x513):02X}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['whirlwind', 'rooms', 'tree'])
    parser.add_argument('--seeds', type=int, default=30)
    args = parser.parse_args()
    require_job()
    globals()[args.what](args)


if __name__ == '__main__':
    main()
