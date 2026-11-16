"""Round 28's walks of Dungeon 8 on the draft's way (docs/routes/d8-walks.md item 2): every room's first frame and
monsters, the crossings by scripted straight walks with the monsters live, the doors, keys, shutters and bomb
walls, the tunnel's ladders, what a room remembers of its kills, the Bubbles.  Staging: `r28_walks.stage_d8`
(`triforce.probe` with the route's inventory, 13 hearts, the magic sword, the doors opened before).

    python scripts/experiments/r28_d8_walks.py rooms [--seeds N]       # first spot, monsters, statue fire per room
    python scripts/experiments/r28_d8_walks.py crossings [--seeds N]   # 0x7e, 0x4e, 0x3f, the tunnel, 0x4c, 0x2c
    python scripts/experiments/r28_d8_walks.py doors                   # keys, key doors, shutters, the keyless pass
    python scripts/experiments/r28_d8_walks.py history                 # kills banked: in parts, past the history
    python scripts/experiments/r28_d8_walks.py tunnel [--seeds N]      # where each ladder of 0x2f leads
    python scripts/experiments/r28_d8_walks.py bubbles [--seeds N]     # which Bubbles, what a touch does

The Manhandla rooms are `r28_manhandla.py`, the Darknut fights `r28_darknut.py`, 0x4c's Pols Voices
`r28_pols.py`, Gleeok `r28_gleeok.py`."""

import argparse
import collections
import statistics

from r28_walks import (ALL_DEAD, BOMBS, CELLAR, CUR_OPENED_DOORS, D8_ROUTE, DARKNUT_BLUE, DARKNUT_RED,
                       FIREBALL_STATUE, GIBDO, HEARTS, KEYS, LEVEL_KILL_COUNTS, OBJ_DIR, OBJ_META, OBJ_TYPE, PLAY,
                       ROOM_HISTORY, ROOM_KILL_COUNT, ROOM_OBJ_COUNT, SWORD_BLOCKED, TRIFORCE, UW_FLAGS_7TO9,
                       Dodger, StagingWalk, Tally28, alive, d8_cellar_states, d8_state, hop, kill_all, load, name,
                       stage_d8, summary)

DOOR_BITS = {'N': 8, 'S': 4, 'W': 2, 'E': 1}


def state_of(room, entry):
    if entry == 'cave':
        return d8_cellar_states()['8_2fc' if room == 0x2F else '8_4cc']
    return d8_state(room, entry)


def enter(room, entry, seed):
    w = load(state_of(room, entry), seed)
    w.until(lambda w: w.mode in (PLAY, CELLAR), 'NONE', 600)
    return w


def kinds(w):
    return dict(sorted(collections.Counter(name(o[1]) for o in w.objects() if o[1] < 0x50).items()))


def immortal(w, buttons='NONE', frames=1):
    """Measurement only: frames with Link's hearts written back each frame."""
    hearts = w.r(HEARTS)
    for _ in range(frames):
        w.press(buttons, 1)
        w.game.write(HEARTS, hearts)
        w.game.write(HEARTS + 1, 0xFF)


def history(w):
    return [f'{w.r(ROOM_HISTORY + i):02x}' for i in range(6)]


# ---- every room's first frame ---------------------------------------------------------------------------------------

def rooms(args):
    for room, entry in D8_ROUTE:
        firsts, made, first_ball, balls, doors, lost = (collections.Counter(), collections.Counter(), [], [],
                                                        collections.Counter(), [])
        for seed in range(args.seeds):
            w = enter(room, entry, seed)
            firsts[f'({w.x:#04x},{w.y:#04x}) facing {w.r(OBJ_DIR)}'] += 1
            d0 = w.r(CUR_OPENED_DOORS)
            h0, seen, ball_at, n_balls, prev = w.hearts(), collections.Counter(), None, 0, set()
            for f in range(600):
                w.press('NONE', 1)
                if f == 60:
                    doors[f'${d0:02X} -> ${w.r(CUR_OPENED_DOORS):02X}'] += 1
                if f == 150:
                    seen = kinds(w)
                    lost.append(h0 - w.hearts())
                now = {o[0] for o in w.objects() if o[1] == FIREBALL_STATUE}
                if now - prev:
                    n_balls += len(now - prev)
                    if ball_at is None:
                        ball_at = f
                prev = now
                if f > 150:
                    w.game.write(HEARTS, w.r(HEARTS) | 0x0C)     # kept alive for the fireball count
            made[tuple(seen.items())] += 1
            if ball_at is not None:
                first_ball.append(ball_at)
            balls.append(n_balls)
        print(f'8:0x{room:02x} from {entry}: Link first at {dict(firsts)}; opened doors {dict(doors)}; monsters at '
              f'frame 150 {dict(made)}; RoomObjCount {w.r(ROOM_OBJ_COUNT)}; hearts lost standing 150 frames: mean '
              f'{statistics.mean(lost):.2f} max {max(lost)}')
        if any(balls):
            print(f'   statue fireballs: the first on frame {min(first_ball)}-{max(first_ball)} (median '
                  f'{int(statistics.median(first_ball))}); {statistics.mean(balls):.1f} made in 600 frames '
                  f'(one every {600 / max(0.01, statistics.mean(balls)):.0f} frames)')


# ---- crossings ------------------------------------------------------------------------------------------------------

def cross(room, entry, seed, points, button, line, limit=900, hearts=None):
    w = enter(room, entry, seed)
    if hearts is not None:
        from r28_walks import hearts_bytes                       # pylint: disable=import-outside-toplevel
        hb = hearts_bytes(hearts, 13)
        w.game.set_value('hearts_and_containers', hb['hearts_and_containers'])
        w.game.set_value('partial_hearts', hb['partial_hearts'])
    t = Tally28(w)
    start = (w.x, w.y)
    keys, blocked = w.r(KEYS), 0
    ok = t.walk(*points, limit=limit) if points else True
    if ok:
        room0 = (w.level, w.room, w.mode in (PLAY, CELLAR))
        ok = False
        for _ in range(limit):
            if (w.level, w.room) != room0[:2] and w.mode in (PLAY, CELLAR):
                ok = True
                break
            if w.mode in (PLAY, CELLAR) and w.hearts() <= 0:
                break
            blocked = max(blocked, w.r(SWORD_BLOCKED))
            cur = None if w.mode not in (PLAY, CELLAR) else (w.x if line[0] == 'x' else w.y)
            if cur is not None and cur != line[1] and (w.level, w.room) == room0[:2]:
                t.press({'x': ('RIGHT', 'LEFT'), 'y': ('DOWN', 'UP')}[line[0]][0 if line[1] > cur else 1])
            else:
                t.press(button)
    return {'ok': ok, 'lost': t.lost(), 'frames': t.frames, 'by': t.by, 'start': start,
            'end': (w.level, w.room, w.x, w.y, w.mode), 'keys': (keys, w.r(KEYS)), 'sword_blocked': w.r(SWORD_BLOCKED),
            'hearts': w.hearts(), 'w': w}


# (label, room, entry, points (x first; ('y', x, y): y first), exit button, line kept)
CROSSINGS = [
    ('0x7e S->N', 0x7E, 'S', [], 'UP', ('x', 0x78)),
    ('0x4e S->N (key door), straight up', 0x4E, 'S', [(0x78, 0x5D)], 'UP', ('x', 0x78)),
    ('0x3f W->stairs, straight E on the door row', 0x3F, 'W', [(0xC0, 0x8D)], 'RIGHT', ('y', 0x8D)),
    ('0x3f W->stairs, by row $7D to X $A0', 0x3F, 'W', [('y', 0x20, 0x7D), (0xA0, 0x7D), (0xA0, 0x8D), (0xC0, 0x8D)],
     'RIGHT', ('y', 0x8D)),
    ('0x2f tunnel: down, W, up the left ladder', 0x2F, 'cave', [('y', 0xC0, 0xBD), (0x30, 0xBD)], 'UP', ('x', 0x30)),
    ('0x2c S->piece', 0x2C, 'S', [], 'UP', ('x', 0x78)),
]


def crossings(args):
    for label, room, entry, points, button, line in CROSSINGS:
        rows, ends, extra = [], collections.Counter(), collections.Counter()
        for seed in range(args.seeds):
            if room == 0x2C:
                w = enter(room, entry, seed)
                t = Tally28(w)
                n = w.until(lambda w: w.r(TRIFORCE) == 0xFF, 'UP', 300)
                m = w.until(lambda w: w.mode == PLAY and w.level == 0, 'NONE', 3000)
                rows.append((m >= 0, t.lost(), n + m, t.by))
                ends[f'{w.level}:0x{w.room:02x} ({w.x:#04x},{w.y:#04x}) hearts {w.hearts()}/{w.max_hearts()} '
                     f'piece after {n} frames of Up'] += 1
                continue
            r = cross(room, entry, seed, points, button, line)
            rows.append((r['ok'], r['lost'], r['frames'], r['by']))
            e = r['end']
            ends[f'{e[0]}:0x{e[1]:02x} ({e[2]:#04x},{e[3]:#04x})' + ('' if r['ok'] else ' dead' if r['hearts'] <= 0
                                                                        else ' stuck')] += 1
            if r['keys'][0] != r['keys'][1]:
                extra[f'keys {r["keys"][0]} -> {r["keys"][1]}'] += 1
            if r['sword_blocked']:
                extra['sword blocked on leaving'] += 1
        print(f'{label}: in at ({r["start"][0]:#04x},{r["start"][1]:#04x}); {summary(rows)}; arrives {dict(ends)}; '
              f'{dict(extra)}' if room != 0x2C else f'{label}: {summary(rows)}; {dict(ends)}')
    # the same crossings stepping aside from what is predicted to touch Link (`r28_walks.Dodger`)
    for label, room, entry, points, button, line in (
            ('0x4e S->N (key door), stepping aside', 0x4E, 'S', [(0x78, 0x5D)], 'UP', ('x', 0x78)),
            ('0x3f W->stairs, stepping aside', 0x3F, 'W', [(0xB0, 0x8D)], 'RIGHT', ('y', 0x8D))):
        for hearts in (None, 6.0, 3.0):
            rows, blocked = [], 0
            for seed in range(args.seeds):
                w = enter(room, entry, seed)
                if hearts is not None:
                    from r28_walks import hearts_bytes           # pylint: disable=import-outside-toplevel
                    hb = hearts_bytes(hearts, 13)
                    w.game.set_value('hearts_and_containers', hb['hearts_and_containers'])
                    w.game.set_value('partial_hearts', hb['partial_hearts'])
                t = Tally28(w)
                d = Dodger(t)
                ok = d.walk(*points, limit=1500)
                if ok:
                    ok = False
                    for _ in range(600):
                        if w.mode not in (PLAY, CELLAR) or w.room != room:
                            ok = True
                            break
                        if w.hearts() <= 0:
                            break
                        cur = w.x if line[0] == 'x' else w.y
                        if cur != line[1]:
                            t.press(d.choose((line[1], w.y) if line[0] == 'x' else (w.x, line[1])))
                        else:
                            t.press(button)
                blocked += 1 if w.r(SWORD_BLOCKED) else 0
                rows.append((ok, t.lost(), t.frames, t.by))
            print(f'{label}, from {hearts or 13.0} hearts: {summary(rows)}; sword blocked on leaving in {blocked}')
    # 0x3f from fewer hearts: the one-way room is where a run is low
    for hearts in (6.0, 3.0):
        rows = []
        for seed in range(args.seeds):
            r = cross(0x3F, 'W', seed, [(0xC0, 0x8D)], 'RIGHT', ('y', 0x8D), hearts=hearts)
            rows.append((r['ok'], r['lost'], r['frames'], r['by']))
        print(f'0x3f straight E from {hearts} hearts: {summary(rows)}')


# ---- doors, keys, shutters ------------------------------------------------------------------------------------------

def watch_doors(w, frames=120):
    """Frames until each change of CurOpenedDoors, the all-dead flag and the keys, Link standing."""
    log, doors, dead = [], w.r(CUR_OPENED_DOORS), w.r(ALL_DEAD)
    for f in range(1, frames + 1):
        immortal(w)
        if w.r(CUR_OPENED_DOORS) != doors:
            doors = w.r(CUR_OPENED_DOORS)
            log.append((f, f'doors ${doors:02X}'))
        if w.r(ALL_DEAD) and not dead:
            dead = 1
            log.append((f, 'room flagged empty'))
    return log


def staged(room, entry, seed=0, **kwargs):
    w, _, _ = stage_d8(room, entry, **kwargs)
    return w


def doors(_args):
    # 0x5e: the key, the shutters
    w = enter(0x5E, 'S', 0)
    print(f'0x5e from S: {w.where()}, doors ${w.r(CUR_OPENED_DOORS):02X}, keys {w.r(KEYS)}, {kinds(w)}')
    immortal(w, 'NONE', 60)
    kill_all(w)
    log = watch_doors(w, 80)
    print(f'   all five put into the death state: {log} (frames after)')
    w.__class__ = StagingWalk
    w.staging = True
    k0 = w.r(KEYS)
    w.goto(y=0x8D)
    n = w.until(lambda w: w.r(KEYS) != k0, 'RIGHT' if w.x < 0x80 else 'LEFT', 100)
    print(f'   the key: walking along row $8D from X $78, taken at ({w.x:#04x},{w.y:#04x}) after {n} frames; keys '
          f'{k0} -> {w.r(KEYS)}; flag ${w.r(UW_FLAGS_7TO9 + 0x5E):02X}')
    snap = w.game.snapshot()
    w.goto(x=0x78)
    w.goto(y=0x5D)
    n = w.until(lambda w: w.room == 0x4E and w.mode == PLAY, 'UP', 400)
    print(f'   N from ($78,$5D): playing in 0x4e after {n} frames at ({w.x:#04x},{w.y:#04x}); doors '
          f'${w.r(CUR_OPENED_DOORS):02X}; history {history(w)}; 0x5e kill count {w.r(LEVEL_KILL_COUNTS + 0x5E)}, '
          f'flag ${w.r(UW_FLAGS_7TO9 + 0x5E):02X}')
    # the E key door of 0x5e (off the route), with the room cleared and the shutters done
    w.game.restore(snap)
    k, spent_at = w.r(KEYS), None
    for _ in range(200):
        if w.x == 0xD0:
            break
        w.press('RIGHT', 1)
        if w.r(KEYS) < k and spent_at is None:
            spent_at = w.x
    for f in range(1, 4):
        if spent_at is None:
            w.press('RIGHT', 1)
            if w.r(KEYS) < k:
                spent_at = f'frame {f} of Right at the door spot'
    n = w.until(lambda w: w.room == 0x5F and w.mode == PLAY, 'RIGHT', 400)
    w.staging = False
    w.press('NONE', 60)
    print(f'   0x5e E key door with {k} keys, walking Right along the door row: key spent at {spent_at}, in 0x5f '
          f'after {n} frames at ({w.x:#04x},{w.y:#04x}), keys {w.r(KEYS)}, {kinds(w)}; doors there '
          f'${w.r(CUR_OPENED_DOORS):02X}')
    # the keyless pass: Right into the E door while the shutters open
    for keys in (2, 0):
        res = collections.Counter()
        for delay in range(0, 20, 2):
            w = enter(0x5E, 'S', 0)
            immortal(w, 'NONE', 60)
            w.game.apply_ram({'link_x': 0xD0, 'link_y': 0x8D})   # written: Link on the E door spot
            w.game.write(OBJ_DIR, 1)
            w.game.write(KEYS, keys)
            kill_all(w)
            n, far, unpaid = None, 0xD0, 0xD0
            for f in range(1, 200):
                immortal(w, 'RIGHT' if n is not None and f - n >= delay else 'NONE')
                if n is None and w.r(ALL_DEAD):
                    n = f
                if w.mode != PLAY:
                    break
                far = max(far, w.x)
                if w.r(KEYS) == keys:
                    unpaid = max(unpaid, w.x)
            w.until(lambda w: w.mode == PLAY, 'RIGHT', 300)
            there, back = w.room, ''
            got, flags = w.r(KEYS), (w.r(UW_FLAGS_7TO9 + 0x5E) & 1, w.r(UW_FLAGS_7TO9 + 0x5F) & 2)
            if keys == 0 and there == 0x5F:
                immortal(w, 'NONE', 120)                         # the close command behind him is long over
                for _ in range(300):
                    if w.room == 0x5E and w.mode == PLAY:
                        break
                    immortal(w, 'LEFT')
                back = (f'; after 120 frames there, Left: {"back in 0x5e" if w.room == 0x5E else "shut in"}, keys '
                        f'{w.r(KEYS)}, flags then {w.r(UW_FLAGS_7TO9 + 0x5E) & 1}/{w.r(UW_FLAGS_7TO9 + 0x5F) & 2}')
            res[f'in 0x{there:02x}, keys {keys} -> {got}, furthest X ${far:02X} in 0x5e (${unpaid:02X} before '
                f'a key was taken), door flags 0x5e E/0x5f W {flags[0]}/{flags[1]}{back}'] += 1
        print(f'   keyless pass with {keys} keys: Link written onto the E door spot, the five put into the death '
              f'state, Right held from 0-18 frames after the room is flagged empty (10 starts): {dict(res)}')
    # 0x4e: the N key door, the W key door
    w = enter(0x4E, 'S', 0)
    w.__class__ = StagingWalk
    w.staging = True
    w.goto(y=0x5D)
    k = w.r(KEYS)
    snap = w.game.snapshot()
    log = []
    for f in range(1, 400):
        w.press('UP', 1)
        if w.r(KEYS) != k:
            log.append(f'key spent on frame {f}')
            k = w.r(KEYS)
        if w.r(CUR_OPENED_DOORS) & 8 and not any('open' in x for x in log):
            log.append(f'door open on frame {f}')
        if w.room == 0x3E and w.mode == PLAY:
            break
    print(f'0x4e N key door from ($78,$5D): {log}; playing in 0x3e after {f} frames at ({w.x:#04x},{w.y:#04x}); '
          f'keys {w.r(KEYS)}; flags 4e/3e ${w.r(UW_FLAGS_7TO9 + 0x4E):02X}/${w.r(UW_FLAGS_7TO9 + 0x3E):02X}; '
          f'doors in 0x3e ${w.r(CUR_OPENED_DOORS):02X}')
    w.game.restore(snap)
    w.goto(y=0x8D)
    w.goto(x=0x20)
    k = w.r(KEYS)
    w.press('LEFT', 1)
    f1 = w.r(KEYS)
    w.staging = False                                             # 0x4d's Gohma is the game's
    n = None
    for f in range(400):
        if w.room == 0x4D and w.mode == PLAY:
            n = f
            break
        immortal(w, 'LEFT')
    immortal(w, 'NONE', 60)
    w.staging = True
    print(f'0x4e W key door (off the route) from ($20,$8D): keys {k} -> {f1} on the first frame of Left; in 0x4d '
          f'after {n} frames, {kinds(w)}, doors ${w.r(CUR_OPENED_DOORS):02X}')
    w.game.restore(snap)
    w.game.write(KEYS, 0)
    n = w.until(lambda w: w.mode != PLAY, 'UP', 200)
    print(f'0x4e N with 0 keys: 200 frames of Up: {"through" if n >= 0 else "nothing"}')
    # 0x3e: the E shutter, the way round the statue
    w = enter(0x3E, 'S', 0)
    print(f'0x3e from S: {w.where()}, doors ${w.r(CUR_OPENED_DOORS):02X}, {kinds(w)}')
    immortal(w, 'NONE', 60)
    kill_all(w)
    print(f'   all six put into the death state: {watch_doors(w, 80)}')
    w.__class__ = StagingWalk
    w.staging = True
    a = w.path(('y', 0x8D), ('x', 0x80))
    b = w.goto(x=0x90, limit=60)
    print(f'   the door row: from ($78,$8D) Right to X $80 {a >= 0}; on to X $90: '
          f'{"walks" if b >= 0 else f"stopped at X ${w.x:02X} (the statue)"}')
    c = w.path(('x', 0x80), ('y', 0x9D), ('x', 0xA0), ('y', 0x8D), ('x', 0xD0))
    w.staging = False                                             # 0x3f's monsters are the game's
    n = w.until(lambda w: w.room == 0x3F and w.mode == PLAY, 'RIGHT', 400)
    d0 = w.r(CUR_OPENED_DOORS)
    first = (w.x, w.y)
    m = None
    for f in range(1, 101):
        immortal(w)
        if m is None and w.r(CUR_OPENED_DOORS) == 0:
            m = f
    out = False
    for f in range(200):
        immortal(w, 'DOWN' if w.y < 0x8D else 'UP' if w.y > 0x8D else 'LEFT')
        out = out or w.mode != PLAY
    print(f'   round the statue by row $9D: {c} frames to ($D0,$8D); E: playing in 0x3f after {n} frames at '
          f'({first[0]:#04x},{first[1]:#04x}), doors ${d0:02X}, shut {m} frames later; 200 frames of Left with its '
          f'monsters alive: {"out W" if out else "the shutter holds"}')
    # 0x3f: its six dead: the bombs, the W shutter
    w = enter(0x3F, 'W', 0)
    immortal(w, 'NONE', 60)
    kill_all(w)
    log = watch_doors(w, 80)
    w.__class__ = StagingWalk
    w.staging = True
    b0 = w.r(BOMBS)
    w.goto(y=0x8D)
    n = w.until(lambda w: w.r(BOMBS) != b0, 'RIGHT' if w.x < 0x80 else 'LEFT', 200)
    print(f'0x3f with its three Darknuts and three Gibdos put into the death state (two Bubbles alive): {log}; bombs '
          f'{b0} -> {w.r(BOMBS)} at ({w.x:#04x},{w.y:#04x})')
    # 0x7e S: out of the level and back
    w = enter(0x5E, 'S', 0)
    immortal(w, 'NONE', 60)
    kill_some(w, 2)
    out_and_back(w, 'DOWN', 'UP', 0x5E, 0x6E)
    immortal(w, 'NONE', 30)
    left = len(alive(w))
    for _ in range(600):
        if w.room == 0x6E and w.mode == PLAY:
            break
        immortal(w, 'DOWN')
    banked = w.r(LEVEL_KILL_COUNTS + 0x5E)
    w.__class__ = StagingWalk
    w.staging = True                                             # the Manhandla emptied for the walk down
    w.goto(x=0x78)
    w.until(lambda w: w.room == 0x7E and w.mode == PLAY, 'DOWN', 600)
    n = w.until(lambda w: w.mode != PLAY, 'DOWN', 300)
    m = w.until(lambda w: w.mode == PLAY and w.level == 0, 'NONE', 600)
    out = w.where()
    from r28_walks import burn_tree                               # pylint: disable=import-outside-toplevel
    burn_tree(w, x=0x80, face='RIGHT')                            # the probe's level entry never burned the tree
    w.until(lambda w: w.mode != PLAY, 'RIGHT', 300)
    w.until(lambda w: w.mode == PLAY and w.level == 8, 'NONE', 600)
    back = (w.where(), w.r(LEVEL_KILL_COUNTS + 0x5E), w.r(UW_FLAGS_7TO9 + 0x6E), w.r(UW_FLAGS_7TO9 + 0x5E),
            history(w))
    w.until(lambda w: w.room == 0x6E and w.mode == PLAY, 'UP', 600)
    w.goto(y=0x5D)
    w.staging = False
    for _ in range(600):
        if w.room == 0x5E and w.mode == PLAY:
            break
        immortal(w, 'UP')
    immortal(w, 'NONE', 60)
    print(f'0x7e S after 2 of 0x5e\'s five were killed (3 left on re-entry: {left}): {n} frames of Down from '
          f'0x7e\'s first spot, playing {m} frames later in {out}; the tree burned, back in by the stairs: {back[0]}; '
          f'0x5e kill count {banked} before, {back[1]} after; flags 6e/5e ${back[2]:02X}/${back[3]:02X} (the '
          f'bombed wall stays); history {back[4]}; 0x5e entered again: {len(alive(w))} Darknuts')


# ---- what a room remembers ------------------------------------------------------------------------------------------

def kill_some(w, n):
    for o in alive(w)[:n]:
        w.game.write(OBJ_META + o[0], 0x10)
    immortal(w, 'NONE', 60)


def out_and_back(w, out_button, in_button, room, other):
    """Leaves `room` by `out_button` into `other` and comes straight back; Link kept alive, monsters untouched."""
    for _ in range(600):
        if w.room == other and w.mode == PLAY:
            break
        immortal(w, out_button)
    there = (w.x, w.y)
    for _ in range(600):
        if w.room == room and w.mode == PLAY:
            break
        immortal(w, in_button)
    return there


def spawn_spots(w):
    return sorted((f'{o[2]:02X},{o[3]:02X}' for o in w.objects() if o[1] in (DARKNUT_BLUE, DARKNUT_RED, GIBDO)))


# Other rooms of the level to enter (the probe's hop); no hop's source room (the id written while Link leaves) is
# 0x5e or 0x3e, whose banked kills are what is read
OTHERS = ((0x7E, 'N'), (0x7F, 'W'), (0x7D, 'E'), (0x7C, 'E'), (0x5D, 'N'), (0x5C, 'E'), (0x4B, 'N'), (0x3B, 'S'))



def history_cmd(_args):
    for room, entry, n, back in ((0x5E, 'S', 5, 0x6E), (0x3E, 'S', 6, 0x4E)):
        w = enter(room, entry, 0)
        immortal(w, 'NONE', 60)
        print(f'8:0x{room:02x} from {entry}: {len(alive(w))} Darknuts at {spawn_spots(w)}, Link ({w.x:#04x},'
              f'{w.y:#04x}); history {history(w)}')
        left = n
        for kills in (2, 2):
            kill_some(w, kills)
            left -= kills
            count = w.r(ROOM_KILL_COUNT)
            out_and_back(w, 'DOWN', 'UP', room, back)
            first = (w.x, w.y, w.r(CUR_OPENED_DOORS))
            immortal(w, 'NONE', 60)
            balls = sum(1 for o in w.objects() if o[1] == FIREBALL_STATUE)
            print(f'   {kills} put into the death state (RoomKillCount {count}), out S into 0x{back:02x} and back: '
                  f'{len(alive(w))} Darknuts (expected {left}) at {spawn_spots(w)}; Link first at ({first[0]:#04x},'
                  f'{first[1]:#04x}), doors ${first[2]:02X}; banked {w.r(LEVEL_KILL_COUNTS + room)}, flag '
                  f'${w.r(UW_FLAGS_7TO9 + room):02X}; statue fireballs in flight 60 frames in: {balls}')
        # past the room history: six other rooms entered (the probe's hop: the room id written), then back in
        snap = w.game.snapshot()
        w.__class__ = StagingWalk
        w.staging = True
        for other, side in OTHERS:
            hop(w, other, side)
        hist = history(w)
        w.staging = False
        hop(w, room, entry)
        immortal(w, 'NONE', 60)
        print(f'   then eight other rooms entered (history {hist}: 0x{room:02x} gone) and back in from {entry}: '
              f'{len(alive(w))} Darknuts (expected {left}); banked {w.r(LEVEL_KILL_COUNTS + room)}, flag '
              f'${w.r(UW_FLAGS_7TO9 + room):02X}')
        # all dead: in the history it stays empty, past it the room is full again
        w.game.restore(snap)
        immortal(w, 'NONE', 30)
        kill_some(w, 9)
        log = watch_doors(w, 40)
        out_and_back(w, 'DOWN', 'UP', room, back)
        first = w.r(CUR_OPENED_DOORS)
        log2 = watch_doors(w, 60)
        print(f'   the rest put into the death state; out S and back: {len(alive(w))} Darknuts, doors ${first:02X} '
              f'then {log2}; banked {w.r(LEVEL_KILL_COUNTS + room)}, flag ${w.r(UW_FLAGS_7TO9 + room):02X}')
        w.__class__ = StagingWalk
        w.staging = True
        for other, side in OTHERS:
            hop(w, other, side)
            if f'{room:02x}' not in history(w):
                break
        hist = history(w)
        w.staging = False
        hop(w, room, entry)
        immortal(w, 'NONE', 60)
        print(f'   then other rooms entered until the history is {hist} and back in: {len(alive(w))} Darknuts; '
              f'banked {w.r(LEVEL_KILL_COUNTS + room)}, flag ${w.r(UW_FLAGS_7TO9 + room):02X}')


# ---- the tunnel -----------------------------------------------------------------------------------------------------

def tunnel(args):
    paths = d8_cellar_states()
    w = load(paths['8_2fc'], 0)
    w.until(lambda w: w.mode == CELLAR, 'NONE', 600)
    print(f'the tunnel from 0x3f\'s stairs: {w.where()}, {kinds(w)}')
    # the right ladder straight back up (Keese and 0x3f's monsters live, Link kept alive)
    n = None
    for f in range(600):
        if w.mode == PLAY:
            n = f
            break
        immortal(w, 'UP')
    first = (w.where(), w.r(OBJ_DIR), w.r(CUR_OPENED_DOORS))
    immortal(w, 'NONE', 60)
    print(f'   Up at once (the right ladder): playing after {n} frames in {first[0]} facing {first[1]}, doors '
          f'${first[2]:02X}; 0x3f then: {kinds(w)}; history {history(w)}')
    # from 0x4c's stairs: the left ladder
    w = load(paths['8_4cc'], 0)
    w.until(lambda w: w.mode == PLAY, 'NONE', 600)
    w.__class__ = StagingWalk
    w.staging = True
    n = w.until(lambda w: w.mode == CELLAR, 'RIGHT', 200)
    print(f'0x4c: Right from the arrival spot ($70,$7D): in the tunnel after {n} frames at {w.where()}')
    w.staging = False
    w.until(lambda w: w.x == 0x30, 'NONE', 200)
    start = w.where()
    m = None
    for f in range(600):
        if w.mode == PLAY:
            m = f
            break
        immortal(w, 'UP')
    first = w.where()
    immortal(w, 'NONE', 60)
    print(f'   there Link starts at {start}; Up at once (the left ladder): playing after {m} frames in {first}; '
          f'0x4c then: {kinds(w)}')
    rows = []
    for seed in range(args.seeds):
        w = load(paths['8_2fc'], seed)
        w.until(lambda w: w.mode == CELLAR, 'NONE', 600)
        t = Tally28(w)
        ok = False
        for _ in range(900):
            if w.mode != CELLAR:
                ok = True
                break
            if w.hearts() <= 0:
                break
            if w.x == 0x30 and w.y <= 0xBD:
                t.press('UP')
            elif w.y < 0xBD and w.x == 0xC0:
                t.press('DOWN')
            elif w.y == 0xBD:
                t.press('LEFT' if w.x > 0x30 else 'RIGHT')
            else:
                t.press('DOWN' if w.x >= 0x80 else 'UP')
        frames = t.frames
        if ok:
            w.until(lambda w: w.mode == PLAY, 'NONE', 300)
            ok = w.room == 0x4C
        rows.append((ok, t.lost(), frames, t.by))
    print(f'the tunnel, right ladder down, W along the bottom, left ladder up, Keese live: {summary(rows)}')


# ---- Bubbles --------------------------------------------------------------------------------------------------------

def bubbles(args):
    for room, entry in ((0x4E, 'S'), (0x3F, 'W')):
        types, timer, lost, back = collections.Counter(), collections.Counter(), collections.Counter(), []
        for seed in range(args.seeds):
            w = enter(room, entry, seed)
            immortal(w, 'NONE', 90)
            bs = [o for o in w.objects() if o[1] in (0x2B, 0x2C, 0x2D)]
            types[tuple(sorted(f'${o[1]:02X}' for o in bs))] += 1
            if not bs:
                continue
            for o in w.objects():                               # only the Bubble touches him
                if o[0] != bs[0][0]:
                    w.game.write(OBJ_TYPE + o[0], 0)
            h = w.hearts()
            w.game.apply_ram({'link_x': bs[0][2], 'link_y': bs[0][3]})
            w.press('NONE', 2)
            timer[f'${w.r(SWORD_BLOCKED):02X}'] += 1
            lost[h - w.hearts()] += 1
            for o in w.objects():
                w.game.write(OBJ_TYPE + o[0], 0)
            n = w.until(lambda w: w.r(SWORD_BLOCKED) == 0, 'NONE', 400)
            back.append(n)
        print(f'8:0x{room:02x}: Bubble types {dict(types)} ($2B flashing, $2C red, $2D blue); Link written onto '
              f'one: SwordBlockedLongTimer {dict(timer)}, hearts lost {dict(lost)}, the timer back at 0 after '
              f'{min(back)}-{max(back)} frames')
    w = enter(0x4E, 'S', 0)
    immortal(w, 'NONE', 90)
    b = next(o for o in w.objects() if o[1] == 0x2B)
    w.game.apply_ram({'link_x': b[2], 'link_y': b[3]})
    w.press('NONE', 30)
    mask = [a for a in ('SWORD N', 'BEAMS N', 'BOMB N', 'BOOMERANG N', 'ARROW N', 'MOVE N') if w.game.is_allowed(a)]
    print(f'   30 frames after the touch the mask offers {mask} (SwordBlockedLongTimer ${w.r(SWORD_BLOCKED):02X}); '
          f'A pressed: sword state ${[w.press("A", 1), w.press("NONE", 2), w.r(0xB9)][2]:02X} (0 = no sword out)')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['rooms', 'crossings', 'doors', 'history', 'tunnel', 'bubbles'])
    parser.add_argument('--seeds', type=int, default=30)
    args = parser.parse_args()
    {'history': history_cmd}.get(args.what, globals().get(args.what))(args)


if __name__ == '__main__':
    main()
