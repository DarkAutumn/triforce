"""Round 32's walk of `d3first`'s leg from the recorder's landing at Level 3 to Level 2's entrance
(docs/routes/d3first-walks-b.md §3): 0x74 W, 0x73 N, 0x63 W, 0x62's tree burned by the blue candle and its cave's 100
rupees, 0x62 N, 0x52 W, 0x51's wrap exit into 0x50, S into 0x60, its wrap exit round the world onto 0x5f's water, the
ladder clip onto the heart container, N 0x4f, 0x3f's raft to 0x2f's take-any heart container, back, W 0x3e and its
wrap exit into 0x3d, the Armos whose touch opens the 30-rupee cave, S 0x4d, W 0x4c, N 0x3c into Level 2.  The
alternative through the Lost Woods (0x62 W into 0x61, its sequence N, W, S, W into 0x60) is walked beside it.

    python scripts/experiments/r32b_leg.py stage            # the leg once, monsters emptied; every room's state
    python scripts/experiments/r32b_leg.py cross [--seeds N] [--rooms 0_62e,...]   # every room with monsters live
    python scripts/experiments/r32b_leg.py woods [--seeds N]   # the Lost Woods way from 0x62 to 0x60

`stage` starts from Level 5's exit (`r32b_walks.out_of(5)`: 8 containers, the wood sword, the blue candle, the
recorder, pieces 1, 3, 4, 5, the index 0), plays the recorder facing right (§1) and walks the leg with the monster
slots emptied every frame (the woken Armos is kept: its fade-in makes the stairs).  It saves every room's pre-entry
state (the scroll into it; for a room entered by a wrap exit, the frame the macro hands back in it) into a temporary
directory.  `cross` loads each with the game's own randomness for its seed and plays the room by a scripted rule
(walks, the candle, the clips by the engine's own macros `Game.act('CLIP E'|'CLIP W')`, the raft, the caves), the
monsters live and never fought: hearts lost, frames, what hit."""

import argparse
import collections
import statistics
import tempfile
from pathlib import Path

from r26_walks import OBJ_DIR, OBJ_STATE, OBJ_TYPE, PLAY, RUPEES, Walk
from r27_walks import ARMOS, StagingWalk, Tally, is_pre_entry, load, name, summary
from r32b_recorder import ride
from r32b_walks import ledger_row, out_of

CAVE = 0x0B
CANDLE_SLOT = 4
STAIRS_TILES = (0x70, 0x71, 0x72, 0x73)
TREE_62 = (16, 4)                     # 0x62's burnable tree (game.yaml entrance tile): X $80, Y $60
ARMOS_3D = (18, 8)                    # 0x3d's secret Armos (game.yaml entrance tile): X $90, Y $80
FACE = {'RIGHT': 1, 'LEFT': 2, 'DOWN': 4, 'UP': 8}
TMP = Path(tempfile.mkdtemp(prefix='r32b-leg-'))


class Stager(StagingWalk):
    """Staging: the monster and shot slots emptied every frame, but the Armos (its fade-in makes 0x3d's stairs)."""

    count = 0                         # frames played (presses and the clip macros)

    def clear_monsters(self):
        for s in range(1, 0x0C):
            if self.r(OBJ_TYPE + s) not in (0, ARMOS) and self.r(OBJ_TYPE + s) < 0x60:
                self.game.write(OBJ_TYPE + s, 0)

    def press(self, buttons, frames: int = 1):
        self.count += frames
        return super().press(buttons, frames)


# ---- the rooms' scripted rules (each from the room's first frame; True when it left the room as the route does) ----

def leave(t: Tally, button, line, limit=600) -> bool:
    return t.leave(button, limit, line=line)


def wrap(t: Tally, home, limit=12) -> bool:
    """The engine's wrap exit (`CLIP E`) from `home` (the zone spot walked back to after a knockback)."""
    w, room = t.w, t.w.room
    for _ in range(limit):
        if w.room != room or w.hearts() <= 0:
            break
        if w.game.is_allowed('CLIP E'):
            change = w.game.act('CLIP E')
            t.frames += change.frames
            w.count = getattr(w, 'count', 0) + change.frames
        elif not t.walk(home, limit=200):
            t.press('RIGHT', 4)
    return w.room != room and w.mode == PLAY


def wait_still(t: Tally, limit=400):
    """Until Link plays and stands (a raft ride or a cave's walk-out over)."""
    w = t.w
    for _ in range(limit):
        if w.mode == PLAY and w.r(OBJ_STATE) == 0:
            return True
        t.press('NONE')
    return False


def r74(t):                         # the landing (0x80, 0x8D): W along its row
    return leave(t, 'LEFT', ('y', 0x8D))


def r73(t):                         # E -> N between the trees at X $B0
    return t.walk((0xB0, 0x8D)) and leave(t, 'UP', ('x', 0xB0))


def r63(t):                         # S -> W: up to row 16, W along it to X $50, up its column, out W on row 8
    return t.walk(('y', 0xB0, 0xBD), (0x50, 0xBD), ('y', 0x50, 0x7D)) and leave(t, 'LEFT', ('y', 0x7D))


def r63_top(t):                     # the first rule tried: up the column to row 4, W along it, down to row 8
    return t.walk(('y', 0xB0, 0x5D), (0x20, 0x5D), ('y', 0x20, 0x7D)) and leave(t, 'LEFT', ('y', 0x7D))


def burn(t: Tally, spot, face, tree, limit=900) -> bool:
    """To `spot`, facing `face`, the candle until the tree's square shows stairs, then onto them (along `face`)."""
    w = t.w
    w.select_b(CANDLE_SLOT)
    if not t.walk(spot):
        return False
    for _ in range(limit):
        if w.mode != PLAY or w.hearts() <= 0:
            break
        burned = w.tile(*tree) in STAIRS_TILES
        fire = any(w.r(OBJ_STATE + s) in (0x21, 0x22) for s in (0x10, 0x11))
        at = w.y == spot[1] and abs(w.x - spot[0]) <= 6 if face in ('LEFT', 'RIGHT') else \
            w.x == spot[0] and abs(w.y - spot[1]) <= 6
        if burned:
            if w.y != spot[1]:
                t.press('DOWN' if w.y < spot[1] else 'UP')
            else:
                t.press(face)
        elif not at:
            t.walk(spot, limit=200)
        elif fire:
            t.press('NONE')
        elif w.r(OBJ_DIR) != FACE[face]:
            t.press(face)
        else:
            t.press('B')
            t.press('NONE', 2)
    return w.mode != PLAY


def r62(t):                         # E -> the tree from E on its row (X $A0, Y $5D), facing W
    return burn(t, (0xA0, 0x5D), 'LEFT', TREE_62)


def secret_cave(t: Tally) -> bool:
    """A secret-rupees cave: in, up to the rupees at X $78, out."""
    w = t.w
    if w.until(lambda w: w.mode == CAVE and w.y == 0xD5, 'NONE', 600) < 0:
        return False
    r0 = w.r(RUPEES)
    t.walk(('y', 0x70, 0xAD), (0x78, 0xAD), (0x78, 0x9D), limit=300)
    for _ in range(300):
        if w.r(RUPEES) > r0 and w.r(0x67D) == 0:
            break
        t.press('NONE')
    t.walk((0x78, 0xBD), (0x70, 0xBD), limit=300)
    for _ in range(600):
        if w.mode == PLAY:
            return True
        t.press('DOWN')
    return False


def r62c(t):                        # out of the cave (0x60, 0x7D): N at X $60
    return leave(t, 'UP', ('x', 0x60))


def r52(t):                         # S -> W
    return t.walk(('y', 0x60, 0xBD), (0x20, 0xBD), ('y', 0x20, 0x8D)) and leave(t, 'LEFT', ('y', 0x8D))


def r51(t):                         # E -> W by the wrap exit
    return wrap(t, (0xF0, 0x8D))


def r50(t):                         # E (the wrap's arrival, X $F0) -> S at X $C0
    return t.walk((0xC0, 0xBD)) and leave(t, 'DOWN', ('x', 0xC0))


def r60(t):                         # N -> W: down to row $8D, the wrap exit round the world into 0x5f
    return t.walk(('y', 0xC0, 0x8D), (0xE0, 0x8D)) and wrap(t, (0xE0, 0x8D))


def r5f(t):                         # E (on the water, X $F0) -> the ladder clip W onto the heart, W to land, N
    w = t.w
    m0 = w.max_hearts()
    for _ in range(6):
        if w.max_hearts() > m0 or w.hearts() <= 0:
            break
        if w.game.is_allowed('CLIP W'):
            n = w.game.act('CLIP W').frames
            t.frames += n
            w.count = getattr(w, 'count', 0) + n
        else:
            t.walk((0xF0, 0x8D), limit=100)
    if w.max_hearts() == m0:
        return False
    return t.walk((0x80, 0x8D), (0x70, 0x8D), ('y', 0x70, 0x5D)) and leave(t, 'UP', ('x', 0x70))


def r4f(t):
    return leave(t, 'UP', ('x', 0x70))


def r3f_raft(t):                    # S -> onto the dock (X $60) -> the raft N into 0x2f
    return t.walk(('y', 0x70, 0x8D), (0x60, 0x8D)) and leave(t, 'UP', ('x', 0x60), limit=900)


def r2f(t):                         # off the raft, up the pier into the take-any cave
    return wait_still(t) and leave_mode(t, 'UP', 400)


def leave_mode(t, button, limit=300):
    w = t.w
    for _ in range(limit):
        if w.mode != PLAY:
            return True
        t.press(button)
    return False


def take_any(t: Tally) -> bool:
    """The take-any cave: the right ware (the heart container), out."""
    w = t.w
    if w.until(lambda w: w.mode == CAVE and w.y == 0xD5, 'NONE', 600) < 0:
        return False
    m0 = w.max_hearts()
    t.walk(('y', 0x70, 0xA5), (0x98, 0xA5), (0x98, 0x9D), limit=300)
    for _ in range(300):
        if w.max_hearts() > m0:
            break
        t.press('NONE')
    t.press('NONE', 60)
    t.walk((0x98, 0xBD), (0x70, 0xBD), limit=300)
    for _ in range(600):
        if w.mode == PLAY:
            return w.max_hearts() > m0
        t.press('DOWN')
    return False


def r2fc(t):                        # out of the cave on the pier: the raft S into 0x3f
    return leave(t, 'DOWN', ('x', 0x60), limit=900)


def r3f_back(t):                    # off the raft on 0x3f's dock -> W
    return wait_still(t) and t.walk(('y', 0x60, 0x8D), (0x10, 0x8D)) and leave(t, 'LEFT', ('y', 0x8D))


def r3e(t):                         # E -> W by the wrap exit
    return wrap(t, (0xF0, 0x8D))


def armos(t: Tally, spot, face, square, back, stab=False, limit=900) -> bool:
    """To `spot` beside the secret Armos, one touch along `face`, `back` (a button and frames: off its way while it
    fades in) or none, then onto the stairs and in.  `stab`: once it has faded in, the sword (or its beam, at full
    hearts) whenever the Armos stands ahead on Link's row, until it is dead; then the stairs."""
    w = t.w
    if not t.walk(*spot):
        return False
    touched = False
    stand = spot[-1][1:] if spot[-1][0] == 'y' else spot[-1]
    vx = {'LEFT': -1, 'RIGHT': 1}.get(face, 0)
    for _ in range(limit):
        if w.mode != PLAY or w.hearts() <= 0:
            break
        armos_up = [o for o in w.objects() if o[1] == ARMOS]
        if stab and touched and armos_up and w.tile(*square) in STAIRS_TILES:
            o = armos_up[0]
            ahead = (o[2] - w.x) * vx
            if abs(o[3] - w.y) <= 8 and (0 < ahead <= 0x1C or (w.hearts() == w.max_hearts() and ahead > 0)):
                if w.r(OBJ_DIR) != FACE[face]:
                    t.press(face)
                else:
                    t.press('A')
                    t.press('NONE', 1)
            elif w.y != stand[1]:
                t.press('DOWN' if w.y < stand[1] else 'UP')
            elif abs(o[2] - square[0] * 8) >= 0x14 or abs(o[3] - (square[1] * 8 + 0x40)) >= 0x14:
                t.press(face)                                    # off the stairs: go for them
            else:
                t.press('NONE')
            continue
        if w.tile(*square) in STAIRS_TILES:
            if w.y != stand[1]:
                t.walk((w.x, stand[1]), limit=100)
            t.press(face)
            continue
        if not touched:
            t.press(face, 2)
            touched = any(o[1] == ARMOS for o in w.objects())
            if touched and back:
                t.press(back[0], back[1])
            continue
        t.press('NONE')
    return w.mode != PLAY


def r3d(t, back=('RIGHT', 12), stab=False):   # E (the wrap's arrival, X $F0) -> the Armos from E on its row -> cave
    return armos(t, [(0xE0, 0x8D), ('y', 0xE0, 0x5D), (0xA0, 0x5D), ('y', 0xA0, 0x7D)], 'LEFT', ARMOS_3D, back, stab)


def r3dc(t):                        # out of the cave above the stairs (X $90, Y $6D) -> E a square, S at X $A0
    return t.walk((0xA0, 0x6D)) and leave(t, 'DOWN', ('x', 0xA0))

def r4d(t):                         # N (X $A0) -> down to row 4, W along it, down to the W exit's row
    return t.walk(('y', 0xA0, 0x5D), (0x10, 0x5D), ('y', 0x10, 0x8D)) and leave(t, 'LEFT', ('y', 0x8D))


def r4c(t):                         # E -> W along row $8D to X $70, N up the gap (d2first's step)
    return t.walk((0x70, 0x8D)) and leave(t, 'UP', ('x', 0x70))


def r3c(t):                         # S -> up the column into Level 2's mouth (d2first's step)
    return t.walk((0x70, 0xCD)) and leave_mode(t, 'UP', 400)


# (state name, the room's rule, how the next room is entered: 'scroll' (the rule leaves by an edge or a mouth;
# the scroll's pre-entry frame is saved) or 'wrap' (the rule's macro hands back the new room's first frame), and the
# room / mode the rule ends in)
LEG = [
    ('0_74r', r74, 'scroll'), ('0_73e', r73, 'scroll'), ('0_63s', r63, 'scroll'), ('0_62e', r62, 'cave'),
    ('0_62cave', secret_cave, 'scroll'), ('0_62c', r62c, 'scroll'), ('0_52s', r52, 'scroll'),
    ('0_51e', r51, 'wrap'), ('0_50e', r50, 'scroll'), ('0_60n', r60, 'wrap'), ('0_5fe', r5f, 'scroll'),
    ('0_4fs', r4f, 'scroll'), ('0_3fs', r3f_raft, 'scroll'), ('0_2fs', r2f, 'cave'), ('0_2fcave', take_any, 'scroll'),
    ('0_2fc', r2fc, 'scroll'), ('0_3fn', r3f_back, 'scroll'), ('0_3ee', r3e, 'wrap'), ('0_3de', r3d, 'cave'),
    ('0_3dcave', secret_cave, 'scroll'), ('0_3dc', r3dc, 'scroll'), ('0_4dn', r4d, 'scroll'),
    ('0_4ce', r4c, 'scroll'), ('0_3cs', r3c, 'scroll'),
]


def walk_states(w, legs, out: Path, last: str) -> dict:
    """Plays `legs` [(state, rule, how)] from where `w` stands (its room is the first state's), monsters emptied;
    saves each next room's pre-entry state (the scroll's or the cave's; after a wrap the macro's hand-back frame),
    the one after the last rule as `last`.  {state: path} of the rooms entered."""
    paths = {}
    w.staging = True
    start = w.count
    for i, (state, rule, how) in enumerate(legs):
        t = Tally(w)
        snap_pre = None
        nxt = legs[i + 1][0] if i + 1 < len(legs) else last
        orig_press = w.press

        def press(buttons, frames=1, _orig=orig_press):
            nonlocal snap_pre
            for _ in range(frames):
                if w.mode != PLAY and is_pre_entry(w):
                    snap_pre = w.game.snapshot()
                _orig(buttons, 1)
        w.press = press
        ok = rule(t)
        w.press = orig_press
        w.until(lambda w: w.mode in (PLAY, CAVE) and w.r(OBJ_STATE) in (0, 0x40) and not is_pre_entry(w), 'NONE', 900)
        print(f'{state}: {"ok" if ok else "FAILED"} in {t.frames} frames -> {w.where()} {ledger_row(w)}')
        if not ok:
            raise RuntimeError(f'staging failed in {state}')
        path = out / f'{nxt}.state'
        if how == 'wrap' or snap_pre is None:
            w.game.save_state_file(str(path))
        else:
            now = w.game.snapshot()
            w.game.restore(snap_pre)
            w.game.save_state_file(str(path))
            w.game.restore(now)
        paths[nxt] = path
    print(f'{legs[0][0]} .. {legs[-1][0]}: {w.count - start} frames in all (scrolls included) to '
          f'{w.where()}')
    return paths


def stage(out: Path = TMP) -> dict:
    """The leg once from Level 5's exit, monsters emptied; {state: path} ('2_7ds': Level 2's first room)."""
    out.mkdir(parents=True, exist_ok=True)
    w0 = out_of(5)
    r = ride(w0, 'RIGHT', 1)
    w = Stager(game=w0.game)
    w.game.set_clips(True)
    print(f'landed {w.where()} after {r["frames"]} frames; {ledger_row(w)}')
    path = out / '0_74r.state'
    w.game.save_state_file(str(path))
    paths = {'0_74r': path}
    paths.update(walk_states(w, LEG, out, '2_7ds'))
    return paths


def measure(paths, variants, seeds) -> tuple:
    """Each (state, rule, label) played on `seeds` trials, monsters live; prints one line each; returns the sums of
    the median frames and mean hearts of the unlabelled ones."""
    frames = hearts = 0
    for state, rule, label in variants:
        rows, ends = [], collections.Counter()
        for seed in range(seeds):
            w = load(paths[state], seed)
            w.game.set_clips(True)
            t = Tally(w)
            h0, m0 = w.hearts(), w.max_hearts()
            ok = rule(t)
            lost = h0 - w.hearts() + (w.max_hearts() - m0)
            rows.append((ok, lost, t.frames, t.by))
            ends[f'{w.level}:0x{w.room:02x}'] += 1
        print(f'{state}{label}: {summary(rows)}; ends {dict(ends)}')
        if not label:
            frames += statistics.median(r[2] for r in rows)
            hearts += statistics.mean(r[1] for r in rows)
    return frames, hearts


def cross(args):
    paths = stage()
    rules = {state: rule for state, rule, _ in LEG}
    wanted = args.rooms.split(',') if args.rooms else [s for s, _, _ in LEG]
    variants = [(s, rules[s], '') for s in wanted]
    if '0_3de' in wanted:
        variants.append(('0_3de', lambda t: r3d(t, back=('RIGHT', 16), stab=True), ' (back off 16, the sword or beam)'))
        variants.append(('0_3de', lambda t: r3d(t, back=None, stab=False), ' (holds on: no back-off, no fight)'))
    if '0_63s' in wanted:
        variants.append(('0_63s', r63_top, ' (up to row 4 first)'))
    frames, hearts = measure(paths, variants, args.seeds)
    print(f'the leg by these rules: {frames:.0f} frames (sum of medians), {hearts:.2f} hearts (sum of means)')


def maze_step(t: Tally, points, button, line, limit=600) -> bool:
    """A Lost Woods scroll: the room id stays 0x61 on a step of the sequence; the scroll is the room change."""
    w = t.w
    if points and not t.walk(*points):
        return False
    for _ in range(limit):
        if w.mode != PLAY:
            return True
        if w.hearts() <= 0:
            return False
        cur = w.x if line[0] == 'x' else w.y
        if cur != line[1]:
            t.press({'x': ('RIGHT', 'LEFT'), 'y': ('DOWN', 'UP')}[line[0]][0 if line[1] > cur else 1])
        else:
            t.press(button)
    return False


# The Lost Woods way from 0x62's cave: 0x62 W, 0x61 from E (N), from S (W), from E (S), from N (W) into 0x60, whose
# wrap exit from E lands on 0x5f's water as the 0x50 way's does.
WOODS = [
    ('0_62c', lambda t: t.walk(('y', 0x60, 0x8D)) and leave(t, 'LEFT', ('y', 0x8D)), 'scroll'),
    ('0_61e', lambda t: maze_step(t, [(0xA0, 0x8D)], 'UP', ('x', 0xA0)), 'scroll'),
    ('0_61s', lambda t: maze_step(t, [('y', 0xA0, 0x8D)], 'LEFT', ('y', 0x8D)), 'scroll'),
    ('0_61e_v2', lambda t: maze_step(t, [(0xA0, 0x8D)], 'DOWN', ('x', 0xA0)), 'scroll'),
    ('0_61n', lambda t: maze_step(t, [('y', 0xA0, 0x8D)], 'LEFT', ('y', 0x8D)), 'scroll'),
    ('0_60e', lambda t: t.walk((0xE0, 0x8D)) and wrap(t, (0xE0, 0x8D)), 'wrap'),
]


def woods(args):
    """The Lost Woods way: out of 0x62's cave W into 0x61, its sequence N, W, S, W into 0x60, its wrap exit."""
    paths = stage()
    w = Stager()
    w.game.load(str(paths['0_62c']), randomize_rng=False)
    w.game.set_clips(True)
    wpaths = walk_states(w, WOODS, TMP / 'woods', '0_5fe')
    wpaths['0_62c'] = paths['0_62c']
    frames, hearts = measure(wpaths, [(s, rule, '') for s, rule, _ in WOODS], args.seeds)
    print(f'the woods way 0x62 cave -> 0x5f: {frames:.0f} frames (sum of medians), {hearts:.2f} hearts')
    rules = {state: rule for state, rule, _ in LEG}
    frames, hearts = measure(paths, [(s, rules[s], '') for s in ('0_62c', '0_52s', '0_51e', '0_50e', '0_60n')],
                             args.seeds)
    print(f'the 0x52 way 0x62 cave -> 0x5f: {frames:.0f} frames (sum of medians), {hearts:.2f} hearts')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('stage')
    for cmd in ('cross', 'woods'):
        p = sub.add_parser(cmd)
        p.add_argument('--seeds', type=int, default=20)
        p.add_argument('--rooms')
    args = parser.parse_args()
    {'stage': lambda a: stage(), 'cross': cross, 'woods': woods}[args.cmd](args)


if __name__ == '__main__':
    main()
