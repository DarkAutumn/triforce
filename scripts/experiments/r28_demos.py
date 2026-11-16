"""Round 28: the walks' winning scripts as rules over the environment's own actions, and their demonstration rows
(docs/routes/d8-walks.md; `train --demos FILE --demo-coeff C`, `triforce.demos`, as round 25's
`r25_d5_37_tracks.py --demos` and round 26's `r26_d7_28_food_demos.py`).

    python scripts/experiments/r28_demos.py scaffold OUT [--hearts H] [--boss-hearts H]
    cd OUT && TRIFORCE_STATES_DIR=OUT/triforce/states PYTHONPATH=OUT \
        python <checkout>/scripts/experiments/r28_demos.py ROOM [--runs N] [--out FILE.npz]

`scaffold` makes a throwaway data root: the checkout's data (the route `d8` as it is in routes.yaml) with staged
start states of steps 281-296 from the walks' staging (`r28_walks.leg_states`, `d8_state`, `d8_cellar_states`: 13
hearts, the magic sword, the route's keys, bombs and doors), named by the route's own visit counts, and the families'
yamls pointed at them (`route: d8`, `ram_randomization: {}`, no curriculum, no `reward.heart_value`: its table is
fitted for d2first).  It is what stands in until the stretch has start states; nothing in the checkout changes.

ROOM is one of:

    whirlwind    0:0x22 (ow-lynel): RECORDER where the mask offers it (twice), else WAIT.
    tree         0:0x6d (ow-moblin): MOVE S to the tree's row, CANDLE W where the mask offers it, three WAIT 32 while
                 the fire burns, then MOVE W onto the stairs.
    darknut-5e   8:0x5e (darknut): the fighter (below), then the key and the N door.
    darknut-3e   8:0x3e (darknut): the fighter, then round the statue by row Y $9D to the E door.
    pols         8:0x4c (pols-voice): straight along the way out of the ring, by the corner key, to the N wall; ARROW
                 along the leg when a Pols Voice is 8-96 px ahead within 8 px of Link's line; BOMB N at the wall.
    gleeok       8:0x3c (boss-gleeok): on row Y $9D under the head that hangs lowest (row $AD while BEAMS is
                 offered), SWORD or BEAMS N when a head is in reach, the heads' timer is 0 and no shot is predicted to
                 hit; else step aside; then the heart container and the N door.

Every rule decides once per environment action from the RAM the env exposes (positions, facings, the flash timers:
what a player sees) and the action mask; the walks' scripts decided once a frame.  What that costs is the point of
the numbers this prints: the share of runs each rule wins through the env, the hearts it loses, the endings.

The fighter (`r28_darknut.Fighter` over actions): SWORD (BEAMS where offered) toward a Darknut 10-27 px ahead on
Link's row or column, at most 9 px off it, that does not face the blade and does not flash, when nothing is predicted
to touch Link; else to a drop when he is not full; else MOVE 4 toward the nearest square beside or behind a Darknut,
taking of the offered moves (and WAIT 4) the one with the fewest predicted overlaps (`r28_walks.Dodger`: every object
carried forward 14 frames along its motion since the last decision).

Rows: a MOVE or WAIT row is cloned without its duration; an item or sword row is written `REPEAT` times (it is one
decision among hundreds).  Only runs that end as the family's success are kept."""

import argparse
import collections
import json
import os
import sys
from pathlib import Path

RUNS = 30
SEED0 = 1000
REPEAT = {'RECORDER': 40, 'CANDLE': 40, 'BOMB': 40, 'ARROW': 20, 'SWORD': 4, 'BEAMS': 4}

OBJ_X, OBJ_Y, OBJ_DIR, OBJ_STATE, OBJ_TYPE, OBJ_META, OBJ_INVINC = 0x70, 0x84, 0x98, 0xAC, 0x34F, 0x405, 0x4F0
FRAME, HEARTS = 0x15, 0x66F
DARKNUT_RED, DARKNUT_BLUE, GIBDO, POLS_VOICE, GLEEOK4, DROP = 0x0B, 0x0C, 0x30, 0x16, 0x45, 0x60
SHOTS = (0x55, 0x56, 0x53, 0x5B, 0x57, 0x58, 0x59)
VEC = {1: (1, 0), 2: (-1, 0), 4: (0, 1), 8: (0, -1)}
NAME = {1: 'E', 2: 'W', 4: 'S', 8: 'N'}
OPP = {1: 2, 2: 1, 4: 8, 8: 4}
SIDE = {1: (4, 8), 2: (4, 8), 4: (1, 2), 8: (1, 2)}
NECK_X, NECK_Y, DEAD_MASK, HEAD_SLOT = (0x438, 0x452, 0x46C, 0x395), (0x445, 0x45F, 0x479, 0x3BD), 0x511, 5
GLEEOK_BASE = (0x7C, 0x6F)
FLOOR = (0x20, 0x5D, 0xD0, 0xBD)

# room -> (family, the route step's index in d8, the family's step limit is its yaml's)
ROOMS = {'whirlwind': ('ow-lynel', 281), 'tree': ('ow-moblin', 286), 'darknut-5e': ('darknut', 289),
         'darknut-3e': ('darknut', 291), 'pols': ('pols-voice', 294), 'gleeok': ('boss-gleeok', 295)}


def objects(ram):
    """[(slot, type, x, y, facing, metastate)] of slots 1-11."""
    return [(s, ram[OBJ_TYPE + s], ram[OBJ_X + s], ram[OBJ_Y + s], ram[OBJ_DIR + s], ram[OBJ_META + s])
            for s in range(1, 12) if ram[OBJ_TYPE + s]]


class Eyes:
    """What moved since the last decision: [(x, y, vx, vy, radius)] of what can hurt (`r28_walks.Dodger.threats`
    with the motion taken between two decisions, not two frames)."""

    def __init__(self, harmless=(), radius=0x13, shot_radius=0x0D, horizon=14):
        self.harmless, self.radius, self.shot_radius, self.horizon = set(harmless), radius, shot_radius, horizon
        self.prev, self.frame = {}, None

    def threats(self, ram):
        frames = ((ram[FRAME] - self.frame) % 256 or 4) if self.frame is not None else 4
        self.frame = ram[FRAME]
        out, now = [], {}
        for s, t, x, y, _d, meta in objects(ram):
            old = self.prev.get(s)
            moved = old is not None and old[0] == t and abs(x - old[1]) < 2 * frames + 8 and abs(y - old[2]) < 2 * frames + 8
            aim = old[3] if moved else (ram[OBJ_X], ram[OBJ_Y])
            now[s] = (t, x, y, aim)
            if t in self.harmless or t >= 0x60 or meta >= 0x10:
                continue
            vx, vy = ((x - old[1]) / frames, (y - old[2]) / frames) if moved else (0.0, 0.0)
            if t in SHOTS and vx == 0 and vy == 0:               # not flying yet: aimed once, at where Link stood
                ax, ay = aim[0] - x, aim[1] - y
                d = max(1.0, (ax * ax + ay * ay) ** 0.5)
                vx, vy = 1.5 * ax / d, 1.5 * ay / d
            out.append((x, y, vx, vy, self.shot_radius if t in SHOTS else self.radius))
        self.prev = now
        return out

    def danger(self, x, y, dx, dy, threats, first=0):
        total = 0.0
        for f in range(first, self.horizon + 1):
            lx, ly = x + dx * 1.5 * f, y + dy * 1.5 * f
            for ox, oy, vx, vy, r in threats:
                if abs(ox + vx * f - lx) < r and abs(oy + vy * f - ly) < r:
                    total += 1.0 / (1 + f)
        return total

    def choose(self, ram, ok, goal, threats, bounds=FLOOR):
        """Of MOVE <d> 4 and WAIT 4 (those the mask offers, `ok`), the one with the fewest predicted overlaps, ties
        by the distance left to `goal`.  From the floor (`bounds`) no move is taken that would end off it; off it
        (in a doorway: sideways moves do not move him, and 50 decisions in one place end the run) Link walks in."""
        x, y = ram[OBJ_X], ram[OBJ_Y]
        inside = bounds and bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]
        if bounds and not inside:
            d = 4 if y < bounds[1] else 8 if y > bounds[3] else 1 if x < bounds[0] else 2
            if ok(f'MOVE {NAME[d]} 4'):
                return f'MOVE {NAME[d]} 4'
        best = None
        for d, (dx, dy) in list(VEC.items()) + [(0, (0, 0))]:
            step = f'MOVE {NAME[d]} 4' if d else 'WAIT 4'
            if not ok(step):
                continue
            ex, ey = x + dx * 6, y + dy * 6
            if d and inside and not (bounds[0] <= ex <= bounds[2] and bounds[1] <= ey <= bounds[3]):
                ex, ey = min(max(ex, bounds[0]), bounds[2]), min(max(ey, bounds[1]), bounds[3])
                if (ex, ey) == (x, y) or (abs(x - 0x78) < 8 if dy else abs(y - 0x8D) < 8):
                    continue                  # at the floor's edge, or on a doorway's line: the move would leave
            left = abs(goal[0] - ex) + abs(goal[1] - ey)
            cost = (round(self.danger(x, y, dx, dy, threats), 3), left)
            if best is None or cost < best[0]:
                best = (cost, step)
        return best[1] if best else 'WAIT 4'


class Legs:
    """The way out once a room is won: `legs` = [(axis, value)] walked in turn, one axis at a time (a MOVE of 8
    frames while 14 px or more are left, else of 4; within 3 px counts as there, the grid's snap does the rest), a
    leg ended early by its `done` test, then `last` while the mask offers it.  No stepping aside: mixing the two
    axes a decision at a time walks Link in circles between grid points."""

    def __init__(self, legs, last, done=None):
        self.legs, self.last, self.done, self.i = list(legs), last, done or {}, 0

    def step(self, ram, ok):
        pos = {'x': ram[OBJ_X], 'y': ram[OBJ_Y]}
        while self.i < len(self.legs):
            axis, value = self.legs[self.i]
            test = self.done.get(self.i)
            if test(ram) if test else abs(value - pos[axis]) <= 3:
                self.i += 1
                continue
            d = {'x': (1, 2), 'y': (4, 8)}[axis][0 if value >= pos[axis] else 1]
            for step in (f'MOVE {NAME[d]} {8 if abs(value - pos[axis]) >= 14 else 4}', f'MOVE {NAME[d]} 4'):
                if ok(step):
                    return step
            # A wall on the way (a statue, a block): along the other axis, toward the next leg's value first.
            other = (4, 8) if axis == 'x' else (1, 2)
            if self.i + 1 < len(self.legs) and self.legs[self.i + 1][0] != axis:
                axis2, value2 = self.legs[self.i + 1]
                other = other if value2 >= pos[axis2] else other[::-1]
            for d2 in other:
                if ok(f'MOVE {NAME[d2]} 4'):
                    return f'MOVE {NAME[d2]} 4'
            return 'WAIT 4'
        if ok(self.last):
            return self.last
        # Not on the doorway's line after all: the four moves by the distance left to it.
        axis, value = self.legs[-2] if len(self.legs) > 1 else self.legs[-1]
        d = {'x': (1, 2), 'y': (4, 8)}[axis][0 if value > pos[axis] else 1]
        return f'MOVE {NAME[d]} 4' if ok(f'MOVE {NAME[d]} 4') else 'WAIT 4'


def whirlwind_rule():
    def rule(_ram, ok):
        return 'RECORDER' if ok('RECORDER') else 'WAIT 8'
    return rule

def swing(ok, d):
    """SWORD (BEAMS where offered) toward `d` (a facing bit), or the step that makes it possible: the env's swing
    turns Link only where the game turns him (on a grid point, or right round), and its shortest MOVE is 4 frames,
    so between grid points facing sideways the rule steps back from the target (MOVE of 4 frames away: Link walks
    on to the grid point and turns) and swings round on the next decision.  None: neither is offered."""
    for step in (f'BEAMS {NAME[d]}', f'SWORD {NAME[d]}', f'MOVE {NAME[OPP[d]]} 4'):
        if ok(step):
            return step
    return None



def tree_rule():
    state = {'fired': False, 'waits': 0}

    def rule(ram, ok):
        x, y = ram[OBJ_X], ram[OBJ_Y]
        if not state['fired']:
            # Onto the tree's row first (Y $5D; a MOVE of 4 frames stops at $5B, and the turn W walks on to the grid
            # point): fired from the row above, Link then walks W over the stairs' square, not onto it.
            if not 0x5B <= y <= 0x61:
                return 'MOVE S 4' if y < 0x5B else 'MOVE N 4'
            if ok('CANDLE W'):
                state['fired'] = True
                return 'CANDLE W'
            return 'MOVE W 4' if x > 0xCF else 'MOVE E 4'
        if state['waits'] < 3:
            state['waits'] += 1
            return 'WAIT 32'
        return 'MOVE W 8'
    return rule


def darknut_rule(exit_path):
    """The fighter, then `exit_path()` (a `Legs`)."""
    eyes, leave = Eyes(), exit_path()
    targets, shielded = (DARKNUT_BLUE, DARKNUT_RED, GIBDO), (DARKNUT_BLUE, DARKNUT_RED)

    def rule(ram, ok):
        x, y = ram[OBJ_X], ram[OBJ_Y]
        threats = eyes.threats(ram)
        foes = [o for o in objects(ram) if o[1] in targets]
        if not foes:
            return leave.step(ram, ok)
        alive = [o for o in foes if o[5] == 0]
        beam = any(ok(f'BEAMS {n}') for n in 'NSWE')
        hits = []
        for s, kind, ox, oy, facing, _m in alive:
            if ram[OBJ_INVINC + s]:
                continue
            for d, (dx, dy) in VEC.items():
                along = (ox - x) * dx + (oy - y) * dy
                lateral = abs((ox - x) * dy) + abs((oy - y) * dx)
                parried = kind in shielded and facing == OPP[d]
                if 10 <= along <= (0x70 if beam else 27) and lateral <= 9 and not parried:
                    hits.append((along, d))
        if hits and eyes.danger(x, y, 0, 0, threats) < 0.35:
            step = swing(ok, min(hits)[1])
            if step:
                return step
        drops = [o for o in objects(ram) if o[1] == DROP]
        full = (ram[HEARTS] >> 4) == (ram[HEARTS] & 0x0F) and ram[HEARTS + 1] == 0xFF
        goal = (x, y)
        if drops and not full:
            goal = min((abs(o[2] - x) + abs(o[3] - y), (o[2], o[3])) for o in drops)[1]
        else:
            best = None
            for _s, _k, ox, oy, facing, _m in alive:
                facing = facing if facing in VEC else 4
                for d in (SIDE[facing][0], SIDE[facing][1], OPP[facing]):
                    gx, gy = ox + VEC[d][0] * 0x12, oy + VEC[d][1] * 0x12
                    if FLOOR[0] <= gx <= FLOOR[2] and FLOOR[1] <= gy <= FLOOR[3]:
                        dist = abs(gx - x) + abs(gy - y)
                        if best is None or dist < best[0]:
                            best = (dist, (gx, gy))
            goal = best[1] if best else goal
        return eyes.choose(ram, ok, goal, threats)
    return rule


def exit_5e():
    """Column X $78, down or up it over the key's square (taken at ($78,$8D)), up to the N door."""
    return Legs([('x', 0x78), ('y', 0x8D), ('y', 0x5D)], 'MOVE N 8')


def exit_3e():
    """Round the statue at X $90 by row Y $9D, back onto the door row, E."""
    return Legs([('y', 0x9D), ('x', 0xA0), ('y', 0x8D), ('x', 0xD0)], 'MOVE E 8')


def pols_rule():
    """Straight along the legs (back onto the leg's line after a knockback), an arrow at a Pols Voice ahead."""
    legs = [(0x70, 0x7D), (0x60, 0x7D), (0x60, 0x9D), (0xB0, 0x9D), (0xB0, 0x5D), (0x20, 0x5D), (0x20, 0xBD),
            (0xD0, 0xBD), (0xD0, 0x5D), (0xD0, 0xBD), (0x20, 0xBD), (0x20, 0x5D), (0x78, 0x5D)]
    state = {'i': 1, 'bombed': False}

    def toward(now, want, low, high):
        return None if abs(want - now) <= 2 else (high if want > now else low)

    def rule(ram, ok):
        x, y = ram[OBJ_X], ram[OBJ_Y]
        while state['i'] < len(legs) and abs(legs[state['i']][0] - x) + abs(legs[state['i']][1] - y) <= 4:
            state['i'] += 1
        if state['i'] >= len(legs):
            if not state['bombed'] and ok('BOMB N'):
                state['bombed'] = True
                return 'BOMB N'
            return 'MOVE N 8' if ok('MOVE N 8') else 'WAIT 4'
        (ax, ay), (px, py) = legs[state['i'] - 1], legs[state['i']]
        if ay == py:                                             # a leg along a row: onto the row, then along it
            d = toward(y, py, 8, 4) or toward(x, px, 2, 1)
        else:
            d = toward(x, px, 2, 1) or toward(y, py, 8, 4)
        if d is None:
            return 'WAIT 4'
        dx, dy = VEC[d]
        ahead = [o for o in objects(ram) if o[1] == POLS_VOICE and o[5] == 0
                 and 8 <= (o[2] - x) * dx + (o[3] - y) * dy <= 0x60 and abs((o[2] - x) * dy) + abs((o[3] - y) * dx) <= 8]
        if ahead and ram[OBJ_STATE + 0x12] == 0 and ok(f'ARROW {NAME[d]}'):
            return f'ARROW {NAME[d]}'
        far = abs(px - x) + abs(py - y) if (ay == py) == (d in (1, 2)) else 0
        step = f'MOVE {NAME[d]} {8 if far >= 14 else 4}'
        return step if ok(step) else 'WAIT 4'
    return rule


def gleeok_heads(ram):
    if ram[OBJ_TYPE + 1] != GLEEOK4 or 0x10 <= ram[OBJ_META + 1] <= 0x13:
        return []
    return [(ram[NECK_X[n] + 4], ram[NECK_Y[n] + 4]) for n in range(4) if not ram[DEAD_MASK] & (1 << n)]


def gleeok_rule(row=0x9D):
    eyes = Eyes(harmless=(GLEEOK4,))
    state = {}

    def taken(ram):
        return (ram[HEARTS] >> 4) > state.setdefault('containers', ram[HEARTS] >> 4)
    # Off the door column first (a MOVE S on it walks out through the bombed S wall), down to the bottom row, W along
    # it until the heart container is taken (at ($28,$BD)), back to the door column, up.
    leave = Legs([('x', 0x60), ('y', 0xBD), ('x', 0x20), ('x', 0x78), ('y', 0x5D)], 'MOVE N 8', done={2: taken})

    def rule(ram, ok):
        x, y = ram[OBJ_X], ram[OBJ_Y]
        seen = eyes.threats(ram)
        heads = gleeok_heads(ram)
        if not heads:
            return leave.step(ram, ok)
        threats = seen + [(hx, hy, 0, 0, 0x0F) for hx, hy in heads] + [(GLEEOK_BASE[0], GLEEOK_BASE[1], 0, 0, 0x12)]
        beam = ok('BEAMS N')
        reach = [h for h in heads if abs(h[0] - x) <= 8 and (10 <= y - h[1] <= 27 or (beam and y - h[1] >= 10))]
        shots = [t for t in threats if t[4] < 0x0F]
        if ram[OBJ_INVINC + HEAD_SLOT] == 0 and reach and eyes.danger(x, y, 0, 0, shots) < 0.3:
            step = swing(ok, 8)
            if step:
                return step
        low = max(heads, key=lambda h: (h[1], -abs(h[0] - x)))
        return eyes.choose(ram, ok, (low[0] & 0xF8, max(row, 0xAD) if beam else row), threats)
    return rule


RULES = {'whirlwind': whirlwind_rule, 'tree': tree_rule, 'darknut-5e': lambda: darknut_rule(exit_5e),
         'darknut-3e': lambda: darknut_rule(exit_3e), 'pols': pols_rule, 'gleeok': gleeok_rule}


def stems():
    """{route step index: the start state's name without its chain}, steps 281-296 of d8."""
    from triforce import _native                               # pylint: disable=import-outside-toplevel
    from triforce.entry_states import state_stem                # pylint: disable=import-outside-toplevel
    return {s['index']: (state_stem(s['level'], s['room'], s['in_cave'], s['entry'], s['visit']), s)
            for s in _native.route_steps('d8')[281:]}


def scaffold(args):
    import yaml                                                 # pylint: disable=import-outside-toplevel
    from r26_walks import ROOT                                  # pylint: disable=import-outside-toplevel
    from r28_walk_scaffold import LEG                           # pylint: disable=import-outside-toplevel
    from r28_walks import d8_cellar_states, d8_state, leg_states  # pylint: disable=import-outside-toplevel
    out = args.out.resolve()
    (out / 'triforce').mkdir(parents=True)
    for f in (ROOT / 'triforce').iterdir():
        if f.name != 'states':
            (out / 'triforce' / f.name).symlink_to(f)
    for name in ('roms', 'models', 'trajectories', 'scripts', 'zelda-asm'):
        if (ROOT / name).exists():
            (out / name).symlink_to(ROOT / name)
    states = out / 'triforce' / 'states'
    states.mkdir()
    for f in (ROOT / 'triforce' / 'states').iterdir():
        (states / f.name).symlink_to(f)
    leg, cellar = leg_states(out / 'leg-states'), d8_cellar_states(out / 'd8-states')
    names = collections.defaultdict(list)
    for stem, step in stems().values():
        level, room, entry = step['level'], step['room'], step['entry']
        if level == 0:
            source = leg[LEG[(room, entry)]]
        elif step['in_cave']:
            source = cellar['8_2fc']
        elif entry == 'cave':
            source = cellar['8_4cc']
        else:
            hearts = args.boss_hearts if room == 0x3C and args.boss_hearts is not None else args.hearts
            source = d8_state(room, entry, path=out / 'd8-states' / f'8_{room:02x}{entry.lower()}.state', hearts=hearts)
        (states / f'{stem}_0.state').write_bytes(Path(source).read_bytes())
        names[step['specialist']].append(stem)
    (out / 'specialists').mkdir()
    for f in (ROOT / 'specialists').glob('*.yaml'):
        data = yaml.safe_load(f.read_text())
        if data['name'] in names:
            data['route'] = 'd8'
            data.get('reward', {}).pop('heart_value', None)
            data['ram_randomization'] = {}
            data['curriculum'] = {'enabled': False}
            data['start_states'] = [f'{stem}_[0-9]' for stem in names[data['name']]]
        (out / 'specialists' / f.name).write_text(yaml.safe_dump(data, sort_keys=False))
    print(f'scaffold in {out}: ' + '; '.join(f'{k}: {v}' for k, v in names.items()))
    print(f'play with: cd {out} && TRIFORCE_STATES_DIR={states} PYTHONPATH={out} python {Path(__file__).resolve()} ROOM')


def play(args):
    from triforce._native import Env                            # pylint: disable=import-outside-toplevel
    from triforce.demos import DemoRecorder, action_indices, legal  # pylint: disable=import-outside-toplevel
    family, index = ROOMS[args.what]
    state = stems()[index][0] + '_0'
    env = Env(family, explicit_starts=True, strict_actions=False, seed=0)
    recorder = DemoRecorder()
    endings, lost, steps_won, kinds = collections.Counter(), [], [], collections.Counter()
    kept = 0
    for seed in range(SEED0, SEED0 + args.runs):
        obs, _ = env.reset(seed=seed, options={'start_state': state})
        rule = RULES[args.what]()
        health, info, n, illegal = env.state.health, {}, 0, 0
        for n in range(1, args.max_steps + 1):
            mask = obs['action_mask']

            def ok(step, mask=mask):
                return legal(mask, action_indices(step))
            step = rule(bytes(env.state.ram), ok)
            if ok(step):
                kind = step.split()[0]
                recorder.add(obs, step, repeat=REPEAT.get(kind, 1), any_duration=kind in ('MOVE', 'WAIT'))
                kinds[kind] += 1
            else:
                illegal += 1
            obs, _, terminated, truncated, info = env.step(step)
            if terminated or truncated:
                break
        won = info.get('outcome') == 'success'
        endings[info.get('ending', 'max-steps (the script\'s)')] += 1
        if won:
            recorder.keep()
            kept += 1
            lost.append(health - env.state.health)
            steps_won.append(n)
        else:
            recorder.drop()
        if args.verbose:
            print(f'  seed {seed}: {info.get("ending")} after {n} decisions, hearts {health} -> {env.state.health}, '
                  f'{illegal} masked')
    rows = recorder.save(args.out) if args.out and kept else 0
    mean = lambda v: sum(v) / len(v) if v else float('nan')     # noqa: E731
    print(json.dumps({'room': args.what, 'family': family, 'state': state, 'runs': args.runs, 'won': kept,
                      'endings': dict(endings), 'hearts_lost_in_wins': round(mean(lost), 2),
                      'hearts_lost_max': max(lost, default=None), 'decisions_in_wins': round(mean(steps_won), 1),
                      'actions_kept': dict(kinds), 'rows': rows, 'file': str(args.out) if rows else None}))
    return 0 if kept else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['scaffold'] + list(ROOMS))
    parser.add_argument('out_dir', nargs='?', type=Path, help='scaffold: the throwaway data root')
    parser.add_argument('--hearts', type=float, default=None, help="scaffold: Dungeon 8 states' hearts (default 13)")
    parser.add_argument('--boss-hearts', type=float, default=None, help="scaffold: Gleeok's state's hearts")
    parser.add_argument('--runs', type=int, default=RUNS)
    parser.add_argument('--max-steps', type=int, default=3000, help='decisions a run may take')
    parser.add_argument('--out', type=Path, default=None, help='write the kept runs\' rows here (.npz)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.what == 'scaffold':
        if args.out_dir is None:
            parser.error('scaffold needs OUT')
        args.out = args.out_dir
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        return scaffold(args) or 0
    if 'TRIFORCE_STATES_DIR' not in os.environ:
        parser.error('run inside a scaffold: TRIFORCE_STATES_DIR=OUT/triforce/states PYTHONPATH=OUT (see scaffold)')
    return play(args)


if __name__ == '__main__':
    sys.exit(main())
