"""Round 32's walks of `d3first`, first half (docs/routes/d3first-walks-a.md): the game start to Level 5's entrance
with the wooden sword, as route steps played in the running game (`r32_walks`).

    python scripts/experiments/r32_d3first.py stage [--legs start,d3,...]      # the staging walk, the ledger
    python scripts/experiments/r32_d3first.py rooms [--legs ...] [--seeds N] [--steps NAME,...] [--hearts H]
    python scripts/experiments/r32_d3first.py chain --legs d3 [--seeds N] [--hearts H]  # the leg in one game

`stage` walks every step once from `start.state` with the monsters pinned and saves each step's pre-entry state under
target/r32-walks/states (the route's state names); `rooms` plays each step alone from its state, 20 seeds, monsters
live; `chain` plays the legs' steps one after another in one game from the first step's state, as a run would."""

import argparse
import collections
import json
import statistics
from pathlib import Path

from r26_walks import ROOT
from r32_walks import Failed, Runner, first_monsters, hp, load, set_hearts, stage, summary, trial

OUT = ROOT / 'target' / 'r32-walks'
STATE_DIR = OUT / 'states'

UP, DOWN, LEFT, RIGHT = 'UP', 'DOWN', 'LEFT', 'RIGHT'

# ---- the route, leg by leg: (state name, actions) ------------------------------------------------------------------
# The first step of each leg starts where the previous leg's last step left Link.

LEGS = collections.OrderedDict()

LEGS['start'] = [
    ('start', [('walk', (0x40, 0x8D), (0x40, 0x5D)), ('go', UP)]),                 # 0x77: into the sword cave
    ('0_77cave', [('cave_item', 0x78, 'sword'), ('go', DOWN)]),                     # the wood sword, out
    ('0_77c', [('walk', (0x40, 0x8D)), ('go', LEFT, ('y', 0x8D))]),                  # 0x77 cave -> W
    ('0_76e', [('walk', (0x90, 0x8D), (0x90, 0x7D), (0x08, 0x7D)), ('go', LEFT, ('y', 0x7D))]),   # 0x76 round its rocks
    ('0_75e', [('clip', 'E')]),                                                     # the wrap exit into 0x74
    ('0_74e', [('walk', (0xD0, 0x7D), (0xD0, 0x8D), (0x80, 0x8D)), ('go', UP, ('x', 0x80))]),   # into the mouth
]

LEGS['d3'] = [
    ('3_7cs', [('walk', (0x78, 0x95), (0x20, 0x95), (0x20, 0x8D)), ('go', LEFT, ('y', 0x8D))]),   # round the statues
    ('3_7be', [('take', (0x78, 0x8D), 'key'), ('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]),
    ('3_6bs', [('clear',), ('take', (0x78, 0x8D), 'key'), ('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]),
    ('3_5bs', [('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]),                    # past the 3 Darknuts
    ('3_4bs', [('clear',), ('take', (0x78, 0x8D), 'key'), ('walk', (0x20, 0x8D)), ('go', LEFT, ('y', 0x8D))]),
    ('3_4ae', [('clear',), ('take', (0x78, 0x8D), 'bombs'), ('walk', (0x20, 0x8D)), ('go', LEFT, ('y', 0x8D))]),
    ('3_49e', [('take', (0x78, 0x8D), 'key'), ('walk', (0x78, 0xBD)), ('go', DOWN, ('x', 0x78))]),
    ('3_59n', [('clear',), ('walk', (0x78, 0xBD)), ('go', DOWN, ('x', 0x78))]),      # the S shutter
    ('3_69n', [('walk', (0xC8, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),                 # past the 8 Darknuts: stairs
    ('3_0fcave', [('cellar_item', 'raft')]),
    ('3_69c', [('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]),
    ('3_59s', [('walk', (0xD0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),                 # the key door E
    ('3_5aw', [('walk', (0xD0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),
    ('3_5bw', [('clear',), ('take', (0x78, 0x8D), 'bombs'), ('walk', (0xC0, 0x8D)), ('bomb', RIGHT),
               ('go', RIGHT, ('y', 0x8D))]),
    ('3_5cw', [('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]),
    ('3_4cs', [('walk', (0xC0, 0x8D)), ('bomb', RIGHT), ('go', RIGHT, ('y', 0x8D))]),
    ('3_4dw', [('clear', 'manhandla'), ('take', (0x78, 0x8D), 'heart'), ('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]),
    ('3_3ds', [('triforce', (0x78, 0x9D))]),
]


LEGS['d3-d4'] = [                                                                  # d2first's steps 105-111
    ('0_74c', [('walk', (0x08, 0x8D)), ('go', LEFT, ('y', 0x8D))]),
    ('0_73e', [('walk', (0x80, 0x45)), ('go', UP, ('x', 0x80))]),
    ('0_63s', [('walk', (0xF0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),
    ('0_64w', [('walk', (0xF0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),
    ('0_65w', [('walk', (0x80, 0x45)), ('go', UP, ('x', 0x80))]),
    ('0_55s', [('raft', UP)]),
    ('0_45s', [('go', UP, ('x', 0x80))]),
]

LEGS['d4'] = [                                                                     # d2first's steps 112-133
    ('4_71s', [('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]),
    ('4_61s', [('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]),
    ('4_51s', [('take', (0x90, 0x8D), 'key'), ('walk', (0x20, 0x8D)), ('go', LEFT, ('y', 0x8D))]),
    ('4_50e', [('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]),
    ('4_40s', [('take', (0x78, 0x7D), 'key'), ('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]),
    ('4_30s', [('walk', (0xD0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),               # the key door E
    ('4_31w', [('clear',), ('walk', (0xD0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),     # the E shutter
    ('4_32w', [('clear',), ('walk', (0x50, 0x8D)), ('push', RIGHT), ('walk', (0xC8, 0x5D)),
               ('go', RIGHT, ('y', 0x5D))]),                                        # the block, the stairs
    ('4_60cave', [('cellar_item', 'ladder')]),
    ('4_32c', [('walk', (0x20, 0x8D)), ('go', LEFT, ('y', 0x8D))]),
    ('4_31e', [('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]),                   # the key door N: the island
    ('4_21s', [('straight', (0x78, 0x7D)), ('clip_once', 'N'), ('straight', (0x78, 0x5D)), ('bomb', UP),
               ('go', UP, ('x', 0x78))]),                                           # the ladder clip, the wall
    ('4_11s', [('walk', (0x78, 0xAD), (0x78, 0x9D), (0x60, 0x8D), (0x90, 0x8D), (0x78, 0x7D), (0x78, 0x6D),
                         (0x78, 0x5D)), ('bomb', UP), ('go', UP, ('x', 0x78))]),     # the ten rupees, the N wall
    ('4_01s', [('straight', (0x78, 0x6D), (0x78, 0x8D), (0xD0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),   # the key
                                                                                    # on the water, the ladder
    ('4_02w', [('walk', (0x78, 0xBD)), ('go', DOWN, ('x', 0x78))]),
    ('4_12n', [('clear',), ('walk', (0x50, 0x8D)), ('push', RIGHT), ('walk', (0xD0, 0x8D)),
               ('go', RIGHT, ('y', 0x8D))]),
    ('4_13w', [('clear', 'gleeok'), ('take', (0xD0, 0xBD), 'heart'), ('walk', (0x78, 0x5D)),
               ('go', UP, ('x', 0x78))]),
    ('4_03s', [('triforce', (0x78, 0x9D))]),
]

LEGS['d4-d1'] = [                                       # by raft to 0x55, then 0x56 N, 0x46, 0x47, 0x48, 0x38, 0x37
    ('0_45c', [('raft', DOWN)]),
    ('0_55n', [('straight', (0x80, 0x8D)), ('walk', (0xF0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),   # off the dock
    ('0_56w', [('walk', (0x78, 0x45)), ('go', UP, ('x', 0x78))]),
    ('0_46s', [('walk', (0xF0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),
    ('0_47w', [('walk', (0xF0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),
    ('0_48w', [('walk', (0x78, 0x45)), ('go', UP, ('x', 0x78))]),
    ('0_38s', [('walk', (0x08, 0x8D)), ('go', LEFT, ('y', 0x8D))]),
    ('0_37e', [('walk', (0x70, 0x85)), ('go', UP, ('x', 0x70))]),
]

LEGS['d1'] = [                     # phase IX's Dungeon 1: out and back in for the N door, 0x53's N wall, steps 56-67
    ('1_73s', [('walk', (0x78, 0xBD)), ('go', DOWN, ('x', 0x78))]),                 # out again at once
    ('0_37c', [('straight', (0x70, 0x85)), ('go', UP, ('x', 0x70))]),                # off the mouth, back in
    ('1_73s_v2', [('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]),                # the N door, open
    ('1_63s', [('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]),
    ('1_53s', [('clear',), ('take', (0x78, 0x6D), 'key'), ('walk', (0x78, 0x5D)), ('bomb', UP),
               ('go', UP, ('x', 0x78))]),
    ('1_43s', [('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]),
    ('1_33s', [('clear',), ('take', None, 'key'), ('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]),  # carried
    ('1_23s', [('clear',), ('take', (0x78, 0x6D), 'key'), ('walk', (0x20, 0x8D)), ('go', LEFT, ('y', 0x8D))]),
    ('1_22e', [('walk', (0xA0, 0x9D)), ('clip_once', 'N'), ('go', LEFT)]),            # into the diamond: stairs
    ('1_7fcave', [('cellar_item', 'bow')]),
    ('1_22c', [('walk', (0x60, 0xB5), (0xC0, 0xB5), (0xC0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),   # round, timed by the traps
    ('1_23w', [('walk', (0x78, 0xBD)), ('go', DOWN, ('x', 0x78))]),
    ('1_33n', [('walk', (0x78, 0xBD)), ('go', DOWN, ('x', 0x78))]),
    ('1_43n', [('walk', (0xD0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),                 # the key door E
    ('1_44w', [('clear',), ('take', (0x78, 0x6D), 'boomerang'), ('walk', (0xD0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),
    ('1_45w', [('take', (0xA0, 0xBD), 'key'), ('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]),
    ('1_35s', [('clear',), ('take', (0xC0, 0x8D), 'heart'), ('walk', (0xD0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),
    ('1_36w', [('triforce', (0x78, 0x9D))]),
]

LEGS['d1-d5'] = [                 # 0x37 E, 0x38 N, 0x28 E, 0x29-0x2b E, heart rock 0x2c, the 30 secret 0x2d,
    ('0_37c_v2', [('walk', (0xF0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),            # 0x2c, 0x1c, 0x0c's candle, 0x0b
    ('0_38w', [('walk', (0x78, 0x45)), ('go', UP, ('x', 0x78))]),
    ('0_28s', [('walk', (0xF0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),
    ('0_29w', [('walk', (0xF0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),
    ('0_2aw', [('walk', (0xF0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]),
    ('0_2bw', [('walk', (0xF0, 0xAD)), ('go', RIGHT, ('y', 0xAD))]),
    ('0_2cw', [('walk', (0x90, 0xA5)), ('bomb', UP), ('walk', (0x90, 0xA5)), ('go', UP, ('x', 0x90))]),
    ('0_2ccave', [('cave_item', 0x98, 'heart'), ('go', DOWN)]),
    ('0_2cc', [('walk', (0xF0, 0xAD)), ('go', RIGHT, ('y', 0xAD))]),
    ('0_2dw', [('walk', (0x50, 0x5D)), ('bomb', UP), ('go', UP, ('x', 0x50))]),
    ('0_2dcave', [('cave_item', 0x78, 'rupees'), ('go', DOWN)]),
    ('0_2dc', [('walk', (0x08, 0x8D)), ('go', LEFT, ('y', 0x8D))]),                 # back W: 0x1d's S part
    ('0_2ce', [('walk', (0x30, 0x55)), ('go', UP, ('x', 0x30))]),                    # leads only E
    ('0_1cs', [('walk', (0x50, 0x5D)), ('go', UP, ('x', 0x50))]),
    ('0_0cs', [('walk', (0x80, 0x55)), ('go', UP, ('x', 0x80))]),
    ('0_0ccave', [('need_rupees', 60), ('cave_item', 0x98, 'candle'), ('go', DOWN)]),
    ('0_0cc', [('walk', (0xF0, 0x8D)), ('clip', 'E')]),
    ('0_0be', [('walk', (0x70, 0x85)), ('go', UP, ('x', 0x70))]),
]

# Other ways to play a step, for the comparisons of the document: {step: {variant: actions}}
VARIANTS = {
    '3_59n': {'bombs': [('clear', 'bombs'), ('walk', (0x78, 0xBD)), ('go', DOWN, ('x', 0x78))]},
    '3_69n': {'dash': [('dash', (0xC8, 0x8D)), ('go', RIGHT, ('y', 0x8D))]},
    '3_5bw': {'bombs': [('clear', 'bombs'), ('take', (0x78, 0x8D), 'bombs'), ('walk', (0xC0, 0x8D)), ('bomb', RIGHT),
                        ('go', RIGHT, ('y', 0x8D))]},
    '3_5bs': {'dash': [('dash', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]},
    '3_4dw': {'sword': [('clear',), ('take', (0x78, 0x8D), 'heart'), ('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]},
    '1_53s': {'nokey': [('walk', (0x78, 0x5D)), ('bomb', UP), ('go', UP, ('x', 0x78))]},
    '1_33s': {'nokey': [('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]},
    '1_23s': {'nokey': [('walk', (0x20, 0x8D)), ('go', LEFT, ('y', 0x8D))]},
    '4_21s': {'clear': [('clear',), ('straight', (0x78, 0x7D)), ('clip_once', 'N'), ('straight', (0x78, 0x5D)),
                        ('bomb', UP), ('go', UP, ('x', 0x78))]},
    '1_22c': {'straight': [('straight', (0x60, 0xB5), (0xC0, 0xB5), (0xC0, 0x8D)), ('go', RIGHT, ('y', 0x8D))]},
    '1_45w': {'nokey': [('walk', (0x78, 0x5D)), ('go', UP, ('x', 0x78))]},
    # the plain screen wrap (`CLIP W|E` offered within 32 px of the bound, `scan`): to the far edge, one step on
    '0_76e': {'wrap': [('clip_once', 'E'), ('go', LEFT, ('y', 0x8D))]},
    '0_64w': {'wrap': [('clip_once', 'W'), ('go', RIGHT, ('y', 0x8D))]},
    '0_47w': {'wrap': [('clip_once', 'W'), ('go', RIGHT, ('y', 0x8D))]},
    '0_29w': {'wrap': [('clip_once', 'W'), ('go', RIGHT, ('y', 0x8D))]},
    '0_2aw': {'wrap': [('clip_once', 'W'), ('go', RIGHT, ('y', 0x8D))]},
    '0_2bw': {'wrap': [('clip_once', 'W'), ('go', RIGHT, ('y', 0x8D))]},
}
# The best play the walks found where it differs from the main actions (the chains' `--best`)
BEST = {'3_5bs': 'dash', '3_59n': 'bombs', '3_69n': 'dash', '4_21s': 'clear', '1_23s': 'nokey', '0_76e': 'wrap',
        '0_64w': 'wrap', '0_47w': 'wrap', '0_29w': 'wrap', '0_2aw': 'wrap', '0_2bw': 'wrap'}


def best_steps(steps):
    return [(n, VARIANTS[n][BEST[n]] if n in BEST else a) for n, a in steps]


def steps_of(legs):
    out = []
    for leg in legs:
        out += LEGS[leg]
    return out


def all_steps():
    return steps_of(LEGS)


def start_of(name):
    """The saved state of step `name` (the first step: start.state)."""
    if name == 'start':
        from r26_walks import STATES        # pylint: disable=import-outside-toplevel
        return STATES / 'start.state'
    return STATE_DIR / f'{name}.state'


def cmd_stage(args):
    steps = all_steps()
    names = [s[0] for s in steps]
    first = names.index(args.start or LEGS[args.legs[0]][0][0])
    last = names.index(LEGS[args.legs[-1]][-1][0])
    paths, rows, cheats = stage(steps[first:last + 1], STATE_DIR, start=start_of(names[first]))
    print('cheats:', cheats)
    ledger = OUT / 'ledger.json'
    old = dict(json.loads(ledger.read_text())) if ledger.exists() else {}
    old.update(dict(rows))
    ledger.write_text(json.dumps(old, indent=1))


def cmd_rooms(args):
    steps = steps_of(args.legs)
    if args.steps:
        steps = [s for s in steps if s[0] in args.steps]
    for name_, actions in steps:
        if args.variant:
            if args.variant not in VARIANTS.get(name_, {}):
                continue
            actions = VARIANTS[name_][args.variant]
        path = start_of(name_)
        setup = None
        if args.containers:
            def setup(w, n=args.containers):                 # more containers than the route gives (full)
                w.game.set_value('hearts_and_containers', ((n - 1) << 4) | (n - 1))
                w.game.set_value('partial_hearts', 0xFF)
        rows = [trial(path, seed, actions, hearts=args.hearts, setup=setup) for seed in range(args.seeds)]
        ends = collections.Counter(r['end']['where'] for r in rows)
        label = f'{name_} ({args.variant})' if args.variant else name_
        hearts = f' from {args.hearts} hearts' if args.hearts is not None else ''
        hearts += f' ({args.containers} containers, full)' if args.containers else ''
        cheats = collections.Counter(c for r in rows for c in r['cheats'])
        print(f'{label}{hearts}: monsters {first_monsters(path)}; {summary(rows)}; ends {dict(ends)}'
              + (f'; cheats {dict(cheats)}' if cheats else '') + (
                  f"; bombs at the end mean {statistics.mean(r['end']['bombs'] for r in rows):.1f}"), flush=True)


def cmd_chain(args):
    steps = steps_of(args.legs)
    if args.best:
        steps = best_steps(steps)
    per = collections.defaultdict(list)
    total = []
    for seed in range(args.seeds):
        w = load(start_of(steps[0][0]), seed)
        if args.hearts is not None:
            set_hearts(w, args.hearts)
        w.until(lambda w: w.mode in (5, 9, 0x0B), 'NONE', 2000)
        reached, why = 0, 'ok'
        for name_, actions in steps:
            run = Runner(w, stage=False)
            h0 = hp(w)
            try:
                run.play(actions)
            except Failed as e:
                why = f'{name_}: {e}'
            per[name_].append((why == 'ok', h0 - hp(w), run.frames(), h0, run.cheats))
            if why != 'ok':
                break
            reached += 1
        total.append((why, reached, hp(w), w.max_hearts()))
        print(f'seed {seed}: {why[:110]}; steps {reached}/{len(steps)}; hearts {hp(w):.2f}/{w.max_hearts()}; '
              f'keys {w.r(0x66E)} bombs {w.r(0x658)} rupees {w.r(0x66D)}', flush=True)
    for name_, _ in steps:
        rows = per.get(name_, [])
        if rows:
            cheats = collections.Counter(c.split(' at ')[0] for r in rows for c in r[4])
            print(f'  {name_}: {sum(r[0] for r in rows)}/{len(rows)}, entered with {statistics.mean(r[3] for r in rows):.2f}'
                  f' hearts, net loss {statistics.mean(r[1] for r in rows):+.2f} (drops count), frames median '
                  f'{int(statistics.median(r[2] for r in rows))}' + (f', cheats {dict(cheats)}' if cheats else ''))
    ok = sum(1 for t in total if t[0] == 'ok')
    print(f'through: {ok}/{len(total)}')


def cmd_scan(args):
    """Where the session offers CLIP (`Game.skip_available`, what a `clips: true` family's mask reads) in each step's
    room: Link set on every 8-pixel grid point of the floor (overworld X $00-$F0, Y $3D-$DD; dungeon X $20-$D0,
    Y $5D-$BD), monsters emptied, one frame, then asked in the four directions (round 26's `r26_clip_scan`)."""
    from r26_walks import Walk                       # pylint: disable=import-outside-toplevel
    seen = set()
    for name_, _ in steps_of(args.legs):
        path = start_of(name_)
        w = Walk(seed=0)
        w.game.load(str(path), randomize_rng=True, seed=0)
        w.until(lambda w: w.mode in (5, 9), 'NONE', 600)
        if w.mode != 5 or (w.level, w.room) in seen:
            continue
        seen.add((w.level, w.room))
        w.game.set_clips(True)
        w.game.set_clips_allowed(True)
        for s in range(1, 0x0C):
            if w.r(0x34F + s) < 0x60:
                w.game.write(0x34F + s, 0)
        snap, offered = w.game.snapshot(), collections.defaultdict(list)
        (x0, x1), (y0, y1) = ((0x20, 0xD0), (0x5D, 0xBD)) if w.level else ((0x00, 0xF0), (0x3D, 0xDD))
        for y in range(y0, y1 + 1, 8):
            for x in range(x0, x1 + 1, 8):
                w.game.restore(snap)
                w.game.apply_ram({'link_x': x, 'link_y': y})
                w.press('NONE', 1)
                if (w.x, w.y) == (x, y) and (w.level, w.room) == next(iter([(w.level, w.room)])):
                    for d in 'NSWE':
                        if w.game.skip_available(d):
                            offered[d].append((x, y))
        text = '; '.join(f'{d} on {len(v)} points (rows {sorted({p[1] for p in v})[0]:#04x}-'
                         f'{sorted({p[1] for p in v})[-1]:#04x}, X {min(p[0] for p in v):#04x}-{max(p[0] for p in v):#04x})'
                         for d, v in offered.items())
        print(f'{w.level}:0x{w.room:02x} ({name_}): {text or "nothing"}', flush=True)


# Families of rooms d2first does not tag (the walk's proposal: the family whose monsters they hold)
NEW_FAMILY = {(0, 0x46, False): 'ow-octorok', (0, 0x29, False): 'ow-leever', (0, 0x2a, False): 'ow-leever',
              (0, 0x2b, False): 'ow-leever', (0, 0x2d, True): 'ow-zora', (0, 0x0c, True): 'ow-tektite',
              (3, 0x4a, False): 'stalfos-keese-gel'}
TAKES = {'cave_item': lambda a: a[2], 'take': lambda a: a[2], 'cellar_item': lambda a: a[1],
         'triforce': lambda a: 'triforce', 'bomb': lambda a: f'bomb {a[1]}', 'clear': lambda a: 'clear',
         'push': lambda a: 'push', 'clip': lambda a: f'CLIP {a[1]} (wrap exit)',
         'clip_once': lambda a: f'CLIP {a[1]}', 'need_rupees': lambda a: f'{a[1]} rupees'}
SIDE = {'n': 'N', 's': 'S', 'w': 'W', 'e': 'E', 'c': 'cave', 'cave': 'S'}


def where_of(name_):
    """(level, room, in_cave, entry, visit) of a state name."""
    import re                                         # pylint: disable=import-outside-toplevel
    if name_ == 'start':
        return 0, 0x77, False, 'start', 1
    m = re.match(r'(\d)_([0-9a-f]{2})(cave|[nswec])(?:_v(\d))?$', name_)
    level, room, entry, visit = int(m.group(1)), int(m.group(2), 16), m.group(3), int(m.group(4) or 1)
    return level, room, entry == 'cave', SIDE[entry], visit


def cmd_table(args):
    """The half as route steps: room, in, out, what is taken, family, and how d2first knows the room."""
    import re                                         # pylint: disable=import-outside-toplevel
    from triforce import _native                      # pylint: disable=import-outside-toplevel
    d2 = _native.route_steps('d2first')
    fam = {}
    for line in (ROOT / 'triforce' / 'routes.yaml').read_text().split('\n'):
        m = re.match(r'  ([a-z0-9-]+): \[(.*)\]$', line)
        if m:
            for r in re.findall(r"'\((\d), (0x[0-9a-f]+)(, cave)?\)'", m.group(2)):
                fam[(int(r[0]), int(r[1], 16), bool(r[2]))] = m.group(1)
    steps = all_steps()
    print('| # | State | Room | In | Out | Taken / done | Family | On d2first |')
    print('|---|---|---|---|---|---|---|---|')
    for i, (name_, actions) in enumerate(steps):
        level, room, cave, entry, _ = where_of(name_)
        nxt = where_of(steps[i + 1][0]) if i + 1 < len(steps) else None
        last = actions[-1]
        if last[0] == 'triforce':
            out = 'triforce'
        elif nxt is None:
            out = 'cave'
        elif cave:
            out = 'N' if level else 'S'                                  # a cellar's ladder; a cave's mouth
        elif level and nxt[0] == 0:
            out = 'S (out to the overworld)'
        elif nxt[2] or nxt[0] != level or (nxt[3] == 'cave' and nxt[1] == room) or last[0] == 'stairs' or \
                (last[0] == 'go' and nxt[3] == 'S' and nxt[0] != level):
            out = 'cave'
        elif last[0] == 'raft':
            out = 'N' if last[1] == 'UP' else 'S'
        elif last[0] == 'clip':
            out = 'W'
        else:
            out = {'UP': 'N', 'DOWN': 'S', 'LEFT': 'W', 'RIGHT': 'E'}.get(last[1], '?')
        if name_ in ('1_22e',):
            out = 'cave'
        key = (level, room, cave)
        family = fam.get(key) or NEW_FAMILY.get(key, '?') + ' (new)'
        same = [s for s in d2 if (s.get('level', 0), s['room'], bool(s.get('in_cave'))) == key]
        if not same:
            known = 'new room'
        elif any(s['entry'] == entry and s['exit'] == out for s in same):
            known = 'same entry and exit'
        elif any(s['entry'] == entry for s in same):
            known = 'same entry, new exit (' + '/'.join(sorted({s['exit'] for s in same if s['entry'] == entry})) + ')'
        else:
            known = 'new entry (' + ', '.join(sorted({f"{s['entry']}->{s['exit']}" for s in same})) + ')'
        took = ', '.join(TAKES[a[0]](a) for a in actions if a[0] in TAKES)
        print(f'| {i} | `{name_}` | {level}:0x{room:02x}{" cave" if cave else ""} | {entry} | {out} | {took} | '
              f'{family} | {known} |')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('cmd', choices=['stage', 'rooms', 'chain', 'scan', 'table'])
    parser.add_argument('--legs', default=','.join(LEGS))
    parser.add_argument('--steps', default=None)
    parser.add_argument('--from', dest='start', default=None, help='stage: the step to start from (its saved state)')
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--hearts', type=float, default=None)
    parser.add_argument('--best', action='store_true', help='chain: the best play found (BEST) where it differs')
    parser.add_argument('--containers', type=int, default=None, help='rooms: this many heart containers, full')
    parser.add_argument('--variant', default=None, help='rooms: play the steps that have this variant (VARIANTS)')
    args = parser.parse_args()
    args.legs = args.legs.split(',')
    args.steps = args.steps.split(',') if args.steps else None
    {'stage': cmd_stage, 'rooms': cmd_rooms, 'chain': cmd_chain, 'scan': cmd_scan, 'table': cmd_table}[args.cmd](args)


if __name__ == '__main__':
    main()
