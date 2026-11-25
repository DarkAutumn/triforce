"""Does a solved room's way hold in every random phase?  Scripted inputs over thousands of seeds, in the route's own
layout (the owner, Nov 25, on 5x37: "I'm not sure if holding pure north is possible with every possible rng").

    triforce job run --agent A --task T --subtask S --desc '...' --express --timeout 10m -- \\
        .venv/bin/python scripts/solved_phases.py ROOM [--seeds N] [--first-seed S] [--workers W] \\
        [--out DIR] [--keep K] [--json FILE]

ROOM is a solved room's key (``5x47-S``, ``5x37-S``, ``3x69-N``, ``3x69-cave``); every script of ``SCRIPTS[ROOM]``
plays the room from the route's start states (``d3first_<level>_<room><side>_<chain>``, chains in turn by seed),
loaded by ``_native.LiveGame.load(state, seed, layout='route')``: the round-43 load rule with the route's planned
``$524`` for the door, the chase flag drawn 50/50 and the natural random phase from the seed (needs a checkout with
that rule: ``r40-int`` and after it).  A script reads the game each frame and presses buttons (no look-ahead, no
state saved and restored): it is a player who sees the screen.

Each run is measured the way ``triforce solved passes`` measures a pass (``triforce/solved_rooms.py``; the same
definitions, inlined so the script runs on a checkout without that module): from the first frame in play in the
room to the first frame of leaving it; a hit is a drop of Link's hearts; the exit is the side of a scroll, ``cave``
for the stairs, ``death``.  Clean = left by the solved exit with no hit.  Printed: per script, clean runs of all and
by the chase flag drawn (``$60`` at the load, its frame-0 edit), frames of the clean runs (min, mean, max), swings.
``--out DIR`` keeps the first ``--keep`` runs of each script and every unclean one up to ``--keep`` as TFREPLAY
files (``<room>_<script>_seed<S>_<clean|hit|...>.tfreplay``) for ``triforce solved passes`` and ``play-zelda.py``.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from triforce import _native, jobs

UP, DOWN, LEFT, RIGHT, A = 0x10, 0x20, 0x40, 0x80, 0x01
MODE, SPAWN_CYCLE, CHASE_FLAG = 0x12, 0x524, 0x60
PLAY_MODE, DEATH_MODE = 5, 0x11
LIMIT = 900                 # frames a run may take before it counts as stuck
DOOR_ROW = 189              # Link's Y in the S doorway's top tile (the doorway ends here)
TOP_ROW = 93                # Link's Y on a dungeon room's top floor row ($5D)


OPPOSITE = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
FACING = {UP: 'N', DOWN: 'S', LEFT: 'W', RIGHT: 'E'}
REACH = (6, 24, 10)         # the wood sword's blade box: 6-24 pixels ahead, 10 aside (round40.md, measured)


def in_reach(link: dict, enemies: list, facing: str) -> bool:
    """A live monster in the blade's box ahead of Link, not facing the blade (a Darknut's shield refuses a hit
    from its front, asm:Damage_HandleWeaponHit@CheckDarknut)."""
    dx, dy = {'N': (0, -1), 'S': (0, 1), 'W': (-1, 0), 'E': (1, 0)}[facing]
    lo, hi, aside = REACH
    for e in enemies:
        if not e.get('active') or e.get('dying') or e.get('direction') == OPPOSITE[facing]:
            continue
        ex, ey = e['x'] - link['x'], e['y'] - link['y']
        if lo <= ex * dx + ey * dy <= hi and abs(ex * dy) + abs(ey * dx) <= aside:
            return True
    return False


class Legs:
    """A path as legs: (button, done(link) -> bool); the last leg is held until Link leaves.  ``swing_at``: Link's
    Y at or above which each swing in turn is made (one swing a value, e.g. ``(0x91, 0x71)``); else ``reach``: a
    swing whenever a monster is in the blade's reach (``in_reach``), at most ``max_swings``.  A swing is A pressed
    with the leg's direction, never while Link is still mid-swing."""

    def __init__(self, legs, reach: bool = False, max_swings: int = 2, swing_at: tuple = ()):
        self.legs, self.leg, self.swing_at, self.reach = legs, 0, tuple(swing_at), reach
        self.max_swings = len(self.swing_at) if self.swing_at else max_swings if reach else 0
        self.swings, self.last_a = 0, False

    def __call__(self, state) -> int:
        link = state.link
        while self.leg < len(self.legs) - 1 and self.legs[self.leg][1](link):
            self.leg += 1
        button = self.legs[self.leg][0]
        if self.swing_at:
            wanted = self.swings < len(self.swing_at) and link['y'] <= self.swing_at[self.swings]
        else:
            wanted = self.reach and in_reach(link, state.enemies, FACING[button])
        press = wanted and not self.last_a and self.swings < self.max_swings and not link['status'] & 0x30
        self.last_a = press
        if press:
            self.swings += 1
            return button | A
        return button


def owner_37(**swing) -> Legs:
    """5x37: N out of the doorway, W half a tile to X $70, N to the wall (the top row), E to X $78, N out."""
    return Legs([(UP, lambda l: l['y'] <= DOOR_ROW), (LEFT, lambda l: l['x'] <= 0x70),
                 (UP, lambda l: l['y'] <= TOP_ROW), (RIGHT, lambda l: l['x'] >= 0x78), (UP, None)], **swing)


def line_69_in(**swing) -> Legs:
    """3x69 in: S to the top row, E to the block wall (X $90), S to the stairs' row (Y $8D), E onto the stairs."""
    return Legs([(DOWN, lambda l: l['y'] >= TOP_ROW), (RIGHT, lambda l: l['x'] >= 0x90),
                 (DOWN, lambda l: l['y'] >= 0x8D), (RIGHT, None)], **swing)


SCRIPTS = {
    # 5x47 from S: hold north from the door to the N door, X $78 the whole way: alone; with a swing whenever a
    # Gibdo is in the blade's reach; with one swing as Link reaches the second row from the top (Y $71); with two,
    # on the ladder over the middle water (Y $91) and at the second row (swing timings swept, 300-600 seeds each).
    '5x47-S': {
        'north': lambda: Legs([(UP, None)]),
        'north+swing-in-reach': lambda: Legs([(UP, None)], reach=True, max_swings=2),
        'north+swing@71': lambda: Legs([(UP, None)], swing_at=(0x71,)),
        'north+swings@91,71': lambda: Legs([(UP, None)], swing_at=(0x91, 0x71)),
    },
    # 5x37 from S: the owner's way, alone, with one swing as Link steps onto the ladder over the water (Y $A5), with
    # a swing whenever a Darknut is in the blade's reach; pure north at X $78 the same three ways.
    '5x37-S': {
        'owner': lambda: owner_37(),
        'owner+swing@a5': lambda: owner_37(swing_at=(0xA5,)),
        'owner+swing-in-reach': lambda: owner_37(reach=True, max_swings=2),
        'north': lambda: Legs([(UP, None)]),
        'north+swing@a5': lambda: Legs([(UP, None)], swing_at=(0xA5,)),
        'north+swing-in-reach': lambda: Legs([(UP, None)], reach=True, max_swings=2),
    },
    # 3x69 from N (the way in, to the stairs; round40.md "3:0x69 crossed by a line", the owner, Nov 23): the line
    # alone and with a swing whenever a Darknut is in the blade's reach (its side or back to the blade).
    '3x69-N': {
        'line': lambda: line_69_in(),
        'line+swing-in-reach': lambda: line_69_in(reach=True, max_swings=4),
    },
    # 3x69 out of the cellar, from the cellar's exit spot ($60, $8D) (the owner, Nov 25: "Leaving is always safe if
    # you head right out"): N to the top row, E to the door's X $78, N out; or E to X $78 first, then N.
    '3x69-cave': {
        'north-first': lambda: Legs([(UP, lambda l: l['y'] <= TOP_ROW), (RIGHT, lambda l: l['x'] >= 0x78),
                                     (UP, None)]),
        'east-first': lambda: Legs([(RIGHT, lambda l: l['x'] >= 0x78), (UP, None)]),
    },
}
EXITS = {'5x47-S': 'N', '5x37-S': 'N', '3x69-N': 'cave', '3x69-cave': 'N'}
STATES = {'5x47-S': 'd3first_5_47s', '5x37-S': 'd3first_5_37s', '3x69-N': 'd3first_3_69n', '3x69-cave': 'd3first_3_69c'}


def side_left(x: int, y: int):
    if y <= 0x45:
        return 'N'
    if y >= 0xD0:
        return 'S'
    if x <= 0x10:
        return 'W'
    if x >= 0xE0:
        return 'E'
    return None


def play(room: str, script: str, seed: int, chains: int = 3):
    """One run: the measurement (a dict) and the take's bytes."""
    game = _native.LiveGame('d3first')
    state = game.load(f"{STATES[room]}_{seed % chains}", seed, layout='route')
    policy = SCRIPTS[room][script]()
    frames = hits = swings = 0
    start, health, exit_, last_a, xy = None, None, None, False, None
    while frames < LIMIT:
        mode = state.ram[MODE]
        if mode == PLAY_MODE:
            if start is None:
                start, health = frames, state.health
            if state.health < health:
                hits += 1
            health = state.health
            xy = (state.link['x'], state.link['y'])
            buttons = policy(state)
            if buttons & A and not last_a:
                swings += 1
            last_a = bool(buttons & A)
        elif start is not None and mode in (0x06, 0x07, 0x10, DEATH_MODE):
            exit_ = 'death' if mode == DEATH_MODE else 'cave' if mode == 0x10 else side_left(*xy)
            break
        else:
            buttons = 0
        state = game.step(buttons)
        frames += 1
    data = bytes(game.replay(json.dumps({'kind': 'human', 'scripted': {'room': room, 'script': script}}),
                             'other' if exit_ not in (None, 'death') else 'death' if exit_ == 'death' else 'unknown'))
    edits = {int(a): int(n) for f, a, _o, n in _native.Replay(data).edits() if int(f) == 0}
    clean = exit_ == EXITS[room] and hits == 0
    return {'seed': seed, 'script': script, 'exit': exit_, 'hits': hits, 'swings': swings,
            'frames': (frames - start) if start is not None else None, 'clean': clean,
            'layout': edits.get(SPAWN_CYCLE), 'chase': edits.get(CHASE_FLAG)}, data


def _work(args):
    room, script, seeds = args
    return [play(room, script, seed) for seed in seeds]


def summarise(results: list) -> dict:
    clean = [r for r in results if r['clean']]
    frames = [r['frames'] for r in clean]
    out = {'runs': len(results), 'clean': len(clean),
           'frames': [min(frames), round(sum(frames) / len(frames), 1), max(frames)] if frames else None,
           'swings_a_run': round(sum(r['swings'] for r in results) / max(1, len(results)), 3),
           'exits': {}, 'layouts': sorted({r['layout'] for r in results if r['layout'] is not None})}
    for r in results:
        out['exits'][str(r['exit'])] = out['exits'].get(str(r['exit']), 0) + 1
    for flag in (0, 1):
        mine = [r for r in results if r['chase'] == flag]
        out[f'chase{flag}'] = [sum(r['clean'] for r in mine), len(mine)]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('room', choices=sorted(SCRIPTS))
    parser.add_argument('--seeds', type=int, default=3000)
    parser.add_argument('--first-seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', help="keep takes here")
    parser.add_argument('--keep', type=int, default=10, help="takes kept a script: the first, and the unclean ones")
    parser.add_argument('--json', help="write the results here")
    args = parser.parse_args(argv)
    jobs.require_job()
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    chunks = [seeds[i:i + 100] for i in range(0, len(seeds), 100)]
    report = {'room': args.room, 'seeds': [seeds[0], seeds[-1]], 'scripts': {}}
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    by_seed = {}
    for script in SCRIPTS[args.room]:
        results, kept, unclean_kept = [], 0, 0
        with ProcessPoolExecutor(args.workers) as pool:
            for done, chunk in enumerate(pool.map(_work, [(args.room, script, c) for c in chunks]), 1):
                for result, data in chunk:
                    results.append(result)
                    keep = kept < args.keep or (not result['clean'] and unclean_kept < args.keep)
                    if out and keep:
                        what = 'clean' if result['clean'] else f"{result['exit']}-hits{result['hits']}"
                        (out / f"{args.room}_{script}_seed{result['seed']}_{what}.tfreplay").write_bytes(data)
                        kept += 1
                        unclean_kept += not result['clean']
                jobs.progress(done, len(chunks), 'chunks', label=script)
        for r in results:
            by_seed.setdefault(r['seed'], {})[script] = r['clean']
        report['scripts'][script] = summarise(results)
        print(script, json.dumps(report['scripts'][script]), flush=True)
    names = list(SCRIPTS[args.room])
    if len(names) > 1:     # what the second script saves where the first is hit, and the other way round
        pairs = {}
        for a in names:
            for b in names:
                if a != b:
                    pairs[f"{b} where {a} fails"] = [sum(1 for s in by_seed.values() if not s[a] and s[b]),
                                                     sum(1 for s in by_seed.values() if not s[a])]
        report['pairs'] = pairs
        print('pairs', json.dumps(pairs))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
