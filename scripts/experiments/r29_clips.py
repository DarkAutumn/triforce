"""Round 29: the block clips of Dungeon 9's ring rooms by the agent's CLIP action (docs/experiments/round29.md §5;
as docs/routes/d7-walks.md §9 did for 7:0x1a).

    python scripts/experiments/r29_clips.py scan              # where each of the eight block rooms offers CLIP
    python scripts/experiments/r29_clips.py walk [--seeds N]  # every clip walked from the route's entry, monsters live
    python scripts/experiments/r29_clips.py returns [--seeds N]  # the return visits a clip leaves full

The rooms are the eight the route empties for a block (9:0x55, 0x07, 0x05, 0x61, 0x20, 0x10, 0x30, 0x03).  Their
states are the route walk's own (`r29_route_walks.route_states`: one game from Dungeon 8's piece, every room's
pre-entry frame, the red ring from cellar 0x00 on); a trial loads one with the game's own randomness for its seed, the
monsters made fresh and live.

`scan`: Link is set on every 8-pixel grid point of the floor with the monsters put away, one frame passes, and
`Game.skip_available` (`zelda::clips::skip_target`, what the mask of a `clips: true` family reads) is asked in the
four directions.

`walk`: from Link's first frame, along the listed points to a grid point where CLIP is offered (back onto the line
after a knockback; held by a Like-Like he stabs), `CLIP <d>` by the engine's own macro, then MOVE toward the stairs
square.  A trial succeeds when Link stands in the cellar.  Hearts are read from the bytes (the ring's quarter hearts).

`returns`: what a clip changes later.  A room that was clipped, not emptied, is full when the route comes back:
0x07 out of the ring cellar to the S wall, 0x61 out of the tunnel to the N door, and 0x20, whose first visit (out of
the tunnel, the bomb at the N wall) and second (from N to the stairs) both meet its five Wizzrobes.  The same
crossings with the room as the fight leaves it (the route as built) are walked beside them.  It plays the game over
many seeded trials: a job of the machine's queue (``jobs.require_job``: ``triforce job run ... --timeout 30m --
.venv/bin/python scripts/experiments/r29_clips.py ...``)."""

import argparse
import collections
import sys
import tempfile
from pathlib import Path

from triforce.jobs import require_job

from r26_walks import BOMBS, PLAY
from r27_walks import StagingWalk
from r29_route_walks import Stager, precise_hearts, route_states

CELLAR = 0x09
STAIRS = (0x80, 0x8D)                                          # the stairs square inside the ring of blocks
ROOMS = [('0x55', '9_55s'), ('0x07', '9_07s'), ('0x05', '9_05e'), ('0x61', '9_61e'), ('0x20', '9_20n'),
         ('0x10', '9_10s'), ('0x30', '9_30e'), ('0x03', '9_03e')]
# room -> (state, fresh room, [(name of the way, points to the clip's grid point, CLIP direction)])
CLIPS = {
    '0x55': ('9_55s', False, [('W lane', [(0x60, 0xBD), (0x60, 0x9D)], 'N'), ('under the ring', [(0x70, 0xBD), (0x70, 0xB5)], 'N')]),
    '0x07': ('9_07s', False, [('W lane', [(0x60, 0xBD), (0x60, 0x9D)], 'N'), ('under the ring', [(0x70, 0xBD), (0x70, 0xB5)], 'N')]),
    '0x61': ('9_61e', False, [('E lane, below', [(0xD0, 0x8D), (0xD0, 0x9D), (0xA0, 0x9D)], 'N'),
                              ('E lane, above', [(0xD0, 0x8D), (0xD0, 0x7D), (0xA0, 0x7D)], 'S')]),
    '0x20': ('9_20n', True, [('from the door', [(0x70, 0x5D)], 'S'), ('W lane', [(0x60, 0x5D), (0x60, 0x7D)], 'S')]),
    '0x03': ('9_03e', False, [('E lane, below', [(0xD0, 0x8D), (0xD0, 0x9D), (0xA0, 0x9D)], 'N'),
                              ('E lane, above', [(0xD0, 0x8D), (0xD0, 0x7D), (0xA0, 0x7D)], 'S')]),
}

_STATES = {}


def states() -> dict:
    if not _STATES:
        paths, _rows, _cheats = route_states(Path(tempfile.mkdtemp(prefix='r29-clips-')))
        _STATES.update(paths)
    return _STATES


def load(name: str, seed: int, fresh: bool = False) -> StagingWalk:
    w = StagingWalk(seed=seed)
    w.game.load(str(states()[name]), randomize_rng=True, fresh_room=fresh, seed=seed)
    w.game.set_clips(True)
    w.game.set_clips_allowed(True)
    return w


def scan(_args):
    for room, name in ROOMS:
        w = load(name, 0)
        snap, offered = w.game.snapshot(), []
        w.staging = True
        for y in range(0x5D, 0xBE, 8):
            for x in range(0x20, 0xD1, 8):
                w.game.restore(snap)
                w.game.apply_ram({'link_x': x, 'link_y': y})
                w.press('NONE', 1)
                if (w.x, w.y) == (x, y):
                    dirs = ''.join(d for d in 'NSWE' if w.game.skip_available(d))
                    if dirs:
                        offered.append(f'(${x:02X},${y:02X}){dirs}')
        print(f'9:{room} ({name}): CLIP offered on {len(offered)} grid points{": " + " ".join(offered) if offered else ""}')


def alive(w) -> bool:
    return w.mode not in (0x11, 0x08) and precise_hearts(w) > 0.01


def follow(w, points, limit: int = 900) -> bool:
    """Along axis-parallel legs to the last point; back onto the leg's line after a knockback; stabs while held."""
    here, held = (w.x, w.y), 0
    room = (w.level, w.room, w.mode)
    for p in points:
        along_x = p[1] == here[1]
        for _ in range(limit):
            if (w.level, w.room, w.mode) != room or not alive(w):
                return False
            x, y = w.x, w.y
            if (x, y) == p:
                break
            if w.r(0x512):                                       # asm:IsLinkParalyzed: a Like-Like
                held += 1
                w.press('A' if held % 12 == 1 else 'NONE', 1)
                continue
            if along_x:
                button = ('DOWN' if p[1] > y else 'UP') if y != p[1] else ('RIGHT' if p[0] > x else 'LEFT')
            else:
                button = ('RIGHT' if p[0] > x else 'LEFT') if x != p[0] else ('DOWN' if p[1] > y else 'UP')
            w.press(button, 1)
        else:
            return False
        here = p
    return True


def clip_trial(name: str, fresh: bool, points, direction: str, seed: int):
    """(what happened, hearts lost, frames)."""
    w = load(name, seed, fresh)
    w.press('NONE', 1)
    h0, f0 = precise_hearts(w), w.game.frame
    what = None
    for _attempt in range(3):
        if not follow(w, points):
            what = 'died' if not alive(w) else 'left the room or stopped on the way'
            break
        waited = 0
        while (w.x, w.y) == points[-1] and not w.game.skip_available(direction) and waited < 30:
            w.press('NONE', 1)
            waited += 1
        if (w.x, w.y) != points[-1]:
            continue                                             # knocked off the point: walk back
        if not w.game.skip_available(direction):
            what = 'CLIP not offered'
            break
        w.game.act(f'CLIP {direction}')
        for _ in range(8):
            if w.mode != PLAY or not alive(w):
                break
            # Onto the stairs' row first (a clip from above ends 8 px over it), then along it.
            dx, dy = STAIRS[0] - w.x, STAIRS[1] - w.y
            before = (w.x, w.y)
            if dy:
                w.game.act('MOVE S' if dy > 0 else 'MOVE N')
            if (w.x, w.y) == before and w.mode == PLAY:
                w.game.act('MOVE E' if dx > 0 else 'MOVE W')
        if w.mode != PLAY and alive(w):
            w.until(lambda w: w.mode == CELLAR, 'NONE', 300)
            what = 'cellar' if w.mode == CELLAR else 'left the room'
            break
        if not alive(w):
            what = 'died'
            break
        if abs(w.x - STAIRS[0]) <= 0x10 and abs(w.y - STAIRS[1]) <= 0x10:
            what = 'inside the ring, no stairs'
            break
    else:
        what = what or 'knocked off three times'
    return what, h0 - precise_hearts(w), w.game.frame - f0


def report(label, rows, good):
    wins = [r for r in rows if r[0] == good]
    lost = [r[1] for r in wins]
    frames = sorted(r[2] for r in wins)
    others = collections.Counter(r[0] for r in rows if r[0] != good)
    mean = sum(lost) / len(lost) if lost else float('nan')
    print(f'{label}: {len(wins)} of {len(rows)}; hearts lost in those {mean:.2f} (max {max(lost, default=0):.2f}); '
          f'frames median {frames[len(frames) // 2] if frames else "-"}'
          + (f'; else {dict(others)}' if others else ''))


def walk(args):
    for room, (name, fresh, ways) in CLIPS.items():
        for label, points, direction in ways:
            rows = [clip_trial(name, fresh, points, direction, seed) for seed in range(args.seeds)]
            ring = load(name, 0).r(0x662)
            report(f'9:{room} {label}: {" ".join(f"(${x:02X},${y:02X})" for x, y in points)} CLIP {direction} '
                   f'({"red ring" if ring == 2 else "no ring"}{", the room full" if fresh else ""})', rows, 'cellar')


def cross_trial(name: str, fresh: bool, points, button: str, seed: int, bomb: str = None):
    """A crossing to the room's exit: along `points`, a bomb at the wall facing `bomb` if given, then `button` held
    until Link is in another room.  (what happened, hearts lost, frames)."""
    w = Stager(seed=seed)
    w.game.load(str(states()[name]), randomize_rng=True, fresh_room=fresh, seed=seed)
    w.press('NONE', 1)
    h0, f0, room = precise_hearts(w), w.game.frame, w.room
    what = 'stopped on the way'
    for _attempt in range(4):
        if follow(w, points):
            if bomb:
                if w.r(BOMBS) == 0:
                    w.game.write(BOMBS, 1)
                w.bomb_wall(bomb, limit=120)
                bomb = None
                if (w.x, w.y) != points[-1]:
                    continue
            held = 0
            for _ in range(200):
                if w.room != room or not alive(w):
                    break
                held += 1 if w.r(0x512) else 0
                w.press(('A' if held % 12 == 1 else 'NONE') if w.r(0x512) else button, 1)
            if w.room != room and alive(w):
                what = 'out'
                break
        if not alive(w):
            what = 'died'
            break
    return what, h0 - precise_hearts(w), w.game.frame - f0


def returns(args):
    cases = [
        ('9:0x07 out of the ring cellar to the S wall (red ring)', '9_07c', [(0x60, 0xBD), (0x78, 0xBD)], 'DOWN', None),
        ('9:0x61 out of the tunnel to the N door (red ring)', '9_61c', [(0x50, 0x9D), (0x50, 0x5D), (0x78, 0x5D)], 'UP', None),
        ('9:0x20 out of the tunnel, the bomb at the N wall (red ring)', '9_20c', [(0x50, 0x9D), (0x50, 0x5D), (0x78, 0x5D)], 'UP', 'UP'),
    ]
    for label, name, points, button, bomb in cases:
        for fresh in (True, False):
            if name == '9_20c' and not fresh:
                continue                                         # the first visit is the fight's own
            rows = [cross_trial(name, fresh, points, button, seed, bomb) for seed in range(args.seeds)]
            report(f'{label}, {"full (after a clip)" if fresh else "as the fight leaves it"}', rows, 'out')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['scan', 'walk', 'returns'])
    parser.add_argument('--seeds', type=int, default=20)
    args = parser.parse_args()
    require_job()
    {'scan': scan, 'walk': walk, 'returns': returns}[args.what](args)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
