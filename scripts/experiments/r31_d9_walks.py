"""Phase IX (docs/routes/phase9-walks.md §5): Dungeon 9's 0x66 out N into 0x56, 0x56's W bomb wall into 0x55 from E,
and the Lanmolas' block clip to the stairs from 0x55's E side.  The trials start from the route's own entry state
`9_66s_<chain>` (what the chains bring to Dungeon 9: 15 hearts, all eight pieces, the magic sword, the ladder, 3 keys,
6 bombs); the pre-entry states of 0x56 from S and 0x55 from E are made once from it by a staging walk (monsters
pinned in their wall, `r29_route_walks.Stager`), and every trial loads one with the game's own randomness for its
seed, the monsters made fresh and live.

    python scripts/experiments/r31_d9_walks.py states            # 9_56s, 9_55e under the walk's temp dir
    python scripts/experiments/r31_d9_walks.py n66 [--seeds 20]  # 0x66 S -> N (the shutter the eight pieces open)
    python scripts/experiments/r31_d9_walks.py w56 [--seeds 20]  # 0x56 S -> the W wall, the bomb, into 0x55
    python scripts/experiments/r31_d9_walks.py scan55            # where CLIP is offered in 0x55 (monsters away)
    python scripts/experiments/r31_d9_walks.py clip55 [--seeds 20]  # from the E arrival: to the clip point, CLIP, stairs

A held Link (a Like-Like) stabs once every 12 frames (`r29_clips.follow`); bombs are written when the bag is empty
(counted).  Hearts are read from the bytes (`precise_hearts`).

It plays the game over many seeded trials: a job of the machine's queue (``jobs.require_job``: ``triforce job run
... --timeout 30m -- .venv/bin/python scripts/experiments/r31_d9_walks.py ...``)."""

import argparse
import collections
import statistics
from pathlib import Path

from triforce.jobs import require_job

import r29_clips
from r26_walks import BOMBS, KEYS, PLAY, STATES
from r27_walks import TMP, StagingWalk
from r29_route_walks import Stager, precise_hearts

STATE_66 = STATES / '9_66s_0.state'


def stager(path, seed=0) -> Stager:
    w = Stager(seed=seed)
    w.game.load(str(path), randomize_rng=True, seed=seed)
    w.staging = True
    w.wait_play()
    return w


def make_states() -> dict:
    out = {'9_56s': TMP / '9_56s.state', '9_55e': TMP / '9_55e.state'}
    if all(p.exists() for p in out.values()):
        return out
    w = stager(STATE_66)
    w.press('NONE', 30)                                   # the shutters open 18 frames after Link's first frame
    assert w.walk((0x78, 0x5D)) and w.transit('UP', save=out['9_56s']), w.where()
    assert w.walk((0x78, 0x8D), (0x20, 0x8D)), w.where()  # over the water row on the ladder
    w.game.write(BOMBS, max(1, w.r(BOMBS)))
    assert w.bomb_wall('LEFT') >= 0, w.where()
    w.press('NONE', 10)
    assert w.transit('LEFT', save=out['9_55e']), w.where()
    print(f'staged: 0x56 from S, its W wall bombed, 0x55 from E: {w.where()}')
    return out


def load(path, seed) -> StagingWalk:
    w = StagingWalk(seed=seed)
    w.game.load(str(path), randomize_rng=True, seed=seed)
    w.game.set_clips(True)
    w.game.set_clips_allowed(True)
    return w


def report(label, rows):
    ok = [r for r in rows if r[0] == 'out']
    lost = [r[1] for r in ok]
    other = collections.Counter(r[0] for r in rows if r[0] != 'out')
    print(f"{label}: {len(ok)} of {len(rows)}; hearts lost mean {statistics.mean(lost) if lost else float('nan'):.2f} "
          f"max {max(lost, default=0):.2f}; frames median {statistics.median(r[2] for r in ok) if ok else '-'}"
          + (f'; else {dict(other)}' if other else '') + (f'; {rows[0][3]}' if len(rows[0]) > 3 else ''))


def crossing(path, points, button, seed, bomb=None, wait=0):
    """Along `points` (a held Link stabs), a bomb facing `bomb` at the last point (written when the bag is empty),
    then `button` held until Link plays in another room.  (what, hearts lost, frames, bombs written)."""
    w = Stager(seed=seed)
    w.game.load(str(path), randomize_rng=True, seed=seed)
    w.wait_play()
    w.press('NONE', wait)
    h0, f0, room = precise_hearts(w), w.game.frame, w.room
    written = 0
    for _attempt in range(4):
        if not r29_clips.follow(w, points):
            if not r29_clips.alive(w):
                return 'died', h0 - precise_hearts(w), w.game.frame - f0, written
            continue
        if bomb:
            if w.r(BOMBS) == 0:
                w.game.write(BOMBS, 1)
                written += 1
            if w.bomb_wall(bomb, limit=120) < 0 and (w.x, w.y) != points[-1]:
                continue
            bomb = None
            if (w.x, w.y) != points[-1]:
                continue
        held = 0
        for _ in range(300):
            if w.room != room or not r29_clips.alive(w):
                break
            held += 1 if w.r(0x512) else 0
            w.press(('A' if held % 12 == 1 else 'NONE') if w.r(0x512) else button, 1)
        if w.room != room and r29_clips.alive(w):
            w.until(lambda v: v.mode == PLAY, 'NONE', 300)
            return 'out', h0 - precise_hearts(w), w.game.frame - f0, written
        if not r29_clips.alive(w):
            return 'died', h0 - precise_hearts(w), w.game.frame - f0, written
    return 'stopped', h0 - precise_hearts(w), w.game.frame - f0, written


def cmd_states(_args):
    for name, path in make_states().items():
        w = load(path, 0)
        w.wait_play()
        print(f'{name}: {path} -> {w.where()} hearts {precise_hearts(w):.2f} bombs {w.r(BOMBS)} keys {w.r(KEYS)} '
              f'monsters {collections.Counter(o[1] for o in w.objects())}')


def cmd_n66(args):
    rows = [crossing(STATES / f'9_66s_{s % 3}.state', [(0x78, 0x5D)], 'UP', s) for s in range(args.seeds)]
    report("9:0x66 S -> N (straight up X $78 to the N shutter)", rows)
    rows = [crossing(STATES / f'9_66s_{s % 3}.state', [(0x78, 0x8D), (0x20, 0x8D)], 'LEFT', s)
            for s in range(args.seeds)]
    report("9:0x66 S -> W (today: up to Y $8D, W)", rows)


def cmd_w56(args):
    path = make_states()['9_56s']
    for label, points in (('up X $78 over the water row, W along Y $8D', [(0x78, 0x8D), (0x20, 0x8D)]),
                          ('up X $28 (W of the middle), up over the water at X $20', [(0x78, 0xBD), (0x20, 0xBD),
                                                                                    (0x20, 0x8D)])):
        rows = [crossing(path, points, 'LEFT', s, bomb='LEFT') for s in range(args.seeds)]
        report(f'9:0x56 S -> W wall ({label}), bomb, into 0x55', rows)
        print(f'    bombs written: {sum(r[3] for r in rows)}')


def cmd_scan55(_args):
    path = make_states()['9_55e']
    w = load(path, 0)
    w.wait_play()
    print(f'0x55 from E: first movable at ({w.x:#04x},{w.y:#04x}), monsters '
          f'{collections.Counter(o[1] for o in w.objects())}')
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
    print(f'9:0x55: CLIP offered on {len(offered)} grid points: {" ".join(offered)}')


WAYS55 = [('E lane, below the ring', [(0xD0, 0x9D), (0xA0, 0x9D)], 'N'),
          ('E lane, above the ring', [(0xD0, 0x7D), (0xA0, 0x7D)], 'S'),
          ('round to the W lane (the S way\'s clip)', [(0xD0, 0xBD), (0x60, 0xBD), (0x60, 0x9D)], 'N'),
          ('under the ring (the S way\'s clip)', [(0xD0, 0xBD), (0x70, 0xBD), (0x70, 0xB5)], 'N')]


def cmd_clip55(args):
    path = make_states()['9_55e']
    r29_clips._STATES['9_55e'] = path                 # r29_clips.clip_trial loads by name
    for label, points, direction in WAYS55:
        rows = [r29_clips.clip_trial('9_55e', False, points, direction, s) for s in range(args.seeds)]
        r29_clips.report(f'9:0x55 {label}: {" ".join(f"(${x:02X},${y:02X})" for x, y in points)} CLIP {direction}',
                         rows, 'cellar')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['states', 'n66', 'w56', 'scan55', 'clip55'])
    p.add_argument('--seeds', type=int, default=20)
    args = p.parse_args()
    require_job()
    {'states': cmd_states, 'n66': cmd_n66, 'w56': cmd_w56, 'scan55': cmd_scan55, 'clip55': cmd_clip55}[args.cmd](args)


if __name__ == '__main__':
    main()
