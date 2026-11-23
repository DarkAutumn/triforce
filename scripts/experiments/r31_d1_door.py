"""Phase IX (docs/routes/phase9-walks.md §1): Dungeon 1's entrance door.  1:0x73's N door is a key door; leaving the
level by 0x73's S door and coming back in draws it open, and Link walks N into 0x63 without a key.  Every trial loads
the route's own entry state `1_73s_<chain>` (Link come down from 0:0x37 with what the chain brings) with the game's
own randomness for its seed; the one Red Octorok of 0:0x37 is live.

    python scripts/experiments/r31_d1_door.py door [--seeds 20]      # the trick and its rule, variant by variant
    python scripts/experiments/r31_d1_door.py frames                 # the frame CurOpenedDoors ($EE) takes bit 8

asm: Door_SetEnteringDoorwayOpened (mode 7's first init step sets $EE to the side opposite Link's facing, also when
the scroll leaves the level), Mode_LeaveLevelToOW, Door_AddRoomFlagsToOpenedDoors, Door_LayOut, Door_TouchKey.

It plays the game over many seeded trials: a job of the machine's queue (``jobs.require_job``: ``triforce job run
... --timeout 30m -- .venv/bin/python scripts/experiments/r31_d1_door.py ...``)."""

import argparse
import collections

from triforce.jobs import require_job

from r26_walks import CUR_OPENED_DOORS, KEYS, PLAY, STATES
from r27_walks import Tally, load

UW_FLAGS_1TO6 = 0x6FF                       # asm:WorldFlags (levels 1-6): room flags, bits 0-3 doors opened
UNDERGROUND_EXIT = 0x5A                     # asm:UndergroundExitType: 2 while Link comes out of a level
N_DOOR = (0x78, 0x5D)                       # the floor before 0x73's N door


def at(w, level, room):
    return w.level == level and w.room == room and w.mode == PLAY


def out_south(t: Tally, limit=400) -> int:
    """From anywhere in 0x73: to X $78, then Down until Link plays in 0:0x37.  Frames from the first Down."""
    w = t.w
    t.walk((0x78, w.y))
    f0 = t.frames
    for _ in range(limit):
        if at(w, 0, 0x37):
            return t.frames - f0
        t.press('DOWN')
    return -1


def in_again(t: Tally, limit=400) -> int:
    """From where Link comes out in 0:0x37 (X $70, Y $7D, on the mouth's foot): one grid step down first (the mouth
    takes him only once asm:UndergroundExitType is 0 again, after a full grid step in mode 5), then Up until he plays
    in 1:0x73."""
    w = t.w
    f0 = t.frames
    if not t.walk((0x70, w.y), limit=200):
        return -1
    for _ in range(40):
        if not w.r(UNDERGROUND_EXIT):
            break
        t.press('DOWN')
    t.leave('UP', limit=limit, line=('x', 0x70))
    return t.frames - f0 if at(w, 1, 0x73) else -1


def north(t: Tally, limit=200) -> tuple:
    """To the N door and Up: (passed into 0x63, frames, keys spent)."""
    w = t.w
    keys = w.r(KEYS)
    f0 = t.frames
    if not t.walk(N_DOOR):
        return False, -1, 0
    for _ in range(limit):
        if at(w, 1, 0x63):
            return True, t.frames - f0, keys - w.r(KEYS)
        t.press('UP')
    return False, t.frames - f0, keys - w.r(KEYS)


VARIANTS = {
    'control': 'no trick: to the N door with 0 keys',
    'trick': 'out S, back in at once, N (0 keys)',
    'trick-keys': 'out S, back in, N with the chain\'s 3 keys (is a key spent?)',
    'wait300': 'out S, 300 frames in 0:0x37, back in, N (0 keys)',
    'twice': 'out S, in, out S, in, N (0 keys)',
    'ow-scroll': 'out S, E into 0:0x38 and back W, in, N (0 keys)',
    'side-room': 'out S, in, W into 0x72 and back E, N (0 keys)',
    'half-out': 'down to Y $DD (the edge) without leaving, back up, N (0 keys)',
    'back-from-63': 'trick, N into 0x63, back S into 0x73, N again (0 keys)',
    'later': 'trick, N, back S into 0x73, out S, in, N (0 keys): a second use',
}


def trial(variant: str, path, seed: int) -> dict:
    w = load(path, seed)
    w.wait_play()
    t = Tally(w)
    row = {'variant': variant, 'seed': seed, 'keys_in': w.r(KEYS)}
    if variant != 'trick-keys':
        w.game.write(KEYS, 0)
    out_f = in_f = None
    if variant in ('trick', 'trick-keys', 'wait300', 'twice', 'ow-scroll', 'side-room', 'back-from-63', 'later'):
        out_f = out_south(t)
        row['ee_outside'] = w.r(CUR_OPENED_DOORS)
        row['spot_ow'] = (w.x, w.y)
        if variant == 'wait300':
            t.press('NONE', 300)
        if variant == 'ow-scroll':
            t.walk((w.x, 0x8D))
            t.leave('RIGHT', line=('y', 0x8D))
            t.leave('LEFT', line=('y', w.y))
            t.walk((0x70, 0x7D))
            row['ee_after_scroll'] = w.r(CUR_OPENED_DOORS)
        in_f = in_again(t)
        if variant == 'twice':
            out_south(t)
            in_again(t)
        if variant == 'side-room':
            t.walk((0x78, 0x8D))
            t.leave('LEFT', line=('y', 0x8D))
            t.leave('RIGHT', line=('y', 0x8D))
    if variant == 'half-out':
        t.walk((0x78, 0xCD))
        t.press('DOWN', 6)
        row['edge_y'] = w.y
        t.walk((0x78, 0xBD))
    row['ee_in'] = w.r(CUR_OPENED_DOORS)
    row['flags_on_entry'] = (w.r(UW_FLAGS_1TO6 + 0x73) & 0x0F, w.r(UW_FLAGS_1TO6 + 0x63) & 0x0F)
    row['out_frames'], row['in_frames'] = out_f, in_f
    ok, frames, spent = north(t)
    if variant in ('back-from-63', 'later') and ok:
        t.leave('DOWN', line=('x', 0x78))
        row['ee_back'] = w.r(CUR_OPENED_DOORS)
        if variant == 'later':
            out_south(t)
            in_again(t)
        ok, frames, spent = north(t)
    row.update(ok=ok, n_frames=frames, spent=spent, flag73=w.r(UW_FLAGS_1TO6 + 0x73) & 0x0F,
               flag63=w.r(UW_FLAGS_1TO6 + 0x63) & 0x0F, lost=t.lost(), by=dict(t.by), frames=t.frames)
    return row


def cmd_door(args):
    for variant, what in VARIANTS.items():
        rows = [trial(variant, STATES / f'1_73s_{s % 3}.state', s) for s in range(args.seeds)]
        oks = sum(r['ok'] for r in rows)
        spent = collections.Counter(r['spent'] for r in rows)
        ee = collections.Counter(r['ee_in'] for r in rows)
        frames = [r['frames'] for r in rows if r['ok']]
        trip = [r['out_frames'] + r['in_frames'] for r in rows if r['out_frames'] and r['in_frames']]
        lost = sum(r['lost'] for r in rows) / len(rows)
        print(f"{variant:13s} {what}\n    N passed {oks}/{len(rows)}, keys spent {dict(spent)}, $EE on entry "
              f"{ {f'{k:#x}': v for k, v in ee.items()} }, flags 0x73/0x63 "
              f"{collections.Counter((r['flag73'], r['flag63']) for r in rows)}, hearts lost {lost:.2f}"
              + (f", out+in {min(trip)}-{max(trip)} frames (mean {sum(trip) / len(trip):.0f})" if trip else '')
              + (f", whole trial {min(frames)}-{max(frames)}" if frames else ''))
        extra = {k: collections.Counter(str(r[k]) for r in rows) for k in ('flags_on_entry', 'ee_outside', 'spot_ow', 'ee_after_scroll',
                                                                         'edge_y', 'ee_back') if k in rows[0]}
        if extra:
            print('   ', {k: dict(v) for k, v in extra.items()})


def cmd_frames(_args):
    """The frame $EE takes bit 8: Link walked down from (`$78`,`$CD`), one frame at a time."""
    w = load(STATES / '1_73s_0.state', 0)
    w.wait_play()
    t = Tally(w)
    t.walk((0x78, 0xCD))
    prev = None
    for i in range(300):
        cur = (w.level, w.room, w.mode, w.r(0x13), w.y, w.r(CUR_OPENED_DOORS))
        if cur[:4] != (prev or (None,) * 6)[:4] or cur[5] != (prev or (None,) * 6)[5]:
            print(f'frame {i:3d}: level {cur[0]} room {cur[1]:#04x} mode {cur[2]:#x}/{cur[3]} y {cur[4]:#04x} '
                  f'$EE {cur[5]:#04x}')
        prev = cur
        if at(w, 0, 0x37):
            break
        t.press('DOWN')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['door', 'frames'])
    p.add_argument('--seeds', type=int, default=20)
    args = p.parse_args()
    require_job()
    {'door': cmd_door, 'frames': cmd_frames}[args.cmd](args)


if __name__ == '__main__':
    main()
