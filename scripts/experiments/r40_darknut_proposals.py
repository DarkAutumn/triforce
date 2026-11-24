"""Round 40 (docs/experiments/round40.md, "The owner's Darknut fighter", the proposals): what the two switched-off
proposals would see in today's darknut network's play, read from its replays frame by frame (a calculation over
replays, nothing played).

    python scripts/experiments/r40_darknut_proposals.py REPLAY_DIR... [--json OUT]

(a) `reward.chase_penalty` (`zelda::darknut_rules::chasing`, read again here as `chasing`): the frames Link stands
    within 8 pixels of the line of a Darknut walking away from him, less than $30 behind it; how often such a frame is
    followed by a hit within `SOON` frames against the other frames with a Darknut in the room; the share of hits
    with such a frame in the `SOON` frames before; the cost a visit would pay at `RATE` a frame.
(b) `reward.bomb_catch_min` / `items.bomb_catch` (`zelda::darknut_rules::bomb_catch`, read again here as `catch`):
    every bomb laid, the Darknuts its blast hurt (HP down or dead while it blasts), against the count `catch` gave
    on the frame it was laid: the bombs a minimum of 2 would leave unpaid, and how well the mask's count tells the
    bombs that hurt two or more.

It reads many replays: a job of the machine's queue (``jobs.require_job``: ``triforce job run ... --express
--timeout 10m -- .venv/bin/python scripts/experiments/r40_darknut_proposals.py ...``).
"""

import argparse
import collections
import json
import sys
from pathlib import Path

OBJ_X, OBJ_Y, OBJ_DIR, OBJ_TYPE, OBJ_META, OBJ_HP = 0x70, 0x84, 0x98, 0x34F, 0x405, 0x485
OBJ_Q_SPEED, OBJ_INVINC, OBJ_MASK = 0x3BC, 0x4F0, 0x4B2
BOMB_STATE, INV_CLOCK, HEARTS = (0xBC, 0xBD), 0x66C, (0x66F, 0x670)
DARKNUTS = (0x0B, 0x0C)
VEC = {1: (1, 0), 2: (-1, 0), 4: (0, 1), 8: (0, -1)}
OPPOSITE = {1: 2, 2: 1, 4: 8, 8: 4}
CHASE_REACH = 0x30
SOON = 32                    # frames after a chasing frame within which a hit counts as following it
RATE = 0.002                 # the cost a frame the measurement prices a visit at (the proposal's suggested value)


def monsters(ram):
    """(slot, type, x, y, facing bits) of the room's monsters out of their spawn cloud and not dying."""
    return [(s, ram[OBJ_TYPE + s], ram[OBJ_X + s], ram[OBJ_Y + s], ram[OBJ_DIR + s] & 0x0F) for s in range(1, 12)
            if 0 < ram[OBJ_TYPE + s] < 0x60 and ram[OBJ_META + s] == 0]


def chasing(ram) -> bool:
    """`zelda::darknut_rules::chasing`."""
    x, y = ram[OBJ_X], ram[OBJ_Y]
    for _, kind, dx, dy, facing in monsters(ram):
        if kind not in DARKNUTS or facing not in VEC:
            continue
        vx, vy = VEC[facing]
        along = (x - dx) * vx + (y - dy) * vy
        across = abs(y - dy) if vx else abs(x - dx)
        if across < 9 and -CHASE_REACH < along < 0:
            return True
    return False


def catch(ram, way: int, bomb) -> int:
    """`zelda::darknut_rules::bomb_catch` for a bomb at `bomb` laid facing `way` (direction bits), on `ram`."""
    still = ram[INV_CLOCK] != 0
    count = 0
    for slot, kind, x, y, facing in monsters(ram):
        if ram[OBJ_MASK + slot] & 0x08 or (kind in DARKNUTS and facing == OPPOSITE[way]):
            continue
        vx, vy = VEC.get(facing, (0, 0))
        speed = 0 if still else ram[OBJ_Q_SPEED + slot]
        flash = 2 * ram[OBJ_INVINC + slot]
        for f in range(max(0x31, flash), 0x31 + 0x18):
            moved = speed * f // 64
            if abs(x + vx * moved - bomb[0]) < 0x18 and abs(y + vy * moved - bomb[1]) < 0x18:
                count += 1
                break
    return count


def health(ram) -> float:
    full = ram[HEARTS[0]] & 0x0F
    partial = ram[HEARTS[1]]
    return full + (1.0 if partial >= 0x80 else 0.5 if partial else 0.0)


def read(path: Path) -> dict:
    """One replay's frames: chasing frames and their hits, the bombs and what they hurt."""
    from triforce import _native
    player = _native.ReplayPlayer(path.read_bytes())
    rams = []
    for _ in range(player.frames):
        player.step(1, False)
        rams.append(bytes(player.ram()))
    hits = [f for f in range(1, len(rams)) if health(rams[f]) < health(rams[f - 1])]
    chase = [f for f, ram in enumerate(rams) if chasing(ram)]
    fighting = [f for f, ram in enumerate(rams) if any(m[1] in DARKNUTS for m in monsters(ram))]
    bombs = []
    for f in range(1, len(rams)):
        for slot in (0x10, 0x11):
            if rams[f][0xAC + slot] in (0x11, 0x12) and rams[f - 1][0xAC + slot] == 0:
                way = rams[f][OBJ_DIR] & 0x0F
                at = (rams[f][OBJ_X + slot], rams[f][OBJ_Y + slot])
                before = rams[f - 1]
                predicted = catch(before, way, at) if way in VEC else 0
                hurt = set()
                for g in range(f + 1, min(len(rams), f + 0x31 + 0x18 + 2)):
                    if rams[g][0xAC + slot] != 0x13 and rams[g - 1][0xAC + slot] != 0x13:
                        continue
                    for s in range(1, 12):
                        if rams[g - 1][OBJ_TYPE + s] in DARKNUTS and rams[g - 1][OBJ_META + s] < 0x10:
                            gone = rams[g][OBJ_META + s] >= 0x10 or rams[g][OBJ_TYPE + s] not in DARKNUTS
                            if gone or rams[g][OBJ_HP + s] < rams[g - 1][OBJ_HP + s]:
                                hurt.add(s)
                bombs.append({'frame': f, 'predicted': predicted, 'hurt': len(hurt)})
    return {'replay': path.name, 'frames': len(rams), 'hits': hits, 'chase': chase, 'fighting': len(fighting),
            'bombs': bombs}


def summarize(visits: list[dict]) -> dict:
    hit_after = lambda frames, hits: sum(any(f < h <= f + SOON for h in hits) for f in frames)  # noqa: E731
    chase_frames = sum(len(v['chase']) for v in visits)
    fight_frames = sum(v['fighting'] for v in visits)
    chase_hit = sum(hit_after(v['chase'], v['hits']) for v in visits)
    chase_set = [set(v['chase']) for v in visits]
    other_hit = 0
    for v, chased in zip(visits, chase_set):
        hits = v['hits']
        other_hit += sum(1 for f in range(v['frames']) if f not in chased and any(f < h <= f + SOON for h in hits))
    hits = sum(len(v['hits']) for v in visits)
    hits_after_chase = sum(sum(any(h - SOON <= f < h for f in v['chase']) for h in v['hits']) for v in visits)
    bombs = [b for v in visits for b in v['bombs']]
    by = collections.Counter(min(b['hurt'], 2) for b in bombs)
    told = collections.Counter((b['predicted'] >= 2, b['hurt'] >= 2) for b in bombs)
    return {
        'visits': len(visits), 'frames': sum(v['frames'] for v in visits), 'darknut_frames': fight_frames,
        'chase': {
            'frames': chase_frames, 'share_of_darknut_frames': chase_frames / max(1, fight_frames),
            'hit_within_soon_after_chase_frame': chase_hit / max(1, chase_frames),
            'hit_within_soon_after_other_frame': other_hit / max(1, sum(v['frames'] for v in visits) - chase_frames),
            'hits': hits, 'hits_with_chase_before': hits_after_chase,
            'cost_per_visit_at_rate': RATE * chase_frames / max(1, len(visits)), 'rate': RATE, 'soon': SOON},
        'bombs': {
            'laid': len(bombs), 'hurt_0': by[0], 'hurt_1': by[1], 'hurt_2plus': by[2],
            'darknuts_hurt_per_bomb': sum(b['hurt'] for b in bombs) / max(1, len(bombs)),
            'unpaid_at_min_2': by[0] + by[1],
            'catch_ge2_hurt_ge2': told[(True, True)], 'catch_ge2_hurt_lt2': told[(True, False)],
            'catch_lt2_hurt_ge2': told[(False, True)], 'catch_lt2_hurt_lt2': told[(False, False)]},
    }


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dirs', nargs='+')
    parser.add_argument('--json', default=None)
    args = parser.parse_args(argv)
    from triforce import jobs
    jobs.require_job()
    paths = sorted(p for d in args.dirs for p in Path(d).glob('*.tfreplay'))
    visits = []
    for i, path in enumerate(paths):
        visits.append(read(path))
        jobs.progress(i + 1, len(paths), 'replays')
    result = {'dirs': args.dirs, **summarize(visits)}
    print(json.dumps(result, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
