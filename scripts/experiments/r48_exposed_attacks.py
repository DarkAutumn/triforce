"""Round 40 ("The Goriya teacher"): how often `reward.exposed_attack_penalty` would fire, measured on recorded play
before it is switched on (the owner, Nov 26: "penalize attacking near projectiles if there's no enemies near the
agent").

    triforce job run --agent A --task T --subtask exposed --express --timeout 20m -- \\
        .venv/bin/python scripts/experiments/r48_exposed_attacks.py ROOT --out target/r48/exposed.json

ROOT holds `<team>-<leg>/*.tfreplay` (the overworld recordings of `target/ow-shots/`: teams `new` and `old`, 50 runs a
leg, every decision recorded). Every decision is read at its frame (`_native.ReplayPlayer`) and judged by the rule
itself (`_native.exposed_probe`, `zelda::exposed`): a harmful shot near for the attack's lockout, a monster near
(walking at Link into a swing), a ranged attack aimed at one, and the penalty's condition. Counted on the overworld
(level 0, game mode 5), by team, by the network that decided (the META's family), and on the subset round 40 looked
at by hand: the new team's SWORD, BEAMS and BOOMERANG with a Zora fireball within 64 px and no monster within 48 px
(the middles' larger axis gap; `/tmp/ow_shots2.py`: 93 of 97 pointed at neither the Zora nor the fireball), split by
whether the attack's lane held the Zora or the fireball.
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from triforce import _native, jobs

ATTACKS = ('SWORD', 'BEAMS', 'BOMB', 'BOOMERANG', 'ARROW')
HAND = ('SWORD', 'BEAMS', 'BOOMERANG')
OBJ_TYPE, OBJ_X, OBJ_Y, OBJ_METASTATE = 0x34F, 0x70, 0x84, 0x405
CUR_LEVEL, GAME_MODE = 0x10, 0x12
FIREBALL, ZORA = 0x55, 0x11
UNIT = {'N': (0, -1), 'S': (0, 1), 'W': (-1, 0), 'E': (1, 0)}


def in_lane(lx, ly, x, y, d):
    """`/tmp/ow_shots2.py`'s lane: within 8 px across, any distance ahead (cardinal attacks only)."""
    if d not in UNIT:
        return False
    dx, dy = UNIT[d]
    if dx:
        return abs(y - ly) <= 8 and (x - lx) * dx > 0
    return abs(x - lx) <= 8 and (y - ly) * dy > 0


def hand_case(ram, direction):
    """The hand-measured subset: (zora in the lane, fireball in the lane) or None outside it."""
    lx, ly = int(ram[OBJ_X]) + 8, int(ram[OBJ_Y]) + 8
    fire, zoras, near = [], [], 999
    for slot in range(1, 0x0C):
        t = int(ram[OBJ_TYPE + slot])
        if t == 0:
            continue
        x, y = int(ram[OBJ_X + slot]) + 8, int(ram[OBJ_Y + slot]) + 8
        d = max(abs(x - lx), abs(y - ly))
        if t == FIREBALL:
            fire.append((x, y, d))
        elif 0x01 <= t <= 0x4F and int(ram[OBJ_METASTATE + slot]) == 0:
            near = min(near, d)
            if t == ZORA:
                zoras.append((x, y))
    if near <= 48 or not any(d <= 64 for _, _, d in fire):
        return None
    return (any(in_lane(lx, ly, x, y, direction) for x, y in zoras),
            any(in_lane(lx, ly, x, y, direction) for x, y, d in fire if d <= 64))


def scan(path, stats, hand):
    data = open(path, 'rb').read()
    replay = _native.Replay(data)
    meta = replay.meta
    kinds, directions = meta['actions']['kinds'], meta['actions']['directions']
    names = {m['index']: m.get('family') or m['name'] for m in meta['models']}
    dec = replay.decisions()
    player = _native.ReplayPlayer(data)
    rows, labels = [], []
    for frame, action, policy in zip(dec['frame'], dec['action'], dec['policy']):
        frame = int(frame)
        if frame > player.frame:
            player.step(frame - player.frame, False)
        ram = np.asarray(player.ram(), dtype=np.uint8)
        if ram[GAME_MODE] != 5 or ram[CUR_LEVEL] != 0:
            continue
        kind = kinds[int(action[0])]
        direction = directions[int(action[1])]          # read only for the directional kinds
        rows.append(ram.copy())
        labels.append((kind, direction, names.get(int(policy), '?')))
    if not rows:
        return
    probe = _native.exposed_probe(np.stack(rows), [k for k, _, _ in labels], [d for _, d, _ in labels])
    for ram, (kind, direction, family), (near, monster, aimed, exposed) in zip(rows, labels, probe):
        stats['decisions'] += 1
        if kind not in ATTACKS:
            continue
        stats['attacks'] += 1
        stats[f'attacks {kind}'] += 1
        stats['shot near'] += int(near)
        stats['shot near, monster near'] += int(near and monster)
        stats['shot near, aimed'] += int(near and aimed and not monster)
        stats['exposed'] += int(exposed)
        if exposed:
            stats[f'exposed {kind}'] += 1
            stats[f'exposed by {family}'] += 1
        if hand is not None and kind in HAND:
            case = hand_case(ram, direction)
            if case is not None:
                hand[case + (bool(exposed),)] += 1


def main():
    jobs.require_job()
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('root')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    folders = sorted(p for p in glob.glob(os.path.join(args.root, '*-*')) if os.path.isdir(p))
    files = [(os.path.basename(f).split('-')[0], p) for f in folders for p in sorted(glob.glob(f'{f}/*.tfreplay'))]
    stats = defaultdict(Counter)
    hand = Counter()
    for i, (team, path) in enumerate(files):
        try:
            scan(path, stats[team], hand if team == 'new' else None)
        except Exception as e:  # pylint: disable=broad-except
            print(f'skip {path}: {e}', file=sys.stderr)
            stats[team]['skipped'] += 1
        jobs.progress(i + 1, len(files), 'replays')
    out = {'teams': {t: dict(c) for t, c in stats.items()},
           'hand': [{'zora_in_lane': z, 'fireball_in_lane': f, 'exposed': e, 'attacks': n}
                    for (z, f, e), n in sorted(hand.items())]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print('| Team | Overworld decisions | Attacks | Shot near | ... and a monster near | ... and aimed | Fires |'
          ' per attack | per 1000 decisions |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for team, c in sorted(stats.items()):
        fires = c['exposed']
        print(f"| {team} | {c['decisions']} | {c['attacks']} | {c['shot near']} | {c['shot near, monster near']} | "
              f"{c['shot near, aimed']} | {fires} | {fires / max(c['attacks'], 1):.1%} | "
              f"{1000 * fires / max(c['decisions'], 1):.1f} |")
    for team, c in sorted(stats.items()):
        print(f'\n{team}: ' + ', '.join(f'{k} {v}' for k, v in sorted(c.items()) if k.startswith(('exposed ', 'attacks '))))
    print('\nThe new team\'s SWORD/BEAMS/BOOMERANG, Zora fireball within 64 px, no monster within 48 px:')
    print('| Zora in its lane | Fireball in its lane | Attacks | Penalty fires |')
    print('|---|---|---:|---:|')
    rows = Counter()
    fired = Counter()
    for (z, f, e), n in hand.items():
        rows[(z, f)] += n
        fired[(z, f)] += n if e else 0
    for (z, f), n in sorted(rows.items(), key=lambda kv: -kv[1]):
        print(f"| {'yes' if z else 'no'} | {'yes' if f else 'no'} | {n} | {fired[(z, f)]} |")
    print(f'| all | | {sum(rows.values())} | {sum(fired.values())} |')


if __name__ == '__main__':
    main()
