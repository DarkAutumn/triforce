"""Round 21 (docs/experiments/round21.md): how a gauntlet policy uses the wrap, from replays recorded with the
decisions stream (`scripts/experiments/r21_gauntlet.sh`: target/r21-gauntlet/replays/<arm>-s<seed>/ next to the
run's eval-replays.json, whose records give each replay's start state and target exit).

    python scripts/experiments/r21_behaviour.py REPLAY_DIR EVAL_JSON [--trace]

A wrap is a decision after which Link's X jumped across the screen (|dX| > 128) with no room change: W when he came
out at the east edge, E at the west.  Prints one JSON line per episode (start state, target, ending, actions, wraps
with the step, the direction and Link's position) and a summary."""
import json
import sys
from pathlib import Path

from triforce import _native as native

KINDS = ['MOVE', 'WAIT', 'SWORD', 'BEAMS', 'BOMB', 'BOOMERANG', 'ARROW', 'CLIP']
DIRS = ['N', 'S', 'W', 'E', 'NW', 'NE', 'SW', 'SE']
DURS = [4, 8, 16, 32]
X, Y, ROOM = 0x70, 0x84, 0xEB


def describe(action):
    kind, direction, duration = (int(v) for v in action)
    name = KINDS[kind]
    if name == 'WAIT':
        return f'WAIT {DURS[duration]}'
    if name == 'MOVE':
        return f'MOVE {DIRS[direction]} {DURS[duration]}'
    return f'{name} {DIRS[direction]}'


def trace(path: Path, states_dir: str):
    data = path.read_bytes()
    replay = native.Replay(data)
    decisions = replay.decisions()
    player = native.ReplayPlayer(data, states_dir=states_dir)
    frames = [int(f) for f in decisions['frame']]
    positions = []
    for frame in frames + [replay.frames]:
        while player.frame < frame and player.step(min(frame - player.frame, 1 << 20)):
            pass
        ram = player.ram()
        positions.append((int(ram[X]), int(ram[Y]), int(ram[ROOM])))
    actions = [describe(a) for a in decisions['action']]
    return actions, positions


def analyse(actions, positions, entry, target):
    wraps = []
    for i, action in enumerate(actions):
        (x0, y0, r0), (x1, y1, r1) = positions[i], positions[i + 1]
        if r0 == r1 and abs(x1 - x0) > 128:
            wraps.append({'step': i, 'dir': 'W' if x1 > x0 else 'E', 'action': action, 'x': x0, 'y': y0,
                          'moved_before': any(positions[j][1] != positions[0][1] for j in range(i + 1)),
                          'y_changed_before': positions[i][1] != positions[0][1]})
    return wraps


def main():
    replay_dir, eval_json = Path(sys.argv[1]), Path(sys.argv[2])
    records = json.loads(eval_json.read_text())['records']
    by_seed = {r['seed']: r for r in records}
    episodes = []
    for path in sorted(replay_dir.glob('*.tfreplay')):
        seed = int(path.name.split('-')[0])
        record = by_seed[seed]
        actions, positions = trace(path, 'experiments/r21/states')
        entry, target = record['entry'], record['target']
        wraps = analyse(actions, positions, entry, target)
        crossing = target != 'S'
        episode = {'state': record['start_state'], 'entry': entry, 'target': target, 'ending': record['ending'],
                   'steps': record['steps'], 'health_change': record['health'] - record['start_health'],
                   'wraps': wraps, 'first_actions': actions[:6]}
        # A crossing won by a wrap: the last wrap went the way of the crossing (a W wrap for a W entry) and Link
        # left within 3 decisions of it.
        if crossing and wraps:
            last = wraps[-1]
            episode['won_by_wrap'] = (record['ending'] == 'exit-correct' and last['dir'] == entry
                                      and len(actions) - last['step'] <= 3)
        episode['wrong_way_wraps'] = sum(w['dir'] != entry for w in wraps) if crossing else None
        episode['first_wrap_at_entry'] = bool(wraps) and wraps[0]['step'] == 0
        episodes.append(episode)
        if '--trace' in sys.argv:
            episode['trace'] = [f'{a} @({p[0]:#04x},{p[1]:#04x})' for a, p in zip(actions, positions)]
        print(json.dumps(episode))
    crossing = [e for e in episodes if e['target'] != 'S']
    south = [e for e in episodes if e['target'] == 'S']

    def share(items, test):
        return f'{sum(map(test, items))}/{len(items)}'
    summary = {
        'episodes': len(episodes),
        'crossing': len(crossing), 'south': len(south),
        'crossing_with_wrap': share(crossing, lambda e: bool(e['wraps'])),
        'crossing_won_by_wrap': share(crossing, lambda e: bool(e.get('won_by_wrap'))),
        'crossing_won_walking': share(crossing, lambda e: e['ending'] == 'exit-correct' and not e['wraps']),
        'crossing_first_wrap_at_once': share(crossing, lambda e: e['first_wrap_at_entry']),
        'crossing_first_wrap_after_row_change': share(
            crossing, lambda e: bool(e['wraps']) and e['wraps'][0]['y_changed_before']),
        'crossing_wrong_way_wraps': sum(e['wrong_way_wraps'] or 0 for e in crossing),
        'crossing_exit_wrong_through_entry': share(crossing, lambda e: e['ending'] == 'exit-wrong'),
        'south_with_wrap': share(south, lambda e: bool(e['wraps'])),
        'south_wraps': sum(len(e['wraps']) for e in south),
        'wraps_per_episode': sum(len(e['wraps']) for e in episodes) / max(1, len(episodes)),
    }
    print(json.dumps({'summary': summary}))


if __name__ == '__main__':
    main()
