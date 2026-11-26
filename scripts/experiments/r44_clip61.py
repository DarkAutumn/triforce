"""Round 44: 9x61's block clip, scripted from the walked state (the owner, Nov 25: "Yes you clip through there").

    python scripts/experiments/r44_clip61.py [--state PATH] [--waits 0-9] [--out FILE]

The walked 9_61e (by default the walk's carried machine, Link's hearts as he arrives) loaded exactly (no draw, no
write), clips on, then MOVE only to Link's spot below the diamond's E block, ($A0, $9D), one CLIP N into it and MOVE W
onto the stairs under the centre block (the room's layout is UW $1A: 5:0x64's diamond clips the same way,
tests/test_d5.py), the Patra alive. Each wait (WAIT 4 that many times on entry, so the Patra stands elsewhere) is a
take; prints its actions, where it ends, the room's kills and the hearts lost. ``--out`` writes the first take that
reaches the tunnel stairs unhit with no kill, as a JSON list of actions (`entry_states walk --script`)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from triforce._native import Game  # noqa: E402  pylint: disable=wrong-import-position
from triforce.paths import STATES_DIR  # noqa: E402  pylint: disable=wrong-import-position

ROOM_KILLS = 0x34F
CARRY = ROOT / 'target/r44/seg229/carry/d3first_9_61e_0.state'


def take(state: str, wait: int) -> dict:
    game = Game(seed=0, states_dir=str(STATES_DIR), route='d3first')
    now = game.load(state, randomize_rng=False, fresh_room=False)
    game.set_clips_allowed(True)
    start, health, actions, kills = now.full_location, now.health, [], 0

    def act(text):
        nonlocal now, kills
        actions.append(text)
        kills = max(kills, game.read(ROOM_KILLS))
        now = game.act(text).state
        return now.full_location != start or now.game_over

    done = False
    for _ in range(wait):
        done = done or act('WAIT 4')
    plan = [('MOVE W 4', lambda: now.link['x'] <= 0xB0), ('MOVE S 4', lambda: now.link['y'] >= 0x9D),
            ('MOVE W 4', lambda: now.link['x'] <= 0xA0)]
    for text, reached in plan:
        for _ in range(16):
            if done or reached():
                break
            done = act(text)
    if not done:
        done = act('CLIP N')
    for _ in range(16):
        if done:
            break
        done = act('MOVE W 16')
    level, room, in_cave = tuple(now.full_location)
    return {'actions': actions, 'ends': (level, hex(room), in_cave), 'kills': kills,
            'hearts lost': health - now.health, 'stairs': (level, room, in_cave) == (9, 0x75, True)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--state', default=str(CARRY))
    parser.add_argument('--waits', default='0-9')
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    first, last = (int(v) for v in args.waits.split('-'))
    chosen = None
    for wait in range(first, last + 1):
        result = take(args.state, wait)
        print(wait, json.dumps(result), flush=True)
        if chosen is None and result['stairs'] and result['kills'] == 0 and result['hearts lost'] == 0:
            chosen = result['actions']
    if args.out and chosen is not None:
        Path(args.out).write_text(json.dumps(chosen) + '\n')
        print('written', args.out)


if __name__ == '__main__':
    main()
