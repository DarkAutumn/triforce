"""A throwaway data root for playing today's models in the fight rooms of Dungeon 9 before round 29 builds the route
(docs/routes/d9-fights-walks.md items 1-3): a copy of the checkout's data with a scouting route `d9fights` (one step
per fight room, extending `d6`; the steps are not joined: each room is evaluated alone), every one of those rooms
tagged with ONE family (the candidate under test), that family's yaml pointed at entry states made by the walks'
staging (`r29_fight_walks.d9_state`: 15 containers, the magic sword, with or without the red ring;
`ram_randomization: {}`).  Nothing in the checkout changes.

    python scripts/experiments/r29_fight_scaffold.py OUT --family like-like [--ring 2] [--hearts 15]
    TRIFORCE_MODELS_DIR=$PWD/models TRIFORCE_STATES_DIR=OUT/triforce/states PYTHONPATH=OUT \\
        python -m triforce.eval_batch NAME --families like-like --episodes-per-state 20 --eval-parallel 8 \\
        --output-dir DIR [--acts temperature:0.5+relax:16+plan:16:1]

The rooms and what the step asks (the walk's guesses, not round 29's decisions): 0x55 (Lanmolas) S -> the stairs;
0x05, 0x10, 0x30 (Wizzrobes) from their bomb wall -> the stairs their block makes; 0x20 (Wizzrobes) N -> the stairs
(the second visit's direction, the room full); 0x03 (Zols, Like-Likes) E -> the stairs; 0x61 (Patra) E -> the stairs;
0x16 (Patra) S -> N, its key door written open.  0x52 and Ganon's room are left out: 0x52 is reached only by its
tunnel, and nothing in the engine aims the silver arrow."""

import argparse
import re
import sys
from pathlib import Path

import yaml

from r26_walks import ROOT
from r29_fight_walks import d9_state

# (room, entry, exit)
STEPS = [(0x55, 'S', 'cave'), (0x05, 'E', 'cave'), (0x61, 'E', 'cave'), (0x20, 'N', 'cave'), (0x10, 'S', 'cave'),
         (0x30, 'E', 'cave'), (0x03, 'E', 'cave'), (0x16, 'S', 'N')]
WORLD_FLAGS_7TO9 = 0x77F                                        # asm:WorldFlags, levels 7-9
OPEN = {0x16: {WORLD_FLAGS_7TO9 + 0x16: 0x08, WORLD_FLAGS_7TO9 + 0x06: 0x04}}     # 0x16's N key door, both sides


def tag_game_yaml(text: str, family: str, rooms) -> str:
    out, level, lines = [], None, text.split('\n')
    for i, line in enumerate(lines):
        out.append(line)
        m = re.match(r'- level: (\d+)$', line)
        if m:
            level = int(m.group(1))
        m = re.match(r"  location: '(0x[0-9a-f]+)'", line)
        if m and level == 9 and int(m.group(1), 16) in rooms and \
                not (i + 1 < len(lines) and 'in_cave: true' in lines[i + 1]):
            out.append(f'  specialist: {family}  # d9fights scaffold')
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('out', type=Path)
    parser.add_argument('--family', required=True, help='the family that plays every fight room')
    parser.add_argument('--ring', type=int, default=2, help='0 none, 2 red')
    parser.add_argument('--hearts', type=float, default=15)
    parser.add_argument('--rooms', default=None, help='hex rooms, comma separated (default: all eight)')
    args = parser.parse_args()
    steps = STEPS if args.rooms is None else [s for s in STEPS if s[0] in {int(r, 16) for r in args.rooms.split(',')}]
    out = args.out.resolve()
    (out / 'triforce').mkdir(parents=True)
    for f in (ROOT / 'triforce').iterdir():
        if f.name not in ('game.yaml', 'routes.yaml', 'states'):
            (out / 'triforce' / f.name).symlink_to(f)
    for name in ('roms', 'models', 'trajectories', 'scripts', 'zelda-asm'):
        if (ROOT / name).exists():
            (out / name).symlink_to(ROOT / name)
    family = args.family
    (out / 'triforce' / 'game.yaml').write_text(
        tag_game_yaml((ROOT / 'triforce' / 'game.yaml').read_text(), family, {s[0] for s in steps}))
    route = '\n# d9fights (round 29 fight walks): a scaffold, never committed.\nd9fights:\n  extends: d6\n  steps:\n'
    for room, entry, leave in steps:
        route += (f"  - {{level: 9, room: '0x{room:02x}', specialist: {family}, entry: {entry}, exit: {leave}}}\n")
    rooms = [f'(9, 0x{room:02x})' for room, _, _ in steps]
    routes = (ROOT / 'triforce' / 'routes.yaml').read_text()
    lines = routes.split('\n')
    for i, line in enumerate(lines):
        m = re.match(r'  ([a-z0-9-]+): \[(.*)\]$', line)
        if m and m.group(1) == family:
            lines[i] = f"  {family}: [{m.group(2)}, " + ', '.join(f"'{r}'" for r in rooms) + ']'
    # the scouting route goes before `alternates:` (routes are the keys above it)
    text = '\n'.join(lines)
    cut = text.index('\nalternates:')
    (out / 'triforce' / 'routes.yaml').write_text(text[:cut] + route + text[cut:])
    states = out / 'triforce' / 'states'
    states.mkdir()
    for f in (ROOT / 'triforce' / 'states').iterdir():
        (states / f.name).symlink_to(f)
    names = []
    for room, entry, _ in steps:
        name = f'9_{room:02x}{entry.lower()}'
        path = d9_state(room, entry, ring=args.ring, hearts=args.hearts, silver=True, ram=OPEN.get(room))
        (states / f'{name}_0.state').write_bytes(Path(path).read_bytes())
        names.append(name)
    (out / 'specialists').mkdir()
    for f in (ROOT / 'specialists').glob('*.yaml'):
        data = yaml.safe_load(f.read_text())
        if data['name'] == family:
            data['rooms'] = list(data['rooms']) + [r for r in rooms if r not in data['rooms']]
            data['route'] = 'd9fights'
            data.get('reward', {}).pop('heart_value', None)      # the table is fitted for d2first
            data['ram_randomization'] = {}
            data['curriculum'] = {'enabled': False}
            data['start_states'] = [f'{name}_[0-9]' for name in names]
        (out / 'specialists' / f.name).write_text(yaml.safe_dump(data, sort_keys=False))
    print(f'scaffold in {out}: {family}: {names} (ring {args.ring}, {args.hearts} hearts)', file=sys.stderr)
    print(f'evaluate with TRIFORCE_STATES_DIR={states} PYTHONPATH={out}', file=sys.stderr)


if __name__ == '__main__':
    main()
