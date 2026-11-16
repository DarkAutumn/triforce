"""A throwaway data root for playing today's models in Dungeon 7 before round 26 builds the route (docs/routes/
d7-walks.md item 12): a copy of the checkout's data with a scouting route `d7walk` (Dungeon 7 with three keys,
extending d2first), D7's rooms tagged with the families that would take them, `requires_bombs` on its five wall rooms,
the families' yamls pointed at D7 entry states (`ram_randomization: {}`: the states' own 11 hearts and bombs), and
those states (`r26_walks.d7_state`: the route's inventory, doors, keys and bombs).  Nothing in the checkout changes.

    python scripts/experiments/r26_walk_scaffold.py OUT
    TRIFORCE_STATES_DIR=OUT/triforce/states PYTHONPATH=OUT python -m triforce.evaluate --specialist goriya \
        --episodes-per-state 20 --parallel 8 --output EVAL.json --replays DIR --no-log
    ... python -m triforce.cli eval-chain --route d7walk --from 7_0dw_0 --runs 20 --parallel 10 ...

The route, tags and family yamls here are the walk's guesses, not round 26's decisions."""

import argparse
import re
import sys
from pathlib import Path

import yaml

from r26_walks import ROOT, d7_state

ROUTE = """
# d7walk (round 26 walks, docs/routes/d7-walks.md): a scaffold, never committed.  Dungeon 7 with three keys.
d7walk:
  extends: d2first
  steps:
  - {level: 7, room: '0x79', specialist: stalfos-keese-gel, entry: S, exit: N}
  - {level: 7, room: '0x69', specialist: goriya, entry: S, exit: N, treasure: skip}
  - {level: 7, room: '0x59', specialist: goriya, entry: S, exit: N}
  - {level: 7, room: '0x49', specialist: goriya, entry: S, exit: N}
  - {level: 7, room: '0x39', specialist: boss-digdogger, entry: S, exit: W}
  - {level: 7, room: '0x38', specialist: goriya, entry: E, exit: N, treasure: skip}
  - {level: 7, room: '0x28', specialist: goriya, entry: S, exit: N}
  - {level: 7, room: '0x18', specialist: goriya, entry: S, exit: E}
  - {level: 7, room: '0x19', specialist: goriya, entry: W, exit: E}
  - {level: 7, room: '0x1a', specialist: goriya, entry: W, exit: cave}
  - {level: 7, room: '0x4a', in_cave: true, specialist: equipment-room, entry: S, exit: N}
  - {level: 7, room: '0x1a', specialist: goriya, entry: cave, exit: E}
  - {level: 7, room: '0x1b', specialist: goriya, entry: W, exit: E, treasure: {if_bombs_below: 8}}
  - {level: 7, room: '0x1c', specialist: boss-digdogger, entry: W, exit: N}
  - {level: 7, room: '0x0c', specialist: dodongo-crossing, entry: S, exit: E, treasure: skip}
  - {level: 7, room: '0x0d', specialist: wallmaster, entry: W, exit: cave}
  - {level: 7, room: '0x7b', in_cave: true, specialist: equipment-room, entry: S, exit: W}
  - {level: 7, room: '0x29', specialist: goriya, entry: cave, exit: E}
  - {level: 7, room: '0x2a', specialist: boss-aquamentus, entry: W, exit: E}
  - {level: 7, room: '0x2b', specialist: stalfos-keese-gel, entry: W, exit: triforce}
"""
FAMILY = {0x79: 'stalfos-keese-gel', 0x69: 'goriya', 0x59: 'goriya', 0x49: 'goriya', 0x39: 'boss-digdogger',
          0x38: 'goriya', 0x28: 'goriya', 0x18: 'goriya', 0x19: 'goriya', 0x1A: 'goriya', 0x1B: 'goriya',
          0x1C: 'boss-digdogger', 0x0C: 'dodongo-crossing', 0x0D: 'wallmaster', 0x29: 'goriya',
          0x2A: 'boss-aquamentus', 0x2B: 'stalfos-keese-gel'}
CELLARS = {0x4A: 'equipment-room', 0x7B: 'equipment-room'}
WALLS = {0x69, 0x19, 0x1A, 0x0C, 0x29}
# The entry states the families play (room, entry) and the bombs brought into Dungeon 7 (the walks' ledger: ~3).
STATES = {'goriya': [(0x69, 'S'), (0x59, 'S'), (0x49, 'S'), (0x19, 'W'), (0x1A, 'W'), (0x1B, 'W')],
          'boss-digdogger': [(0x39, 'S'), (0x1C, 'W')], 'dodongo-crossing': [(0x0C, 'S')],
          'wallmaster': [(0x0D, 'W')], 'boss-aquamentus': [(0x2A, 'W')]}


def tag_game_yaml(text: str) -> str:
    out, level, lines = [], None, text.split('\n')
    for i, line in enumerate(lines):
        out.append(line)
        m = re.match(r'- level: (\d+)$', line)
        if m:
            level = int(m.group(1))
        m = re.match(r"  location: '(0x[0-9a-f]+)'", line)
        if m and level == 7:
            room, in_cave = int(m.group(1), 16), i + 1 < len(lines) and 'in_cave: true' in lines[i + 1]
            family = (CELLARS if in_cave else FAMILY).get(room)
            if family:
                out.append(f'  specialist: {family}  # d7walk scaffold')
            if room in WALLS and not in_cave:
                out.append('  requires_bombs: true  # d7walk scaffold')
    return '\n'.join(out)


def room_name(room, cave=False):
    return f'(7, 0x{room:02x}' + (', cave)' if cave else ')')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    (out / 'triforce').mkdir(parents=True)
    for f in (ROOT / 'triforce').iterdir():
        if f.name not in ('game.yaml', 'routes.yaml', 'states'):
            (out / 'triforce' / f.name).symlink_to(f)
    for name in ('roms', 'models', 'trajectories', 'scripts', 'zelda-asm'):
        if (ROOT / name).exists():
            (out / name).symlink_to(ROOT / name)
    (out / 'triforce' / 'game.yaml').write_text(tag_game_yaml((ROOT / 'triforce' / 'game.yaml').read_text()))
    routes = (ROOT / 'triforce' / 'routes.yaml').read_text()
    added = {}
    for room, family in FAMILY.items():
        added.setdefault(family, []).append(room_name(room))
    for room, family in CELLARS.items():
        added.setdefault(family, []).append(room_name(room, cave=True))
    lines = routes.split('\n')
    for i, line in enumerate(lines):
        m = re.match(r'  ([a-z0-9-]+): \[(.*)\]$', line)
        if m and m.group(1) in added:
            lines[i] = f"  {m.group(1)}: [{m.group(2)}, " + ', '.join(f"'{r}'" for r in added[m.group(1)]) + ']'
    (out / 'triforce' / 'routes.yaml').write_text('\n'.join(lines) + ROUTE)
    (out / 'specialists').mkdir()
    for f in (ROOT / 'specialists').glob('*.yaml'):
        data = yaml.safe_load(f.read_text())
        name = data['name']
        if name in added:
            data['rooms'] = list(data['rooms']) + added[name]
            data['route'] = 'd7walk'
            data.get('reward', {}).pop('heart_value', None)      # the table is fitted for d2first
            data['ram_randomization'] = {}
            data['curriculum'] = {'enabled': False}
            data['start_states'] = [f'7_{r:02x}{e.lower()}_[0-9]' for r, e in STATES.get(name, [])] or \
                data['start_states']
            if name == 'dodongo-crossing':
                data.setdefault('items', {})['bombs'] = 'required'      # 0x0c's wall: the round's rule, not today's
        (out / 'specialists' / f.name).write_text(yaml.safe_dump(data, sort_keys=False))
    states = out / 'triforce' / 'states'
    states.mkdir()
    for f in (ROOT / 'triforce' / 'states').iterdir():
        (states / f.name).symlink_to(f)
    for entries in STATES.values():
        for room, entry in entries:
            d7_state(states / f'7_{room:02x}{entry.lower()}_0.state', room, entry, bombs=3,
                     extra_inventory={'rupees': 31})
    print(f'scaffold in {out}; evaluate with TRIFORCE_STATES_DIR={states} PYTHONPATH={out}', file=sys.stderr)


if __name__ == '__main__':
    main()
