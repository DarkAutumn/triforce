"""A throwaway data root for playing today's models on the leg from Dungeon 7 to Dungeon 6 and in Dungeon 6 before
round 27 builds the route (docs/routes/d6-walks.md): a copy of the checkout's data with a scouting route `d6walk`
(the draft's steps from Dungeon 7's exit to Dungeon 6's piece, extending `d7`), the rooms tagged with the families
that would take them, the families' yamls pointed at entry states made by the walks' staging (`r27_walks.leg_states`,
`d6_state`: 12 hearts, the magic sword after 0x21, the route's keys, bombs and doors; `ram_randomization: {}`).
Nothing in the checkout changes.

    python scripts/experiments/r27_walk_scaffold.py OUT [--wizzrobe FAMILY] [--ghini FAMILY] [--boss FAMILY]
    TRIFORCE_MODELS_DIR=$PWD/models TRIFORCE_STATES_DIR=OUT/triforce/states PYTHONPATH=OUT \
        python -m triforce.eval_batch NAME --families goriya,vire --episodes-per-state 20 --output-dir DIR

`--wizzrobe` names the family that plays the six Wizzrobe rooms (0x78, 0x28, 0x29, 0x2c crossed; 0x38, 0x3a fought):
one scaffold per candidate (goriya, darknut, like-like, vire, zol, ...).  The route, tags and family yamls here are
the walk's guesses, not round 27's decisions.  What the engine cannot say yet is left out or played without it: 0x21
is entered and left S (no gravestone push, no cave), Gohma's room has no arrow rule, 0x38's block is the engine's
`push_block_opens`."""

import argparse
import re
import sys
from pathlib import Path

import yaml

from r26_walks import ROOT
from r27_walks import d6_state, leg_states

ROUTE = """
# d6walk (round 27 walks, docs/routes/d6-walks.md): a scaffold, never committed.
d6walk:
  extends: d7
  steps:
  - {level: 0, room: '0x42', specialist: ow-moblin, entry: cave, exit: S}
  - {level: 0, room: '0x52', specialist: ow-moblin, entry: N, exit: W}
  - stretch: woods
    parts: {level: 0, room: '0x51', entry: E}
    rejoins: {level: 0, room: '0x40', entry: S}
    clips:
    - {level: 0, room: '0x51', specialist: ow-moblin, entry: E, exit: W}
    - {level: 0, room: '0x50', specialist: ow-lynel, entry: E, exit: N}
    plain:
    - {level: 0, room: '0x51', specialist: ow-moblin, entry: E, exit: S}
    - {level: 0, room: '0x61', specialist: ow-moblin, entry: N, exit: maze}
    - {level: 0, room: '0x60', specialist: ow-lynel, entry: E, exit: N}
    - {level: 0, room: '0x50', specialist: ow-lynel, entry: S, exit: N}
  - {level: 0, room: '0x40', specialist: GHINI, entry: S, exit: E}
  - {level: 0, room: '0x41', specialist: GHINI, entry: W, exit: N}
  - {level: 0, room: '0x31', specialist: GHINI, entry: S, exit: N}
  - {level: 0, room: '0x21', specialist: GHINI, entry: S, exit: S}
  - {level: 0, room: '0x31', specialist: GHINI, entry: N, exit: E}
  - {level: 0, room: '0x32', specialist: ow-lynel, entry: W, exit: N}
  - {level: 0, room: '0x22', specialist: ow-lynel, entry: S, exit: cave}
  - {level: 6, room: '0x79', specialist: WIZZROBE, entry: S, exit: W, bomb_reserve: 1}
  - {level: 6, room: '0x78', specialist: WIZZROBE, entry: E, exit: N}
  - {level: 6, room: '0x68', specialist: zol, entry: S, exit: N, treasure: skip}
  - {level: 6, room: '0x58', specialist: stalfos-keese-gel, entry: S, exit: N}
  - {level: 6, room: '0x48', specialist: trap-room, entry: S, exit: N}
  - {level: 6, room: '0x38', specialist: WIZZROBE, entry: S, exit: N}
  - {level: 6, room: '0x28', specialist: WIZZROBE, entry: S, exit: E, treasure: skip}
  - {level: 6, room: '0x29', specialist: WIZZROBE, entry: W, exit: S, treasure: skip, bomb_reserve: 0}
  - {level: 6, room: '0x39', specialist: vire, entry: N, exit: E}
  - {level: 6, room: '0x3a', specialist: WIZZROBE, entry: W, exit: cave}
  - {level: 6, room: '0x08', in_cave: true, specialist: equipment-room, entry: S, exit: E}
  - {level: 6, room: '0x1d', specialist: like-like, entry: cave, exit: S}
  - {level: 6, room: '0x2d', specialist: vire, entry: N, exit: W}
  - {level: 6, room: '0x2c', specialist: WIZZROBE, entry: E, exit: N}
  - {level: 6, room: '0x1c', specialist: BOSS, entry: S, exit: N}
  - {level: 6, room: '0x0c', specialist: WIZZROBE, entry: S, exit: triforce}
"""
WIZZROBE_ROOMS = (0x79, 0x78, 0x38, 0x28, 0x29, 0x3A, 0x2C, 0x0C)
FAMILY6 = {0x68: 'zol', 0x58: 'stalfos-keese-gel', 0x48: 'trap-room', 0x39: 'vire', 0x2D: 'vire', 0x1D: 'like-like'}
GRAVEYARD = (0x40, 0x41, 0x31, 0x21)
OVERWORLD = {0x61: 'ow-moblin', 0x32: 'ow-lynel', 0x22: 'ow-lynel'}        # 0x42, 0x52, 0x51, 0x50, 0x60 are tagged
# The entry states: {family key: [(state name, how it is made)]}; 'leg:<name>' is a `leg_states` state, (room, entry)
# a `d6_state`.  The names follow the state rule (a later visit from the same side adds _v<N> along d6walk).
LEG_STATES = {'ow-moblin': [('0_42c_v2', '0_42c'), ('0_52n', '0_52n'), ('0_51e_v2', '0_51e'), ('0_61n', '0_61n')],
              'ow-lynel': [('0_50e_v2', '0_50e'), ('0_60e_v2', '0_60e'), ('0_50s', '0_50s'), ('0_32w', '0_32w'),
                           ('0_22s', '0_22s')],
              'GHINI': [('0_40s', '0_40s'), ('0_41w', '0_41w'), ('0_31s', '0_31s'), ('0_31n', '0_31n')]}
D6_STATES = {'WIZZROBE': [(0x78, 'E'), (0x38, 'S'), (0x28, 'S'), (0x29, 'W'), (0x3A, 'W'), (0x2C, 'E')],
             'zol': [(0x68, 'S')], 'stalfos-keese-gel': [(0x58, 'S')], 'trap-room': [(0x48, 'S')],
             'vire': [(0x39, 'N'), (0x2D, 'N')], 'BOSS': [(0x1C, 'S')]}
WALLS = {0x28}


def tag_game_yaml(text: str, wizzrobe: str, ghini: str, boss: str) -> str:
    out, level, lines = [], None, text.split('\n')
    for i, line in enumerate(lines):
        out.append(line)
        m = re.match(r'- level: (\d+)$', line)
        if m:
            level = int(m.group(1))
        m = re.match(r"  location: '(0x[0-9a-f]+)'", line)
        if not m:
            continue
        room, in_cave = int(m.group(1), 16), i + 1 < len(lines) and 'in_cave: true' in lines[i + 1]
        family = None
        if level == 6:
            family = ('equipment-room' if in_cave and room == 0x08 else None if in_cave else
                      wizzrobe if room in WIZZROBE_ROOMS else boss if room == 0x1C else FAMILY6.get(room))
        elif level == 0 and not in_cave:
            family = ghini if room in GRAVEYARD else OVERWORLD.get(room)
        if family:
            out.append(f'  specialist: {family}  # d6walk scaffold')
        if level == 6 and room in WALLS and not in_cave:
            out.append('  requires_bombs: true  # d6walk scaffold')
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('out', type=Path)
    parser.add_argument('--wizzrobe', default='goriya', help='the family that plays the Wizzrobe rooms')
    parser.add_argument('--ghini', default='ow-tektite', help='the family that plays the graveyard')
    parser.add_argument('--boss', default='boss-aquamentus', help="the family that plays Gohma's room")
    parser.add_argument('--hearts', type=float, default=None, help="Dungeon 6 states' hearts (default 12 full)")
    parser.add_argument('--sword', type=int, default=3, help="Dungeon 6 states' sword (3 magic, 2 white)")
    args = parser.parse_args()
    out = args.out.resolve()
    (out / 'triforce').mkdir(parents=True)
    for f in (ROOT / 'triforce').iterdir():
        if f.name not in ('game.yaml', 'routes.yaml', 'states'):
            (out / 'triforce' / f.name).symlink_to(f)
    for name in ('roms', 'models', 'trajectories', 'scripts', 'zelda-asm'):
        if (ROOT / name).exists():
            (out / name).symlink_to(ROOT / name)
    sub = {'WIZZROBE': args.wizzrobe, 'GHINI': args.ghini, 'BOSS': args.boss}
    (out / 'triforce' / 'game.yaml').write_text(
        tag_game_yaml((ROOT / 'triforce' / 'game.yaml').read_text(), args.wizzrobe, args.ghini, args.boss))
    route = ROUTE
    for key, family in sub.items():
        route = route.replace(key, family)
    # the families' room lists (routes.yaml `families`) and yamls
    added = {}
    for step in yaml.safe_load(route)['d6walk']['steps']:
        steps = [step] if 'room' in step else step['clips'] + step['plain']
        for s in steps:
            room = f"({s['level']}, {s['room']}" + (', cave)' if s.get('in_cave') else ')')
            added.setdefault(s['specialist'], [])
            if room not in added[s['specialist']]:
                added[s['specialist']].append(room)
    routes = (ROOT / 'triforce' / 'routes.yaml').read_text()
    lines = routes.split('\n')
    for i, line in enumerate(lines):
        m = re.match(r'  ([a-z0-9-]+): \[(.*)\]$', line)
        if m and m.group(1) in added:
            new = [r for r in added[m.group(1)] if f"'{r}'" not in m.group(2)]
            if new:
                lines[i] = f"  {m.group(1)}: [{m.group(2)}, " + ', '.join(f"'{r}'" for r in new) + ']'
    (out / 'triforce' / 'routes.yaml').write_text('\n'.join(lines) + route)
    states = out / 'triforce' / 'states'
    states.mkdir()
    for f in (ROOT / 'triforce' / 'states').iterdir():
        (states / f.name).symlink_to(f)
    names = {}
    leg = leg_states(out / 'leg-states')
    for key, entries in LEG_STATES.items():
        for state, source in entries:
            (states / f'{state}_0.state').write_bytes(Path(leg[source]).read_bytes())
            names.setdefault(sub.get(key, key), []).append(state)
    for key, entries in D6_STATES.items():
        for room, entry in entries:
            state = f'6_{room:02x}{entry.lower()}'
            d6_state(room, entry, path=states / f'{state}_0.state', hearts=args.hearts, sword=args.sword,
                     extra={'rupees': 31})
            names.setdefault(sub.get(key, key), []).append(state)
    (out / 'specialists').mkdir()
    for f in (ROOT / 'specialists').glob('*.yaml'):
        data = yaml.safe_load(f.read_text())
        name = data['name']
        if name in added:
            data['rooms'] = list(data['rooms']) + [r for r in added[name] if r not in data['rooms']]
        if name in names:
            data['route'] = 'd6walk'
            data.get('reward', {}).pop('heart_value', None)      # the table is fitted for d2first
            data['ram_randomization'] = {}
            data['curriculum'] = {'enabled': False}
            data['start_states'] = [f'{state}_[0-9]' for state in names[name]]
        (out / 'specialists' / f.name).write_text(yaml.safe_dump(data, sort_keys=False))
    print(f'scaffold in {out}: ' + '; '.join(f'{k}: {v}' for k, v in names.items()), file=sys.stderr)
    print(f'evaluate with TRIFORCE_STATES_DIR={states} PYTHONPATH={out}', file=sys.stderr)


if __name__ == '__main__':
    main()
