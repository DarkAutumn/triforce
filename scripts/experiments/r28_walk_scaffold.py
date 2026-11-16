"""A throwaway data root for playing today's models on the leg from Dungeon 6 to Dungeon 8 and in Dungeon 8 before
round 28 builds the route (docs/routes/d8-walks.md): a copy of the checkout's data with a scouting route `d8walk`
(the draft's steps from Dungeon 6's exit to Dungeon 8's piece, extending `d6`), the rooms tagged with the families
that would take them, the families' yamls pointed at entry states made by the walks' staging (`r28_walks.leg_states`,
`d8_state`, `d8_cellar_states`: 13 hearts, the magic sword, the route's keys, bombs and doors;
`ram_randomization: {}`).  Nothing in the checkout changes.

    python scripts/experiments/r28_walk_scaffold.py OUT [--manhandla F] [--darknut F] [--cross F] [--pols F]
                                                        [--boss F] [--skip-key] [--boss-hearts H] [--hearts H]
    TRIFORCE_MODELS_DIR=$PWD/models TRIFORCE_STATES_DIR=OUT/triforce/states PYTHONPATH=OUT \
        python -m triforce.eval_batch NAME --families darknut,gibdo --episodes-per-state 20 --output-dir DIR

One scaffold per assignment of families to the roles: `--manhandla` plays 0x6e (cross, bomb the N wall), `--darknut`
the two forced fights 0x5e and 0x3e, `--cross` the mixed rooms 0x4e and 0x3f, `--pols` 0x4c, `--boss` Gleeok's room.
The route, tags and family yamls here are the walk's guesses, not round 28's decisions.  What the engine cannot say
yet is played without it: 0x6d's tree has no burn rule (game.yaml `how: burn` is read by nothing: CANDLE is turned
on for the family and the objective is whatever `exit: cave` gives), 0x4c's corner key is the room's `treasure`
(`--skip-key`: `treasure: skip`, the wall only), 0x6e and 0x4c are tagged `requires_bombs` (0x4c's state holds the
one bomb the no-drop run would have cheated in)."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from r26_walks import ROOT
from r28_walks import d8_cellar_states, d8_state, leg_states

ROUTE = """
# d8walk (round 28 walks, docs/routes/d8-walks.md): a scaffold, never committed.
d8walk:
  extends: d6
  steps:
  - {level: 0, room: '0x22', specialist: ow-lynel, entry: cave, exit: recorder, bomb_reserve: 1}
  - {level: 0, room: '0x3c', specialist: ow-octorok, entry: recorder, exit: S}
  - {level: 0, room: '0x4c', specialist: ow-octorok, entry: N, exit: E}
  - {level: 0, room: '0x4d', specialist: ow-moblin, entry: W, exit: S}
  - {level: 0, room: '0x5d', specialist: ow-moblin, entry: N, exit: S}
  - {level: 0, room: '0x6d', specialist: ow-moblin, entry: N, exit: cave}
  - {level: 8, room: '0x7e', specialist: stalfos-keese-gel, entry: S, exit: N}
  - {level: 8, room: '0x6e', specialist: MANHANDLA, entry: S, exit: N, treasure: skip}
  - {level: 8, room: '0x5e', specialist: DARKNUT, entry: S, exit: N, bomb_reserve: 0}
  - {level: 8, room: '0x4e', specialist: CROSS, entry: S, exit: N, treasure: skip}
  - {level: 8, room: '0x3e', specialist: DARKNUT, entry: S, exit: E}
  - {level: 8, room: '0x3f', specialist: CROSS, entry: W, exit: cave, treasure: skip}
  - {level: 8, room: '0x2f', in_cave: true, specialist: equipment-room, entry: S, exit: W}
  - {level: 8, room: '0x4c', specialist: POLS, entry: cave, exit: N KEY}
  - {level: 8, room: '0x3c', specialist: BOSS, entry: S, exit: N}
  - {level: 8, room: '0x2c', specialist: stalfos-keese-gel, entry: S, exit: triforce}
"""
ROLE_ROOMS = {'MANHANDLA': (0x6E,), 'DARKNUT': (0x5E, 0x3E), 'CROSS': (0x4E, 0x3F), 'POLS': (0x4C,), 'BOSS': (0x3C,)}
FIXED8 = {0x7E: 'stalfos-keese-gel', 0x2C: 'stalfos-keese-gel'}
OVERWORLD = {0x22: 'ow-lynel', 0x3C: 'ow-octorok', 0x4C: 'ow-octorok', 0x4D: 'ow-moblin', 0x5D: 'ow-moblin',
             0x6D: 'ow-moblin'}
WALLS = {0x6E, 0x4C}
# the leg's states: (room, entry) -> the `leg_states` name
LEG = {(0x22, 'cave'): '0_22c', (0x3C, 'recorder'): '0_3cr', (0x4C, 'N'): '0_4cn', (0x4D, 'W'): '0_4dw',
       (0x5D, 'N'): '0_5dn', (0x6D, 'N'): '0_6dn'}
ITEMS = {'ow-lynel': {'recorder': True}, 'ow-moblin': {'candle': True}}


def tag_game_yaml(text: str, families: dict) -> str:
    """game.yaml with the stretch's rooms tagged (a `specialist:` line replaces the room's own)."""
    out, level, lines = [], None, text.split('\n')
    by_room = {room: families[role] for role, rooms in ROLE_ROOMS.items() for room in rooms}
    by_room.update(FIXED8)
    skip = False
    for i, line in enumerate(lines):
        m = re.match(r'- level: (\d+)$', line)
        if m:
            level, skip = int(m.group(1)), False
        if skip and re.match(r'  specialist:', line):
            skip = False
            continue
        out.append(line)
        m = re.match(r"  location: '(0x[0-9a-f]+)'", line)
        if not m:
            continue
        room, in_cave = int(m.group(1), 16), i + 1 < len(lines) and 'in_cave: true' in lines[i + 1]
        family = None
        if level == 8:
            family = 'equipment-room' if in_cave and room == 0x2F else None if in_cave else by_room.get(room)
        elif level == 0 and not in_cave:
            family = OVERWORLD.get(room)
        if family:
            out.append(f'  specialist: {family}  # d8walk scaffold')
            skip = True
        if level == 8 and room in WALLS and not in_cave:
            out.append('  requires_bombs: true  # d8walk scaffold')
    return '\n'.join(out)


def route_steps(out: Path, route: str):
    """The scaffold's route steps as the engine reads them (their `visit` names the states)."""
    code = f"import json; from triforce import _native; print(json.dumps(_native.route_steps('{route}')))"
    res = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=False,
                         env=dict(os.environ, PYTHONPATH=str(out)), cwd=out)
    if res.returncode != 0:
        raise RuntimeError(f'the scaffold route does not load:\n{res.stderr[-2000:]}')
    return json.loads(res.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('out', type=Path)
    parser.add_argument('--manhandla', default='boss-manhandla', help='the family that crosses 0x6e')
    parser.add_argument('--darknut', default='darknut', help='the family that fights 0x5e and 0x3e')
    parser.add_argument('--cross', default='gibdo', help='the family that crosses 0x4e and 0x3f')
    parser.add_argument('--pols', default='pols-voice', help='the family that plays 0x4c')
    parser.add_argument('--boss', default='boss-gleeok', help="the family that plays Gleeok's room")
    parser.add_argument('--skip-key', action='store_true', help="0x4c: `treasure: skip` (the wall, not the key)")
    parser.add_argument('--hearts', type=float, default=None, help="Dungeon 8 states' hearts (default 13 full)")
    parser.add_argument('--boss-hearts', type=float, default=None, help="Gleeok's state's hearts (default --hearts)")
    parser.add_argument('--arrows', action='store_true', help='ARROW on for the --pols family')
    parser.add_argument('--sword', type=int, default=3, help="Dungeon 8 states' sword (3 magic, 2 white)")
    parser.add_argument('--bombs', type=int, default=None, help="Dungeon 8 states' bombs (default the ledger's)")
    args = parser.parse_args()
    out = args.out.resolve()
    (out / 'triforce').mkdir(parents=True)
    for f in (ROOT / 'triforce').iterdir():
        if f.name not in ('game.yaml', 'routes.yaml', 'states'):
            (out / 'triforce' / f.name).symlink_to(f)
    for name in ('roms', 'models', 'trajectories', 'scripts', 'zelda-asm'):
        if (ROOT / name).exists():
            (out / name).symlink_to(ROOT / name)
    families = {'MANHANDLA': args.manhandla, 'DARKNUT': args.darknut, 'CROSS': args.cross, 'POLS': args.pols,
                'BOSS': args.boss}
    (out / 'triforce' / 'game.yaml').write_text(tag_game_yaml((ROOT / 'triforce' / 'game.yaml').read_text(),
                                                              families))
    route = ROUTE.replace(' KEY', ', treasure: skip' if args.skip_key else '')
    for key, family in families.items():
        route = route.replace(key, family)
    steps = yaml.safe_load(route)['d8walk']['steps']
    added = {}
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
    (out / 'specialists').mkdir()
    for f in (ROOT / 'specialists').glob('*.yaml'):
        data = yaml.safe_load(f.read_text())
        if data['name'] in added:
            data['rooms'] = list(data['rooms']) + [r for r in added[data['name']] if r not in data['rooms']]
        (out / 'specialists' / f.name).write_text(yaml.safe_dump(data, sort_keys=False))
    # the states, named by the route's own visit counts
    from triforce.entry_states import state_stem                # pylint: disable=import-outside-toplevel
    walked = route_steps(out, 'd8walk')[-len(steps):]
    leg = leg_states(out / 'leg-states')
    cellar = d8_cellar_states(out / 'd8-states')
    names = {}
    for step in walked:
        level, room, entry = step['level'], step['room'], step['entry']
        stem = state_stem(level, room, step['in_cave'], entry, step['visit'])
        target = states / f'{stem}_0.state'
        if level == 0:
            source = leg[LEG[(room, entry)]]
        elif step['in_cave']:
            source = cellar['8_2fc']
        elif entry == 'cave':
            source = cellar['8_4cc']
        else:
            hearts = args.boss_hearts if room == 0x3C and args.boss_hearts is not None else args.hearts
            extra = {'sword': args.sword}
            if args.bombs is not None:
                extra['bombs'] = args.bombs
            source = d8_state(room, entry, path=out / 'd8-states' / f'8_{room:02x}{entry.lower()}.state',
                              hearts=hearts, extra=extra)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.write_bytes(Path(source).read_bytes())
        names.setdefault(step['specialist'], []).append(stem)
    for f in (out / 'specialists').glob('*.yaml'):
        data = yaml.safe_load(f.read_text())
        name = data['name']
        if name in names:
            data['route'] = 'd8walk'
            data.get('reward', {}).pop('heart_value', None)      # the table is fitted for d2first
            data['ram_randomization'] = {}
            data['curriculum'] = {'enabled': False}
            data['start_states'] = [f'{state}_[0-9]' for state in names[name]]
            data.setdefault('items', {}).update(ITEMS.get(name, {}))
            if name == args.pols and args.arrows:
                data['items']['arrows'] = True
        f.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f'scaffold in {out}: ' + '; '.join(f'{k}: {v}' for k, v in names.items()), file=sys.stderr)
    print(f'evaluate with TRIFORCE_STATES_DIR={states} PYTHONPATH={out}', file=sys.stderr)


if __name__ == '__main__':
    main()
