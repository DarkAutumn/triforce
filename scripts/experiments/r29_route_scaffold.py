"""A throwaway data root for playing today's models on the leg from Dungeon 8 to Dungeon 9 and on the crossed rooms
of Dungeon 9 before round 29 builds the route (docs/routes/d9-walks.md): a copy of the checkout's data with a
scouting route `d9walk` (the draft's steps from Dungeon 8's exit to Ganon's door, appended to `d6`; of Dungeon 8
only the piece's room is in it, so that the route's progress holds eight pieces: a step is matched by exactly what
Link holds, and without it every room is off the route), the rooms tagged with the families that would take them, the families' yamls pointed at entry states
made by the walks' staging (`r29_route_walks.route_states`: 15 hearts, the magic sword, the route's keys, bombs and
doors, the red ring after cellar 0x00; `ram_randomization: {}`).  Nothing in the checkout changes.

    python scripts/experiments/r29_route_scaffold.py OUT [--wizzrobe FAMILY] [--likelike FAMILY] [--trap FAMILY]
        [--peahat FAMILY] [--zol FAMILY] [--whirlwind FAMILY] [--no-ring]
    TRIFORCE_MODELS_DIR=$PWD/models TRIFORCE_STATES_DIR=OUT/triforce/states PYTHONPATH=OUT \
        python -m triforce.eval_batch NAME --families ow-lynel,like-like --episodes-per-state 20 --output-dir DIR

One scaffold per candidate assignment.  The route, tags and family yamls here are the walk's guesses, not round 29's
decisions.  What the engine cannot say yet is left out or played without it: 0x47 is entered and left E (no burned
tree, no cave), the treasure cellars 0x00 and 0x4f have no waypoint (the ring and the silver arrow are in the staged
states that follow), and the rooms that are another walk's fights (0x55, 0x07, 0x16 from S, 0x05, 0x61 from E, 0x20
out of the tunnel, 0x10 from S, 0x30, 0x52) are steps without start states."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from r26_walks import ROOT
from r29_route_walks import route_states

# (level, room, in_cave, entry, exit, family key, the staged state that enters it (None: not played), extra step keys)
STEPS = [
    (8, 0x2C, False, 'S', 'triforce', 'stalfos-keese-gel', None, {}),   # Dungeon 8's piece: the route's progress
    (0, 0x6D, False, 'cave', 'recorder', 'WHIRLWIND', '0_6dc', {'bomb_reserve': 8}),   # bombs for the walls only
    (0, 0x37, False, 'recorder', 'E', 'ow-octorok', '0_37r', {}),
    (0, 0x38, False, 'W', 'S', 'ow-zora', '0_38w', {}),
    (0, 0x48, False, 'N', 'W', 'ow-leever', '0_48n', {}),
    (0, 0x47, False, 'E', 'E', 'ow-zora', None, {}),
    (0, 0x48, False, 'W', 'N', 'ow-leever', '0_48w', {}),
    (0, 0x38, False, 'S', 'N', 'ow-zora', '0_38s', {}),
    (0, 0x28, False, 'S', 'W', 'PEAHAT', '0_28s', {}),
    (0, 0x27, False, 'E', 'N', 'PEAHAT', '0_27e', {}),
    (0, 0x17, False, 'S', 'N', 'ow-boulder', '0_17s', {}),
    (0, 0x07, False, 'S', 'W', 'ow-lynel', '0_07s', {}),
    (0, 0x06, False, 'E', 'W', 'ow-lynel', '0_06e', {}),
    (0, 0x05, False, 'E', 'cave', 'ow-lynel', '0_05e', {}),
    (9, 0x76, False, 'S', 'N', 'stalfos-keese-gel', '9_76s', {}),
    (9, 0x66, False, 'S', 'W', 'stalfos-keese-gel', '9_66s', {}),
    (9, 0x65, False, 'E', 'N', 'WIZZROBE', '9_65e', {}),
    (9, 0x55, False, 'S', 'cave', 'WIZZROBE', None, {}),
    (9, 0x60, True, 'S', 'W', 'equipment-room', '9_60c', {}),
    (9, 0x14, False, 'cave', 'E', 'LIKELIKE', '9_14c', {}),
    (9, 0x15, False, 'W', 'E', 'WIZZROBE', '9_15w', {}),
    (9, 0x16, False, 'W', 'S', 'WIZZROBE', '9_16w', {'treasure': 'skip'}),
    (9, 0x26, False, 'N', 'E', 'stalfos-keese-gel', '9_26n', {'treasure': 'skip'}),
    (9, 0x27, False, 'W', 'N', 'WIZZROBE', '9_27w', {'treasure': 'skip'}),
    (9, 0x17, False, 'S', 'N', 'WIZZROBE', '9_17s', {}),
    (9, 0x07, False, 'S', 'cave', 'WIZZROBE', None, {}),
    (9, 0x00, True, 'S', 'N', 'equipment-room', None, {}),
    (9, 0x07, False, 'cave', 'S', 'WIZZROBE', '9_07c', {}),
    (9, 0x17, False, 'N', 'S', 'WIZZROBE', '9_17n', {}),
    (9, 0x27, False, 'N', 'W', 'WIZZROBE', '9_27n', {'treasure': 'skip'}),
    (9, 0x26, False, 'E', 'N', 'stalfos-keese-gel', '9_26e', {'treasure': 'skip'}),
    (9, 0x16, False, 'S', 'N', 'WIZZROBE', None, {}),
    (9, 0x06, False, 'S', 'W', 'stalfos-keese-gel', '9_06s', {}),
    (9, 0x05, False, 'E', 'cave', 'WIZZROBE', None, {}),
    (9, 0x70, True, 'S', 'W', 'equipment-room', '9_70c', {}),
    (9, 0x63, False, 'cave', 'W', 'ZOL', '9_63c', {}),
    (9, 0x62, False, 'E', 'W', 'stalfos-keese-gel', '9_62e', {'treasure': 'skip'}),
    (9, 0x61, False, 'E', 'cave', 'WIZZROBE', None, {}),
    (9, 0x75, True, 'S', 'W', 'equipment-room', '9_75c', {}),
    (9, 0x20, False, 'cave', 'N', 'WIZZROBE', None, {}),
    (9, 0x10, False, 'S', 'cave', 'WIZZROBE', None, {}),
    (9, 0x4F, True, 'S', 'N', 'equipment-room', None, {}),
    (9, 0x10, False, 'cave', 'S', 'WIZZROBE', '9_10c', {}),
    (9, 0x20, False, 'N', 'cave', 'WIZZROBE', '9_20n', {}),
    (9, 0x75, True, 'S', 'E', 'equipment-room', '9_75c_v2', {}),
    (9, 0x61, False, 'cave', 'N', 'WIZZROBE', '9_61c', {}),
    (9, 0x51, False, 'S', 'N', 'LIKELIKE', '9_51s', {}),
    (9, 0x41, False, 'S', 'N', 'TRAP', '9_41s', {}),
    (9, 0x31, False, 'S', 'W', 'WIZZROBE', '9_31s', {}),
    (9, 0x30, False, 'E', 'cave', 'WIZZROBE', None, {}),
    (9, 0x67, True, 'S', 'E', 'equipment-room', '9_67c', {}),
    (9, 0x04, False, 'cave', 'W', 'WIZZROBE', '9_04c', {}),
    (9, 0x03, False, 'E', 'cave', 'ZOL', '9_03e', {}),
    (9, 0x77, True, 'S', 'W', 'equipment-room', '9_77c', {}),
    (9, 0x52, False, 'cave', 'N', 'WIZZROBE', None, {}),
]
NO_RING = {(9, 0x26), (9, 0x27), (9, 0x17), (9, 0x07), (9, 0x00)}          # the detour's rooms
WALLS = {(9, 0x65), (9, 0x26), (9, 0x27), (9, 0x17), (9, 0x06), (9, 0x20), (9, 0x31), (9, 0x04), (0, 0x05)}
BOMB_MODES = ('free', 'required')


def write_specialists(out: Path, rooms_of: dict, names: dict):
    """The family yamls with the scaffold's rooms; a family with staged states (`names`) plays them as they are."""
    (out / 'specialists').mkdir(exist_ok=True)
    for f in (ROOT / 'specialists').glob('*.yaml'):
        data = yaml.safe_load(f.read_text())
        name = data['name']
        mine = rooms_of.get(name, [])
        data['rooms'] = [r for r in data['rooms'] if r in mine or not any(
            r in rooms for family, rooms in rooms_of.items() if family != name)] + [
                r for r in mine if r not in data['rooms']]
        if name in names:
            data['route'] = 'd9walk'
            data.get('reward', {}).pop('heart_value', None)      # the table is fitted for d2first
            data['ram_randomization'] = {}
            data['curriculum'] = {'enabled': False}
            data['start_states'] = [f'{stem}_[0-9]' for stem in names[name]]
            if any(f'({level}, 0x{room:02x})' in mine for level, room in WALLS):
                items = data.setdefault('items', {})
                if items.get('bombs') not in BOMB_MODES:
                    items['bombs'] = 'required'                  # the wall's bomb must be offered
        (out / 'specialists' / f.name).write_text(yaml.safe_dump(data, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('out', type=Path)
    parser.add_argument('--wizzrobe', default='goriya', help='plays the Wizzrobe and Patra rooms that are crossed')
    parser.add_argument('--likelike', default='like-like', help='plays 0x14 and 0x51')
    parser.add_argument('--trap', default='like-like', help='plays 0x41')
    parser.add_argument('--peahat', default='ow-zora', help='plays overworld 0x28 and 0x27 (tagged ow-zora today)')
    parser.add_argument('--zol', default='zol', help='plays 0x63 and 0x03')
    parser.add_argument('--whirlwind', default='ow-moblin', help="plays 0x6d (the recorder must be in its `items`)")
    parser.add_argument('--no-ring', action='store_true', help='the way without the red ring detour')
    parser.add_argument('--hearts', type=float, default=None, help="the states' hearts (default: as staged, full)")
    parser.add_argument('--extends', default='d6', help='the route d9walk is appended to (d6, d2first)')
    args = parser.parse_args()
    out = args.out.resolve()
    sub = {'WIZZROBE': args.wizzrobe, 'LIKELIKE': args.likelike, 'TRAP': args.trap, 'PEAHAT': args.peahat,
           'ZOL': args.zol, 'WHIRLWIND': args.whirlwind}
    steps = [s for s in STEPS if not (args.no_ring and (s[0], s[1]) in NO_RING)]
    if args.no_ring:                                             # 0x16 is entered once, from W, and left N
        steps = [s for s in steps if (s[1], s[3]) != (0x16, 'S')]
        steps = [(s[0], s[1], s[2], s[3], 'N', s[5], None, {}) if (s[1], s[3]) == (0x16, 'W') else s for s in steps]
    (out / 'triforce').mkdir(parents=True)
    for f in (ROOT / 'triforce').iterdir():
        if f.name not in ('game.yaml', 'routes.yaml', 'states'):
            (out / 'triforce' / f.name).symlink_to(f)
    for name in ('roms', 'models', 'trajectories', 'scripts', 'zelda-asm'):
        if (ROOT / name).exists():
            (out / name).symlink_to(ROOT / name)
    # game.yaml: the tags, the walls
    game = yaml.safe_load((ROOT / 'triforce' / 'game.yaml').read_text())
    family_of = {(s[0], s[1], s[2]): sub.get(s[5], s[5]) for s in steps}
    for room in game['rooms']:
        key = (room['level'], int(room['location'], 16), bool(room.get('in_cave')))
        if key in family_of:
            room['specialist'] = family_of[key]
            if key[:2] in WALLS and not key[2]:
                room['requires_bombs'] = True
            if key == (0, 0x05, False):
                room['bomb_secret'] = list(room['entrance']['tile'])
    (out / 'triforce' / 'game.yaml').write_text(yaml.safe_dump(game, sort_keys=False, width=200))
    # routes.yaml: the route and the families' room lists
    lines = ['', '# d9walk (round 29 walks, docs/routes/d9-walks.md): a scaffold, never committed.', 'd9walk:',
             f'  extends: {args.extends}', '  steps:']
    rooms_of = {}
    for level, room, in_cave, entry, exit_, family, _, extra in steps:
        family = sub.get(family, family)
        step = {'level': level, 'room': f'0x{room:02x}'}
        if in_cave:
            step['in_cave'] = True
        step.update(specialist=family, entry=entry, exit=exit_, **extra)
        lines.append('  - ' + yaml.safe_dump(step, default_flow_style=True, sort_keys=False, width=200).strip())
        name = f'({level}, 0x{room:02x}' + (', cave)' if in_cave else ')')
        if name not in rooms_of.setdefault(family, []):
            rooms_of[family].append(name)
    routes = (ROOT / 'triforce' / 'routes.yaml').read_text().split('\n')
    for i, line in enumerate(routes):
        m = re.match(r'  ([a-z0-9-]+): \[(.*)\]$', line)
        if m:
            kept = m.group(2)
            for family, rooms in rooms_of.items():               # a room leaves the family it was listed under
                if family != m.group(1):
                    for r in rooms:
                        kept = re.sub(r"(, )?'" + re.escape(r) + r"'", '', kept).lstrip(', ')
            new = [r for r in rooms_of.get(m.group(1), []) if f"'{r}'" not in kept]
            routes[i] = f"  {m.group(1)}: [" + ', '.join([kept] * bool(kept) + [f"'{r}'" for r in new]) + ']'
    (out / 'triforce' / 'routes.yaml').write_text('\n'.join(routes) + '\n'.join(lines) + '\n')
    # the states, under the names the route gives its steps
    states = out / 'triforce' / 'states'
    states.mkdir()
    for f in (ROOT / 'triforce' / 'states').iterdir():
        (states / f.name).symlink_to(f)
    code = ("import json; from triforce import _native; from triforce.entry_states import state_stem; "
            "print(json.dumps([state_stem(s['level'], s['room'], s['in_cave'], s['entry'], s['visit']) "
            "for s in _native.route_steps('d9walk')]))")
    write_specialists(out, rooms_of, {})                         # the route loads only beside the family yamls
    stems = json.loads(subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True,
                                      env=dict(os.environ, PYTHONPATH=str(out)), cwd=out).stdout)[-len(steps):]
    staged, _, cheats = route_states(out / 'route-states', ring=not args.no_ring)
    names = {}
    for (level, room, in_cave, entry, exit_, family, state, _), stem in zip(steps, stems):
        if state is None:
            continue
        data = Path(staged[state]).read_bytes()
        (states / f'{stem}_0.state').write_bytes(data)
        names.setdefault(sub.get(family, family), []).append(stem)
    if args.hearts is not None:
        from triforce._native import Game            # pylint: disable=import-outside-toplevel
        from r27_walks import hearts_bytes           # pylint: disable=import-outside-toplevel
        game_ = Game(states_dir=str(states))
        for stems_ in names.values():
            for stem in stems_:
                path = states / f'{stem}_0.state'
                game_.load(str(path), randomize_rng=False)
                containers = (game_.read(0x66F) >> 4) + 1
                for key, value in hearts_bytes(args.hearts, containers).items():
                    game_.set_value(key, value)
                game_.save_state_file(str(path))
    write_specialists(out, rooms_of, names)
    print(f'scaffold in {out}: ' + '; '.join(f'{k}: {v}' for k, v in names.items()), file=sys.stderr)
    print(f'bombs written by the staging walk: {cheats}', file=sys.stderr)
    print(f'evaluate with TRIFORCE_STATES_DIR={states} PYTHONPATH={out}', file=sys.stderr)


if __name__ == '__main__':
    main()
