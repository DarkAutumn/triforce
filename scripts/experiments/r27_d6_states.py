"""Round 27: the sources and the inventory table of the start states of the leg from Dungeon 7 to the magic sword and
of Dungeon 6 (`d2first` steps 248-280, scouted as the route `d6`; docs/experiments/round27.md "The start states").

    python scripts/experiments/r27_d6_states.py write STATE OUT [--bombs N] [--rupees N] [--keys N]
    python scripts/experiments/r27_d6_states.py table [--dir DIR] [--markdown] [--route d2first] [--first-step 248]
                                                      [--last-step N]

``write`` writes ``OUT/<STATE's name>``, the pre-entry state STATE (not run: it stays pre-entry) with the given
counts: what round 26's ``r26_d7_states.py bombs`` does, for any of the three ledgers.  Used where a state after a
skipped fight (``entry_states build --skip-fights``) would not hold what a run that fought holds.

``table`` prints the sword, hearts and containers, bombs, keys, rupees, arrows and triforce pieces of every state of
steps 248-280 in DIR (default: the states dir), chains 0 / 1 / 2, as each loads (the first playable frame, the room
fresh), with the manifest's ``fight_skipped`` mark.  Round 28 reads its stretch with ``--first-step 281``.
"""

import argparse
import json
import os
import sys

from triforce._native import Game, Snapshot, route_steps
from triforce.entry_states import state_stem
from triforce.paths import STATES_DIR

ROUTE, FIRST_STEP = 'd2first', 248
# RAM (docs/ram-catalog.md): the sword (2 white, 3 magic), bombs, arrows, rupees, keys, hearts and containers, pieces.
SWORD, BOMBS, ARROWS, RUPEES, KEYS, HEARTS, TRIFORCE = 0x657, 0x658, 0x659, 0x66D, 0x66E, 0x66F, 0x671
SWORDS = {0: 'none', 1: 'wood', 2: 'white', 3: 'magic'}
FIELDS = {'bombs': 'bombs', 'rupees': 'rupees', 'keys': 'keys'}


def write(source: str, out: str, values: dict) -> str:
    """Writes ``<out>/<source's name>``: the state file ``source`` with ``values`` (Game.set_value names)."""
    game = Game(seed=0, states_dir=str(STATES_DIR))
    os.makedirs(out, exist_ok=True)
    with open(source, 'rb') as file:
        game.restore(Snapshot.from_state_file(file.read()))
    had = {name: game.read({'bombs': BOMBS, 'rupees': RUPEES, 'keys': KEYS}[name]) for name in values}
    for name, value in values.items():
        game.set_value(FIELDS[name], value)
    path = os.path.join(out, os.path.basename(source))
    game.save_state_file(path)
    print(f"{path}: " + ', '.join(f"{name} {had[name]} -> {value}" for name, value in values.items()))
    return path


def inventory(game: Game, path: str) -> dict:
    """What a state holds at its first playable frame."""
    game.load(path, fresh_room=True, seed=0, randomize_rng=False)
    containers = game.read(HEARTS)
    return {'sword': SWORDS.get(game.read(SWORD), '?'), 'hearts': (containers & 0x0F) + 1,
            'containers': (containers >> 4) + 1, 'bombs': game.read(BOMBS), 'keys': game.read(KEYS),
            'rupees': game.read(RUPEES), 'arrows': game.read(ARROWS), 'pieces': bin(game.read(TRIFORCE)).count('1')}


def table(directory: str, markdown: bool, route: str = ROUTE, first: int = FIRST_STEP, last: int | None = None) -> int:
    """Prints one row per step of ``route`` from ``first`` to ``last`` (the route's end); returns the number of states
    missing."""
    game = Game(seed=0, states_dir=directory)
    manifest_path = os.path.join(directory, 'entry_states.json')
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding='utf-8') as file:
            data = json.load(file)
        manifest = {entry['file']: entry for entry in (data if isinstance(data, list) else data.get('states', []))}
    missing = 0
    if markdown:
        print('| Step | State | Way | Family | Sword | Hearts / containers | Bombs | Keys | Rupees | Arrows | Pieces '
              '| After a skipped fight |')
        print('|---:|---|---|---|---|---|---|---|---|---|---|---|')
    for step in route_steps(route)[first:None if last is None else last + 1]:
        stem = state_stem(step['level'], step['room'], step['in_cave'], step['entry'], step['visit'])
        rows, skipped = [], []
        for chain in (0, 1, 2):
            name = f'{stem}_{chain}.state'
            path = os.path.join(directory, name)
            if not os.path.exists(path):
                missing += 1
                rows.append(None)
                continue
            rows.append(inventory(game, path))
            skipped.append(bool(manifest.get(name, {}).get('fight_skipped')))

        def cell(key, rows=rows):
            values = ['-' if row is None else str(row[key]) for row in rows]
            return values[0] if len(set(values)) == 1 else ' / '.join(values)
        hearts = ['-' if row is None else f"{row['hearts']} / {row['containers']}" for row in rows]
        hearts = hearts[0] if len(set(hearts)) == 1 else '; '.join(hearts)
        mark = 'yes' if skipped and all(skipped) else ('some' if any(skipped) else '')
        cells = [str(step['index']), f'`{stem}`', step['way'] or '', step['specialist'], cell('sword'), hearts,
                 cell('bombs'), cell('keys'), cell('rupees'), cell('arrows'), cell('pieces'), mark]
        print('| ' + ' | '.join(cells) + ' |' if markdown else '  '.join(f'{c:12}' for c in cells))
    return missing


def main(argv=None) -> int:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    one = sub.add_parser('write')
    one.add_argument('state')
    one.add_argument('out')
    for name in FIELDS:
        one.add_argument(f'--{name}', type=int, default=None)
    tab = sub.add_parser('table')
    tab.add_argument('--dir', default=str(STATES_DIR))
    tab.add_argument('--markdown', action='store_true')
    tab.add_argument('--route', default=ROUTE)
    tab.add_argument('--first-step', type=int, default=FIRST_STEP)
    tab.add_argument('--last-step', type=int, default=None)
    args = parser.parse_args(argv)
    if args.command == 'write':
        values = {name: getattr(args, name) for name in FIELDS if getattr(args, name) is not None}
        if not values:
            parser.error('write: give at least one of --bombs, --rupees, --keys')
        write(args.state, args.out, values)
        return 0
    return 1 if table(args.dir, args.markdown, args.route, args.first_step, args.last_step) else 0


if __name__ == '__main__':
    sys.exit(main())
