"""Round 26: the sources and the inventory table of the start states of the leg to Dungeon 7 and Dungeon 7 (route
steps 209-247; docs/experiments/round26.md "The start states").

    python scripts/experiments/r26_d7_states.py sources OUT
    python scripts/experiments/r26_d7_states.py bombs STATE N OUT
    python scripts/experiments/r26_d7_states.py table [--dir DIR] [--markdown]

``sources`` writes ``OUT/5_14s_<chain>.state`` for chains 0-2: the installed D5 triforce-room states with the bombs
a run really brings out of Dungeon 5 (docs/routes/d7-walks.md §1: 3 in half of the arrivals, mean 3.3, none in
14-17%) instead of whatever their own build left: chain 0 (the chain segments start from it) 3, chain 1 its own 4,
chain 2 none (the arrival whose three first walls are cheated).  Nothing else is written; the states stay pre-entry.
``entry_states build --start-step 208 --source OUT/5_14s_<chain>.state --chains <chain>`` goes on from them.

``bombs`` writes ``OUT/<STATE's name>``, the pre-entry state STATE with N bombs.  Used once: chain 2 enters 7:0x1a
with no bomb, the cheat ledger hands it the room's bomb, and the explorer's cell restores take a granted item for
Link's own (docs/experiments/backlog.md), so its cellar state ``7_4acave_2`` held 5 bombs where a run holds the 4
of the drop it picked up; the chain was built on from the corrected state.

``table`` prints rupees, bombs, keys, hearts and the route's items of every state of steps 209-247 in DIR (default:
the states dir), as each loads (the first playable frame, the room fresh).
"""

import argparse
import os
import sys

from triforce._native import Game, Snapshot, route_steps
from triforce.entry_states import state_stem
from triforce.paths import STATES_DIR

FIRST_STEP, SOURCE_STEP = 209, 208
BOMBS = {0: 3, 1: 4, 2: 0}
# RAM (docs/ram-catalog.md): rupees, bombs, keys, hearts and containers, arrows, the recorder, the food, the candle.
RUPEES, BOMBS_ADDR, KEYS, HEARTS, ARROWS, RECORDER, FOOD, CANDLE = 0x66D, 0x658, 0x66E, 0x66F, 0x659, 0x65C, 0x65D, 0x65B
TRIFORCE = 0x671


def with_bombs(game: Game, source: str, bombs: int, out: str) -> str:
    """Writes ``<out>/<source's name>``: the state file ``source`` (not run: it stays pre-entry) with ``bombs``."""
    os.makedirs(out, exist_ok=True)
    with open(source, 'rb') as file:
        game.restore(Snapshot.from_state_file(file.read()))
    had = game.read(BOMBS_ADDR)
    game.set_value('bombs', bombs)
    path = os.path.join(out, os.path.basename(source))
    game.save_state_file(path)
    print(f"{path}: bombs {had} -> {bombs}")
    return path


def write_sources(out: str) -> list:
    """The three sources; returns their paths."""
    game = Game(seed=0, states_dir=str(STATES_DIR))
    names = {chain: f"{state_stem(5, 0x14, False, 'S')}_{chain}.state" for chain in BOMBS}
    return [with_bombs(game, os.path.join(str(STATES_DIR), names[chain]), bombs, out)
            for chain, bombs in BOMBS.items()]


def inventory(game: Game, path: str) -> dict:
    """What a state holds at its first playable frame."""
    game.load(path, fresh_room=True, seed=0, randomize_rng=False)
    containers = game.read(HEARTS)
    return {'rupees': game.read(RUPEES), 'bombs': game.read(BOMBS_ADDR), 'keys': game.read(KEYS),
            'hearts': (containers & 0x0F) + 1, 'max_hearts': (containers >> 4) + 1,
            'arrows': game.read(ARROWS), 'recorder': game.read(RECORDER), 'food': game.read(FOOD),
            'candle': game.read(CANDLE), 'pieces': bin(game.read(TRIFORCE)).count('1')}


def table(directory: str, markdown: bool) -> int:
    """Prints one row per step and chain triple; returns the number of states missing."""
    game = Game(seed=0, states_dir=directory)
    missing = 0
    if markdown:
        print('| Step | State | Way | Rupees | Bombs | Keys | Hearts | Arrows | Food | Candle |')
        print('|---:|---|---|---|---|---|---|---|---|---|')
    for step in route_steps()[FIRST_STEP:]:
        stem = state_stem(step['level'], step['room'], step['in_cave'], step['entry'], step['visit'])
        rows = []
        for chain in (0, 1, 2):
            path = os.path.join(directory, f'{stem}_{chain}.state')
            if not os.path.exists(path):
                missing += 1
                rows.append(None)
                continue
            rows.append(inventory(game, path))

        def cell(key, rows=rows):
            return ' / '.join('-' if row is None else str(row[key]) for row in rows)
        hearts = ' / '.join('-' if row is None else f"{row['hearts']}" for row in rows)
        cells = [str(step['index']), f'`{stem}`', step['way'] or '', cell('rupees'), cell('bombs'), cell('keys'),
                 hearts, cell('arrows'), cell('food'), cell('candle')]
        print('| ' + ' | '.join(cells) + ' |' if markdown else '  '.join(f'{c:14}' for c in cells))
    return missing


def main(argv=None) -> int:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    sources = sub.add_parser('sources')
    sources.add_argument('out')
    one = sub.add_parser('bombs')
    one.add_argument('state')
    one.add_argument('n', type=int)
    one.add_argument('out')
    tab = sub.add_parser('table')
    tab.add_argument('--dir', default=str(STATES_DIR))
    tab.add_argument('--markdown', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'sources':
        write_sources(args.out)
        return 0
    if args.command == 'bombs':
        with_bombs(Game(seed=0, states_dir=str(STATES_DIR)), args.state, args.n, args.out)
        return 0
    return 1 if table(args.dir, args.markdown) else 0


if __name__ == '__main__':
    sys.exit(main())
