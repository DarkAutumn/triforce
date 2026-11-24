#!/usr/bin/env python3
"""First draft of a dungeon's game.yaml rooms from the ROM's level data (zelda-asm/tools/romdata.py).

    .venv/bin/python scripts/level_rooms.py 5 6 7 8 9                         # print the entries
    .venv/bin/python scripts/level_rooms.py 5 --append-to triforce/game.yaml  # append them (refused if level 5 is there)

Quest 1.  Per room, as Dungeons 3 and 4 were written by hand (docs/routes/d3.md "Data sources"): exits from the door
types (attr A/B: 0 open, 1 wall, 4 bomb wall, 5/6 key door, 7 shutter) toward the level's own rooms, the stairs'
cellar as the `cave` exit, monsters from attr C/D and the level's FoeCounts, the room item, the secret trigger (attr
F & 7), the push block and the dark flag.  A treasure cellar is an in-cave room with N back up to its room; a tunnel
cellar has W (the left ladder, attr A's room) and E (the right ladder, attr B's room).

Not monsters, so left out of `enemies` and named in a comment: blade traps ($49, $4A), fires ($3F, $40), the rupee
stash ($35).  People keep the `Unknown_<decimal>` names the other levels use.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'zelda-asm' / 'tools'))
import romdata  # noqa: E402

SIDE_ORDER = ('N', 'E', 'S', 'W')                  # game.yaml's order of a room's exits
SIDES = {name: (index, delta) for index, (name, delta) in enumerate(romdata.SIDES)}
WALL, BOMB, KEYS, SHUTTER = 1, 4, (5, 6), 7
TRIGGERS = {1: 'kill-all', 2: 'ringleader', 3: 'boss-dead', 4: 'push-block', 5: 'push-block-stairs',
            6: 'life-or-money', 7: 'kill-all-item'}
TREASURES = {0x00: 'bombs', 0x05: 'recorder', 0x07: 'red-candle', 0x09: 'silver-arrow', 0x0B: 'magic-key',
             0x0E: 'triforce-of-power', 0x0F: '5-rupees', 0x10: 'wand', 0x11: 'book', 0x13: 'red-ring',
             0x16: 'compass', 0x17: 'map', 0x19: 'key', 0x1A: 'heart-container', 0x1B: 'triforce'}
NOT_MONSTERS = {0x35: 'a rupee stash ($35)', 0x3F: 'guard fires ($3F)', 0x40: 'standing fires ($40)',
                0x49: 'blade traps ($49)', 0x4A: 'blade traps ($4A)'}
PERSONS = range(0x4B, 0x53)                        # dungeon people; the hungry Goriya is $36, Zelda $37
BLOCK_STAIRS = (26, 4)                             # where a pushed block's stairs appear (trigger 5)


def enemy_names() -> dict[int, str]:
    """Object type -> the repo's enemy name (`describe::enemy_name`)."""
    text = (ROOT / 'crates/triforce-py/src/describe.rs').read_text()
    body = text[text.index('pub fn enemy_name'):text.index('pub fn item_name')]
    return {int(value, 16): name for value, name in re.findall(r'0x([0-9A-F]{2}) => "(\w+)"', body)}


def stairs_tile(game, room: dict):
    """Top-left tile of the layout's stairs square, None if the layout has none."""
    grid = game.uw_grid(romdata.uw_layout_desc(game, room['lay']))
    for row, squares in enumerate(grid):
        for col, square in enumerate(squares):
            if romdata.uw_square_kind(game, square)[1] == 'stairs':
                return 4 + 2 * col, 4 + 2 * row
    return None


def room_lines(game, level, room_id: int, names: dict[int, str]) -> list[str]:
    lv, room = level.lv, level.room[room_id]
    notes = [word for word, yes in (('the start room: S leaves to the overworld', room_id == level.start),
                                    ('the boss room', room_id == level.boss),
                                    ('the triforce room', room_id == level.tri and level.tri_has)) if yes]
    exits, bomb_walls, shutters = [], [], []
    for side in SIDE_ORDER:
        index, delta = SIDES[side]
        own, other = room['doors'][index], level.neighbour(room_id, delta)
        if own == WALL or other is None:
            continue
        theirs = level.room[other]['doors'][index ^ 1]
        comment = ('  # bomb wall' if own == BOMB else '  # shutter' if own == SHUTTER
                   else f'  # open here; a shutter on 0x{other:02x}\'s side' if theirs == SHUTTER else '')
        exits.append((side, comment, other, False, own in KEYS, own == SHUTTER))
        if own == BOMB:
            bomb_walls.append(side)
        if own == SHUTTER:
            shutters.append(side)
    cellar = level.stairs_cellar(room_id)
    stairs = stairs_tile(game, room)
    if cellar is not None:
        kind = 'tunnel' if level.room[cellar]['lay'] == 0x3E else 'item cellar'
        how = ('they appear when the block is pushed' if room['secret'] == 5
               else 'in the layout' if stairs else 'not in the layout')
        exits.append(('cave', f'  # the stairs ({how}) down to the {kind} 0x{cellar:02x}', cellar, True, False, False))
    lines = [f'- level: {lv}', f"  location: '0x{room_id:02x}'" + (f"  # {'; '.join(notes)}" if notes else '')]
    if exits:
        lines.append('  exits:')
        for side, comment, to, in_cave, locked, shutter in exits:
            lines += [f'    {side}:{comment}', f'      level: {lv}', f"      location: '0x{to:02x}'"]
            lines += ['      in_cave: true'] * in_cave + ['      locked: true'] * locked + ['      shutter: true'] * shutter
    monsters = [(count, kind) for count, kind in room['mons'] if kind and kind not in NOT_MONSTERS]
    others = [f'{count} x {NOT_MONSTERS[kind]}' for count, kind in room['mons'] if kind in NOT_MONSTERS]
    people = [romdata.hx(kind) for _, kind in monsters if kind in PERSONS or kind in (0x36, 0x37)]
    remark = '; '.join(others + [f'{p}: a person' for p in people])
    if monsters:
        lines.append('  enemies:' + (f'  # {remark}' if remark else ''))
        lines += [f"    {names.get(kind, f'Unknown_{kind}')}: {count}" for count, kind in monsters]
    else:
        lines.append('  enemies: null' + (f'  # {remark}' if remark else ''))
    lines.append(f"  treasure: {TREASURES[room['item']] if room['item'] is not None else 'null'}")
    if bomb_walls:
        lines.append(f"  bomb_walls: [{', '.join(bomb_walls)}]")
    if room['secret']:
        lines.append(f"  trigger: {TRIGGERS[room['secret']]}  # attr F & 7 = {room['secret']}")
    if room['block'] and room['secret'] == 5:
        lines.append('  push_block: [N, S, E, W]  # any push reveals the stairs (trigger 5); it needs the room clear')
        stairs = BLOCK_STAIRS
    elif room['block'] and room['secret'] == 4 and shutters:
        lines.append(f'  push_block_opens: {shutters[0]}  # the shutter (trigger 4); the block needs the room clear')
    elif room['block'] and cellar is not None and stairs:
        # Stairs in the layout closed in by blocks (the diamond of 1:0x22; every such room of levels 5-9 walked on
        # Nov 13 has it): the west block is the push block.
        lines.append('  push_block: [N, S]  # the W block of the ring round the stairs: N or S opens the way, E closes it; it needs the room clear')
    if cellar is not None and stairs:
        lines.append(f'  stairs: [{stairs[0]}, {stairs[1]}]')
    if room['dark']:
        lines.append('  dark: true')
    return lines


def cellar_lines(level, cellar: int, names: dict[int, str]) -> list[str]:
    lv, room = level.lv, level.room[cellar]
    left, right = room['a'], room['b']
    lines = [f'- level: {lv}', f"  location: '0x{cellar:02x}'"]
    if left == right:
        lines += [f'  in_cave: true  # the item cellar under 0x{left:02x} (side view)', '  exits:',
                  f'    N:  # up the ladder (attr A = attr B = ${left:02X})', f'      level: {lv}',
                  f"      location: '0x{left:02x}'"]
    else:
        lines += [f'  in_cave: true  # the tunnel between 0x{left:02x} and 0x{right:02x} (side view)', '  exits:',
                  f'    E:  # up the right ladder (attr B = ${right:02X})', f'      level: {lv}',
                  f"      location: '0x{right:02x}'",
                  f'    W:  # up the left ladder (attr A = ${left:02X})', f'      level: {lv}',
                  f"      location: '0x{left:02x}'"]
    lines.append("  enemies:  # the cellar's keese (asm: not the attr C/D list)")
    lines += [f"    {names.get(kind, f'Unknown_{kind}')}: {count}" for count, kind in room['mons']]
    lines.append(f"  treasure: {TREASURES[room['item']] if room['item'] is not None else 'null'}")
    return lines


def level_text(game, lv: int, names: dict[int, str]) -> str:
    level = romdata.Level(game, lv)
    lines = [f'# --- Dungeon {lv} (level {lv}) (scripts/level_rooms.py from the ROM; zelda-asm/levels/level-{lv}.md) ---']
    for room_id in level.rooms:
        lines += room_lines(game, level, room_id, names)
    for cellar in level.cellars:
        lines += cellar_lines(level, cellar, names)
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('levels', type=int, nargs='+', choices=range(1, 10), metavar='LEVEL')
    parser.add_argument('--rom', type=Path, default=None, help="the ROM (default: triforce.paths.rom())")
    parser.add_argument('--append-to', type=Path, metavar='GAME_YAML', help='append the entries to this file')
    args = parser.parse_args(argv)
    root = ROOT / 'zelda-asm'
    if args.rom is None:
        from triforce.paths import rom  # pylint: disable=import-outside-toplevel
        args.rom = rom()
    game = romdata.Game(romdata.Source(str(root), romdata.load_prg(args.rom), str(root / 'data')))
    names = enemy_names()
    text = ''.join(level_text(game, lv, names) for lv in args.levels)
    if not args.append_to:
        sys.stdout.write(text)
        return 0
    existing = args.append_to.read_text()
    present = [lv for lv in args.levels if re.search(rf'^- level: {lv}$', existing, re.M)]
    if present:
        print(f'{args.append_to} already has rooms of level {present}', file=sys.stderr)
        return 1
    args.append_to.write_text(existing.rstrip('\n') + '\n' + text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
