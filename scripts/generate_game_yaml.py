#!/usr/bin/env python3
"""Generate triforce/game.yaml from per-savestate YAML files.

Reads all *.yaml files in triforce/custom_integrations/Zelda-NES/, groups
them by (level, location, in_cave), and merges into a single game.yaml
with one entry per room. Enemy counts use the max across savestates,
treasure uses the first non-null value, and exits are unioned.
"""

import glob
import os

import yaml


STATE_DIR = os.path.join(os.path.dirname(__file__), '..', 'triforce', 'custom_integrations', 'Zelda-NES')
OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'triforce', 'game.yaml')


def main():
    from collections import defaultdict

    rooms = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(STATE_DIR, '*.yaml'))):
        with open(f, encoding='utf-8') as fh:
            d = yaml.safe_load(fh)
        key = (d['level'], d['location'], d.get('in_cave', False))
        rooms[key].append(d)

    game_rooms = []
    for (level, location, in_cave), entries in sorted(rooms.items()):
        # Merge exits: union across all savestates
        merged_exits = {}
        for entry in entries:
            for direction, dest in (entry.get('exits') or {}).items():
                if direction not in merged_exits:
                    merged_exits[direction] = dest

        # Merge enemies: max count of each type
        merged_enemies = {}
        for entry in entries:
            for etype, count in (entry.get('enemies') or {}).items():
                merged_enemies[etype] = max(merged_enemies.get(etype, 0), count)

        # Merge treasure: first non-null value
        merged_treasure = None
        for entry in entries:
            t = entry.get('treasure')
            if t is not None:
                merged_treasure = t
                break

        room = {'level': level, 'location': location}
        if in_cave:
            room['in_cave'] = True
        room['exits'] = merged_exits if merged_exits else None
        room['enemies'] = merged_enemies if merged_enemies else None
        room['treasure'] = merged_treasure

        game_rooms.append(room)

    game = {'rooms': game_rooms}
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        yaml.dump(game, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f'Wrote {len(game_rooms)} rooms to {OUTPUT}')


if __name__ == '__main__':
    main()
