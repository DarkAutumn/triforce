"""game.yaml's dungeon exits follow one convention and agree with the ROM's room attributes (review F030).

Doors come from attr A ($687E: N = bits 7-5, S = bits 4-2) and attr B ($68FE: W = bits 7-5, E = bits 4-2) of the
loaded level (Z_05 FindDoorTypeByDoorBit; 0 open, 1 wall, 4 bomb wall, 5/6 key, 7 shutter).  Convention (game.yaml
"Exit conventions"): every door is an exit on both sides, a key door is `locked` on both sides, a bomb wall is an
exit listed in `bomb_walls`, and the entrance room's S door back to the overworld is no exit.
"""

import pytest
import yaml

from triforce import _native as native
from triforce.paths import PACKAGE_DIR

from .conftest import REPO_STATES_DIR, require_state

# Dungeon 4's entrance state is a scouting state (triforce/states/d4: the route d2first-d4, not trained yet).
ENTRANCES = {1: ('1_73s_0', 0x73), 2: ('2_7ds_0', 0x7d), 3: ('3_7cs_0', 0x7c), 4: ('d4/4_71s_0', 0x71)}
OPPOSITE = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
DOOR = {0: 'open', 1: 'wall', 2: 'open', 3: 'open', 4: 'bomb', 5: 'key', 6: 'key', 7: 'shutter'}


def rooms():
    data = yaml.safe_load((PACKAGE_DIR / 'game.yaml').read_text(encoding='utf-8'))
    return {(r['level'], int(r['location'], 16), bool(r.get('in_cave'))): r for r in data['rooms']}


def test_every_exit_has_its_way_back():
    by_key = rooms()
    for (level, room, in_cave), entry in by_key.items():
        if in_cave or entry.get('maze'):
            continue
        for side, exit in (entry['exits'] or {}).items():
            if side not in OPPOSITE:
                continue
            target = (exit['level'], int(exit['location'], 16), bool(exit.get('in_cave')))
            assert target in by_key, f"{level}:{room:#04x} {side} -> {target}: not in game.yaml"
            other = by_key[target]
            if other.get('maze'):
                continue
            back = (other['exits'] or {}).get(OPPOSITE[side])
            assert back is not None, f"{level}:{room:#04x} {side} -> {target} has no way back"
            assert bool(back.get('locked')) == bool(exit.get('locked')), f"{level}:{room:#04x} {side}: locked"
            walls, other_walls = entry.get('bomb_walls') or [], other.get('bomb_walls') or []
            assert (side in walls) == (OPPOSITE[side] in other_walls), f"{level}:{room:#04x} {side}: bomb wall"


@pytest.mark.parametrize('level', sorted(ENTRANCES))
def test_dungeon_exits_match_the_roms_doors(level):
    state, entrance = ENTRANCES[level]
    require_state(state)
    game = native.Game(seed=0, states_dir=str(REPO_STATES_DIR))
    game.load(state, fresh_room=True, seed=0)
    assert (game.read(0x12), game.read(0x10)) == (5, level)
    attr_a = [game.read(0x687E + i) for i in range(0x80)]
    attr_b = [game.read(0x68FE + i) for i in range(0x80)]
    checked = 0
    for (lvl, room, in_cave), entry in rooms().items():
        if lvl != level or in_cave:
            continue
        a, b = attr_a[room], attr_b[room]
        doors = {'N': DOOR[a >> 5], 'S': DOOR[(a >> 2) & 7], 'W': DOOR[b >> 5], 'E': DOOR[(b >> 2) & 7]}
        exits, walls = entry['exits'] or {}, entry.get('bomb_walls') or []
        for side, door in doors.items():
            where = f"{level}:{room:#04x} {side} (ROM {door})"
            exit = exits.get(side)
            if door == 'wall' or (room == entrance and side == 'S'):
                assert exit is None, where
                continue
            assert exit is not None, where
            assert bool(exit.get('locked')) == (door == 'key'), where
            assert (side in walls) == (door == 'bomb'), where
            checked += 1
    assert checked > 20
