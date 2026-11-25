"""Round 43: the natural entry draw (`natural_entries`, off by default): a start at a door takes the spawn cycle, edge
cell and chase state that power-on runs of the route brought there (`triforce/entry_draws.d3first.json`,
`zelda::room_entry::EntryDraws`), not the uniform spawn draw, which gives most doors layouts real play never shows."""

import json
from pathlib import Path

from triforce import _native, emu

from .conftest import REPO_STATES_DIR, require_rom

TABLE = Path(__file__).resolve().parents[1] / 'triforce' / 'entry_draws.d3first.json'
SPAWN_CYCLE = 0x524
UP = 8                                  # Link's facing on the frame before an entry from the south


def real_cycles(level: int, room: int, facing: int) -> set:
    doors = json.loads(TABLE.read_text())['doors']
    return {e[0] for d in doors if (d['level'], d['room'], d['facing']) == (level, room, facing) for e in d['entries']}


def cycle_after(state: str, cycle: int) -> int:
    """`$524` after the game entered the room from `state` with spawn cycle `cycle` (asm:Spawn_AssignPositions
    stores where its walk ended: one value per layout)."""
    game = _native.Game(states_dir=str(REPO_STATES_DIR))
    game.load(state, randomize_rng=False, controllable=True)
    game.write(SPAWN_CYCLE, cycle)
    game.press(0, 1)
    return game.read(SPAWN_CYCLE)


def test_the_table_holds_only_values_the_game_holds():
    table = json.loads(TABLE.read_text())
    assert table['route'] == 'd3first' and table['doors']
    for door in table['doors']:
        assert 1 <= door['level'] <= 9 or door['level'] == 0
        assert door['facing'] in (1, 2, 4, 8) and door['entries']
        for spawn_cycle, edge_cell, chase_flag, chase_x, chase_y, chase_timer, timer_cycle in door['entries']:
            assert 0 <= spawn_cycle <= 8 and chase_flag in (0, 1) and 0 <= timer_cycle <= 9 and 0 <= chase_timer <= 7
            assert all(0 <= b <= 0xFF for b in (edge_cell, chase_x, chase_y))


def test_natural_entries_place_5x47_as_power_on_runs_do_and_the_uniform_draw_does_not():
    require_rom()
    spec = emu.Spec(5, 0x47)
    state = emu.find_start(spec, REPO_STATES_DIR).state
    real = {cycle_after(state, c) for c in real_cycles(5, 0x47, UP)}
    assert real, "5x47 from S has real entries in the table"
    natural = emu.Session(spec, seed=0, states_dir=REPO_STATES_DIR, natural_entries=True)
    uniform = emu.Session(spec, seed=0, states_dir=REPO_STATES_DIR)
    seen_natural, seen_uniform = set(), set()
    for seed in range(24):
        natural.reset(seed)
        uniform.reset(seed)
        seen_natural.add(natural.state.read(SPAWN_CYCLE))
        seen_uniform.add(uniform.state.read(SPAWN_CYCLE))
    assert seen_natural <= real
    assert seen_uniform - real, "the uniform draw shows layouts real play does not bring to 5x47"
    assert natural.meta()['human']['natural_entries'] is True
    assert '_natural_' in natural.file_name() and '_natural_' not in uniform.file_name()


def test_a_door_the_table_lacks_keeps_the_uniform_draw():
    require_rom()
    # A cave entry (game mode $0B) is no door of the table: both loads draw the same way and give the same machine.
    natural = _native.LiveGame(route='d3first', states_dir=str(REPO_STATES_DIR), natural_entries=True)
    uniform = _native.LiveGame(route='d3first', states_dir=str(REPO_STATES_DIR))
    a = natural.load('d3first_0_77cave_0', 5)
    b = uniform.load('d3first_0_77cave_0', 5)
    assert a.ram == b.ram
