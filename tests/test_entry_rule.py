"""Round 43: the reset rule (`zelda::room_entry::EntryRule`, `triforce/entry_rule.d3first.json`): a start at a door
writes only values some play of the game can bring there: `$524` the route's planned value on half the resets and
the door's other reachable values evenly on the rest (training, evaluation, a chain's first room), the planned value
alone in the emulator tool by default; the chase flag 50/50, the divider, `$4A` and `$525` from the door's real
values.  The uniform draw it replaced gave most doors layouts no game shows there."""

import json
from pathlib import Path

from triforce import _native, emu, start_audit
from triforce.specialist import load_specialist

from .conftest import REPO_STATES_DIR, require_rom

TABLE = Path(__file__).resolve().parents[1] / 'triforce' / 'entry_rule.d3first.json'
SPAWN_CYCLE = 0x524
GIBDOS = 'd3first_5_47s_0'             # 5x47 from S: Link faces up, on the bottom edge
DOOR_5X47 = (5, 0x47, 8, False)


def door(key) -> dict:
    return next(d for d in json.loads(TABLE.read_text())['doors']
                if (d['level'], d['room'], d['facing'], d['inside']) == key)


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
    assert table['route'] == 'd3first' and table['planned_share'] == 0.5 and table['doors']
    keys = set()
    for d in table['doors']:
        key = (d['level'], d['room'], d['facing'], d['inside'])
        assert key not in keys
        keys.add(key)
        assert d['facing'] in (1, 2, 4, 8) and d['proof'] in ('maps', 'measured')
        assert 0 <= d['planned'] <= 8 and all(0 <= v <= 8 for v in d['others']) and d['planned'] not in d['others']
        assert all(0 <= t <= 7 for t in d['chase_timers'])
        assert all(0 <= b <= 0xFF for b in d['edge_cells'])
    gibdos = door(DOOR_5X47)
    assert gibdos['planned'] == 6 and gibdos['proof'] == 'maps'


def test_the_emulator_tool_serves_the_routes_layout_and_any_valid_one_on_request():
    require_rom()
    rule = door(DOOR_5X47)
    planned = cycle_after(GIBDOS, rule['planned'])
    valid = {cycle_after(GIBDOS, c) for c in [rule['planned'], *rule['others']]}
    session = emu.Session(emu.Spec(5, 0x47), seed=0, states_dir=REPO_STATES_DIR)
    live = _native.LiveGame(route='d3first', states_dir=str(REPO_STATES_DIR))
    route_seen, valid_seen = set(), set()
    for seed in range(24):
        session.reset(seed)
        route_seen.add(session.state.read(SPAWN_CYCLE))
        valid_seen.add(live.load(GIBDOS, seed, layout='valid').read(SPAWN_CYCLE))
    assert route_seen == {planned}
    assert valid_seen <= valid and len(valid_seen) > 1


def test_a_training_reset_draws_the_planned_layout_half_the_time_and_only_valid_ones():
    require_rom()
    rule = door(DOOR_5X47)
    planned = cycle_after(GIBDOS, rule['planned'])
    valid = {cycle_after(GIBDOS, c) for c in [rule['planned'], *rule['others']]}
    fields = load_specialist('gibdo').to_dict()
    fields['curriculum']['enabled'] = False
    config = _native.SpecialistConfig.from_json(json.dumps(fields))
    env = _native.Env(config, start_state=GIBDOS, states_dir=str(REPO_STATES_DIR))
    seen = []
    for seed in range(200):
        env.reset(seed=seed)
        seen.append(env.state.read(SPAWN_CYCLE))
    assert set(seen) <= valid
    assert 0.4 <= seen.count(planned) / len(seen) <= 0.6


def test_a_start_the_rule_has_no_door_for_keeps_the_uniform_draw():
    require_rom()
    # A cave entry (game mode $0B) is no door of the table: both layouts load the same machine.
    live = _native.LiveGame(route='d3first', states_dir=str(REPO_STATES_DIR))
    a = live.load('d3first_0_77cave_0', 5, layout='route').ram
    b = live.load('d3first_0_77cave_0', 5, layout='valid').ram
    assert a == b


def test_the_start_audit_reads_a_seed_state_against_the_rule():
    require_rom()
    game = _native.Game(states_dir=str(REPO_STATES_DIR))
    row = start_audit.audit_state(game, str(REPO_STATES_DIR / f'{GIBDOS}.state'), start_audit.load_rule('d3first'))
    assert row['door'] == list(DOOR_5X47) and row['planned'] == 6 and not row['problems']
    cave = start_audit.audit_state(game, str(REPO_STATES_DIR / 'd3first_0_77cave_0.state'), {})
    assert cave['cave'] and not cave['problems']
