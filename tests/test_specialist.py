"""Specialist configs, start-state discovery/verification, the probabilistic selector, RAM randomization and
savestate capture."""

import gzip
import os
import random
import shutil

import pytest
import yaml
from pydantic import ValidationError

from triforce.actions import Action
from triforce.emulator import GAME_DIR
from triforce.game_map import ROUTES_YAML, GameMap
from triforce.room_entry import is_pre_entry
from triforce.specialist import (RamRandomization, SpecialistConfig, format_room, list_specialists, load_specialist,
                                 parse_room)
from triforce.start_states import (DAMAGE_WEIGHT, DECAY_FACTOR, FAILURE_WEIGHT, ProbabilisticSelector,
                                   SavestateCapture, StartStatePool, entry_side, ram_randomization_values)
from triforce.zelda_enums import MapLocation

from .conftest import act_until


@pytest.fixture(scope='module')
def routes():
    with open(ROUTES_YAML, encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope='module')
def game_map():
    return GameMap.load()


def minimal(**update):
    data = {'name': 't', 'rooms': ['(1, 0x73)'], 'success': ['exit-correct']}
    data.update(update)
    return data


# --- configs ---

def test_every_family_has_a_config(routes):
    assert list_specialists() == sorted(routes['families'])


@pytest.mark.parametrize('name', list_specialists())
def test_config_rooms_match_game_yaml_tags_and_routes(name, routes, game_map):
    config = load_specialist(name)
    assert config.name == name
    assert set(config.rooms) == set(game_map.rooms_for_specialist(name)), "rooms must be tagged in game.yaml"
    assert {format_room(r) for r in config.rooms} == {format_room(parse_room(r)) for r in routes['families'][name]}
    for room in config.rooms:
        assert game_map[room].specialist == name


def test_every_route_room_belongs_to_its_family(routes):
    for step in routes['milestone1']:
        location = MapLocation(step['level'], int(step['room'], 16), bool(step.get('in_cave')))
        assert location in load_specialist(step['specialist']).rooms


def test_parse_room_forms():
    assert parse_room('(1, 0x73)') == MapLocation(1, 0x73, False)
    assert parse_room('(0, 0x77, cave)') == MapLocation(0, 0x77, True)
    assert parse_room([2, '0x0e']) == MapLocation(2, 0x0e, False)
    with pytest.raises(ValueError):
        parse_room('(1; 0x73)')


@pytest.mark.parametrize('update', [
    {'success': ['exit-sideways']},
    {'success': []},
    {'endings': {'exit-wrong': 'meh'}},
    {'ram_randomization': {'hp': [0.0, 'max']}},
    {'ram_randomization': {'hp': [0.9, 0.5]}},
    {'ram_randomization': {'keys': [3, 1]}},
    {'ram_randomization': {'rupees': [0, 1]}},
    {'max_steps': 0},
    {'rooms': []},
    {'colour': 'blue'},
])
def test_invalid_configs_are_rejected(update):
    with pytest.raises(ValidationError):
        SpecialistConfig(**minimal(**update))


def test_yaml_on_off_and_max_parse():
    config = SpecialistConfig(**yaml.safe_load(
        "name: t\nrooms: ['(1, 0x73)']\nsuccess: exit-correct\nreward: {pbrs: off}\nobs: {objective: on}\n"
        "ram_randomization: {hp: [0.5, max], bombs: [0, 4]}\n"))
    assert config.reward.pbrs is False and config.obs.objective is True
    assert config.success == ['exit-correct'] and config.ram_randomization.hp == (0.5, 'max')


def test_trainable_specialists_have_start_states():
    for name in list_specialists():
        config = load_specialist(name)
        if config.trainable and name != 'd2-goriya':          # d2-goriya: seed states still missing (route.md)
            assert config.resolve_start_states(), name


def test_start_states_load_into_their_rooms(session):
    """RAM decides: every seed state a config resolves must load into one of its rooms."""
    for name in list_specialists():
        config = load_specialist(name)
        for path in config.resolve_start_states():
            state = session.load(path, randomize_rng=False)
            assert state.full_location in config.rooms, f"{name}: {os.path.basename(path)} -> {state.full_location}"


# --- probabilistic selector ---

def test_selector_round_robins_until_memory_covers_every_state():
    selector = ProbabilisticSelector(['a', 'b', 'c'], random.Random(0), memory_size=6)
    assert [selector.next() for _ in range(4)] == ['a', 'b', 'c', 'a']
    for _ in range(6):
        selector.record('a', True, False)
    assert selector.next() == 'b'                       # memory full but b and c never seen: still round-robin


def test_selector_weights_failures_damage_and_recency():
    selector = ProbabilisticSelector(['a', 'b'], random.Random(0), memory_size=4)
    selector.record('a', True, False)
    selector.record('b', False, True)
    selector.record('a', True, False)
    selector.record('b', True, False)
    weights = selector.weights()
    assert weights['b'] == pytest.approx(1.0 + DECAY_FACTOR ** 2 * FAILURE_WEIGHT * DAMAGE_WEIGHT)
    assert weights['a'] == pytest.approx(DECAY_FACTOR + DECAY_FACTOR ** 3)


def test_selector_samples_failing_states_more():
    selector = ProbabilisticSelector(['easy', 'hard'], random.Random(0), memory_size=128)
    for i in range(128):
        selector.record('easy' if i % 2 else 'hard', i % 2 == 1, False)
    picks = [selector.next() for _ in range(2000)]
    assert picks.count('hard') > 1.6 * picks.count('easy')


def test_pool_mixes_harvested_states(tmp_path):
    shutil.copy(os.path.join(GAME_DIR, '1_73n.state'), tmp_path / 'h1.state')
    config = load_specialist('d1-melee').model_copy(update={'harvested_states': str(tmp_path)})
    pool = StartStatePool(config, random.Random(0))
    picks = [pool.next() for _ in range(400)]
    harvested = sum(p.endswith('h1.state') for p in picks)
    assert 150 < harvested < 250                       # 50/50 by default
    only = StartStatePool(config.model_copy(update={'harvested_fraction': 1.0}), random.Random(0))
    assert all(only.next().endswith('h1.state') for _ in range(20))


def test_pool_without_any_states_raises(tmp_path):
    from triforce.start_states import StartStateError   # pylint: disable=import-outside-toplevel
    config = load_specialist('boss-dodongo').model_copy(update={'harvested_states': str(tmp_path)})
    with pytest.raises(StartStateError):
        StartStatePool(config)


# --- RAM randomization ---

@pytest.mark.parametrize('hp,max_health', [((0.5, 'max'), 3), ((1.0, 'max'), 16), ((0.01, 0.01), 3)])
def test_hp_randomization_bounds(session, hp, max_health):
    rng = random.Random(0)
    spec = RamRandomization(hp=hp, bombs=(1, 3), keys=(2, 2))
    lo = 1.0 if hp[0] == 'max' else hp[0]
    for _ in range(10):
        values = ram_randomization_values(spec, max_health, rng)
        session.load('start', randomize_rng=False, ram_overrides={'hearts_and_containers': (max_health - 1) << 4})
        state = session.apply_ram(values)
        assert state.link.max_health == max_health
        assert max(0.5, round(lo * 2 * max_health) / 2) <= state.link.health <= max_health
        assert 1 <= state.link.bombs <= 3 and state.link.keys == 2
        if state.link.health == max_health:
            assert state.link.is_health_full            # beams need the NES full-health encoding


def test_full_hp_share_splits_resets_between_beams_and_no_beams(session):
    """full_hp: half the resets at full health (the sword fires beams), the rest strictly below full."""
    rng = random.Random(0)
    spec = RamRandomization(hp=(0.5, 'max'), full_hp=0.5)
    draws = [ram_randomization_values(spec, 3, rng) for _ in range(400)]
    full = [v for v in draws if v['partial_hearts'] == 0xFF]
    assert 0.4 < len(full) / len(draws) < 0.6
    for values in (full[0], next(v for v in draws if v['partial_hearts'] != 0xFF)):
        session.load('start', randomize_rng=False, ram_overrides={'hearts_and_containers': 0x20, 'sword': 1})
        link = session.apply_ram(values).link
        assert link.is_health_full == (values in full) and 1.5 <= link.health <= 3.0
        assert link.are_beams_available == link.is_health_full


# --- savestate capture ---

def test_entry_side_letters():
    assert entry_side(MapLocation(0, 0x77, False), MapLocation(0, 0x67, False)) == 's'
    assert entry_side(MapLocation(0, 0x77, False), MapLocation(0, 0x78, False)) == 'w'
    assert entry_side(MapLocation(0, 0x37, False), MapLocation(1, 0x73, False)) == 'c'
    assert entry_side(MapLocation(0, 0x77, True), MapLocation(0, 0x77, False)) == 'c'


def test_capture_saves_first_entry_once_before_monsters_spawn(session, tmp_path):
    capture = SavestateCapture(str(tmp_path))
    session.load('start', randomize_rng=False)
    session.executor.capture_pre_entry = True
    try:
        changes = act_until(session, 'MOVE E 32', lambda c: c.changed_location)
    finally:
        session.executor.capture_pre_entry = False
    path = capture.on_step(changes[-1], session)
    assert os.path.basename(path) == '0_78w.state'
    assert capture.on_step(changes[-1], session) is None          # already exists
    with gzip.open(path) as f:
        session.emu.load_state(f.read())
    assert is_pre_entry(session.emu.read)                           # saved before the room's monsters exist
    state = session.load(path, randomize_rng=False)
    assert state.full_location == MapLocation(0, 0x78, False)
    assert capture.on_step(session.act(Action.parse('MOVE E 8')), session) is None   # no room change
