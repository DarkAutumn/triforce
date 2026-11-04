"""ObservationBuilder: contract conformance and decoded values against known savestate facts."""

import glob
import os
from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest

from triforce.actions import MASK_SIZE, Action
from triforce.emulator import GAME_DIR
from triforce.observation import (ENTITY_FEATURES, ENTITY_VOCAB, HISTORY_FEATURES, ITEM_TYPE_BASE, LINK_FEATURES,
                                  NUM_ENTITY_SLOTS, OBJECTIVE_FEATURES, OBS_SCHEMA_VERSION, ROOM_STATE_FEATURES,
                                  ROOM_VOCAB, TREASURE_SLOT, ObservationBuilder, room_index)
from triforce.tiles import NUM_TILE_CLASSES, TileClass
from triforce.zelda_enums import MapLocation

from .conftest import load_and_place

R = {n: i for i, n in enumerate(ROOM_STATE_FEATURES)}
L = {n: i for i, n in enumerate(LINK_FEATURES)}
E = {n: i for i, n in enumerate(ENTITY_FEATURES)}
H = {n: i for i, n in enumerate(HISTORY_FEATURES)}
O = {n: i for i, n in enumerate(OBJECTIVE_FEATURES)}
SPACE = ObservationBuilder.observation_space()
ALL_STATES = sorted(os.path.basename(p)[:-len('.state')] for p in glob.glob(os.path.join(GAME_DIR, '*.state')))


def _reset(session, name, objective=False, **kwargs):
    builder = ObservationBuilder(objective=objective)
    state = session.load(name, randomize_rng=False, **kwargs)
    return builder, builder.reset(state, session.mask)


def _step(session, builder, command):
    change = session.act(Action.parse(command))
    return change, builder.build(change, session.mask)


def _wait_until_spawned(session, builder, slots=3, max_steps=10):
    """Pre-entry savestates load with the room's monsters just created, in their spawn clouds: WAIT until the first
    `slots` entities are out.  Returns the last (change, obs)."""
    for _ in range(max_steps):
        change, obs = _step(session, builder, 'WAIT 8')
        if not obs['entity_features'][:slots, E['spawning']].any():
            return change, obs
    raise AssertionError(f"monsters still spawning after {max_steps} WAITs")


def test_space_matches_contract():
    assert OBS_SCHEMA_VERSION >= 1 and NUM_ENTITY_SLOTS == 12
    assert set(SPACE.spaces) == {'room_id', 'room_state', 'link', 'entity_types', 'entity_features', 'grid',
                                 'history', 'objective', 'action_mask'}
    assert SPACE['room_id'].shape == (1,) and SPACE['room_id'].high.max() == ROOM_VOCAB - 1
    assert SPACE['entity_types'].high.max() == ENTITY_VOCAB - 1
    assert SPACE['grid'].shape == (22, 32) and SPACE['grid'].high.max() == NUM_TILE_CLASSES - 1
    assert SPACE['entity_features'].shape == (12, len(ENTITY_FEATURES)) and ENTITY_FEATURES[0] == 'presence'
    assert SPACE['action_mask'].shape == (MASK_SIZE,)
    for names in (ROOM_STATE_FEATURES, LINK_FEATURES, ENTITY_FEATURES, HISTORY_FEATURES, OBJECTIVE_FEATURES):
        assert len(set(names)) == len(names)


def test_every_savestate_builds_a_valid_observation(session):
    """Acceptance: every savestate in the integration dir produces a contract-conforming observation."""
    assert len(ALL_STATES) > 100
    builder = ObservationBuilder(objective=False)
    for name in ALL_STATES:
        state = session.load(name, randomize_rng=False)
        obs = builder.reset(state, session.mask)
        bad = [k for k, v in obs.items() if not SPACE[k].contains(v)]
        assert not bad, f"{name}: {bad}"
        types, feats = obs['entity_types'], obs['entity_features']
        assert np.all(feats[types == 0] == 0), f"{name}: empty slot with features"
        assert np.all(feats[types != 0, 0] == 1), f"{name}: occupied slot without presence"
        assert obs['room_id'][0] == room_index(state.level, state.in_cave, state.location)
        assert np.array_equal(obs['action_mask'], session.mask.to_flat())


def test_goriya_room_entities_and_grid(session):
    builder, obs = _reset(session, '1_44w')
    assert np.all(obs['entity_features'][:3, E['spawning']] == 1)       # just created, in their spawn clouds
    assert np.all(obs['entity_features'][:3, E['hurts_on_touch']] == 0)
    assert obs['history'][H['entered_from_W']] == 1 and obs['history'][H['has_last_action']] == 0
    _, obs = _wait_until_spawned(session, builder)
    ram = session.state.ram
    for i in range(3):          # NES slots 1-3 hold the three RedGoriyas
        assert obs['entity_types'][i] == 0x06
        f = obs['entity_features'][i]
        x, y = int(ram[0x70 + i + 1]), int(ram[0x84 + i + 1])
        assert f[E['x']] == pytest.approx(x / 255) and f[E['y']] == pytest.approx((y - 64) / 176)
        assert f[E['dx']] == pytest.approx((x - int(ram[0x70])) / 255)
        assert f[E['hurts_on_touch']] == 1 and f[E['killable']] == 1 and f[E['projectile']] == 0
        assert f[E['hp']] > 0
    assert not obs['entity_types'][3:TREASURE_SLOT].any()

    # Boomerang appears when the room is cleared: treasure slot shows kind + position, flagged hidden.
    treasure = obs['entity_features'][TREASURE_SLOT]
    assert obs['entity_types'][TREASURE_SLOT] == ITEM_TYPE_BASE + 0x1D
    assert treasure[E['hidden']] == 1 and treasure[E['is_treasure']] == 1 and treasure[E['active']] == 0
    assert treasure[E['x']] == pytest.approx(0x80 / 255) and treasure[E['y']] == pytest.approx((0x70 - 64) / 176)

    rs = obs['room_state']
    assert rs[R['door_S_bomb']] == 1 and rs[R['door_S_is_open']] == 0
    assert rs[R['door_W_key']] == 1 and rs[R['door_W_is_open']] == 1
    assert rs[R['door_N_none']] == 1 and rs[R['door_E_open']] == 1
    assert rs[R['trigger_all_dead_item']] == 1 and rs[R['item_hidden_until_clear']] == 1
    assert rs[R['enemies_remaining']] == pytest.approx(3 / 8)

    grid = obs['grid']
    assert np.all(grid[18:22, 15:17] == TileClass.BOMBABLE)
    assert np.all(grid[10:12, 0:4] == TileClass.DOOR_W) and np.all(grid[10:12, 28:32] == TileClass.DOOR_E)
    assert np.all(grid[0:4, 15:17] == TileClass.WALL)
    assert grid[8, 8] == TileClass.BLOCK and grid[4, 4] == TileClass.WALKABLE
    assert obs['history'][H['entered_from_W']] == 1
    assert builder.room_state.room == 0x44


def test_push_block_in_grid_and_room_state(session):
    state = load_and_place(session, '1_22e', {'link_x': 96, 'link_y': 125}, randomize_rng=False)
    builder = ObservationBuilder(objective=False)
    obs = builder.reset(state, session.mask)
    cells = {tuple(c) for c in np.argwhere(obs['grid'] == TileClass.PUSH_BLOCK)}
    assert cells == {(10, 12), (10, 13), (11, 12), (11, 13)}
    assert obs['grid'][10, 16] == TileClass.STAIRS
    assert obs['room_state'][R['push_block_pending']] == 1 and obs['room_state'][R['push_block_present']] == 1
    assert obs['entity_types'][10] == 0x68 and obs['entity_features'][10, E['is_tile_object']] == 1

    _step(session, builder, 'MOVE S 32')
    _, obs = _step(session, builder, 'WAIT 32')
    assert not (obs['grid'] == TileClass.PUSH_BLOCK).any()
    assert obs['room_state'][R['push_block_done']] == 1 and obs['room_state'][R['push_block_pending']] == 0


def test_start_state_link_and_room(session):
    _, obs = _reset(session, 'start')
    assert obs['room_id'][0] == room_index(0, False, 0x77)
    link = obs['link']
    assert link[L['sword_0']] == 1 and link[L['beams_ready']] == 0
    assert link[L['x']] == pytest.approx(120 / 255) and link[L['y']] == pytest.approx((141 - 64) / 176)
    assert link[L['hp_fraction']] == pytest.approx(session.state.link.health / session.state.link.max_health)
    assert link[L['max_hp']] == pytest.approx(3 / 16)
    assert obs['room_state'][R['overworld']] == 1 and not obs['entity_types'].any()
    assert (obs['grid'] == TileClass.CAVE).sum() == 4       # $F3 pair over $24 pair at the sword cave mouth


def test_key_room_treasure(session):
    _, obs = _reset(session, '1_72e')
    assert obs['entity_types'][TREASURE_SLOT] == ITEM_TYPE_BASE + 0x19
    assert obs['room_state'][R['item_is_key']] == 1 and obs['room_state'][R['item_visible']] == 0
    assert obs['room_state'][R['enemies_unspawned']] == pytest.approx(3 / 8)
    spawning = obs['entity_features'][:3, E['spawning']]
    assert np.all(spawning == 1) and np.all(obs['entity_features'][:3, E['hurts_on_touch']] == 0)


def test_sword_cave_exposes_ware_and_fires(session):
    _, obs = _reset(session, '0_77cave_0')
    types, feats = obs['entity_types'], obs['entity_features']
    assert list(types[:3]) == [0x6A, 0x40, 0x40]
    assert feats[0, E['hurts_on_touch']] == 0 and np.all(feats[1:3, E['hurts_on_touch']] == 1)
    assert not feats[:3, E['killable']].any()
    assert types[TREASURE_SLOT] == ITEM_TYPE_BASE + 0x01
    assert feats[TREASURE_SLOT, E['x']] == pytest.approx(0x78 / 255)
    assert feats[TREASURE_SLOT, E['y']] == pytest.approx((0x98 - 64) / 176)
    assert obs['room_state'][R['in_cave']] == 1 and obs['room_state'][R['cave_ware_1']] == 1


def test_velocity_is_position_delta(session):
    builder, obs = _reset(session, '1_44w')
    assert not obs['entity_features'][:, [E['vx'], E['vy']]].any()
    _, obs = _wait_until_spawned(session, builder)                  # monsters don't move inside spawn clouds
    before = obs['entity_features'][:3, [E['x'], E['y']]].copy()
    change, obs = _step(session, builder, 'WAIT 16')
    after = obs['entity_features'][:3, [E['x'], E['y']]]
    velocity = obs['entity_features'][:3, [E['vx'], E['vy']]]
    assert np.abs(velocity).sum() > 0
    moved_px = (after - before) * np.array([255, 176])
    np.testing.assert_allclose(velocity, np.clip(moved_px / (change.frames * 4.0), -1, 1), atol=1e-5)

    hist = obs['history']
    assert hist[H['kind_WAIT']] == 1 and hist[H['duration_16']] == 1 and hist[H['has_last_action']] == 1
    assert not hist[[H[f'dir_{d}'] for d in ('N', 'S', 'W', 'E', 'NW', 'NE', 'SW', 'SE')]].any()
    assert hist[H['frames']] == pytest.approx(min(change.frames / 64, 1))


def test_room_change_resets_velocity_and_records_entry(session):
    builder, obs = _reset(session, 'start')
    for _ in range(30):
        change, obs = _step(session, builder, 'MOVE E 32')
        if change.changed_location:
            break
    assert session.state.location == 0x78
    assert obs['history'][H['room_changed']] == 1 and obs['history'][H['entered_from_W']] == 1
    assert not obs['entity_features'][:, [E['vx'], E['vy']]].any()
    assert obs['history'][H['dir_E']] == 1 and obs['history'][H['kind_MOVE']] == 1


class _Kind(Enum):
    NONE = 0
    MOVE = 1


def test_objective_block(session):
    state = session.load('1_44w', randomize_rng=False)
    state.objectives = SimpleNamespace(kind=_Kind.MOVE, next_rooms={MapLocation(1, 0x45, False)})

    off = ObservationBuilder(objective=False).reset(state, session.mask)
    assert not off['objective'].any()

    on = ObservationBuilder(objective=True).reset(state, session.mask)
    assert on['objective'][O['next_room_E']] == 1 and on['objective'][O['objective_MOVE']] == 1
    assert on['objective'].sum() == 2
    for key in off:
        if key != 'objective':
            assert np.array_equal(off[key], on[key])


@pytest.mark.parametrize('value,phase', [(0x17, 'outbound'), (0x28, 'spark'), (0x37, 'slow_outward'),
                                       (0x47, 'slow_return'), (0x57, 'fast_return')])
def test_boomerang_animation_bits_preserve_tactical_phase(session, value, phase):
    state = session.load('2_7ds_2', randomize_rng=False)
    state = session.apply_ram({'bait_or_boomerang_animation': value})
    obs = ObservationBuilder(objective=False).reset(state, session.mask)
    phases = [L[f'boomerang_{name}'] for name in ('outbound', 'spark', 'slow_outward',
                                               'slow_return', 'fast_return')]
    assert obs['link'][L['boomerang_active']] == 1
    assert obs['link'][L[f'boomerang_{phase}']] == 1
    assert obs['link'][phases].sum() == 1


def test_bait_does_not_expose_stale_boomerang_position(session):
    builder, _ = _reset(session, '2_7ds_2')
    _step(session, builder, 'MOVE N 16')
    _step(session, builder, 'MOVE N 16')
    _, flight = _step(session, builder, 'BOOMERANG E')
    assert flight['link'][L['boomerang_active']] == 1
    assert flight['link'][L['boomerang_x']] > 0
    state = session.apply_ram({'bait_or_boomerang_animation': 0x80})
    bait = builder.reset(state, session.mask)['link']
    assert bait[L['boomerang_active']] == 0
    assert bait[L['boomerang_x']] == bait[L['boomerang_y']] == 0
    assert not bait[[L[f'boomerang_{name}'] for name in ('outbound', 'spark', 'slow_outward',
                                                     'slow_return', 'fast_return')]].any()


@pytest.mark.parametrize('kind,state_byte,stun,timer,harm', [
    (0x06, 0, 0x10, 0, False),       # stunned Goriya remains present/active, but harmless
    (0x5C, 0x10, 0, 0, True),       # outbound Goriya boomerang
    (0x5C, 0x30, 0, 0, False),      # slow-outward is not return, but cannot hit Link
    (0x5C, 0x50, 0, 0, False),      # fast-return boomerang
    (0x53, 0x30, 0, 0, False),      # shield-bounced rock
    (0x5B, 0x10, 0, 4, False),      # delayed arrow only draws/checks shooter
    (0x55, 0x10, 0, 4, True),       # delayed fireball still checks collision
    (0x49, 2, 0, 0, True),          # returning trap uses different state semantics
])
def test_contact_hazard_preserves_presence_and_kind_specific_gates(session, kind, state_byte, stun, timer, harm):
    state = session.load('1_44w', randomize_rng=False)
    # Decode a single native object-table boundary without advancing its next update.
    ram = state.ram
    ram[0x350], ram[0xAD], ram[0x406] = kind, state_byte, 0
    ram[0x3E], ram[0x29], ram[0x66C] = stun, timer, 0
    obs = ObservationBuilder(objective=False).reset(state, session.mask)
    entity = obs['entity_features'][0]
    assert obs['entity_types'][0] == kind
    assert entity[E['presence']] == entity[E['active']] == 1
    assert bool(entity[E['hurts_on_touch']]) is harm
