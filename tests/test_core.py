"""Emulator, object model, action schema, option executor and session behavior on real savestates."""

from types import SimpleNamespace

import numpy as np
import pytest

from triforce.actions import (ACTION_SCHEMA_VERSION, MASK_SIZE, Action, ActionMask, DIRECTIONS, DURATIONS,
                              compute_action_mask)
from triforce.frame_skip import Interrupt, OptionExecutor
from triforce.projectile_blocks import ProjectileBlock, ShieldCollisionWatch
from triforce.zelda_enums import (ActionKind, AnimationState, Direction, MapLocation, SwordKind,
                                ZeldaAnimationKind, ZeldaEnemyKind)

from .conftest import act_until, full_health, load_and_place


# --- action schema (no emulator) ---

def test_action_indices_round_trip():
    for action in [Action(ActionKind.MOVE, Direction.W, 32), Action(ActionKind.WAIT, duration=4),
                   Action(ActionKind.BOOMERANG, Direction.SE), Action(ActionKind.BOMB, Direction.N)]:
        assert Action.from_indices(*action.to_indices()) == action


def test_from_indices_ignores_heads_that_do_not_apply():
    assert Action.from_indices(ActionKind.WAIT, 7, 2) == Action(ActionKind.WAIT, Direction.NONE, 16)
    assert Action.from_indices(ActionKind.SWORD, 1, 3) == Action(ActionKind.SWORD, Direction.S, 0)


@pytest.mark.parametrize('text', ['MOVE NW 8', 'SWORD SE', 'MOVE N 5', 'WAIT 12'])
def test_invalid_actions_are_rejected(text):
    with pytest.raises(AssertionError):
        Action.parse(text)


def test_mask_flat_round_trip():
    mask = ActionMask()
    mask.allow(ActionKind.MOVE, Direction.E)
    mask.allow(ActionKind.WAIT)
    mask.allow(ActionKind.BOOMERANG, Direction.NW)
    flat = mask.to_flat()
    assert flat.shape == (MASK_SIZE,) and flat.dtype == bool
    again = ActionMask.from_flat(flat)
    assert again.is_allowed(Action(ActionKind.BOOMERANG, Direction.NW))
    assert again.is_allowed(Action(ActionKind.WAIT, duration=32))
    assert not again.is_allowed(Action(ActionKind.MOVE, Direction.W, 8))
    assert ACTION_SCHEMA_VERSION >= 1 and len(DIRECTIONS) == 8 and DURATIONS == (4, 8, 16, 32)


# --- savestate decoding ---

def test_decodes_dungeon_savestate(session):
    state = session.load('1_44w', randomize_rng=False)
    assert (state.level, state.location, state.in_cave) == (1, 0x44, False)
    assert state.link.sword == SwordKind.WOOD
    assert state.link.bombs == 4
    goriyas = [e for e in state.enemies if e.id == ZeldaEnemyKind.RedGoriya]
    assert len(goriyas) == 3
    assert state.bomb_walls and state.bomb_walls[0].wall_direction == Direction.S


def test_decodes_overworld_start(session):
    state = session.load('start', randomize_rng=False)
    assert (state.level, state.location) == (0, 0x77)
    assert state.link.sword == SwordKind.NONE
    assert state.link.max_health == 3


def test_north_shutter_is_barred_not_a_wall(session):
    state = session.load('2_2es_0', randomize_rng=False, fresh_room=True)
    assert state.is_door_barred(Direction.N)
    assert not state.is_door_open(Direction.N)
    assert not state.room.is_wall_intact(Direction.N, state.current_tiles)


def test_boomerang_stays_active_on_return_until_caught(session):
    session.load('1_35s_0', randomize_rng=False, fresh_room=True)
    act_until(session, 'MOVE N 32', lambda c: session.emu.read(0x53) == 0)
    session.act(Action.parse('BOOMERANG E'))
    change = act_until(session, 'WAIT 4', lambda c: 0x40 <= session.emu.read(0xBB) <= 0x57)[-1]
    assert change.state.link.get_animation_state(ZeldaAnimationKind.BOOMERANG) == AnimationState.ACTIVE
    change = act_until(session, 'WAIT 4', lambda c: session.emu.read(0xBB) == 0)[-1]
    assert change.state.link.get_animation_state(ZeldaAnimationKind.BOOMERANG) == AnimationState.INACTIVE
    bait = session.apply_ram({'bait_or_boomerang_animation': 0x80})
    assert bait.link.get_animation_state(ZeldaAnimationKind.BOOMERANG) == AnimationState.INACTIVE


# --- masks ---

# Converted savestates load with full hearts (beams); these give 3 containers with 2 hearts, so the sword is offered.
HURT = {'hearts_and_containers': 0x21, 'partial_hearts': 0xFF}


def leave_west_door(session):
    """From 1_44w (pre-entry, Link on the W door tile): MOVE E until Link is off the door and the Goriyas have left
    their spawn clouds."""
    act_until(session, 'MOVE E 16', lambda c: c.state.link.position.x >= 32 and c.state.active_enemies)


def test_doorway_restricts_moves_and_weapons(session):
    session.load('1_44w', randomize_rng=False, ram_overrides=HURT)
    mask = session.mask
    assert mask.directions(ActionKind.MOVE) == [Direction.W, Direction.E]
    assert not mask.kind[ActionKind.SWORD]      # next to the wall a weapon press would leave the room
    leave_west_door(session)
    assert session.mask.kind[ActionKind.SWORD] and session.mask.kind[ActionKind.BOMB]


def test_sword_and_beams_are_exclusive(session):
    session.load('1_44w', randomize_rng=False, ram_overrides=HURT)
    leave_west_door(session)
    assert session.mask.kind[ActionKind.SWORD] and not session.mask.kind[ActionKind.BEAMS]

    state = session.load('1_44w', randomize_rng=False, ram_overrides=full_health(session, 16))
    leave_west_door(session)
    assert state.link.sword == SwordKind.WOOD
    assert session.mask.kind[ActionKind.BEAMS] and not session.mask.kind[ActionKind.SWORD]


def test_no_weapons_in_empty_room(session):
    session.load('1_73s_0', randomize_rng=False)
    session.act(Action.parse('MOVE N 16'))      # off the S door tile, where weapons are masked anyway
    mask = session.mask
    assert mask.kind[ActionKind.MOVE] and mask.kind[ActionKind.WAIT]
    assert not mask.kind[ActionKind.SWORD] and not mask.kind[ActionKind.BOMB]


def place_at_push_block(session):
    """Link standing just N of 1_22e's push block, after the room has created it."""
    return load_and_place(session, '1_22e', {'link_x': 96, 'link_y': 125}, randomize_rng=False)


def test_push_block_move_is_legal(session):
    state = place_at_push_block(session)
    assert state.push_block_pending
    assert state.is_push_block_direction(Direction.S)
    assert not state.can_link_move(Direction.S)
    assert compute_action_mask(state).is_allowed(Action(ActionKind.MOVE, Direction.S, 32))


# --- option executor ---

def test_wall_move_costs_at_most_two_frames(session):
    session.load('1_73s_0', randomize_rng=False)
    session.act(Action.parse('MOVE N 16'))      # off the S door tile
    act_until(session, 'MOVE W 32', lambda c: c.interrupt == Interrupt.WALL)
    change = session.act(Action.parse('MOVE W 32'))
    assert change.interrupt == Interrupt.WALL
    assert change.frames <= 2


def test_timed_move_runs_requested_duration(session):
    session.load('1_73s_0', randomize_rng=False)
    start_y = session.state.link.position.y
    change = session.act(Action.parse('MOVE N 16'))
    assert change.interrupt is None
    assert 16 <= change.frames <= 18                  # + the one frame the executor uses to settle
    assert change.state.link.position.y < start_y


def test_wait_is_interrupted_in_combat(session):
    session.load('1_44w', randomize_rng=False)
    session.act(Action.parse('MOVE E 16'))
    reasons = set()
    for _ in range(40):
        change = session.act(Action.parse('WAIT 32'))
        reasons.add(change.interrupt)
        if change.state.game_over:
            break
    assert reasons - {None}, "Goriyas never interrupted a WAIT"
    assert reasons <= set(Interrupt.ALL) | {None}


def _shield_ram(kind=0x5C):
    """Shield-eligible Link with one outgoing hostile shot, already within the collision box."""
    ram = np.zeros(0x800, dtype=np.uint8)
    for address, value in {0x10: 1, 0x12: 5, 0xEB: 0x44, 0x70: 128, 0x84: 144, 0x98: 1,
                           0x66F: 0x22, 0x670: 255, 0x34F + 11: kind, 0xAC + 11: 0x17,
                           0x70 + 11: 140, 0x84 + 11: 144, 0x98 + 11: 2,
                           0x4BF + 11: 0xE1, 0x3D0 + 11: 2}.items():
        ram[address] = value
    ram[0x346:0x34A] = (24, 224, 80, 200)
    return ram


def _parry_frame(ram, kind=0x5C):
    ram[0x604] = 1
    ram[0x70 + 11] = 137
    ram[0xAC + 11] = 0x20 if kind == 0x5C else 0x30
    ram[0x3D0 + 11] = 3
    ram[0x380 + 11] = ram[0x98]
    ram[0x394 + 11] = 0
    if kind == 0x55:
        ram[0x34F + 11] = 0
        ram[0xAC + 11] = 0


@pytest.mark.parametrize('kind', [0x53, 0x55, 0x57, 0x5C])
def test_shield_collision_types_use_request_and_signature(kind):
    ram = _shield_ram(kind)
    ram[0x676] = 1
    before = ShieldCollisionWatch(ram.__getitem__)
    _parry_frame(ram, kind)
    assert ram[0x605] == 0  # the sound is delayed, but this collision is already confirmed
    assert before.collisions(ram.__getitem__, 100) == (ProjectileBlock(100, 11, kind, MapLocation(1, 0x44, False)),)
    repeated = ShieldCollisionWatch(ram.__getitem__)
    ram[0x605] = 1
    assert repeated.collisions(ram.__getitem__, 101) == ()


@pytest.mark.parametrize('address,value', [(0x66C, 1), (0x4F0, 1), (0x3D, 1), (0x3D + 11, 1),
                                         (0x512, 1), (0xAC, 0x10), (0xAC, 0x40), (0x98, 2),
                                         (0x66F, 0x21), (0x670, 127), (0x604, 0), (0x12, 6)])
def test_shield_collision_rejects_clock_stun_wrong_facing_damage_and_missing_request(address, value):
    ram = _shield_ram()
    before = ShieldCollisionWatch(ram.__getitem__)
    _parry_frame(ram)
    ram[address] = value
    assert before.collisions(ram.__getitem__, 100) == ()


@pytest.mark.parametrize('phase', [0, 0x10, 0x28, 0x30, 0x40, 0x50])
def test_boomerang_disappearance_and_natural_return_are_not_blocks(phase):
    ram = _shield_ram()
    before = ShieldCollisionWatch(ram.__getitem__)
    _parry_frame(ram)
    ram[0xAC + 11] = phase
    assert before.collisions(ram.__getitem__, 100) == ()


def test_magic_shield_does_not_block_unshieldable_fireballs():
    ram = _shield_ram(0x56)
    ram[0x676] = 1
    before = ShieldCollisionWatch(ram.__getitem__)
    _parry_frame(ram, 0x56)
    assert before.collisions(ram.__getitem__, 100) == ()


def test_immune_clink_with_nearby_unchanged_projectile_is_not_a_block():
    ram = _shield_ram()
    before = ShieldCollisionWatch(ram.__getitem__)
    ram[0x604] = ram[0x605] = 1
    assert before.collisions(ram.__getitem__, 100) == ()


@pytest.mark.parametrize('kind', [0x55, 0x57])
def test_magic_shots_require_magic_shield(kind):
    ram = _shield_ram(kind)
    before = ShieldCollisionWatch(ram.__getitem__)
    _parry_frame(ram, kind)
    assert before.collisions(ram.__getitem__, 100) == ()


@pytest.mark.parametrize('address,value', [(0x380 + 11, 2), (0x394 + 11, 2)])
def test_fireball_disappearance_needs_collision_bounce_evidence(address, value):
    ram = _shield_ram(0x55)
    ram[0x676] = 1
    before = ShieldCollisionWatch(ram.__getitem__)
    _parry_frame(ram, 0x55)
    ram[address] = value
    assert before.collisions(ram.__getitem__, 100) == ()


def test_shield_collision_uses_half_width_midpoint_and_strict_axis_thresholds():
    ram = _shield_ram()
    before = ShieldCollisionWatch(ram.__getitem__)
    _parry_frame(ram)
    ram[0x70 + 11], ram[0x84 + 11] = 140, 152  # midpoint distances 8 and 8, even though Euclidean >9
    assert len(before.collisions(ram.__getitem__, 100)) == 1
    ram[0x70 + 11] = 141  # half-width midpoint now exactly 9 away
    assert before.collisions(ram.__getitem__, 100) == ()


def test_unrelated_clink_does_not_credit_a_stationary_wall_spark_or_boundary_fireball():
    ram = _shield_ram()
    before = ShieldCollisionWatch(ram.__getitem__)
    _parry_frame(ram)
    ram[0x70 + 11] = 140  # blocked by a tile without moving; another actor supplied the clink
    assert before.collisions(ram.__getitem__, 100) == ()

    ram = _shield_ram(0x55)
    ram[0x676] = 1
    ram[0x70] = ram[0x70 + 11] = 16  # west-facing fireball outside the left bound: no Link check executes
    before = ShieldCollisionWatch(ram.__getitem__)
    _parry_frame(ram, 0x55)
    ram[0x70 + 11] = 16
    assert before.collisions(ram.__getitem__, 100) == ()


def test_real_before_trace_credits_collision_even_when_sound_endpoints_are_silent():
    # Captured before the fix: seed 0, WAIT 32 action 106, frame 2973 -> 2974 in 1:0x44.
    # Both controllable endpoint sounds were silent; old critique_block paid nothing. Actual $604=1,
    # slot 11 outgoing $17 -> spark $20, no clock/invincibility/stun/health loss, half-width midpoint (84,181)
    # collides with Link's (88,189). This is real RAM evidence, not a fabricated sound-onset scenario.
    ram = _shield_ram()
    ram[0x70], ram[0x84], ram[0x98], ram[0x66F], ram[0x670] = 80, 181, 2, 33, 254
    ram[0x70 + 11], ram[0x84 + 11], ram[0x98 + 11] = 77, 173, 1
    before = ShieldCollisionWatch(ram.__getitem__)
    ram[0x70 + 11], ram[0xAC + 11], ram[0x3D0 + 11], ram[0x604] = 80, 0x20, 3, 1
    assert before.collisions(ram.__getitem__, 2974) == (ProjectileBlock(2974, 11, 0x5C, MapLocation(1, 0x44, False)),)


class _ScriptedShieldEmulator:
    """Deterministic RAM frames to exercise executor branches without relying on monster movement."""
    def __init__(self, collision_frame, stunned_until=0):
        self.ram = _shield_ram()
        self.frame = 0
        self.collision_frame = collision_frame
        self.stunned_until = stunned_until
        self.on_frame = None
        self.buttons = []

    def read(self, address):
        return int(self.ram[address])

    def set_value(self, name, value):
        if name == 'link_direction':
            self.ram[0x98] = value

    def step(self, buttons=None, frames=1):
        for _ in range(frames):
            self.buttons.append(None if buttons is None else tuple(buttons))
            self.frame += 1
            self.ram[0xAC] = 0x40 if self.frame < self.stunned_until else 0
            if self.frame == self.collision_frame:
                _parry_frame(self.ram)
            if self.on_frame:
                self.on_frame()


@pytest.mark.parametrize('duration', [4, 32])
def test_executor_counts_every_collision_without_interrupting_or_replacing_callback(duration):
    emu = _ScriptedShieldEmulator(collision_frame=2)
    observed = []
    callback = lambda: observed.append(emu.frame)
    emu.on_frame = callback
    state = SimpleNamespace(level=1, location=0x44, in_cave=False)
    result = OptionExecutor(emu).execute(Action(ActionKind.WAIT, duration=duration), state)
    assert result.frames == duration + 1 and result.control_frames == duration and result.interrupt is None
    assert result.blocks == 1 and result.projectile_blocks[0].frame == 2
    assert observed == list(range(1, duration + 2)) and emu.on_frame is callback
    assert emu.buttons == [None] * result.frames


def test_executor_counts_blocks_in_attack_settling_and_uncontrollable_frames():
    state = SimpleNamespace(level=1, location=0x44, in_cave=False)
    emu = _ScriptedShieldEmulator(collision_frame=3)
    result = OptionExecutor(emu).execute(Action(ActionKind.SWORD, Direction.E), state)
    assert (result.control_frames, result.frames, result.blocks) == (2, 3, 1)
    assert emu.buttons[0][-1] == 1 and emu.buttons[1:] == [None, None]
    emu = _ScriptedShieldEmulator(collision_frame=6, stunned_until=6)
    result = OptionExecutor(emu).execute(Action(ActionKind.WAIT, duration=4), state)
    assert (result.control_frames, result.frames, result.blocks) == (4, 7, 1)
    assert result.projectile_blocks[0].frame == 6


def _place_shield_projectile(session, slots=(11,)):
    session.load('1_73s_0', randomize_rng=False)
    session.act(Action.parse('MOVE N 16'))  # off the entry door, in playable mode before RAM placement
    session.apply_ram({'link_x': 128, 'link_y': 144, 'link_direction': 1,
                       'hearts_and_containers': 0x22, 'partial_hearts': 255})
    for address in (0xAC, 0x3D, 0x4F0, 0x66C, 0x512):
        session.emu.write(address, 0)
    for slot in slots:
        for base, value in {0x34F: 0x5C, 0xAC: 0x17, 0x70: 140, 0x84: 144, 0x98: 2, 0x4BF: 0xE1,
                            0x3D0: 2, 0x3BC: 0xA0, 0x3A8: 0, 0x380: 0x51, 0x394: 0,
                            0x28: 0, 0x3D: 0, 0x405: 0, 0x492: 0}.items():
            session.emu.write(base + slot, value)
    session.emu.write(0x34C, len(slots))
    session.apply_ram({})


def test_real_shield_block_can_coincide_with_a_heart_pickup(session):
    _place_shield_projectile(session)
    session.apply_ram({'hearts_and_containers': 0x21, 'partial_hearts': 255})
    for base, value in {0x34F: 0x60, 0xAC: 0x22, 0x70: 128, 0x84: 144, 0x405: 0, 0x3A8: 0xC0}.items():
        session.emu.write(base + 1, value)
    session.apply_ram({})
    change = session.act(Action.parse('WAIT 4'))
    assert change.health_gained == 1 and change.health_lost == 0
    assert change.blocks == 1


@pytest.mark.parametrize('duration', [4, 32])
def test_real_projectile_block_counts_are_independent_of_option_duration(session, duration):
    _place_shield_projectile(session, slots=(10, 11))
    start = session.emu.frame
    change = session.act(Action(ActionKind.WAIT, duration=duration))
    assert change.blocks == change.option.blocks == 2
    assert {event.slot for event in change.projectile_blocks} == {10, 11}
    assert all(start < event.frame <= start + change.frames for event in change.projectile_blocks)
    assert change.health_lost == 0 and change.interrupt is None
    assert session.act(Action.parse('WAIT 4')).blocks == 0


def test_block_collection_preserves_real_seeded_timing_state_and_user_callback(session, monkeypatch):
    _place_shield_projectile(session)
    snapshot = session.snapshot()
    observed = []
    callback = lambda: observed.append(session.emu.frame)
    session.emu.on_frame = callback
    try:
        active = session.act(Action.parse('WAIT 16'))
        active_ram = session.emu.ram.copy()
        active_screen = session.emu.get_screen().copy()
        assert len(observed) == active.frames and session.emu.on_frame is callback
        session.restore(snapshot)
        observed.clear()
        with monkeypatch.context() as patch:
            patch.setattr(ShieldCollisionWatch, 'collisions', lambda self, read, frame: ())
            unobserved = session.act(Action.parse('WAIT 16'))
        assert len(observed) == unobserved.frames and session.emu.on_frame is callback
        assert active.blocks == 1 and unobserved.blocks == 0
        assert (active.frames, active.option.control_frames, active.interrupt) == \
               (unobserved.frames, unobserved.option.control_frames, unobserved.interrupt)
        assert np.array_equal(active_ram, session.emu.ram)
        assert np.array_equal(active_screen, session.emu.get_screen())
    finally:
        session.emu.on_frame = None


def test_push_block_opens(session):
    place_at_push_block(session)
    session.act(Action.parse('MOVE S 32'))
    session.act(Action.parse('WAIT 32'))
    assert int(session.emu.ram[0x4CF]) >= 1          # BlockPushComplete
    assert not session.state.push_block_pending


def test_scripted_walk_through_three_rooms(session):
    """Phase 1 acceptance: timed MOVEs carry Link through three overworld rooms."""
    session.load('start', randomize_rng=False)
    visited = [session.state.location]
    for direction in ('E', 'W', 'W'):
        changes = act_until(session, f'MOVE {direction} 32', lambda c: c.changed_location)
        visited.append(changes[-1].state.location)
    assert visited == [0x77, 0x78, 0x77, 0x76]


def test_beam_look_ahead_credits_firing_step(session):
    """Beams fired at an enemy are credited to the firing step and not again when they land."""
    session.load('1_44w', randomize_rng=False, ram_overrides=full_health(session, 16))
    leave_west_door(session)
    credited = 0
    for _ in range(40):
        state = session.state
        target = min(state.active_enemies, key=lambda e: e.distance, default=None)
        if target is None:
            break
        dx, dy = target.position.x - state.link.position.x, target.position.y - state.link.position.y
        if abs(dy) < 8 and session.mask.kind[ActionKind.BEAMS]:
            change = session.act(Action(ActionKind.BEAMS, Direction.E if dx > 0 else Direction.W))
            credited += change.damage_dealt
            follow = session.act(Action.parse('WAIT 32'))
            assert not follow.enemies_hit or credited == 0, "beam damage counted twice"
            if credited:
                return
        else:
            session.act(Action.parse('MOVE S 8' if dy > 0 else 'MOVE N 8'))
    pytest.skip("never lined up a beam shot from this savestate")


def test_without_look_ahead_a_beam_emulates_only_real_frames(session):
    """look_ahead=False (watching games): firing beams runs no hidden simulated frames through the emulator."""
    session.load('1_44w', randomize_rng=False, ram_overrides=full_health(session, 16))
    leave_west_door(session)
    assert session.mask.kind[ActionKind.BEAMS]
    snap = session.snapshot()
    emulated = []
    session.emu.on_frame = lambda: emulated.append(1)
    try:
        counts = {}
        for look_ahead in (True, False):
            session.restore(snap)
            session.look_ahead = look_ahead
            emulated.clear()
            start = session.emu.frame
            session.act(Action(ActionKind.BEAMS, Direction.E))
            counts[look_ahead] = (len(emulated), session.emu.frame - start)
    finally:
        session.emu.on_frame = None
        session.look_ahead = True
    hidden_on = counts[True][0] - counts[True][1]
    assert hidden_on > 0, f"the beam shot never triggered a look-ahead: {counts}"
    assert counts[False][0] == counts[False][1], f"look_ahead=False still emulated hidden frames: {counts}"


def test_snapshot_restore_is_deterministic(session):
    session.load('1_44w', randomize_rng=False)
    snap = session.snapshot()
    script = ['MOVE E 16', 'WAIT 8', 'MOVE S 8', 'SWORD E']
    first = [session.act(Action.parse(a)).state.link.position for a in script]
    session.restore(snap)
    second = [session.act(Action.parse(a)).state.link.position for a in script]
    assert first == second



def test_cave_person_line_blocks_north(session):
    """Link can't walk above the cave dweller's line; MOVE N must be masked there, and the sword is reachable."""
    session.load('0_77cave_0', randomize_rng=False)
    assert session.state.in_cave
    act_until(session, 'MOVE N 16', lambda c: not session.mask.direction[ActionKind.MOVE, 0])
    assert session.state.link.position.y < 0x8E
    assert session.state.link.sword == SwordKind.NONE
