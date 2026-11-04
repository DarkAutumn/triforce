"""Emulator, object model, action schema, option executor and session behavior on real savestates."""

import pytest

from triforce.actions import (ACTION_SCHEMA_VERSION, MASK_SIZE, Action, ActionMask, DIRECTIONS, DURATIONS,
                              compute_action_mask)
from triforce.frame_skip import Interrupt
from triforce.zelda_enums import ActionKind, AnimationState, Direction, SwordKind, ZeldaAnimationKind, ZeldaEnemyKind

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
