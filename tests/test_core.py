"""Emulator, object model, action schema, option executor and session behavior on real savestates."""

import pytest

from triforce.actions import (ACTION_SCHEMA_VERSION, MASK_SIZE, Action, ActionMask, DIRECTIONS, DURATIONS,
                              compute_action_mask)
from triforce.frame_skip import Interrupt
from triforce.zelda_enums import ActionKind, Direction, SwordKind, ZeldaEnemyKind

from .conftest import act_until, full_health


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


# --- masks ---

def test_doorway_restricts_moves_and_weapons(session):
    session.load('1_44w', randomize_rng=False)
    mask = session.mask
    assert mask.directions(ActionKind.MOVE) == [Direction.W, Direction.E]
    assert not mask.kind[ActionKind.SWORD]      # next to the wall a weapon press would leave the room
    session.act(Action.parse('MOVE E 16'))
    assert session.mask.kind[ActionKind.SWORD] and session.mask.kind[ActionKind.BOMB]


def test_sword_and_beams_are_exclusive(session):
    session.load('1_44w', randomize_rng=False)
    session.act(Action.parse('MOVE E 16'))
    assert session.mask.kind[ActionKind.SWORD] and not session.mask.kind[ActionKind.BEAMS]

    state = session.load('1_44w', randomize_rng=False, ram_overrides=full_health(session, 16))
    session.act(Action.parse('MOVE E 16'))
    assert state.link.sword == SwordKind.WOOD
    assert session.mask.kind[ActionKind.BEAMS] and not session.mask.kind[ActionKind.SWORD]


def test_no_weapons_in_empty_room(session):
    session.load('1_73n', randomize_rng=False)
    mask = session.mask
    assert mask.kind[ActionKind.MOVE] and mask.kind[ActionKind.WAIT]
    assert not mask.kind[ActionKind.SWORD] and not mask.kind[ActionKind.BOMB]


def test_push_block_move_is_legal(session):
    session.load('1_22e', randomize_rng=False, ram_overrides={'link_x': 96, 'link_y': 125})
    state = session.state
    assert state.push_block_pending
    assert state.is_push_block_direction(Direction.S)
    assert not state.can_link_move(Direction.S)
    assert compute_action_mask(state).is_allowed(Action(ActionKind.MOVE, Direction.S, 32))


# --- option executor ---

def test_wall_move_costs_at_most_two_frames(session):
    session.load('1_73n', randomize_rng=False)
    act_until(session, 'MOVE W 32', lambda c: c.interrupt == Interrupt.WALL)
    change = session.act(Action.parse('MOVE W 32'))
    assert change.interrupt == Interrupt.WALL
    assert change.frames <= 2


def test_timed_move_runs_requested_duration(session):
    session.load('1_73n', randomize_rng=False)
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
    session.load('1_22e', randomize_rng=False, ram_overrides={'link_x': 96, 'link_y': 125})
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
    session.act(Action.parse('MOVE E 16'))
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
    session.load('0_77cave', randomize_rng=False)
    assert session.state.in_cave
    act_until(session, 'MOVE N 16', lambda c: not session.mask.direction[ActionKind.MOVE, 0])
    assert session.state.link.position.y < 0x8E
    assert session.state.link.sword == SwordKind.NONE
