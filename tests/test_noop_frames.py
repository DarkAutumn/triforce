"""Frames where Link can't act are the executor's, not a decision's (Noops52, Nov 27; Noop51's census, round40.md "No-op
decisions"): the boomerang's catch pose is played inside the running action, and a shove offers one MOVE, its own,
that ends where the shove does (the mask's no-ops: `test_noop_decisions.py`).
"""

import pytest

from .test_noop_decisions import OBJ_TYPE, OBJ_X, OBJ_Y, clear_monsters, game_at, link

STATUS, SHOVE_DIR, BOOMERANG_SLOT = 0xAC, 0xC0, 0xBB
DURATIONS = (4, 8, 16, 32)


def moves(game):
    return {(d, u) for d in 'NSWE' for u in DURATIONS if game.is_allowed(f'MOVE {d} {u}')}


# --- 2. The boomerang's catch pose -----------------------------------------------------------------------------------

def test_no_decision_comes_in_the_boomerang_catch_pose():
    """`$20` one frame, `$21` six (asm:Boomerang_UpdateSlowAndReturn): the game zeroes Link's move and refuses A and
    B. Those frames don't count toward a WAIT (it plays 4 + 7, or 8) and no decision comes inside them; before, a WAIT
    4 every five frames left one of them in the pose."""
    game = game_at('d3first_6_38s_0')
    game.act('MOVE N 8')
    clear_monsters(game)
    game.act('BOOMERANG N')
    caught = None
    for _ in range(40):
        flying = game.read(BOOMERANG_SLOT) != 0
        change = game.act('WAIT 4')
        assert game.read(STATUS) & 0xF0 != 0x20, 'a decision in the catch pose'
        if flying and game.read(BOOMERANG_SLOT) == 0:
            caught = change.frames
            break
    assert caught is not None, 'the boomerang came back'
    assert caught >= 4 + 7 + 1, f'the catch pose counted toward the WAIT ({caught} frames)'


# --- 4. Knockback ----------------------------------------------------------------------------------------------------

def shove_duration(pixels):
    return min(DURATIONS, key=lambda d: abs(d * 3 - pixels * 2))


def shoved(state, seed):
    """Link walks into the nearest monster until one shoves him (a real hit)."""
    game = game_at(state, seed=seed, randomize=True)
    for _ in range(80):
        if game.read(SHOVE_DIR):
            return game
        x, y = link(game)
        monsters = [(game.read(OBJ_X + s), game.read(OBJ_Y + s)) for s in range(1, 7)
                    if 0 < game.read(OBJ_TYPE + s) < 0x40]
        mx, my = min(monsters, key=lambda m: abs(m[0] - x) + abs(m[1] - y))
        dx, dy = mx - x, my - y
        step = f"MOVE {('E' if dx > 0 else 'W') if abs(dx) > abs(dy) else ('S' if dy > 0 else 'N')} 4"
        game.act(step if game.is_allowed(step) else 'WAIT 4')
    pytest.fail(f'{state} seed {seed}: no shove in 80 steps')


@pytest.mark.parametrize(('state', 'seed', 'pixels'), [
    ('d3first_6_38s_0', 2, 32),          # a Like-Like's touch, the whole shove
    ('d3first_7_18s_0', 0, 7),           # cut short by the wall
    ('d3first_1_33s_0', 0, 3),
])
def test_a_shove_offers_one_move_its_way_that_lands_where_it_ends(state, seed, pixels):
    """While the shove runs the game reads no direction (asm:Link_HandleInput): the mask offers the attacks and one
    MOVE, the shove's way, of the duration whose walk (1.5 pixels a frame) is closest to the shove's distance, and no
    WAIT. That MOVE ends on the shove's last frame: the next decision sees where Link landed."""
    game = shoved(state, seed)
    snapshot, start = game.snapshot(), link(game)
    frames = 0
    while game.read(SHOVE_DIR):
        game.press(0, 1)
        frames += 1
    landing = link(game)
    travel = abs(landing[0] - start[0]) + abs(landing[1] - start[1])
    assert travel == pixels
    game.restore(snapshot)
    offered = moves(game)
    assert len(offered) == 1 and not game.is_allowed('WAIT 4'), game.legal
    (direction, duration), = offered
    assert duration == shove_duration(travel)
    change = game.act(f'MOVE {direction} {duration}')
    assert (change.frames, link(game), game.read(SHOVE_DIR)) == (frames, landing, 0)
    assert game.is_allowed('WAIT 4'), 'free again'


def test_an_attack_chosen_in_a_shove_still_strikes():
    """A and B are read during the shove (asm:Link_HandleInput): the strikes the mask offers there work."""
    game = shoved('d3first_6_38s_0', 2)
    swords = [d for d in 'NSWE' if game.is_allowed(f'SWORD {d}') or game.is_allowed(f'BEAMS {d}')]
    assert swords, game.legal
    kind = 'SWORD' if game.is_allowed(f'SWORD {swords[0]}') else 'BEAMS'
    change = game.act(f'{kind} {swords[0]}')
    assert change.frames >= 12, f'{kind} {swords[0]}: no swing ({change.frames} frames)'
