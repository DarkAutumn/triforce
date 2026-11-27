"""Decisions that did nothing (Noops52, Nov 27; Noop51's census, round40.md "No-op decisions"): the mask no longer
offers a MOVE or a CLIP the game refuses (the frames where Link can't act: `test_noop_frames.py`).

SETUP (RAM writes, where a test needs a spot the route's states don't start on): Link's position, facing, grid offset
and doorway byte, monsters removed or added, a Like-Like's hold. What the game does there is then the game's own:
every masked press is also pressed, and moves nothing.
"""

from triforce import _native

from .conftest import REPO_STATES_DIR, require_state

LINK_X, LINK_Y, FACING, GRID_OFFSET, DOORWAY_DIR = 0x70, 0x84, 0x98, 0x394, 0x053
STATUS, HELD, LADDER_SLOT = 0xAC, 0x512, 0x64
OBJ_TYPE, OBJ_X, OBJ_Y, OBJ_TIMER = 0x34F, 0x70, 0x84, 0x28
FACINGS = {'E': 1, 'W': 2, 'S': 4, 'N': 8}


def game_at(state, seed=0, randomize=False):
    require_state(state)
    game = _native.Game(route='d3first', seed=seed, states_dir=str(REPO_STATES_DIR))
    game.load(state, randomize_rng=randomize)
    return game


def link(game):
    return game.read(LINK_X), game.read(LINK_Y)


def place_link(game, x, y, facing):
    for addr, value in ((DOORWAY_DIR, 0), (LINK_X, x), (LINK_Y, y), (GRID_OFFSET, 0), (FACING, FACINGS[facing])):
        game.write(addr, value)


def clear_monsters(game, keep=()):
    for slot in range(1, 12):
        if game.read(OBJ_TYPE + slot) not in keep:
            game.write(OBJ_TYPE + slot, 0)


def pressed(game, buttons, frames=8, see=lambda game: None):
    """The game's own answer: `buttons` held `frames` frames from here: whether Link moved, and what `see` reads then
    (before the machine is put back)."""
    snapshot, at = game.snapshot(), link(game)
    game.press(buttons, frames)
    moved, seen = link(game) != at, see(game)
    game.restore(snapshot)
    return moved, seen


def pressed_moves_nothing(game, buttons, frames=8):
    return not pressed(game, buttons, frames)[0]


def armos_count(game):
    return sum(game.read(OBJ_TYPE + s) == 0x1E for s in range(1, 12))


# --- 5. The woken secret Armos ---------------------------------------------------------------------------------------

def test_the_secret_armos_push_is_not_offered_while_its_armos_fades_in():
    """0:0x34's secret statue: the press from (`$50`, `$7D`) wakes the Armos, which fades in on the square for `$3F`
    frames; a press then makes nothing (asm:Spawn_CheckTouchedArmosOrGravestone), and it is no longer offered."""
    game = game_at('d3first_0_34r_0')
    for _ in range(30):                                    # the whirlwind sets Link down
        if game.read(0x522) == 0 and all(game.read(OBJ_TYPE + s) != 0x2E for s in range(1, 12)):
            break
        game.act('WAIT 8')
    clear_monsters(game)
    place_link(game, 0x50, 0x7D, 'W')
    assert game.is_allowed('MOVE W 4')
    game.act('MOVE W 4')
    armos = [s for s in range(1, 12) if game.read(OBJ_TYPE + s) == 0x1E]
    assert armos and (game.read(OBJ_X + armos[0]), game.read(OBJ_Y + armos[0])) == (0x40, 0x80), 'woken'
    fading = 0
    while game.read(OBJ_TIMER + armos[0]):
        assert link(game) == (0x50, 0x7D) and not game.is_allowed('MOVE W 4'), game.legal
        moved, armos_after = pressed(game, 'LEFT', 4, armos_count)
        assert not moved and armos_after == len(armos), 'a press makes no second Armos'
        game.act('WAIT 8')
        fading += 1
    assert fading >= 6


# --- 6. CLIP in a Like-Like's hold -----------------------------------------------------------------------------------

def test_no_clip_is_offered_in_a_like_likes_hold():
    """The game discards every direction in a hold (asm:Link_Update): the skip is not offered, as MOVE is not. The
    mask: where CLIP S is offered, writing the hold takes it away (SETUP). The game: in a real hold (9:0x51, walked into
    a Like-Like) no direction moves Link, and no CLIP is offered."""
    game = game_at('d3first_9_51s_0')
    game.act('MOVE N 8')
    game.set_clips(True)
    clear_monsters(game)
    place_link(game, 0x20, 0x9D, 'N')
    assert game.is_allowed('CLIP S'), game.legal
    assert not pressed_moves_nothing(game, 'DOWN'), 'free, S moves him'
    game.write(HELD, 1)
    assert 'CLIP' not in game.legal and 'MOVE' not in game.legal, game.legal

    game = game_at('d3first_9_51s_0', seed=1, randomize=True)
    game.set_clips(True)
    for _ in range(60):
        if game.read(HELD):
            break
        x, y = link(game)
        likes = [(game.read(OBJ_X + s), game.read(OBJ_Y + s)) for s in range(1, 12) if game.read(OBJ_TYPE + s) == 0x17]
        lx, ly = min(likes, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
        step = f"MOVE {('E' if lx > x else 'W') if abs(lx - x) > abs(ly - y) else ('S' if ly > y else 'N')} 4"
        shove = [f'MOVE {d} {u}' for d in 'NSWE' for u in (4, 8, 16, 32) if game.is_allowed(f'MOVE {d} {u}')]
        game.act(next(a for a in [step, 'WAIT 4', *shove] if game.is_allowed(a)))
    assert game.read(HELD), 'held'
    assert 'CLIP' not in game.legal and 'MOVE' not in game.legal, game.legal
    for buttons in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
        assert pressed_moves_nothing(game, buttons), buttons


# --- 7. The small ones -----------------------------------------------------------------------------------------------

def test_move_n_stops_at_a_dungeon_persons_line():
    """9:0x06's old man (slot 1, type `$4B`-`$52`): Link can't move up while his Y is below `$8E`
    (asm:Person_BlockLinkMovingUp, called by asm:Walker_Move in a dungeon too, not only in caves)."""
    game = game_at('d3first_9_06s_0')
    assert 0x4B <= game.read(OBJ_TYPE + 1) <= 0x52
    for _ in range(30):
        if not game.is_allowed('MOVE N 4'):
            break
        game.act('MOVE N 4')
    assert link(game) == (0x78, 0x8D), 'walked up to the line'
    assert not game.is_allowed('MOVE N 4') and pressed_moves_nothing(game, 'UP')


def test_the_person_line_lets_link_through_the_north_doorway():
    """7:0x28's Grumble (slot 1 `$36`, waiting for food): the line holds Link inside the room, but at the N doorway's
    mouth asm:Doorway_CheckLink sets his move up again after asm:Person_BlockLinkMovingUp, and MOVE N is offered."""
    game = game_at('d3first_7_28s_0')
    for _ in range(20):                                    # the Grumble's text halts Link first
        if game.read(STATUS) == 0 and game.read(STATUS + 1) == 1:
            break
        game.act('WAIT 8')
    assert game.read(OBJ_TYPE + 1) == 0x36
    place_link(game, 0x50, 0x8D, 'N')
    assert not game.is_allowed('MOVE N 4') and pressed(game, 'UP', see=lambda g: g.read(LINK_Y))[1] == 0x8D
    place_link(game, 0x78, 0x5D, 'N')
    assert game.is_allowed('MOVE N 4')
    moved, doorway = pressed(game, 'UP', see=lambda g: g.read(DOORWAY_DIR))
    assert moved and doorway == FACINGS['N'], 'into the N doorway'


def test_no_ladder_crossing_without_a_free_object_slot():
    """7:0x18's one-square water: the game lays the ladder in a free monster slot (asm:Obj_FindEmptyMonsterSlot);
    with all eleven taken it lays none and MOVE E onto the water is not offered."""
    game = game_at('d3first_7_18s_0')
    game.act('MOVE N 8')
    place_link(game, 0xA0, 0x8D, 'E')
    assert game.is_allowed('MOVE E 4')
    moved, ladder = pressed(game, 'RIGHT', see=lambda g: g.read(LADDER_SLOT))
    assert moved and ladder, 'a slot free: the ladder is laid and Link steps on'
    for slot in range(1, 12):
        if game.read(OBJ_TYPE + slot) == 0:
            for addr, value in ((OBJ_TYPE + slot, 0x1B), (OBJ_X + slot, 0x28), (OBJ_Y + slot, 0x64)):
                game.write(addr, value)
    assert not game.is_allowed('MOVE E 4')
    assert pressed(game, 'RIGHT', see=lambda g: g.read(LADDER_SLOT)) == (False, 0), 'no ladder laid, no step'


def test_no_move_off_the_grid_past_a_dungeon_bound():
    """Off the grid no tile is tested, but the room bounds are (asm:Walker_Move@CheckBoundary): one pixel past 8:0x6e's
    N bound below its bombable wall, MOVE N is not offered; MOVE S is, and moves him. Through an open N door
    (1:0x43) asm:Doorway_CheckLink lets him on: MOVE N is offered there and walks into the doorway. (With the ladder
    out asm:Ladder_CheckAndDraw moves him on too: zelda/tests/attacks.rs's clip frames, 4:0x21's ladder clip N.)"""
    game = game_at('d3first_8_6es_0')
    game.act('MOVE N 8')
    clear_monsters(game)
    place_link(game, 0x78, 0x5C, 'N')
    game.write(GRID_OFFSET, 0xFF)
    assert not game.is_allowed('MOVE N 4') and pressed_moves_nothing(game, 'UP')
    assert game.is_allowed('MOVE S 4')
    game.act('MOVE S 4')
    assert link(game)[1] > 0x5C, 'MOVE S walked'
    game = game_at('d3first_1_43s_0')
    game.act('MOVE N 8')
    clear_monsters(game)
    place_link(game, 0x78, 0x5C, 'N')
    game.write(GRID_OFFSET, 0xFF)
    assert game.is_allowed('MOVE N 4')
    moved, doorway = pressed(game, 'UP', see=lambda g: g.read(DOORWAY_DIR))
    assert moved and doorway == FACINGS['N']
