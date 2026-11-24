"""Link's hearts as the game holds them (zelda-asm/mechanics.md "Hearts, rings and death"), played on the emulator.

`HeartValues` ($66F): high nibble = heart containers - 1, low nibble = the full hearts BESIDES the partial heart;
`HeartPartial` ($670) fills that partial heart ($80-$FF drawn full, $01-$7F half, 0 empty).  A hit takes its
fraction off the partial heart (borrowing a full heart as $FF when it lacks) and then its whole hearts off the low
nibble, and kills when they exceed it (`asm:Link_BeHarmed`).  So the game's own two hearts are (1, $FF), and
(2, $00), which `env::start_states::hearts_encoding` wrote for two hearts until Nov 24, draws two hearts on the HUD
but survives a two-heart hit at none.  Play never makes a full heart with an empty partial: only death writes
$00 (`@HandleDied`).  The monsters are moved onto Link by memory, as the walk tests do, so the hits are forced."""

import pytest

from triforce import _native

from .conftest import REPO_STATES_DIR, require_rom

MODE, DEATH = 0x12, 0x11
HEARTS, PARTIAL = 0x66F, 0x670
OBJ_X, OBJ_Y = 0x70, 0x84


def hits(state: str, kinds: list, hearts: tuple, limit: int = 12, frames: int = 4000) -> tuple:
    """Loads ``state`` (no RNG redraw) with ``hearts`` = (HeartValues, HeartPartial) and moves a monster of
    ``kinds`` (taken in turn, one per hit) onto Link every frame until he dies or took ``limit`` hits: the
    (HeartValues, HeartPartial) after each hit, and whether he died."""
    require_rom()
    game = _native.Game(states_dir=str(REPO_STATES_DIR))
    game.load(state, randomize_rng=False)
    game.set_value('hearts_and_containers', hearts[0])
    game.set_value('partial_hearts', hearts[1])
    seen = [hearts]
    for _ in range(frames):
        now = game.state
        foes = [e for e in now.enemies if e['name'] == kinds[(len(seen) - 1) % len(kinds)] and e['active']]
        if foes:
            game.write(OBJ_X + foes[0]['index'], now.link['x'])
            game.write(OBJ_Y + foes[0]['index'], now.link['y'])
        game.press(0, 1)
        after = (game.read(HEARTS), game.read(PARTIAL))
        if after != seen[-1]:
            seen.append(after)
        if game.read(MODE) == DEATH or len(seen) > limit:
            break
    return seen[1:], game.read(MODE) == DEATH


def test_two_hearts_die_to_a_blue_darknut_and_the_old_bytes_did_not():
    """The Olive Garden (5:0x05, no ring at that step of the route): a Blue Darknut's touch is two hearts."""
    garden = 'd3first_5_05e_0'
    assert hits(garden, ['BlueDarknut'], (0x77, 0xFF))[0][0] == (0x75, 0xFF)          # 2 hearts a touch, no ring
    assert hits(garden, ['BlueDarknut'], (0x71, 0xFF)) == ([(0x70, 0x00)], True)       # the game's 2: dies
    seen, died = hits(garden, ['BlueDarknut'], (0x72, 0x00), limit=1)                 # the old 2: alive at none
    assert seen[0] == (0x70, 0x00) and not died
    assert hits(garden, ['BlueDarknut'], (0x72, 0x7F))[0][0] == (0x70, 0x7F)          # 2.5 survive one
    assert hits(garden, ['BlueDarknut'], (0x70, 0x7F)) == ([(0x70, 0x00)], True)       # half a heart: dead


@pytest.mark.parametrize('state, kinds, trace', [
    # Half a heart a hit (Blue Keese): each second one borrows a full heart as $FF (1/256 extra).
    ('d3first_3_4ae_0', ['BlueKeese'], [(0x22, 0x7F), (0x21, 0xFE), (0x21, 0x7E), (0x20, 0xFD), (0x20, 0x7D),
                                        (0x20, 0x00)]),
    # A heart a hit (Red Darknut): the partial heart stays $FF.
    ('d3first_3_69n_0', ['RedDarknut'], [(0x21, 0xFF), (0x20, 0xFF), (0x20, 0x00)]),
    # Half and whole hearts in turn (Blue Keese, Zol), both orders.
    ('d3first_3_49e_0', ['BlueKeese', 'Zol'], [(0x22, 0x7F), (0x21, 0x7F), (0x20, 0xFE), (0x20, 0x00)]),
    ('d3first_3_49e_0', ['Zol', 'BlueKeese'], [(0x21, 0xFF), (0x21, 0x7F), (0x20, 0x7F), (0x20, 0x00)]),
])
def test_play_never_leaves_a_full_heart_with_an_empty_partial(state, kinds, trace):
    """From 3 full hearts ($22, $FF) down to death: the bytes after every hit.  $00 comes only with death."""
    seen, died = hits(state, kinds, (0x22, 0xFF))
    assert (seen, died) == (trace, True)
    assert all(partial != 0 for _, partial in seen[:-1])


def test_the_encoding_is_the_games():
    """``_native.hearts_encoding`` (the training resets' own) writes what play leaves: the Blue Darknut's steps from
    8 full hearts, and n half hearts drawn as n / 2 on the HUD."""
    def encoded(half_hearts, max_health=8):
        out = _native.hearts_encoding(half_hearts, max_health)
        return out['hearts_and_containers'], out['partial_hearts']

    assert [encoded(n) for n in (16, 12, 8, 4)] == [(0x77, 0xFF), (0x75, 0xFF), (0x73, 0xFF), (0x71, 0xFF)]
    assert [encoded(n, 3) for n in (6, 5, 4, 3, 2, 1)] == [(0x22, 0xFF), (0x22, 0x7F), (0x21, 0xFF), (0x21, 0x7F),
                                                           (0x20, 0xFF), (0x20, 0x7F)]
    for n in range(1, 17):
        hearts, partial = encoded(n)
        assert (hearts & 0x0F) + (1.0 if partial >= 0x80 else 0.5) == n / 2     # the HUD's count
    with pytest.raises(ValueError):
        _native.hearts_encoding(4, 0)
