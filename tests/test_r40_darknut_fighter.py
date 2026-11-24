"""Round 40: the owner's Darknut fighter (``scripts/experiments/r40_darknut_fighter.py``; docs/owner/darknuts.md).

The owner's take ``replays/owner/darknut/blue_darknut-play_with_anticipation.tfreplay`` read frame by frame: his first
six swings, each against the fighter's geometry (the game's sword box, the shield, the grid point, the flash) and his
stance against the fighter's shadow band.  Then the fighter's own visits on fixed seeds through the case tasks."""

import importlib.util
from pathlib import Path

import pytest

from .conftest import REPO_STATES_DIR, require_rom

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'experiments' / 'r40_darknut_fighter.py'
TAKE = ROOT / 'replays' / 'owner' / 'darknut' / 'blue_darknut-play_with_anticipation.tfreplay'


@pytest.fixture(scope='module')
def fighter():
    require_rom()
    spec = importlib.util.spec_from_file_location('r40_darknut_fighter', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def take(fighter):
    """The take's RAM and controller byte a frame, and the room's floor."""
    from triforce import _native
    player = _native.ReplayPlayer(TAKE.read_bytes())
    rams, buttons, grid = [], [], None
    for _ in range(player.frames):
        player.step(1, False)
        rams.append(bytes(player.ram()))
        buttons.append(player.buttons)
        if grid is None and fighter.darknuts(rams[-1]):
            game = _native.Game(states_dir=str(REPO_STATES_DIR))
            grid = game.load(player.save_state(), randomize_rng=False, controllable=True).render_grid()
    return rams, buttons, grid


def swings(fighter, rams, buttons, count):
    """The first `count` swings: the frame A is read (newly pressed, Link's state $1x), Link, the Darknut nearest him
    then, and whether its HP fell (or it died) within the swing."""
    out = []
    for f in range(1, len(rams)):
        ram = rams[f]
        if buttons[f] & 1 and not buttons[f - 1] & 1 and ram[0xAC] & 0xF0 == 0x10:   # the replay's A is bit 0
            x, y = ram[fighter.OBJ_X], ram[fighter.OBJ_Y]
            d = min(fighter.darknuts(ram), key=lambda n: abs(n.x - x) + abs(n.y - y))
            later = rams[f + 13]
            landed = later[fighter.OBJ_TYPE + d.slot] != d.kind or later[fighter.OBJ_HP + d.slot] >> 4 < d.hp
            out.append((f, x, y, ram, d, landed))
            if len(out) == count:
                break
    return out


def test_the_owners_first_six_swings_against_the_fighters_geometry(fighter, take):
    """His first four swings strike the same Blue Darknut walking E from 15-24 pixels S of its line and 5-13 ahead
    of it (the fighter's shadow spot, `OFF` 20 and `AHEAD` 8, is inside that band); the next two are his anticipation
    at the E wall ("I miss, my positioning was a little off, but it was the right idea").  The fighter's own test of
    each swing (`Fighter.lands`, the game's sword box) agrees with the game on all six where it accepts or refuses
    for the game's reason:
    1-3 land, and the fighter would make the same strike (SWORD N at that Darknut);
    4 lands on the frame its flash from swing 3 ends and the frame before it reaches its grid point: the fighter's
      margins (two frames past the flash, two frames in the box) refuse it;
    5 misses: the Darknut, turned S at the wall, is 21 pixels above the E blade's box;
    6 misses: the Darknut is 1.6 frames from its grid point, where it turns W, its shield to the blade (the fighter
      refuses a strike before a grid point)."""
    rams, buttons, grid = take
    rule = fighter.Fighter('garden', fighter.Floor(grid))
    seen = []
    for f, x, y, ram, d, landed in swings(fighter, rams, buttons, 6):
        facing = fighter.GAME_DIR[ram[fighter.OBJ_DIR] & 0x0F]
        off = abs(x - d.x) if d.dir in 'NS' else abs(y - d.y)
        vx, vy = fighter.VEC[d.dir]
        ahead = (x - d.x) * vx + (y - d.y) * vy
        accepted = rule.lands(d, x, y, facing, 0) is not None
        strikes = [(text, n.slot) for text, _, n, _ in rule.strikes(ram, fighter.darknuts(ram),
                                                                     lambda t: t.startswith('SWORD'))]
        seen.append((f, facing, d.slot, d.dir, off, ahead, landed, accepted, (f'SWORD {facing}', d.slot) in strikes))
    assert seen == [
        (87, 'N', 5, 'E', 15, 13, True, True, True),
        (118, 'N', 5, 'E', 24, 10, True, True, True),
        (148, 'N', 5, 'E', 22, 8, True, True, True),
        (179, 'N', 5, 'E', 16, 5, True, False, False),
        (220, 'E', 5, 'S', 17, 20, False, False, False),
        (237, 'E', 5, 'S', 17, 9, False, False, False),
    ]
    for _, _, _, _, off, ahead, *_ in seen[:4]:
        assert 15 <= off <= 24 and 5 <= ahead <= 13
        assert abs(off - fighter.OFF) <= 5 and abs(ahead - fighter.AHEAD) <= 5


def test_the_fighter_clears_the_garden_with_no_bomb_unhit(fighter):
    """The Olive Garden from E with no bomb, episode seed 101 (entry state 1, as `triforce evaluate` lays it out): six
    Blue Darknuts killed by the sword, the block pushed, down the stairs, no hit."""
    record = fighter.play(fighter.make_env('garden', 0, REPO_STATES_DIR), 'garden', 101, 'd3first_5_05e_1')
    assert (record['ending'], record['hits'], record['bombs']) == ('exit-correct', 0, 0), record


def test_the_fighter_keeps_the_reserve_in_3x59(fighter):
    """3:0x59 from N with 4 bombs, reserve 5 (the owner, decisions 2026-11-24): cleared by the sword, no bomb laid,
    out by the S door."""
    record = fighter.play(fighter.make_env('59', 4, REPO_STATES_DIR), '59', 1, 'd3first_3_59n_0')
    assert (record['ending'], record['bombs']) == ('exit-correct', 0), record


def test_a_bomb_is_laid_only_for_two_or_more(fighter):
    """3:0x59 with 8 bombs (three above the reserve), seed 2: the one bomb laid hurt two or more Darknuts."""
    record = fighter.play(fighter.make_env('59', 8, REPO_STATES_DIR), '59', 2, 'd3first_3_59n_2')
    assert record['ending'] == 'exit-correct' and record['bombs'] >= 1, record
    assert all(hurt >= 2 for hurt in record['bomb_hurt']), record
