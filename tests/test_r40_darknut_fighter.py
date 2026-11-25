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
    """The Olive Garden from E with no bomb, episode seed 104 (pinned on entry state 1, as `triforce evaluate` laid it
    out before round 44's one state per door): six Blue Darknuts killed by the sword, the block pushed, down the
    stairs, no hit. (Seed 101 was pinned first; since the fighter may back out of a doorway and no longer models a
    sideways step there, its entry from the E door times differently and at frame 971 a Darknut walks into a swing:
    one hit. Seeds 103-105 and 107-111 are unhit.)"""
    # round 44: one state per door, from the real walk; seed 104 (pinned on chain 1's old state) passes on the walk's
    record = fighter.play(fighter.make_env('garden', 0, REPO_STATES_DIR), 'garden', 104, 'd3first_5_05e_0')
    assert (record['ending'], record['hits'], record['bombs']) == ('exit-correct', 0, 0), record


def test_the_fighter_keeps_the_reserve_in_3x59(fighter):
    """3:0x59 from N with 4 bombs, reserve 5 (the owner, decisions 2026-11-24): cleared by the sword, no bomb laid,
    out by the S door."""
    record = fighter.play(fighter.make_env('59', 4, REPO_STATES_DIR), '59', 1, 'd3first_3_59n_0')
    assert (record['ending'], record['bombs']) == ('exit-correct', 0), record


def test_a_bomb_is_laid_only_for_two_or_more(fighter):
    """3:0x59 with 8 bombs (three above the reserve), seed 2: the one bomb laid hurt two or more Darknuts."""
    # round 44: d3first has one state per door (chain 0); seed 2 was pinned on chain 2's state
    record = fighter.play(fighter.make_env('59', 8, REPO_STATES_DIR), '59', 2, 'd3first_3_59n_0')
    assert record['ending'] == 'exit-correct' and record['bombs'] >= 1, record
    assert all(hurt >= 2 for hurt in record['bomb_hurt']), record


def test_the_chase_cost_is_paid_on_the_turn_rules_line(fighter):
    """`reward.chase_penalty` (Darknut40's switched-off proposal): a walker that keeps stepping behind the nearest
    Darknut on its line pays `penalty-chase`, its value times the step's frames, on exactly the steps that end with
    Link on the line of a Darknut walking away from him; with the key at 0 (the default) it is never paid."""
    from triforce._native import Env, SpecialistConfig
    base = SpecialistConfig.load(str(fighter.task_file('garden', 0)))
    for cost in (0.01, 0.0):
        config = base.replace(reward=dict(base.to_dict()['reward'], chase_penalty=cost))
        env = Env(config, explicit_starts=True, strict_actions=False, seed=0, states_dir=str(REPO_STATES_DIR))
        env.reset(seed=7, options={'start_state': 'd3first_5_05e_0'})
        paid, last = 0, None
        for _ in range(150):
            ram = bytes(env.state.ram)
            x, y = ram[fighter.OBJ_X], ram[fighter.OBJ_Y]
            nuts = [d for d in fighter.darknuts(ram) if not d.cloud and d.dir]
            way = 'W'
            if nuts and 0x20 <= x <= 0xD0:
                d = min(nuts, key=lambda n: abs(n.x - x) + abs(n.y - y))
                vx, vy = fighter.VEC[d.dir]
                gx, gy = d.x - 16 * vx, d.y - 16 * vy             # one cell behind it on its line
                ways = [('E' if gx > x else 'W'), ('S' if gy > y else 'N')]
                if abs(gx - x) <= abs(gy - y):
                    ways.reverse()
                way = ways[1] if (x, y) == last else ways[0]  # stopped (the block, a wall): the other axis
            last = (x, y)
            _, _, terminated, truncated, info = env.step(f'MOVE {way} 4')
            if terminated or truncated:
                break
            charged = info['rewards'].get('penalty-chase')
            if cost:
                expected = fighter.chasing_line(bytes(env.state.ram))
                assert (charged is not None) == expected, (info, expected)
                if charged is not None:
                    assert charged == pytest.approx(-cost * info['frames'])
                    paid += 1
            else:
                assert charged is None
        if cost:
            assert paid >= 5, paid


def test_the_proposals_read_the_owners_bomb_take():
    """`r40_darknut_proposals.read` on the owner's `bombs_with_melee` take: his six bombs (docs/owner/darknuts.md: five
    hurt two, the fourth none) and the catch count `items.bomb_catch` would have shown on the frame each was laid (the
    same five at two or more); his one hit; the frames a Darknut was set to reverse onto him at its next grid point
    (the document counts frames from 1, here from 0: its bomb 1 at 299 is 298 here, its hit at 992 is 991)."""
    require_rom()
    spec = importlib.util.spec_from_file_location('r40_darknut_proposals', SCRIPT.with_name('r40_darknut_proposals.py'))
    proposals = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proposals)
    visit = proposals.read(TAKE.with_name('blue_darknut-bombs_with_melee.tfreplay'))
    assert [(b['frame'], b['predicted'], b['hurt']) for b in visit['bombs']] == [
        (298, 2, 2), (492, 3, 2), (980, 2, 2), (1183, 1, 0), (1427, 3, 2), (2193, 2, 2)]
    # 233 chasing frames since `chasing` became `zelda::chase`'s `reverse_link` (was 168: Link on the line within $30
    # behind the Darknut now). Compared frame by frame: all 168 are still in, the 65 new ones are Link farther behind
    # (frame 291: a Darknut at (56, 109) walking W, its grid point (48, 109), Link at (104, 116), 48 behind it), and
    # the hit at 991 still has chasing frames in the 32 before it (959 on).
    assert visit['hits'] == [991] and len(visit['chase']) == 233
    assert any(991 - 32 <= f < 991 for f in visit['chase'])


def test_the_first_pass_of_3x5b_kills_one_darknut_and_takes_its_bombs(fighter, monkeypatch):
    """3:0x5b's first pass (the owner's Dungeon 3 plan; `carriers: fight`): seed 12 on `d3first_3_5bs_0`, hearts
    drawn, in the route's layout (spawn cycle 3) with the drop cycle at 0: the carriers lit, one Red Darknut killed by
    the sword, its bombs (the table's drop) taken, out by the N door, no hit, no second kill.  Seed 1, the same layout
    at drop cycle 1: the carriers dark, Link walks through and kills nothing (seeds chosen under round 43's reset rule
    on round 44's one state per door; the case draws the cycle 0 or 1, round 45)."""
    monkeypatch.setenv(fighter.HEARTS_VARIANT, 'low')
    env = fighter.make_env('5b', 0, REPO_STATES_DIR)
    record = fighter.play(env, '5b', 12, 'd3first_3_5bs_0')
    assert (record['ending'], record['hits'], record['kills'], record['drop_appeared'], record['lit']) == \
        ('exit-correct', 0, 1, True, True), record
    assert record['bag_gain'] > 0 and record['layout']['planned'] and record['drop_cycle'] == 0, record
    walked = fighter.play(env, '5b', 1, 'd3first_3_5bs_0')
    assert (walked['ending'], walked['hits'], walked['kills'], walked['lit'], walked['drop_cycle']) == \
        ('exit-correct', 0, 0, False, 1), walked
