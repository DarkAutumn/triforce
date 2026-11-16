"""Schema 6's new columns on walked traces (round 26; docs/architecture.md "Observation", decisions 2026-11-16):
Link is put into the room by ``triforce.probe``, the game runs frame by frame with his hearts held, and
``_native.observe`` reads the state as the env's observation would (scripts/experiments/r26_obs_probes.py walked the
same rooms)."""

import pytest

from .conftest import require_rom

TYPE, X, Y, STATE = 0x34F, 0x70, 0x84, 0xAC
HEARTS, PARTIAL = 0x66F, 0x670
SWORD_BLOCKED_LONG_TIMER = 0x4C
GANON_SCENE_PHASE = 0x445
RED_WIZZROBE, MAGIC_SHOT, GANON, BUBBLE = 0x24, 0x59, 0x3E, 0x2B


@pytest.fixture(scope='module')
def probe():
    require_rom()
    from triforce.paths import STATES_DIR
    from triforce.probe import Probe
    return Probe(STATES_DIR)


class Frames:
    """Steps ``game`` one frame at a time with Link's hearts held (and ``before(game)`` ahead of each frame)."""

    def __init__(self, game):
        self.game = game
        self.hearts, self.partial = game.read(HEARTS), game.read(PARTIAL)

    def step(self, before=None):
        if before is not None:
            before(self.game)
        self.game.press('NONE', 1)
        self.game.write(HEARTS, self.hearts)
        self.game.write(PARTIAL, self.partial)

    def until(self, done, limit, before=None):
        for _ in range(limit):
            if done(self.game):
                return
            self.step(before)
        raise AssertionError(f'not reached in {limit} frames')


def slots(game, types):
    return [s for s in range(1, 0xC) if game.read(TYPE + s) in types]


def entity(game, slot):
    """{feature: value} of NES slot ``slot``'s row in the observation of ``game``'s state."""
    from triforce._native import ENTITY_FEATURES, observe
    row = observe(game.state)['entity_features'][slot - 1]
    return {name: float(row[i]) for i, name in enumerate(ENTITY_FEATURES)}


def test_a_red_wizzrobe_shoots_when_its_phase_time_runs_out(probe):
    """6:0x78 (five Red Wizzrobes, secret trigger 2): slot 1 is the ringleader; at state $B0 it is solid with
    `phase_time` 0, and its magic shot ($59) is out by the frame after, where it stands
    (asm:Wizzrobe_Red_State2Solid)."""
    probe.enter_room(6, 0x78)
    game = probe.game
    frames = Frames(game)
    frames.step()
    assert game.read(TYPE + 1) == RED_WIZZROBE
    assert [entity(game, s)['ringleader'] for s in range(1, 7)] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    frames.until(lambda g: g.read(STATE + 1) == 0xBF, 200)
    solid = entity(game, 1)
    assert (solid['visible'], solid['touch_harms'], solid['weapon_target']) == (1.0, 1.0, 1.0)
    assert solid['phase_time'] == pytest.approx(15 / 512)
    frames.until(lambda g: g.read(STATE + 1) == 0xB1, 30)
    shots = set(slots(game, {MAGIC_SHOT}))
    frames.step()
    assert game.read(STATE + 1) == 0xB0 and entity(game, 1)['phase_time'] == 0.0
    frames.step()
    new = set(slots(game, {MAGIC_SHOT})) - shots
    assert any(abs(game.read(X + n) - game.read(X + 1)) <= 16 and abs(game.read(Y + n) - game.read(Y + 1)) <= 16
               for n in new), 'its magic shot came out'


def test_ganon_is_invisible_and_harmful_once_the_fight_starts(probe):
    """9:0x42 (left dark, so that the entry scene ends): Ganon_ScenePhase 0-1 nothing touches; from scene 2 he is
    invisible, harms on touch and the sword can hit him."""
    probe.enter_room(9, 0x42, lit=False)
    game = probe.game
    frames = Frames(game)
    frames.step()
    s = slots(game, {GANON})[0]
    scene = entity(game, s)
    assert (scene['touch_harms'], scene['weapon_target'], scene['visible']) == (0.0, 0.0, 1.0)
    frames.until(lambda g: g.read(GANON_SCENE_PHASE) == 2, 400)
    fight = entity(game, s)
    assert (fight['touch_harms'], fight['weapon_target'], fight['visible'], fight['phase']) == (1.0, 1.0, 0.0, 0.25)


def test_a_bubble_blocks_the_sword_and_the_block_counts_down(probe):
    """7:0x49: Link held on a Bubble until it touches him: `sword_blocked` 1, `sword_blocked_time` 1.0
    (SwordBlockedLongTimer $10), then 1/16 less every 10 frames (asm:Bubble_Update, asm:Nmi_Isr)."""
    from triforce._native import LINK_FEATURES, observe
    blocked_at, time_at = LINK_FEATURES.index('sword_blocked'), LINK_FEATURES.index('sword_blocked_time')

    def link(game):
        v = observe(game.state)['link']
        return float(v[blocked_at]), float(v[time_at])

    probe.enter_room(7, 0x49)
    game = probe.game
    frames = Frames(game)
    frames.until(lambda g: slots(g, {BUBBLE}), 120)
    s = slots(game, {BUBBLE})[0]
    assert link(game) == (0.0, 0.0)
    frames.until(lambda g: g.read(SWORD_BLOCKED_LONG_TIMER), 300,
                 before=lambda g: (g.write(X, g.read(X + s)), g.write(Y, g.read(Y + s))))
    assert link(game) == (1.0, 1.0)
    times = [1.0]
    for _ in range(32):
        frames.step()
        blocked, t = link(game)
        assert blocked == 1.0
        times.append(t)
    drops = [i for i in range(1, len(times)) if times[i] != times[i - 1]]
    assert times[drops[0]] == 0.9375 and drops[0] <= 10, 'the first 10-frame tick'
    assert [times[i] for i in drops] == [1.0 - k / 16 for k in range(1, len(drops) + 1)]
    assert len(drops) >= 3 and all(b - a == 10 for a, b in zip(drops, drops[1:]))
