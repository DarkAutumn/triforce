"""Observation schema 10 (round40.md "The Goriya teacher"): ``entity_shooter`` is ``_native.shooter_blocks`` of the same
RAM with ``obs.shooter`` on, all zero with it off (the default), and the switch changes nothing else the env returns;
against a recorded game the throw rule (``zelda::shooter``) says exactly when a blue Goriya throws and gives a red one's
throws their 2-in-256 chance."""

from pathlib import Path

import numpy as np

from triforce import _native
from triforce.observation import ENTITY_SHOOTER_FEATURES, observation_space
from triforce.specialist import load_specialist

from .conftest import REPO_STATES_DIR

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / 'corpus' / 'replays' / 'golden-d3first-51-networks.tfreplay'
ACTIONS = ['WAIT 8', 'MOVE N 8', 'SWORD N', 'MOVE W 8', 'WAIT 16', 'MOVE E 8', 'SWORD E', 'MOVE S 8']
OBJ_TYPE, OBJ_STATE, OBJ_SHOVE_DIR, OBJ_GRID_OFFSET = 0x34F, 0xAC, 0xC0, 0x394
CHASE_X, CHASE_Y, GAME_MODE = 0x61, 0x62, 0x12
# `_native.shooter_probe` columns.
P_GORIYA, P_LOADED, P_TIME, P_WILL, P_NEXT = range(5)


def _play(shooter: bool, steps: int = 60):
    config = load_specialist('goriya')
    config = config.replace(obs={**vars(config.obs), 'shooter': shooter})
    env = _native.Env(config, start_state='7_1bw_0', seed=3, strict_actions=False, states_dir=REPO_STATES_DIR)
    obs, _ = env.reset(seed=3)
    out = []
    for step in range(steps):
        ram = np.frombuffer(env.state.ram, dtype=np.uint8)[:0x800].copy()
        out.append((obs, ram))
        obs, _, terminated, truncated, _ = env.step(ACTIONS[step % len(ACTIONS)])
        if terminated or truncated:
            break
    return out


def test_the_block_is_shooter_blocks_with_the_switch_and_zero_without():
    on, off = _play(True), _play(False)
    assert len(on) == len(off) > 10
    space = observation_space()
    seen = np.zeros(len(ENTITY_SHOOTER_FEATURES))
    for (a, ram), (b, _) in zip(on, off):
        for key in a:
            if key != 'entity_shooter':
                assert np.array_equal(a[key], b[key]), key
        assert not b['entity_shooter'].any()
        assert np.array_equal(a['entity_shooter'], _native.shooter_blocks(ram[None])[0])
        assert space['entity_shooter'].contains(a['entity_shooter'])
        goriya = np.isin(a['entity_types'], (0x05, 0x06))
        assert not a['entity_shooter'][~goriya].any(), "only the Goriyas' rows"
        seen = np.maximum(seen, a['entity_shooter'].max(axis=0))
    names = list(ENTITY_SHOOTER_FEATURES)
    assert seen[names.index('loaded')] == 1 and seen[names.index('throw_time')] > 0, "7x1b's Blue Goriyas hold"


def test_a_recorded_game_throws_when_the_rule_says():
    """Every frame of the corpus's golden ``d3first`` run in a Goriya room (Dungeons 1, 2 and 7): a throw is the frame a
    Goriya's ObjState bit 7 rises. The probe of the frame before (``next``: every condition but a red one's random
    byte holds on the next frame) must agree with it exactly for a blue Goriya, but where the rule's input moved under
    it: the chase target moved and the Goriya decides on a grid point that frame (Link's own move comes first in the
    frame), or a knockback a wall cut short; a red one never throws without ``next``, and with it about 2 frames in
    256. ``will_throw`` (16 frames) is followed by a blue's throw in most frames that say 1 (Link's hits, stuns and
    moves cancel the rest)."""
    player = _native.ReplayPlayer(GOLDEN.read_bytes())
    prev = None
    counts = {'blue': np.zeros((2, 2), int), 'red': np.zeros((2, 2), int)}
    excused = 0
    windows, throws = [], set()
    while player.frame < player.frames:
        player.step(1, False)
        ram = np.asarray(player.ram(), dtype=np.uint8)[:0x800].copy()
        types = ram[OBJ_TYPE + 1:OBJ_TYPE + 12]
        if ram[GAME_MODE] != 5 or not np.isin(types, (0x05, 0x06)).any():
            prev = None
            continue
        probe = _native.shooter_probe(ram[None])[0]
        if prev is not None:
            pram, pprobe = prev
            for s in range(1, 12):
                t = pram[OBJ_TYPE + s]
                if t not in (0x05, 0x06) or ram[OBJ_TYPE + s] != t:
                    continue
                row = pprobe[s - 1]
                threw = not pram[OBJ_STATE + s] & 0x80 and bool(ram[OBJ_STATE + s] & 0x80)
                said = bool(row[P_NEXT])
                if threw:
                    throws.add((player.frame, s))
                if t == 0x05 and said != threw:
                    moved = (pram[CHASE_X], pram[CHASE_Y]) != (ram[CHASE_X], ram[CHASE_Y])
                    on_grid = ram[OBJ_GRID_OFFSET + s] & 0x0F == 0
                    if (moved and on_grid) or pram[OBJ_SHOVE_DIR + s]:
                        excused += 1
                        continue
                counts['blue' if t == 0x05 else 'red'][int(said), int(threw)] += 1
                if t == 0x05:
                    windows.append((player.frame, s, row[P_WILL]))
        prev = (ram, probe)
    blue, red = counts['blue'], counts['red']
    assert blue[1, 1] >= 50 and blue[0, 1] == 0 and blue[1, 0] == 0, blue      # exact, every throw said
    assert excused <= 5, excused
    assert red[0, 1] == 0 and red[1, 1] >= 10, red
    rate, n = red[1, 1] / red[1].sum(), red[1].sum()
    assert abs(rate - 2 / 256) < 4 * np.sqrt(2 / 256 * (1 - 2 / 256) / n), (rate, n)
    said = [(f, s) for f, s, w in windows if w == 1.0]
    followed = sum(any((f + k, s) in throws for k in range(16)) for f, s in said)
    assert len(said) > 500 and followed / len(said) > 0.7, (followed, len(said))
