"""Observation schema 10 (round40.md "The Goriya teacher", "Every shooter"): ``entity_shooter`` is
``_native.shooter_blocks`` of the same RAM with ``obs.shooter`` on, all zero with it off (the default), and the switch
changes nothing else the env returns; against recorded games the shot rules (``zelda::shooter``) say exactly when a
blue Goriya throws and give a red one's throws their 2-in-256 chance, and say every shot of the walking shooters
(Lynels, Moblins, Octoroks), the Zora, Aquamentus, Gohma and the red Wizzrobe on the frame before it and through the
16 frames before it, with the walking shooters' random gates read from the game's own register."""

import functools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from triforce import _native
from triforce.observation import ENTITY_SHOOTER_FEATURES, observation_space
from triforce.specialist import load_specialist

from .conftest import REPO_STATES_DIR

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / 'corpus' / 'replays' / 'golden-d3first-51-networks.tfreplay'
GOLDEN_D2 = ROOT / 'corpus' / 'replays' / 'golden-d2first-34-networks.tfreplay'
ACTIONS = ['WAIT 8', 'MOVE N 8', 'SWORD N', 'MOVE W 8', 'WAIT 16', 'MOVE E 8', 'SWORD E', 'MOVE S 8']
OBJ_TYPE, OBJ_DIR, OBJ_STATE, OBJ_SHOVE_DIR, OBJ_GRID_OFFSET = 0x34F, 0x98, 0xAC, 0xC0, 0x394
CHASE_X, CHASE_Y, GAME_MODE = 0x61, 0x62, 0x12
# `_native.shooter_probe` columns.
P_COVERED, P_LOADED, P_TIME, P_WILL, P_NEXT = range(5)
# The types `zelda::shooter::throw` covers.
COVERED = (0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x11, 0x3D, 0x33, 0x34, 0x24)


def _play(family: str, state: str, shooter: bool, actions=ACTIONS, steps: int = 60):
    config = load_specialist(family)
    config = config.replace(obs={**vars(config.obs), 'shooter': shooter})
    env = _native.Env(config, start_state=state, seed=3, strict_actions=False, states_dir=REPO_STATES_DIR)
    obs, _ = env.reset(seed=3)
    out = []
    for step in range(steps):
        ram = np.frombuffer(env.state.ram, dtype=np.uint8)[:0x800].copy()
        out.append((obs, ram))
        obs, _, terminated, truncated, _ = env.step(actions[step % len(actions)])
        if terminated or truncated:
            break
    return out


@pytest.mark.parametrize('family, state, types, actions', [
    ('goriya', '7_1bw_0', (0x05,), ACTIONS),
    ('ow-moblin', '0_52e_0', (0x03, 0x04), ['WAIT 8', 'SWORD W', 'MOVE W 8', 'WAIT 16', 'SWORD N', 'MOVE E 8'])])
def test_the_block_is_shooter_blocks_with_the_switch_and_zero_without(family, state, types, actions):
    """7x1b's Blue Goriyas and 0x52's Moblins: the block fills the covered shooters' rows and nothing else."""
    on, off = _play(family, state, True, actions), _play(family, state, False, actions)
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
        covered = np.isin(a['entity_types'], COVERED)
        assert not a['entity_shooter'][~covered].any(), "only the covered shooters' rows"
        seen = np.maximum(seen, a['entity_shooter'][np.isin(a['entity_types'], types)].max(axis=0, initial=0))
    names = list(ENTITY_SHOOTER_FEATURES)
    assert seen[names.index('loaded')] == 1 and seen[names.index('throw_time')] > 0, f"{state}'s shooters hold"


def _kept_shove(ram, s) -> bool:
    """A knockback the game keeps (asm:Shove_Update drops one across the facing axis off a grid point on its first
    frame): the only knockback whose end the rule does not walk exactly (a wall can end it sooner)."""
    shove = int(ram[OBJ_SHOVE_DIR + s])
    across = shove & (0x0C if ram[OBJ_DIR + s] < 3 else 0x03)
    return shove != 0 and not (shove & 0x80 and across and ram[OBJ_GRID_OFFSET + s] != 0)


def test_a_recorded_game_throws_when_the_rule_says():
    """Every frame of the corpus's golden ``d3first`` run in a Goriya room (Dungeons 1, 2 and 7): a throw is the frame a
    Goriya's ObjState bit 7 rises. The probe of the frame before (``next``: every condition but a red one's random
    byte holds on the next frame) must agree with it exactly for a blue Goriya, but where the rule's input moved under
    it: the chase target moved and the Goriya decides on a grid point that frame (Link's own move comes first in the
    frame), or a knockback the game keeps (a wall may cut it short); a red one never throws without ``next``, and with
    it about 2 frames in 256. Every blue throw had ``will_throw`` = 1 on each of the 15 frames before it (its spawn
    cloud's included), but where the chase target moved between that frame and the throw, a kept knockback ran, or the
    room had no free slot or 4 shots flying then and a shot ended since (the rule reads them as they stand); and
    ``will_throw`` = 1 is followed by a blue's throw in most frames (Link's hits, stuns and moves cancel the rest)."""
    player = _native.ReplayPlayer(GOLDEN.read_bytes())
    prev = None
    counts = {'blue': np.zeros((2, 2), int), 'red': np.zeros((2, 2), int)}
    excused = 0
    windows, throws = [], set()
    history = {}                # frame -> (chase target, room for a shot, {blue slot: (will_throw, kept knockback)})
    while player.frame < player.frames:
        player.step(1, False)
        ram = np.asarray(player.ram(), dtype=np.uint8)[:0x800].copy()
        types = ram[OBJ_TYPE + 1:OBJ_TYPE + 12]
        if ram[GAME_MODE] != 5 or not np.isin(types, (0x05, 0x06)).any():
            prev = None
            continue
        probe = _native.shooter_probe(ram[None])[0]
        room = bool(np.any(types == 0)) and ram[0x34C] < 4
        history[player.frame] = ((int(ram[CHASE_X]), int(ram[CHASE_Y])), room,
                                 {s: (float(probe[s - 1][P_WILL]), _kept_shove(ram, s))
                                  for s in range(1, 12) if ram[OBJ_TYPE + s] == 0x05})
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
                    throws.add((player.frame, s, int(t)))
                if t == 0x05 and said != threw:
                    moved = (pram[CHASE_X], pram[CHASE_Y]) != (ram[CHASE_X], ram[CHASE_Y])
                    on_grid = ram[OBJ_GRID_OFFSET + s] & 0x0F == 0
                    if (moved and on_grid) or _kept_shove(pram, s):
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
    # The window: each blue throw at frame f said 1 on frames f-15..f-1 (the probe of frame g covers g+1..g+16).
    checked, window_excused, missed = 0, 0, []
    for f, s, t in sorted(throws):
        if t != 0x05:
            continue
        target = history[f][0]
        for g in range(f - 15, f):
            if g not in history or s not in history[g][2]:
                continue                                   # before the room's first play frame
            checked += 1
            if history[g][2][s][0] == 1.0:
                continue
            span = [history[k] for k in range(g, f) if k in history]
            if (any(h[0] != target for h in span) or any(h[2].get(s, (0, False))[1] for h in span)
                    or not history[g][1]):
                window_excused += 1
                continue
            missed.append((f, s, g, history[g][2][s][0]))
    assert checked >= 50 * 10 and not missed, (checked, window_excused, missed[:10])
    assert window_excused <= checked // 5, (window_excused, checked)
    said = [(f, s) for f, s, w in windows if w == 1.0]
    blue_throws = {(f, s) for f, s, t in throws if t == 0x05}
    followed = sum(any((f + k, s) in blue_throws for k in range(16)) for f, s in said)
    assert len(said) > 500 and followed / len(said) > 0.7, (followed, len(said))


# ---------------------------------------------------------------------------------------------- every other shooter

OBJ_X, OBJ_Y, OBJ_TIMER, OBJ_METASTATE, OBJ_STUN, OBJ_QSPEED = 0x70, 0x84, 0x28, 0x405, 0x3D, 0x3BC
SHOOT_TIMER, WANTS, INVINCIBLE, UNINITIALIZED, TURN_TIMER = 0x451, 0x412, 0x4F0, 0x492, 0x478
GOHMA_SHOOT_TIMER, SHOT_COUNT, FRAME_COUNTER, RANDOM = 0x380, 0x34C, 0x15, 0x18
# What freezes every update for a frame (asm:Mode5Play_Update): the flute's tune, a pause, the submenu, a lighting room.
FREEZES = (0x3C, 0xE0, 0xE1, 0x51E)
FAMILY = {0x01: 'lynel', 0x02: 'lynel', 0x03: 'moblin', 0x04: 'moblin', 0x07: 'octorok', 0x08: 'octorok',
          0x09: 'octorok', 0x0A: 'octorok', 0x11: 'zora', 0x3D: 'aquamentus', 0x33: 'gohma', 0x34: 'gohma',
          0x24: 'red wizzrobe'}
WALKING = (0x01, 0x02, 0x03, 0x04, 0x07, 0x08, 0x09, 0x0A)
RED_WALKING = (0x02, 0x04, 0x07, 0x08)
# Wanderers' turn rates (asm:Moblin_Update, asm:Octorok_Update); a Lynel decides by the Goriya rule.
TURN_RATE = {0x03: 0xA0, 0x04: 0xA0, 0x07: 0x70, 0x08: 0x70, 0x09: 0xA0, 0x0A: 0xA0}
# A shot type and the shooters that make it.
SHOTS = {0x53: (0x07, 0x08, 0x09, 0x0A), 0x57: (0x01, 0x02), 0x5B: (0x03, 0x04), 0x55: (0x11, 0x3D),
         0x56: (0x33, 0x34), 0x59: (0x24,)}


def _signature(pram, ram, s) -> bool:
    """The shooter's own record of the frame its rule fires (made or refused): the shoot timer reaching `$10`, the
    Zora's update reading `$FD`, Aquamentus's and Gohma's timers reloaded, the red Wizzrobe reaching `$B0`."""
    t = int(pram[OBJ_TYPE + s])
    if t in WALKING:
        return pram[SHOOT_TIMER + s] == 0x11 and ram[SHOOT_TIMER + s] == 0x10
    if t == 0x11:
        return pram[OBJ_STATE + s] == 3 and pram[OBJ_TIMER + s] == 0xFE
    if t == 0x3D:
        return pram[OBJ_TIMER + s] <= 1 and ram[OBJ_TIMER + s] >= 0x70
    if t in (0x33, 0x34):
        return pram[GOHMA_SHOOT_TIMER + s] == 1 and ram[GOHMA_SHOOT_TIMER + s] == 0x41
    return t == 0x24 and pram[OBJ_STATE + s] == 0xB1 and ram[OBJ_STATE + s] == 0xB0


def _shots_made(pram, ram) -> set:
    """The shooter slots whose shot appeared this frame: each new shot object (a slot taking a shot type, or a shot
    slot uninitialized again: one ended and another took it in the same frame), one to one with the nearest shooter of
    a type that makes it, a shooter whose own rule fired this frame first. Aquamentus's two other fireballs and the
    shots of shooters the rule does not cover (statues, Manhandla, Gleeok) stay unmatched."""
    fresh = [k for k in range(1, 12) if int(ram[OBJ_TYPE + k]) in SHOTS
             and (pram[OBJ_TYPE + k] != ram[OBJ_TYPE + k] or (ram[UNINITIALIZED + k] and not pram[UNINITIALIZED + k]))]
    pairs = sorted((not _signature(pram, ram, s),
                    max(abs(int(ram[OBJ_X + k]) - int(pram[OBJ_X + s])), abs(int(ram[OBJ_Y + k]) - int(pram[OBJ_Y + s]))),
                    k, s)
                   for k in fresh for s in range(1, 12)
                   if s != k and int(pram[OBJ_TYPE + s]) in SHOTS[int(ram[OBJ_TYPE + k])]
                   and ram[OBJ_TYPE + s] == pram[OBJ_TYPE + s])
    made, used = set(), set()
    for _, distance, k, s in pairs:
        if distance <= 12 and k not in used and s not in made:
            made.add(s)
            used.add(k)
    return made


def _room_for(ram, t) -> bool:
    """A free monster slot and, for a counted shot, fewer than 4 flying."""
    counted = t in WALKING or t == 0x24
    return bool(np.any(ram[OBJ_TYPE + 1:OBJ_TYPE + 12] == 0)) and (not counted or ram[SHOT_COUNT] < 4)


def _held(pram, ram) -> bool:
    """No update ran its course this frame: a lag frame (the frame counter did not move) or a freeze."""
    return ram[FRAME_COUNTER] == pram[FRAME_COUNTER] or any(ram[a] for a in FREEZES)


@functools.lru_cache(maxsize=None)
def _walk(path: Path) -> dict:
    """Every play frame of a recorded game with a covered shooter other than a Goriya: per type, the probe of the frame
    before against the shots made (`said`, `made`: a 2x2 count; unexcused disagreements listed), each shot's 16
    frames before, `will_throw` = 1 followed by its shot, the walking shooters' start gate and wanderers' wish against
    the random register."""
    player = _native.ReplayPlayer(path.read_bytes())
    out = {'table': defaultdict(lambda: np.zeros((2, 2), int)), 'wrong': [], 'excused': defaultdict(int),
           'windows': defaultdict(lambda: [0, 0]), 'window_missed': [], 'followed': {},
           'start': np.zeros((2, 2), int), 'start_rate': [0, 0], 'wish': np.zeros((2, 2), int)}
    history, shots, freed, held, prev = {}, [], set(), set(), None
    while player.frame < player.frames:
        player.step(1, False)
        ram = np.asarray(player.ram(), dtype=np.uint8)[:0x800].copy()
        if ram[GAME_MODE] != 5 or not np.isin(ram[OBJ_TYPE + 1:OBJ_TYPE + 12], tuple(FAMILY)).any():
            prev = None
            continue
        probe = _native.shooter_probe(ram[None])[0]
        history[player.frame] = {s: (float(probe[s - 1][P_WILL]), _room_for(ram, int(ram[OBJ_TYPE + s])),
                                     FAMILY[int(ram[OBJ_TYPE + s])])
                                 for s in range(1, 12) if int(ram[OBJ_TYPE + s]) in FAMILY}
        if prev is not None:
            pram, pprobe = prev
            if _held(pram, ram):
                held.add(player.frame)
            made = _shots_made(pram, ram)
            for s in range(1, 12):
                t = int(pram[OBJ_TYPE + s])
                if t not in FAMILY or ram[OBJ_TYPE + s] != t:
                    continue
                said, shot = bool(pprobe[s - 1][P_NEXT]), s in made
                if shot:
                    shots.append((player.frame, s, FAMILY[t]))
                if said and not shot and (_held(pram, ram) or not _room_for(ram, t)):
                    out['excused']['held, or the room filled first'] += 1        # a shot of a higher slot came first
                    continue
                if shot and not said and not _room_for(pram, t):
                    out['excused']['a shot ended first'] += 1          # a higher slot's shot ended, freeing the room
                    freed.add((player.frame, s))
                    continue
                hit = ram[INVINCIBLE + s] > pram[INVINCIBLE + s] or ram[OBJ_STUN + s] > pram[OBJ_STUN + s]
                if said and not shot and t in WALKING and hit:
                    out['excused']['hit or stunned'] += 1
                    continue
                out['table'][FAMILY[t]][int(said), int(shot)] += 1
                if said != shot:
                    out['wrong'].append((player.frame, s, hex(t), said, shot))
                if t in WALKING and not _held(pram, ram) and pram[OBJ_METASTATE + s] == 0 \
                        and not pram[UNINITIALIZED + s]:
                    _gates(out, pram, ram, s, t)
        prev = (ram, probe)
    made = {(f, s) for f, s, _ in shots}
    for f, s, family in shots:
        for g in range(f - 16, f):
            if g not in history or s not in history[g]:
                continue                                            # before the room's first play frame
            will, room, _ = history[g][s]
            out['windows'][family][0] += 1
            if will == 1.0:
                continue
            later = [history[k][s][1] for k in range(g, f) if k in history and s in history[k]]
            if not room and (any(later) or (f, s) in freed):
                out['windows'][family][1] += 1                      # no room then, freed since
                continue
            out['window_missed'].append((f, s, family, g))
    for g, rows in history.items():
        for s, (will, _, family) in rows.items():
            if will == 1.0 and not any(g + k in held for k in range(1, 17)):     # a freeze delays every shot
                out['followed'][s, g] = (family, any((g + k, s) in made for k in range(1, 17)))
    return out


def _gates(out, pram, ram, s, t):
    """The walking shooters' random gates read from the register the game steps before the frame's updates (`$18-$24`,
    the frame's own bytes): a red one starts its stand with the timer at 0 and the wish standing only when `Random,X`
    is `$F8` or more (asm:Shooter_RunShootTimer); a wanderer at a grid point wishes to shoot when it turns, toward the
    chase target in line on a turn rate at least `Random+1,X` or perpendicular when its turn timer ran out
    (asm:Wanderer_MoveAndTurn)."""
    if pram[SHOOT_TIMER + s] == 0 and ram[WANTS + s] and not pram[INVINCIBLE + s] and not ram[INVINCIBLE + s]:
        started = ram[SHOOT_TIMER + s] == 0x30
        if t in RED_WALKING:
            out['start'][int(ram[RANDOM + s] >= 0xF8), int(started)] += 1
            out['start_rate'][0] += 1
            out['start_rate'][1] += int(started)
        else:
            out['start'][1, int(started)] += 1
    if t in TURN_RATE and pram[OBJ_QSPEED + s] and not ram[OBJ_SHOVE_DIR + s] and ram[OBJ_GRID_OFFSET + s] & 0x0F == 0 \
            and ram[SHOOT_TIMER + s] in (0, 0x30):
        def gap(a, b):
            return abs(((int(a) - int(b) + 128) & 0xFF) - 128)
        in_line = gap(ram[CHASE_X], ram[OBJ_X + s]) < 9 or gap(ram[CHASE_Y], ram[OBJ_Y + s]) < 9
        wish = (in_line and TURN_RATE[t] >= ram[RANDOM + 1 + s]) or pram[TURN_TIMER + s] <= 1
        out['wish'][int(wish), int(ram[WANTS + s] != 0)] += 1


# The fewest shots of each type a game holds (both golden runs hold more).
FEWEST = {'lynel': 15, 'moblin': 45, 'octorok': 35, 'zora': 30, 'aquamentus': 4, 'gohma': 1, 'red wizzrobe': 35}
# The least share of a type's frames saying `will_throw` = 1 (out of freezes) that its shot follows within 16 frames.
FOLLOWED = 0.8


@pytest.mark.parametrize('path', [GOLDEN, GOLDEN_D2], ids=['d3first', 'd2first'])
def test_every_shot_is_said_on_the_frame_before_it(path):
    """The probe of the frame before (``next``: every condition holds on the next frame) against the shots made, for
    every covered type but the Goriyas (above): exact, but where the rule's reading of the room moved under it in the
    frame itself: a lag frame or a freeze held every update, a higher slot's shot filled the room first or ended and
    freed it, Link's hit or stun landed on a walking shooter."""
    out = _walk(path)
    for family, fewest in FEWEST.items():
        table = out['table'][family]
        assert table[1, 1] >= fewest, (family, table)
    assert not out['wrong'], out['wrong'][:10]
    assert sum(out['excused'].values()) <= 10, dict(out['excused'])


@pytest.mark.parametrize('path', [GOLDEN, GOLDEN_D2], ids=['d3first', 'd2first'])
def test_every_shot_had_will_throw_on_the_16_frames_before_it(path):
    """``will_throw`` = 1 on each of the 16 frames before every shot (the probe of frame g covers g+1..g+16), but where
    the room had no space then and a shot ended since; and for each type most frames saying 1 are followed by the shot
    (Link's hits, stuns and the room filling cancel the rest), but Gohma: Link's arrows into its eye kill it within a
    few of its 65-frame cycles, and the frames saying 1 before its death are a large share of its few."""
    out = _walk(path)
    checked = sum(v[0] for v in out['windows'].values())
    excused = sum(v[1] for v in out['windows'].values())
    assert checked >= 2000 and not out['window_missed'], (checked, out['window_missed'][:10])
    assert excused <= checked // 50, (excused, checked)
    followed = defaultdict(list)
    for family, hit in out['followed'].values():
        followed[family].append(hit)
    rates = {family: (sum(hits), len(hits)) for family, hits in followed.items()}
    for family in FEWEST:
        if family != 'gohma':
            hits, frames = rates.get(family, (0, 0))
            assert frames >= 40 and hits / frames > FOLLOWED, (family, rates)


def test_the_walking_shooters_random_gates_are_the_registers():
    """The two random gates of the walking shooters, outside the 16 frames (a stand starts 33 frames before its shot),
    read from the game's register: a red one starts its stand exactly when ``Random,X`` is ``$F8`` or more, a blue one
    whenever it may; a wanderer's wish is exactly its turn (in line on a turn rate at least ``Random+1,X``, or its turn
    timer out). The red gate's rate over the frames it could start: the register shifts one bit a frame, so after a
    frame that failed the next passes 1 time in 62 (it needs the old byte's top four bits set and its fifth clear, and
    a new bit set), below a fair byte's 8 in 256."""
    out = _walk(GOLDEN)
    start, wish = out['start'], out['wish']
    assert start[0, 1] == 0 and start[1, 0] == 0 and start[1, 1] >= 100, start
    assert wish[0, 1] == 0 and wish[1, 0] == 0 and wish[1, 1] >= 200 and wish[0, 0] >= 400, wish
    eligible, started = out['start_rate']
    rate = started / eligible
    assert 1 / 100 < rate < 8 / 256, (started, eligible)
