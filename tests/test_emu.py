"""triforce.emu: one room played by hand, with no window: the start, the DONE rule, resets, the recording."""

import hashlib
import re
from datetime import datetime

import pytest

from triforce import _native, emu
from triforce.emu import DOWN, LEFT, RIGHT, UP, Spec, Where

from .conftest import REPO_STATES_DIR, require_rom

PRESS = {'U': UP, 'D': DOWN, 'L': LEFT, 'R': RIGHT, 'N': 0}


def find_start(spec: Spec) -> emu.Start:
    return emu.find_start(spec, REPO_STATES_DIR)


def session_of(spec: Spec, seed: int) -> emu.Session:
    require_rom()
    return emu.Session(spec, seed=seed, states_dir=REPO_STATES_DIR)


def play(session, script: str) -> list:
    """Runs ``script`` ('R200U40': a d-pad direction or N for none, then frames) until it ends or the take is DONE;
    returns where Link was each time it changed: (GameMode, level, room, in_cave)."""
    seen = []
    for key, frames in re.findall(r'([UDLRN])(\d+)', script):
        for _ in range(int(frames)):
            done = session.step(PRESS[key])
            state = session.state
            where = (state.mode, *state.full_location)
            if not seen or seen[-1] != where:
                seen.append(where)
            if done:
                return seen
    return seen


def picture_hash(session) -> str:
    return hashlib.sha1(session.screen().tobytes()).hexdigest()


# ---------------------------------------------------------------------------------------------- the spec

def test_rooms_and_levels_in_the_owners_notation():
    assert (emu.parse_level('3'), emu.parse_level('o'), emu.parse_level('0')) == (3, 0, 0)
    assert (emu.parse_room('59'), emu.parse_room('0f'), emu.parse_room('0x7C')) == (0x59, 0x0f, 0x7c)
    assert (emu.room_text(3, 0x59), emu.room_text(0, 0x1f)) == ('3x59', 'ox1f')
    for bad in ('10', 'x', ''):
        with pytest.raises(ValueError):
            emu.parse_level(bad)
    for bad in ('zz', '80'):
        with pytest.raises(ValueError):
            emu.parse_room(bad)


def test_the_route_start_its_entries_visits_and_chains():
    assert find_start(Spec(3, 0x59)) == emu.Start('d3first_3_59n_0', 'N')        # the route's first entry
    assert find_start(Spec(3, 0x59, entry='S', chain=2)) == emu.Start('d3first_3_59s_2', 'S')
    assert find_start(Spec(3, 0x69, entry='cave')) == emu.Start('d3first_3_69c_0', 'cave')
    assert find_start(Spec(3, 0x0f)) == emu.Start('d3first_3_0fcave_0', 'S')      # the raft's cellar
    assert find_start(Spec(5, 0x05)) == emu.Start('d3first_5_05e_0', 'E')         # the Olive Garden, from E
    assert find_start(Spec(3, 0x59, route='d2first')).state == '3_59e_0'          # bare names: d2first's, from E
    with pytest.raises(emu.StartError, match='never enters 3x22.*: 0f 3d 49'):
        find_start(Spec(3, 0x22))
    with pytest.raises(emu.StartError, match='by N, S, not E'):
        find_start(Spec(3, 0x59, entry='E'))
    with pytest.raises(emu.StartError, match='visit 3 chain 0.*visit 1 chain 0'):
        find_start(Spec(3, 0x59, visit=3))
    with pytest.raises(emu.StartError, match='no route'):
        find_start(Spec(3, 0x59, route='nope'))


def test_overrides_write_bombs_keys_and_hearts_as_the_env_does():
    assert emu.overrides(Spec(3, 0x59, bombs=4, keys=2), 3) == {'bombs': 4, 'keys': 2}
    assert emu.overrides(Spec(3, 0x59, hearts=3), 3) == {'hearts_and_containers': 0x22, 'partial_hearts': 0xFF}
    assert emu.overrides(Spec(3, 0x59, hearts=2), 3) == {'hearts_and_containers': 0x21, 'partial_hearts': 0xFF}
    assert emu.overrides(Spec(3, 0x59, hearts=1.5), 3) == {'hearts_and_containers': 0x21, 'partial_hearts': 0x7F}
    assert emu.overrides(Spec(3, 0x59, hearts=5), 3) == {'hearts_and_containers': 0x44, 'partial_hearts': 0xFF}
    assert emu.overrides(Spec(5, 0x05, hearts=2), 8) == dict(_native.hearts_encoding(4, 8))   # the env's function
    with pytest.raises(ValueError):
        emu.overrides(Spec(3, 0x59, hearts=0), 3)


def hud_hearts(session) -> float:
    """The hearts the HUD draws on the last frame: its bottom row of heart tiles (hearts 1-8, from x 176 at y 48),
    a red right half a full heart, a red left half alone a half heart."""
    picture = session.screen()
    count = 0.0
    for i in range(8):
        cell = picture[48:56, 176 + 8 * i:184 + 8 * i]
        red = (cell[..., 0] > 150) & (cell[..., 1] < 100)
        count += 1.0 if red[:, 4:].any() else 0.5 if red[:, :4].any() else 0.0
    return count


def darknut_touches(session, limit: int = 1, frames: int = 600) -> list:
    """A Blue Darknut moved onto Link every frame (by memory, as the walk tests force hits; not recorded as play):
    the (HeartValues, HeartPartial) after each touch, until ``limit`` touches or DONE."""
    seen, last = [], (session.state.read(0x66F), session.state.read(0x670))
    for _ in range(frames):
        state = session.state
        foes = [e for e in state.enemies if e['name'] == 'BlueDarknut' and e['active']]
        if foes:
            session.game.poke(0x70 + foes[0]['index'], state.link['x'])
            session.game.poke(0x84 + foes[0]['index'], state.link['y'])
        session.step(0)
        now = (session.state.read(0x66F), session.state.read(0x670))
        if now != last:
            seen.append(now)
            last = now
        if session.done or len(seen) >= limit:
            break
    return seen


@pytest.mark.parametrize('hearts, touches, died', [(2, 1, True), (2.5, 1, False), (2.5, 2, True), (0.5, 1, True)])
def test_hearts_in_the_olive_garden_die_as_the_game_says(hearts, touches, died):
    """``--hearts 2``: a Blue Darknut's touch (two hearts; Link holds no ring at the route's Olive Garden) kills;
    2.5 survive one; half a heart dies to anything.  The status line's hearts are what the HUD draws from the second
    frame on (the first frame's picture is drawn before the game rewrites the status bar's text)."""
    session = session_of(Spec(5, 0x05, bombs=0, hearts=hearts), 1)
    assert session.state.read(0x662) == 0                                   # InvRing: no ring
    for _ in range(2):
        session.step(0)
    assert hud_hearts(session) == session.state.health == hearts
    darknut_touches(session, touches)
    assert (session.done == 'died') == died
    assert _native.replay_verify(session.replay())['frames'] == session.frames


# ---------------------------------------------------------------------------------------------- DONE

def test_the_room_rule_on_sequences_of_where_link_is():
    home = Where(3, 0x69, False, True, False)
    rule = emu.DONE_RULES['room']
    stairs = [Where(3, 0x0f, False, False, False),        # down the stairs (mode 16: the room id is the cellar's)
              Where(3, 0x0f, True, True, False),          # in the cellar, on the raft's ledge
              Where(3, 0x69, False, False, False),        # up the ladder (mode 10, then 4)
              Where(3, 0x69, False, True, False)]         # back in play in 3x69
    assert [rule(home, now) for now in stairs] == [None] * 4
    assert rule(home, Where(3, 0x59, False, False, False)) is None                    # scrolling N
    assert rule(home, Where(3, 0x59, False, True, False)) == 'left for 3x59'
    assert rule(home, Where(0, 0x74, False, True, False)) == 'left for ox74'           # out of the level
    assert rule(home, Where(3, 0x69, False, False, True)) == 'died'
    overworld = Where(0, 0x0f, False, True, False)
    assert rule(overworld, Where(0, 0x0f, True, True, False)) is None                  # a cave under the screen
    assert rule(overworld, Where(0, 0x0e, False, True, False)) == 'left for ox0e'
    cellar = Where(3, 0x0f, True, True, False)
    assert rule(cellar, Where(3, 0x69, False, True, False)) == 'left for 3x69'         # up out of a cellar start


def test_the_cellar_under_the_room_is_the_same_take_and_the_next_room_ends_it():
    """3x69 from its stairs: right onto them, down to the raft's cellar, up the ladder, then N into 3x59.  Back in
    3x69 Link walks right until he is in the N door's column (X $78), then up.  (A fixed 24 frames right brought him
    to X $7B on main's start state, from which he turns into the doorway, and to X $80 on the generation's rebuilt
    one, beside it, where he stood until the Darknuts killed him.)"""
    session = session_of(Spec(3, 0x69, entry='cave'), 1)
    seen = play(session, 'R200N200U200')
    assert (9, 3, 0x0f, True) in seen and seen[-1] == (5, 3, 0x69, False)            # there and back
    assert session.done is None
    for _ in range(60):                         # right to the door's column (asm:ObjX, Link's slot 0)
        if session.state.ram[0x70] >= 0x78 or session.step(RIGHT):
            break
    seen = play(session, 'U200')
    assert session.done == 'left for 3x59' and seen[-1] == (5, 3, 0x59, False)
    with pytest.raises(RuntimeError, match='DONE'):
        session.step(0)


def test_death_ends_the_take():
    """Into the Darknuts with half a heart (the first touch kills; with the start state's own hearts whether he
    dies within the 200 frames depended on that state's contents)."""
    session = session_of(Spec(3, 0x69, entry='cave', hearts=0.5), 1)
    play(session, 'D4R200')
    assert session.done == 'died' and session.state.game_over


# ---------------------------------------------------------------------------------------------- resets

def test_a_start_loads_ready_and_a_reset_with_the_same_seed_plays_the_same_frames():
    session = session_of(Spec(3, 0x59), 7)
    assert session.done is None and session.seed == 7 and session.home == Where(3, 0x59, False, True, False)
    loaded = session.frames                     # the load's frames to control, before any button
    assert 0 < loaded < 120 and session.screen().shape == (240, 256, 3)

    def run():
        hashes = [picture_hash(session)]
        for frame in range(150):
            session.step(DOWN if frame < 60 else LEFT)
            hashes.append(picture_hash(session))
        return hashes

    first = run()
    assert session.frames == loaded + 150
    session.reset(7)
    assert session.frames == loaded and session.done is None
    assert run() == first
    session.reset(8)
    assert run() != first                        # the monsters walk otherwise
    session.reset()
    assert 0 <= session.seed < emu.SEED_RANGE


# ---------------------------------------------------------------------------------------------- the recording

def test_the_saved_replay_verifies_replays_to_the_same_frame_and_keeps_the_bombs(tmp_path):
    session = session_of(Spec(3, 0x59, bombs=4), 123)
    assert session.state.bombs == 4
    play(session, 'D40L30N10')
    when = datetime(2026, 11, 24, 10, 15, 2)
    path = session.save(tmp_path / 'played', when)
    assert path.name == '3x59_N_bombs4_seed123_261124-101502.tfreplay'
    data = path.read_bytes()
    recording = _native.Replay(data)
    assert recording.frames == session.frames and recording.ending == 'unknown'
    meta = recording.meta
    assert meta['kind'] == 'human' and meta['seed'] == 123 and meta['route'] == 'd3first'
    assert meta['human']['state'] == 'd3first_3_59n_0' and meta['human']['overrides'] == {'bombs': 4}
    assert recording.source['name'] == 'd3first_3_59n_0'                              # embedded, by its name
    edits = recording.edits()
    bombs = [tuple(int(v) for v in e) for e in edits if e[1] == 0x658]
    assert bombs == [(0, 0x658, bombs[0][2], 4)] and bombs[0][2] != 4                # a frame-0 EDIT
    assert {e[1] for e in edits if e[0] == 0} >= set(range(0x18, 0x25))               # the RNG phase
    assert _native.replay_verify(data)['frames'] == session.frames

    player = _native.ReplayPlayer(data)
    player.step(recording.frames, True)
    assert (player.screen() == session.screen()).all()

    play(session, 'R20')                         # saving keeps the recording going
    again = _native.Replay(session.replay())
    assert again.frames == recording.frames + 20


def test_endings_of_a_done_take(tmp_path):
    session = session_of(Spec(3, 0x69, entry='cave', hearts=0.5), 1)     # the first touch kills (test_death_...)
    play(session, 'D4R200')
    recording = _native.Replay(session.replay())
    assert recording.ending == 'death' and recording.meta['human']['done'] == 'died'
    assert _native.replay_verify(session.replay())['frames'] == session.frames


def test_the_olive_garden_from_its_route_entry_with_no_bombs():
    """The owner's first take: ``emu-zelda.py 5 05 --bombs 0``, Dungeon 5's six Blue Darknuts by the sword alone."""
    session = session_of(Spec(5, 0x05, bombs=0), 5)
    assert session.start == emu.Start('d3first_5_05e_0', 'E') and session.state.bombs == 0
    assert session.home == Where(5, 0x05, False, True, False)
    play(session, 'L30N30')
    assert session.done is None
    assert session.file_name(datetime(2026, 11, 24, 10, 15, 2)) == '5x05_E_bombs0_seed5_261124-101502.tfreplay'
    assert _native.replay_verify(session.replay())['frames'] == session.frames


def test_an_agent_reads_link_and_the_monsters_frame_by_frame():
    """The recipe of docs/architecture.md "Playing a room by hand": a ReplayPlayer stepped a frame at a time, its
    machine read through Game.load(save_state(), controllable=True)."""
    session = session_of(Spec(5, 0x05, bombs=0), 3)
    seen = []
    for _ in range(40):
        session.step(LEFT)
        seen.append((session.state.link['x'], [e['name'] for e in session.state.enemies]))
    data = session.replay()
    player = _native.ReplayPlayer(data)
    reader = _native.Game(states_dir=str(REPO_STATES_DIR))
    player.step(_native.Replay(data).frames - 40, False)
    read = []
    for _ in range(40):
        player.step(1, False)
        state = reader.load(player.save_state(), randomize_rng=False, controllable=True)
        read.append((state.link['x'], [e['name'] for e in state.enemies]))
    assert read == seen and 'BlueDarknut' in read[-1][1]


# ---------------------------------------------------------------------------------------------- --item

B_BOX = (slice(24, 48), slice(120, 144))          # the status bar's B box (y, x)


def test_the_b_item_is_selected_at_the_load_and_drawn():
    """``--item``: `SelectedItemSlot` ($656) written at frame 0, held through the frames to control, the HUD's B box
    drawing it after the first frame; only items Link holds (the Olive Garden's start: the boomerang and bombs)."""
    def box(spec):
        session = session_of(spec, 1)
        session.step(0)
        edits = [tuple(int(v) for v in e) for e in _native.Replay(session.replay()).edits() if e[1] == 0x656]
        return session.state.read(0x656), edits, session.screen()[B_BOX].tobytes(), session
    plain = box(Spec(5, 0x05))
    boomerang = box(Spec(5, 0x05, item='boomerang'))
    bombs = box(Spec(5, 0x05, bombs=3, item='bombs'))
    assert list(plain[3].state.b_slots) == [0, 1] and plain[0] == 0 and plain[1] == []
    assert boomerang[:3] == (0, [(0, 0x656, 0, 0)], plain[2])
    assert bombs[0] == 1 and bombs[1] == [(0, 0x656, 0, 1)] and bombs[2] != plain[2]
    assert _native.replay_verify(bombs[3].replay())['frames'] == bombs[3].frames
    assert bombs[3].file_name(datetime(2026, 11, 24, 10, 15, 2)) == \
        '5x05_E_bombs3_itembombs_seed1_261124-101502.tfreplay'
    with pytest.raises(ValueError, match='candle: Link does not hold it at this start; he holds boomerang, bombs'):
        session_of(Spec(5, 0x05, item='candle'), 1)
    with pytest.raises(ValueError, match=r'bombs: .*StatusBar_CheckMissingItem.*he holds boomerang$'):
        session_of(Spec(5, 0x05, bombs=0, item='bombs'), 1)


# ---------------------------------------------------------------------------------------------- --enemies

OBJ_TYPE = 0x34F                 # ObjType, slots 0 (Link) to 11 (zelda::tables::OBJ_ID)


def monsters(session, frames: int = 120) -> list:
    for _ in range(frames):
        session.step(0)
    return [(e['index'], e['name'], e['x'], e['y']) for e in session.state.enemies]


def kill(session, slots, frames: int = 64):
    """Empties ``slots`` (the room counts those monsters dead, as the walk tests kill: tests/test_d5.py) and lets the
    game notice; the writes are EDITS of the take."""
    for slot in slots:
        session.game.poke(OBJ_TYPE + slot, 0)
    for _ in range(frames):
        session.step(0)


def test_enemies_leaves_n_drawn_from_the_seed():
    """``--enemies 2`` in 3x59 (5 Red Darknuts) and the Olive Garden (6 Blue Darknuts): the game makes 2, where its
    spawn draw from the seed puts them: the same seed the same, another seed elsewhere."""
    first = monsters(session_of(Spec(3, 0x59, enemies=2), 1))
    assert len(first) == 2 and {name for _, name, _, _ in first} == {'RedDarknut'}
    assert monsters(session_of(Spec(3, 0x59, enemies=2), 1)) == first
    assert monsters(session_of(Spec(3, 0x59, enemies=2), 2)) != first
    assert len(monsters(session_of(Spec(3, 0x59), 1))) == 5
    garden = session_of(Spec(5, 0x05, bombs=0, enemies=2), 1)
    assert garden.enemies == (2, 6) and garden.banked == [((5, 0x05), 4)]
    assert [name for _, name, _, _ in monsters(garden)] == ['BlueDarknut'] * 2
    assert len(monsters(session_of(Spec(3, 0x59, enemies=9), 1))) == 5            # more than the room has: all
    assert session_of(Spec(3, 0x59, enemies=9), 1).enemies == (5, 5)


def test_the_room_opens_when_the_n_left_die():
    """3x59 is a kill-all room (its S shutter): with ``--enemies 2`` it opens once those two die; with all five
    there, two dead leave it shut.  The banked kills and the scripted kills are EDITS: the take verifies."""
    session = session_of(Spec(3, 0x59, enemies=2), 1)
    left = monsters(session)
    assert session.state.is_door_barred('S')
    kill(session, [slot for slot, _, _, _ in left])
    assert session.state.enemies == [] and not session.state.is_door_barred('S')
    edits = [tuple(int(v) for v in e) for e in _native.Replay(session.replay()).edits()]
    assert (0, 0x560 + 0x59, edits[[e[1] for e in edits].index(0x560 + 0x59)][2], 3) in edits    # 3 kills banked
    assert _native.replay_verify(session.replay())['frames'] == session.frames

    full = session_of(Spec(3, 0x59), 1)
    kill(full, [slot for slot, _, _, _ in monsters(full)][:2])
    assert len(full.state.enemies) == 3 and full.state.is_door_barred('S')


def test_enemies_is_for_dungeon_rooms_with_monsters():
    with pytest.raises(ValueError, match='dungeon rooms only'):
        session_of(Spec(0, 0x0f, enemies=1), 1)
    with pytest.raises(ValueError, match='no monsters in 3x7c'):
        session_of(Spec(3, 0x7c, enemies=1), 1)
