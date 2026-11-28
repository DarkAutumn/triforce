"""triforce.emu, the whole game and the cheats, with no window: power on to the hand-over, the console's RESET
and its hand-over (the same twice for a name and a delay), the name and the second quest, --invincible (hits put
back, a killing hit undone), infinite counts, the ending's detection, the state key's file, the replays verifying."""

from datetime import datetime

import numpy as np
import pytest

from triforce import _native, emu
from triforce.emu import B, DOWN, LEFT, RIGHT, UP, Cheats, GameSpec, Spec

from .conftest import REPO_STATES_DIR, ROOT, require_rom

GOLDEN = ROOT / 'corpus' / 'replays' / 'golden-d3first-51-networks.tfreplay'


def game_of(spec: GameSpec) -> emu.GameSession:
    require_rom()
    return emu.GameSession(spec)


def room_of(spec: Spec, seed: int) -> emu.Session:
    require_rom()
    return emu.Session(spec, seed=seed, states_dir=REPO_STATES_DIR)


def events(data: bytes) -> list:
    return [(e['frame'], e['type']) for e in _native.Replay(data).events()]


# ---------------------------------------------------------------------------------------------- pure parts

def test_counts_and_the_cheats_writes():
    assert (emu.parse_count('8'), emu.parse_count('infinite'), emu.parse_count('INF')) == (8, emu.INFINITE,
                                                                                          emu.INFINITE)
    for bad in ('-1', '256', 'lots'):
        with pytest.raises(ValueError):
            emu.parse_count(bad)
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[emu.GAME_MODE], ram[emu.HEARTS], ram[emu.PARTIAL_HEART] = emu.PLAY_MODE, 0x22, 0xFF   # 3 of 3
    ram[0x658], ram[0x66E], ram[0x66D] = 7, 8, 254
    cheats = Cheats(invincible=True, infinite=frozenset({'bombs', 'keys', 'rupees'}))
    assert cheats.writes(ram) == [(0x658, 8), (0x66D, 255)]          # one bomb and one rupee spent: back up
    assert Cheats().writes(ram) == []
    ram[emu.HEARTS], ram[emu.PARTIAL_HEART] = 0x22, 0x7F              # half a heart lost
    assert Cheats(invincible=True).writes(ram) == [(emu.PARTIAL_HEART, 0xFF)]
    ram[emu.HEARTS], ram[emu.PARTIAL_HEART] = 0x21, 0xFF              # a whole heart lost: the partial stays full
    assert Cheats(invincible=True).writes(ram) == [(emu.HEARTS, 0x22)]
    ram[emu.HEARTS], ram[emu.PARTIAL_HEART] = 0x20, 0x00              # the hit that kills (asm:Link_BeHarmed)
    ram[emu.GAME_MODE], ram[emu.IS_UPDATING], ram[emu.GAME_SUBMODE] = emu.DEATH_MODE, 0, 0
    assert Cheats(invincible=True).writes(ram) == [(emu.GAME_MODE, emu.PLAY_MODE), (emu.IS_UPDATING, 1),
                                                   (emu.HEARTS, 0x22), (emu.PARTIAL_HEART, 0xFF)]


def test_the_game_spec_its_label_and_the_second_quest():
    assert GameSpec().label == 'start LINK' and GameSpec(name='zelda', reset=120).label == 'start ZELDA reset 120'
    assert GameSpec(name='Zelda').second_quest and not GameSpec(name='LINK').second_quest
    with pytest.raises(ValueError, match='32766'):
        GameSpec(reset=32767)


# ---------------------------------------------------------------------------------------------- the whole game

def test_the_whole_game_from_power_on_hands_over_on_the_first_screen(tmp_path):
    game = game_of(GameSpec())
    state = game.state
    assert (state.mode, state.full_location) == (emu.PLAY_MODE, (0, 0x77, False))
    assert bytes(state.ram[0x638:0x640]) == bytes([0x15, 0x12, 0x17, 0x14, 0x24, 0x24, 0x24, 0x24])   # LINK
    assert game.hand_over.frame == game.frames and game.hand_over.rng == bytes(state.ram[emu.RNG])
    assert game.status_line().startswith('ox77   LINK   hearts 3/3   bombs 0   keys 0   rupees 0   frame ')
    for buttons in [UP] * 30 + [LEFT] * 30:
        assert game.step(buttons) is None
    path = game.save(tmp_path, datetime(2026, 11, 25, 10, 15, 2))
    assert path.name == 'start_LINK_261125-101502.tfreplay'
    data = path.read_bytes()
    replay = _native.Replay(data)
    assert replay.source == {'kind': 'power-on', 'name': 'LINK', 'delay': 0}
    assert replay.version == _native.REPLAY_VERSION == 4
    assert events(data) == [(game.hand_over.frame, 'control')]
    assert _native.replay_verify(data)['frames'] == game.frames
    assert replay.meta['human']['hand_over'] == {'frame': game.hand_over.frame, 'rng': game.hand_over.rng.hex()}


def test_a_reset_hands_over_the_same_game_for_the_same_name_and_delay(tmp_path):
    """--reset: RESET on the first gameplay frame, START on the title as soon as it takes it, the delay on the
    file-select screen, slot 1; the same name and delay twice are the same machine at the hand-over, another delay
    other random bytes, another name the same random bytes (the reset seeds them; the name only moves the frame);
    the replay holds the RESET and verifies."""
    first, again, other = (game_of(GameSpec(name='Q7 0X9', reset=delay)) for delay in (120, 120, 121))
    assert first.hand_over == again.hand_over and first.hand_over.frame + 1 == other.hand_over.frame
    assert first.hand_over.rng != other.hand_over.rng
    assert game_of(GameSpec(reset=120)).hand_over.rng == first.hand_over.rng
    assert bytes(first.game.state_file()) == bytes(again.game.state_file())
    assert first.state.full_location == (0, 0x77, False) and first.state.mode == emu.PLAY_MODE
    plain = game_of(GameSpec(name='Q7 0X9'))
    data = first.replay()
    reset_at = [frame for frame, kind in events(data) if kind == 'reset']
    assert reset_at == [plain.hand_over.frame - 1]                   # on the first gameplay frame
    assert _native.replay_verify(data)['frames'] == first.frames
    assert first.file_name(datetime(2026, 11, 25, 10, 15, 2)) == 'start_Q7-0X9_reset120_261125-101502.tfreplay'
    assert first.start_text().startswith('start Q7 0X9 reset 120: handed over at frame ')


def test_zelda_plays_the_second_quest():
    game = game_of(GameSpec(name='zelda'))
    assert game.state.ram[0x062D] == 1                               # QuestNumbers, slot 1: the second quest
    assert game.start_text().endswith('the second quest')


def test_a_name_the_board_cannot_type_is_refused():
    require_rom()
    with pytest.raises(ValueError, match='only A-Z, 0-9 and spaces'):
        emu.GameSession(GameSpec(name='A-B'))


def test_infinite_counts_and_invincible_from_the_hand_over(tmp_path):
    game = game_of(GameSpec(reset=0, invincible=True, bombs=emu.INFINITE, keys=3, rupees=emu.INFINITE))
    state = game.state
    assert (state.bombs, state.keys, state.ram[0x66D]) == (8, 3, 255)
    game.game.poke(0x658, 5)                                         # five bombs: back to 8 the frame after
    game.step(0)
    assert game.state.bombs == 8
    assert game.file_name(datetime(2026, 11, 25, 10, 15, 2)) == \
        'start_LINK_reset0_invincible_bombsinf_keys3_rupeesinf_261125-101502.tfreplay'
    assert _native.replay_verify(game.replay())['frames'] == game.frames


def test_the_state_key_writes_a_state_named_for_the_room_and_the_frame(tmp_path):
    game = game_of(GameSpec())
    when = datetime(2026, 11, 25, 10, 15, 2)
    path = game.save_state(tmp_path, when)
    assert path.name == f"ox77_frame{game.frames}_261125-101502.state"
    assert game.save_state(tmp_path, when).name == f"ox77_frame{game.frames}_261125-101502-2.state"
    loaded = _native.Game().load(path.read_bytes(), randomize_rng=False, controllable=True)
    assert loaded.full_location == (0, 0x77, False) and bytes(loaded.ram) == bytes(game.state.ram)


def test_the_ending_is_the_first_frame_of_game_mode_13():
    """The corpus's golden race from power on ends the game: ``emu.ended`` holds from the frame GameMode turns $13,
    never before; the same frames played live from a state 240 frames earlier (Dungeon 9, Zelda's room) end it."""
    require_rom()
    player = _native.ReplayPlayer(GOLDEN.read_bytes())
    player.seek(player.frames - 400)
    first, start, buttons = None, None, []
    while player.frame < player.frames:
        if first is None and player.frame == player.frames - 300:
            start = (player.frame, bytes(player.save_state()))
        player.step(1, False)
        if start is not None and first is None:
            buttons.append(player.buttons)
        if emu.ended(player.ram()):
            first = first or player.frame
        else:
            assert first is None, 'the ending, once begun, stays'
    assert first is not None and player.ram()[emu.GAME_MODE] == emu.ENDING_MODE
    live = _native.LiveGame()
    live.load(start[1], 0)
    for frame, pressed in enumerate(buttons, 1):
        live.step(pressed)
        if emu.ended(live.state.ram):
            break
    assert emu.ended(live.state.ram) and frame > 200                 # the walk up to her, then the ending
    assert live.state.full_location[0] == 9


# ---------------------------------------------------------------------------------------------- one room

def test_invincible_puts_the_hits_back_in_the_darknuts_room():
    """3x69 from its stairs into the Darknuts with three hearts dies (test_emu); invincible, the same walk lives,
    every hit (a Red Darknut's: one heart) put back as an edit, and the take verifies."""
    session = room_of(Spec(3, 0x69, entry='cave', invincible=True), 1)
    for buttons in [DOWN] * 4 + [RIGHT] * 400:
        assert session.step(buttons) is None
    assert session.state.health == session.state.max_health
    edits = [tuple(int(v) for v in e) for e in _native.Replay(session.replay()).edits() if e[1] == emu.HEARTS]
    assert len(edits) >= 3 and all(value == (prev & 0xF0) | (prev >> 4) > prev for _, _, prev, value in edits)
    assert _native.replay_verify(session.replay())['frames'] == session.frames
    assert '_invincible_seed1_' in session.file_name()


def test_invincible_undoes_a_killing_hit():
    """One heart container (a Blue Darknut's touch is two hearts, more than Link's filled hearts): each killing hit
    starts the death (GameMode $11) and is undone before its first frame; Link never dies."""
    session = room_of(Spec(5, 0x05, invincible=True), 1)
    session.game.poke(emu.HEARTS, 0x00)
    killed = 0
    for buttons in [UP] * 300 + [LEFT] * 300 + [RIGHT] * 300 + [DOWN] * 300:
        session.game.step(buttons)
        killed += session.game.state.ram[emu.GAME_MODE] == emu.DEATH_MODE
        session.hold()
        assert session.game.state.ram[emu.GAME_MODE] != emu.DEATH_MODE
    assert killed >= 3 and session.state.mode == emu.PLAY_MODE and not session.state.game_over
    assert _native.replay_verify(session.replay())['frames'] == session.frames


def test_infinite_bombs_come_back_after_one_is_set():
    session = room_of(Spec(5, 0x05, bombs=emu.INFINITE, item='bombs'), 1)
    assert session.state.bombs == 8
    counts = []
    for buttons in [LEFT] * 10 + [B] + [0] * 3:                      # off the doorway, then a bomb
        session.step(buttons)
        counts.append(session.state.bombs)
    assert set(counts) == {8}
    edits = [tuple(int(v) for v in e) for e in _native.Replay(session.replay()).edits() if e[1] == 0x658]
    assert edits[-1][2:] == (7, 8)                                   # spent one: 7, written back to 8
    assert '_bombsinf_itembombs_' in session.file_name()
