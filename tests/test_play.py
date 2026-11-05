"""``triforce.play`` (headless): a session persists between invocations; actions, raw presses, RAM pokes and dumps,
screenshots, savestates and room-entry captures."""

import pickle

import numpy as np
from PIL import Image

from triforce import _native, cli, play


def _start(states_dir, session):
    assert play.main(['--state', str(states_dir / 'start.state'), '--seed-rng', '--session', str(session),
                      '--quiet']) == 0


def _session_game(session):
    with open(session, 'rb') as file:
        saved = pickle.load(file)
    game = _native.Game()
    game.restore(saved['snapshot'])
    return game, saved['screen']


def test_without_a_session_play_asks_for_a_state(tmp_path, capsys):
    assert play.main(['MOVE N 16', '--session', str(tmp_path / 'none.pkl')]) == 1
    assert 'start one with --state' in capsys.readouterr().err


def test_session_continues_across_invocations(states_dir, tmp_path, capsys):
    session = tmp_path / 'session.pkl'
    _start(states_dir, session)
    start_x = _session_game(session)[0].state.link['x']
    capsys.readouterr()

    assert play.main(['MOVE W 8', 'SWORD N', 'press LEFT+B 4', 'press NONE 2', '--session', str(session),
                      '--no-grid', '--ram', '0x70', '--ram', '0x70:2']) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith('MOVE W 8 ->') and 'WARNING' not in out[0]
    assert out[1].endswith('(WARNING: action was masked out)')         # no sword yet
    assert out[2:4] == ['press LEFT+B for 4 frames', 'press NONE for 2 frames']
    game, screen = _session_game(session)
    x = game.state.link['x']
    assert x < start_x - 8
    pair = ' '.join(f"{v:02x}" for v in game.ram()[0x70:0x72])
    assert out[-2:] == [f"RAM 0x070: {x:02x}", f"RAM 0x070: {pair}"]
    assert screen.shape == _native.FRAME_SHAPE and screen.max() > 0
    assert out[4].startswith('Location: level 0 room 0x77') and not any('####' in line for line in out)

    # The next call starts where this one stopped.
    assert play.main(['MOVE E 8', '--session', str(session), '--quiet']) == 0
    assert _session_game(session)[0].state.link['x'] > x


def test_set_screenshot_and_save_state(states_dir, tmp_path, capsys):
    session = tmp_path / 'session.pkl'
    _start(states_dir, session)
    png, state = tmp_path / 'shot.png', tmp_path / 'here.state'
    assert play.main(['WAIT 4', '--session', str(session), '--quiet', '--set', 'rupees=0x2a',
                      '--screenshot', str(png), '--save-state', str(state)]) == 0
    out = capsys.readouterr().out
    assert f"screenshot: {png}" in out and f"saved state: {state}" in out
    image = Image.open(png)
    assert image.size == (512, 480)
    game, screen = _session_game(session)
    assert np.array_equal(np.asarray(image.convert('RGB'))[::2, ::2], screen)
    assert game.state.rupees == 42
    loaded = _native.Game().load(str(state), randomize_rng=False, controllable=True)
    assert bytes(loaded.ram) == bytes(game.state.ram)


def test_capture_saves_each_new_room_once(states_dir, tmp_path, capsys):
    session, captures = tmp_path / 'session.pkl', tmp_path / 'captures'
    captures.mkdir()
    _start(states_dir, session)
    north = ['MOVE N 16'] * 6
    assert cli.main(['play', *north, '--session', str(session), '--capture', str(captures), '--quiet']) == 0
    assert f"captured {captures / '0_67n.state'}" in capsys.readouterr().out
    assert sorted(p.name for p in captures.iterdir()) == ['0_67n.state']
    state = _native.Game().load(str(captures / '0_67n.state'), randomize_rng=False, controllable=True)
    assert state.full_location == (0, 0x67, False)

    _start(states_dir, session)
    assert play.main([*north, '--session', str(session), '--capture', str(captures), '--quiet']) == 0
    assert 'capture skipped, exists' in capsys.readouterr().out


def test_list_states_filters_by_glob(tmp_path, monkeypatch, capsys):
    for name in ('1_44w_0.state', '1_44w_1.state', '0_77c_0.state', 'notes.txt'):
        (tmp_path / name).write_text('')
    monkeypatch.setattr(play, 'STATES_DIR', tmp_path)
    assert play.main(['--list-states']) == 0
    assert capsys.readouterr().out.split() == ['0_77c_0', '1_44w_0', '1_44w_1']
    assert play.main(['--list-states', '1_*']) == 0
    assert capsys.readouterr().out.split() == ['1_44w_0', '1_44w_1']
