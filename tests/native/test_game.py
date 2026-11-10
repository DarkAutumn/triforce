"""`Game`: loading, acting, snapshots, pokes, rendering and text dumps on zcore."""

import gzip
import pickle

import numpy as np
import pytest

from triforce import _native as native


def test_load_from_name_path_bytes_and_snapshot(game, states_dir):
    state = game.load('start', randomize_rng=False)
    assert state.full_location == (0, 0x77, False)
    assert (state.health, state.max_health) == (3.0, 3)
    by_path = native.Game(states_dir=str(states_dir)).load(str(states_dir / 'start.state'), randomize_rng=False)
    by_bytes = native.Game().load((states_dir / 'start.state').read_bytes(), randomize_rng=False)
    assert bytes(by_path.ram) == bytes(by_bytes.ram) == bytes(state.ram)
    assert game.state.mode == 5
    assert game.steps == 0 and game.total_frames == 0


def test_act_snapshot_restore_round_trip(game):
    game.load('start')
    snap = game.snapshot()
    actions = ['MOVE N 16', 'MOVE W 8', (0, 2, 1), 'WAIT 8', 'MOVE S 32', 'MOVE E 16']
    first = [game.act(a) for a in actions]
    ram_after = bytes(game.state.ram)
    assert game.steps == len(actions)
    assert game.total_frames == sum(c.frames for c in first)
    game.restore(snap)
    assert game.steps == 0
    second = [game.act(a) for a in actions]
    assert [c.frames for c in first] == [c.frames for c in second]
    assert [c.interrupt for c in first] == [c.interrupt for c in second]
    assert bytes(game.state.ram) == ram_after

    # A pickled snapshot restores the same machine (another game, same results).
    other = native.Game()
    other.restore(pickle.loads(pickle.dumps(snap)))
    third = [other.act(a) for a in actions]
    assert [c.frames for c in third] == [c.frames for c in first]
    assert bytes(other.state.ram) == ram_after


def test_change_fields(game):
    game.load('start', randomize_rng=False)
    change = game.act('MOVE N 16')
    assert change.action == 'MOVE N 16'
    assert change.action_indices == (0, 0, 2)
    assert change.frames >= 16 and change.control_frames >= 1
    assert change.previous.full_location == (0, 0x77, False)
    assert change.state.link['y'] < change.previous.link['y']
    assert change.action_mask.shape == (native.MASK_SIZE,) and change.action_mask.dtype == bool
    assert (change.health_lost, change.health_gained, change.enemies_hit, change.blocks) == (0.0, 0.0, [], 0)
    assert 'MOVE N 16 ->' in change.describe() == native.Game.describe_change(change)


def test_masked_and_malformed_actions(game):
    game.load('start')
    assert not game.is_allowed('SWORD N')     # no sword yet
    assert game.is_allowed('MOVE N 16')
    with pytest.raises(native.InvalidActionError):
        game.act('JUMP N')
    with pytest.raises(native.InvalidActionError):
        game.act((0, 0))
    assert game.mask.shape == (native.MASK_SIZE,)
    assert game.legal.startswith('MOVE[')


def test_ram_read_write_and_variables(game):
    game.load('start')
    ram = game.ram()
    assert ram.shape == (10240,) and ram.dtype == np.uint8
    assert game.read(0x10) == 0 and game.read(0xEB) == 0x77
    game.write(0x66F, 0x22)
    assert game.read(0x66F) == 0x22
    state = game.set_value('hearts_and_containers', 0x44)
    assert state.max_health == 5
    state = game.apply_ram({'rupees': 42})
    assert state.rupees == 42 and state.var('rupees') == 42
    with pytest.raises(ValueError):
        game.read(0x2000)
    with pytest.raises(ValueError):
        game.set_value('no_such_variable', 1)


def test_ram_values_must_fit_a_byte(game):
    game.load('start')
    with pytest.raises(ValueError, match='rupees = 300 does not fit a byte'):
        game.set_value('rupees', 300)
    with pytest.raises(ValueError, match='does not fit a byte'):
        game.apply_ram({'rupees': 256})
    with pytest.raises(ValueError, match='does not fit a byte'):
        game.load('start', ram_overrides={'rupees': 300})
    assert game.set_value('rupees', 255).rupees == 255
    assert game.set_value('rupees', -1).rupees == 255                     # two's complement of a signed byte


def test_rendering(game):
    game.load('start')
    with pytest.raises(native.RenderDisabledError):
        game.screen()
    game.render = True
    with pytest.raises(native.RenderDisabledError):
        game.screen()                          # nothing rendered since rendering was turned on
    game.act('WAIT 4')
    screen = game.screen()
    assert screen.shape == native.FRAME_SHAPE and screen.dtype == np.uint8
    assert screen.reshape(-1, 3).max(axis=0).max() > 0
    game.render = False
    with pytest.raises(native.RenderDisabledError):
        game.screen()


def test_render_does_not_change_emulation(game):
    game.load('start', randomize_rng=False)
    snap = game.snapshot()
    off = [game.act('MOVE W 16').frames for _ in range(3)]
    ram_off = bytes(game.state.ram)
    game.restore(snap)
    game.render = True
    on = [game.act('MOVE W 16').frames for _ in range(3)]
    assert on == off and bytes(game.state.ram) == ram_off


def test_press_buttons(game):
    game.load('start')
    x = game.state.link['x']
    state = game.press('RIGHT', 8)
    assert state.link['x'] > x
    state = game.press(['LEFT'], 8)
    assert state.link['x'] < x + 8
    with pytest.raises(ValueError):
        game.press('JUMP')


def test_state_files_and_formats(game, tmp_path):
    game.load('start')
    game.act('MOVE N 16')
    path = tmp_path / 'here.state'
    game.save_state_file(str(path))
    assert path.read_bytes()[:8] == b'TFZSTATE'
    assert path.read_bytes() == game.save_state()
    other = native.Game()
    state = other.load(str(path), randomize_rng=False, controllable=True)
    assert bytes(state.ram) == bytes(game.state.ram)

    legacy = tmp_path / 'legacy.state'
    legacy.write_bytes(gzip.compress(b'not a state'))
    with pytest.raises(native.StateFormatError, match='legacy stable-retro state'):
        game.load(str(legacy))
    with pytest.raises(native.StateFormatError, match='not a TFZSTATE'):
        game.load(b'garbage bytes that are long enough to have a header')
    with pytest.raises(OSError):
        game.load('no_such_state_name')


def test_pre_entry_capture(game):
    game.load('start', randomize_rng=False)
    game.capture_pre_entry = True
    assert game.capture_pre_entry
    for _ in range(12):
        change = game.act('MOVE N 16')
        if change.changed_location:
            break
    assert change.changed_location and change.state.full_location == (0, 0x67, False)
    pre = game.last_pre_entry
    assert pre is not None
    replay = native.Game(seed=5)
    state = replay.load(pre)
    assert state.full_location == (0, 0x67, False)


def test_text_dumps(game):
    game.load('start')
    text = game.describe_state()
    assert text.startswith('Location: level 0 room 0x77') and 'Legal: MOVE[' in text
    grid = game.render_grid()
    assert grid.count('\n') == 22 and 'L' in grid
    assert game.state.describe().splitlines()[0] == text.splitlines()[0]


def test_play_movie_from_power_on():
    from tests.native.conftest import MOVIE  # pylint: disable=import-outside-toplevel
    game = native.Game()
    with pytest.raises(RuntimeError):
        _ = game.state
    state = game.play_movie(MOVIE)
    assert (state.mode, state.level, state.location) == (5, 0, 0x77)
