"""``triforce.demos``: demonstration rows from a scripted rule, and ``train --demos``'s cloning term."""

import numpy as np
import pytest

from triforce import _native, demos, train
from triforce.actions import MASK_SIZE, NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS
from triforce.metrics import read_jsonl
from triforce.model import MASK_KEY

from .conftest import start_task

KINDS, DIRECTIONS, DURATIONS = list(_native.ACTION_KINDS), list(_native.DIRECTIONS), list(_native.DURATIONS)


def _mask(kind: str, directions=(), durations=()) -> np.ndarray:
    """A mask offering one kind with the given directions and durations."""
    mask = np.zeros(MASK_SIZE, dtype=bool)
    k = KINDS.index(kind)
    mask[k] = True
    for d in directions:
        mask[NUM_KINDS + k * NUM_DIRECTIONS + DIRECTIONS.index(d)] = True
    for u in durations:
        mask[NUM_KINDS + NUM_KINDS * NUM_DIRECTIONS + k * NUM_DURATIONS + [int(x) for x in DURATIONS].index(u)] = True
    return mask


def test_action_text_becomes_the_policy_indices():
    assert demos.action_indices('MOVE W 8') == (KINDS.index('MOVE'), DIRECTIONS.index('W'), 1)
    assert demos.action_indices('SWORD N') == (KINDS.index('SWORD'), DIRECTIONS.index('N'), 0)
    assert demos.action_indices('WAIT 16') == (KINDS.index('WAIT'), 0, 2)


def test_legal_reads_the_kind_and_its_rows():
    mask = _mask('MOVE', ['N', 'W'], [4, 8])
    assert demos.legal(mask, demos.action_indices('MOVE W 8'))
    assert not demos.legal(mask, demos.action_indices('MOVE E 8'))
    assert not demos.legal(mask, demos.action_indices('MOVE N 32'))
    assert not demos.legal(mask, demos.action_indices('SWORD N'))


def test_recorder_keeps_only_kept_episodes_and_round_trips(tmp_path):
    recorder = demos.DemoRecorder()
    obs = {'link': np.arange(3, dtype=np.float32), MASK_KEY: _mask('MOVE', ['N'], [4])}
    recorder.add(obs, 'MOVE N 4')
    recorder.drop()
    for value in (1.0, 2.0):     # the second row written twice
        recorder.add({**obs, 'link': np.full(3, value, dtype=np.float32)}, 'MOVE N 4', repeat=int(value))
    recorder.add({**obs, 'link': np.full(3, 3.0, dtype=np.float32)}, 'MOVE N 4', any_duration=True)
    recorder.keep()
    with pytest.raises(ValueError, match='masked'):
        recorder.add(obs, 'MOVE S 4')
    assert recorder.save(tmp_path / 'd.npz') == 4
    loaded, masks, actions = demos.load_demos([tmp_path / 'd.npz', tmp_path / 'd.npz'])
    assert list(loaded) == ['link'] and loaded['link'][:, 0].tolist() == [1.0, 2.0, 2.0, 3.0] * 2
    move = KINDS.index('MOVE')
    assert masks.shape == (8, MASK_SIZE)
    assert actions.tolist() == [[move, 0, 0], [move, 0, 0], [move, 0, 0], [move, 0, demos.ANY_DURATION]] * 2


def test_train_clones_the_demonstrations(states_dir, tmp_path):
    """--demos adds the cloning term every update (and --ent-coeff sets the entropy bonus)."""
    env = _native.Env('ow-start', states_dir=str(states_dir), seed=0)
    obs, _ = env.reset(seed=0)
    recorder = demos.DemoRecorder()
    recorder.add(obs, 'MOVE N 8')
    recorder.keep()
    recorder.save(tmp_path / 'demos.npz')
    common = ['--specialist', start_task(), '--states-dir', str(states_dir), '--models-dir', str(tmp_path),
              '--parallel', '2', '--steps-per-update', '16', '--iterations', '2', '--no-curriculum', '--seed', '3',
              '--device', 'cpu', '--demos', str(tmp_path / 'demos.npz'), '--demo-coeff', '0.5',
              '--ent-coeff', '0.03']
    with pytest.raises(SystemExit, match='--best-of-n'):
        train.main(common + ['--run-id', 'both', '--best-of-n', '2'])
    run = train.main(common + ['--run-id', 'demos'])
    records = [r for r in read_jsonl(str(run / 'metrics.jsonl')) if 'losses/demo_bc_loss' in r]
    assert len(records) == 2 and all(r['charts/demo_bc_coeff'] == 0.5 for r in records)
    assert all(r['charts/ent_coeff'] == pytest.approx(0.03) for r in records)
