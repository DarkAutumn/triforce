"""``triforce.demo``: demo trace parsing and replaying a trace through a native ``Env`` into a demo batch."""

import numpy as np
import pytest
import torch

from triforce import _native
from triforce.actions import Action
from triforce.demo import collect_demo_batch, compute_demo_accuracy, parse_demo_actions

# ow-start from the game start: west to the sword cave's column, north into the cave, take the sword, back out,
# east to the north exit's column and north into 0x67 (its objective-correct exit once Link has the sword).
CAVE_TRACE = ("# into the sword cave\n" + 'MOVE W 4\n' * 10 + 'MOVE N 8  # up the cave column\n' * 7
              + "\nMOVE E 4   # line up with the sword\n" + 'MOVE N 8\n' * 6
              + "\n# out and north\n" + 'MOVE S 8\n' * 8 + 'MOVE E 4\n' * 11 + 'MOVE N 8\n' * 4)
CAVE_ACTIONS = ([Action.parse('MOVE W 4')] * 10 + [Action.parse('MOVE N 8')] * 7 + [Action.parse('MOVE E 4')]
                + [Action.parse('MOVE N 8')] * 6 + [Action.parse('MOVE S 8')] * 8 + [Action.parse('MOVE E 4')] * 11
                + [Action.parse('MOVE N 8')] * 4)


def test_trace_parsing_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / 'cave.txt'
    path.write_text(CAVE_TRACE)
    assert parse_demo_actions(str(path)) == CAVE_ACTIONS
    (tmp_path / 'empty.txt').write_text('# nothing\n\n')
    with pytest.raises(ValueError, match='no actions'):
        parse_demo_actions(str(tmp_path / 'empty.txt'))
    (tmp_path / 'bad.txt').write_text('MOVE N 16 extra\n')
    with pytest.raises(ValueError, match='Unexpected tokens'):
        parse_demo_actions(str(tmp_path / 'bad.txt'))


def _env(states_dir):
    config = _native.SpecialistConfig.load('ow-start').replace(start_states=['start'], max_steps=100)
    return _native.Env(config, states_dir=str(states_dir), seed=0)


def test_demo_batch_from_a_successful_trace(states_dir):
    steps = len(CAVE_ACTIONS)
    obs, masks, targets = collect_demo_batch(_env(states_dir), CAVE_ACTIONS)
    assert targets.dtype == torch.int64
    assert torch.equal(targets, torch.as_tensor(np.stack([a.to_indices() for a in CAVE_ACTIONS])))
    assert masks.shape == (steps, _native.MASK_SIZE) and masks.dtype == torch.bool
    assert set(obs) == set(_native.OBS_SPEC) - {'action_mask'}
    assert all(value.shape[0] == steps for value in obs.values())
    assert compute_demo_accuracy(targets, targets) == 1.0
    wrong = targets.clone()
    wrong[0, 2] += 1
    assert compute_demo_accuracy(wrong, targets) == pytest.approx((steps - 1) / steps)


def test_demo_replay_refuses_masked_actions_and_unfinished_traces(states_dir):
    with pytest.raises(RuntimeError, match='Demo step 2: SWORD N is not allowed'):
        collect_demo_batch(_env(states_dir), [Action.parse('MOVE W 4'), Action.parse('SWORD N')])
    short = CAVE_ACTIONS[:20]
    with pytest.raises(RuntimeError, match='did not end in success'):
        collect_demo_batch(_env(states_dir), short)
    _, _, targets = collect_demo_batch(_env(states_dir), short, require_success=False)
    assert targets.shape == (20, 3)
    # Steps after the episode ends are not part of the batch.
    _, _, targets = collect_demo_batch(_env(states_dir), CAVE_ACTIONS + [Action.parse('WAIT 8')])
    assert len(targets) == len(CAVE_ACTIONS)
