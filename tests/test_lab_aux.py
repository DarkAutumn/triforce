"""The lab's auxiliary targets (docs/lab.md "Auxiliary targets"): what each decision's action did, the look-ahead's
labels for candidates not taken (training only), the losses on a network's named outputs, and the hit-head example
network (``nets.hithead``) that trains on them and plays without any trial."""

import json

import numpy as np
import pytest
import torch

from triforce import _native, nets
from triforce.evaluate import evaluate_specialist
from triforce.lab import train
from triforce.lab.auxiliary import AuxLoss, aux_loss, joint_index
from triforce.rollout_buffer import RolloutBuffer

from .conftest import REPO_STATES_DIR, require_state
from .networks import native_spec, random_network


def _pool(tmp_path, seed=1):
    """A darknut room (monsters that hit), two envs."""
    require_state('3_5bs_0')
    config = _native.SpecialistConfig.load('darknut').replace(state_routes=['d2first'], start_states=['3_5bs_0'],
                                                              max_steps=40)
    task = _native.TrainTask(config, str(tmp_path / f'task{seed}'), states_dir=str(REPO_STATES_DIR))
    return _native.EnvPool(task, num_envs=2, threads=2, seed=seed)


def _rollout(tmp_path, policy, steps=48, tried=None):
    with _pool(tmp_path) as pool:
        buffer = RolloutBuffer(steps, pool.num_envs, pool.obs_spec, 0.99, 0.95,
                               tried=0 if tried is None else tried['candidates'])
        extra = {} if tried is None else {'tried': tried}
        assert pool.rollout([policy], steps, buffer.rollout_buffers(), **extra) == 2
    return buffer


def test_the_rollout_carries_what_each_action_did_and_tries_others_without_changing_play(tmp_path):
    native = random_network(5).to_native()
    plain = _rollout(tmp_path, native)
    tried = _rollout(tmp_path, native, tried={'candidates': 3, 'share': 0.5})
    # Trials go back to exactly where the env stood: the same play, decision for decision.
    for name in ('actions', 'rewards', 'frames', 'health_lost', 'hits', 'dones'):
        assert torch.equal(getattr(plain, name), getattr(tried, name)), name
    assert plain.tried_actions is None and torch.isfinite(plain.health_lost).all() and (plain.hits >= 0).all()
    assert (plain.health_lost > 0).any() or (plain.hits > 0).any()      # a random policy in a darknut room
    lost = tried.tried_lost.reshape(-1, 3)
    chosen = (lost >= 0).any(-1)
    assert 0.2 < chosen.float().mean() < 0.8                                # share 0.5, drawn per decision
    assert set(lost.unique().tolist()) <= {-1, 0, 1}
    taken = joint_index(tried.actions.reshape(-1, 3))
    candidates = joint_index(tried.tried_actions.reshape(-1, 3, 3))
    for row in chosen.nonzero().squeeze(-1).tolist():
        legal = candidates[row][lost[row] >= 0].tolist()
        assert legal and taken[row].item() not in legal and len(set(legal)) == len(legal)
    assert ((tried.tried_frames.reshape(-1) > 0) == chosen).all()
    assert (candidates[~chosen] == joint_index(torch.tensor([-1, -1, -1]))).all()


def test_a_served_network_tries_its_own_ranked_candidates(tmp_path):
    network = nets.create('hithead', native_spec(), hidden_dim=32, depth=1, hit_dim=32)
    buffer = _rollout(tmp_path, network.to_native(), steps=16, tried={'candidates': 2, 'share': 1.0})
    lost = buffer.tried_lost.reshape(-1, 2)
    assert (lost >= 0).all(-1).float().mean() > 0.9         # nearly every decision has two other legal actions


def test_aux_losses_read_the_taken_and_the_tried_actions():
    batch = {'actions': torch.tensor([[1, 2, 3], [0, 0, 0]]), 'health_lost': torch.tensor([0.5, 0.0]),
             'hits': torch.tensor([0, 2]), 'frames': torch.tensor([16.0, 32.0]),
             'tried_actions': torch.tensor([[[2, 0, 0], [-1, -1, -1]], [[-1, -1, -1], [-1, -1, -1]]]),
             'tried_lost': torch.tensor([[1, -1], [-1, -1]])}
    logits = torch.zeros(2, 352)
    rows = torch.arange(2)
    loss, stats = aux_loss([AuxLoss.parse('hit=hit:2.0')], {'hit': logits}, batch, rows)
    assert stats['charts/aux_hit_labels'] == 1.5 and abs(float(loss) - 2.0 * np.log(2.0)) < 1e-5
    logits[0, joint_index(torch.tensor([1, 2, 3]))] = 20.0       # sure of the hit taken, and of the tried one
    logits[0, joint_index(torch.tensor([2, 0, 0]))] = 20.0
    logits[1, 0] = -20.0
    loss, _ = aux_loss([AuxLoss.parse('hit=hit:1')], {'hit': logits}, batch, rows)
    assert loss < 1e-6
    loss, _ = aux_loss([AuxLoss.parse('d=damaged:1'), AuxLoss.parse('f=frames:1')],
                       {'d': torch.tensor([-20.0, 20.0]), 'f': torch.tensor([1.0, 2.0])}, batch, rows)
    assert loss < 1e-6
    with pytest.raises(ValueError):
        aux_loss([AuxLoss.parse('nope=hit:1')], {'hit': logits}, batch, rows)
    with pytest.raises(ValueError):
        AuxLoss.parse('hit=unknown:1')


def test_the_hit_head_learns_only_from_its_own_loss():
    network = nets.create('hithead', native_spec(), hidden_dim=32, depth=1, hit_dim=32)
    obs = {k: torch.zeros(4, *v['shape'], dtype=torch.long if v['vocab'] else torch.float32)
           for k, v in native_spec().items() if k != 'action_mask'}
    out = network.evaluate(obs, torch.ones(4, 11 * 13, dtype=torch.bool))
    assert out.aux['hit'].shape == (4, 352)
    (out.log_prob.sum() + out.value.sum()).backward()           # the policy's and the value's terms only
    assert network.hit_head.weight.grad is None and all(p.grad is None for p in network.hit_tower.parameters())


def test_a_hit_head_run_trains_on_its_labels_and_plays_without_trials(states_dir, tmp_path):
    run = train.main(['--name', 'hit', '--run-id', 'h1', '--families', 'ow-start', '--parallel', '2',
                      '--steps-per-update', '64', '--device', 'cpu', '--models-dir', str(tmp_path), '--seed', '5',
                      '--no-wait', '--iterations', '2', '--arch', 'hithead',
                      '--arch-config', '{"hidden_dim": 32, "depth": 1, "hit_dim": 32}',
                      '--aux', 'hit=hit:0.5', '--try-others', '2:0.5'])
    lines = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    metrics = lines[-1].get('metrics', lines[-1])
    assert 'losses/aux_hit' in metrics and metrics['charts/aux_hit_labels'] > 1.0     # tried candidates add labels
    # At play the network is served from the observation alone: it plays with no trial, and foresight (which tries
    # actions in the emulator) is refused for it.
    config = _native.SpecialistConfig.load('ow-start').replace(start_states=['start'], max_steps=12)

    def play(act):
        return evaluate_specialist(config, tmp_path / 'ow-start' / 'h1', episodes_per_state=2, parallel=2, seed=2,
                                   states_dir=states_dir, act=act, clips='on')

    assert play('argmax')['episodes'] == 2
    with pytest.raises(ValueError, match='no foresight'):
        play('argmax+guard:2')
