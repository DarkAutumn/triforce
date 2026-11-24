"""``nets.hitnet`` (round36.md "Observation v7 and the hit network"): Link as a token over the entity tokens of
schema 7, a hit head on its own tower; shapes, masks, the stopped gradient, a lab run on a toy family, and play from
the observation alone."""

import json

import pytest
import torch

from triforce import _native, nets
from triforce.evaluate import evaluate_specialist
from triforce.lab import train
from triforce.lab.auxiliary import matched_precision

from .conftest import start_task
from .networks import native_spec

SMALL = {'d_model': 32, 'layers': 1, 'link_out': 32, 'hidden_dim': 32, 'depth': 1, 'hit_dim': 32, 'hit_depth': 1}


def _obs(batch=4, seed=0):
    """Random observations of the native layout (every slot present)."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for key, spec in native_spec().items():
        if key == 'action_mask':
            continue
        shape = (batch, *spec['shape'])
        out[key] = (torch.randint(0, spec['vocab'], shape, generator=gen) if spec['vocab']
                    else torch.rand(shape, generator=gen))
    out['entity_features'][..., 0] = 1.0
    return out


def _mask(batch=4):
    return torch.ones(batch, 11 * 13, dtype=torch.bool)


@pytest.mark.parametrize('hit, outputs', [('joint', 352), ('kind_direction', 88)])
def test_the_hit_output_is_one_logit_a_joint_action(hit, outputs):
    network = nets.create('hitnet', native_spec(), hit=hit, feed=True, **SMALL)
    assert network.hit_head.out_features == outputs
    assert network.policy_tower[0].in_features == network.feature_dim + outputs
    out = network.evaluate(_obs(), _mask())
    assert out.aux['hit'].shape == (4, 352) and out.log_prob.shape == (4,) and out.value.shape == (4,)
    if hit == 'kind_direction':      # the four durations of a kind and direction share their logit
        per = out.aux['hit'].view(4, 88, 4)
        assert torch.equal(per, per[..., :1].expand_as(per))


def test_without_a_hit_head_there_is_no_auxiliary_output_and_groups_switch_off():
    network = nets.create('hitnet', native_spec(), hit='none', tokens=['type', 'features'], room=False, **SMALL)
    assert not hasattr(network, 'hit_head') and not hasattr(network, 'room_embedding')
    assert network.token_proj.in_features == native_spec()['entity_features']['shape'][1] + 16
    assert network.evaluate(_obs(), _mask()).aux == {}
    with pytest.raises(ValueError, match='feed needs a hit head'):
        nets.create('hitnet', native_spec(), hit='none', feed=True)
    with pytest.raises(ValueError, match='tokens'):
        nets.create('hitnet', native_spec(), tokens=['walls'])


def test_the_chase_group_feeds_the_entity_tokens_and_links_token():
    spec = native_spec()
    plain = nets.create('hitnet', spec, tokens=['type', 'features'], **SMALL)
    chase = nets.create('hitnet', spec, tokens=['type', 'features', 'chase'], **SMALL)
    assert chase.token_proj.in_features == plain.token_proj.in_features + spec['entity_chase']['shape'][1]
    assert chase.link_proj.in_features == plain.link_proj.in_features + spec['chase']['shape'][0]
    assert 'chase' not in chase.vector_keys and 'entity_chase' not in chase.vector_keys
    assert chase.feature_dim == plain.feature_dim                # read through Link's token, not the trunk
    obs = _obs()
    other = {k: v.clone() for k, v in obs.items()}
    other['chase'] = torch.rand_like(other['chase'])
    other['entity_chase'] = torch.rand_like(other['entity_chase'])
    chase.eval(), plain.eval()
    assert not torch.allclose(chase.evaluate(obs, _mask()).value, chase.evaluate(other, _mask()).value)
    assert torch.equal(plain.evaluate(obs, _mask()).value, plain.evaluate(other, _mask()).value)


def test_empty_slots_are_masked_and_link_alone_is_fine():
    network = nets.create('hitnet', native_spec(), **SMALL).eval()
    obs = _obs()
    obs['entity_features'][:, 6:, 0] = 0.0
    other = {k: v.clone() for k, v in obs.items()}
    for key in ('entity_features', 'entity_rel', 'entity_contact'):
        other[key][:, 6:, 1:] = torch.rand_like(other[key][:, 6:, 1:]) * 5
    other['entity_types'][:, 6:] = 3
    a, b = network.evaluate(obs, _mask()), network.evaluate(other, _mask())
    assert torch.allclose(a.aux['hit'], b.aux['hit'], atol=1e-5) and torch.allclose(a.value, b.value, atol=1e-5)
    obs['entity_features'][..., 0] = 0.0                    # no entity at all: Link's token attends to itself
    assert torch.isfinite(network.evaluate(obs, _mask()).aux['hit']).all()


def test_the_policy_cannot_bend_the_hit_head():
    network = nets.create('hitnet', native_spec(), feed=True, **SMALL)
    out = network.evaluate(_obs(), _mask())
    (out.log_prob.sum() + out.value.sum()).backward()
    assert network.hit_head.weight.grad is None and all(p.grad is None for p in network.hit_tower.parameters())
    assert network.token_proj.weight.grad is not None       # the body is shared with the policy


def test_matched_precision():
    scores = torch.tensor([0.9, 0.8, 0.1, 0.7, 0.2])
    assert matched_precision(scores, torch.tensor([1, 0, 0, 1, 1])) == pytest.approx(2 / 3)
    assert matched_precision(scores, torch.zeros(5)) != matched_precision(scores, torch.zeros(5))   # NaN


def test_a_hitnet_run_trains_its_head_and_plays_without_trials(states_dir, tmp_path):
    run = train.main(['--name', 'hitnet', '--run-id', 'h1', '--families', f'ow-start={start_task()}', '--parallel',
                      '2',
                      '--steps-per-update', '64', '--device', 'cpu', '--models-dir', str(tmp_path), '--seed', '5',
                      '--iterations', '2', '--arch', 'hitnet', '--arch-config',
                      json.dumps({**SMALL, 'feed': True}), '--aux', 'hit=hit:0.5', '--try-others', '2:0.5'])
    lines = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    metrics = lines[-1].get('metrics', lines[-1])
    assert 'losses/aux_hit' in metrics and metrics['charts/aux_hit_labels'] > 1.0
    assert metrics['charts/aux_hit_lost_judged'] >= 0          # the held-out read (its P = R is NaN without a hit)
    # Served from the observation alone: plays with no trial; foresight (actions tried in the emulator) is refused.
    config = _native.SpecialistConfig.load('ow-start').replace(start_states=['start'], max_steps=12)

    def play(act):
        return evaluate_specialist(config, tmp_path / 'ow-start' / 'h1', episodes_per_state=2, parallel=2, seed=2,
                                   states_dir=states_dir, act=act, clips='on')

    assert play('argmax')['episodes'] == 2
    with pytest.raises(ValueError, match='no foresight'):
        play('argmax+guard:2')
