"""Policy network: encoders, factored heads, masking, log-prob/entropy bookkeeping (synthetic observations)."""

import math

import numpy as np
import pytest
import torch

from triforce.actions import Action, ActionMask, CARDINALS
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.zelda_enums import ActionKind

from . import synthetic


def _batch(observations):
    return {k: torch.as_tensor(np.stack([o[k] for o in observations])) for k in observations[0]}


def _model(**kwargs):
    torch.manual_seed(0)
    return ZeldaPolicy(synthetic.observation_space(), ModelConfig(hidden_dim=64, **kwargs))


def _uniform_heads(model):
    """Zero the policy heads so every legal choice is equally likely."""
    with torch.no_grad():
        for head in (model.kind_head, model.direction_head, model.duration_head):
            head.weight.zero_()
            head.bias.zero_()


def test_forward_backward_reaches_every_parameter():
    rng = np.random.default_rng(1)
    model = _model(entity_layers=2)
    obs = _batch([synthetic.random_observation(rng, synthetic.random_mask(rng)) for _ in range(64)])
    out = model.evaluate(obs)
    assert out.actions.shape == (64, 3) and out.log_prob.shape == (64,) and out.value.shape == (64,)
    (out.log_prob.sum() + out.value.sum() + out.entropy.sum()).backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert not missing, f"parameters without gradient: {missing}"
    assert model.parameter_count() == sum(p.numel() for p in model.parameters())


def test_masked_actions_are_never_sampled():
    rng = np.random.default_rng(2)
    model = _model()
    masks = [synthetic.random_mask(rng) for _ in range(32)]
    observations = [synthetic.random_observation(rng, mask) for mask in masks for _ in range(32)]
    with torch.no_grad():
        actions = model.get_action(_batch(observations)).numpy()
    for obs, action in zip(observations, actions):
        mask = ActionMask.from_flat(obs['action_mask'])
        chosen = Action.from_indices(*action)
        assert mask.is_allowed(chosen), f"{chosen} sampled under {mask}"
        assert (action == chosen.to_indices()).all(), "non-applicable heads must report index 0"


def test_no_direction_kind_ignores_direction_head():
    rng = np.random.default_rng(3)
    model = _model()
    obs = synthetic.random_observation(rng)
    wait = torch.as_tensor(Action.parse('WAIT 8').to_indices()).unsqueeze(0)
    move = torch.as_tensor(Action.parse('MOVE N 8').to_indices()).unsqueeze(0)
    sword = torch.as_tensor(Action.parse('SWORD E').to_indices()).unsqueeze(0)
    with torch.no_grad():
        before = {name: model.evaluate(obs, actions=a) for name, a in (('wait', wait), ('move', move))}
        model.direction_head.weight.normal_(0, 5.0)
        model.direction_head.bias.normal_(0, 5.0)
        after_dir = {name: model.evaluate(obs, actions=a) for name, a in (('wait', wait), ('move', move),
                                                                          ('sword', sword))}
        model.duration_head.weight.normal_(0, 5.0)
        model.duration_head.bias.normal_(0, 5.0)
        after_dur = {name: model.evaluate(obs, actions=a) for name, a in (('move', move), ('sword', sword))}

    assert torch.allclose(before['wait'].log_prob, after_dir['wait'].log_prob)
    assert torch.allclose(before['wait'].entropy, after_dir['wait'].entropy)
    assert not torch.allclose(before['move'].log_prob, after_dir['move'].log_prob)
    # SWORD has no duration: duration-head changes can't touch it, but they do change MOVE.
    assert torch.allclose(after_dir['sword'].log_prob, after_dur['sword'].log_prob)
    assert torch.allclose(after_dir['sword'].entropy, after_dur['sword'].entropy)
    assert not torch.allclose(after_dir['move'].log_prob, after_dur['move'].log_prob)


def test_entropy_and_log_prob_count_only_applicable_heads():
    rng = np.random.default_rng(4)
    model = _model()
    _uniform_heads(model)
    mask = ActionMask()
    for direction in CARDINALS:
        mask.allow(ActionKind.MOVE, direction)
    mask.allow(ActionKind.WAIT)
    mask.allow(ActionKind.SWORD, CARDINALS[0])
    mask.allow(ActionKind.SWORD, CARDINALS[1])
    obs = synthetic.random_observation(rng, mask)

    expected = {'MOVE N 8': math.log(3) + math.log(4) + math.log(4),
                'WAIT 32': math.log(3) + math.log(4),
                'SWORD S': math.log(3) + math.log(2)}
    with torch.no_grad():
        for text, value in expected.items():
            out = model.evaluate(obs, actions=torch.as_tensor(Action.parse(text).to_indices()).unsqueeze(0))
            assert out.entropy.item() == pytest.approx(value, abs=1e-5), text
            assert out.log_prob.item() == pytest.approx(-value, abs=1e-5), text


def test_forced_kind_is_excluded_from_log_prob_and_entropy():
    rng = np.random.default_rng(5)
    model = _model()
    _uniform_heads(model)
    mask = ActionMask()
    for direction in CARDINALS:
        mask.allow(ActionKind.MOVE, direction)
    obs = synthetic.random_observation(rng, mask)
    with torch.no_grad():
        model.kind_head.bias[ActionKind.WAIT] = 50.0     # a masked kind's logit must not matter
        out = model.evaluate(obs, actions=torch.as_tensor([[int(ActionKind.MOVE), 2, 1]]))
    assert out.entropy.item() == pytest.approx(2 * math.log(4), abs=1e-5)
    assert out.head_entropy['kind'].item() == 0.0
    assert not out.head_applies['kind'].item()


def test_empty_entity_slots_are_ignored():
    rng = np.random.default_rng(6)
    model = _model()
    obs = synthetic.random_observation(rng, present=3)
    noisy = {k: v.copy() for k, v in obs.items()}
    noisy['entity_features'][3:, 1:] = rng.uniform(-1, 1, noisy['entity_features'][3:, 1:].shape)
    noisy['entity_types'][3:] = 5
    empty = synthetic.random_observation(rng, present=0)
    with torch.no_grad():
        policy, value = model(obs)
        policy_noisy, value_noisy = model(noisy)
        policy_empty, value_empty = model(empty)
    assert torch.allclose(policy, policy_noisy, atol=1e-6) and torch.allclose(value, value_noisy, atol=1e-6)
    assert torch.isfinite(policy_empty).all() and torch.isfinite(value_empty).all()


def test_deterministic_picks_best_legal_action():
    rng = np.random.default_rng(7)
    model = _model()
    _uniform_heads(model)
    with torch.no_grad():
        model.kind_head.bias[ActionKind.BOMB] = 10.0       # illegal under standard_mask
        model.kind_head.bias[ActionKind.SWORD] = 5.0
        model.direction_head.bias[3] = 5.0                 # E
    action = model.get_action(synthetic.random_observation(rng), deterministic=True)[0]
    assert Action.from_indices(*action.tolist()) == Action.parse('SWORD E')


def test_empty_kind_mask_raises():
    rng = np.random.default_rng(8)
    with pytest.raises(ValueError, match='Empty kind mask'):
        _model().evaluate(synthetic.random_observation(rng, ActionMask()))


def test_config_round_trip_rebuilds_identical_network():
    rng = np.random.default_rng(9)
    model = _model()
    clone = ZeldaPolicy(config=ModelConfig.from_dict(model.config.to_dict()))
    clone.load_state_dict(model.state_dict())
    obs = synthetic.random_observation(rng)
    actions = torch.as_tensor([[0, 1, 2]])
    with torch.no_grad():
        a, b = model.evaluate(obs, actions=actions), clone.evaluate(obs, actions=actions)
    assert torch.equal(a.log_prob, b.log_prob) and torch.equal(a.value, b.value)
    with pytest.raises(ValueError):
        ModelConfig.from_dict({'hiden_dim': 3})
