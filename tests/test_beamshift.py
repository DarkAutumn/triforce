"""``nets.beamshift`` (round39.md "A beam head"): a policy and a supervised beam network in one network, the BEAMS
entries of the joint action raised by ``weight * sigmoid(beam logit)``. The shift is the formula, only BEAMS moves,
weight 0 is the policy exactly, a ``hitshift`` nests as the policy, and a combination saves and loads as a run; the
recorded labels (``beam_lined``) become the head's candidates (``r38_hit.beam_rows``)."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

from triforce import registry
from triforce.actions import CARDINALS, DIRECTION_INDEX
from triforce.nets.base import factored_joint
from triforce.nets.beamshift import BEAMS, BeamShift
from triforce.nets.hitshift import HitShift
from triforce.observation import OBJECTIVE_FEATURES

from .networks import random_network
from .test_hitshift import MASK, _batch, _hitnet

ROOT = Path(__file__).resolve().parents[1]


def _expected(p: torch.Tensor, beam: torch.Tensor, weight: float) -> torch.Tensor:
    """q ∝ p · exp(shift): each BEAMS direction raised by weight · σ(its duration-0 logit), for every duration."""
    shift = torch.zeros_like(p).view(-1, 11, 8, 4)
    shift[:, BEAMS] = weight * torch.sigmoid(beam.view(-1, 11, 8, 4)[:, BEAMS, :, :1]).expand(-1, -1, 4)
    q = p + shift.view(p.shape)
    return torch.exp(q - torch.logsumexp(q, dim=-1, keepdim=True)).view(-1, 11, 8, 4)


def test_the_shift_raises_beams_alone():
    policy, beamnet = random_network(3, sharpen=3.0), _hitnet(4)
    network = BeamShift.of(policy, beamnet, 4.0).eval()
    obs = _batch(6, seed=1)
    obs[MASK][:, BEAMS] = True                                      # _batch rules kind 3 out; BEAMS is kind 3
    with torch.no_grad():
        heads, _ = network.heads(obs)
        beam = beamnet.hit_logits(beamnet.trunk(obs))
        p = factored_joint(policy.heads(obs)[0], obs[MASK])
    q = _expected(p, beam, 4.0)
    f = torch.exp(factored_joint(heads, obs[MASK])).view(6, 11, 8, 4)
    assert torch.allclose(f.sum((2, 3)), q.sum((2, 3)), atol=1e-5)
    assert torch.allclose(f.sum(3), q.sum(3), atol=1e-5)
    assert torch.equal(heads.aux['beam'], beam)
    # Among the other kinds the policy's odds are kept: only BEAMS's mass moved.
    others = [k for k in range(11) if k != BEAMS]
    pk, fk = torch.exp(p).view(6, 11, 32).sum(-1)[:, others], f.sum((2, 3))[:, others]
    assert torch.allclose(fk / fk.sum(-1, keepdim=True), pk / pk.sum(-1, keepdim=True), atol=1e-5)
    assert float(f.sum((2, 3))[:, BEAMS].min()) >= float(torch.exp(p).view(6, 11, 32).sum(-1)[:, BEAMS].min()) - 1e-6


def test_weight_zero_is_the_policy_and_the_head_never_trains():
    policy, beamnet = random_network(5, sharpen=3.0), _hitnet(6)
    network = BeamShift.of(policy, beamnet, 0.0).eval()
    obs = _batch(4, seed=2)
    with torch.no_grad():
        alone, combined = policy.heads(obs)[0], network.heads(obs)[0]
    for name in ('kind', 'direction', 'duration', 'value'):
        assert torch.equal(getattr(alone, name), getattr(combined, name)), name
    assert not any(p.requires_grad for p in network.beam.parameters())


def test_the_fight_gate_shifts_fights_alone_and_saves(tmp_path):
    """``objectives=['FIGHT']``: rows whose objective is a fight get the shift, rows that cross (MOVE) play the policy."""
    policy, beamnet = random_network(11, sharpen=3.0), _hitnet(12)
    network = BeamShift.of(policy, beamnet, 8.0, ['FIGHT']).eval()
    obs = _batch(6, seed=4)
    obs[MASK][:, BEAMS] = True
    kind = torch.zeros(6, len(OBJECTIVE_FEATURES))
    kind[:3, OBJECTIVE_FEATURES.index('objective_FIGHT')] = 1.0
    kind[3:, OBJECTIVE_FEATURES.index('objective_MOVE')] = 1.0
    obs['objective'] = kind
    with torch.no_grad():
        f = torch.exp(factored_joint(network.heads(obs)[0], obs[MASK])).view(6, 11, 8, 4)
        p = factored_joint(policy.heads(obs)[0], obs[MASK])
        beam = beamnet.hit_logits(beamnet.trunk(obs))
    q = _expected(p, beam, 8.0)
    assert torch.allclose(f[:3].sum((2, 3)), q[:3].sum((2, 3)), atol=1e-5)
    assert torch.allclose(f[3:].sum((2, 3)), torch.exp(p[3:]).view(3, 11, 32).sum(-1), atol=1e-5)
    path = registry.save_checkpoint('wizzrobe', 'beam-fight-w8', network, root=tmp_path)
    loaded = registry.load_model(path, check_schema=False)[0].eval()
    assert loaded.config['objectives'] == ['FIGHT'] and loaded.gate == network.gate


def test_a_hitshift_nests_as_the_policy_and_saves_as_a_run(tmp_path):
    inner = HitShift.of(random_network(7, sharpen=3.0), _hitnet(8), 8.0).eval()
    network = BeamShift.of(inner, _hitnet(9), 4.0).eval()
    obs = _batch(3, seed=3)
    obs[MASK][:, BEAMS] = True
    with torch.no_grad():
        p = factored_joint(inner.heads(obs)[0], obs[MASK])
        beam = network.beam_logits(obs)
        f = torch.exp(factored_joint(network.heads(obs)[0], obs[MASK])).view(3, 11, 8, 4)
    assert torch.allclose(f.sum((2, 3)), _expected(p, beam, 4.0).sum((2, 3)), atol=1e-5)
    path = registry.save_checkpoint('darknut', 'beam-w4', network, root=tmp_path)
    loaded = registry.load_model(path, check_schema=False)[0].eval()
    assert isinstance(loaded, BeamShift) and isinstance(loaded.policy, HitShift)
    with torch.no_grad():
        assert torch.allclose(loaded.heads(obs)[0].kind, network.heads(obs)[0].kind)


def test_the_recorded_labels_become_one_candidate_a_legal_direction():
    spec = importlib.util.spec_from_file_location('r38_hit', ROOT / 'scripts' / 'experiments' / 'r38_hit.py')
    r38_hit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r38_hit)
    mask = np.zeros((3, 11 + 88 + 44), bool)
    mask[0, BEAMS] = True                                            # decision 0: BEAMS legal N and E
    for d in ('N', 'E'):
        mask[0, 11 + BEAMS * 8 + DIRECTION_INDEX[next(c for c in CARDINALS if c.name == d)]] = True
    mask[1, 11 + BEAMS * 8:11 + BEAMS * 8 + 8] = True               # decision 1: directions but not the kind
    lined = np.array([8, 15, 0])                                     # N lined up for decision 0
    rows = r38_hit.beam_rows({MASK: mask}, lined, np.zeros(3, bool), 'test')
    assert sorted(rows.c_row.tolist()) == [0, 0]
    labels = {int(j) % 32 // 4: bool(y) for j, y in zip(rows.c_joint, rows.c_lost)}
    north = DIRECTION_INDEX[next(c for c in CARDINALS if c.name == 'N')]
    east = DIRECTION_INDEX[next(c for c in CARDINALS if c.name == 'E')]
    assert labels == {north: True, east: False}
    assert all(int(j) // 32 == BEAMS and int(j) % 4 == 0 for j in rows.c_joint)
