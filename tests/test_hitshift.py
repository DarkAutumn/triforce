"""``nets.hitshift`` (round38.md "The hit head at play"): a policy and a supervised hit network in one network, the
joint action shifted by ``-weight * softplus(hit logit)``. The shift is the formula, weight 0 is the policy exactly,
and a combination saved as a family's run is loaded and played by every tool (chains with replays, the golden race,
``zelda-headless.py``, ``play-zelda.py``'s simulations, evaluations) from the observation alone: no trial in the
emulator at play."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from triforce import _native, golden, nets, registry, replay
from triforce.evaluate import evaluate_specialist
from triforce.nets.base import factored_joint
from triforce.nets.hitshift import HitShift
from triforce.play_grid import ReplayGrid
from triforce.runner import Policies, Route, Runner, chain_meta

from .conftest import SEGMENT_STATE, SEGMENT_STOP_STEP, promote_untrained, require_state
from .networks import native_spec, random_network, scripted
from .test_hitnet import SMALL, _obs

ROOT = Path(__file__).resolve().parents[1]
MASK = 'action_mask'


def _hitnet(seed: int):
    """A small joint-head hit network with perturbed weights (its logits vary by action)."""
    torch.manual_seed(seed)
    network = nets.create('hitnet', native_spec(), hit='joint', **SMALL).eval()
    with torch.no_grad():
        for param in network.parameters():
            param.add_(0.3 * torch.randn_like(param))
    return network


def _batch(batch: int, seed: int) -> dict:
    """Random observations with a mask that rules out two kinds and some directions and durations."""
    obs = _obs(batch, seed)
    mask = torch.ones(batch, 11 + 11 * 8 + 11 * 4, dtype=torch.bool)
    mask[:, [3, 7]] = False
    mask[: batch // 2, 11 + 8 * 1: 11 + 8 * 1 + 3] = False          # kind 1's first three directions
    mask[batch // 2:, 11 + 88 + 4 * 2] = False                       # kind 2's shortest duration
    return {**obs, MASK: mask}


@pytest.mark.parametrize('weight', [1.0, 4.0, 8.0])
def test_the_shift_is_the_formula(weight):
    """q(k, d, u) ∝ p(k, d, u) · exp(−w · softplus(h(k, d, u))): the shifted heads' factored joint has q's
    marginals over (kind), (kind, direction) and (kind, duration), and the hit logits ride along in ``aux``."""
    policy, hitnet = random_network(3, sharpen=3.0), _hitnet(4)
    network = HitShift.of(policy, hitnet, weight).eval()
    obs = _batch(6, seed=1)
    mask = obs[MASK]
    with torch.no_grad():
        heads, _ = network.heads(obs)
        hit = hitnet.hit_logits(hitnet.trunk(obs))
        p = factored_joint(policy.heads(obs)[0], mask)
    q = p - weight * F.softplus(hit)
    q = torch.exp(q - torch.logsumexp(q, dim=-1, keepdim=True)).view(6, 11, 8, 4)
    f = torch.exp(factored_joint(heads, mask)).view(6, 11, 8, 4)
    assert torch.allclose(f.sum((2, 3)), q.sum((2, 3)), atol=1e-5)
    assert torch.allclose(f.sum(3), q.sum(3), atol=1e-5)
    assert torch.allclose(f.sum(2), q.sum(2), atol=1e-5)
    assert torch.equal(heads.aux['hit'], hit)
    assert float(q.sum((2, 3))[:, [3, 7]].max()) == 0.0              # nothing illegal comes back
    # A head that reads every action alike shifts nothing.
    flat = p - weight * F.softplus(torch.full_like(hit, 1.5))
    assert torch.allclose(torch.softmax(flat, -1), torch.exp(p), atol=1e-6)


def test_weight_zero_is_the_policy_exactly():
    policy, hitnet = random_network(5, sharpen=3.0), _hitnet(6)
    network = HitShift.of(policy, hitnet, 0.0).eval()
    obs = _batch(4, seed=2)
    with torch.no_grad():
        alone, combined = policy.heads(obs)[0], network.heads(obs)[0]
        values = network.get_value(obs), policy.heads(obs)[0].value
    for name in ('kind', 'direction', 'duration', 'value'):
        assert torch.equal(getattr(alone, name), getattr(combined, name)), name
    assert torch.equal(*values)


def test_the_policy_never_trains_the_hit_head():
    """The shift reads the hit logits with the gradient stopped; ``freeze_hit`` takes the hit body out of
    training."""
    network = HitShift.of(random_network(7), _hitnet(8), 4.0)
    heads, _ = network.heads(_batch(3, seed=3))
    (heads.kind.logsumexp(-1).sum() + heads.direction.sum()).backward()
    assert all(p.grad is None or not p.grad.any() for p in network.hit.parameters())
    frozen = HitShift.of(random_network(7), _hitnet(8), 4.0, freeze_hit=True)
    assert not any(p.requires_grad for p in frozen.hit.parameters())


@pytest.fixture(scope='module')
def shift_models(tmp_path_factory):
    """The untrained team with ``ow-octorok`` promoted to a combination: the network that walks south
    (``networks.scripted``) and a small hit network, w = 8, saved as ``r38_hit.py combine`` saves it."""
    root = promote_untrained(tmp_path_factory.mktemp('shift-models'))
    network = HitShift.of(scripted('MOVE S 16'), _hitnet(9), 8.0)
    path = registry.save_checkpoint('ow-octorok', 'shift-w8', network, root=root)
    registry.set_act_mode(path, registry.DEFAULT_ACT)
    registry.promote('ow-octorok', 'shift-w8', root=root)
    return root


def _segment():
    return Route.load('d2first', start_from=str(require_state(SEGMENT_STATE)), stop_step=SEGMENT_STOP_STEP)


def test_a_combination_is_a_run_like_any_other(shift_models):
    assert registry.list_runs('ow-octorok', shift_models) == ['shift-w8', 'untrained']
    assert registry.promoted_run('ow-octorok', shift_models) == 'shift-w8'
    assert registry.act_mode(registry.checkpoint_path('ow-octorok', 'shift-w8', shift_models)) == registry.DEFAULT_ACT
    loaded = registry.load_model(registry.checkpoint_path('ow-octorok', 'shift-w8', shift_models))[0]
    assert isinstance(loaded, HitShift) and loaded.config['weight'] == 8.0
    policies = Policies(shift_models)
    assert policies.checkpoints['ow-octorok'].endswith('shift-w8/ckpt.pt') or \
        Path(policies.checkpoints['ow-octorok']).name == 'shift-w8'


def test_a_chain_plays_and_records_it(shift_models, states_dir, tmp_path):
    """Recorded like any other, its hit head's danger with every decision it took and named in META."""
    runner = Runner(models_dir=shift_models, max_steps=20, parallel=2, states_dir=states_dir)
    segment = _segment()
    results = runner.run_many(segment, [0, 1], record=replay.record_options(runner.record_meta(segment)),
                              replays=replay.ReplayDir(tmp_path))
    assert [r.result for r in results] == ['success', 'success']
    for path in sorted(tmp_path.iterdir()):
        recording = _native.Replay(path.read_bytes())
        assert _native.replay_verify(recording.data)['frames'] == recording.frames
        models = recording.meta['models']
        shift = next(m for m in models if m['name'] == 'ow-octorok')
        assert shift['served'] and shift['architecture']['name'] == 'hitshift' and shift['run'] == 'shift-w8'
        assert [o['name'] for o in shift['outputs']] == ['distribution', 'value', 'hit']
        decisions = recording.decisions()
        mine = [row for policy, row in zip(decisions['policy'], decisions['danger']) if policy == shift['index']]
        assert mine and all(row and all(0.0 <= p <= 1.0 for p in row.values()) for row in mine)


def test_the_golden_race_plays_it(shift_models, states_dir):
    policies = Policies(shift_models)
    data, summary = golden.golden_path(
        policies, 'd2first', 2, 10, replay.record_options(chain_meta(policies, 'sample', Route.load('d2first'))),
        start_from=str(require_state(SEGMENT_STATE)), stops=[SEGMENT_STOP_STEP], states_dir=states_dir, attempts=4)
    assert summary['result'] == 'success'
    assert _native.replay_verify(data)['frames'] == _native.Replay(data).frames


def test_zelda_headless_plays_it(shift_models, tmp_path, capsys):
    """Its runs, with every decision recorded (the served network's too)."""
    spec = importlib.util.spec_from_file_location('zelda_headless', ROOT / 'zelda-headless.py')
    headless = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(headless)
    out = tmp_path / 'runs'
    common = ['--route', 'd2first', '--from', str(require_state(SEGMENT_STATE)), '--parallel', '2', '--models-dir',
              str(shift_models)]
    assert headless.main([str(out), '2', '--stop-at', f'step:{SEGMENT_STOP_STEP}', *common]) == 0
    files = list(out.iterdir())
    assert len(files) == 2
    assert all(len(_native.Replay(path.read_bytes()).decisions()['frame']) > 0 for path in files)
    capsys.readouterr()


@pytest.mark.parametrize('source', ['best', 'golden'])
def test_play_zeldas_simulations_start_with_it(shift_models, states_dir, source):
    grid = ReplayGrid(1, 0.0, source, models_dir=shift_models, states_dir=states_dir, route='d2first', parallel=2,
                      start_from=str(require_state(SEGMENT_STATE)), stop_step=SEGMENT_STOP_STEP)
    try:
        grid._ensure(source)  # pylint: disable=protected-access
        assert grid._failed == {}  # pylint: disable=protected-access
        assert 'ow-octorok' in grid._policies.names  # pylint: disable=protected-access
    finally:
        for part in (grid._feed, grid._pool, grid._golden):  # pylint: disable=protected-access
            if part is not None:
                part.close()


def test_it_plays_without_trials(shift_models):
    """An evaluation plays it from the observation alone; an act mode that tries actions in the emulator (the
    planner) is refused for it."""
    state = require_state(SEGMENT_STATE)
    config = _native.SpecialistConfig.load('ow-octorok').replace(start_states=[state.stem], max_steps=12)
    run = registry.run_dir('ow-octorok', 'shift-w8', shift_models)

    def play(act):
        return evaluate_specialist(config, run, episodes_per_state=2, parallel=2, seed=2, states_dir=state.parent,
                                   act=act, clips='on')

    assert play(registry.act_mode(run))['episodes'] >= 2
    with pytest.raises(ValueError, match='no foresight'):
        play('temperature:0.5+relax:16+plan:16:1')
