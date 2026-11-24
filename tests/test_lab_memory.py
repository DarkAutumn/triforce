"""The lab's training of networks with memory (``nets.gru``, ``LabNet.unroll``, ``RolloutBuffer.recompute_sequence``),
sampling across families by need (``lab.collect.NeedSampler``), and what a lab run shares with ``triforce train``:
harvested starts, numbered checkpoints for ``select-checkpoint``, the early stop."""

import json
import shutil
from collections import Counter

import torch

from triforce import _native, nets, registry, select_checkpoint
from triforce.evaluate import evaluate_specialist
from triforce.lab import train
from triforce.lab.collect import Family, MultiCollector, NeedSampler
from triforce.metrics import RolloutMetrics
from triforce.rollout_buffer import RolloutBuffer

from .conftest import REPO_STATES_DIR, ROOT, require_state, start_task
from .networks import native_spec


def _gru():
    torch.manual_seed(0)
    return nets.create('gru', native_spec(), hidden_dim=32, core_dim=16)


def _obs(rows):
    torch.manual_seed(1)
    return {k: (torch.randint(0, v['vocab'], (rows, *v['shape'])) if v['vocab'] else torch.randn(rows, *v['shape']))
            for k, v in native_spec().items() if k != 'action_mask'}


def test_unroll_is_step_by_step_with_resets_at_episode_starts():
    network = _gru()
    assert network.recurrent and not nets.create('wide', native_spec(), hidden_dim=16, depth=1).recurrent
    obs = _obs(2 * 4)                                   # two sequences of four
    starts = torch.tensor([False, False, True, False, True, False, False, False])
    state = torch.randn(2, 16)
    heads, final = network.unroll(obs, starts, state)
    with torch.no_grad():
        for c in range(2):
            s = state[c:c + 1]
            for t in range(4):
                row = c * 4 + t
                if starts[row]:
                    s = network.initial_state(1)
                step, s = network.heads({k: v[row:row + 1] for k, v in obs.items()}, s)
                assert torch.allclose(step.kind, heads.kind[row:row + 1], atol=1e-6)
                assert torch.allclose(step.value, heads.value[row:row + 1], atol=1e-6)
            assert torch.allclose(s, final[c:c + 1], atol=1e-6)


def test_sequence_recompute_keeps_start_states_and_carries_across_rollouts(states_dir, tmp_path):
    network = _gru()
    config = _native.SpecialistConfig.load('ow-start').replace(start_states=['start'], max_steps=6)
    task = _native.TrainTask(config, str(tmp_path / 'task'), states_dir=str(states_dir))
    with _native.EnvPool(task, num_envs=2, threads=2, seed=3) as pool:
        buffers = [RolloutBuffer(8, 2, pool.obs_spec, 0.99, 0.95) for _ in range(2)]
        for buffer in buffers:
            pool.rollout([network.to_native()], 8, buffer.rollout_buffers())
    _, carry = buffers[0].recompute_sequence(network, 'cpu', None, 4)
    assert buffers[0].chunk_states.shape == (2, 2, 16)
    assert torch.equal(buffers[0].chunk_states[:, 0], network.initial_state(2))
    batch = buffers[0].flatten('cpu')
    heads, final = network.unroll(batch['obs'], batch['starts'], batch['chunk_states'])
    with torch.no_grad():
        logp = buffers[0].logp_ent_val.reshape(-1, 3)
        assert torch.allclose(heads.value, logp[:, 2], atol=1e-5)
        assert torch.allclose(final.view(2, 2, 16)[:, 1], carry, atol=1e-5)
    buffers[1].recompute_sequence(network, 'cpu', carry, 4)
    first = buffers[1].chunk_states[:, 0]
    fresh = buffers[1].dones[:, 0].bool()
    assert torch.allclose(first[~fresh], carry[~fresh]) and torch.equal(first[fresh], network.initial_state(2)[fresh])


def test_a_recurrent_network_trains_on_sequences_and_plays(states_dir, tmp_path):
    for mode in ('ppo', 'imitate'):
        extra = ['--teacher-coeff', '0'] if mode == 'ppo' else ['--actor', 'student']
        if mode == 'imitate':
            registry.save_checkpoint('ow-start', 'teach', nets.create('zelda', native_spec()), root=tmp_path)
            registry.promote('ow-start', 'teach', tmp_path)
        # ow-start from the bare state ``start``: no state_routes, so --all-routes (round 40, lab.recipe.check_routes)
        run = train.main(['--name', f'mem-{mode}', '--run-id', 'g1', '--families', f'ow-start={start_task()}',
                          '--parallel', '2', '--all-routes',
                          '--steps-per-update', '64', '--device', 'cpu', '--models-dir', str(tmp_path), '--seed', '6',
                          '--iterations', '2', '--arch', 'gru', '--bptt', '8', '--mode', mode,
                          '--arch-config', '{"hidden_dim": 32, "core_dim": 16}', *extra])
        lines = (run / 'metrics.jsonl').read_text().splitlines()
        assert len(lines) == 2
    config = _native.SpecialistConfig.load('ow-start').replace(start_states=['start'], max_steps=12)
    result = evaluate_specialist(config, tmp_path / 'ow-start' / 'g1', episodes_per_state=2, parallel=2, seed=2,
                                 states_dir=states_dir, act='temperature:0.5+relax:16', clips='on')
    assert result['episodes'] == 2


def test_need_sampling_gives_the_rows_to_the_families_below_target():
    sampler = NeedSampler({'a': 0.9, 'b': 0.9}, floor=0.1, window=10)
    assert sampler.shares() == {'a': 0.5, 'b': 0.5}                 # no episodes yet: both need their target
    sampler.observe('a', [{'info': {'outcome': 'success'}}] * 10)
    sampler.observe('b', [{'info': {'outcome': 'failure'}}] * 5 + [{'info': {'outcome': 'success'}}] * 5)
    assert sampler.need('a') == 0.0 and abs(sampler.need('b') - 0.4) < 1e-9
    shares = sampler.shares()
    assert abs(shares['a'] - 1 / 6) < 1e-9 and abs(shares['b'] - 5 / 6) < 1e-9      # 0.1 : 0.5 (floor + need)
    steps = sampler.steps({'a': 2, 'b': 2}, 512, unit=16)
    assert steps == {'a': 48, 'b': 208} and all(s % 16 == 0 for s in steps.values())


def test_a_need_sampled_run_changes_and_logs_each_familys_rows(tmp_path):
    yaml = tmp_path / 'short.yaml'
    yaml.write_text(_native.SpecialistConfig.load(start_task()).replace(max_steps=8).to_json())
    run = train.main(['--name', 'need', '--run-id', 'n1', '--families', f'ow-start={start_task()}',
                      f'ow-start@d3first={yaml}', '--all-routes',
                      '--parallel', '2', '--steps-per-update', '64', '--device', 'cpu', '--models-dir', str(tmp_path),
                      '--seed', '7', '--iterations', '3', '--arch', 'wide', '--sampling', 'need',
                      '--need-target', '1.0', '--arch-config', '{"hidden_dim": 16, "depth": 1}',
                      '--checkpoint-every', '1', '--checkpoint-keep', '2'])
    lines = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    shares = [lines[i].get('metrics', lines[i]) for i in range(3)]
    for metrics in shares:
        total = metrics['family/ow-start/rows_share'] + metrics['family/ow-start@d3first/rows_share']
        assert abs(total - 1.0) < 1e-9 and 'family/ow-start/need' in metrics
    # Numbered checkpoints are select-checkpoint's candidates, through the family's link to the lab run.
    found = select_checkpoint.candidates('ow-start', registry.run_dir('ow-start', 'n1', tmp_path), tmp_path,
                                         promoted=False)
    labels = [candidate.label for candidate in found]
    assert 'last' in labels and 'it000002' in labels and 'it000001' not in labels     # --checkpoint-keep 2


def test_harvested_starts_are_drawn_in_a_multi_family_run(states_dir, tmp_path):
    seed = require_state('0_5bw_0')
    harvested = tmp_path / 'harvested'
    harvested.mkdir()
    for i in range(2):
        shutil.copy(seed, harvested / f'd3first_0_5bw_h{i}_0.state')
    states = tmp_path / 'states'
    states.mkdir()
    shutil.copy(seed, states / '0_5bw_0.state')
    config = _native.SpecialistConfig.load('ow-moblin').replace(harvested_states=str(harvested),
                                                               harvested_fraction=1.0, max_steps=2)
    other = _native.SpecialistConfig.load('ow-start').replace(start_states=['start'], max_steps=2)
    shutil.copy(states_dir / 'start.state', states / 'start.state')
    collector = MultiCollector([Family('ow-moblin', config, 1), Family('ow-start', other, 1)], tmp_path / 'run', 4,
                               0.99, 0.95, seed=1, states_dir=states)
    try:
        collector.start(nets.create('wide', native_spec(), hidden_dim=16, depth=1))
        collector.take()
    finally:
        collector.close()
    moblin = [e['info']['start_state'] for e in collector.episodes if e['family'] == 'ow-moblin']
    assert moblin and all('_h' in s for s in moblin), moblin


class _Collector:
    """What ``train.FamilyStop`` reads of a ``MultiCollector``."""

    def __init__(self, config):
        self.families = [Family('ow-start', config, 1)]
        self.pools = [type('Pool', (), {'start_states': ['start']})()]
        metrics = RolloutMetrics()
        metrics.cohorts = [(0, Counter({('start', True): 6}))]
        metrics.episodes = 6
        self.by_family = {'ow-start': metrics}


def test_the_early_stop_needs_every_family_and_a_confirming_evaluation(monkeypatch):
    config = _native.SpecialistConfig.load('ow-start').replace(max_steps=8)
    args = train.parse_args(['--name', 'x', '--families', 'ow-start', '--stop-success', '0.9', '--stop-patience', '1',
                             '--steps-per-update', '64', '--parallel', '2'])
    rates = iter([{'start': (1, 10)}, {'start': (10, 10)}])
    monkeypatch.setattr(train, 'success_by_state', lambda *a, **k: next(rates))
    stop = train.FamilyStop(args, _Collector(config), 32)
    network = type('Net', (), {'to_native': lambda self: None})()
    metrics = {}
    assert not stop.update(network, 1, metrics)            # min_iterations: ceil(8 / 32) + 1 = 2
    assert metrics['family/ow-start/balanced_success_rate'] == 1.0
    assert not stop.update(network, 2, {})                 # proposed; the evaluation says 0.1: continue
    assert stop.update(network, 3, {})                     # proposed again, confirmed
