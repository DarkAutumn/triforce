"""``python -m triforce.train`` against the native ``EnvPool`` (tiny rollouts from the base movie's start state)."""

import argparse
import importlib.util
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from triforce import registry, train
from triforce.metrics import EpisodeTracker, read_jsonl

SPECIALIST = 'ow-start'
GAMEPLAY_PREFIXES = ('kind/', 'duration/', 'interrupt/', 'ending/', 'outcome/', 'reward/', 'health', 'hearts/',
                     'linger/', 'rollout/')
GAMEPLAY_KEYS = ('episodes', 'episode_steps', 'episode_frames', 'success_rate', 'blocks')


def _common(states_dir, tmp_path, *extra):
    return ['--specialist', SPECIALIST, '--states-dir', str(states_dir), '--models-dir', str(tmp_path), *extra]


def _check_run(run_dir, iterations):
    checkpoint = registry.load_checkpoint(run_dir)          # raises on schema mismatch
    assert checkpoint['specialist'] == SPECIALIST and checkpoint['iterations'] == iterations
    assert checkpoint['optimizer'] is not None and checkpoint['optimizer']['state']
    records = read_jsonl(str(run_dir / 'metrics.jsonl'))
    assert records, "no metrics logged"
    last = records[-1]
    for key in ('rollout/steps', 'rollout/frames', 'episodes', 'entropy/kind', 'entropy/direction',
                'entropy/duration', 'charts/SPS', 'losses/approx_kl', 'losses/kl_rollback'):
        assert key in last, key
    assert any(k.startswith('interrupt/') for k in last)
    assert any(k.startswith('kind/') for k in last)
    assert (run_dir / 'tb').is_dir() and (run_dir / 'run.json').is_file()
    return checkpoint, records


def _gameplay(run_dir):
    return [{key: value for key, value in record.items() if key.startswith(GAMEPLAY_PREFIXES) or key in GAMEPLAY_KEYS}
            for record in read_jsonl(str(run_dir / 'metrics.jsonl'))]


def test_train_then_resume(states_dir, tmp_path):
    common = _common(states_dir, tmp_path, '--parallel', '2', '--steps-per-update', '32', '--iterations', '1',
                     '--seed', '1')
    first = train.main(common + ['--run-id', 'first'])
    checkpoint, records = _check_run(first, iterations=1)
    assert checkpoint['steps'] == 32 and records[0]['rollout/steps'] == 32

    # Resuming the same specialist continues the counters and the optimizer state.
    second = train.main(common + ['--run-id', 'second', '--load', str(first)])
    resumed, records = _check_run(second, iterations=2)
    assert resumed['steps'] == 64 and records[0]['iteration'] == 2
    first_steps = {p: s['step'] for p, s in checkpoint['optimizer']['state'].items()}
    assert all(resumed['optimizer']['state'][p]['step'] > first_steps[p] for p in first_steps)
    run = json.loads((second / 'run.json').read_text())
    assert run['args']['parallel'] == 2 and run['args']['threads'] == 2


def test_supplied_config_and_explicit_start_state(states_dir, tmp_path):
    config = tmp_path / 'alternate.yaml'
    config.write_text("name: ow-start\nrooms: ['(0, 0x77)', '(0, 0x77, cave)', '(0, 0x39)']\n"
                      "start_states: ['start']\nsuccess: [exit-correct]\nmax_steps: 1\n")
    run = train.main(['--specialist', str(config), '--parallel', '2', '--iterations', '1',
                      '--steps-per-update', '8', '--no-curriculum', '--models-dir', str(tmp_path),
                      '--start-state', str(states_dir / 'start.state'), '--run-id', 'alternate', '--seed', '7'])
    metrics = read_jsonl(str(run / 'metrics.jsonl'))[-1]
    assert metrics.get('ending/max-steps') == 1
    assert metrics['episode_steps'] == 1
    assert metrics['episodes'] == 8


@pytest.mark.parametrize('target', [False, True])
def test_load_widens_an_older_checkpoint_only_with_obs_target(states_dir, tmp_path, target):
    from triforce.model import ModelConfig, ZeldaPolicy, obs_spec_from_space  # pylint: disable=import-outside-toplevel
    from triforce.observation import observation_space  # pylint: disable=import-outside-toplevel
    spec = obs_spec_from_space(observation_space())
    del spec['target']
    old = registry.save_checkpoint(SPECIALIST, 'schema1', ZeldaPolicy(config=ModelConfig(obs_spec=spec, hidden_dim=32)),
                                   iterations=3, steps=96, root=tmp_path)
    checkpoint = torch.load(old, weights_only=False)
    checkpoint['obs_schema_version'] = 1
    torch.save(checkpoint, old)
    config = tmp_path / 'target.yaml'
    config.write_text("name: ow-start\nrooms: ['(0, 0x77)', '(0, 0x77, cave)', '(0, 0x39)']\n"
                      f"start_states: ['start']\nsuccess: [exit-correct]\nmax_steps: 4\nobs: {{target: {target}}}\n")
    run = train.main(['--specialist', str(config), '--parallel', '2', '--iterations', '1', '--steps-per-update', '16',
                      '--no-curriculum', '--models-dir', str(tmp_path), '--start-state', str(states_dir / 'start.state'),
                      '--run-id', 'continued', '--seed', '7', '--load', str(old.parent)])
    resumed = registry.load_checkpoint(run)
    assert ('target' in resumed['model_config']['obs_spec']) == target
    assert resumed['iterations'] == 4 and resumed['steps'] == 96 + 16
    # The optimizer state matches the saved (widened or not) first layers.
    feature_dim = ZeldaPolicy(config=resumed['model_config']).feature_dim
    assert (32, feature_dim) in {tuple(s['exp_avg'].shape) for s in resumed['optimizer']['state'].values()}


LEARNER_KEYS = ('losses/approx_kl', 'losses/policy_loss', 'losses/value_loss')


def _seeded_run(states_dir, tmp_path, run_id, parallel, threads, device, *extra):
    # Four iterations: rollouts run one update behind, so rollouts 3 and 4 are the first to act with learned weights.
    run = train.main(_common(states_dir, tmp_path, '--parallel', str(parallel), '--threads', str(threads),
                             '--iterations', '4', '--steps-per-update', '24', '--no-curriculum',
                             '--run-id', run_id, '--seed', '31', '--device', device, *extra))
    history = _gameplay(run)
    if device == 'cpu':     # the CPU update is bit-stable with one torch thread; CUDA compares gameplay only
        for entry, record in zip(history, read_jsonl(str(run / 'metrics.jsonl'))):
            entry.update({key: record[key] for key in LEARNER_KEYS})
    return history


@pytest.mark.parametrize('parallel,device', [(1, 'cpu'), (3, 'cpu'), pytest.param(
    3, 'cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA'))],
    ids=['repeat-1-thread-cpu', '1-vs-3-threads-cpu', '1-vs-3-threads-cuda'])
def test_seeded_training_repeats_gameplay_for_any_thread_count(states_dir, tmp_path, parallel, device):
    torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        first = _seeded_run(states_dir, tmp_path, f'one-thread-{parallel}-{device}', parallel, 1, device)
        second = _seeded_run(states_dir, tmp_path, f'threads-{parallel}-{device}', parallel, parallel, device)
        assert len(first) == 4 and first[0]['rollout/steps'] == 24
        assert first == second
        if parallel == 1 and device == 'cpu':
            # The comparison sees the learner: a 10x learning rate leaves the first rollouts (initial weights) alike
            # and changes the ones that act with updated weights.
            faster = _seeded_run(states_dir, tmp_path, 'faster-lr', parallel, 1, device, '--learning-rate', '3e-3')
            gameplay = [[{k: v for k, v in e.items() if k not in LEARNER_KEYS} for e in h] for h in (first, faster)]
            assert gameplay[0][:2] == gameplay[1][:2]
            assert gameplay[0][2:] != gameplay[1][2:]
    finally:
        torch.set_num_threads(torch_threads)


def test_early_stop_balances_start_states():
    """An oversampled failing state counts as one state; an iteration that finished no entry episode resets."""
    early = train.EarlyStop(0.75, 2, window=4, min_episodes=2)
    sequence = [([Counter({('a', True): 1, ('b', False): 9})], 10, False, None),   # a has too few episodes
                ([Counter({('a', True): 1})], 1, False, 0.5),                     # a 2/2, b 0/9 (one whole cohort)
                ([Counter({('b', True): 4})], 4, False, 1.0),                     # b 4/4 covers the window: streak 1
                ([], 0, False, 1.0),                                              # nothing finished: reset
                ([Counter({('b', True): 4})], 4, False, 1.0),                     # streak 1
                ([Counter({('a', True): 1}), Counter({('b', False): 1})], 2, True, 0.9)]   # b 4/5: streak 2
    for cohorts, finished, stop, balanced in sequence:
        assert early.update(cohorts, finished) == stop
        assert early.balanced == balanced
    never = train.EarlyStop(None, 1, min_episodes=1)
    assert not any(never.update([Counter({('a', True): 5})], 5) for _ in range(10)) and never.balanced == 1.0


def test_early_stop_balances_seed_states_only():
    """F140: 3 seed states at 50% and 100 harvested labels at 100%: the rate stays 0.5 and needs only the seeds."""
    early = train.EarlyStop(0.9, 1, min_episodes=2, seed_states=['a', 'b', 'c'])
    harvested = Counter({(f'0_77s_h{i}_3', True): 1 for i in range(100)})
    assert not early.update([harvested], 100) and early.balanced is None and early.harvested == (100, 100)
    seeds = Counter({(s, ok): 1 for s in 'abc' for ok in (True, False)})
    for _ in range(5):
        assert not early.update([seeds, harvested], 106)
        assert early.balanced == 0.5 and early.harvested == (100, 100)
    assert set(early.results) == {'a', 'b', 'c'}
    pinned = train.EarlyStop(0.9, 1, min_episodes=1)          # --start-state: no seed list, every label counts
    assert pinned.update([Counter({('x', True): 2})], 2) and pinned.harvested == (0, 0)


def test_training_logs_its_harvested_source_and_rate(states_dir, tmp_path, logs_dir):
    """F069/F140: run.json and the start event name the harvested dir, its state count and fraction; harvested
    episodes are logged as harvested/success_rate and the early stop balances over the seed state only."""
    harvest = tmp_path / 'harvest'
    harvest.mkdir()
    shutil.copyfile(states_dir / 'start.state', harvest / '0_77s_h1_0.state')
    config = tmp_path / 'short.yaml'
    config.write_text("name: ow-start\nrooms: ['(0, 0x77)', '(0, 0x77, cave)', '(0, 0x39)']\n"
                      f"start_states: ['start']\nsuccess: [exit-correct]\nmax_steps: 4\n"
                      f"harvested_states: {harvest}\nharvested_fraction: 0.5\n")
    run_dir = train.main(['--specialist', str(config), '--states-dir', str(states_dir), '--parallel', '4',
                          '--steps-per-update', '32', '--iterations', '4', '--models-dir', str(tmp_path),
                          '--run-id', 'harvest', '--seed', '3'])
    harvested = json.loads((run_dir / 'run.json').read_text())['harvested']
    assert (harvested['dir'], harvested['states'], harvested['fraction']) == (str(harvest), 1, 0.5)
    assert harvested['drawn'] == 'drawn on 50% of the resets (harvested_fraction)'
    (log_dir,) = [path for path in logs_dir.iterdir() if path.is_dir()]
    plan = json.loads((log_dir / 'training.json').read_text())
    start = next(e for e in read_jsonl(str(log_dir / plan['runs'][0]['events'])) if e['event'] == 'start')
    assert start['harvested'] == harvested and start['curriculum_tracks'] is None
    records = read_jsonl(str(run_dir / 'metrics.jsonl'))
    assert any('harvested/success_rate' in r for r in records)


def test_early_stop_guards_min_iterations_and_overdue_episodes():
    early = train.EarlyStop(0.9, 1, min_episodes=1, min_iterations=3)
    perfect = [Counter({('a', True): 20})]
    assert [early.update(perfect, 20) for _ in range(3)] == [False, False, True]     # not before iteration 3
    assert not early.update(perfect, 20, over_half_budget=0.2) and early.streak == 0  # 20% of envs overdue
    assert early.update(perfect, 20, over_half_budget=0.05)
    early.reset()
    assert early.streak == 0


@pytest.mark.parametrize('failure', [0, 49, 112])
def test_early_stop_rate_is_exact_over_whole_iterations(failure):
    """One iteration of 113 episodes with one failure reads 112/113 wherever the failure fell (F130)."""
    outcomes = [i != failure for i in range(113)]
    early = train.EarlyStop(0.99, 1)
    early.update([Counter((('a', success) for success in outcomes))], 113)
    assert early.balanced == pytest.approx(112 / 113)


@pytest.mark.parametrize('order', ['failures-first', 'alternating'])
def test_early_stop_half_rate_reads_half(order):
    outcomes = [i >= 20 for i in range(40)] if order == 'failures-first' else [i % 2 == 0 for i in range(40)]
    early = train.EarlyStop(0.99, 1)
    early.update([Counter((('a', success) for success in outcomes))], 40)
    assert early.balanced == pytest.approx(0.5)
    early.update([Counter({('a', True): 25})], 25)     # covers the window alone: the 40 drop out
    assert early.balanced == 1.0


def _stream(envs, success_rate, fail_steps, rollouts, seed, success_steps=50, states=('s0', 's1', 's2')):
    """A synthetic training pool (as probe_synthetic.py): every env plays episodes back to back, each a success after
    ``success_steps`` decisions with probability ``success_rate``, else a failure at the ``fail_steps`` budget.
    Yields per rollout (jobs, steps, fresh, events, truth): the rollout's rows as ``EnvPool.rollout`` writes them,
    and every episode's (start rollout, start state, success) by job."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    rng = random.Random(seed)
    steps_per_env = 4096 // envs
    truth, jobs_made = {}, iter(range(1, 1 << 40))
    live = []

    def begin(rollout):
        job, ok = next(jobs_made), rng.random() < success_rate
        truth[job] = (rollout, states[job % len(states)], ok)
        return [job, 0, success_steps if ok else fail_steps]

    live = [begin(1) for _ in range(envs)]
    for rollout in range(1, rollouts + 1):
        jobs = np.zeros((envs, steps_per_env), np.int64)
        steps = np.zeros((envs, steps_per_env), np.int64)
        fresh = np.zeros(envs, bool)
        events = []
        for env in range(envs):
            for t in range(steps_per_env):
                episode = live[env]
                jobs[env, t], steps[env, t] = episode[0], episode[1]
                episode[1] += 1
                if episode[1] == episode[2]:
                    _, state, ok = truth[episode[0]]
                    events.append({'env': env, 'job': episode[0], 'info': {
                        'outcome': 'success' if ok else 'soft-failure', 'start_state': state,
                        'episode_steps': episode[2]}})
                    live[env] = begin(rollout if t + 1 < steps_per_env else rollout + 1)
            fresh[env] = live[env][1] == 0
        yield jobs, steps, fresh, events, truth


@pytest.mark.parametrize('envs,fail_steps', [(16, 2000), (8, 3000)])
def test_a_cold_started_policy_with_slow_failures_does_not_trip_the_early_stop(envs, fail_steps):
    """p = 0.9 with 2000/3000-step failures: completion-time counting read ~1.0 for the first rollouts and stopped
    at iteration 5 in most runs (F071); the cohorts and guards never propose a stop."""
    for seed in range(20):
        tracker = EpisodeTracker(fail_steps)
        early = train.EarlyStop(0.99, 5, min_iterations=math.ceil(fail_steps / (4096 // envs)) + 5)
        for jobs, steps, fresh, events, _ in _stream(envs, 0.9, fail_steps, 30, seed):
            cohorts, values = tracker.update(jobs, steps, fresh, events)
            finished = len(events)
            assert not early.update([c for _, c in cohorts], finished, values['in_flight/over_half_budget'])


def test_an_iteration_with_every_env_in_flight_resets_the_streak():
    early = train.EarlyStop(0.9, 3, min_episodes=1)
    tracker = EpisodeTracker(600)
    early.update([Counter({('a', True): 10})], 10)
    early.update([Counter({('a', True): 10})], 10)
    assert early.streak == 2
    # Four envs, 64 rows each, all inside episodes that started earlier: nothing ends, nothing is released.
    jobs = np.repeat(np.arange(1, 5)[:, None], 64, axis=1)
    steps = np.tile(np.arange(400, 464), (4, 1))
    cohorts, values = tracker.update(jobs, steps, np.zeros(4, bool), [])
    assert cohorts == [] and values['in_flight/episodes'] == 4 and values['in_flight/over_half_budget'] == 1.0
    assert values['in_flight/mean_steps'] == values['in_flight/max_steps'] == 464
    assert not early.update(cohorts, 0, values['in_flight/over_half_budget']) and early.streak == 0


def test_cohort_rate_is_the_success_of_the_episodes_started_in_that_rollout():
    tracker = EpisodeTracker(2000)
    released = []
    for jobs, steps, fresh, events, truth in _stream(16, 0.7, 2000, 40, seed=3):
        cohorts, values = tracker.update(jobs, steps, fresh, events)
        released += cohorts
        if cohorts:
            pooled = sum((counts for _, counts in cohorts), Counter())
            assert values['started/success_rate'] == sum(n for (_, ok), n in pooled.items() if ok) / sum(
                pooled.values())
    # In rollout order; every rollout that started episodes is released once they (and earlier ones) all ended.
    started = sorted({start for start, _, _ in truth.values() if start <= tracker.released})
    assert [k for k, _ in released] == started and len(released) > 10
    for rollout, counts in released:
        expected = Counter((state, ok) for start, state, ok in truth.values() if start == rollout)
        assert counts == expected, rollout


@pytest.mark.parametrize('evaluated,stops', [((10, 50), False), ((50, 50), True)])
def test_a_stop_proposal_is_confirmed_by_an_evaluation(states_dir, tmp_path, logs_dir, monkeypatch, evaluated, stops):
    """Every proposal runs an evaluation of the current weights; below --stop-success the run goes on."""
    calls = []

    def fake_eval(config, policy, runs, seed=0, parallel=4, threads=None, states_dir=None, deterministic=False):
        # pylint: disable=too-many-arguments, too-many-positional-arguments, unused-argument
        calls.append((runs, seed, parallel, deterministic))
        return {'start': evaluated}

    monkeypatch.setattr(train, 'success_by_state', fake_eval)
    monkeypatch.setattr(train.EarlyStop, 'update', lambda self, *_: True)
    run_dir = train.main(['--specialist', SPECIALIST, '--states-dir', str(states_dir), '--parallel', '2',
                          '--steps-per-update', '16', '--iterations', '3', '--models-dir', str(tmp_path),
                          '--run-id', 'confirm', '--seed', '5', '--no-curriculum', '--stop-success', '0.5'])
    assert train.run_status(run_dir) == ('early_stop' if stops else 'completed')
    (log_dir,) = [path for path in logs_dir.iterdir() if path.is_dir()]
    events = read_jsonl(str(next(log_dir.glob('*.jsonl'))))
    checks = [e for e in events if e['event'] == 'early_stop_check']
    assert len(checks) == (1 if stops else 3) and len(calls) == len(checks)
    assert [c['decision'] for c in checks] == (['stop'] if stops else ['continue'] * 3)
    assert checks[0]['rate'] == evaluated[0] / evaluated[1] and checks[0]['states'] == {'start': checks[0]['rate']}
    assert calls[0] == (train.CONFIRM_EPISODES, 5 + 1, 2, False)
    assert ('early_stop' in [e['event'] for e in events]) == stops


def test_train_stops_early_and_writes_summary_and_events(states_dir, tmp_path, logs_dir):
    config = tmp_path / 'short.yaml'
    config.write_text("name: ow-start\nrooms: ['(0, 0x77)', '(0, 0x77, cave)', '(0, 0x39)']\n"
                      "start_states: ['start']\nsuccess: [exit-correct]\nmax_steps: 4\n")
    run_dir = train.main(['--specialist', str(config), '--states-dir', str(states_dir), '--parallel', '4',
                          '--steps-per-update', '16', '--iterations', '50', '--models-dir', str(tmp_path),
                          '--run-id', 'early', '--seed', '1', '--stop-success', '0.0', '--stop-patience', '1'])
    with open(run_dir / 'train_summary.json', encoding='utf-8') as file:
        summary = json.load(file)
    checkpoint = registry.load_checkpoint(run_dir)
    records = read_jsonl(str(run_dir / 'metrics.jsonl'))
    assert summary['stopped_early'] and checkpoint['iterations'] < 50
    assert summary['status'] == train.run_status(run_dir) == 'early_stop'
    assert summary['iterations_trained'] == summary['iterations']      # a fresh run
    assert summary['iterations'] == checkpoint['iterations'] == len(records)
    # Proposed once every start state has min_episodes cohort results and the guards allow it (threshold 0), then
    # confirmed by the evaluation.
    assert 'balanced_success_rate' in records[-1] and records[-1]['iteration'] >= 2

    # Standalone training gets its own log folder: training.json plus one event stream ending in 'end'.
    (log_dir,) = [path for path in logs_dir.iterdir() if path.is_dir()]
    plan = json.loads((log_dir / 'training.json').read_text())
    assert plan['status'] == 'completed' and [r['specialist'] for r in plan['runs']] == [SPECIALIST]
    assert plan['runs'][0]['status'] == 'early_stop' and plan['runs'][0]['steps'] == checkpoint['steps']
    events = read_jsonl(str(log_dir / plan['runs'][0]['events']))
    kinds = [e['event'] for e in events]
    assert kinds[0] == 'start' and kinds[-2:] == ['train_end', 'end'] and kinds.count('end') == 1
    assert kinds.index('early_stop') > max(i for i, k in enumerate(kinds) if k == 'progress')
    (check,) = [e for e in events if e['event'] == 'early_stop_check']
    assert check['decision'] == 'stop' and check['episodes'] == train.CONFIRM_EPISODES
    assert kinds.index('early_stop_check') < kinds.index('early_stop')
    progress = [e for e in events if e['event'] == 'progress']
    assert [e['iteration'] for e in progress] == [r['iteration'] for r in records]
    assert progress[-1]['steps'] == checkpoint['steps'] and progress[-1]['target_steps'] == 50 * 16
    assert [e['seq'] for e in events] == list(range(len(events)))


def test_train_error_is_logged_before_raising(states_dir, tmp_path):
    events_path = tmp_path / 'stream.jsonl'
    with pytest.raises(Exception) as error:
        train.main(_common(states_dir, tmp_path, '--parallel', '1', '--iterations', '1', '--run-id', 'bad',
                           '--curriculum', str(tmp_path / 'missing'), '--events', str(events_path)))
    events = read_jsonl(str(events_path))
    assert [e['event'] for e in events] == ['error', 'train_end']    # the caller owns 'end'
    assert str(error.value) in events[0]['message'] and 'Traceback' in events[0]['traceback']
    assert events[1]['status'] == 'error'


def test_an_interrupted_run_saves_but_exits_130(states_dir, tmp_path):
    """SIGTERM mid-run: the checkpoint and summary are saved, the summary says interrupted, the exit code is not 0,
    so no driver mistakes a partial run for a finished fixed-budget one."""
    command = [sys.executable, '-m', 'triforce.train', *_common(states_dir, tmp_path, '--parallel', '2',
               '--steps-per-update', '32', '--iterations', '100000', '--device', 'cpu', '--run-id', 'cut',
               '--events', str(tmp_path / 'stream.jsonl'))]
    metrics = tmp_path / SPECIALIST / 'cut' / 'metrics.jsonl'
    with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          env={**os.environ, 'TRIFORCE_STATES_DIR': str(states_dir)}) as process:
        deadline = time.monotonic() + 120
        while not (metrics.is_file() and metrics.read_text()) and time.monotonic() < deadline:
            assert process.poll() is None, "training ended on its own"
            time.sleep(0.05)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=120) == train.EXIT_INTERRUPTED
    run_dir = metrics.parent
    summary = json.loads((run_dir / 'train_summary.json').read_text())
    assert summary['status'] == train.run_status(run_dir) == 'interrupted' and summary['iterations_trained'] >= 1
    assert registry.load_checkpoint(run_dir)['iterations'] == summary['iterations']


def test_training_into_an_existing_run_dir_sets_the_old_run_aside(states_dir, tmp_path, monkeypatch):
    """A second training into run x moves x's files to x.prev-1 first: a crash in its first iteration leaves no
    'completed' status and no doubled metrics behind (F172)."""
    common = _common(states_dir, tmp_path, '--parallel', '2', '--steps-per-update', '32', '--iterations', '1',
                     '--seed', '4', '--no-curriculum', '--run-id', 'x')
    run_dir = train.main(common)
    (run_dir / 'act').write_text('argmax\n')
    (run_dir / 'eval.json').write_text('{}')
    assert train.run_status(run_dir) == 'completed'

    def crash(*_args, **_kwargs):
        raise RuntimeError("crash in the first iteration")

    with monkeypatch.context() as patch:
        patch.setattr(train.MetricsLogger, 'log', crash)
        with pytest.raises(RuntimeError):
            train.main(common)
    assert train.run_status(run_dir) not in ('completed', 'early_stop')
    assert not (run_dir / 'metrics.jsonl').exists() and not (run_dir / 'eval.json').exists()
    previous = run_dir.with_name('x.prev-1')
    assert train.run_status(previous) == 'completed'
    assert [r['iteration'] for r in read_jsonl(str(previous / 'metrics.jsonl'))] == [1]
    assert {'ckpt.pt', 'act', 'eval.json', 'run.json'} <= {p.name for p in previous.iterdir()}

    # A same-dir --load continuation appends its metrics; act/eval files of the old weights still go.
    train.main(common)
    (run_dir / 'act').write_text('argmax\n')
    (run_dir / 'eval-sample.json').write_text('{}')
    train.main(common + ['--load', str(run_dir)])
    assert [r['iteration'] for r in read_jsonl(str(run_dir / 'metrics.jsonl'))] == [1, 2]
    assert not (run_dir / 'act').exists() and not (run_dir / 'eval-sample.json').exists()
    assert (run_dir.with_name('x.prev-3') / 'eval-sample.json').is_file()
    assert registry.load_checkpoint(run_dir)['iterations'] == 2


def test_a_load_continuation_resumes_the_curriculum(states_dir, tmp_path):
    """A finished curriculum stays finished in a --load continuation; a fresh run starts it over (F088)."""
    trajectories = tmp_path / 'trajectories'
    track = trajectories / 'start'
    track.mkdir(parents=True)
    for index in range(4):
        shutil.copyfile(states_dir / 'start.state', track / f'{index:04d}.state')
    (track / 'trajectory.json').write_text(json.dumps({'start_state': 'start', 'success': True}))
    config = tmp_path / 'short.yaml'
    config.write_text("name: ow-start\nrooms: ['(0, 0x77)', '(0, 0x77, cave)', '(0, 0x39)']\n"
                      "start_states: ['start']\nsuccess: [exit-correct]\nmax_steps: 4\n")
    common = ['--specialist', str(config), '--states-dir', str(states_dir), '--models-dir', str(tmp_path),
              '--parallel', '2', '--steps-per-update', '32', '--iterations', '1', '--seed', '2',
              '--curriculum', str(trajectories), '--curriculum-fraction', '1.0']
    first = train.main(common + ['--run-id', 'first'])
    assert read_jsonl(str(first / 'metrics.jsonl'))[-1]['curriculum/episodes'] > 0
    tracks = json.loads((first / 'run.json').read_text())['curriculum_tracks']       # F131: what it loaded
    assert list(tracks) == ['start'] and tracks['start'].endswith('+00:00')
    progress = json.loads((first / 'curriculum.json').read_text())
    progress['start'].update(point=0, stages_passed=4)
    (first / 'curriculum.json').write_text(json.dumps(progress))

    second = train.main(common + ['--run-id', 'second', '--load', str(first)])
    assert 'curriculum/episodes' not in read_jsonl(str(second / 'metrics.jsonl'))[-1]
    assert json.loads((second / 'curriculum.json').read_text())['start']['point'] == 0
    assert json.loads((second / 'run.json').read_text())['curriculum'] == str(trajectories)

    fresh = train.main(common + ['--run-id', 'fresh'])
    assert json.loads((fresh / 'curriculum.json').read_text())['start']['point'] > 0


def test_an_interrupt_mid_update_saves_the_last_finished_iteration(states_dir, tmp_path, monkeypatch):
    """KeyboardInterrupt on the 2nd optimizer step of iteration 2: the checkpoint holds iteration 1's weights and
    optimizer state, not a partial update (F084)."""
    from triforce.ppo import PPO  # pylint: disable=import-outside-toplevel
    original, seen = PPO.optimize, {}

    def optimize(self, network, buffer):
        seen['calls'] = seen.get('calls', 0) + 1
        if seen['calls'] == 2:
            step, steps = self.optimizer.step, []

            def interrupted(*args, **kwargs):
                steps.append(1)
                if len(steps) == 2:
                    raise KeyboardInterrupt("test")
                return step(*args, **kwargs)
            self.optimizer.step = interrupted
        stats = original(self, network, buffer)
        seen['weights'] = {k: v.detach().clone() for k, v in network.state_dict().items()}
        seen['adam_steps'] = {int(s['step']) for s in self.optimizer.state_dict()['state'].values()}
        return stats

    monkeypatch.setattr(PPO, 'optimize', optimize)
    run_dir = train.main(_common(states_dir, tmp_path, '--parallel', '2', '--steps-per-update', '32', '--iterations',
                                 '5', '--learning-rate', '1e-6', '--run-id', 'cut-mid', '--seed', '3'))
    assert train.run_status(run_dir) == 'interrupted' and seen['calls'] == 2
    checkpoint = registry.load_checkpoint(run_dir)
    assert checkpoint['iterations'] == 1
    assert all(torch.equal(value, seen['weights'][key]) for key, value in checkpoint['state_dict'].items())
    assert {int(s['step']) for s in checkpoint['optimizer']['state'].values()} == seen['adam_steps']


def _train_round():
    spec = importlib.util.spec_from_file_location('train_round', Path(__file__).parents[1] / 'scripts' /
                                                  'train_round.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('flags', [['--save-every', '0'], ['--iterations', '0'], ['--stop-patience', '0'],
                                   ['--best-of-n', '1', '--best-every', '0'], ['--steps-per-update', '1']])
def test_bad_counts_are_rejected_before_the_run_dir_is_written(states_dir, tmp_path, flags):
    with pytest.raises(SystemExit):
        train.main(_common(states_dir, tmp_path, '--parallel', '2', '--run-id', 'bad', *flags))
    assert not (tmp_path / SPECIALIST).exists()


def test_target_steps_are_what_the_parallel_envs_train(states_dir, tmp_path, logs_dir):
    """--parallel 3 --steps-per-update 32 trains 30 steps per iteration; the targets say so (F136)."""
    run_dir = train.main(_common(states_dir, tmp_path, '--parallel', '3', '--steps-per-update', '32',
                                 '--iterations', '2', '--no-curriculum', '--run-id', 'thirty', '--seed', '1'))
    (log_dir,) = [path for path in logs_dir.iterdir() if path.is_dir()]
    plan = json.loads((log_dir / 'training.json').read_text())
    events = read_jsonl(str(log_dir / plan['runs'][0]['events']))
    start = next(e for e in events if e['event'] == 'start')
    assert start['target_steps'] == 60 == plan['runs'][0]['target_steps']
    assert registry.load_checkpoint(run_dir)['steps'] == 60
    assert plan['command'][3:5] == ['--specialist', SPECIALIST]


def test_round_driver_sets_unfinished_runs_aside_and_skips_finished_ones(tmp_path, monkeypatch):
    round_ = _train_round()
    monkeypatch.setattr(round_, 'say', lambda *_: None)
    run_dir = registry.run_dir(SPECIALIST, 'r', tmp_path)
    ckpt = registry.checkpoint_path(SPECIALIST, 'r', tmp_path)

    def finish(status, code=0):
        def fake(_name, _args, _models, directory, _log_dir):
            directory.mkdir(parents=True)
            ckpt.write_bytes(b'not a real checkpoint')
            (directory / 'train_summary.json').write_text(json.dumps({'status': status, 'iterations': 3}))
            calls.append(directory)
            return code
        return fake

    calls = []
    monkeypatch.setattr(round_, 'train', finish('interrupted', train.EXIT_INTERRUPTED))
    args = argparse.Namespace(round='r', skip_existing=True)
    record = round_.train_specialist(SPECIALIST, args, tmp_path, tmp_path)
    assert record['status'] == 'train-failed' and record['train_status'] == 'interrupted'
    assert not (tmp_path / SPECIALIST / registry.PROMOTED_NAME).exists()

    monkeypatch.setattr(round_, 'train', finish('completed'))
    record = round_.train_specialist(SPECIALIST, args, tmp_path, tmp_path)
    assert calls == [run_dir, run_dir] and record['status'] == 'ok' and record['train_status'] == 'completed'
    partial = run_dir.with_name('r.partial-1')
    assert train.run_status(partial) == 'interrupted'                      # kept, not overwritten

    round_.train_specialist(SPECIALIST, args, tmp_path, tmp_path)
    assert calls == [run_dir, run_dir]                                     # finished: skipped
    kinds = [e['event'] for e in read_jsonl(str(tmp_path / f'{SPECIALIST}.jsonl'))]
    assert kinds.count('partial_set_aside') == 1 and kinds[-1] == 'skipped'
