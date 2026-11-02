"""``python -m triforce.train`` against the real ZeldaEnv (tiny rollouts)."""

import json
from collections import Counter

from triforce import registry, train
from triforce.metrics import read_jsonl

SPECIALIST = 'd1-melee'


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


def test_train_in_process_then_resume(session, tmp_path):
    common = ['--specialist', SPECIALIST, '--parallel', '1', '--steps-per-update', '32', '--iterations', '1',
              '--models-dir', str(tmp_path), '--seed', '1']
    first = train.main(common + ['--run-id', 'first'], session=session)
    checkpoint, _ = _check_run(first, iterations=1)
    assert checkpoint['steps'] == 32

    # Resuming the same specialist continues the counters and the optimizer state.
    second = train.main(common + ['--run-id', 'second', '--load', str(first)], session=session)
    resumed, records = _check_run(second, iterations=2)
    assert resumed['steps'] == 64 and records[0]['iteration'] == 2
    first_steps = {p: s['step'] for p, s in checkpoint['optimizer']['state'].items()}
    assert all(resumed['optimizer']['state'][p]['step'] > first_steps[p] for p in first_steps)


def test_train_parallel_workers(tmp_path):
    run_dir = train.main(['--specialist', SPECIALIST, '--parallel', '2', '--steps-per-update', '128',
                          '--iterations', '2', '--models-dir', str(tmp_path), '--run-id', 'parallel'])
    checkpoint, records = _check_run(run_dir, iterations=2)
    assert checkpoint['steps'] == 256 and [r['iteration'] for r in records] == [1, 2]
    assert sum(r['rollout/steps'] for r in records) == 256


def test_early_stop_balances_start_states():
    """An oversampled failing state counts as one state; iterations with no entry episodes keep the streak."""
    early = train.EarlyStop(0.75, 2, window=4, min_episodes=2)
    sequence = [(Counter({('a', True): 1, ('b', False): 9}), False, None),     # a has too few episodes
                (Counter({('a', True): 1}), False, 0.5),                       # a 2/2, b 0/4 (window)
                (Counter({('b', True): 4}), False, 1.0),                       # streak 1
                (Counter(), False, 1.0),                                       # unchanged
                (Counter({('a', True): 1, ('b', False): 1}), True, 0.875)]     # b 3/4: streak 2
    for start_results, stop, balanced in sequence:
        assert early.update(start_results) == stop
        assert early.balanced == balanced
    never = train.EarlyStop(None, 1, min_episodes=1)
    assert not any(never.update(Counter({('a', True): 5})) for _ in range(10)) and never.balanced == 1.0


def test_train_stops_early_and_writes_summary(session, tmp_path):
    run_dir = train.main(['--specialist', 'boss-aquamentus', '--parallel', '1', '--steps-per-update', '512',
                          '--iterations', '50', '--models-dir', str(tmp_path), '--run-id', 'early', '--seed', '1',
                          '--stop-success', '0.0', '--stop-patience', '1'], session=session)
    with open(run_dir / 'train_summary.json', encoding='utf-8') as file:
        summary = json.load(file)
    checkpoint = registry.load_checkpoint(run_dir)
    records = read_jsonl(str(run_dir / 'metrics.jsonl'))
    assert summary['stopped_early'] and checkpoint['iterations'] < 50
    assert summary['iterations'] == checkpoint['iterations'] == len(records)
    # Stops on the first iteration where every start state has min_episodes results (threshold 0).
    assert 'balanced_success_rate' in records[-1] and all('balanced_success_rate' not in r for r in records[:-1])
