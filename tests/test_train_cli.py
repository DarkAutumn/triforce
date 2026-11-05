"""``python -m triforce.train`` against the native ``EnvPool`` (tiny rollouts from the base movie's start state)."""

import json
from collections import Counter

import pytest

from triforce import registry, train
from triforce.metrics import read_jsonl

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


@pytest.mark.parametrize('parallel', [1, 3])
def test_seeded_training_repeats_gameplay_for_any_thread_count(states_dir, tmp_path, parallel):
    histories = []
    for run_id, threads in (('one-thread', 1), ('threads', parallel)):
        run = train.main(_common(states_dir, tmp_path, '--parallel', str(parallel), '--threads', str(threads),
                                 '--iterations', '2', '--steps-per-update', '48', '--no-curriculum',
                                 '--run-id', f'{run_id}-{parallel}', '--seed', '31', '--device', 'cpu'))
        histories.append(_gameplay(run))
    assert histories[0] == histories[1]
    assert histories[0][0]['rollout/steps'] == 48


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
    assert summary['iterations'] == checkpoint['iterations'] == len(records)
    # Stops on the first iteration where every start state has min_episodes results (threshold 0).
    assert 'balanced_success_rate' in records[-1] and all('balanced_success_rate' not in r for r in records[:-1])

    # Standalone training gets its own log folder: training.json plus one event stream ending in 'end'.
    (log_dir,) = logs_dir.iterdir()
    plan = json.loads((log_dir / 'training.json').read_text())
    assert plan['status'] == 'completed' and [r['specialist'] for r in plan['runs']] == [SPECIALIST]
    assert plan['runs'][0]['status'] == 'early_stop' and plan['runs'][0]['steps'] == checkpoint['steps']
    events = read_jsonl(str(log_dir / plan['runs'][0]['events']))
    kinds = [e['event'] for e in events]
    assert kinds[0] == 'start' and kinds[-2:] == ['train_end', 'end'] and kinds.count('end') == 1
    assert kinds.index('early_stop') > max(i for i, k in enumerate(kinds) if k == 'progress')
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
