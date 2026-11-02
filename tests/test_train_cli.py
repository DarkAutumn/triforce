"""``python -m triforce.train`` against the real ZeldaEnv (tiny rollouts)."""

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
