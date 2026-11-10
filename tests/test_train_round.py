"""scripts/train_round.py end to end with a stub trainer and evaluation: what gets promoted, when, and what the
round summary keeps of the models it replaces."""

import importlib.util
import json
from pathlib import Path

import pytest

from triforce import registry
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.observation import observation_space

SPECIALIST = 'goriya'
SMALL = ModelConfig(hidden_dim=32, entity_d_model=32, entity_ff_dim=32, grid_out_dim=32)
REPO_STATES_DIR = Path(__file__).resolve().parents[1] / 'triforce' / 'states'


class _RepoStates:     # pylint: disable=too-few-public-methods
    """A specialist config whose seed states resolve in the repo's route states (the session default holds only
    ``start.state``)."""
    def __init__(self, config):
        self._config = config

    def __getattr__(self, name):
        return getattr(self._config, name)

    def resolve_start_states(self, states_dir=None):
        return self._config.resolve_start_states(str(states_dir or REPO_STATES_DIR))


@pytest.fixture(name='round_')
def fixture_round(monkeypatch):
    spec = importlib.util.spec_from_file_location('train_round', Path(__file__).parents[1] / 'scripts' /
                                                  'train_round.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'say', lambda *_: None)
    load_specialist = module.load_specialist
    monkeypatch.setattr(module, 'load_specialist', lambda name: _RepoStates(load_specialist(name)))
    network = ZeldaPolicy(observation_space(), SMALL)

    def train(name, args, models, run_dir, _log_dir):
        registry.save_checkpoint(name, args.round, network, root=models)
        (run_dir / 'train_summary.json').write_text(json.dumps({'status': 'completed', 'iterations': 3}))
        return 0
    module.real_train = module.train
    monkeypatch.setattr(module, 'train', train)
    return module


def _result(success_rate):
    stats = {'success_rate': success_rate, 'episodes': 2, 'failures': {}}
    return {'format': 2, 'specialist': SPECIALIST, 'act': 'sample', 'deterministic': False,
            'success_rate': success_rate, 'episodes': 2, 'mean_health_fraction': 0.5,
            'mean_start_health_fraction': 0.6, 'mean_health_change': -0.5, 'mean_health_change_total': -0.5,
            'hearts_dropped': 0.0, 'hearts_taken': 0.0, 'hearts_left': 0.0, 'by_room': {'1:0x23': stats},
            'by_state': {'1_23s_0': stats}, 'records': []}


def _evaluate(fail=False):
    def evaluate(*_args, **_kwargs):
        if fail:
            raise RuntimeError("evaluation broke")
        return _result(0.75)
    return evaluate


def _previous(models):
    registry.save_checkpoint(SPECIALIST, 'old', ZeldaPolicy(observation_space(), SMALL), root=models)
    registry.promote(SPECIALIST, 'old', root=models, act='argmax')


def _round(round_, models, *extra, code=0, names=SPECIALIST):
    assert round_.main(['--round', 'r', '--models-dir', str(models), '--specialists', names,
                        '--chain-runs', '0', *extra]) == code
    return json.loads((models / 'r-summary.json').read_text())


def _plan(logs_dir):
    """The round's training.json and its event streams (the only log folder under ``logs_dir``)."""
    folder, = [p for p in logs_dir.iterdir() if p.is_dir()]
    streams = {p.stem: [json.loads(line)['event'] for line in p.read_text().splitlines()]
               for p in folder.glob('*.jsonl') if p.stem != 'round'}
    return json.loads((folder / 'training.json').read_text()), streams


def test_without_promote_the_promoted_model_stays(round_, tmp_path, monkeypatch):
    _previous(tmp_path)
    monkeypatch.setattr(round_, 'evaluate_specialist', _evaluate())
    summary = _round(round_, tmp_path)
    assert registry.promoted_run(SPECIALIST, tmp_path) == 'old'
    assert summary['previous_promoted'] == {SPECIALIST: {'run': 'old', 'act': 'argmax'}}
    record = summary['specialists'][SPECIALIST]
    assert record['status'] == 'ok' and not record.get('promoted') and record['eval_act'] == 'sample'
    assert json.loads(Path(record['eval_path']).read_text())['success_rate'] == 0.75
    assert Path(record['eval_path']) == registry.run_dir(SPECIALIST, 'r', tmp_path) / 'eval.json'


def test_promote_waits_for_a_successful_evaluation(round_, tmp_path, monkeypatch):
    _previous(tmp_path)
    monkeypatch.setattr(round_, 'evaluate_specialist', _evaluate(fail=True))
    summary = _round(round_, tmp_path, '--promote', code=1)                 # a failed evaluation fails the round
    assert summary['specialists'][SPECIALIST]['status'] == 'eval-failed'
    assert registry.promoted_run(SPECIALIST, tmp_path) == 'old'          # an unevaluated run is never promoted
    assert registry.act_mode(registry.promoted_checkpoint(SPECIALIST, tmp_path)) == 'argmax'


def test_promote_after_the_evaluation_records_what_it_replaced(round_, tmp_path, monkeypatch):
    _previous(tmp_path)
    monkeypatch.setattr(round_, 'evaluate_specialist', _evaluate())
    summary = _round(round_, tmp_path, '--promote')
    assert registry.promoted_run(SPECIALIST, tmp_path) == 'r'
    assert summary['specialists'][SPECIALIST]['promoted'] is True
    assert summary['previous_promoted'][SPECIALIST] == {'run': 'old', 'act': 'argmax'}


def test_a_specialist_without_a_promoted_model_records_none(round_, tmp_path, monkeypatch):
    monkeypatch.setattr(round_, 'evaluate_specialist', _evaluate())
    summary = _round(round_, tmp_path)
    assert summary['previous_promoted'] == {SPECIALIST: {'run': None, 'act': None}}
    with pytest.raises(FileNotFoundError):
        registry.promoted_run(SPECIALIST, tmp_path)


def test_training_success_weights_metrics_rows_by_their_episodes(round_, tmp_path):
    rows = [{'iteration': i, 'steps': 100 * i, 'success_rate': rate, 'episodes': episodes}
            for i, (rate, episodes) in enumerate([(0.0, 50), (1.0, 1), (0.5, 4), (0.0, 0), (0.25, 4), (1.0, 1)])]
    (tmp_path / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    record = round_.training_record(tmp_path)
    assert record['train_success_last5'] == pytest.approx((0 * 50 + 1 + 0.5 * 4 + 0.25 * 4 + 1) / 60)


def test_train_args_expand_only_the_name_placeholder(round_, tmp_path, monkeypatch):
    commands = []

    class Popen:    # pylint: disable=too-few-public-methods
        """Records the command; prints nothing and exits 0."""
        def __init__(self, command, **_kwargs):
            commands.append(command)
            self.stdout, self.returncode = [], 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False
    monkeypatch.setattr(round_.subprocess, 'Popen', Popen)
    args = round_.parse_args(['--round', 'r', '--train-args', '--load /x/{name} --search \'{"candidates": 4}\''])
    assert round_.real_train(SPECIALIST, args, tmp_path, tmp_path / 'run', tmp_path) == 0
    assert commands[0][-4:] == ['--load', '/x/goriya', '--search', '{"candidates": 4}']


def test_unknown_specialists_stop_the_round_before_anything_trains(round_, tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(round_, 'train', lambda *args: calls.append(args) or 0)
    assert round_.main(['--round', 'r', '--models-dir', str(tmp_path), '--specialists', 'goriya,wallmastr']) == 2
    assert 'wallmastr' in capsys.readouterr().err and not calls


def test_a_failing_specialist_fails_the_round_but_not_the_others(round_, tmp_path, monkeypatch, logs_dir):
    """goriya's trainer saves a checkpoint and exits 3; zol's driver step raises; wallmaster trains.  The round
    goes on, ends 'failed' with exit 1, every stream ends with `end`, and the chain uses only wallmaster's run."""
    stub = round_.train

    def train(name, args, models, run_dir, log_dir):
        if name == 'zol':
            raise RuntimeError("set_aside broke")
        code = stub(name, args, models, run_dir, log_dir)
        return 3 if name == 'goriya' else code
    monkeypatch.setattr(round_, 'train', train)
    monkeypatch.setattr(round_, 'evaluate_specialist', _evaluate())
    chains = []

    def eval_chain(_route, runs, _parallel, **kwargs):
        chains.append(kwargs['overrides'])
        return [], {'runs': runs, 'successes': 0, 'success_rate': 0.0, 'failures': {}, 'errors': []}
    monkeypatch.setattr(round_, 'eval_chain', eval_chain)
    assert round_.main(['--round', 'r', '--models-dir', str(tmp_path), '--specialists', 'goriya,zol,wallmaster',
                        '--chain-runs', '1']) == 1
    plan, streams = _plan(logs_dir)
    assert plan['status'] == 'failed' and 'goriya train-failed' in plan['error'] and 'zol train-failed' in plan['error']
    assert [run['status'] for run in plan['runs']] == ['train-failed', 'train-failed', 'ok']
    assert all(events[-1] == 'end' for events in streams.values()) and len(streams) == 3
    assert list(chains[0]) == ['wallmaster']


def test_a_summary_write_failure_does_not_fail_the_round(round_, tmp_path, monkeypatch, logs_dir):
    monkeypatch.setattr(round_, 'evaluate_specialist', _evaluate())
    writes = []
    write = round_.write_json_atomic

    def flaky(path, data, **kwargs):
        writes.append(path)
        if len(writes) == 2:        # the evaluation thread's summary write
            raise RuntimeError("dictionary changed size during iteration")
        return write(path, data, **kwargs)
    monkeypatch.setattr(round_, 'write_json_atomic', flaky)
    _round(round_, tmp_path)
    plan, streams = _plan(logs_dir)
    assert plan['status'] == 'completed' and plan['runs'][0]['status'] == 'ok'
    assert streams[SPECIALIST][-1] == 'end'


DRIVER = '''
import importlib.util, json, sys, time
from pathlib import Path
from triforce import registry
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.observation import observation_space
spec = importlib.util.spec_from_file_location('train_round', sys.argv[1])
round_ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(round_)
network = ZeldaPolicy(observation_space(), ModelConfig(hidden_dim=32, entity_d_model=32, entity_ff_dim=32,
                                                       grid_out_dim=32))

def train(name, args, models, run_dir, _log_dir):
    if name != 'goriya':
        time.sleep(120)                     # still training when the signal comes
    registry.save_checkpoint(name, args.round, network, root=models)
    (run_dir / 'train_summary.json').write_text(json.dumps({'status': 'completed', 'iterations': 3}))
    return 0

def evaluate(*_args, on_episode=None, **_kwargs):
    for done in range(1, 2401):             # two minutes of episodes
        time.sleep(0.05)
        on_episode({'success': True}, done, 2400)
    raise AssertionError("not stopped")

round_.train = train
round_.evaluate_specialist = evaluate
round_.interrupt_on_sigterm()
sys.exit(round_.main(['--round', 'r', '--models-dir', sys.argv[2], '--specialists', 'goriya,zol',
                      '--chain-runs', '0']))
'''


def test_sigterm_stops_training_and_evaluation_and_closes_every_run(tmp_path, logs_dir):
    import os, signal, subprocess, sys, time   # pylint: disable=import-outside-toplevel,multiple-imports
    script = tmp_path / 'driver.py'
    script.write_text(DRIVER)
    # The repo's states (the session `states_dir` fixture may have pointed TRIFORCE_STATES_DIR at a bare one).
    env = {**{k: v for k, v in os.environ.items() if k != 'TRIFORCE_STATES_DIR'}, 'CUDA_VISIBLE_DEVICES': ''}
    with subprocess.Popen([sys.executable, str(script), str(Path(__file__).parents[1] / 'scripts' /
                                                         'train_round.py'), str(tmp_path / 'models')],
                          env=env, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL) as process:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                statuses = [run['status'] for run in _plan(logs_dir)[0]['runs']]
            except (ValueError, OSError):   # the folder or plan is not written yet
                statuses = []
            if statuses == ['evaluating', 'running']:
                break
            assert process.poll() is None
            time.sleep(0.1)
        assert statuses == ['evaluating', 'running']
        signalled = time.monotonic()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=30) != 0
        assert time.monotonic() - signalled < 10
    plan, streams = _plan(logs_dir)
    assert plan['status'] == 'interrupted'
    assert [run['status'] for run in plan['runs']] == ['interrupted', 'interrupted']
    assert streams['goriya'][-1] == 'end' and streams['zol'][-1] == 'end'
    assert not (registry.run_dir('goriya', 'r', tmp_path / 'models') / 'eval.json').exists()


def _chain_stub(calls, error=None):
    def eval_chain(route, runs, parallel, **kwargs):
        calls.append({'route': route, 'runs': runs, 'parallel': parallel, **kwargs})
        if error is not None:
            raise error
        return [], {'runs': runs, 'successes': 1, 'success_rate': 1 / runs, 'failures': {'0:0x77 death': 1},
                    'errors': []}
    return eval_chain


def test_a_promoted_round_chains_the_promoted_models_and_keeps_the_chain_summary(round_, tmp_path, monkeypatch,
                                                                                  logs_dir):
    monkeypatch.setattr(round_, 'evaluate_specialist', _evaluate())
    calls = []
    monkeypatch.setattr(round_, 'eval_chain', _chain_stub(calls))
    stale = tmp_path / 'r-chain.jsonl'
    stale.write_text('{"seed": 0}\n')
    summary = _round(round_, tmp_path, '--promote', '--chain-runs', '2', '--chain-parallel', '1')
    call, = calls
    assert call['overrides'] is None and (call['runs'], call['parallel']) == (2, 1)    # promoted models play
    assert call['output'] == stale and not stale.exists()      # a rerun replaces the chain file
    assert summary['chain']['successes'] == 1
    assert json.loads((tmp_path / 'r-chain-summary.json').read_text()) == summary['chain']
    record = summary['specialists'][SPECIALIST]
    assert (record['eval_success_rate'], record['eval_episodes'], record['eval_health_change']) == (0.75, 2, -0.5)
    assert record['eval_by_room'] == {'1:0x23': 0.75}
    plan, _ = _plan(logs_dir)
    assert plan['status'] == 'completed'


def test_an_unpromoted_round_chains_its_trained_checkpoints_and_a_chain_error_fails_it(round_, tmp_path,
                                                                                         monkeypatch, logs_dir):
    monkeypatch.setattr(round_, 'evaluate_specialist', _evaluate(fail=True))
    calls = []
    monkeypatch.setattr(round_, 'eval_chain', _chain_stub(calls, RuntimeError("route broke")))
    summary = _round(round_, tmp_path, '--chain-runs', '1', code=1)
    # A run whose evaluation failed still finished training: its checkpoint plays in the chain.
    assert calls[0]['overrides'] == {SPECIALIST: str(registry.checkpoint_path(SPECIALIST, 'r', tmp_path))}
    assert summary['chain'] == {'error': 'RuntimeError: route broke'}
    assert json.loads((tmp_path / 'r-chain-summary.json').read_text()) == summary['chain']
    plan, _ = _plan(logs_dir)
    assert plan['status'] == 'failed' and 'eval-chain failed' in plan['error']


def test_format_table_shows_each_specialist_status_and_its_worst_room(round_):
    base = {'rooms': 3, 'seed_states': 6, 'steps': 1000, 'stopped_early': True, 'train_success_last5': 0.9,
            'train_balanced_last': 0.8, 'steps_per_second': 1234.4}
    table = round_.format_table({
        'goriya': {**base, 'status': 'ok', 'eval_success_rate': 0.95,
                   'eval_by_room': {'1:0x23': 0.9, '1:0x33': 0.5, '1:0x44': 0.5}},
        'zol': {**base, 'status': 'eval-failed', 'stopped_early': False},
        'wallmaster': {'rooms': None, 'seed_states': None, 'steps': None, 'stopped_early': None,
                       'train_success_last5': None, 'steps_per_second': None, 'status': 'train-failed'},
    }).splitlines()
    assert len(table) == 5 and table[0].startswith('| specialist |')
    assert table[2] == '| goriya | 3 | 6 | 1000 | yes | 90% | 80% | 95% | 1:0x33 (50%) | 1234 |'
    assert table[3] == '| zol | 3 | 6 | 1000 | no | 90% | 80% | eval-failed | - | 1234 |'
    assert table[4] == '| wallmaster | None | None | - | - | - | - | train-failed; - | - | - |'
