"""The ``triforce`` command (``triforce.cli.main``) in-process: list, promote, argument errors and passthrough."""

import json

import pytest

from triforce import cli, registry
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.observation import observation_space
from triforce.runner import RunnerError
from triforce.specialist import list_specialists

SMALL = ModelConfig(hidden_dim=32, entity_d_model=32, entity_ff_dim=32, grid_out_dim=32)


@pytest.fixture(scope='module')
def network():
    return ZeldaPolicy(observation_space(), SMALL)


def _listing(capsys):
    rows = {}
    for line in capsys.readouterr().out.splitlines():
        name, promoted, runs = line.split()
        rows[name] = (promoted.removeprefix('promoted='), int(runs.removeprefix('runs=')))
    return rows


def test_list_shows_every_specialist_with_its_promoted_run_and_run_count(tmp_path, network, capsys):
    for run_id in ('a', 'b'):
        registry.save_checkpoint('ow-start', run_id, network, root=tmp_path)
    registry.save_checkpoint('goriya', 'x', network, root=tmp_path)
    registry.promote('ow-start', 'b', root=tmp_path)
    assert cli.main(['--models-dir', str(tmp_path), 'list']) == 0
    rows = _listing(capsys)
    assert list(rows) == list_specialists()
    assert rows['ow-start'] == ('b', 2) and rows['goriya'] == ('-', 1)
    assert all(rows[name] == ('-', 0) for name in rows if name not in ('ow-start', 'goriya'))


def test_promote_switches_the_listed_run_and_rejects_missing_runs(tmp_path, network, capsys):
    for run_id in ('a', 'b'):
        registry.save_checkpoint('goriya', run_id, network, root=tmp_path)
    assert cli.main(['--models-dir', str(tmp_path), 'promote', 'goriya', 'a']) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / 'goriya' / 'promoted')
    cli.main(['--models-dir', str(tmp_path), 'promote', 'goriya', 'b'])
    capsys.readouterr()
    cli.main(['--models-dir', str(tmp_path), 'list'])
    assert _listing(capsys)['goriya'] == ('b', 2)
    with pytest.raises(FileNotFoundError, match='No checkpoint for goriya/c'):
        cli.main(['--models-dir', str(tmp_path), 'promote', 'goriya', 'c'])
    assert registry.promoted_run('goriya', tmp_path) == 'b'


@pytest.mark.parametrize('argv', [[], ['no-such-command'], ['promote', 'goriya'], ['eval-chain', '--runs', 'x']])
def test_bad_arguments_exit_with_usage(argv, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2 and 'usage: triforce' in capsys.readouterr().err


def test_passthrough_commands_hand_their_arguments_to_the_module(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(['evaluate', '--help'])
    assert error.value.code == 0 and 'python -m triforce.evaluate' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(['run', '--no-such-flag'])
    assert 'python -m triforce.run' in capsys.readouterr().err


def test_eval_chain_rejects_an_unknown_route(tmp_path):
    with pytest.raises(RunnerError, match='no-such-route'):
        cli.main(['eval-chain', '--route', 'no-such-route', '--runs', '1', '--output', str(tmp_path / 'c.jsonl')])
    assert not (tmp_path / 'c.jsonl').exists()


def test_eval_chain_stopping_in_the_start_room_succeeds_without_acting(models_dir, tmp_path, capsys):
    output = tmp_path / 'nested' / 'chain.jsonl'
    assert cli.main(['--models-dir', str(models_dir), 'eval-chain', '--runs', '2', '--stop-at', '0:0x77',
                     '--output', str(output)]) == 0
    assert 'route milestone1: 2/2 = 100%' in capsys.readouterr().out
    runs = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(r['seed'], r['result'], r['steps']) for r in runs] == [(0, 'success', 0), (1, 'success', 0)]
    # A second evaluation appends to the same file.
    cli.main(['--models-dir', str(models_dir), 'eval-chain', '--runs', '1', '--stop-at', '0:0x77',
              '--first-seed', '5', '--output', str(output)])
    assert [json.loads(line)['seed'] for line in output.read_text().splitlines()] == [0, 1, 5]
