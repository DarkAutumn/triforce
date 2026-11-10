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


def test_eval_chain_rejects_an_unknown_route(tmp_path, capsys):
    assert cli.main(['eval-chain', '--route', 'no-such-route', '--runs', '1', '--output',
                     str(tmp_path / 'c.jsonl')]) == 2
    assert 'no-such-route' in capsys.readouterr().err
    assert not (tmp_path / 'c.jsonl').exists()


@pytest.mark.parametrize('args, message', [
    (['--from', 'nope_0'], "Unknown start state 'nope_0'"),
    (['--stop-at', '9:0x99'], "--stop-at 9:0x99 is not a room of route 'd2first' after the start step"),
    (['--from', '2_7ds_0', '--stop-at', '0:0x67'], "--stop-at 0:0x67 is not a room"),   # only before Dungeon 2
    (['--stop-at', '1:73'], "is not a room (LEVEL:0xROOM"),
    (['--checkpoint', 'models/goriya/x'], "is not SPECIALIST=PATH"),
    (['--checkpoint', 'gorya=models/goriya'], "unknown specialist(s) ['gorya']"),
])
def test_eval_chain_checks_its_inputs_before_anything_runs(tmp_path, capsys, args, message):
    output = tmp_path / 'c.jsonl'
    assert cli.main(['--models-dir', str(tmp_path / 'models'), 'eval-chain', '--runs', '2', '--no-log', *args,
                     '--output', str(output)]) == 2
    assert message in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_eval_chain_exits_non_zero_when_a_run_ends_in_an_error(tmp_path, capsys):
    output = tmp_path / 'chain.jsonl'
    assert cli.main(['--models-dir', str(tmp_path / 'models'), 'eval-chain', '--runs', '1', '--no-log',
                     '--output', str(output)]) == 1
    assert '1 of 1 runs ended in an error' in capsys.readouterr().err
    assert json.loads(output.read_text())['result'] == 'error'


def test_eval_chain_writes_provenance_and_appends_only_the_same_team(models_dir, tmp_path, capsys):
    output = tmp_path / 'nested' / 'chain.jsonl'
    chain = ['--models-dir', str(models_dir), 'eval-chain', '--stop-at', '0:0x77', '--output', str(output)]
    assert cli.main([*chain, '--runs', '2']) == 0
    assert 'route d2first: 2/2 = 100%' in capsys.readouterr().out
    runs = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(r['seed'], r['result'], r['steps']) for r in runs] == [(0, 'success', 0), (1, 'success', 0)]
    meta = json.loads(cli.meta_path(output).read_text())
    models = {model['name']: model for model in meta['provenance']['models']}
    assert sorted(models) == list_specialists() and meta['batches'] == [{'first_seed': 0, 'runs': 2}]
    goriya = registry.promoted_checkpoint('goriya', models_dir)
    assert models['goriya']['checkpoint'] == str(goriya)
    assert models['goriya']['checkpoint_sha1'] == registry.checkpoint_sha1(goriya)
    assert models['goriya']['act'] == 'sample' and meta['provenance']['stop_at'] == '0:0x77'
    # An existing output is never appended to implicitly.
    assert cli.main([*chain, '--runs', '1', '--first-seed', '5']) == 2
    assert 'exists; pass --append' in capsys.readouterr().err
    # Another act mode is another team: refused even with --append.
    assert cli.main([*chain, '--runs', '1', '--first-seed', '5', '--append', '--act', 'argmax']) == 2
    assert 'models differ' in capsys.readouterr().err
    assert cli.main([*chain, '--runs', '1', '--first-seed', '1', '--append']) == 2
    assert 'seeds [1] are already in' in capsys.readouterr().err
    assert cli.main([*chain, '--runs', '1', '--first-seed', '5', '--append']) == 0
    assert [json.loads(line)['seed'] for line in output.read_text().splitlines()] == [0, 1, 5]
    assert json.loads(cli.meta_path(output).read_text())['batches'][1] == {'first_seed': 5, 'runs': 1}
