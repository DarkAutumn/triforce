"""The ``triforce`` command (``triforce.cli.main``) in-process: list, promote, argument errors and passthrough."""

import json
import shutil

import pytest

import triforce
from triforce import cli, registry
from triforce.model import ModelConfig
from triforce.runner import Policies, RunnerError
from triforce.specialist import list_specialists

from .conftest import REPO_STATES_DIR, seeded_network

SMALL = ModelConfig(hidden_dim=32, entity_d_model=32, entity_ff_dim=32, grid_out_dim=32)


@pytest.fixture(scope='module')
def network():
    return seeded_network(1, SMALL)


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
def test_bad_arguments_exit_with_usage(argv):
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2


def test_passthrough_commands_hand_their_arguments_to_the_module(monkeypatch):
    with pytest.raises(SystemExit) as error:
        cli.main(['evaluate', '--help'])
    assert error.value.code == 0
    with pytest.raises(SystemExit) as error:
        cli.main(['run', '--no-such-flag'])
    assert error.value.code == 2
    seen = []
    monkeypatch.setattr('triforce.evaluate.main', lambda argv: seen.append(argv) or 0)
    assert cli.main(['evaluate', '--episodes-per-state', '3']) == 0
    assert seen == [['--episodes-per-state', '3']]


def test_passthrough_after_the_global_models_dir(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as error:
        cli.main(['--models-dir', str(tmp_path), 'evaluate', '--help'])
    assert error.value.code == 0
    with pytest.raises(SystemExit) as error:
        cli.main(['--models-dir', 'x', 'train'])            # the trainer's own usage error, not a KeyError
    assert error.value.code == 2
    seen = []
    monkeypatch.setattr('triforce.run.main', lambda argv: seen.append(argv) or 0)
    assert cli.main([f'--models-dir={tmp_path}', 'run', '--seed', '3']) == 0
    assert seen == [['--models-dir', str(tmp_path), '--seed', '3']]
    assert cli.main(['--models-dir', 'x', 'play']) == 2


def test_eval_chain_rejects_an_unknown_route(tmp_path, capsys):
    assert cli.main(['eval-chain', '--route', 'no-such-route', '--runs', '1', '--output',
                     str(tmp_path / 'c.jsonl')]) == 2
    assert 'no-such-route' in capsys.readouterr().err
    assert not (tmp_path / 'c.jsonl').exists()


@pytest.mark.parametrize('args, message', [
    (['--from', 'nope_0'], "Unknown start state 'nope_0'"),
    (['--stop-at', '9:0x99'], "--stop-at 9:0x99 is not a room of route 'd2first' after the start step"),
    (['--from', str(REPO_STATES_DIR / '2_7ds_0.state'), '--stop-at', '0:0x59'], "--stop-at 0:0x59 is not a room"),
    (['--stop-at', '1:73'], "is not a room (LEVEL:0xROOM"),
    (['--stop-at', 'step:0'], "--stop-at step:0 is not a step of route 'd2first' after the start step 0"),
    (['--stop-at', 'step:x'], "is not a route step (step:N"),
    (['--stop-at', 'step:24', '--from', str(REPO_STATES_DIR / '0_3cc_0.state')], "after the start step 24"),
    (['--name', 'BOB'], "--name needs --power-on"),
    (['--name', 'BOB', '--delay', '3'], "--name and --delay need --power-on"),
    (['--power-on', '--name', 'ZELDA'], "power on: the name ZELDA starts the second quest"),
    (['--power-on', '--delay', '32767'], "the power-on delay 32767 must be a frame count in 0..=32766"),
    (['--power-on', '--delay', 'soon'], "--delay 'soon' must be a frame count in 0..=32766, or 'random'"),
    (['--power-on', '--from', str(REPO_STATES_DIR / '2_7ds_0.state')], "are exclusive: each replaces the game start"),
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
    assert sorted(models) == list_specialists()
    assert meta['batches'] == [{'first_seed': 0, 'runs': 2, 'finished': True}]
    goriya = registry.promoted_checkpoint('goriya', models_dir)
    assert models['goriya']['checkpoint'] == f'goriya/{goriya.parent.name}/ckpt.pt'    # no local directory
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
    assert json.loads(cli.meta_path(output).read_text())['batches'][1] == {'first_seed': 5, 'runs': 1,
                                                                            'finished': True}


def test_eval_chain_events_into_a_folder_with_or_without_a_plan(models_dir, tmp_path):
    """``--events FILE``: a folder without a plan gets one (kind chain, the stream as its events, finished
    completed); a driver's folder with a plan keeps it untouched."""
    chain = ['--models-dir', str(models_dir), 'eval-chain', '--stop-at', '0:0x77', '--runs', '1']
    alone = tmp_path / 'alone' / 'chain.jsonl'
    assert cli.main([*chain, '--output', str(tmp_path / 'a.jsonl'), '--events', str(alone)]) == 0
    plan = json.loads((alone.parent / 'training.json').read_text())
    assert plan['kind'] == 'chain' and plan['events'] == 'chain.jsonl' and plan['status'] == 'completed'
    assert [json.loads(line)['event'] for line in alone.read_text().splitlines()] == [
        'chain_start', 'chain_progress', 'chain']
    driven = tmp_path / 'driven'
    driven.mkdir()
    (driven / 'training.json').write_text('{"kind": "eval-batch"}')
    assert cli.main([*chain, '--output', str(tmp_path / 'b.jsonl'), '--events', str(driven / 'x.jsonl')]) == 0
    assert json.loads((driven / 'training.json').read_text()) == {'kind': 'eval-batch'}
    assert (driven / 'x.jsonl').read_text().count('"chain"') >= 1


def test_eval_chain_keeps_the_runs_that_finished_before_a_failure(models_dir, tmp_path, monkeypatch):
    def failing_progress(_events, _runs):
        calls = []

        def on_result(_result):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("stopped after two runs")
        return on_result
    monkeypatch.setattr(cli, 'chain_progress_events', failing_progress)
    output = tmp_path / 'chain.jsonl'
    with pytest.raises(RuntimeError, match='stopped after two runs'):
        cli.main(['--models-dir', str(models_dir), 'eval-chain', '--runs', '3', '--stop-at', '0:0x77',
                  '--output', str(output)])
    assert len(output.read_text().splitlines()) == 2
    assert json.loads(cli.meta_path(output).read_text())['batches'] == [{'first_seed': 0, 'runs': 3,
                                                                          'finished': False}]


def test_an_incompatible_model_plays_only_with_the_override(models_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(registry.ALLOW_INCOMPATIBLE_ENV, '0')       # restored afterwards whatever the CLI sets
    root = tmp_path / 'models'
    shutil.copytree(models_dir, root)
    registry.write_model_version(registry.promoted_checkpoint('goriya', root), '1.2.9.9')
    with pytest.raises(registry.IncompatibleModelError, match='goriya.*model version 1.2.9.9'):
        Policies(root)
    chain = ['--models-dir', str(root), 'eval-chain', '--runs', '1', '--stop-at', '0:0x77', '--no-log']
    assert cli.main([*chain, '--output', str(tmp_path / 'a.jsonl')]) == 2
    assert 'pass --allow-incompatible-models' in capsys.readouterr().err
    assert not (tmp_path / 'a.jsonl').exists()
    with pytest.warns(UserWarning, match='model version 1.2.9.9 is not compatible'):
        assert cli.main([*chain, '--allow-incompatible-models', '--output', str(tmp_path / 'b.jsonl')]) == 0
    provenance = json.loads(cli.meta_path(tmp_path / 'b.jsonl').read_text())['provenance']
    versions = {model['name']: model['version'] for model in provenance['models']}
    assert versions['goriya'] == '1.2.9.9' and versions['ow-start'] == provenance['version'] == triforce.__version__
    monkeypatch.setenv(registry.ALLOW_INCOMPATIBLE_ENV, '0')
    with pytest.warns(UserWarning, match='model version 1.2.9.9'):       # the flag before the command works too
        assert cli.main(['--allow-incompatible-models', *chain, '--output', str(tmp_path / 'c.jsonl')]) == 0


def test_stamp_writes_missing_versions_and_refuses_to_change_one(tmp_path, network, capsys):
    for name, run_id in (('ow-start', 'a'), ('ow-start', 'b'), ('goriya', 'x')):
        checkpoint = registry.save_checkpoint(name, run_id, network, root=tmp_path)
        (checkpoint.parent / registry.VERSION_NAME).unlink()
    registry.promote('ow-start', 'b', root=tmp_path)
    registry.promote('goriya', 'x', root=tmp_path)
    sha1 = registry.checkpoint_sha1(tmp_path / 'ow-start' / 'b')
    stamp = ['--models-dir', str(tmp_path), 'stamp']
    assert cli.main(stamp) == 0                                 # every promoted run, this code's version
    assert capsys.readouterr().out.splitlines() == [f'goriya/x: {triforce.__version__}',
                                                    f'ow-start/b: {triforce.__version__}']
    assert registry.model_version(tmp_path / 'ow-start' / 'b') == triforce.__version__
    assert registry.model_version(tmp_path / 'ow-start' / 'a') is None
    assert registry.checkpoint_sha1(tmp_path / 'ow-start' / 'b') == sha1        # the checkpoint is not rewritten
    assert cli.main([*stamp, 'ow-start', 'a', '--version', '1.2.0.4']) == 0
    assert registry.model_version(tmp_path / 'ow-start' / 'a') == '1.2.0.4'
    assert cli.main([*stamp, 'ow-start', 'a']) == 1             # stamped otherwise: refused, both named
    error = capsys.readouterr().err
    assert 'ow-start/a is stamped 1.2.0.4' in error and triforce.__version__ in error
    assert registry.model_version(tmp_path / 'ow-start' / 'a') == '1.2.0.4'
    assert cli.main([*stamp, 'ow-start', 'a', '--force']) == 0
    assert registry.model_version(tmp_path / 'ow-start' / 'a') == triforce.__version__
    assert cli.main([*stamp, 'ow-start', 'nope']) == 1
    assert cli.main([*stamp, 'zol']) == 1                       # nothing promoted
    assert cli.main([*stamp, 'goriya', '--version', '1.3']) == 2
