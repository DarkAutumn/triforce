"""The ``triforce`` command (``triforce.cli.main``) in-process: list, promote, argument errors and passthrough."""

import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

import triforce
from triforce import _native, cli, jobs, registry
from triforce.model import ModelConfig
from triforce.runner import Policies, RunnerError, chain_steps
from triforce.specialist import list_specialists

from .conftest import REPO_STATES_DIR, SEGMENT_STATE, SEGMENT_STOP_STEP, require_state, seeded_network

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
    (['--from', str(REPO_STATES_DIR / '2_7ds_0.state'), '--stop-at', '0:0x5e'], "--stop-at 0:0x5e is not a room"),
    (['--from', str(REPO_STATES_DIR / '2_7ds_0.state'), '--stop-at', '0:0x59'],
     "--stop-at 0:0x59 is not a room of the clips way of route 'd2first'"),
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
    assert cli.main(['--models-dir', str(tmp_path / 'models'), 'eval-chain', '--route', 'd2first', '--runs', '2', '--no-log', *args,
                     '--output', str(output)]) == 2
    assert message in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_eval_chain_answers_the_wrapper_dry_call_and_creates_nothing(tmp_path):
    """``triforce job run`` asks the chain its work first (``$TRIFORCE_JOB_ESTIMATE``, no job id yet): its route
    steps, written as ``{kind, work, name}`` once its arguments are checked, before any log folder, output or model."""
    estimate = tmp_path / 'estimate.json'
    environ = {**{k: v for k, v in os.environ.items() if k != jobs.JOB_ID_ENV}, jobs.JOB_ESTIMATE_ENV: str(estimate)}
    out = subprocess.run([sys.executable, '-m', 'triforce.cli', '--models-dir', str(tmp_path / 'models'), 'eval-chain',
                          '--route', 'd2first', '--stop-at', '0:0x77', '--runs', '3', '--output',
                          str(tmp_path / 'c.jsonl'), '--logs-dir', str(tmp_path / 'logs')],
                         env=environ, capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    steps = chain_steps('d2first', 3, None, '0:0x77')
    assert steps > 0 and json.loads(estimate.read_text()) == {
        'kind': 'chain', 'work': {'chain_steps': steps}, 'name': 'eval-chain-d2first'}
    assert sorted(p.name for p in tmp_path.iterdir()) == ['estimate.json']


def test_eval_chain_exits_non_zero_when_a_run_ends_in_an_error(tmp_path, capsys):
    output = tmp_path / 'chain.jsonl'
    assert cli.main(['--models-dir', str(tmp_path / 'models'), 'eval-chain', '--runs', '1', '--no-log',
                     '--output', str(output)]) == 1
    assert '1 of 1 runs ended in an error' in capsys.readouterr().err
    assert json.loads(output.read_text())['result'] == 'error'


def test_eval_chain_writes_provenance_and_appends_only_the_same_team(models_dir, tmp_path, capsys):
    output = tmp_path / 'nested' / 'chain.jsonl'
    chain = ['--models-dir', str(models_dir), 'eval-chain', '--route', 'd2first', '--stop-at', '0:0x77', '--output',
             str(output)]
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
    assert models['goriya']['act'] == registry.DEFAULT_ACT and meta['provenance']['stop_at'] == '0:0x77'
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


def test_eval_chain_in_an_act_mode_with_the_switch_off(scripted_models, states_dir, tmp_path):
    """Rounds 22 and 23 merged: ``eval-chain --act temperature:0.75 --clips off``: every model acts by the rule,
    every run and room record plays with the switch off and counts its decisions, the provenance and the
    recordings name both, and the recordings replay bit for bit."""
    output, folder = tmp_path / 'chain.jsonl', tmp_path / 'replays'
    assert cli.main(['--models-dir', str(scripted_models), 'eval-chain', '--route', 'd2first',
                     '--from', str(require_state(SEGMENT_STATE)),
                     '--stop-at', f'step:{SEGMENT_STOP_STEP}', '--act', 'temperature:0.75', '--clips', 'off',
                     '--runs', '2', '--parallel', '2', '--no-log', '--output', str(output),
                     '--replays', str(folder)]) == 0
    runs = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(r['seed'], r['result'], r['clips']) for r in runs] == [(0, 'success', False), (1, 'success', False)]
    for run in runs:
        assert all(room['clips_allowed'] is False and room['clips_flips'] == [] for room in run['rooms'])
        played = [room for room in run['rooms'] if room['outcome'] != 'milestone']
        assert played and all(room['decisions']['n'] == room['steps'] > 0 for room in played)
    provenance = json.loads(cli.meta_path(output).read_text())['provenance']
    assert {m['act'] for m in provenance['models']} == {'temperature:0.75'} and provenance['clips'] is False
    files = sorted(folder.iterdir())
    assert len(files) == 2
    for path in files:
        recording = _native.Replay(path.read_bytes())
        assert {m['act'] for m in recording.meta['models']} == {'temperature:0.75'}
        assert _native.replay_verify(path.read_bytes(), states_dir=str(states_dir))['frames'] == recording.frames


def test_eval_chain_events_into_a_folder_with_or_without_a_plan(models_dir, tmp_path, monkeypatch):
    """``--events FILE``: a folder without a plan gets one (kind chain, the stream as its events, finished
    completed) and the chain's counter is the job's 'runs' (the 250-run test's: it was named 'events' for its
    stream); a driver's folder with a plan keeps it untouched and the counter is named for the stream (the batch's
    entry).  Each ``chain_progress`` carries the runs so far (``runner.ChainTally``)."""
    counters = []
    monkeypatch.setattr(jobs, 'progress', lambda done, total, unit, label=None, force=False: counters.append(
        (label or unit, done, total)))
    chain = ['--models-dir', str(models_dir), 'eval-chain', '--stop-at', '0:0x77', '--runs', '1']
    alone = tmp_path / 'alone' / 'chain.jsonl'
    assert cli.main([*chain, '--output', str(tmp_path / 'a.jsonl'), '--events', str(alone)]) == 0
    plan = json.loads((alone.parent / 'training.json').read_text())
    assert plan['kind'] == 'chain' and plan['events'] == 'chain.jsonl' and plan['status'] == 'completed'
    events = [json.loads(line) for line in alone.read_text().splitlines()]
    assert [event['event'] for event in events] == ['chain_start', 'chain_progress', 'chain']
    progress = events[1]
    assert progress['done'] == 1 and progress['deaths'] + progress['stucks'] + progress['errors'] + progress[
        'successes'] == 1
    assert progress['route_steps'] == len(_native.route_steps('d3first')) and progress['triforces'] == []
    assert counters == [('runs', 1, 1)]
    driven = tmp_path / 'driven'
    driven.mkdir()
    (driven / 'training.json').write_text('{"kind": "eval-batch"}')
    assert cli.main([*chain, '--output', str(tmp_path / 'b.jsonl'), '--events', str(driven / 'x.jsonl')]) == 0
    assert json.loads((driven / 'training.json').read_text()) == {'kind': 'eval-batch'}
    assert (driven / 'x.jsonl').read_text().count('"chain"') >= 1
    assert counters[-1] == ('x', 1, 1)


def test_chain_progress_carries_the_runs_so_far():
    """``chain_progress`` (what the training GUI draws of a chain, the 250-run test's): wins, deaths, stuck runs
    and errors, the runs that took each triforce in the route's order (a full route's win takes Dungeon 9's, the
    game's end), the furthest route step and its room; at most once every ``seconds`` and always on the last run."""
    from triforce.runner import RoomRecord, Route, RunResult
    steps = _native.route_steps('d3first')

    def run(seed, played, result='failure', reason='', last=''):
        rooms = [RoomRecord(index=i, level=steps[s]['level'], room=f"0x{steps[s]['room']:02x}",
                            in_cave=steps[s]['in_cave'], specialist='', step=s,
                            outcome=last if last and i == len(played) - 1 else 'exit-correct')
                 for i, s in enumerate(played)]
        return RunResult(route='d3first', seed=seed, result=result, reason=reason, rooms=rooms)

    class Events:
        def __init__(self):
            self.emitted = []

        def emit(self, event, **fields):
            self.emitted.append((event, fields))

    events = Events()
    progress = cli.chain_progress_events(events, 4, route=Route.load('d3first'), seconds=3600.0)
    progress(run(0, list(range(60)), reason='died in (level 0, room 0x37)', last='death'))
    progress(run(1, list(range(31)), reason='bounce-limit: 0:0x45', last='stuck'))    # within the hour: not sent
    progress(run(2, [], result='error', reason='no model'))
    assert [fields['done'] for _, fields in events.emitted] == [1]
    progress(run(3, list(range(len(steps))), result='success'))                       # the last run: sent
    (_, first), (_, last) = events.emitted
    assert {key: first[key] for key in ('deaths', 'stucks', 'errors', 'successes')} == {
        'deaths': 1, 'stucks': 0, 'errors': 0, 'successes': 0}
    assert last == {'done': 4, 'runs': 4, 'successes': 1, 'deaths': 1, 'stucks': 1, 'errors': 1,
                    'triforces': [[3, 3], [4, 2], [1, 1], [2, 1], [5, 1], [7, 1], [6, 1], [8, 1], [9, 1]],
                    'furthest_step': len(steps), 'furthest_room': 'the end', 'route_steps': len(steps)}
    assert first['furthest_step'] == 59 and first['furthest_room'] == f"0:0x{steps[59]['room']:02x}" + (
        ':cave' if steps[59]['in_cave'] else '')
    # a chain to a stop: only the triforces before it, a win is not the game's end
    segment = cli.chain_progress_events(events, 1, route=Route.load('d3first', stop_step=24))
    segment(run(4, list(range(24)), result='success'))
    assert events.emitted[-1][1]['triforces'] == [[3, 1]] and events.emitted[-1][1]['furthest_step'] == 23


def test_eval_chain_keeps_the_runs_that_finished_before_a_failure(models_dir, tmp_path, monkeypatch):
    def failing_progress(_events, _runs, _label=None, _route=None):
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


def test_a_model_plays_by_its_schemas_not_its_version(models_dir, tmp_path, capsys):
    """AGENTS.md "Versioning": the run's version is provenance; the checkpoint's schema numbers decide."""
    root = tmp_path / 'models'
    shutil.copytree(models_dir, root)
    goriya = registry.promoted_checkpoint('goriya', root)
    registry.write_model_version(goriya, '1.2.9.9')                 # another version, the same schemas: plays
    chain = ['--models-dir', str(root), 'eval-chain', '--runs', '1', '--stop-at', '0:0x77', '--no-log']
    assert cli.main([*chain, '--output', str(tmp_path / 'a.jsonl')]) == 0
    provenance = json.loads(cli.meta_path(tmp_path / 'a.jsonl').read_text())['provenance']
    versions = {model['name']: model['version'] for model in provenance['models']}
    assert versions['goriya'] == '1.2.9.9' and versions['ow-start'] == provenance['version'] == triforce.__version__
    checkpoint = torch.load(goriya, map_location='cpu', weights_only=True)
    torch.save({**checkpoint, 'action_schema_version': _native.ACTION_SCHEMA_VERSION - 1}, goriya)
    registry.write_model_version(goriya, triforce.__version__)     # this code's version, another action schema
    with pytest.raises(registry.SchemaMismatchError, match="goriya.*can't load this model: action schema"):
        Policies(root)
    assert cli.main([*chain, '--output', str(tmp_path / 'b.jsonl')]) == 2
    error = capsys.readouterr().err
    assert (f"action schema {_native.ACTION_SCHEMA_VERSION - 1}, this code's is {_native.ACTION_SCHEMA_VERSION}"
            f" (trained by triforce {triforce.__version__}" in error)
    assert not (tmp_path / 'b.jsonl').exists()
