"""Round 32: a route's own runs play only its part (routes.yaml ``own_models_until``; d3first's: up to the magic
sword's step).  The engine names each step's policy (``_native.step_policy``: ``family@d3first`` before the cutoff,
``family`` from it on), ``registry.policy_run`` names the run of a key, and ``runner.Policies`` loads both runs of a
family for a chain.  Here: the resolution, ``eval-chain`` playing two runs of ow-ghini on d3first at the grave (a
scripted route run walks off 0x21 W, the scripted default leaves the sword's cave S), the overrides, the replays
naming both runs, ``promote --route`` and ``list``."""

import json
from pathlib import Path

import pytest

from triforce import _native, cli, registry
from triforce.runner import Policies, RunnerError

from .conftest import promote_untrained, require_state
from .networks import scripted

FAMILY = 'ow-ghini'
PART = f'{FAMILY}@d3first'


@pytest.fixture(name='two_runs', scope='module')
def fixture_two_runs(tmp_path_factory):
    """Every family untrained; ow-ghini's default run ``south`` leaves a room S, its d3first run ``west`` W."""
    root = promote_untrained(tmp_path_factory.mktemp('route-parts'))
    registry.save_checkpoint(FAMILY, 'south', scripted('MOVE S 16'), root=root)
    registry.promote(FAMILY, 'south', root=root)
    registry.save_checkpoint(FAMILY, 'west', scripted('MOVE W 16'), root=root)
    registry.promote(FAMILY, 'west', root=root, route='d3first')
    return root


def test_the_step_names_the_part_and_the_registry_its_run(two_runs):
    until = _native.own_models_until('d3first')
    assert _native.route_steps('d3first')[until]['waypoint'] == 'sword' and _native.own_models_until('d2first') is None
    assert [_native.step_policy('d3first', step, FAMILY) for step in (0, until - 1, until)] == [PART, PART, FAMILY]
    assert _native.step_policy('d2first', 0, FAMILY) == FAMILY
    assert registry.split_policy(PART) == (FAMILY, 'd3first') and registry.split_policy(FAMILY) == (FAMILY, None)
    # d3first: the route's run plays its part only, the default run the rest; d2first never sees the route's run.
    assert registry.policy_run(PART, two_runs, 'd3first') == 'west'
    assert registry.policy_run(FAMILY, two_runs, 'd3first') == 'south'
    assert registry.policy_run(FAMILY, two_runs, 'd2first') == registry.policy_run(FAMILY, two_runs) == 'south'
    assert registry.route_part_key(FAMILY, two_runs, 'd3first') == PART
    assert registry.route_part_key('darknut', two_runs, 'd3first') is None      # no run of its own: the default
    assert registry.route_part_key(FAMILY, two_runs, 'd2first') is None
    with pytest.raises(FileNotFoundError, match='promoted for route d3first'):
        registry.policy_run('darknut@d3first', two_runs, 'd3first')


def test_policies_load_both_runs_and_overrides_pick_the_part(two_runs):
    def keys(route, checkpoints=None):
        policies = Policies(two_runs, checkpoints=checkpoints, names=[FAMILY, 'darknut'], route=route)
        return {name: (Path(path).parent if path.endswith('.pt') else Path(path)).name
                for name, path in policies.checkpoints.items()}

    assert keys('d3first') == {FAMILY: 'south', PART: 'west', 'darknut': registry.promoted_run('darknut', two_runs)}
    assert keys('d2first') == {FAMILY: 'south', 'darknut': registry.promoted_run('darknut', two_runs)}
    west, south = (str(registry.run_dir(FAMILY, run, two_runs)) for run in ('west', 'south'))
    # A family's override plays both parts; FAMILY@ROUTE the route's part only (a family without a run of its own
    # gets one).
    assert {k: v for k, v in keys('d3first', {FAMILY: west}).items() if k != 'darknut'} == {FAMILY: 'west'}
    assert keys('d3first', {PART: south})[PART] == 'south'
    assert keys('d3first', {'darknut@d3first': west})['darknut@d3first'] == 'west'
    with pytest.raises(RunnerError, match='own_models_until'):
        keys('d2first', {f'{FAMILY}@d2first': west})
    with pytest.raises(RunnerError, match='own_models_until'):
        keys('d3first', {f'{FAMILY}@d2first': west})


def test_eval_chain_plays_the_route_run_before_the_grave_and_the_default_after(two_runs, tmp_path, capsys):
    """0x21 from S (step 205) is d3first's part: the route run plays it (walks off W, the run ends ``missed``); the
    sword's cave (step 206) is the default run's (leaves S without the sword).  Both chains load both runs; the room
    records, the provenance and the replays name the run that played."""
    plays = {}
    for start in ('d3first_0_21s_0', 'd3first_0_21cave_0'):
        output, folder = tmp_path / f'{start}.jsonl', tmp_path / start
        assert cli.main(['--models-dir', str(two_runs), 'eval-chain', '--no-log', '--route', 'd3first', '--from',
                         str(require_state(start)), '--runs', '1', '--act', 'argmax', '--output', str(output),
                         '--replays', str(folder)]) == 0
        run = json.loads(output.read_text())
        models = {m['name']: m for m in json.loads(cli.meta_path(output).read_text())['provenance']['models']}
        assert models[FAMILY]['checkpoint'] == f'{FAMILY}/south/ckpt.pt'
        assert models[PART]['checkpoint'] == f'{FAMILY}/west/ckpt.pt'
        (path,) = folder.iterdir()
        recording = _native.Replay(path.read_bytes())
        header = recording.meta['models']
        assert [m['checkpoint'] for m in header if m['name'] in (FAMILY, PART)] == [
            f'{FAMILY}/south/ckpt.pt', f'{FAMILY}/west/ckpt.pt']
        room = next(event for event in recording.events() if event['type'] == 'room')
        first = run['rooms'][0]
        plays[start] = (first['specialist'], first.get('policy'), first['step'], header[room['policy']]['name'])
    capsys.readouterr()
    until = _native.own_models_until('d3first')
    assert plays == {'d3first_0_21s_0': (FAMILY, PART, until - 1, PART),
                     'd3first_0_21cave_0': (FAMILY, None, until, FAMILY)}


def test_eval_chain_refuses_a_part_the_route_does_not_have(two_runs, tmp_path, capsys):
    west = registry.run_dir(FAMILY, 'west', two_runs)
    assert cli.main(['--models-dir', str(two_runs), 'eval-chain', '--no-log', '--runs', '1', '--checkpoint',
                     f'{FAMILY}@d3first={west}', '--output', str(tmp_path / 'c.jsonl')]) == 2
    assert 'own_models_until' in capsys.readouterr().err


def test_promote_route_and_list_show_where_the_routes_run_plays(tmp_path, capsys):
    for run_id in ('a', 'b'):
        registry.save_checkpoint(FAMILY, run_id, scripted('WAIT 8'), root=tmp_path)
    assert cli.main(['--models-dir', str(tmp_path), 'promote', FAMILY, 'a']) == 0
    assert cli.main(['--models-dir', str(tmp_path), 'promote', FAMILY, 'b', '--route', 'd3first']) == 0
    assert capsys.readouterr().out.splitlines()[-1] == str(tmp_path / FAMILY / 'promoted.d3first')
    assert cli.main(['--models-dir', str(tmp_path), 'list']) == 0
    (line,) = [line for line in capsys.readouterr().out.splitlines() if line.startswith(f'{FAMILY} ')]
    until = _native.own_models_until('d3first')
    assert line.split()[1:] == ['promoted=a', 'runs=2', 'd3first=b', '(steps', f'0-{until - 1},', 'before',
                                '0:0x21:cave)']
    with pytest.raises(ValueError, match='unknown route'):
        cli.main(['--models-dir', str(tmp_path), 'promote', FAMILY, 'b', '--route', 'd9first'])
