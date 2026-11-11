"""Runner, registry hand-off and chain-eval summary on the native chain pool with untrained or scripted networks.

The chain's game rules (untagged rooms, step budgets, retreats, visit clocks) are tested natively in
``crates/env/tests/chain_explore.rs``; these tests cover the Python side: routes, results, missing models,
determinism, replay files and the ``eval-chain`` command line."""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

import triforce
from triforce import _native, provenance as provenance_module, registry, replay, run
from triforce.runner import Policies, Route, RoomRecord, Runner, RunnerError, RunResult, chain_summary, \
    parse_location, parse_stop, write_jsonl
from triforce.zelda_enums import ActionKind, MapLocation

from .conftest import REPO_STATES_DIR, require_state
from .networks import scripted


def test_route_definition():
    route = Route.load('d2first')
    assert route.start_state == 'start'
    assert route.terminal is None                   # the route ends on taking the D3 triforce
    assert Route.load('d2first', parse_location('1:0x73')).terminal == MapLocation(1, 0x73, False)
    assert Route.load('d2first', start_from='2_7ds_0', states_dir=REPO_STATES_DIR).start_state == '2_7ds_0'
    with pytest.raises(RunnerError, match='Unknown route'):
        Route.load('no-such-route')


def test_route_checks_the_start_state_and_the_stop_room(tmp_path):
    with pytest.raises(RunnerError, match=r"Unknown start state 'nope_0'.*states in"):
        Route.load('d2first', start_from='nope_0')
    (tmp_path / '1_73s_0.state').write_bytes(b'')
    assert Route.load('d2first', start_from='1_73s_0', states_dir=tmp_path).start_state == '1_73s_0'
    # The stop room must come after the start state's step: 2:0x7d is D2's entrance, 1:0x73 is on the way on.
    assert Route.load('d2first', parse_location('1:0x73'), '2_7ds_0',
                      states_dir=REPO_STATES_DIR).terminal == MapLocation(1, 0x73, False)
    with pytest.raises(RunnerError, match=r"--stop-at 2:0x7d is not a room of route 'd2first' after the start"):
        Route.load('d2first', parse_location('2:0x7d'), '2_7ds_0', states_dir=REPO_STATES_DIR)
    with pytest.raises(RunnerError, match='--stop-at 9:0x99'):
        Route.load('d2first', parse_location('9:0x99'))
    # 0x73 is entered from S first (step 46) and from W later (step 48): stopping at it again is after the start.
    assert Route.load('d2first', parse_location('1:0x73'), '1_73s_0',
                      states_dir=REPO_STATES_DIR).terminal == MapLocation(1, 0x73, False)


def test_checkpoint_overrides_must_name_a_specialist_and_a_checkpoint(models_dir, tmp_path):
    checkpoint = registry.promoted_checkpoint('ow-start', models_dir)
    with pytest.raises(RunnerError, match=r"unknown specialist\(s\) \['no-such'\]"):
        Policies(models_dir, checkpoints={'no-such': str(checkpoint)})
    with pytest.raises(RunnerError, match='no checkpoint file there'):
        Policies(models_dir, checkpoints={'ow-start': str(tmp_path / 'nope')})
    policies = Policies(models_dir, checkpoints={'ow-start': str(checkpoint.parent)})
    assert policies.checkpoints['ow-start'] == str(checkpoint.parent)


def test_eval_chain_summary_names_every_model_and_its_act_mode(models_dir, tmp_path):
    from triforce.cli import eval_chain     # pylint: disable=import-outside-toplevel
    root = tmp_path / 'models'
    shutil.copytree(models_dir, root)
    registry.promote('goriya', 'untrained', root=root, act='argmax')
    other = registry.save_checkpoint('zol', 'other', registry.load_model(
        registry.promoted_checkpoint('zol', root))[0], root=root)
    _, summary = eval_chain('d2first', 1, models_dir=root, overrides={'zol': str(other)}, stop_at='0:0x77')
    provenance = summary['provenance']
    models = {model['name']: model for model in provenance['models']}
    assert sorted(models) == sorted(Policies(root).names)
    assert models['goriya']['act'] == 'argmax' and models['ow-start']['act'] == 'sample'
    assert models['zol']['checkpoint'] == str(other)
    assert models['zol']['checkpoint_sha1'] == registry.checkpoint_sha1(other)
    assert (provenance['route'], provenance['start_from'], provenance['stop_at']) == ('d2first', None, '0:0x77')
    assert provenance['first_seed'] == 0 and provenance['runs'] == 1
    assert provenance['code'] == provenance_module.code_version()
    assert provenance['data'] == provenance_module.data_hash() and provenance['version'] == triforce.__version__
    assert models['zol']['version'] == models['ow-start']['version'] == triforce.__version__


def test_run_result_round_trips_the_native_dict(tmp_path):
    room = RoomRecord(0, 1, '0x44', False, 'stalfos-keese-gel', steps=3, outcome='death',
                      linger={'max_steps': 2, 'tail_steps': 1})
    result = RunResult('d2first', 4, 'failure', 'died', 3, 90, retreats=1, rooms=[room])
    assert RunResult.from_dict(result.to_dict()) == result
    path = tmp_path / 'runs.jsonl'
    write_jsonl(str(path), [result, result])
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0]) == result.to_dict()


def test_chain_summary_counts_failures_by_room():
    def run(result, room, outcome):
        r = RunResult('d2first', 0, result=result)
        r.rooms.append(RoomRecord(0, 1, room, False, 'stalfos-keese-gel', outcome=outcome))
        return r
    summary = chain_summary([run('success', '0x0e', 'milestone'), run('failure', '0x44', 'death'),
                             run('failure', '0x44', 'death'), run('failure', '0x73', 'stuck')])
    assert summary['runs'] == 4 and summary['successes'] == 1 and summary['success_rate'] == 0.25
    assert summary['failures'] == {'1:0x44 stalfos-keese-gel death': 2, '1:0x73 stalfos-keese-gel stuck': 1}


def test_chain_summary_totals_cheated_items_and_the_runs_that_used_any():
    def run(*rooms):
        r = RunResult('d2first', 0, result='success')
        r.rooms.extend(RoomRecord(i, 1, '0x33', False, 'stalfos-keese-gel', cheated_bombs=bombs, cheated_keys=keys)
                       for i, (bombs, keys) in enumerate(rooms))
        return r
    results = [run((1, 0), (1, 1)), run((0, 0)), run((0, 2))]
    assert results[0].cheated == {'bombs': 2, 'keys': 1, 'rupees': 0}
    summary = chain_summary(results)
    assert (summary['cheated_bombs'], summary['runs_with_cheated_bombs']) == (2, 1)
    assert (summary['cheated_keys'], summary['runs_with_cheated_keys']) == (3, 2)
    assert (summary['cheated_rupees'], summary['runs_with_cheated_rupees']) == (0, 0)
    assert 'bombs_cheated' not in summary


def test_chain_summary_counts_keys_carried_into_the_milestone_separately():
    def run(result, keys):
        r = RunResult('d2first', 0, result=result)
        r.rooms.append(RoomRecord(0, 2, '0x0e', False, '-', outcome='milestone', keys_in=keys))
        return r
    summary = chain_summary([run('success', 3), run('success', 3), run('success', 1), run('success', None),
                             run('failure', 3)])
    assert summary['successes'] == 4
    assert summary['arrival_keys'] == {'1': 1, '3': 2, 'unknown': 1}   # failures don't count; old records: unknown


def test_policies_load_promoted_models_and_name_the_missing(models_dir, tmp_path):
    policies = Policies(models_dir)
    assert not policies.missing and policies.names == sorted(policies.names)
    assert [config.name for config in policies.configs] == policies.names
    empty = Policies(tmp_path)
    assert not empty.names and 'ow-start' in empty.missing['ow-start']


def test_route_checks_a_stop_step_a_power_on_start_and_a_checkpoint():
    assert parse_stop('step:24') == (None, 24) and parse_stop(None) == (None, None)
    assert parse_stop('1:0x73') == (MapLocation(1, 0x73, False), None)
    route = Route.load('d2first', stop_step=24)
    assert (route.stop_step, route.terminal, route.start_state) == (24, None, 'start')
    assert Route.load('d2first', power_on={}).power_on == {'name': 'LINK', 'delay': None, 'max_delay': 32766}
    assert Route.load('d2first', power_on={'name': 'bob 2', 'delay': 9}).power_on == {
        'name': 'BOB 2', 'delay': 9, 'max_delay': 32766}
    checkpoint = {'state': b'a state', 'step': 24, 'granted': [1, 0, 0], 'kind': 'end', 'frame': 7}
    route = Route.load('d2first', checkpoint=checkpoint, stop_step=25)
    assert route.checkpoint == {'state': b'a state', 'step': 24, 'granted': [1, 0, 0], 'randomize_rng': False}
    from_3c = {'start_from': '0_3cc_0', 'states_dir': REPO_STATES_DIR}
    for kwargs, message in [
            ({'stop_step': 0}, "--stop-at step:0 is not a step of route 'd2first' after the start step 0"),
            ({'stop_step': 10_000}, 'is not a step of route'),
            ({'stop_step': 24, **from_3c}, 'after the start step 24; steps: 25..='),
            ({'stop_step': 24, 'checkpoint': checkpoint}, 'after the start step 24'),
            ({'stop_step': 30, 'stop_at': parse_location('1:0x73')}, 'a stop room and a stop step are exclusive'),
            ({'power_on': {}, **from_3c}, r'a start state \(--from\) and a power-on start are exclusive'),
            ({'power_on': {}, 'checkpoint': checkpoint}, 'a power-on start and a checkpoint are exclusive'),
            ({'checkpoint': {**checkpoint, 'step': 10_000}}, "the checkpoint's step 10000 is not a step"),
            ({'power_on': {'name': 'ZELDA'}}, 'second quest'),
            ({'power_on': {'delay': 32767}}, r'the power-on delay 32767 must be a frame count in 0\.\.=32766'),
            ({'power_on': {'max_delay': -1}}, 'the power-on max delay -1 must be'),
            ({'power_on': {'frames': 3}}, r"unknown power-on options \['frames'\]")]:
        with pytest.raises(RunnerError, match=message):
            Route.load('d2first', **kwargs)


def test_power_on_runs_start_the_game_by_inputs_and_draw_their_delay_from_the_seed(states_dir, models_dir):
    runner = Runner(models_dir=models_dir, max_steps=12, parallel=2, states_dir=states_dir)
    drawn = Route.load('d2first', power_on={'max_delay': 40})
    twice = [[r.to_dict() for r in runner.run_many(drawn, [1, 2, 3])] for _ in range(2)]
    assert twice[0] == twice[1]
    for result in twice[0]:
        room = result['rooms'][0]
        assert (room['level'], room['room'], room['specialist'], room['health_in']) == (0, '0x77', 'ow-start', 3.0)
        assert result['steps'] == 12 or result['result'] == 'failure'
    provenance = runner.provenance(drawn)
    assert provenance['power_on'] == {'name': 'LINK', 'delay': None, 'max_delay': 40}
    assert (provenance['start_from'], provenance['checkpoint'], provenance['stop_step']) == (None, None, None)


def _play_recorded(runner, route, seed):
    """(the run's result, its recording) of one recorded chain run."""
    record = replay.record_options(replay.meta('chain', [], route.name))
    events = []
    with runner.pool(route, [seed], record=record) as pool:
        pool.play()
        idle = False
        while not idle:
            idle = pool.pool.wait_idle(timeout=0.1)
            events += pool.poll_events()
    (_, result), = pool.results(events)
    recording, = [_native.Replay(e['data']) for e in events if e['type'] == 'replay']
    return result, recording


def test_a_run_ends_on_its_stop_step_and_another_continues_from_its_end(states_dir, models_dir):
    """0x3c (step 24), where Link comes out of Dungeon 2: straight south is 0x4c (step 25)."""
    runner = Runner(models_dir=models_dir, max_steps=40, states_dir=states_dir,
                    networks={'ow-octorok': scripted('MOVE S 16')})
    route = Route.load('d2first', start_from=str(require_state('0_3cc_0')), stop_step=25)
    result, recording = _play_recorded(runner, route, 0)
    assert (result.result, result.reason) == ('success', 'reached route step 25 (level 0, room 0x4c)')
    assert [(r.room, r.outcome) for r in result.rooms] == [('0x3c', 'exit-correct'), ('0x4c', 'milestone')]
    assert recording.ending == 'win'
    end = recording.states()[-1]
    assert (end['kind'], end['step'], end['frame']) == ('end', 25, recording.frames)
    health = result.rooms[-1].health_in

    onward = Route.load('d2first', checkpoint=end, stop_step=26)
    provenance = runner.provenance(onward)
    assert provenance['checkpoint']['step'] == 25 and len(provenance['checkpoint']['state_sha1']) == 40
    runner.max_steps = 6
    result, continued = _play_recorded(runner, onward, 3)
    room = result.rooms[0]
    assert (room.level, room.room, room.health_in) == (0, '0x4c', health)
    assert continued.source == {'kind': 'checkpoint'}
    assert continued.states()[0]['state'] == end['state'] and continued.events()[0] == {'frame': 0, 'type': 'control'}
    joined = _native.Replay(_native.replay_stitch(recording.data, continued.data))
    assert joined.frames == recording.frames + continued.frames
    assert _native.replay_verify(joined.data)['frames'] == joined.frames



def test_missing_promoted_model_is_an_error(states_dir, tmp_path):
    result = Runner(models_dir=tmp_path, states_dir=states_dir).run(Route.load('d2first'), seed=0)
    assert result.result == 'error' and result.steps == 0
    assert 'ow-start' in result.reason


def test_runs_record_rooms_and_end_cleanly(states_dir, models_dir, tmp_path):
    runner = Runner(models_dir=models_dir, max_steps=40, parallel=2, states_dir=states_dir)
    results = runner.run_many(Route.load('d2first'), [5, 3, 4])
    assert [r.seed for r in results] == [3, 4, 5]
    for result in results:
        assert result.result in ('failure', 'success')
        assert result.rooms[0].specialist == 'ow-start'
        assert result.steps == sum(r.steps for r in result.rooms) <= 40
        assert all(r.outcome != 'in-progress' for r in result.rooms)
        assert result.rooms[-1].outcome in ('stuck', 'death', 'milestone', 'exit-wrong')
        room = result.rooms[0]
        assert room.hearts_dropped == room.hearts_taken + room.hearts_left + room.hearts_expired

    path = tmp_path / 'runs.jsonl'
    write_jsonl(str(path), results)
    assert json.loads(path.read_text().splitlines()[0])['rooms'][0]['specialist'] == 'ow-start'


def test_same_seed_same_run(states_dir, models_dir):
    runner = Runner(models_dir=models_dir, max_steps=30, parallel=2, threads=1, states_dir=states_dir)
    first = runner.run_many(Route.load('d2first'), [7, 8])
    second = runner.run_many(Route.load('d2first'), [7, 8])
    assert [r.to_dict() for r in first] == [r.to_dict() for r in second]


def test_runs_act_the_way_each_model_is_registered(states_dir, models_dir, tmp_path):
    """A run's ``act`` file (promote --act) makes that model act by argmax; the other models keep sampling, and
    an explicit ``deterministic`` overrides every model."""
    root = tmp_path / 'models'
    shutil.copytree(models_dir, root)
    registry.promote('ow-start', 'untrained', root=root, act='argmax')
    policies = Policies(root)
    assert [name for name, argmax in zip(policies.names, policies.deterministic) if argmax] == ['ow-start']

    def start_rooms(deterministic):
        """Each run's first room visit (the start room, ow-start's)."""
        runner = Runner(models_dir=root, deterministic=deterministic, max_steps=30, parallel=2, states_dir=states_dir)
        return [r.to_dict()['rooms'][0] for r in runner.run_many(Route.load('d2first'), [3, 4])]

    registered, argmax, sampled = start_rooms(None), start_rooms(True), start_rooms(False)
    assert all(room['specialist'] == 'ow-start' for room in registered)
    assert registered == argmax and registered != sampled


def test_sampled_runs_do_not_depend_on_envs_or_threads(states_dir, models_dir):
    """Sampled actions are keyed to (run seed, decision), so a seed's run is the same on any env or thread."""
    seeds = [3, 4, 5, 6]
    runs = []
    for parallel, threads in ((1, 1), (4, 1), (4, 4), (3, 2)):
        runner = Runner(models_dir=models_dir, max_steps=40, parallel=parallel, threads=threads,
                        states_dir=states_dir)
        runs.append([r.to_dict() for r in runner.run_many(Route.load('d2first'), seeds)])
    assert all(run == runs[0] for run in runs[1:])
    assert len({json.dumps(run['rooms']) for run in runs[0]}) > 1, "different seeds play different runs"


def test_supplied_config_no_progress_ends_runs(states_dir, models_dir):
    config = _native.SpecialistConfig.load('ow-start').replace(no_progress_steps=4)
    runner = Runner(models_dir=models_dir, networks={'ow-start': scripted('WAIT 8')},
                    specialist_configs={'ow-start': config}, parallel=3, states_dir=states_dir)
    results = runner.run_many(Route.load('d2first'), [0, 1, 2])
    for result in results:
        assert result.result == 'failure' and result.rooms[-1].outcome == 'stuck' and result.steps == 4
        assert 'no-room-progress' in result.reason
        assert result.rooms[-1].linger['tail_steps'] == 4
    assert chain_summary(results)['successes'] == 0


def test_a_chain_run_replays_its_actions_to_the_same_probabilities(states_dir, models_dir):
    """A chain run's recorded actions (``EnvPool.play(record='actions')``) fed back to the same models
    (``replay=``) give the same run and, decision by decision, the same probabilities and values, bit for bit.  A
    replay that outlives its actions raises."""
    runner = Runner(models_dir=models_dir, max_steps=40, parallel=3, threads=2, states_dir=states_dir)
    route, policies = Route.load('d2first'), runner.policies.policies

    def play(seeds, **kwargs):
        with runner.pool(route, seeds) as chain:
            chain.pool.play(policies, deterministic=chain.deterministic, **kwargs)
            chain.pool.wait_idle()
            plays = {int(p['job']): p for p in chain.pool.poll_plays()}
            runs = {r.seed: r.to_dict() for _, r in chain.results(chain.poll_events())}
        return plays, runs

    plays, runs = play([3, 4, 5], record='actions')
    assert sorted(plays) == sorted(runs) == [3, 4, 5]
    assert all(len(plays[seed]['actions']) == runs[seed]['steps'] for seed in runs)
    again, replayed = play([4], record='observations', replay={4: plays[4]['actions']})
    assert replayed == {4: runs[4]}
    for key in ('actions', 'log_prob', 'entropy', 'value', 'policy'):
        assert again[4][key].tobytes() == plays[4][key].tobytes(), key
    assert len(again[4]['obs']['room_id']) == runs[4]['steps']
    with pytest.raises(ValueError):
        play([3], record='actions', replay={3: plays[3]['actions'][:-1]})


def test_chain_cli_uses_supplied_specialist_config(states_dir, models_dir, tmp_path):
    name = 'ow-start'
    network, _ = registry.load_model(models_dir / name / 'untrained')
    with torch.no_grad():
        network.kind_head.weight.zero_()
        network.kind_head.bias.fill_(-100)
        network.kind_head.bias[ActionKind.WAIT] = 100
    root = tmp_path / 'wait-model'
    registry.save_checkpoint(name, 'wait', network, root=root)
    registry.promote(name, 'wait', root=root)
    config = tmp_path / 'alternate.yaml'
    config.write_text("name: ow-start\nrooms: ['(0, 0x77)', '(0, 0x77, cave)', '(0, 0x39)']\n"
                      "start_states: ['start']\nsuccess: [exit-correct]\nno_progress_steps: 2\nmax_steps: 100\n")
    output = tmp_path / 'chain.jsonl'
    subprocess.run([sys.executable, '-m', 'triforce.cli', '--models-dir', str(root), 'eval-chain',
                    '--parallel', '2', '--runs', '2', '--act', 'argmax', '--specialist-config', str(config),
                    '--output', str(output)],
                   check=True, capture_output=True, text=True,
                   env={**os.environ, 'TRIFORCE_STATES_DIR': str(states_dir)})
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record['seed'] for record in records] == [0, 1]
    assert [record['steps'] for record in records] == [2, 2]
    assert all(record['result'] == 'failure' and 'no-room-progress' in record['reason'] for record in records)
    assert all(record['rooms'][-1]['linger']['tail_steps'] == 2 for record in records)


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='no ffmpeg')
def test_video_replays_every_frame_of_the_run(states_dir, models_dir, tmp_path):
    runner = Runner(models_dir=models_dir, max_steps=6, states_dir=states_dir)
    result, data = run.record_replay(runner, Route.load('d2first'), seed=2)
    recorded = _native.Replay(data)
    assert result.steps == 6 and recorded.source == {'kind': 'state', 'name': 'start'}
    assert recorded.frames >= result.frames and recorded.meta['seed'] == 2
    assert recorded.meta['kind'] == 'chain' and recorded.meta['route'] == 'd2first'
    assert _native.replay_verify(data) == {'frames': recorded.frames, 'edits': len(recorded.edits()), 'states': 2}
    path = tmp_path / 'run.mp4'
    assert run.write_video(data, str(path)) == recorded.frames     # the start state is in the replay
    probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_entries',
                            'stream=width,height,nb_read_frames', '-of', 'csv=p=0', str(path)],
                           check=True, capture_output=True, text=True)
    assert probe.stdout.strip() == f"256,240,{recorded.frames}"
