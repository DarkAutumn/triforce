"""Runner, registry hand-off and chain-eval summary on the native chain pool with untrained or scripted networks.

The chain's game rules (untagged rooms, step budgets, fairy healing, visit clocks) are tested natively in
``crates/env/tests/chain_explore.rs``; these tests cover the Python side: routes, results, batched per-specialist
inference, missing models, determinism and the ``eval-chain`` command line."""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from triforce import _native, registry, run
from triforce.actions import Action
from triforce.runner import Policies, Route, RoomRecord, Runner, RunnerError, RunResult, chain_summary, \
    parse_location, write_jsonl
from triforce.zelda_enums import ActionKind, MapLocation

NUM_KINDS, NUM_DIRECTIONS, NUM_DURATIONS = len(_native.ACTION_KINDS), len(_native.DIRECTIONS), len(_native.DURATIONS)


class ScriptedPolicy:
    """A network stand-in: ``action`` where the row's mask allows it, else ``fallback``; records batch sizes."""

    def __init__(self, action: str, fallback: str = 'WAIT 8'):
        self.action = Action.parse(action).to_indices()
        self.fallback = Action.parse(fallback).to_indices()
        self.batches = []

    @staticmethod
    def allowed(mask, indices) -> bool:
        """Whether the flat mask allows ``[kind, direction, duration]``."""
        kind, direction, duration = (int(i) for i in indices)
        dirs = mask[NUM_KINDS + kind * NUM_DIRECTIONS:NUM_KINDS + (kind + 1) * NUM_DIRECTIONS]
        base = NUM_KINDS + NUM_KINDS * NUM_DIRECTIONS
        durs = mask[base + kind * NUM_DURATIONS:base + (kind + 1) * NUM_DURATIONS]
        return bool(mask[kind]) and (not dirs.any() or bool(dirs[direction])) \
            and (not durs.any() or bool(durs[duration]))

    def get_action(self, obs, deterministic=False):  # pylint: disable=unused-argument
        masks = np.asarray(obs['action_mask'])
        self.batches.append(len(masks))
        return torch.as_tensor(np.stack([self.action if self.allowed(m, self.action) else self.fallback
                                         for m in masks]))


def test_milestone_route_definition():
    route = Route.load('milestone1')
    assert route.start_state == 'start'
    assert route.terminal == MapLocation(2, 0x0e, False)
    assert Route.load('milestone1', parse_location('1:0x73')).terminal == MapLocation(1, 0x73, False)
    assert Route.load('milestone1', start_from='0_37c_2').start_state == '0_37c_2'
    with pytest.raises(RunnerError, match='Unknown route'):
        Route.load('no-such-route')


@pytest.mark.parametrize('route', ['milestone1', 'milestone1-fairy'])
def test_every_route_room_has_a_specialist(route):
    for step in _native.route_steps(route):
        location = (step['level'], step['room'], step['in_cave'])
        assert _native.specialist_for(location) == step['specialist'], step


def test_run_result_round_trips_the_native_dict(tmp_path):
    room = RoomRecord(0, 1, '0x44', False, 'stalfos-keese-gel', steps=3, outcome='death',
                      linger={'max_steps': 2, 'tail_steps': 1})
    result = RunResult('milestone1', 4, 'failure', 'died', 3, 90, [room])
    assert RunResult.from_dict(result.to_dict()) == result
    path = tmp_path / 'runs.jsonl'
    write_jsonl(str(path), [result, result])
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0]) == result.to_dict()


def test_chain_summary_counts_failures_by_room():
    def run(result, room, outcome):
        r = RunResult('milestone1', 0, result=result)
        r.rooms.append(RoomRecord(0, 1, room, False, 'stalfos-keese-gel', outcome=outcome))
        return r
    summary = chain_summary([run('success', '0x0e', 'milestone'), run('failure', '0x44', 'death'),
                             run('failure', '0x44', 'death'), run('failure', '0x73', 'stuck')])
    assert summary['runs'] == 4 and summary['successes'] == 1 and summary['success_rate'] == 0.25
    assert summary['failures'] == {'1:0x44 stalfos-keese-gel death': 2, '1:0x73 stalfos-keese-gel stuck': 1}


def test_policies_load_promoted_models_and_name_the_missing(models_dir, tmp_path):
    policies = Policies(models_dir)
    assert not policies.missing and policies.names == sorted(policies.names)
    assert [config.name for config in policies.configs] == policies.names
    empty = Policies(tmp_path)
    assert not empty.names and 'ow-start' in empty.missing['ow-start']


def test_missing_promoted_model_is_an_error(states_dir, tmp_path):
    result = Runner(models_dir=tmp_path, states_dir=states_dir).run(Route.load('milestone1'), seed=0)
    assert result.result == 'error' and result.steps == 0
    assert 'ow-start' in result.reason


def test_runs_record_rooms_and_end_cleanly(states_dir, models_dir, tmp_path):
    runner = Runner(models_dir=models_dir, max_steps=40, parallel=2, states_dir=states_dir)
    results = runner.run_many(Route.load('milestone1'), [5, 3, 4])
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
    first = runner.run_many(Route.load('milestone1'), [7, 8])
    second = runner.run_many(Route.load('milestone1'), [7, 8])
    assert [r.to_dict() for r in first] == [r.to_dict() for r in second]


def test_sampled_runs_do_not_depend_on_envs_or_threads(states_dir, models_dir):
    """Sampled actions are keyed to (run seed, decision), so a seed's run is the same on any env or thread."""
    seeds = [3, 4, 5, 6]
    runs = []
    for parallel, threads in ((1, 1), (4, 1), (4, 4), (3, 2)):
        runner = Runner(models_dir=models_dir, max_steps=40, parallel=parallel, threads=threads,
                        states_dir=states_dir)
        runs.append([r.to_dict() for r in runner.run_many(Route.load('milestone1'), seeds)])
    assert all(run == runs[0] for run in runs[1:])
    assert len({json.dumps(run['rooms']) for run in runs[0]}) > 1, "different seeds play different runs"


def test_wrong_exit_into_untagged_room_is_a_failure(states_dir, models_dir):
    """Walking W out of 0x77 (the objective wants the sword cave) into untagged 0x76 ends the run as a gameplay
    failure."""
    west = ScriptedPolicy('MOVE W 16')
    runner = Runner(models_dir=models_dir, max_steps=200, networks={'ow-start': west}, states_dir=states_dir)
    result = runner.run(Route.load('milestone1'), seed=0)
    assert result.result == 'failure'
    assert 'untagged' in result.reason and '0x76' in result.reason
    assert (result.rooms[-1].room, result.rooms[-1].outcome) == ('0x77', 'exit-wrong')
    assert west.batches and set(west.batches) == {1}


def test_specialist_rows_are_batched_and_no_progress_ends_runs(states_dir, models_dir):
    """Every ready env of one specialist goes into one batch; a supplied config's no_progress_steps ends the run."""
    wait = ScriptedPolicy('WAIT 8')
    config = _native.SpecialistConfig.load('ow-start').replace(no_progress_steps=4)
    runner = Runner(models_dir=models_dir, networks={'ow-start': wait}, specialist_configs={'ow-start': config},
                    parallel=3, states_dir=states_dir)
    results = runner.run_many(Route.load('milestone1'), [0, 1, 2])
    assert sum(wait.batches) == sum(result.steps for result in results) and max(wait.batches) <= 3
    for result in results:
        assert result.result == 'failure' and result.rooms[-1].outcome == 'stuck' and result.steps == 4
        assert 'no-room-progress' in result.reason and '4 quiet actions' in result.reason
        assert result.rooms[-1].linger['tail_steps'] == 4
    assert chain_summary(results)['successes'] == 0


def test_run_cap_ends_runs(states_dir, models_dir):
    wait = ScriptedPolicy('WAIT 8')
    config = _native.SpecialistConfig.load('ow-start').replace(no_progress_steps=20)
    runner = Runner(models_dir=models_dir, max_steps=5, networks={'ow-start': wait},
                    specialist_configs={'ow-start': config}, states_dir=states_dir)
    result = runner.run(Route.load('milestone1'), seed=0)
    assert result.result == 'failure' and result.rooms[-1].outcome == 'stuck'
    assert result.steps == 5 and result.reason == 'run exceeded 5 steps'


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
                    '--parallel', '2', '--runs', '2', '--deterministic', '--specialist-config', str(config),
                    '--output', str(output)],
                   check=True, capture_output=True, text=True,
                   env={**os.environ, 'TRIFORCE_STATES_DIR': str(states_dir)})
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record['seed'] for record in records] == [0, 1]
    assert [record['steps'] for record in records] == [2, 2]
    assert all(record['result'] == 'failure' and 'no-room-progress' in record['reason'] for record in records)
    assert all(record['rooms'][-1]['linger']['tail_steps'] == 2 for record in records)


def test_video_needs_ffmpeg(monkeypatch):
    monkeypatch.setattr(run.shutil, 'which', lambda _: None)
    with pytest.raises(RuntimeError, match='ffmpeg not found on PATH'):
        run.ffmpeg_command('/tmp/never.mp4')


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='no ffmpeg')
def test_video_replays_every_frame_of_the_run(states_dir, models_dir, tmp_path):
    runner = Runner(models_dir=models_dir, max_steps=6, states_dir=states_dir)
    result, movie = run.record_movie(runner, Route.load('milestone1'), seed=2)
    assert result.steps == 6 and movie['origin'] == 'start'
    assert len(movie['buttons']) >= result.frames
    path = tmp_path / 'run.mp4'
    assert run.write_video(movie, str(path), states_dir) == len(movie['buttons'])
    probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_entries',
                            'stream=width,height,nb_read_frames', '-of', 'csv=p=0', str(path)],
                           check=True, capture_output=True, text=True)
    assert probe.stdout.strip() == f"256,240,{len(movie['buttons'])}"
