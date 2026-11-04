"""Runner, registry hand-off and chain-eval summary on the real game with untrained networks."""

import json

import pytest
import torch

from triforce import cli, registry
from triforce.actions import Action
from triforce.game_map import GameMap, load_routes
from triforce.runner import Route, RunResult, RoomRecord, Runner, RunnerError, chain_summary, parse_location, \
    write_jsonl
from triforce.zelda_enums import ActionKind, MapLocation


def test_milestone_route_definition():
    route = Route.load('milestone1')
    assert route.start_state == 'start'
    assert route.terminal == MapLocation(2, 0x0e, False)
    assert Route.load('milestone1', parse_location('1:0x73')).terminal == MapLocation(1, 0x73, False)
    with pytest.raises(RunnerError):
        Route.load('no-such-route')


@pytest.mark.parametrize('route', sorted(load_routes()))
def test_every_route_room_has_a_specialist(session, route):
    runner = Runner(session=session)
    for step in load_routes()[route]:
        location = MapLocation(step['level'], int(step['room'], 16), bool(step.get('in_cave')))
        assert runner.specialist_for(location) == step['specialist'], step


def test_untagged_room_stops_with_clear_error(session, models_dir):
    game_map = GameMap.load()
    game_map[MapLocation(0, 0x77, False)].specialist = None
    runner = Runner(session=session, models_dir=models_dir, game_map=game_map)
    result = runner.run(Route.load('milestone1'), seed=0)
    assert result.result == 'error'
    assert 'no specialist' in result.reason and '0x77' in result.reason


def test_wrong_exit_into_untagged_room_is_a_failure(session, models_dir):
    """Walking W out of 0x77 (the objective wants the sword cave) into untagged 0x76 ends the run as a gameplay
    failure; only a room the objective wanted can be a game.yaml error."""
    runner = Runner(session=session, models_dir=models_dir, max_steps=200)
    runner._load('ow-start').act = lambda obs, deterministic: Action.parse('MOVE W 16')  # pylint: disable=protected-access
    result = runner.run(Route.load('milestone1'), seed=0)
    assert result.result == 'failure'
    assert 'untagged' in result.reason and '0x76' in result.reason
    assert (result.rooms[-1].room, result.rooms[-1].outcome) == ('0x77', 'exit-wrong')


def test_missing_promoted_model_is_an_error(session, tmp_path):
    result = Runner(session=session, models_dir=tmp_path).run(Route.load('milestone1'), seed=0)
    assert result.result == 'error'
    assert 'ow-start' in result.reason
    assert result.rooms[0].outcome == 'error'


def test_run_records_rooms_and_ends_cleanly(session, models_dir, tmp_path):
    runner = Runner(session=session, models_dir=models_dir, max_steps=150)
    result = runner.run(Route.load('milestone1'), seed=3)
    assert result.result in ('failure', 'success')
    assert result.rooms[0].specialist == 'ow-start'
    assert result.steps == sum(r.steps for r in result.rooms)
    assert all(r.outcome != 'in-progress' for r in result.rooms)
    assert result.rooms[-1].outcome in ('stuck', 'death', 'milestone')

    path = tmp_path / 'runs.jsonl'
    write_jsonl(str(path), [result])
    room = json.loads(path.read_text().splitlines()[0])['rooms'][0]
    assert room['specialist'] == 'ow-start'
    assert room['hearts_dropped'] == room['hearts_taken'] + room['hearts_left'] + room['hearts_expired']


def test_same_seed_same_run(session, models_dir):
    runner = Runner(session=session, models_dir=models_dir, max_steps=60)
    first = runner.run(Route.load('milestone1'), seed=7)
    second = runner.run(Route.load('milestone1'), seed=7)
    assert first.to_dict() == second.to_dict()


def test_chain_summary_counts_failures_by_room():
    def run(result, room, outcome):
        r = RunResult('milestone1', 0, result=result)
        r.rooms.append(RoomRecord(0, 1, room, False, 'stalfos-keese-gel', outcome=outcome))
        return r
    summary = chain_summary([run('success', '0x0e', 'milestone'), run('failure', '0x44', 'death'),
                             run('failure', '0x44', 'death'), run('failure', '0x73', 'stuck')])
    assert summary['runs'] == 4 and summary['successes'] == 1 and summary['success_rate'] == 0.25
    assert summary['failures'] == {'1:0x44 stalfos-keese-gel death': 2, '1:0x73 stalfos-keese-gel stuck': 1}


def test_runner_no_room_progress_threshold_replaces_flat_budget(session, models_dir, monkeypatch):
    monkeypatch.setattr('triforce.runner.MIN_ROOM_STEPS', 2)
    runner = Runner(session=session, models_dir=models_dir, max_steps=20)
    specialist = runner._load('ow-start')
    specialist.config = specialist.config.model_copy(update={'no_progress_steps': 4, 'max_steps': 1})
    specialist.act = lambda obs, deterministic: Action.parse('WAIT 8')
    tails = []

    class Renderer:
        def show(self, session, specialist, action, change):
            tails.append(change.frames)

    runner.renderer = Renderer()
    result = runner.run(Route.load('milestone1'), seed=0)
    assert len(tails) == result.steps == 4
    assert result.result == 'failure' and result.rooms[-1].outcome == 'stuck'
    assert 'no-room-progress' in result.reason and '4 quiet actions' in result.reason
    assert result.rooms[-1].linger['tail_steps'] == 4
    assert result.rooms[-1].linger['tail_frames'] == sum(tails)
    assert chain_summary([result])['successes'] == 0


@pytest.mark.parametrize('enabled', [False, True])
def test_runner_no_room_progress_preserves_global_guard_and_disabled_budget(session, models_dir, monkeypatch,
                                                                          enabled):
    monkeypatch.setattr('triforce.runner.MIN_ROOM_STEPS', 3)
    runner = Runner(session=session, models_dir=models_dir, max_steps=5)
    specialist = runner._load('ow-start')
    specialist.config = specialist.config.model_copy(update={'no_progress_steps': 20 if enabled else None,
                                                            'max_steps': 1})
    specialist.act = lambda obs, deterministic: Action.parse('WAIT 8')
    result = runner.run(Route.load('milestone1'), seed=0)
    assert result.result == 'failure' and result.rooms[-1].outcome == 'stuck'
    assert result.steps == (5 if enabled else 3)
    assert result.reason == ('run exceeded 5 steps' if enabled else 'ow-start exceeded 3 steps')
    assert result.rooms[-1].linger['tail_steps'] == result.steps


def test_runner_room_exit_resets_visit_clock(session, models_dir):
    runner = Runner(session=session, models_dir=models_dir, max_steps=100)
    specialist = runner._load('ow-start')
    specialist.config = specialist.config.model_copy(update={'no_progress_steps': 20})

    def act(obs, deterministic):
        if session.state.in_cave:
            return Action.parse('MOVE S 16')
        return Action.parse('WAIT 8')

    specialist.act = act
    route = Route.load('milestone1', start_from='0_77cave_0')
    result = runner.run(route, seed=0)
    assert result.result == 'failure' and 'no-room-progress' in result.reason
    assert len(result.rooms) == 2
    old, new = result.rooms
    assert old.in_cave and old.linger['tail_steps'] == 0
    assert not new.in_cave and new.steps == new.linger['steps'] == new.linger['tail_steps'] == 20


def test_chain_cli_spawn_workers_preserve_supplied_specialist_config(models_dir, tmp_path):
    name = 'stalfos-keese-gel'
    network, _ = registry.load_model(models_dir / name / 'untrained')
    with torch.no_grad():
        network.kind_head.weight.zero_()
        network.kind_head.bias.fill_(-100)
        network.kind_head.bias[ActionKind.WAIT] = 100
    root = tmp_path / 'wait-model'
    registry.save_checkpoint(name, 'wait', network, root=root)
    registry.promote(name, 'wait', root=root)
    config = tmp_path / 'alternate.yaml'
    config.write_text("name: stalfos-keese-gel\nrooms: ['(1, 0x73)']\nstart_states: ['1_73s_0']\n"
                      "success: [exit-correct]\nno_progress_steps: 2\nmax_steps: 100\nitems: {bombs: off}\n")
    output = tmp_path / 'chain.jsonl'
    cli.main(['--models-dir', str(root), 'eval-chain', '--parallel', '2', '--runs', '2',
              '--from', '1_73s_0', '--deterministic', '--specialist-config', str(config),
              '--output', str(output)])
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record['steps'] for record in records] == [2, 2]
    assert all(record['result'] == 'failure' and 'no-room-progress' in record['reason'] for record in records)
    assert all(record['rooms'][-1]['linger']['tail_steps'] == 2 for record in records)
