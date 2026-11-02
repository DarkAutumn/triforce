"""Runner, registry hand-off and chain-eval summary on the real game with untrained networks."""

import json

import pytest
import yaml

from triforce.game_map import GameMap
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.observation import ObservationBuilder
from triforce.registry import promote, save_checkpoint
from triforce.runner import ROUTES_PATH, Route, RunResult, RoomRecord, Runner, RunnerError, chain_summary, \
    parse_location, write_jsonl
from triforce.specialist import list_specialists
from triforce.zelda_enums import MapLocation


@pytest.fixture(scope='module')
def models_dir(tmp_path_factory):
    """Every specialist promoted to a small untrained network."""
    root = tmp_path_factory.mktemp('models')
    config = ModelConfig(hidden_dim=32, entity_d_model=32, entity_ff_dim=32, grid_out_dim=32)
    for name in list_specialists():
        network = ZeldaPolicy(ObservationBuilder.observation_space(), config)
        save_checkpoint(name, 'untrained', network, root=root)
        promote(name, 'untrained', root=root)
    return root


def test_milestone_route_definition():
    route = Route.load('milestone1')
    assert route.start_state == 'start'
    assert route.terminal == MapLocation(2, 0x0e, False)
    assert Route.load('milestone1', parse_location('1:0x73')).terminal == MapLocation(1, 0x73, False)
    with pytest.raises(RunnerError):
        Route.load('no-such-route')


def test_every_route_room_has_a_specialist(session):
    with open(ROUTES_PATH, 'r', encoding='utf-8') as f:
        steps = yaml.safe_load(f)['milestone1']
    runner = Runner(session=session)
    for step in steps:
        location = MapLocation(step['level'], int(step['room'], 16), bool(step.get('in_cave')))
        assert runner.specialist_for(location) == step['specialist']


def test_untagged_room_stops_with_clear_error(session, models_dir):
    game_map = GameMap.load()
    game_map[MapLocation(0, 0x77, False)].specialist = None
    runner = Runner(session=session, models_dir=models_dir, game_map=game_map)
    result = runner.run(Route.load('milestone1'), seed=0)
    assert result.result == 'error'
    assert 'no specialist' in result.reason and '0x77' in result.reason


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
    assert json.loads(path.read_text().splitlines()[0])['rooms'][0]['specialist'] == 'ow-start'


def test_same_seed_same_run(session, models_dir):
    runner = Runner(session=session, models_dir=models_dir, max_steps=60)
    first = runner.run(Route.load('milestone1'), seed=7)
    second = runner.run(Route.load('milestone1'), seed=7)
    assert first.to_dict() == second.to_dict()


def test_chain_summary_counts_failures_by_room():
    def run(result, room, outcome):
        r = RunResult('milestone1', 0, result=result)
        r.rooms.append(RoomRecord(0, 1, room, False, 'd1-melee', outcome=outcome))
        return r
    summary = chain_summary([run('success', '0x0e', 'milestone'), run('failure', '0x44', 'death'),
                             run('failure', '0x44', 'death'), run('failure', '0x73', 'stuck')])
    assert summary['runs'] == 4 and summary['successes'] == 1 and summary['success_rate'] == 0.25
    assert summary['failures'] == {'1:0x44 d1-melee death': 2, '1:0x73 d1-melee stuck': 1}
