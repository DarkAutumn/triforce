"""scripts/experiments/r32c_models.py: the goal step it cuts (`--goal-step`) and its pick by bombs."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location('r32c_models', ROOT / 'scripts' / 'experiments' / 'r32c_models.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


models = _script()


def test_the_goal_is_the_next_step_by_default():
    step, goal = models.route_goal('4_71s')
    assert (step.index, goal.index) == (31, 32)


def test_goal_step_cuts_dungeon_4s_entrance_from_dungeon_3s_start():
    step, goal = models.route_goal('3_7cs', 31)
    assert step.index < 31
    assert (goal.index, goal.level, goal.room, goal.entry) == (31, 4, 0x71, 'S')
    assert goal.filename(0) == 'd3first_4_71s_0.state'


def test_a_goal_step_not_after_the_stem_is_refused():
    with pytest.raises(ValueError):
        models.route_goal('4_71s', 31)


def test_pick_by_bombs_takes_the_quantiles_first_seed_first_and_never_twice():
    arrivals = [('a', 4), ('b', 0), ('c', 8), ('d', 4), ('e', 5), ('f', 2), ('g', 4)]
    picks = models.pick_by_bombs(arrivals, [0.5, 0.1, 0.9])
    assert picks == [('a', 4), ('f', 2), ('e', 5)]
    assert models.pick_by_bombs(arrivals, [0.5, 0.5]) == [('a', 4), ('d', 4)]
    assert models.pick_by_bombs([], [0.5]) == [None]
