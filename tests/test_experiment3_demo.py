# pylint: disable=all
"""Experiment 3 demo trace tests."""

import pytest

from diagnose import parse_demo_trace
import torch

from scripts.behavior_clone import compute_demo_accuracy
from triforce.action_space import ActionKind
from triforce.zelda_enums import Direction

EXPECTED_DIRECTIONS = [
    Direction.E, Direction.E, Direction.S, Direction.S,
    Direction.E, Direction.E, Direction.E, Direction.E, Direction.E, Direction.E, Direction.E,
    Direction.E, Direction.E, Direction.E, Direction.E, Direction.E, Direction.E, Direction.E,
    Direction.S, Direction.S, Direction.S,
    Direction.E, Direction.E, Direction.W, Direction.W,
    Direction.N, Direction.N, Direction.N, Direction.N, Direction.N, Direction.N,
    Direction.N, Direction.N, Direction.N, Direction.N, Direction.N,
    Direction.W, Direction.W, Direction.W,
    Direction.N, Direction.N, Direction.N, Direction.N, Direction.N,
]


def test_parse_reverse_demo_trace(tmp_path):
    trace = tmp_path / "trace.txt"
    trace.write_text("""
#3\tMOVE S\t\t+0.000
#2\tMOVE E\t\t+0.000
#1\tNone\t\t+0.000
""", encoding="utf-8")

    actions = parse_demo_trace(str(trace))

    assert actions == [(ActionKind.MOVE, Direction.E), (ActionKind.MOVE, Direction.S)]


def test_parse_normalized_demo_trace_takes_precedence(tmp_path):
    trace = tmp_path / "trace.txt"
    trace.write_text("""
## Normalized forward trace
MOVE N
MOVE E
## Original reverse trace
#2\tMOVE S\t\t+0.000
#1\tNone\t\t+0.000
""", encoding="utf-8")

    actions = parse_demo_trace(str(trace))

    assert actions == [(ActionKind.MOVE, Direction.N), (ActionKind.MOVE, Direction.E)]


def test_parse_full_demo_trace():
    actions = parse_demo_trace("docs/experiments/demos/wallmaster-north-exit.txt")

    assert len(actions) == 44
    assert all(action == ActionKind.MOVE for action, _ in actions)
    assert [direction for _, direction in actions] == EXPECTED_DIRECTIONS


def test_parse_empty_demo_trace_raises(tmp_path):
    trace = tmp_path / "trace.txt"
    trace.write_text("#1\tNone\t\t+0.000\n", encoding="utf-8")

    with pytest.raises(ValueError):
        parse_demo_trace(str(trace))


def test_compute_demo_accuracy():
    pred = torch.tensor([[0, 3], [0, 0], [0, 1]])
    target = torch.tensor([[0, 3], [0, 2], [0, 1]])

    assert compute_demo_accuracy(pred, target) == pytest.approx(2 / 3)
