"""Tests for StepHistoryModel, StepHistoryWidget, and helper functions."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt, QModelIndex
from PySide6.QtWidgets import QApplication

from triforce_debugger.step_history import (
    StepHistory,
    StepEntry,
    StepHistoryModel,
    StepHistoryWidget,
    format_action,
    format_reward_value,
    reward_color,
    COLOR_POSITIVE,
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
)


# ── Ensure QApplication exists ────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ── Helpers ───────────────────────────────────────────────────────────

def _entry(step_number, action="MOVE_N", reward_val=0.0, **overrides):
    """Build a StepEntry with sensible defaults."""
    defaults = dict(
        step_number=step_number,
        action=action,
        reward=reward_val,
        observation=None,
        state=None,
        action_mask=None,
        action_probabilities=None,
        terminated=False,
        truncated=False,
        frame=None,
    )
    defaults.update(overrides)
    return StepEntry(**defaults)


@dataclass
class _FakeOutcome:
    name: str
    value: float


class _FakeReward:
    """Minimal StepRewards-like object for testing."""
    def __init__(self, total, outcomes=None):
        self._total = total
        self._outcomes = outcomes or []

    @property
    def value(self):
        return self._total

    def __iter__(self):
        return iter(self._outcomes)


def _entry_with_outcomes(step_number, total, outcomes):
    """Build a StepEntry whose reward is a FakeReward with outcomes."""
    reward = _FakeReward(total, [_FakeOutcome(n, v) for n, v in outcomes])
    return _entry(step_number, reward=reward)


# ── format_action ─────────────────────────────────────────────────────

class TestFormatAction:
    def test_string_action(self):
        assert format_action("SWORD_E") == "SWORD_E"

    def test_action_taken_like(self):
        act = MagicMock()
        act.kind.value = "MOVE"
        act.direction.name = "N"
        assert format_action(act) == "MOVE N"


# ── format_reward_value ───────────────────────────────────────────────

class TestFormatRewardValue:
    def test_positive_float(self):
        assert format_reward_value(0.05) == "+0.050"

    def test_negative_float(self):
        assert format_reward_value(-0.12) == "-0.120"

    def test_zero(self):
        assert format_reward_value(0.0) == "+0.000"

    def test_step_rewards_like(self):
        r = _FakeReward(0.5)
        assert format_reward_value(r) == "+0.500"


# ── reward_color ──────────────────────────────────────────────────────

class TestRewardColor:
    def test_positive(self):
        assert reward_color(0.1) == COLOR_POSITIVE

    def test_negative(self):
        assert reward_color(-0.1) == COLOR_NEGATIVE

    def test_zero(self):
        assert reward_color(0.0) == COLOR_NEUTRAL

    def test_step_rewards_like(self):
        assert reward_color(_FakeReward(0.5)) == COLOR_POSITIVE
        assert reward_color(_FakeReward(-0.5)) == COLOR_NEGATIVE


# ── StepHistoryModel ──────────────────────────────────────────────────

class TestStepHistoryModel:
    """Verify the Qt item model over the ring buffer."""

    def _make_model(self, entries=None, maxlen=100):
        history = StepHistory(maxlen=maxlen)
        for e in (entries or []):
            history.append(e)
        return StepHistoryModel(history), history

    def test_empty_model(self):
        model, _ = self._make_model()
        assert model.rowCount() == 0

    def test_row_count_matches_buffer(self):
        model, _ = self._make_model([_entry(i) for i in range(5)])
        assert model.rowCount() == 5

    def test_newest_first_ordering(self):
        model, _ = self._make_model([_entry(10), _entry(20), _entry(30)])
        # Display row 0 = newest = step 30
        idx = model.index(0, 0)
        assert model.data(idx) == "#30"
        # Display row 2 = oldest = step 10
        idx = model.index(2, 0)
        assert model.data(idx) == "#10"

    def test_step_columns(self):
        model, _ = self._make_model([_entry(7, action="SWORD_W", reward_val=0.25)])
        assert model.data(model.index(0, 0)) == "#7"
        assert model.data(model.index(0, 1)) == "SWORD_W"
        assert model.data(model.index(0, 2)) == "+0.250"

    def test_reward_color_positive(self):
        model, _ = self._make_model([_entry(1, reward_val=0.5)])
        color = model.data(model.index(0, 2), Qt.ItemDataRole.ForegroundRole)
        assert color == COLOR_POSITIVE

    def test_reward_color_negative(self):
        model, _ = self._make_model([_entry(1, reward_val=-0.5)])
        color = model.data(model.index(0, 2), Qt.ItemDataRole.ForegroundRole)
        assert color == COLOR_NEGATIVE

    def test_reward_color_zero(self):
        model, _ = self._make_model([_entry(1, reward_val=0.0)])
        color = model.data(model.index(0, 2), Qt.ItemDataRole.ForegroundRole)
        assert color == COLOR_NEUTRAL

    def test_column_count(self):
        model, _ = self._make_model()
        assert model.columnCount() == 3

    def test_header_data(self):
        model, _ = self._make_model()
        assert model.headerData(0, Qt.Orientation.Horizontal) == "Step"
        assert model.headerData(1, Qt.Orientation.Horizontal) == "Action"
        assert model.headerData(2, Qt.Orientation.Horizontal) == "Reward"

    # ── Child (reward breakdown) rows ─────────────────────────────────

    def test_child_row_count(self):
        entry = _entry_with_outcomes(1, 0.5, [("reward-a", 0.3), ("reward-b", 0.2)])
        model, _ = self._make_model([entry])
        parent_idx = model.index(0, 0)
        assert model.rowCount(parent_idx) == 2

    def test_child_row_data(self):
        entry = _entry_with_outcomes(1, 0.3, [("reward-hit", 0.5), ("penalty-hp", -0.2)])
        model, _ = self._make_model([entry])
        parent_idx = model.index(0, 0)
        # First child
        child0 = model.index(0, 0, parent_idx)
        assert model.data(child0) == "reward-hit"
        child0_val = model.index(0, 2, parent_idx)
        assert model.data(child0_val) == "+0.500"
        # Second child
        child1 = model.index(1, 0, parent_idx)
        assert model.data(child1) == "penalty-hp"
        child1_val = model.index(1, 2, parent_idx)
        assert model.data(child1_val) == "-0.200"

    def test_child_color(self):
        entry = _entry_with_outcomes(1, 0.0, [("reward-a", 0.1), ("penalty-b", -0.1)])
        model, _ = self._make_model([entry])
        parent_idx = model.index(0, 0)
        pos_color = model.data(model.index(0, 2, parent_idx), Qt.ItemDataRole.ForegroundRole)
        neg_color = model.data(model.index(1, 2, parent_idx), Qt.ItemDataRole.ForegroundRole)
        assert pos_color == COLOR_POSITIVE
        assert neg_color == COLOR_NEGATIVE

    def test_no_children_for_float_reward(self):
        model, _ = self._make_model([_entry(1, reward_val=0.5)])
        parent_idx = model.index(0, 0)
        assert model.rowCount(parent_idx) == 0

    def test_no_grandchildren(self):
        entry = _entry_with_outcomes(1, 0.5, [("reward-a", 0.5)])
        model, _ = self._make_model([entry])
        parent_idx = model.index(0, 0)
        child_idx = model.index(0, 0, parent_idx)
        assert model.rowCount(child_idx) == 0

    # ── Parent / index consistency ────────────────────────────────────

    def test_top_level_parent_is_invalid(self):
        model, _ = self._make_model([_entry(1)])
        idx = model.index(0, 0)
        assert not model.parent(idx).isValid()

    def test_child_parent_is_correct(self):
        entry = _entry_with_outcomes(1, 0.5, [("reward-a", 0.5)])
        model, _ = self._make_model([entry])
        parent_idx = model.index(0, 0)
        child_idx = model.index(0, 0, parent_idx)
        assert model.parent(child_idx).row() == 0

    def test_invalid_index_returns_none(self):
        model, _ = self._make_model()
        assert model.data(QModelIndex()) is None


# ── StepHistoryWidget ─────────────────────────────────────────────────

class TestStepHistoryWidget:
    """Verify the widget wrapping the model and tree view."""

    def test_creates_without_crash(self):
        widget = StepHistoryWidget()
        assert widget.objectName() == "step_history"

    def test_tree_view_exists(self):
        widget = StepHistoryWidget()
        assert widget.tree is not None
        assert widget.tree.model() is widget.model

    def test_append_step_updates_model(self):
        widget = StepHistoryWidget()
        widget.append_step(_entry(1))
        widget.append_step(_entry(2))
        assert widget.model.rowCount() == 2

    def test_clear_history(self):
        widget = StepHistoryWidget()
        for i in range(5):
            widget.append_step(_entry(i))
        widget.clear_history()
        assert widget.model.rowCount() == 0

    def test_step_selected_signal(self):
        widget = StepHistoryWidget()
        widget.append_step(_entry(10))
        widget.append_step(_entry(20))

        received = []
        widget.step_selected.connect(received.append)

        # Select display row 0 (newest = step 20, buffer index 1)
        idx = widget.model.index(0, 0)
        widget.tree.setCurrentIndex(idx)
        assert len(received) == 1
        assert received[0] == 1  # buffer index of step 20

    def test_append_when_full_does_not_crash(self):
        widget = StepHistoryWidget()
        # Use a tiny capacity so we hit the full case
        widget._history._buffer = __import__('collections').deque(maxlen=3)
        for i in range(5):
            widget.append_step(_entry(i))
        assert widget.model.rowCount() == 3

    def test_newest_at_top(self):
        widget = StepHistoryWidget()
        widget.append_step(_entry(100))
        widget.append_step(_entry(200))
        widget.append_step(_entry(300))
        # Row 0 should be step 300
        idx = widget.model.index(0, 0)
        assert widget.model.data(idx) == "#300"
