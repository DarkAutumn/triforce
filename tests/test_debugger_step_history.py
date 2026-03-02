"""Tests for StepHistory ring buffer."""

import pytest
from triforce_debugger.step_history import StepHistory, StepEntry


def _make_entry(step_number, **overrides):
    """Helper to build a StepEntry with sensible defaults."""
    defaults = dict(
        step_number=step_number,
        action=f"action-{step_number}",
        reward=step_number * 0.1,
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


# -- basic operations -------------------------------------------------------

class TestStepHistoryBasics:
    """Add, clear, retrieve operations."""

    def test_empty_on_creation(self):
        hist = StepHistory()
        assert len(hist) == 0

    def test_append_and_length(self):
        hist = StepHistory()
        hist.append(_make_entry(1))
        hist.append(_make_entry(2))
        assert len(hist) == 2

    def test_get_by_index(self):
        hist = StepHistory()
        hist.append(_make_entry(10))
        hist.append(_make_entry(20))
        assert hist.get_by_index(0).step_number == 10
        assert hist.get_by_index(1).step_number == 20

    def test_getitem(self):
        hist = StepHistory()
        hist.append(_make_entry(5))
        assert hist[0].step_number == 5
        assert hist[-1].step_number == 5

    def test_clear(self):
        hist = StepHistory()
        for i in range(10):
            hist.append(_make_entry(i))
        hist.clear()
        assert len(hist) == 0

    def test_newest_and_oldest(self):
        hist = StepHistory()
        hist.append(_make_entry(1))
        hist.append(_make_entry(2))
        hist.append(_make_entry(3))
        assert hist.newest.step_number == 3
        assert hist.oldest.step_number == 1

    def test_newest_oldest_empty(self):
        hist = StepHistory()
        assert hist.newest is None
        assert hist.oldest is None


# -- capacity limits ---------------------------------------------------------

class TestStepHistoryCapacity:
    """Verify ring-buffer eviction behavior."""

    def test_evicts_oldest_at_capacity(self):
        hist = StepHistory(maxlen=5)
        for i in range(8):
            hist.append(_make_entry(i))
        # Only the last 5 should remain
        assert len(hist) == 5
        assert hist.oldest.step_number == 3
        assert hist.newest.step_number == 7

    def test_default_capacity(self):
        hist = StepHistory()
        assert hist._buffer.maxlen == StepHistory.MAX_STEPS  # pylint: disable=protected-access

    def test_custom_capacity(self):
        hist = StepHistory(maxlen=100)
        assert hist._buffer.maxlen == 100  # pylint: disable=protected-access

    def test_index_error_on_empty(self):
        hist = StepHistory()
        with pytest.raises(IndexError):
            hist.get_by_index(0)

    def test_index_error_out_of_range(self):
        hist = StepHistory(maxlen=5)
        hist.append(_make_entry(1))
        with pytest.raises(IndexError):
            hist.get_by_index(5)


# -- data integrity ----------------------------------------------------------

class TestStepEntryData:
    """Verify that stored data comes back unchanged."""

    def test_all_fields_preserved(self):
        entry = StepEntry(
            step_number=42,
            action="MOVE_N",
            reward=0.5,
            observation={"image": [1, 2, 3]},
            state={"link": "pos"},
            action_mask=[True, False],
            action_probabilities={"MOVE": 0.9},
            terminated=True,
            truncated=False,
            frame=b"\x00\x01\x02",
        )
        hist = StepHistory()
        hist.append(entry)
        got = hist[0]
        assert got.step_number == 42
        assert got.action == "MOVE_N"
        assert got.reward == 0.5
        assert got.observation == {"image": [1, 2, 3]}
        assert got.state == {"link": "pos"}
        assert got.action_mask == [True, False]
        assert got.action_probabilities == {"MOVE": 0.9}
        assert got.terminated is True
        assert got.truncated is False
        assert got.frame == b"\x00\x01\x02"
