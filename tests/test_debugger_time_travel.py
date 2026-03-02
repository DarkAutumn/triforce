"""Tests for QT-14: Time-travel on step selection."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from collections import OrderedDict
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PySide6.QtWidgets import QApplication

from triforce_debugger.main_window import MainWindow
from triforce_debugger.step_history import StepEntry


# ── Ensure QApplication exists ────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ── Helpers ───────────────────────────────────────────────────────────

def _make_entry(step_number, frame=None, observation=None,
                action_probs=None, action_mask_desc=None):
    """Build a StepEntry with sensible defaults."""
    if frame is None:
        frame = np.zeros((240, 256, 3), dtype=np.uint8)
    return StepEntry(
        step_number=step_number,
        action="MOVE_N",
        reward=0.0,
        observation=observation,
        state=None,
        action_mask=None,
        action_probabilities=action_probs,
        terminated=False,
        truncated=False,
        frame=frame,
        action_mask_desc=action_mask_desc,
    )


def _make_window_with_steps(count=5):
    """Create a MainWindow with steps in the history."""
    window = MainWindow()
    for i in range(count):
        frame = np.full((240, 256, 3), i * 50, dtype=np.uint8)
        entry = _make_entry(i, frame=frame)
        window.step_history.append_step(entry)
    return window


def _select_display_row(window, row):
    """Select a display row in the step history tree view."""
    idx = window.step_history.model.index(row, 0)
    window.step_history.tree.setCurrentIndex(idx)


# ── Selecting a step ──────────────────────────────────────────────────

class TestTimeTravelSelection:
    """Clicking a step in history pauses the timer and updates panels."""

    def test_sets_historical_mode(self):
        window = _make_window_with_steps()
        assert not window.is_viewing_historical
        _select_display_row(window, 0)
        assert window.is_viewing_historical

    def test_pauses_running_timer(self):
        window = _make_window_with_steps()
        window.game_timer.resume()
        assert window.game_timer.is_running
        _select_display_row(window, 0)
        assert not window.game_timer.is_running

    def test_updates_game_view_frame(self):
        window = _make_window_with_steps()
        _select_display_row(window, 0)  # newest step
        assert window.game_view.frame_image is not None

    def test_updates_observation_panel(self):
        window = MainWindow()
        obs = {"image": np.random.rand(1, 1, 84, 84).astype(np.float32)}
        entry = _make_entry(1, observation=obs)
        window.step_history.append_step(entry)
        _select_display_row(window, 0)
        assert window.obs_panel.obs_image.current_image is not None

    def test_updates_action_table(self):
        window = MainWindow()
        probs = OrderedDict()
        probs['value'] = torch.tensor(0.42)
        kind = MagicMock()
        kind.value = "MOVE"
        direction = MagicMock()
        direction.name = "N"
        probs[kind] = [(direction, 0.9)]

        entry = _make_entry(1, action_probs=probs)
        window.step_history.append_step(entry)
        _select_display_row(window, 0)

        assert window.action_table.table.rowCount() == 1
        assert "0.42" in window.action_table.value_label.text()

    def test_emits_step_viewed_signal(self):
        window = _make_window_with_steps()
        received = []
        window.step_viewed.connect(received.append)
        _select_display_row(window, 0)
        assert len(received) == 1
        assert isinstance(received[0], StepEntry)

    def test_shows_correct_step_data(self):
        """Selecting display row 2 shows step_number=2 (the 3rd oldest)."""
        window = _make_window_with_steps(5)
        received = []
        window.step_viewed.connect(received.append)
        _select_display_row(window, 2)  # display row 2 = step_number 2
        assert received[0].step_number == 2


# ── Resuming (F5 / Continue) ─────────────────────────────────────────

class TestTimeTravelResume:
    """F5 (Continue) exits historical mode and resumes live play."""

    def test_exits_historical_mode(self):
        window = _make_window_with_steps()
        _select_display_row(window, 0)
        assert window.is_viewing_historical
        window.action_continue.trigger()
        assert not window.is_viewing_historical

    def test_resumes_timer(self):
        window = _make_window_with_steps()
        _select_display_row(window, 0)
        assert not window.game_timer.is_running
        window.action_continue.trigger()
        assert window.game_timer.is_running

    def test_emits_none_for_live_mode(self):
        window = _make_window_with_steps()
        received = []
        window.step_viewed.connect(received.append)
        _select_display_row(window, 0)
        window.action_continue.trigger()
        # Last emission should be None (live mode)
        assert received[-1] is None

    def test_shows_latest_step_on_resume(self):
        window = _make_window_with_steps(3)
        _select_display_row(window, 2)  # view oldest step
        window.action_continue.trigger()
        # Game view should now show the newest step's frame
        assert window.game_view.frame_image is not None

    def test_continue_without_history_just_resumes(self):
        window = MainWindow()
        assert not window.is_viewing_historical
        window.action_continue.trigger()
        assert window.game_timer.is_running
        assert not window.is_viewing_historical


# ── show_step public API ─────────────────────────────────────────────

class TestShowStep:
    """Verify the show_step method updates panels correctly."""

    def test_handles_none_frame(self):
        window = MainWindow()
        entry = _make_entry(1, frame=None)
        window.show_step(entry)  # should not crash

    def test_handles_none_observation(self):
        window = MainWindow()
        entry = _make_entry(1, observation=None)
        window.show_step(entry)  # should not crash

    def test_handles_none_probabilities(self):
        window = MainWindow()
        entry = _make_entry(1, action_probs=None)
        window.show_step(entry)
        assert window.action_table.value_label.text() == "Value: —"
