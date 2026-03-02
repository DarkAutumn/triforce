"""Tests for QT-05: Game timer and step loop — QTimer-based game loop."""

import sys

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication


def get_app():
    """Get or create a QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _make_timer():
    """Create a GameTimer for testing."""
    from triforce_debugger.game_timer import GameTimer  # pylint: disable=import-outside-toplevel
    _app = get_app()
    return GameTimer()


# ── Initial state ─────────────────────────────────────────────


def test_initial_state_paused():
    """GameTimer starts paused."""
    timer = _make_timer()
    assert not timer.is_running
    assert not timer.is_uncapped


def test_default_interval():
    """Default interval is capped (16ms)."""
    from triforce_debugger.game_timer import GameTimer  # pylint: disable=import-outside-toplevel
    assert GameTimer.CAPPED_INTERVAL_MS == 16
    assert GameTimer.UNCAPPED_INTERVAL_MS == 0


# ── Resume / pause ────────────────────────────────────────────


def test_resume_starts_timer():
    """Resume starts the timer and sets running state."""
    timer = _make_timer()
    timer.resume()
    assert timer.is_running
    assert timer.interval == 16
    timer.stop()


def test_pause_stops_timer():
    """Pause stops a running timer."""
    timer = _make_timer()
    timer.resume()
    assert timer.is_running
    timer.pause()
    assert not timer.is_running
    timer.stop()


def test_resume_idempotent():
    """Calling resume when already running is a no-op."""
    timer = _make_timer()
    timer.resume()
    timer.resume()  # should not error or double-start
    assert timer.is_running
    timer.stop()


def test_pause_idempotent():
    """Calling pause when already paused is a no-op."""
    timer = _make_timer()
    timer.pause()  # already paused, should not error
    assert not timer.is_running


# ── Single step ───────────────────────────────────────────────


def test_single_step_emits_signal():
    """Single step emits step_requested exactly once."""
    timer = _make_timer()
    steps = []
    timer.step_requested.connect(lambda: steps.append(1))
    timer.single_step()
    assert len(steps) == 1
    assert not timer.is_running


def test_single_step_pauses_running_timer():
    """Single step pauses the timer if it was running."""
    timer = _make_timer()
    timer.resume()
    assert timer.is_running
    timer.single_step()
    assert not timer.is_running
    timer.stop()


# ── Uncapped mode ─────────────────────────────────────────────


def test_set_uncapped():
    """Setting uncapped mode updates the property."""
    timer = _make_timer()
    assert not timer.is_uncapped
    timer.set_uncapped(True)
    assert timer.is_uncapped
    timer.set_uncapped(False)
    assert not timer.is_uncapped


def test_uncapped_changes_interval_when_running():
    """Changing uncapped mode while running adjusts the interval immediately."""
    timer = _make_timer()
    timer.resume()
    assert timer.interval == 16

    timer.set_uncapped(True)
    assert timer.interval == 0

    timer.set_uncapped(False)
    assert timer.interval == 16
    timer.stop()


def test_uncapped_mode_remembered_across_pause():
    """Uncapped mode is remembered: after pause+resume, interval stays 0ms."""
    timer = _make_timer()
    timer.set_uncapped(True)
    timer.resume()
    assert timer.interval == 0
    timer.pause()
    timer.resume()
    assert timer.interval == 0
    timer.stop()


# ── State changed signal ─────────────────────────────────────


def test_state_changed_on_resume():
    """state_changed emits True on resume."""
    timer = _make_timer()
    states = []
    timer.state_changed.connect(states.append)
    timer.resume()
    assert states == [True]
    timer.stop()


def test_state_changed_on_pause():
    """state_changed emits False on pause."""
    timer = _make_timer()
    states = []
    timer.state_changed.connect(states.append)
    timer.resume()
    timer.pause()
    assert states == [True, False]


def test_state_changed_on_single_step_from_running():
    """state_changed emits False when single_step pauses a running timer."""
    timer = _make_timer()
    states = []
    timer.state_changed.connect(states.append)
    timer.resume()
    timer.single_step()
    assert states == [True, False]


def test_no_state_changed_on_single_step_from_paused():
    """No state_changed when single_step is called while already paused."""
    timer = _make_timer()
    states = []
    timer.state_changed.connect(states.append)
    timer.single_step()
    assert states == []


# ── Step emission on tick ─────────────────────────────────────


def test_timer_fires_step_requested():
    """When running, the timer fires step_requested signals."""
    import time
    timer = _make_timer()
    steps = []
    timer.step_requested.connect(lambda: steps.append(1))
    timer.resume()

    # Process events with small delays to let the timer fire
    app = get_app()
    for _ in range(50):
        time.sleep(0.005)
        app.processEvents()
        if steps:
            break

    timer.stop()
    assert len(steps) >= 1


# ── MainWindow wiring ────────────────────────────────────────


def test_main_window_has_game_timer():
    """MainWindow has a game_timer attribute."""
    _app = get_app()
    from triforce_debugger.main_window import MainWindow  # pylint: disable=import-outside-toplevel
    window = MainWindow()
    assert hasattr(window, 'game_timer')
    from triforce_debugger.game_timer import GameTimer  # pylint: disable=import-outside-toplevel
    assert isinstance(window.game_timer, GameTimer)
    window.close()


def test_run_menu_continue_resumes_timer():
    """Run > Continue triggers game_timer.resume."""
    _app = get_app()
    from triforce_debugger.main_window import MainWindow  # pylint: disable=import-outside-toplevel
    window = MainWindow()
    assert not window.game_timer.is_running
    window.action_continue.trigger()
    assert window.game_timer.is_running
    window.game_timer.stop()
    window.close()


def test_run_menu_pause_stops_timer():
    """Run > Pause triggers game_timer.pause."""
    _app = get_app()
    from triforce_debugger.main_window import MainWindow  # pylint: disable=import-outside-toplevel
    window = MainWindow()
    window.game_timer.resume()
    window.action_pause.trigger()
    assert not window.game_timer.is_running
    window.close()


def test_run_menu_step_fires_signal():
    """Run > Step triggers a single step_requested."""
    _app = get_app()
    from triforce_debugger.main_window import MainWindow  # pylint: disable=import-outside-toplevel
    window = MainWindow()
    steps = []
    window.game_timer.step_requested.connect(lambda: steps.append(1))
    window.action_step.trigger()
    assert len(steps) == 1
    window.close()


def test_uncap_fps_toggle_changes_mode():
    """View > Uncap FPS toggles uncapped mode on the timer."""
    _app = get_app()
    from triforce_debugger.main_window import MainWindow  # pylint: disable=import-outside-toplevel
    window = MainWindow()
    assert not window.game_timer.is_uncapped
    window.action_uncap_fps.setChecked(True)
    assert window.game_timer.is_uncapped
    window.action_uncap_fps.setChecked(False)
    assert not window.game_timer.is_uncapped
    window.close()
