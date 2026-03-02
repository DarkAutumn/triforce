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
    """GameTimer starts paused and uncapped."""
    timer = _make_timer()
    assert not timer.is_running
    assert timer.is_uncapped


def test_default_constants():
    """Default constants are correct."""
    from triforce_debugger.game_timer import GameTimer  # pylint: disable=import-outside-toplevel
    assert GameTimer.MS_PER_NES_FRAME == 16
    assert GameTimer.UNCAPPED_INTERVAL_MS == 0


# ── Resume / pause ────────────────────────────────────────────


def test_resume_starts_timer():
    """Resume starts the timer and sets running state."""
    timer = _make_timer()
    timer.resume()
    assert timer.is_running
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


def test_pause_clears_buffer():
    """Pausing clears any buffered frames."""
    timer = _make_timer()
    timer.enqueue_step(["frame1", "frame2"], "result")
    assert timer.buffer_depth == 2
    timer.resume()
    timer.pause()
    assert timer.buffer_depth == 0
    timer.stop()


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


def test_single_step_flushes_buffer():
    """Single step emits all enqueued frames immediately."""
    timer = _make_timer()
    frames_seen = []
    results_seen = []
    timer.frame_ready.connect(frames_seen.append)
    timer.step_completed.connect(results_seen.append)
    # Pre-load step_requested to enqueue frames
    timer.step_requested.connect(
        lambda: timer.enqueue_step(["f1", "f2", "f3"], "step_result"))
    timer.single_step()
    assert frames_seen == ["f1", "f2", "f3"]
    assert results_seen == ["step_result"]


# ── Uncapped mode ─────────────────────────────────────────────


def test_set_uncapped():
    """Setting uncapped mode updates the property."""
    timer = _make_timer()
    assert timer.is_uncapped  # default is uncapped
    timer.set_uncapped(False)
    assert not timer.is_uncapped
    timer.set_uncapped(True)
    assert timer.is_uncapped


# ── enqueue_step / buffer ────────────────────────────────────


def test_enqueue_step_fills_buffer():
    """enqueue_step adds frames to the buffer."""
    timer = _make_timer()
    timer.enqueue_step(["f1", "f2", "f3"], "result")
    assert timer.buffer_depth == 3


def test_enqueue_step_empty_frames():
    """enqueue_step with empty frames is a no-op."""
    timer = _make_timer()
    timer.enqueue_step([], "result")
    assert timer.buffer_depth == 0


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


# ── Tick drives step_requested then renders ──────────────────


def test_tick_requests_step_when_buffer_empty():
    """When the render tick fires with an empty buffer, step_requested is emitted."""
    import time
    timer = _make_timer()
    step_requests = []
    timer.step_requested.connect(lambda: step_requests.append(1))
    # Enqueue frames when requested so _tick has something to pop.
    timer.step_requested.connect(
        lambda: timer.enqueue_step(["frame"], "result"))
    timer.resume()

    app = get_app()
    for _ in range(50):
        time.sleep(0.005)
        app.processEvents()
        if step_requests:
            break

    timer.stop()
    assert len(step_requests) >= 1


def test_frame_ready_emitted_on_tick():
    """Frames from enqueue_step are emitted via frame_ready on render ticks."""
    import time
    timer = _make_timer()
    frames_seen = []
    timer.frame_ready.connect(frames_seen.append)
    timer.step_requested.connect(
        lambda: timer.enqueue_step(["f1", "f2"], "result"))
    timer.resume()

    app = get_app()
    for _ in range(100):
        time.sleep(0.005)
        app.processEvents()
        if len(frames_seen) >= 2:
            break

    timer.stop()
    assert len(frames_seen) >= 2


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
    assert window.game_timer.is_uncapped  # default checked
    window.action_uncap_fps.setChecked(False)
    assert not window.game_timer.is_uncapped
    window.action_uncap_fps.setChecked(True)
    assert window.game_timer.is_uncapped
    window.close()
