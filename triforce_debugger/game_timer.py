"""QTimer-based game loop for the Triforce Debugger.

Provides capped (60fps) and uncapped modes, pause/resume/single-step control.
Emits signals so the main window can update panels after each step.
"""

from PySide6.QtCore import QObject, QTimer, Signal


class GameTimer(QObject):
    """Drives the environment step loop via QTimer.

    Modes:
        - PAUSED: timer stopped, no automatic stepping.
        - RUNNING_CAPPED: timer fires every 16ms (~60fps).
        - RUNNING_UNCAPPED: timer fires every 0ms (as fast as Qt event loop).

    Signals:
        step_requested: Emitted each time the timer fires and a step should occur.
        state_changed: Emitted when the timer mode changes (paused/resumed).
    """

    CAPPED_INTERVAL_MS = 16   # ~60 fps
    UNCAPPED_INTERVAL_MS = 0  # as fast as possible

    # Emitted each time the timer fires to request one env step.
    step_requested = Signal()

    # Emitted when running/paused state changes. Bool is True when running.
    state_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)  # pylint: disable=no-member

        self._running = False
        self._uncapped = False

    # ── Properties ────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """True when the timer is actively firing (not paused)."""
        return self._running

    @property
    def is_uncapped(self) -> bool:
        """True when running in uncapped (0ms) mode."""
        return self._uncapped

    @property
    def interval(self) -> int:
        """Current timer interval in milliseconds."""
        return self._timer.interval()

    # ── Control methods ───────────────────────────────────────

    def resume(self):
        """Start or resume automatic stepping."""
        if self._running:
            return
        self._running = True
        self._timer.start(self._current_interval())
        self.state_changed.emit(True)

    def pause(self):
        """Pause automatic stepping."""
        if not self._running:
            return
        self._running = False
        self._timer.stop()
        self.state_changed.emit(False)

    def single_step(self):
        """Execute exactly one step (pauses if currently running)."""
        was_running = self._running
        if was_running:
            self._running = False
            self._timer.stop()
            self.state_changed.emit(False)
        self.step_requested.emit()

    def set_uncapped(self, uncapped: bool):
        """Switch between capped (16ms) and uncapped (0ms) modes."""
        self._uncapped = uncapped
        if self._running:
            self._timer.setInterval(self._current_interval())

    def stop(self):
        """Fully stop the timer (for shutdown)."""
        self._running = False
        self._timer.stop()

    # ── Internal ──────────────────────────────────────────────

    def _current_interval(self) -> int:
        """Return the interval based on the current mode."""
        return self.UNCAPPED_INTERVAL_MS if self._uncapped else self.CAPPED_INTERVAL_MS

    def _on_tick(self):
        """Timer callback — emit step_requested."""
        self.step_requested.emit()
