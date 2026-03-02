"""QTimer-based game loop for the Triforce Debugger.

Architecture:
  A single render timer drives everything.  On each tick it pops one NES frame
  from a small buffer and emits ``frame_ready``.  When the buffer is empty it
  calls back to the main window to perform an env step, which refills the
  buffer with that step's frames.  Then it schedules the next tick.

  - **Capped mode (60 fps):** render timer fires every 16ms.  One NES frame
    per tick → smooth real-time playback with no jitter.
  - **Uncapped mode:** render timer fires at 0ms.  Drains the buffer as fast
    as the event loop allows, stepping whenever it runs dry.

This keeps the buffer tiny (at most one step's worth of frames) so the user
can pause at any time and immediately take manual control.

Signals:
  frame_ready(object)        — a single NES frame (numpy array) to render.
  step_completed(object)     — a StepResult, emitted at step boundaries.
  state_changed(bool)        — True when running, False when paused.
"""

from collections import deque
from PySide6.QtCore import QObject, QTimer, Signal


class GameTimer(QObject):
    """Drives the environment step loop and frame rendering."""

    MS_PER_NES_FRAME = 16     # ~60 fps  (1000/60 ≈ 16.67, truncated)
    UNCAPPED_INTERVAL_MS = 0  # as fast as possible

    # Emitted for every NES frame that should be drawn.
    frame_ready = Signal(object)

    # Emitted at step boundaries with the full StepResult.
    step_completed = Signal(object)

    # Emitted when the timer needs the main window to perform an env step.
    # The main window should call ``enqueue_step()`` with the result.
    step_requested = Signal()

    # Emitted when running/paused state changes.  Bool is True when running.
    state_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._tick)

        # Frame buffer: each entry is (frame, step_result_or_None).
        # step_result is non-None only on the *last* frame of a step.
        self._buffer: deque[tuple] = deque()

        self._running = False
        self._draining = False  # True when playing back a manual step's frames
        self._uncapped = True  # default to uncapped (fast as possible)

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
    def buffer_depth(self) -> int:
        """Number of frames waiting in the render buffer."""
        return len(self._buffer)

    # ── Control methods ───────────────────────────────────────

    def resume(self):
        """Start or resume automatic stepping."""
        if self._running:
            return
        self._draining = False
        self._running = True
        self._schedule_next()
        self.state_changed.emit(True)

    def pause(self):
        """Pause automatic stepping."""
        if not self._running:
            return
        self._running = False
        self._draining = False
        self._timer.stop()
        self._buffer.clear()
        self.state_changed.emit(False)

    def single_step(self):
        """Execute exactly one step (pauses if currently running)."""
        was_running = self._running
        if was_running:
            self._running = False
            self._timer.stop()
            self._buffer.clear()
            self.state_changed.emit(False)
        # Ask the main window for one step — it will call enqueue_step(),
        # and we emit all those frames immediately.
        self.step_requested.emit()
        self._flush_buffer()

    def set_uncapped(self, uncapped: bool):
        """Switch between capped and uncapped modes."""
        self._uncapped = uncapped
        if self._running:
            self._timer.stop()
            self._schedule_next()

    def enqueue_step(self, frames, step_result):
        """Called by the main window after performing an env step.

        *frames* is the list of NES frames from the step.
        *step_result* is the full StepResult.
        """
        if not frames:
            return
        # Tag only the last frame with the step result.
        for i, f in enumerate(frames):
            sr = step_result if i == len(frames) - 1 else None
            self._buffer.append((f, sr))

    def play_frames(self, frames, step_result):
        """Enqueue frames from a manual/one-off step and play them back.

        In uncapped mode the frames are flushed immediately.
        In capped mode the render timer drains them at 16ms per frame.
        """
        self.enqueue_step(frames, step_result)
        if self._uncapped:
            self._flush_buffer()
        else:
            # Start a drain cycle — render timer pops frames but doesn't
            # request new env steps.
            self._draining = True
            self._timer.start(self.MS_PER_NES_FRAME)

    def stop(self):
        """Fully stop the timer (for shutdown)."""
        self._running = False
        self._draining = False
        self._timer.stop()
        self._buffer.clear()

    # ── Internal ──────────────────────────────────────────────

    def _schedule_next(self):
        """Schedule the next render tick."""
        interval = self.UNCAPPED_INTERVAL_MS if self._uncapped else self.MS_PER_NES_FRAME
        self._timer.start(interval)

    def _tick(self):
        """Render-timer callback.

        Normal (running) mode:
          1. If the buffer is empty, request a step (which refills it).
          2. Pop one frame and emit it.
          3. If it's a step boundary, emit step_completed.
          4. Schedule the next tick.

        Draining mode (manual step playback):
          Pop one frame per tick.  When empty, stop — don't request new steps.
        """
        if self._draining:
            if self._buffer:
                frame, step_result = self._buffer.popleft()
                self.frame_ready.emit(frame)
                if step_result is not None:
                    self.step_completed.emit(step_result)
            if self._buffer:
                self._timer.start(self.MS_PER_NES_FRAME)
            else:
                self._draining = False
            return

        if not self._running:
            return

        # Refill if empty — step synchronously via signal.
        if not self._buffer:
            self.step_requested.emit()

        # Pop and render one frame.
        if self._buffer:
            frame, step_result = self._buffer.popleft()
            self.frame_ready.emit(frame)
            if step_result is not None:
                self.step_completed.emit(step_result)

        # Keep going.
        if self._running:
            self._schedule_next()

    def _flush_buffer(self):
        """Emit all buffered frames immediately (used for single_step)."""
        while self._buffer:
            frame, step_result = self._buffer.popleft()
            self.frame_ready.emit(frame)
            if step_result is not None:
                self.step_completed.emit(step_result)
