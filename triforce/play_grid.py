"""Many live games at once for ``play-zelda.py``: one spawned process per game (stable-retro allows one emulator
per process), each playing route ``milestone1`` with the promoted specialists at a paced frame rate.

Every emulated frame is copied into a shared buffer the GUI reads; room/health changes and run results go through
one shared status queue as dicts with ``cell`` and ``event`` (``start``, ``room``, ``end``, ``crash``).  After a
run ends the cell shows its result for ``RESULT_PAUSE`` seconds and plays the next seed (``cell + k * cells``); a
configuration error (e.g. a missing promoted model) stops the cell instead of retrying in a loop.
"""

import multiprocessing as mp
import queue
import time
from typing import Optional

import numpy as np

FRAME_SHAPE = (224, 240, 3)        # Emulator.get_screen(): rows, columns, RGB
FRAME_BYTES = 224 * 240 * 3
ROUTE = 'milestone1'
RESULT_PAUSE = 3.0
RESYNC_SECONDS = 0.25


class StopPlaying(Exception):
    """Raised from the per-frame hook to unwind out of ``Runner.run`` when the grid stops."""


class FramePacer:
    """Holds a frame rate: ``wait()`` sleeps until the next frame is due.  Falling more than RESYNC_SECONDS behind
    restarts the schedule instead of fast-forwarding to catch up.  ``fps <= 0`` disables pacing."""

    def __init__(self, fps: float, clock=time.monotonic, sleep=time.sleep):
        self.period = 1.0 / fps if fps > 0 else 0.0
        self.clock = clock
        self.sleep = sleep
        self._next = None

    def wait(self) -> None:
        if not self.period:
            return
        if self._next is None:
            self._next = self.clock()
        self._next += self.period
        delay = self._next - self.clock()
        if delay > 0:
            self.sleep(delay)
        elif delay < -RESYNC_SECONDS:
            self._next = self.clock()


class _Reporter:
    """The Runner's renderer hook: reports the specialist, room and health whenever one of them changes."""

    def __init__(self, cell: int, status):
        self.cell = cell
        self.status = status
        self._last = None

    def show(self, session, specialist, action, change):  # pylint: disable=unused-argument
        state = change.state
        info = (specialist, state.level, state.location, state.link.health, state.link.max_health)
        if info != self._last:
            self._last = info
            self.status.put({'cell': self.cell, 'event': 'room', 'specialist': specialist, 'level': state.level,
                             'room': state.location, 'health': float(state.link.health),
                             'max_health': int(state.link.max_health)})


def play_cell(cell: int, cells: int, device: str, fps: float, models_dir, frame_buf, frame_no, status,
              stop) -> None:
    """Process target: plays seeds ``cell, cell + cells, ...`` until ``stop`` is set."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, import-outside-toplevel
    import torch
    from .runner import Route, Runner
    from .session import GameSession

    torch.set_num_threads(1)    # N processes × all-core torch threads oversubscribes the machine
    session = GameSession()
    try:
        runner = Runner(session=session, models_dir=models_dir, renderer=_Reporter(cell, status), device=device)
        route = Route.load(ROUTE)
        pacer = FramePacer(fps)
        screen = np.frombuffer(frame_buf.get_obj(), dtype=np.uint8)

        def on_frame():
            if stop.is_set():
                raise StopPlaying
            with frame_buf.get_lock():
                screen[:] = session.emu.get_screen().reshape(-1)
            frame_no.value += 1
            pacer.wait()

        session.emu.on_frame = on_frame
        k = 0
        while True:
            seed = cell + k * cells
            status.put({'cell': cell, 'event': 'start', 'seed': seed})
            result = runner.run(route, seed)
            outcome = result.rooms[-1].outcome if result.rooms else result.result
            status.put({'cell': cell, 'event': 'end', 'seed': seed, 'result': result.result, 'outcome': outcome,
                        'reason': result.reason})
            if result.result == 'error':
                stop.wait()
                return
            if stop.wait(RESULT_PAUSE):
                return
            k += 1
    except StopPlaying:
        return
    except Exception as err:  # pylint: disable=broad-exception-caught
        status.put({'cell': cell, 'event': 'crash', 'reason': f"{type(err).__name__}: {err}"})
    finally:
        session.close()


class GameGrid:
    """Parent side: starts one game process per cell and exposes their latest frames and status messages."""

    def __init__(self, cells: int, device: str, fps: float, models_dir=None):
        ctx = mp.get_context('spawn')
        self.cells = cells
        self._buffers = [ctx.Array('B', FRAME_BYTES) for _ in range(cells)]
        self._counters = [ctx.Value('q', 0, lock=False) for _ in range(cells)]
        self._status = ctx.Queue()
        self._stop = ctx.Event()
        self._processes = [
            ctx.Process(target=play_cell, name=f'play_grid-{cell}', daemon=True,
                        args=(cell, cells, device, fps, None if models_dir is None else str(models_dir),
                              self._buffers[cell], self._counters[cell], self._status, self._stop))
            for cell in range(cells)]
        self._stopped = False

    def start(self) -> None:
        for process in self._processes:
            process.start()

    def frame(self, cell: int, since: int) -> Optional[tuple[int, np.ndarray]]:
        """(frame number, RGB copy) of a cell's latest frame, or None when it is still frame ``since``."""
        number = self._counters[cell].value
        if number == since:
            return None
        buffer = self._buffers[cell]
        with buffer.get_lock():
            pixels = np.frombuffer(buffer.get_obj(), dtype=np.uint8).reshape(FRAME_SHAPE).copy()
        return number, pixels

    def poll_status(self) -> list[dict]:
        """Every status message received since the last call."""
        messages = []
        while True:
            try:
                messages.append(self._status.get_nowait())
            except (queue.Empty, OSError, ValueError):
                return messages

    def stop(self, timeout: float = 5.0) -> None:
        """Stops every game (idempotent): signal, wait while draining the queue, then terminate stragglers."""
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and any(p.is_alive() for p in self._processes):
            self.poll_status()
            for process in self._processes:
                if process.is_alive():
                    process.join(0.1)
        for process in self._processes:
            if process.is_alive():
                process.terminate()
                process.join(1)
        self._status.close()
        self._status.cancel_join_thread()
