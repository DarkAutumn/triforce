"""Many live games at once for ``play-zelda.py``: one spawned process per game (stable-retro allows one emulator
per process), each playing route ``milestone1`` with the promoted specialists at a paced frame rate.

A game copies its emulated frames into a shared buffer only while the GUI shows it (``GameGrid.show``); hidden
games keep playing at the same pace.  Room/health changes and run results go through one shared status queue as
dicts with ``game`` and ``event`` (``start``, ``room``, ``end``, ``crash``); ``room`` carries ``rooms``, the rooms
entered so far in the run.  After a run ends the game keeps its result for ``RESULT_PAUSE`` seconds and plays the
next seed (``game + k * games``); a configuration error (e.g. a missing promoted model) stops the game instead of
retrying in a loop.
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
    """The Runner's renderer hook: reports the specialist, room, rooms entered and health whenever one changes."""

    def __init__(self, game: int, status):
        self.game = game
        self.status = status
        self._last = None
        self._location = None
        self.rooms = 0

    def start_run(self):
        """Forgets the previous run's progress."""
        self._last = self._location = None
        self.rooms = 0

    def show(self, session, specialist, action, change):  # pylint: disable=unused-argument
        state = change.state
        location = (state.level, state.location, state.in_cave)
        if location != self._location:
            self._location = location
            self.rooms += 1
        info = (specialist, state.level, state.location, state.link.health, state.link.max_health)
        if info != self._last:
            self._last = info
            self.status.put({'game': self.game, 'event': 'room', 'specialist': specialist, 'level': state.level,
                             'room': state.location, 'rooms': self.rooms, 'health': float(state.link.health),
                             'max_health': int(state.link.max_health)})


def end_message(game: int, seed: int, result) -> tuple[dict, bool]:
    """The 'end' status message for a finished ``RunResult`` and whether it was a setup error.  ``outcome`` is the
    last room's (milestone | death | stuck) or 'error' for any error.  An error before the first step (e.g. no
    promoted model) is a setup error; one mid-game (e.g. wandering into an untagged room) is just another ending."""
    if result.result == 'error':
        outcome = 'error'
    else:
        outcome = result.rooms[-1].outcome if result.rooms else result.result
    message = {'game': game, 'event': 'end', 'seed': seed, 'result': result.result, 'outcome': outcome,
               'reason': result.reason}
    return message, result.result == 'error' and result.steps == 0


def play_game(game: int, games: int, device: str, fps: float, models_dir, frame_buf, copied, emulated, shown,
              status, stop) -> None:
    """Process target: plays seeds ``game, game + games, ...`` until ``stop`` is set.  ``emulated`` counts every
    frame; while ``shown`` is set each frame is also copied into ``frame_buf`` and counted in ``copied``."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, import-outside-toplevel
    import torch
    from .runner import Route, Runner
    from .session import GameSession

    torch.set_num_threads(1)    # N processes × all-core torch threads oversubscribes the machine
    session = GameSession(look_ahead=False)     # nothing is scored here; look-ahead frames would be shown and paced
    try:
        reporter = _Reporter(game, status)
        runner = Runner(session=session, models_dir=models_dir, renderer=reporter, device=device)
        route = Route.load(ROUTE)
        pacer = FramePacer(fps)
        screen = np.frombuffer(frame_buf.get_obj(), dtype=np.uint8)

        def on_frame():
            if stop.is_set():
                raise StopPlaying
            emulated.value += 1
            if shown.value:
                with frame_buf.get_lock():
                    screen[:] = session.emu.get_screen().reshape(-1)
                copied.value += 1
            pacer.wait()

        session.emu.on_frame = on_frame
        k = 0
        while True:
            seed = game + k * games
            reporter.start_run()
            status.put({'game': game, 'event': 'start', 'seed': seed})
            result = runner.run(route, seed)
            message, setup_error = end_message(game, seed, result)
            status.put(message)
            if setup_error:
                stop.wait()     # every seed would fail the same way at once; don't restart in a loop
                return
            if stop.wait(RESULT_PAUSE):
                return
            k += 1
    except StopPlaying:
        return
    except Exception as err:  # pylint: disable=broad-exception-caught
        status.put({'game': game, 'event': 'crash', 'reason': f"{type(err).__name__}: {err}"})
    finally:
        session.close()


class GameGrid:
    """Parent side: starts one process per game, exposes the shown games' latest frames and every status message.
    The first ``shown`` games start shown (default: all)."""

    def __init__(self, games: int, device: str, fps: float, models_dir=None, shown: Optional[int] = None):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        ctx = mp.get_context('spawn')
        self.games = games
        shown = games if shown is None else shown
        self._buffers = [ctx.Array('B', FRAME_BYTES) for _ in range(games)]
        self._copied = [ctx.Value('q', 0, lock=False) for _ in range(games)]
        self._emulated = [ctx.Value('q', 0, lock=False) for _ in range(games)]
        self._shown = [ctx.Value('b', int(game < shown), lock=False) for game in range(games)]
        self._status = ctx.Queue()
        self._stop = ctx.Event()
        self._processes = [
            ctx.Process(target=play_game, name=f'play_grid-{game}', daemon=True,
                        args=(game, games, device, fps, None if models_dir is None else str(models_dir),
                              self._buffers[game], self._copied[game], self._emulated[game], self._shown[game],
                              self._status, self._stop))
            for game in range(games)]
        self._stopped = False

    def start(self) -> None:
        for process in self._processes:
            process.start()

    def show(self, game: int, shown: bool) -> int:
        """Starts or stops copying a game's frames; returns its copied-frame count (a later frame is fresh)."""
        self._shown[game].value = int(shown)
        return self._copied[game].value

    def emulated_frames(self) -> list[int]:
        """Frames each game has emulated so far (shown or not)."""
        return [counter.value for counter in self._emulated]

    def frame(self, game: int, since: int) -> Optional[tuple[int, np.ndarray]]:
        """(frame number, RGB copy) of a shown game's latest frame, or None when it is still frame ``since``."""
        number = self._copied[game].value
        if number == since:
            return None
        buffer = self._buffers[game]
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
