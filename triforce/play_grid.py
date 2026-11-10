"""Many live games at once for ``play-zelda.py``: one native chain pool (the default route with the promoted
specialists), one worker thread per game paced at ``fps`` frames per second, every game acting on its own inside
the pool (``EnvPool.play``, the specialists' Rust networks) and a Python thread that turns the pool's events into
status messages and holds each finished run's result on screen.

Shown games render every frame into the pool's frame buffers (``show(game, on)`` toggles a game; hidden games keep
playing at the same pace without rendering) and ``frame(game, since)`` copies a game's latest one.
``emulated_frames()`` counts every game's emulated frames, shown or not.  Room/health changes and run results go
through one status queue as dicts with ``game`` and ``event`` (``start``, ``room``, ``end``, ``crash``); ``room``
carries ``rooms``, the rooms entered so far in the run (caves and cellars count),
``stalled``: with ``stall_steps`` set, the room has gone that many actions without damage dealt or taken (the
run ends ``stuck`` at twice as many), and ``cheated``: cheated items used so far in the run, ``{bombs, keys,
rupees}`` (a room that needs an item Link lacks grants it; crates/zelda/src/cheats.rs).
After a run ends its game holds its result for ``RESULT_PAUSE`` seconds, showing the run's last frame (the next
run has reset but waits, parked in the pool, and its messages wait too), before the next seed plays (seeds are
handed out in order to whichever game is free, so ``start`` carries no seed; ``end`` does).  A setup error (an
error before the first step, e.g. a missing promoted model for the route's first room) stops the game instead of
retrying in a loop.  ``plays``: keep every run in a replay file (``triforce.replay``).

``set_fps(fps)`` changes every game's frame-rate cap (``<= 0``: uncapped, as fast as they run; ``NES_FPS`` is the
console's own rate).  ``set_paused(True)`` stops every game before its next emulated frame (a step in progress
stops mid-step) and stops the result-pause clock; ``set_paused(False)`` resumes both.
"""

import queue
import threading
import time
from typing import Dict, Iterable, List, Optional

import numpy as np

from . import _native
from .runner import ChainPool, Policies, Route, RunResult

FRAME_SHAPE = _native.FRAME_SHAPE   # rows, columns, RGB
ROUTE = _native.DEFAULT_ROUTE
RESULT_PAUSE = 3.0
SEEDS_PER_GAME = 1000
POLL_SECONDS = 0.02
NES_FPS = 60.0988           # frames per second of an NTSC NES


def end_message(game: int, seed: Optional[int], result: RunResult) -> tuple[dict, bool]:
    """The 'end' status message for a finished ``RunResult`` and whether it was a setup error.  ``outcome`` is the
    last room's (milestone | death | stuck | exit-wrong | missed | retreat-limit; see triforce/runner.py), the run's
    result when it has no rooms, or 'error' for any error.  An error before the first step (e.g. no promoted model)
    is a setup error; one mid-game (e.g. wandering into an untagged room) is just another ending."""
    if result.result == 'error':
        outcome = 'error'
    else:
        outcome = result.rooms[-1].outcome if result.rooms else result.result
    message = {'game': game, 'event': 'end', 'seed': seed, 'result': result.result, 'outcome': outcome,
               'reason': result.reason}
    return message, result.result == 'error' and result.steps == 0


class _Held:
    """A game whose next run is ready but waits until ``until`` (the last result is on screen), and the status
    messages its new run produced meanwhile."""

    def __init__(self, until: float):
        self.until = until
        self.messages: List[dict] = []


class GameGrid:
    """Plays ``games`` games of the default route in one chain pool and exposes their latest frames and status
    messages.  ``shown``: the games that render (default all)."""

    def __init__(self, games: int, fps: float, models_dir=None, shown: Optional[Iterable[int]] = None,
                 states_dir=None, stall_steps: Optional[int] = None, plays=None):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.games = games
        self.fps = fps
        self.models_dir = None if models_dir is None else str(models_dir)
        self.states_dir = states_dir
        self.stall_steps = stall_steps
        self.plays = plays
        self._shown = set(range(games)) if shown is None else set(shown)
        self._status: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pool = None
        self._paused_at: Optional[float] = None     # when the current pause began (None: playing)
        self._paused_total = 0.0                    # seconds of earlier pauses
        self._thread = threading.Thread(target=self._run, name='play_grid', daemon=True)

    def start(self) -> None:
        """Loads the models and starts playing (on the status thread)."""
        self._thread.start()

    def show(self, game: int, on: bool) -> int:
        """Renders ``game``'s frames from its next frame on (or stops rendering it); returns its frame counter (a
        later frame is fresh)."""
        with self._lock:
            if on:
                self._shown.add(game)
            else:
                self._shown.discard(game)
            if self._pool is None:
                return 0
            self._pool.set_render(game, on)
            return self._pool.frame_counter(game)

    def frame(self, game: int, since: int) -> Optional[tuple[int, np.ndarray]]:
        """(frame number, RGB copy) of a game's latest frame, or None when it is still frame ``since``."""
        with self._lock:
            if self._pool is None:
                return None
            return self._pool.frame(game, since=since)

    def set_fps(self, fps: float) -> None:
        """Every game's frame-rate cap from its next frame on (``<= 0``: uncapped)."""
        with self._lock:
            self.fps = fps
            if self._pool is not None:
                for game in range(self.games):
                    self._pool.set_fps(game, fps)

    def set_paused(self, on: bool) -> None:
        """Pauses (or resumes) every game before its next emulated frame, and the result-pause clock with them."""
        with self._lock:
            now = time.monotonic()
            if on and self._paused_at is None:
                self._paused_at = now
            elif not on and self._paused_at is not None:
                self._paused_total += now - self._paused_at
                self._paused_at = None
            if self._pool is not None:
                self._pool.set_paused(on)

    @property
    def paused(self) -> bool:
        return self._paused_at is not None

    def _now(self) -> float:
        """Seconds of play: monotonic time minus every pause (frozen while paused)."""
        with self._lock:
            now = time.monotonic() if self._paused_at is None else self._paused_at
            return now - self._paused_total

    def emulated_frames(self) -> list[int]:
        """Frames each game has emulated so far (shown or not), counted as its steps finish."""
        with self._lock:
            if self._pool is None:
                return [0] * self.games
            return [self._pool.emulated_frames(game) for game in range(self.games)]

    def frame_counters(self) -> list[int]:
        """Frames each game has shown (published) so far; only shown games render, frame by frame."""
        with self._lock:
            if self._pool is None:
                return [0] * self.games
            return [self._pool.frame_counter(game) for game in range(self.games)]

    def poll_status(self) -> list[dict]:
        """Every status message received since the last call."""
        messages = []
        while True:
            try:
                messages.append(self._status.get_nowait())
            except queue.Empty:
                return messages

    def stop(self, timeout: float = 5.0) -> None:
        """Stops every game (idempotent): signal the status thread, which closes the pool."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout)

    @property
    def stopped(self) -> bool:
        """The status thread has exited."""
        return not self._thread.is_alive()

    # ------------------------------------------------------------------ status thread

    def _run(self) -> None:
        try:
            route = Route.load(ROUTE)
            policies = Policies(self.models_dir)
            first = _native.route_steps(ROUTE)[0]['specialist']
            if first in policies.missing:
                # Every run would end before its first step; don't start games that fail in a loop.
                for game in range(self.games):
                    self._status.put({'game': game, 'event': 'start', 'seed': None})
                    result = RunResult(ROUTE, None, 'error', policies.missing[first])
                    self._status.put(end_message(game, None, result)[0])
                return
            seeds = range(self.games * SEEDS_PER_GAME)
            chain = ChainPool(route, seeds, policies, num_envs=self.games, threads=self.games, fps=self.fps,
                              states_dir=self.states_dir, stall_steps=self.stall_steps, hold_finished_frame=True,
                              plays=self.plays)
        except Exception as err:  # pylint: disable=broad-exception-caught
            self._crash(err)
            return
        try:
            with self._lock:
                self._pool = chain.pool
                for game in self._shown:
                    chain.pool.set_render(game, True)
                for game in range(self.games):
                    chain.pool.set_fps(game, self.fps)      # in case set_fps ran while the pool was being built
                chain.pool.set_paused(self._paused_at is not None)
            for game in range(self.games):
                self._status.put({'game': game, 'event': 'start', 'seed': None})
            chain.play()
            self._play(chain)
        except Exception as err:  # pylint: disable=broad-exception-caught
            self._crash(err)
        finally:
            with self._lock:
                self._pool = None
            chain.close()

    def _crash(self, err: Exception) -> None:
        for game in range(self.games):
            self._status.put({'game': game, 'event': 'crash', 'reason': f"{type(err).__name__}: {err}"})

    def _play(self, chain: ChainPool) -> None:
        """Turns events into status messages; holds games showing a result and releases them later."""
        held: Dict[int, _Held] = {}
        while not self._stop.wait(POLL_SECONDS) and chain.pool.active:
            chain.pool.wait_idle(timeout=0)                 # raises an env's failure
            for event in chain.poll_events():
                self._event(event, held)
            now = self._now()
            for env in [env for env, hold in held.items() if hold.until <= now]:
                hold = held.pop(env)
                self._status.put({'game': env, 'event': 'start', 'seed': None})
                for message in hold.messages:
                    self._status.put(message)
                chain.pool.set_hold(env, False)         # the next run plays; its frames show from its first action

    def _event(self, event: dict, held: Dict[int, _Held]) -> None:
        env = event['env']
        if event['type'] == 'run':
            result = RunResult.from_dict(event['result'])
            message, setup_error = end_message(env, result.seed, result)
            self._status.put(message)
            # A setup error stops the game: it stays parked, never released.
            held[env] = _Held(float('inf') if setup_error else self._now() + RESULT_PAUSE)
        elif event['type'] == 'room':
            message = {'game': env, 'event': 'room', 'specialist': event['specialist'], 'level': event['level'],
                       'room': event['room'], 'rooms': int(event['rooms']), 'health': float(event['health']),
                       'max_health': int(event['max_health']), 'stalled': bool(event['stalled']),
                       'cheated': {item: int(n) for item, n in event['cheated'].items()}}
            if env in held:
                held[env].messages.append(message)
            else:
                self._status.put(message)
