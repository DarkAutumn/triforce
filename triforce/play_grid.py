"""A grid of cells that play replays, for ``play-zelda.py`` (the window is ``triforce.play_view``).

Every cell plays a replay (a TFREPLAY file's bytes) on the emulator alone (``_native.ReplayScreen``: one thread
per cell at the grid's frame rate), and the grid decides which replay a cell plays next; only a ``live`` grid's
cell 0 is a live game (below).  Where the replays come from is the grid's *source*:

- One of the queues ``best``, ``success``, ``stuck``, ``died``, ``random`` (``QUEUES``).  The promoted models play
  the route headless in a chain pool that records every run, and a background thread in Rust
  (``_native.ReplayFeed``, ``crates/theater``) files each finished run: into its ending's queue, a copy of any
  run into ``random``, and the best runs seen (a win, then the furthest route step, then the most hearts, then
  the fewest frames) into ``best``.  Each queue holds ``capacity`` runs.  A cell on ``best`` takes the top run;
  for every run taken, ``parallel`` more are simulated and the best of them kept.  The simulations pause once
  nothing wants a run and resume when a cell takes one; after ``patience`` finished runs in a row that no queue
  took (a queue for an ending that hardly ever happens) they pause until the next take (0: never give up).
- ``golden``: golden paths (``triforce.golden``: games race from checkpoint to checkpoint, the winners stitched
  into one replay from the start to the route's last triforce), made one after another on a Python thread while
  fewer than ``golden_capacity`` are waiting.  Paths that fail a segment are dropped.
- Replay files (``path``): one file, played on loop, or a directory's top-level ``*.tfreplay`` files in name
  order, each cell taking the next one that no other cell is showing, round and round (the directory is read
  again each time round, so files written meanwhile join in).  A grid with more cells than files leaves the
  rest dark.

``select(source)`` switches between the queues and ``golden`` at any time (every cell cuts to the new source);
their producers start the first time they are selected and keep their queues stocked from then on.  ``skip()``
sends every cell to its next replay.

The run-time clips switch (round 22): the simulations start with ``clips``; ``set_clips(on)`` changes it for every
run started afterwards (``EnvPool.set_clips_allowed``).  Each queued run keeps the switch it was played with, and
a cell's ``start`` message carries it (``clips``: False = played without clips; None for a replay file).  Golden
paths always play with clips on.

``live``: cell 0 shows a run the models play while it is watched (``LiveCell``: a one-env chain pool paced at the
grid's frame rate, no sound, no controller drawn), one run after another; ``set_clips`` reaches that run from its
next decision and the route re-plans from where Link stands.  Each finished live run is filed into the queues like
a simulated one (``ReplayFeed.file``; saved with ``replays`` too), the simulations starting with the live cell.

``poll_status()`` returns what happened since the last call, as dicts with ``cell`` and ``event``: ``wait`` (the
source has nothing yet), ``start`` (``seed``, ``source``, ``name``, ``clips``), ``room`` (``specialist``,
``level``, ``room``, ``rooms``: rooms entered so far, ``cheated``: cheated items used so far), ``end`` (``seed``,
``result``, ``outcome``, ``reason``), ``crash`` (``reason``: the cell's replay could not be played, or the source
failed) and, from a live cell, ``clips`` (``clips``, ``clips_from``: the room the switch went off in during the
run, '' at its start; also on its ``start`` and ``room`` messages).  A finished
replay's last frame stays up for ``RESULT_PAUSE`` seconds before the cell's next one.  ``frame(cell, since)``
copies a cell's latest frame; ``set_fps`` changes every cell's frame rate (``<= 0``: as fast as it plays;
``NES_FPS`` is the console's own); ``set_paused(True)`` stops every cell before its next frame and the result
clock with it (the simulations are not the picture: they keep filling their queues).

``set_sound(cell)`` makes one cell heard on the sound device (None: none), from whatever its game is playing at
that moment; a cell's sound is made as its frames are played, so it is only right at ``NES_FPS``.  ``controller``
draws the controller, its held buttons lit, into every replayed frame.
"""

import bisect
import collections
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import _native, golden, registry, replay
from .runner import ChainPool, Policies, Route, RunResult, chain_meta

FRAME_SHAPE = _native.FRAME_SHAPE   # rows, columns, RGB
ROUTE = _native.DEFAULT_ROUTE
RESULT_PAUSE = 3.0
POLL_SECONDS = 0.02
NES_FPS = 60.0988           # frames per second of an NTSC NES
QUEUES = ('best', 'success', 'stuck', 'died', 'random')
GOLDEN = 'golden'
SOURCES = QUEUES + (GOLDEN,)
FILES = 'files'
LIVE = 'live'               # the source of a live cell (``ReplayGrid(live=True)``: cell 0)
DEFAULT_CAPACITY = 32
DEFAULT_PATIENCE = 64
SEEDS = 1_000_000           # seeds a simulation pool is given: more than a session plays
# The banner of a replay by its header ending; other endings show their last room's outcome.
OUTCOME_OF_ENDING = {'win': 'milestone', 'death': 'death', 'stuck': 'stuck', 'error': 'error'}


class Show:
    """One replay ready for a cell: its bytes and what the strip under the picture says as it plays.  ``clips``:
    False when the run was played with the run-time clips switch off (any part of it), True when on, None when
    unknown (a replay file)."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, data: bytes, source: str, name: Optional[str] = None, clips: Optional[bool] = None):
        recording = _native.Replay(data)
        meta, events = recording.meta, recording.events()
        ending = meta.get('ending') or {}
        models = [model.get('name') for model in meta.get('models') or []]
        self.data, self.source, self.name, self.clips = data, source, name, clips
        self.seed = meta.get('seed')
        self.frames = recording.frames
        self.result, self.reason = ending.get('result') or '', ending.get('reason') or ''
        self.outcome = OUTCOME_OF_ENDING.get(recording.ending) or ending.get('outcome') or recording.ending
        # Watching starts where the models took over: a power-on run's boot and idle are skipped.
        self.start = next((event['frame'] for event in events if event['type'] == 'control'), 0)
        self.specialist = meta.get('specialist')        # an evaluation episode's (it has no room events)
        names = room_names()
        self.rooms = [(event['frame'], models[event['policy']] if event['policy'] is not None
                       and event['policy'] < len(models) else None, event['level'], event['room'],
                       names.get((event['level'], event['room'], event['in_cave'])))
                      for event in events if event['type'] == 'room']
        self._room_frames = [room[0] for room in self.rooms]
        self.cheats = recording.cheats()

    def status(self, frame: int) -> dict:
        """The ``room`` message's fields ``frame`` input frames into the replay."""
        entered = bisect.bisect_right(self._room_frames, frame)
        _, specialist, level, room, name = (self.rooms[entered - 1] if entered
                                            else (0, self.specialist, None, None, None))
        cheated = dict.fromkeys(_native.CHEAT_ITEMS, 0)
        for at, item, count in self.cheats:
            if at <= frame:
                cheated[item] += count
        return {'specialist': specialist, 'level': level, 'room': room, 'name': name, 'rooms': entered,
                'cheated': cheated}


_ROOM_NAMES: Optional[Dict[tuple, str]] = None


def room_names() -> Dict[tuple, str]:
    """game.yaml's room names (``name: Breakfast Room``) by (level, room, in_cave); read once."""
    global _ROOM_NAMES  # pylint: disable=global-statement
    if _ROOM_NAMES is None:
        _ROOM_NAMES = {room: info['name'] for room, info in _native.game_rooms().items() if info['name']}
    return _ROOM_NAMES


class FileSource:
    """The replay files of ``path``: the file itself, or a directory's top-level ``*.tfreplay`` files in name
    order.  ``next(busy)`` goes round them forever."""

    def __init__(self, path):
        self.path = Path(path)
        self.files = self._list()
        if not self.files:
            raise ValueError(f"{self.path} has no {replay.SUFFIX} files" if self.path.is_dir()
                             else f"{self.path} is not a replay file or a directory of them")
        self._cursor = 0

    def _list(self) -> List[Path]:
        if self.path.is_dir():
            return sorted(p for p in self.path.iterdir() if p.suffix == replay.SUFFIX and p.is_file())
        return [self.path] if self.path.is_file() else []

    def next(self, busy=()) -> Show:
        """The next file's replay; a file in ``busy`` (names other cells are showing) is passed over while
        another is free.  Raises ``OSError`` / ``_native.ReplayError`` for a file that can't be read."""
        for _ in range(len(self.files)):
            if self._cursor >= len(self.files):     # once round: read the directory again
                self.files, self._cursor = self._list() or self.files, 0
            path = self.files[self._cursor]
            self._cursor += 1
            if path.name not in busy or len(self.files) <= len(busy):
                break
        return Show(path.read_bytes(), FILES, path.name)


class GoldenSource:
    """Golden paths made one after another on a thread, while fewer than ``capacity`` are waiting.  ``kwargs``:
    ``golden.golden_path``'s (``start_from``, ``power_on``, ``attempts``, ``states_dir``, ``deterministic``,
    ``max_steps``)."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, policies: Policies, route: str, parallel: int, record: dict, capacity: int,
                 first_seed: int = 0, **kwargs):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.policies, self.route, self.parallel, self.record = policies, route, parallel, record
        self.capacity, self.seed, self.kwargs = capacity, first_seed, kwargs
        self.attempts = kwargs.get('attempts', golden.DEFAULT_ATTEMPTS)
        self.parts = len(golden.checkpoints(route)) + 1
        self.made = self.failed = 0
        self.racing: Optional[int] = None       # the segment being raced (1-based), None between paths
        self.error: Optional[str] = None
        self._ready: collections.deque = collections.deque()
        self._wake = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name='golden', daemon=True)
        self._thread.start()

    def next(self) -> Optional[Show]:
        """The oldest waiting golden path, or None."""
        with self._wake:
            if not self._ready:
                return None
            self._wake.notify()
            return self._ready.popleft()

    def stats(self) -> dict:
        with self._wake:
            return {'ready': len(self._ready), 'capacity': self.capacity, 'made': self.made, 'failed': self.failed,
                    'racing': self.racing, 'parts': self.parts, 'error': self.error}

    def close(self, timeout: float = 10.0) -> None:
        with self._wake:
            self._stop = True
            self._wake.notify()
        self._thread.join(timeout)

    def _segment(self, entry: dict) -> None:
        with self._wake:
            self.racing = entry['index'] + 2

    def _run(self) -> None:
        while True:
            with self._wake:
                self.racing = None
                self._wake.wait_for(lambda: self._stop or len(self._ready) < self.capacity)
                if self._stop:
                    return
                self.racing, seed = 1, self.seed
                self.seed += self.parts * self.attempts
            try:
                data, summary = golden.golden_path(self.policies, self.route, self.parallel, seed, self.record,
                                                   on_segment=self._segment, cancel=lambda: self._stop,
                                                   **self.kwargs)
            except golden.Cancelled:
                return
            except Exception as error:  # pylint: disable=broad-exception-caught
                with self._wake:
                    self.error, self.racing = f"{type(error).__name__}: {error}", None
                return
            with self._wake:
                if summary['result'] == 'success':
                    self.made += 1
                    self._ready.append(Show(data, GOLDEN, clips=True))   # golden paths play with clips on
                else:
                    self.failed += 1


def room_text(level, room) -> str:
    """'L4 0x31'."""
    return f"L{level} 0x{room:02x}"


def without_clips(result: RunResult) -> bool:
    """Whether any part of a finished run was played with the run-time clips switch off: it started off, a room
    was entered with it off, or it was turned off during a room."""
    return not result.clips or any(room.clips_allowed is False or any(not flip['clips'] for flip in room.clips_flips)
                                   for room in result.rooms)


class LiveCell:
    """Cell 0 of a ``live`` grid: runs the models play while they are watched, one at a time on a one-env chain
    pool that renders every frame at the grid's frame rate (``ChainPool(fps=..., hold_finished_frame=True)``: a
    finished run's last picture stays up through its result pause while the next run waits).  Runs take seeds in
    order from ``seeds``.  ``set_clips(on)`` turns the run-time clips switch on or off for the run being watched,
    from its next decision (``ChainPool(clips_live=True)``: the route re-plans from where Link stands), and for
    every run after it.

    ``poll(now)`` (``now``: the grid's paused-aware clock) returns the cell's status messages, as
    ``ReplayGrid.poll_status``'s with ``source`` 'live' (``start`` and ``room`` carry ``clips``, the switch, and
    ``clips_from``, the room it went off in during the run, else None; a ``clips`` message says the same when the
    key is pressed), and the finished runs' recordings as (seed, TFREPLAY bytes, whether it was played with clips
    throughout)."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, route: Route, policies: Policies, seeds, fps: float, record: dict, clips: bool, *,
                 deterministic=None, states_dir=None, stall_steps: Optional[int] = None):
        # pylint: disable=too-many-arguments
        self.seeds = seeds
        self.pool = ChainPool(route, seeds, policies, num_envs=1, threads=1, fps=fps, deterministic=deterministic,
                              states_dir=states_dir, stall_steps=stall_steps, record=record,
                              hold_finished_frame=True, clips_live=True)
        self.pool.pool.set_render(0, True)
        self.pool.pool.set_clips_allowed(clips)
        self._lock = threading.Lock()
        self.clips = clips
        self._runs = 0                          # runs started
        self._room: Optional[str] = None        # where Link is ('L4 0x31')
        self._clips_from: Optional[str] = None  # where the switch went off during this run ('' = at its start)
        self._held_until: Optional[float] = None    # a result is up until then; the next run waits
        self._held: List[dict] = []             # the waiting run's messages
        self._out: List[dict] = []
        self._recordings: Dict[int, bytes] = {}     # finished runs' replays waiting for their result, and back
        self._results: Dict[int, bool] = {}
        self._start()
        self.pool.play()

    def _start(self) -> None:
        """A run begins on screen (the lock is held, or nobody else runs yet)."""
        self._room = None
        self._clips_from = None if self.clips else ''
        seed = self.seeds[self._runs] if self._runs < len(self.seeds) else None
        self._runs += 1
        self._out.append({'cell': 0, 'event': 'start', 'seed': seed, 'source': LIVE, 'name': None,
                          'clips': self.clips, 'clips_from': self._clips_from})

    def set_clips(self, on: bool) -> None:
        with self._lock:
            if on == self.clips:
                return
            self.clips = on
            self.pool.pool.set_clips_allowed(on)
            if self._held_until is None and not on and self._clips_from is None:
                self._clips_from = self._room or ''
            self._out.append({'cell': 0, 'event': 'clips', 'clips': on, 'clips_from': self._clips_from})

    def frame(self, since: int):
        return self.pool.pool.frame(0, since=since)

    def poll(self, now: float) -> Tuple[List[dict], List[Tuple[int, bytes, bool]]]:
        self.pool.pool.wait_idle(timeout=0)         # raises the env's failure
        finished = []
        with self._lock:
            for event in self.pool.poll_events():
                self._event(event, now)
            if self._held_until is not None and now >= self._held_until:
                self._held_until = None
                self._start()
                for message in self._held:
                    self._enter(message)
                self._held = []
                self.pool.pool.set_hold(0, False)       # the next run plays; its frames show from its first action
            for seed in [seed for seed in self._recordings if seed in self._results]:
                finished.append((seed, self._recordings.pop(seed), self._results.pop(seed)))
            messages, self._out = self._out, []
        return messages, finished

    def _event(self, event: dict, now: float) -> None:
        kind = event['type']
        if kind == 'replay':
            self._recordings[int(event['job'])] = event['data']
        elif kind == 'run':
            result = RunResult.from_dict(event['result'])
            outcome = 'error' if result.result == 'error' else (result.rooms[-1].outcome if result.rooms
                                                                 else result.result)
            self._out.append({'cell': 0, 'event': 'end', 'seed': result.seed, 'result': result.result,
                              'outcome': outcome, 'reason': result.reason})
            if result.seed is not None:
                self._results[int(result.seed)] = not without_clips(result)
            # An error before the first step (e.g. no model for the first room) would repeat: the cell stays on it.
            setup_error = result.result == 'error' and result.steps == 0
            self._held_until = float('inf') if setup_error else now + RESULT_PAUSE
        elif kind == 'room':
            name = room_names().get((event['level'], event['room'], event['in_cave']))
            message = {'cell': 0, 'event': 'room', 'specialist': event['specialist'], 'level': event['level'],
                       'room': event['room'], 'name': name, 'rooms': int(event['rooms']),
                       'cheated': {item: int(n) for item, n in event['cheated'].items()}}
            if self._held_until is not None:
                self._held.append(message)
            else:
                self._enter(message)

    def _enter(self, message: dict) -> None:
        """A room message of the run on screen, with the switch."""
        self._room = room_text(message['level'], message['room'])
        self._out.append({**message, 'clips': self.clips, 'clips_from': self._clips_from})

    def close(self) -> None:
        self.pool.close()


class _Cell:
    """What one cell is doing: waiting for a replay, playing one, or holding a finished one's result while the
    next one loads behind it."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.reset()

    def reset(self):
        self.show: Optional[Show] = None    # the replay on screen: playing, or its result held
        self.number = 0                     # its screen show number
        self.next: Optional[Show] = None    # the replay loading behind it (held at its start)
        self.next_number = 0
        self.phase = 'idle'                 # idle | playing | result
        self.until = 0.0                    # when a result's hold ends
        self.sent: Optional[dict] = None    # the last room message
        self.waiting = False                # a `wait` message is out
        self.error: Optional[str] = None    # why the next replay could not be had (shown once the hold ends)
        self.backoff = False                # after a failure: no new try until the hold ends


class ReplayGrid:
    """Plays replays in ``cells`` cells (module doc).  ``source``: one of ``SOURCES``, or ignored with ``path``
    (a replay file or a directory of them).  The simulations play ``route`` (``stop_at``: a room, ``stop_step``,
    ``start_from``, ``power_on``: ``Route.load``'s) on ``parallel`` games with the promoted models of
    ``models_dir`` acting as ``act`` says (``sample``, ``argmax``, ``model``).  ``replays``: a folder every
    simulated run is also saved into; ``streams``, ``state`` and
    ``keyframe_frames``: ``replay.record_options``'s.  ``controller``: draw the controller into every frame.
    ``clips``: the run-time clips switch the simulations start with (``set_clips`` changes it).  ``live``: cell 0 is
    a ``LiveCell`` (the models play as it is watched; not with ``path``)."""
    # pylint: disable=too-many-instance-attributes, too-many-public-methods

    def __init__(self, cells: int, fps: float, source: str = 'best', *, path=None, models_dir=None, states_dir=None,
                 route: str = ROUTE, parallel: int = 8, capacity: int = DEFAULT_CAPACITY,
                 patience: int = DEFAULT_PATIENCE, golden_capacity: Optional[int] = None, act: str = 'sample',
                 power_on: Optional[dict] = None, start_from: Optional[str] = None, stop_at=None,
                 stop_step: Optional[int] = None, stall_steps: Optional[int] = None, first_seed: int = 0,
                 attempts: int = golden.DEFAULT_ATTEMPTS, replays=None, streams=(), state: str = 'embed',
                 keyframe_frames: int = replay.DEFAULT_KEYFRAME_FRAMES, controller: bool = False,
                 clips: bool = True, live: bool = False):
        # pylint: disable=too-many-arguments, too-many-locals
        if path is None and source not in SOURCES:
            raise ValueError(f"unknown source {source!r}; sources: {SOURCES}")
        if live and path is not None:
            raise ValueError("a live cell plays the models: not on a grid of replay files")
        self.cells = cells
        self.fps = fps
        self.files = None if path is None else FileSource(path)
        self.models_dir = None if models_dir is None else str(models_dir)
        self.states_dir = states_dir
        self.route, self.parallel, self.capacity, self.patience = route, parallel, capacity, patience
        self.golden_capacity = max(2, cells) if golden_capacity is None else golden_capacity
        self.deterministic = registry.act_override(act)
        self.begin = {'power_on': power_on, 'start_from': start_from}      # where a simulated run starts
        self.stop_at, self.stop_step = stop_at, stop_step
        self.stall_steps, self.first_seed, self.attempts = stall_steps, first_seed, attempts
        self.replays = replays
        self.record = {'streams': streams, 'state': state, 'keyframe_frames': keyframe_frames}
        self.controller = controller
        self.live = live
        self._clips = clips
        self._live: Optional[LiveCell] = None
        self._sound: Optional[int] = None   # the cell that is heard
        self._sound_error: Optional[str] = None
        self._source = FILES if path is not None else source
        self._selection = 0                 # bumped by `select` and `skip`: every cell starts over
        self._status: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._screen = None
        self._policies: Optional[Policies] = None
        self._pool: Optional[ChainPool] = None
        self._feed = None
        self._golden: Optional[GoldenSource] = None
        self._failed: Dict[str, str] = {}   # source -> why its producer could not start
        self._paused_at: Optional[float] = None
        self._paused_total = 0.0
        self._thread = threading.Thread(target=self._run, name='play_grid', daemon=True)

    # ------------------------------------------------------------------ the caller's side

    def start(self) -> None:
        """Starts the cells and, on the status thread, whatever the source needs (models, simulations)."""
        self._thread.start()

    @property
    def source(self) -> str:
        """What the cells play: one of ``SOURCES``, or ``'files'``."""
        return self._source

    def select(self, source: str) -> None:
        """Every cell cuts to ``source`` (one of ``SOURCES``).  ``ValueError`` for another name, or on a grid
        that plays files."""
        if self.files is not None:
            raise ValueError("this grid plays replay files: it has no queues")
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}; sources: {SOURCES}")
        with self._lock:
            if source != self._source:
                self._source = source
                self._selection += 1

    def skip(self) -> None:
        """Every cell goes on to its next replay (a live cell plays on)."""
        with self._lock:
            self._selection += 1

    @property
    def clips(self) -> Optional[bool]:
        """The run-time clips switch of the runs simulated from now on (and of the live cell's run); None on a grid
        that plays files."""
        return None if self.files is not None else self._clips

    def set_clips(self, on: bool) -> None:
        """Turns the run-time clips switch on or off: for every run the simulations start from now on (the runs in
        the queues keep the switch they were played with) and, live, for the run on screen from its next decision
        (the route re-plans from where Link stands).  ``ValueError`` on a grid that plays files.  Golden paths
        always play with clips on."""
        if self.files is not None:
            raise ValueError("this grid plays replay files: it has no runs to switch clips for")
        with self._lock:
            self._clips = on
            if self._pool is not None:
                self._pool.pool.set_clips_allowed(on)
            if self._live is not None:
                self._live.set_clips(on)

    def frame(self, cell: int, since: int) -> Optional[Tuple[int, np.ndarray]]:
        """(frame number, RGB copy) of a cell's latest frame, or None when it is still frame ``since``."""
        if self.live and cell == 0:
            live = self._live
            return None if live is None else live.frame(since)
        screen = self._screen
        return None if screen is None else screen.frame(cell, since)

    def set_fps(self, fps: float) -> None:
        """Every cell's frame rate from its next frame on (``<= 0``: as fast as it plays)."""
        with self._lock:
            self.fps = fps
            if self._screen is not None:
                self._screen.set_fps(fps)
            if self._live is not None:
                self._live.pool.pool.set_fps(0, fps)

    def set_paused(self, on: bool) -> None:
        """Pauses (or resumes) every cell before its next frame, and the result clock with them."""
        with self._lock:
            now = time.monotonic()
            if on and self._paused_at is None:
                self._paused_at = now
            elif not on and self._paused_at is not None:
                self._paused_total += now - self._paused_at
                self._paused_at = None
            if self._screen is not None:
                self._screen.set_paused(on)
            if self._live is not None:
                self._live.pool.pool.set_paused(on)

    def set_sound(self, cell: Optional[int]) -> None:
        """Makes ``cell`` the one that is heard (None: none).  Without a usable sound device nothing is heard and
        ``stats()['sound_error']`` says why."""
        with self._lock:
            self._sound = cell
            self._apply_sound()

    def _apply_sound(self) -> None:
        """Tells the screen who is heard (the lock is held); a live cell has no sound."""
        if self._screen is None:
            return
        try:
            self._screen.set_sound(None if self.live and self._sound == 0 else self._sound)
            self._sound_error = None
        except RuntimeError as error:       # no sound device
            self._sound_error = str(error)

    @property
    def paused(self) -> bool:
        return self._paused_at is not None

    def _now(self) -> float:
        """Seconds of play: monotonic time minus every pause (frozen while paused)."""
        with self._lock:
            now = time.monotonic() if self._paused_at is None else self._paused_at
            return now - self._paused_total

    def emulated_frames(self) -> List[int]:
        """Frames each cell has played so far."""
        screen, live = self._screen, self._live
        played = [0] * self.cells if screen is None else screen.played()
        if self.live:
            played[0] = 0 if live is None else live.pool.pool.emulated_frames(0)
        return played

    def frame_counters(self) -> List[int]:
        """Frames each cell has shown (published) so far."""
        screen, live = self._screen, self._live
        counters = [0] * self.cells if screen is None else screen.frame_counters()
        if self.live:
            counters[0] = 0 if live is None else live.pool.pool.frame_counter(0)
        return counters

    def stats(self) -> dict:
        """``source``; ``feed`` (``ReplayFeed.stats()``: the queues' fill, the runs simulated and their endings,
        whether the simulations run) once the simulations started, else None; ``golden``
        (``GoldenSource.stats()``) once golden paths are being made, else None; ``files`` (how many, or None);
        ``sound_error`` (why no cell can be heard, else None); ``clips`` (the run-time clips switch, None for
        files); ``live`` (cell 0 is live)."""
        feed, paths = self._feed, self._golden
        return {'source': self._source, 'feed': None if feed is None else feed.stats(),
                'golden': None if paths is None else paths.stats(),
                'files': None if self.files is None else len(self.files.files), 'sound_error': self._sound_error,
                'clips': self.clips, 'live': self.live}

    def poll_status(self) -> List[dict]:
        """Every status message since the last call."""
        messages = []
        while True:
            try:
                messages.append(self._status.get_nowait())
            except queue.Empty:
                return messages

    def stop(self, timeout: float = 10.0) -> None:
        """Stops the cells and the simulations (idempotent)."""
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
            screen = _native.ReplayScreen(self.cells, self.fps,
                                          None if self.states_dir is None else str(self.states_dir))
        except Exception as error:  # pylint: disable=broad-exception-caught
            self._crash(range(self.cells), error)
            return
        with self._lock:
            self._screen = screen
            screen.set_fps(self.fps)                # in case these ran before the screen existed
            screen.set_paused(self._paused_at is not None)
            screen.set_controller(self.controller)
            self._apply_sound()
        cells = [_Cell() for _ in range(self.cells)]
        replayed = range(1 if self.live else 0, self.cells)    # the cells that play replays
        selection = -1
        try:
            while not self._stop.wait(POLL_SECONDS):
                with self._lock:
                    source, current = self._source, self._selection
                if current != selection:            # a new source, or a skip: every cell starts over
                    selection = current
                    for index in replayed:
                        screen.clear(index)
                        cells[index].reset()
                self._ensure(source)
                if self.live:
                    self._ensure_live()
                for index in replayed:
                    self._advance(screen, index, cells[index], source, cells)
                self._advance_live()
        except Exception as error:  # pylint: disable=broad-exception-caught
            self._crash(range(self.cells), error)
        finally:
            with self._lock:
                self._screen = None
                live, self._live = self._live, None
            screen.close()
            if live is not None:
                live.close()
            if self._golden is not None:
                self._golden.close()
            if self._feed is not None:
                self._feed.close()
            if self._pool is not None:
                self._pool.close()

    def _crash(self, cells, error) -> None:
        reason = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        for cell in cells:
            self._status.put({'cell': cell, 'event': 'crash', 'reason': reason})

    def _ensure(self, source: str) -> None:
        """Starts what ``source`` needs the first time it is selected: the models, then the simulations and their
        feed (a queue) or the golden-path thread.  A failure is kept and shown by every cell on that source."""
        if source == FILES or source in self._failed:
            return
        if (self._feed is not None) if source in QUEUES else (self._golden is not None):
            return
        try:
            if self._policies is None:
                self._policies = Policies(self.models_dir)
            policies = self._policies
            first = _native.route_steps(self.route)[0]['specialist']
            if first in policies.missing and self.begin['start_from'] is None:
                raise RuntimeError(policies.missing[first])     # every run would end before its first step
            route = Route.load(self.route, self.stop_at, states_dir=self.states_dir, stop_step=self.stop_step,
                               clips=self._clips, **self.begin)
            if source in QUEUES:
                record = replay.record_options(chain_meta(policies, self.deterministic, route), **self.record)
                seeds = range(self.first_seed, self.first_seed + SEEDS)
                self._pool = ChainPool(route, seeds, policies, num_envs=self.parallel, threads=self.parallel,
                                       deterministic=self.deterministic, states_dir=self.states_dir,
                                       stall_steps=self.stall_steps, record=record)
                with self._lock:                    # the key may have been pressed while the pool was built
                    self._pool.pool.set_clips_allowed(self._clips)
                self._pool.play()
                self._feed = _native.ReplayFeed(self._pool.pool, capacity=self.capacity, candidates=self.parallel,
                                                patience=self.patience,
                                                save_dir=None if self.replays is None else str(self.replays))
            else:
                whole = Route.load(self.route, states_dir=self.states_dir, **self.begin)
                record = replay.record_options(chain_meta(policies, self.deterministic, whole, golden=True),
                                               **self.record)
                self._golden = GoldenSource(policies, self.route, self.parallel, record, self.golden_capacity,
                                            self.first_seed, attempts=self.attempts, states_dir=self.states_dir,
                                            deterministic=self.deterministic,
                                            **self.begin)
        except Exception as error:  # pylint: disable=broad-exception-caught
            self._failed[source] = f"{type(error).__name__}: {error}"

    def _ensure_live(self) -> None:
        """Starts the live cell, after the simulations (its finished runs join their queues).  A failure stays in
        the cell."""
        if self._live is not None or LIVE in self._failed:
            return
        self._ensure(QUEUES[0])
        try:
            if QUEUES[0] in self._failed:
                raise RuntimeError(self._failed[QUEUES[0]])
            route = Route.load(self.route, self.stop_at, states_dir=self.states_dir, stop_step=self.stop_step,
                               clips=self._clips, **self.begin)
            record = replay.record_options(chain_meta(self._policies, self.deterministic, route), **self.record)
            # Seeds after the simulations' own: a replay saved with --replays never overwrites one of theirs.
            seeds = range(self.first_seed + SEEDS, self.first_seed + 2 * SEEDS)
            live = LiveCell(route, self._policies, seeds, self.fps, record, self._clips,
                            deterministic=self.deterministic, states_dir=self.states_dir,
                            stall_steps=self.stall_steps)
        except Exception as error:  # pylint: disable=broad-exception-caught
            self._failed[LIVE] = f"{type(error).__name__}: {error}"
            self._crash([0], self._failed[LIVE])
            return
        with self._lock:                            # what changed while the pool was built
            self._live = live
            live.pool.pool.set_fps(0, self.fps)
            live.pool.pool.set_paused(self._paused_at is not None)
            live.set_clips(self._clips)

    def _advance_live(self) -> None:
        """Passes the live cell's messages on and files its finished runs into the queues.  A failure (the env's,
        or a recording the feed can't read) stops the cell, with the reason on it."""
        live = self._live
        if live is None:
            return
        try:
            messages, finished = live.poll(self._now())
            for message in messages:
                self._status.put(message)
            for seed, data, clips in finished:
                self._feed.file(seed, data, clips)
        except Exception as error:  # pylint: disable=broad-exception-caught
            with self._lock:
                self._live = None
            live.close()
            self._failed[LIVE] = f"{type(error).__name__}: {error}"
            self._crash([0], self._failed[LIVE])

    def _next(self, source: str, cell: int, cells: List[_Cell]) -> Optional[Show]:
        """The next replay for ``cell`` from ``source``, or None when it has none yet.  Raises what makes the
        source unusable."""
        if source == FILES:
            if cell >= len(self.files.files):
                return None                 # more cells than files: this one stays dark
            busy = {show.name for other in cells for show in (other.show, other.next) if show is not None}
            return self.files.next(busy)
        if source in self._failed:
            raise RuntimeError(self._failed[source])
        if source == GOLDEN:
            if self._golden.error:
                raise RuntimeError(self._golden.error)
            return self._golden.next()
        entry = self._feed.pop(source)
        if entry is None:
            error = self._feed.stats()['error']
            if error:
                raise RuntimeError(error)
            return None
        return Show(entry['data'], source, clips=entry['clips'])

    def _advance(self, screen, index: int, cell: _Cell, source: str, cells: List[_Cell]) -> None:
        """Moves one cell along: reports the replay it plays; when that ends, holds its result ``RESULT_PAUSE``
        seconds while the next replay loads behind it (``ReplayScreen.play(hold=True)``), then lets that one
        play."""
        # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-return-statements
        # pylint: disable=too-many-branches
        now = self._now()
        if cell.phase == 'playing':
            state = screen.state(index)
            if state['show'] != cell.number or state['status'] in ('loading', 'ready'):
                return
            if state['status'] == 'failed':
                self._crash([index], state['error'])
                cell.phase, cell.until = 'result', now + RESULT_PAUSE
                return
            show = cell.show
            room = show.status(state['frame'])
            if room != cell.sent:
                cell.sent = room
                self._status.put({'cell': index, 'event': 'room', **room})
            if state['status'] != 'finished':
                return
            self._status.put({'cell': index, 'event': 'end', 'seed': show.seed, 'result': show.result,
                              'outcome': show.outcome, 'reason': show.reason})
            cell.phase, cell.until = 'result', now + RESULT_PAUSE
        # Idle, or a result is up: line up the next replay.
        if cell.next is None and cell.error is None and not (cell.backoff and now < cell.until):
            cell.backoff = False
            try:
                show = self._next(source, index, cells)
            except Exception as error:  # pylint: disable=broad-exception-caught
                cell.error = f"{type(error).__name__}: {error}"
            else:
                if show is not None:
                    cell.next = show
                    cell.next_number = screen.play(index, show.data, show.start, hold=True)
        if cell.phase == 'result' and now < cell.until:
            return
        if cell.error is not None:
            self._crash([index], cell.error)
            cell.show, cell.error = None, None
            cell.phase, cell.until, cell.backoff, cell.waiting = 'result', now + RESULT_PAUSE, True, False
            return
        if cell.next is None:
            if cell.phase != 'idle' or not cell.waiting:
                cell.show, cell.phase, cell.waiting = None, 'idle', True
                self._status.put({'cell': index, 'event': 'wait', 'source': source})
            return
        state = screen.state(index)
        if state['status'] == 'loading':
            return
        if state['status'] == 'failed':
            self._crash([index], state['error'])
            cell.show = cell.next = None
            cell.phase, cell.until, cell.backoff, cell.waiting = 'result', now + RESULT_PAUSE, True, False
            return
        screen.release(index)
        cell.show, cell.number, cell.next = cell.next, cell.next_number, None
        cell.phase, cell.sent, cell.waiting = 'playing', None, False
        self._status.put({'cell': index, 'event': 'start', 'seed': cell.show.seed, 'source': cell.show.source,
                          'name': cell.show.name, 'clips': cell.show.clips})
