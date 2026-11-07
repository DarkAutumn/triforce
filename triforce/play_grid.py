"""Many live games at once for ``play-zelda.py``: one native chain pool (route ``milestone1`` with the promoted
specialists), one worker thread per game paced at ``fps`` frames per second, and a Python inference thread that
batches every waiting game's action per specialist on ``device``.

Shown games render every frame into the pool's frame buffers (``show(game, on)`` toggles a game; hidden games keep
playing at the same pace without rendering) and ``frame(game, since)`` copies a game's latest one.
``emulated_frames()`` counts every game's emulated frames, shown or not.  Room/health changes and run results go
through one status queue as dicts with ``game`` and ``event`` (``start``, ``room``, ``end``, ``crash``); ``room``
carries ``rooms``, the rooms entered so far in the run (a change of the observation's room id: level, cave, room),
and ``stalled``: with ``stall_steps`` set, the room has gone that many actions without damage dealt or taken (the
run ends ``stuck`` at twice as many).
After a run ends its game holds its result for ``RESULT_PAUSE`` seconds, showing the run's last frame (the next
run's reset publishes nothing until then), before the next seed plays (seeds are handed
out in order to whichever game is free, so ``start`` carries no seed; ``end`` does).  A setup error (an error before
the first step, e.g. a missing promoted model for the route's first room) stops the game instead of retrying in a
loop.
"""

import queue
import threading
import time
from typing import Dict, Iterable, List, Optional

import numpy as np

from . import _native
from .runner import ChainPool, Policies, Route, RunResult

FRAME_SHAPE = _native.FRAME_SHAPE   # rows, columns, RGB
ROUTE = 'milestone1'
RESULT_PAUSE = 3.0
SEEDS_PER_GAME = 1000
POLL_SECONDS = 0.02


def end_message(game: int, seed: Optional[int], result: RunResult) -> tuple[dict, bool]:
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


class _Held:
    """A game whose next run is ready but waits until ``until`` (the last result is on screen): its first
    observation and the status messages its new run produced meanwhile."""

    def __init__(self, until: float):
        self.until = until
        self.obs: Optional[Dict[str, np.ndarray]] = None
        self.keys: Optional[tuple] = None        # (policy, job, step) of ``obs``
        self.messages: List[dict] = []


class _Rooms:
    """One game's rooms entered this run.  A step's room event arrives before (or with) its row, so room messages
    wait in ``pending`` until the row's room id says how many rooms the run has entered; entering a room without a
    room event of its own (a cave: same level, room, specialist and health) repeats the last room message."""

    def __init__(self):
        self.count = 0
        self.room_id: Optional[int] = None
        self.last: Optional[dict] = None
        self.pending: List[dict] = []

    def new_run(self) -> None:
        self.count, self.room_id, self.last = 0, None, None

    def row(self, room_id: int) -> List[dict]:
        """The row of this game's latest step arrived: its room messages, stamped with ``rooms``."""
        entered = room_id != self.room_id
        if entered:
            self.room_id = room_id
            self.count += 1
        if entered and not self.pending and self.last is not None:
            self.pending.append(dict(self.last))
        return self.flush()

    def flush(self) -> List[dict]:
        messages, self.pending = self.pending, []
        for message in messages:
            message['rooms'] = self.count
        if messages:
            self.last = messages[-1]
        return messages


class GameGrid:
    """Plays ``games`` games of route ``milestone1`` in one chain pool and exposes their latest frames and status
    messages.  ``shown``: the games that render (default all)."""

    def __init__(self, games: int, device: str, fps: float, models_dir=None, shown: Optional[Iterable[int]] = None,
                 states_dir=None, stall_steps: Optional[int] = None):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.games = games
        self.device = device
        self.fps = fps
        self.models_dir = None if models_dir is None else str(models_dir)
        self.states_dir = states_dir
        self.stall_steps = stall_steps
        self._shown = set(range(games)) if shown is None else set(shown)
        self._emulated = [0] * games
        self._status: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pool = None
        self._thread = threading.Thread(target=self._run, name='play_grid', daemon=True)

    def start(self) -> None:
        """Loads the models and starts playing (on the inference thread)."""
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

    def emulated_frames(self) -> list[int]:
        """Frames each game has emulated so far (shown or not), counted as its steps finish."""
        return list(self._emulated)

    def poll_status(self) -> list[dict]:
        """Every status message received since the last call."""
        messages = []
        while True:
            try:
                messages.append(self._status.get_nowait())
            except queue.Empty:
                return messages

    def stop(self, timeout: float = 5.0) -> None:
        """Stops every game (idempotent): signal the inference thread, which closes the pool."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout)

    @property
    def stopped(self) -> bool:
        """The inference thread has exited."""
        return not self._thread.is_alive()

    # ------------------------------------------------------------------ inference thread

    def _run(self) -> None:
        try:
            route = Route.load(ROUTE)
            policies = Policies(self.models_dir, device=self.device)
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
                              states_dir=self.states_dir, stall_steps=self.stall_steps, hold_finished_frame=True)
        except Exception as err:  # pylint: disable=broad-exception-caught
            self._crash(err)
            return
        try:
            with self._lock:
                self._pool = chain.pool
                for game in self._shown:
                    chain.pool.set_render(game, True)
            for game in range(self.games):
                self._status.put({'game': game, 'event': 'start', 'seed': None})
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
        """Lockstep-free loop: act for every game that is ready, hold games showing a result, release them later."""
        held: Dict[int, _Held] = {}
        rooms = [_Rooms() for _ in range(self.games)]
        stopped = set()
        while not self._stop.is_set() and chain.pool.active:
            count = chain.recv(min_ready=1, timeout=POLL_SECONDS)
            for env, frames in zip(chain.buffers['env_ids'][:count], chain.buffers['frames'][:count]):
                self._emulated[int(env)] += int(frames)
            for event in chain.pool.poll_events():
                self._event(event, held, rooms, stopped)
            rows = chain.live_rows(count)
            ids = chain.buffers['env_ids'][rows]
            room_ids = chain.buffers['obs']['room_id'][rows, 0]
            act = []
            for row, env, room_id in zip(rows, ids, room_ids):
                env = int(env)
                if env in stopped:
                    continue
                messages = rooms[env].row(int(room_id))
                if env in held:
                    hold = held[env]
                    hold.obs = {key: value[row].copy() for key, value in chain.buffers['obs'].items()}
                    hold.keys = tuple(int(chain.buffers[k][row]) for k in ('policy', 'job', 'step'))
                    hold.messages += messages
                else:
                    for message in messages:
                        self._status.put(message)
                    act.append(row)
            if act:
                act = np.asarray(act)
                chain.send(chain.buffers['env_ids'][act], chain.actions(act))
            self._release(chain, held)

    def _event(self, event: dict, held: Dict[int, _Held], rooms: List[_Rooms], stopped: set) -> None:
        env = event['env']
        if event['type'] == 'run':
            for message in rooms[env].flush():      # the last step's room messages, before its result
                self._status.put(message)
            rooms[env].new_run()
            result = RunResult.from_dict(event['result'])
            message, setup_error = end_message(env, result.seed, result)
            self._status.put(message)
            if setup_error:
                stopped.add(env)
            else:
                held[env] = _Held(time.monotonic() + RESULT_PAUSE)
        elif event['type'] == 'room':
            rooms[env].pending.append({'game': env, 'event': 'room', 'specialist': event['specialist'],
                                       'level': event['level'], 'room': event['room'],
                                       'health': float(event['health']), 'max_health': int(event['max_health']),
                                       'stalled': bool(event['stalled'])})

    def _release(self, chain: ChainPool, held: Dict[int, _Held]) -> None:
        now = time.monotonic()
        due = [env for env, hold in held.items() if hold.obs is not None and hold.until <= now]
        if not due:
            return
        obs = {key: np.stack([held[env].obs[key] for env in due]) for key in held[due[0]].obs}
        policy, job, step = (np.array(column) for column in zip(*(held[env].keys for env in due)))
        actions = chain.inference.actions(np.asarray(due), obs, policy, job, step)
        for env in due:
            hold = held.pop(env)
            chain.pool.set_hold(env, False)         # the next run's frames show from its first action on
            self._status.put({'game': env, 'event': 'start', 'seed': None})
            for message in hold.messages:
                self._status.put(message)
        chain.send(np.asarray(due), actions)
