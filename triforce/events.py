"""Training event logs for live monitoring (an external "experiment watcher" GUI tails these files).

Layout, one folder per training invocation under ``logs/`` (repo root; ``$TRIFORCE_LOGS_DIR`` or ``--logs-dir``
override; gitignored)::

    logs/<yymmdd>_<NNN>_<what>/
        training.json        plan + status: every specialist run in order (pending / running / evaluating / final)
        <specialist>.jsonl   append-only event stream of one specialist's run
        <specialist>.log     its console output
        round.jsonl          run-level events (scripts/train_round.py, triforce eval-chain)

``NNN`` counts up per day (001, 002, ...).  ``training.json`` is rewritten atomically on every status change, so a
watcher can re-read it at any time.  Every JSONL line is one event object with:

- ``seq`` (0, 1, ... per file), ``ts`` (Unix seconds), ``time`` (ISO 8601 UTC), ``event``, plus the stream's
  context (``specialist``, ``run_id``) and event fields.

Events written by ``triforce.train`` (one specialist):

- ``start``: target_steps, target_iterations, steps_per_update, parallel, device, curriculum, run_dir, pid.
- ``progress``: after every PPO iteration (~4 s at 4,096 steps): iteration, iterations, steps, target_steps,
  fraction, elapsed, sps, eta, and the iteration's headline metrics (``metrics``; ``health_change`` is the mean
  hearts at exit minus hearts at entry, signed; ``health_change_total`` adds the hearts wasted by picking up dropped
  hearts at full health; ``hearts/dropped``, ``hearts/taken``, ``hearts/healed``, ``hearts/left``,
  ``hearts/wasted`` are dropped hearts per episode, see ``crates/zelda/src/hearts.rs``; ``bombs/used`` are bombs placed
  per episode and ``cheated/bombs``, ``cheated/keys``, ``cheated/rupees`` cheated items used per episode
  (crates/zelda/src/cheats.rs); ``boss/stuns``, ``boss/eats`` are Dodongo stuns and swallowed bombs per episode;
  ``linger/max_steps``,
  ``linger/max_frames``, ``linger/tail_steps``, ``linger/tail_frames`` are means and ``linger/worst_steps``,
  ``linger/worst_frames`` maxima over measured episodes, also under ``curriculum/linger/*``).
- ``checkpoint``: iteration, steps, path.
- ``early_stop``: iteration, steps, balanced_success_rate.
- ``early_stop_check``: an early-stop proposal's confirmation evaluation (``train.EarlyStop``): iteration, rate
  (the evaluation's success), episodes, seed, balanced_success_rate, train_rates (per start state, training
  cohorts), states (per start state, evaluation), decision (stop | continue).
- ``error``: message, traceback.
- ``train_end``: status (completed | early_stop | interrupted | error), iterations, steps, wall_seconds, sps,
  checkpoint.

Added by ``scripts/train_round.py`` and ``triforce.evaluate``: ``eval_start``, ``eval_progress`` (at most once a
second: done, episodes, running success_rate) and ``eval`` (specialist evaluation, ``evaluate.eval_event``:
success_rate, episodes, deterministic, health, health_change, health_change_total, hearts_dropped / hearts_taken /
hearts_left per episode, flat ``linger_*`` means/maxima, by_room, by_room_failures, by_state, by_room_linger,
by_state_linger, path; in a round it runs on a
background thread while the next specialist trains, so these can arrive after the next stream's ``start``) and
``error`` (non-zero exit, failed evaluation), ``skipped`` (``--skip-existing`` and the run finished) and
``partial_set_aside`` (run_dir, train_status: an unfinished run renamed ``<run>.partial-N`` before training fresh).
The last line of every stream is ``end`` (status, plus a summary);
nothing follows it.  A stream without ``end`` belongs to a run that is still going or was killed.

Standalone jobs log too (``logged_job``): ``python -m triforce.evaluate`` writes ``<yymmdd>_<NNN>_eval-<name>``
(plan kind ``eval``, one run evaluating, its ``<specialist>.jsonl`` with the eval events above) and ``triforce
eval-chain`` writes ``<yymmdd>_<NNN>_eval-chain-<route>`` (plan kind ``chain``, no runs, ``round.jsonl`` with
``chain_start`` (runs, parallel, overrides, act), ``chain_progress`` (done, runs, successes, at most once a second
and on the last run) and ``chain`` (the summary)).  ``--no-log`` turns it off.
"""

import fcntl
import json
import os
import signal
import socket
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOGS_DIR_ENV = 'TRIFORCE_LOGS_DIR'
PLAN_FILE = 'training.json'
SCHEMA_VERSION = 1

# Metrics copied into progress events (when present).
PROGRESS_METRICS = ('episodes', 'success_rate', 'balanced_success_rate', 'reward/total', 'health_change',
                    'health_change_total', 'hearts/dropped', 'hearts/taken', 'hearts/healed', 'hearts/left',
                    'hearts/wasted', 'bombs/used', 'cheated/bombs', 'cheated/keys', 'cheated/rupees', 'boss/stuns',
                    'boss/eats',
                    'health_at_exit', 'episode_steps', 'curriculum/episodes', 'curriculum/success_rate',
                    'curriculum/point_fraction', 'losses/approx_kl', 'charts/SPS')
PROGRESS_METRICS += ('blocks', 'curriculum/blocks', 'success_rate_pessimistic', 'started/success_rate',
                     'started/episodes', 'started/iteration', 'in_flight/episodes', 'in_flight/mean_steps',
                     'in_flight/max_steps', 'in_flight/over_half_budget')
PROGRESS_METRICS += tuple(f'{prefix}/{key}' for prefix in ('linger', 'curriculum/linger')
                          for key in ('max_steps', 'max_frames', 'tail_steps', 'tail_frames',
                                      'worst_steps', 'worst_frames'))


def logs_root(root: Optional[str | os.PathLike] = None) -> Path:
    """``root``, else ``$TRIFORCE_LOGS_DIR``, else ``<repo>/logs``."""
    if root is not None:
        return Path(root)
    if os.environ.get(LOGS_DIR_ENV):
        return Path(os.environ[LOGS_DIR_ENV])
    return Path(__file__).resolve().parents[1] / 'logs'


def new_log_dir(what: str, root=None, now: Optional[datetime] = None) -> Path:
    """Creates ``<root>/<yymmdd>_<NNN>_<what>`` with the next free NNN for that day (any width: 999 is followed by
    1000).  The scan and the mkdir hold an exclusive lock on ``<root>/.lock``, so concurrent jobs get distinct NNN
    whatever their names."""
    base = logs_root(root)
    base.mkdir(parents=True, exist_ok=True)
    day = (now or datetime.now()).strftime('%y%m%d')
    safe = ''.join(c if c.isalnum() or c in '-_.' else '-' for c in what)
    with open(base / '.lock', 'a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        while True:
            taken = []
            for p in base.glob(f'{day}_*'):
                try:
                    taken.append(int(p.name.split('_')[1]))
                except (IndexError, ValueError):
                    continue
            path = base / f"{day}_{max(taken, default=0) + 1:03d}_{safe}"
            try:
                path.mkdir()
                return path
            except FileExistsError:      # a process without the lock took the number; the rescan sees it
                continue


def _now() -> dict:
    ts = time.time()
    return {'ts': round(ts, 3), 'time': datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec='seconds')}


def write_json_atomic(path, data, *, indent: Optional[int] = 2, default=None) -> Path:
    """Writes ``data`` as JSON to ``path`` through a temp file in the same directory and ``os.replace``: the file
    is either the previous one (or absent) or complete, never cut off.  A failed write leaves no temp file."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=indent, default=default)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _write_json(path: Path, data: dict):
    write_json_atomic(path, data, default=str)


class EventLog:
    """Append-only JSONL event stream; ``context`` fields go into every event."""

    def __init__(self, path: str | os.PathLike, **context):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.context = context
        self.seq = 0
        self._lock = threading.Lock()
        if self.path.exists():               # continue a stream another process started
            with open(self.path, encoding='utf-8') as file:
                self.seq = sum(1 for line in file if line.strip())

    def emit(self, event: str, **fields) -> dict:
        """Appends one event and flushes it (thread-safe)."""
        with self._lock:
            record = {'seq': self.seq, **_now(), 'event': event, **self.context, **fields}
            with open(self.path, 'a', encoding='utf-8') as file:
                file.write(json.dumps(record, default=str) + '\n')
            self.seq += 1
        return record


class TrainingPlan:
    """``training.json``: what this log folder is running, in order, and the status of each run."""

    def __init__(self, directory: str | os.PathLike, name: str, kind: str, runs: list[dict], **extra):
        self.directory = Path(directory)
        self.path = self.directory / PLAN_FILE
        self.data = {'schema': SCHEMA_VERSION, 'name': name, 'kind': kind, 'log_dir': str(self.directory),
                     'status': 'running', 'created': _now()['time'], 'updated': None, 'finished': None,
                     'host': socket.gethostname(), 'pid': os.getpid(), **extra,
                     'runs': [{'index': i, 'status': 'pending', 'started': None, 'finished': None, **run}
                              for i, run in enumerate(runs)]}
        self._lock = threading.RLock()
        self.save()

    def save(self):
        """Rewrites training.json atomically."""
        with self._lock:
            self.data['updated'] = _now()['time']
            _write_json(self.path, self.data)

    def update_run(self, index: int, **fields):
        """Sets fields on one run; status 'running' stamps started, a final status stamps finished (thread-safe)."""
        with self._lock:
            run = self.data['runs'][index]
            if fields.get('status') == 'running' and run['started'] is None:
                run['started'] = _now()['time']
            if fields.get('status') not in (None, 'pending', 'running', 'evaluating'):
                run['finished'] = _now()['time']
            run.update(fields)
            self.save()

    def finish(self, status: str, **fields):
        """Marks the whole plan finished (completed | failed | interrupted)."""
        with self._lock:
            self.data.update(status=status, finished=_now()['time'], **fields)
            self.save()


@contextmanager
def logged_job(what: str, kind: str, runs: list[dict], root=None, **extra):
    """A log folder (``new_log_dir(what, root)``) with its ``training.json`` plan for a job outside a training
    round, e.g. a standalone evaluation (``kind`` 'eval') or eval-chain ('chain'), so the training GUI shows it
    while it runs.  Yields the ``TrainingPlan``; the plan finishes ``completed``, ``interrupted`` (Ctrl-C, or
    SIGTERM, which the block turns into KeyboardInterrupt when it runs in the main thread) or ``failed`` with the
    block."""
    plan = TrainingPlan(new_log_dir(what, root), what, kind, runs, **extra)
    installed = threading.current_thread() is threading.main_thread()
    if installed:
        def interrupt(signum, _frame):
            raise KeyboardInterrupt(f"signal {signum}")
        previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield plan
    except KeyboardInterrupt:
        plan.finish('interrupted')
        raise
    except BaseException as error:
        plan.finish('failed', error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if installed:
            signal.signal(signal.SIGTERM, previous)
    plan.finish('completed')
