"""Training event logs for live monitoring (an external "experiment watcher" GUI tails these files).

Layout, one folder per training invocation under ``logs/`` (repo root; ``$TRIFORCE_LOGS_DIR`` or ``--logs-dir``
override; gitignored)::

    logs/<yymmdd>_<NNN>_<what>/
        training.json        plan + status: every specialist run in order (pending / running / evaluating / final)
        <specialist>.jsonl   append-only event stream of one specialist's run
        <specialist>.log     its console output
        round.jsonl          run-level events (scripts/train_round.py only)

``NNN`` counts up per day (001, 002, ...).  ``training.json`` is rewritten atomically on every status change, so a
watcher can re-read it at any time.  Every JSONL line is one event object with:

- ``seq`` (0, 1, ... per file), ``ts`` (Unix seconds), ``time`` (ISO 8601 UTC), ``event``, plus the stream's
  context (``specialist``, ``run_id``) and event fields.

Events written by ``triforce.train`` (one specialist):

- ``start``: target_steps, target_iterations, steps_per_update, parallel, device, curriculum, run_dir, pid.
- ``progress``: after every PPO iteration (~4 s at 4,096 steps): iteration, iterations, steps, target_steps,
  fraction, elapsed, sps, eta, and the iteration's headline metrics (``metrics``; ``health_change`` is the mean
  hearts at exit minus hearts at entry, signed; ``health_change_total`` adds the hearts wasted by picking up dropped
  hearts at full health; ``hearts/dropped``, ``hearts/taken``, ``hearts/left``, ``hearts/wasted`` are dropped
  hearts per episode, see ``triforce/hearts.py``).
- ``checkpoint``: iteration, steps, path.
- ``early_stop``: iteration, steps, balanced_success_rate.
- ``error``: message, traceback.
- ``train_end``: status (completed | early_stop | interrupted | error), iterations, steps, wall_seconds, sps,
  checkpoint.

Added by ``scripts/train_round.py``: ``eval_start``, ``eval_progress`` (at most once a second: done, episodes,
running success_rate) and ``eval`` (specialist evaluation: success_rate, episodes, health, health_change,
health_change_total, hearts_dropped / hearts_taken / hearts_left per episode, by_room, by_state, path; it runs on a
background thread while the next specialist trains, so these can arrive after the next stream's ``start``) and
``error`` (non-zero exit, failed evaluation).  The last line of every stream is ``end`` (status, plus a summary);
nothing follows it.  A stream without ``end`` belongs to a run that is still going or was killed.
"""

import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOGS_DIR_ENV = 'TRIFORCE_LOGS_DIR'
PLAN_FILE = 'training.json'
SCHEMA_VERSION = 1

# Metrics copied into progress events (when present).
PROGRESS_METRICS = ('episodes', 'success_rate', 'balanced_success_rate', 'reward/total', 'health_change',
                    'health_change_total', 'hearts/dropped', 'hearts/taken', 'hearts/left', 'hearts/wasted',
                    'health_at_exit', 'episode_steps', 'curriculum/episodes', 'curriculum/success_rate',
                    'curriculum/point_fraction', 'losses/approx_kl', 'charts/SPS')


def logs_root(root: Optional[str | os.PathLike] = None) -> Path:
    """``root``, else ``$TRIFORCE_LOGS_DIR``, else ``<repo>/logs``."""
    if root is not None:
        return Path(root)
    if os.environ.get(LOGS_DIR_ENV):
        return Path(os.environ[LOGS_DIR_ENV])
    return Path(__file__).resolve().parents[1] / 'logs'


def new_log_dir(what: str, root=None, now: Optional[datetime] = None) -> Path:
    """Creates ``<root>/<yymmdd>_<NNN>_<what>`` with the next free NNN for that day."""
    base = logs_root(root)
    base.mkdir(parents=True, exist_ok=True)
    day = (now or datetime.now()).strftime('%y%m%d')
    safe = ''.join(c if c.isalnum() or c in '-_.' else '-' for c in what)
    while True:
        taken = [int(p.name.split('_')[1]) for p in base.glob(f'{day}_[0-9][0-9][0-9]_*') if p.is_dir()]
        path = base / f"{day}_{max(taken, default=0) + 1:03d}_{safe}"
        try:
            path.mkdir()
            return path
        except FileExistsError:      # another process took the number; try the next one
            continue


def _now() -> dict:
    ts = time.time()
    return {'ts': round(ts, 3), 'time': datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec='seconds')}


def _write_json(path: Path, data: dict):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, default=str)
    os.replace(tmp, path)


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
