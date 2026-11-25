"""A training's steps a second on record, its normal and the SLOW mark (round40.md "Steps a second on record"; the
owner, Nov 25: "I want the resulting steps per second to be visible to the agent that started the training and waits
on it [...] total sps should also go into the logs/records").

- **The report.** Every training (``triforce train``, ``triforce.lab.train``) reports through ``jobs.speed`` after
  each iteration: its steps so far, the last iteration's steps a second, the run's mean, the rollout's alone, and its
  *shape* (``shape``: the network, the games played side by side, the look-ahead): the job's ``speed`` column.
- **The time series.** The job's wrapper (``triforce job run``) takes a sample every ``SAMPLE_SECONDS`` (``sample``):
  the time, the steps, the rates, the load average and the other jobs running then (their ids; how many of them were
  trainings), into ``speed_samples`` (``jobqueue.SCHEMA``).  ``rate`` is the steps between two samples over the time
  between them.
- **Normal.** A training's normal is its shape's, beside as many other trainings as run beside it then (``normal``):
  the median steps a second a game of the ended trainings of that shape (``NORMAL_RUNS`` of them or more, each with at
  least ``BUCKET_SECONDS`` of samples beside that many trainings), else ``REFERENCE`` (Perf43's measurements of
  Nov 25), else none (the line says so; nothing is judged).
- **SLOW.** A running training whose samples are more than ``SLOW_BELOW`` below their normal for more than
  ``SLOW_SECONDS`` in a row is SLOW (``judge``): ``triforce job wait`` and ``list`` say so with how far below, since
  when and what else runs, the training GUI's card in the warning colour, and the job's log gets a notice (the exit
  code is the command's).  The lead's standing instruction (AGENTS.md "The job queue"): an agent that sees SLOW
  reports it at once.
- **The end.** The last line the wrapper prints for a training (``final_line``): its mean steps a second, its normal
  and IN BOUNDS or OUT OF BOUNDS (how far below, for how long, what ran beside it); ``triforce job wait`` prints the
  same line last.  ``triforce job speed`` answers from the db alone, for any training past or running.
"""

import json
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import jobqueue as q

SAMPLE_SECONDS = 120.0      # one sample every two minutes a training (about 30 rows an hour: costs nothing)
SLOW_BELOW = 0.25           # SLOW: more than a quarter below the normal ...
SLOW_SECONDS = 600.0        # ... for more than ten minutes in a row
NORMAL_RUNS = 3             # ended trainings of a shape (beside as many trainings) before the record is the normal
BUCKET_SECONDS = 600.0      # a training counts toward a normal with at least this long beside that many trainings
GAP_SECONDS = 3 * SAMPLE_SECONDS    # two samples further apart than this are no interval (a wrapper that stalled)
MAX_TRAININGS = 3           # 3 or more other trainings are one bucket

# Steps a second a game before the record has ``NORMAL_RUNS`` ended trainings of a shape, by shape and the other
# trainings beside it (Perf43, Nov 25: round40.md "Where a teacher's time goes", the medians of the generation's
# stage-1 teachers, gen1-* s1 jobs #1258-#1377, each beside 1 or 2 other trainings; nothing was measured alone).
REFERENCE = {
    'hitnet-768, 8 games, look-ahead 4:0.25': {1: 66.9, 2: 52.9},
    'hitnet-768, 20 games, look-ahead 4:0.25': {1: 44.9, 2: 36.9},
}


# ------------------------------------------------------------------------------------------------ the shape

def network_name(config: Optional[dict]) -> str:
    """``hitnet-768``, ``zelda-256``: the architecture and its width (``model_config``)."""
    config = config or {}
    width = config.get('hidden_dim')
    return f"{config.get('arch') or 'zelda'}{'' if width is None else f'-{width}'}"


def shape(network: Optional[str], games: Optional[int], lookahead: Optional[str]) -> str:
    """A training's shape: ``hitnet-768, 20 games, look-ahead 4:0.25`` (``look-ahead off`` without one)."""
    return f"{network or '?'}, {games or '?'} games, look-ahead {lookahead or 'off'}"


def lookahead_text(tried: Optional[dict]) -> Optional[str]:
    """``EnvPool.rollout``'s ``tried`` settings as ``--try-others`` spells them (``4:0.25``), None without."""
    if not tried:
        return None
    return f"{tried['candidates']}:{float(tried['share']):g}"


# ------------------------------------------------------------------------------------------------ the samples

def read_load() -> Optional[float]:
    """The 1-minute load average (``$TRIFORCE_LOADAVG``'s file in tests), None where unreadable."""
    return q.read_load()


def samples(con, job_id: int) -> list[dict]:
    """A job's samples, oldest first, ``others`` decoded."""
    rows = [dict(row) for row in con.execute("SELECT * FROM speed_samples WHERE job = ? ORDER BY ts", (job_id,))]
    for row in rows:
        row['others'] = json.loads(row['others']) if row.get('others') else []
    return rows


def running_beside(con, job_id: int, now: float) -> list[dict]:
    """The other jobs running at ``now`` (by the record: started, not ended then)."""
    return [dict(row) for row in con.execute(
        "SELECT id, name, kind FROM jobs WHERE id != ? AND admitted_ts IS NOT NULL AND admitted_ts <= ? "
        "AND (finished_ts IS NULL OR finished_ts > ?) ORDER BY id", (job_id, now, now))]


def sample(con, job_id: int, now: Optional[float] = None, report: Optional[dict] = None,
           live: bool = True) -> Optional[dict]:
    """Appends the job's sample at ``now`` from its last report (``report``: the row's ``speed`` when not given);
    None (nothing written) for a job that reports no speed.  ``live``: the load average is read now (a backfill's
    past sample has none)."""
    now = time.time() if now is None else now
    if report is None:
        row = con.execute("SELECT speed FROM jobs WHERE id = ?", (job_id,)).fetchone()
        report = json.loads(row['speed']) if row and row['speed'] else None
    if not report:
        return None
    prev = con.execute("SELECT ts, steps FROM speed_samples WHERE job = ? ORDER BY ts DESC LIMIT 1",
                       (job_id,)).fetchone()
    steps = float(report.get('steps') or 0)
    rate = None
    if prev is not None and now - prev['ts'] > 0 and now - prev['ts'] <= GAP_SECONDS:
        rate = max(0.0, steps - prev['steps']) / (now - prev['ts'])
    others = running_beside(con, job_id, now)
    entry = {'job': job_id, 'ts': now, 'steps': steps, 'sps': report.get('sps'), 'mean': report.get('mean'),
             'rollout': report.get('rollout'), 'rate': rate, 'load': read_load() if live else None,
             'trainings': sum(1 for other in others if other.get('kind') == 'train'),
             'others': [other['id'] for other in others]}
    con.execute("INSERT INTO speed_samples (job, ts, steps, sps, mean, rollout, rate, load, trainings, others) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*(entry[key] for key in ('job', 'ts', 'steps', 'sps', 'mean', 'rollout', 'rate', 'load',
                                          'trainings')), json.dumps(entry['others'])))
    return entry


def intervals(rows: list[dict]) -> list[dict]:
    """The intervals between a job's consecutive samples (``rate`` set): {'start', 'end', 'steps', 'seconds',
    'rate', 'trainings', 'others', 'load'} (the later sample's beside-jobs and load)."""
    out = []
    for before, after in zip(rows, rows[1:]):
        if after.get('rate') is None:
            continue
        seconds = after['ts'] - before['ts']
        out.append({'start': before['ts'], 'end': after['ts'], 'seconds': seconds,
                    'steps': max(0.0, after['steps'] - before['steps']), 'rate': after['rate'],
                    'trainings': min(MAX_TRAININGS, after.get('trainings') or 0), 'others': after.get('others') or [],
                    'load': after.get('load')})
    return out


# ------------------------------------------------------------------------------------------------ the normal

@dataclass
class Normal:
    """A shape's normal beside ``trainings`` other trainings: ``per_game`` steps a second a game, from ``basis``
    (``record``: the median of ``runs`` ended trainings; ``reference``: ``REFERENCE``)."""
    per_game: float
    basis: str
    trainings: int
    runs: list = field(default_factory=list)

    def text(self, games: Optional[int]) -> str:
        total = '' if not games else f"{self.per_game * games:,.0f} steps/s, "
        where = (f"median of {len(self.runs)} ended runs" if self.basis == 'record'
                 else "Perf43's reference, Nov 25")
        return f"{total}{self.per_game:.1f} a game beside {trainings_text(self.trainings)} ({where})"


def trainings_text(count: int) -> str:
    return ('no other training' if count == 0 else '1 other training' if count == 1
            else f"{count}{'+' if count >= MAX_TRAININGS else ''} other trainings")


def job_rates(con, job: dict) -> dict[int, tuple[float, float]]:
    """A job's steps and seconds by the other trainings beside it (``intervals``)."""
    out: dict[int, list] = {}
    for item in intervals(samples(con, job['id'])):
        steps, seconds = out.setdefault(item['trainings'], [0.0, 0.0])
        out[item['trainings']] = [steps + item['steps'], seconds + item['seconds']]
    return {key: (value[0], value[1]) for key, value in out.items()}


class Normals:
    """The normals of one db (``normal``), each shape's ended trainings read once."""

    def __init__(self, con, exclude: Optional[int] = None):
        self.con = con
        self.exclude = exclude
        self._shapes: dict[str, list] = {}

    def record(self, shape_name: str) -> list[tuple[int, int, dict]]:
        """(job id, games, {trainings: (steps, seconds)}) of the ended trainings of ``shape_name``."""
        if shape_name not in self._shapes:
            rows = [q._row(row) for row in self.con.execute(   # pylint: disable=protected-access
                f"SELECT * FROM jobs WHERE status IN ({', '.join('?' * len(q.ENDED))}) AND speed IS NOT NULL "
                "AND json_extract(speed, '$.shape') = ?", (*q.ENDED, shape_name))]
            self._shapes[shape_name] = [(job['id'], job['speed'].get('games'), job_rates(self.con, job))
                                        for job in rows if job['id'] != self.exclude]
        return self._shapes[shape_name]

    def normal(self, shape_name: str, trainings: int) -> Optional[Normal]:
        """The normal of ``shape_name`` beside ``trainings`` other trainings: the median rate a game of the ended
        trainings with ``BUCKET_SECONDS`` or more beside as many (``NORMAL_RUNS`` of them or more), else
        ``REFERENCE``'s, else None."""
        trainings = min(MAX_TRAININGS, trainings)
        per_game = []
        for job_id, games, rates in self.record(shape_name):
            steps, seconds = rates.get(trainings, (0.0, 0.0))
            if games and seconds >= BUCKET_SECONDS:
                per_game.append((job_id, steps / seconds / games))
        if len(per_game) >= NORMAL_RUNS:
            return Normal(statistics.median(rate for _, rate in per_game), 'record', trainings,
                          [job_id for job_id, _ in per_game])
        reference = REFERENCE.get(shape_name, {}).get(trainings)
        return None if reference is None else Normal(reference, 'reference', trainings)


# ------------------------------------------------------------------------------------------------ the judgement

def names(con, ids: list) -> str:
    """The jobs ``ids`` for a line: the trainings by name, the rest by id and kind: ``#1376 lead-gen1-others-s1
    (train), #1372 lead-gen1-rooms-s1 (train) and 2 other jobs (#1398 eval, #1401 test)``; ``nothing else`` for
    none; ids alone without a connection (``con`` None)."""
    if not ids:
        return 'nothing else'
    if con is None:
        return ' '.join(f'#{i}' for i in ids)
    rows = {row['id']: dict(row) for row in con.execute(
        f"SELECT id, name, kind FROM jobs WHERE id IN ({', '.join('?' * len(ids))})", tuple(ids))}
    trainings = [q.label(rows[i]) for i in ids if i in rows and rows[i]['kind'] == 'train']
    rest = [f"#{i} {rows[i]['kind'] if i in rows else '?'}" for i in ids if i not in rows or rows[i]['kind'] != 'train']
    text = ', '.join(trainings) or 'no other training'
    return text + ('' if not rest else f" and {len(rest)} other job{'s' if len(rest) > 1 else ''} ({', '.join(rest)})")


def stretches(items: list[dict], normals: Normals, shape_name: str, games: int) -> list[dict]:
    """The runs of consecutive intervals more than ``SLOW_BELOW`` below their normal: {'start', 'end', 'steps',
    'seconds', 'expected' (steps a normal would have made), 'others' (every job beside it then), 'load'}."""
    out, current = [], None
    for item in items:
        normal = normals.normal(shape_name, item['trainings'])
        expected = None if normal is None else normal.per_game * games
        slow = expected is not None and item['rate'] < (1 - SLOW_BELOW) * expected
        if not slow:
            current = None
            continue
        if current is None or item['start'] > current['end'] + 1:
            current = {'start': item['start'], 'end': item['end'], 'steps': 0.0, 'seconds': 0.0, 'expected': 0.0,
                       'others': [], 'loads': []}
            out.append(current)
        current['end'] = item['end']
        current['steps'] += item['steps']
        current['seconds'] += item['seconds']
        current['expected'] += expected * item['seconds']
        current['others'] = sorted(set(current['others']) | set(item['others']))
        if item.get('load') is not None:
            current['loads'].append(item['load'])
    for stretch in out:
        stretch['below'] = 1 - stretch['steps'] / stretch['expected'] if stretch['expected'] else 0.0
        stretch['load'] = statistics.mean(stretch['loads']) if stretch['loads'] else None
    return out


def judge(con, job: dict, now: Optional[float] = None, normals: Optional[Normals] = None) -> Optional[dict]:
    """The job's mark from its samples (None for a job that reports no speed): {'state': 'ok' | 'slow' | 'no-normal',
    'normal' (``Normal.text`` beside the trainings of its last interval), 'rate' (its last ten minutes' steps a
    second), and while SLOW 'below', 'since', 'others' (every job beside it since then), 'load'; 'ts'}."""
    report = job.get('speed')
    if not report:
        return None
    now = time.time() if now is None else now
    normals = normals or Normals(con, exclude=job['id'])
    shape_name, games = report.get('shape'), report.get('games') or 0
    items = intervals(samples(con, job['id']))
    trainings = items[-1]['trainings'] if items else min(MAX_TRAININGS, sum(
        1 for other in running_beside(con, job['id'], now) if other.get('kind') == 'train'))
    normal = normals.normal(shape_name, trainings)
    recent = [item for item in items if item['end'] > (items[-1]['end'] if items else now) - SLOW_SECONDS]
    seconds = sum(item['seconds'] for item in recent)
    mark = {'state': 'no-normal' if normal is None else 'ok', 'ts': now, 'trainings': trainings,
            'normal': None if normal is None else normal.text(games),
            'normal_sps': None if normal is None else normal.per_game * games,
            'rate': sum(item['steps'] for item in recent) / seconds if seconds else None}
    slow = stretches(items, normals, shape_name, games)
    last = slow[-1] if slow else None
    if last is not None and items and last['end'] >= items[-1]['end'] and last['end'] - last['start'] > SLOW_SECONDS:
        mark.update(state='slow', below=last['below'], since=last['start'], others=last['others'],
                    load=last['load'], slow_rate=last['steps'] / last['seconds'] if last['seconds'] else 0.0)
    return mark


def verdict(con, job: dict) -> Optional[dict]:
    """A training's judgement over its whole run (None for a job that reports no speed): {'bounds': 'in' | 'out' |
    'no normal', 'mean' (steps a second over its samples, else its own report), 'expected' (its normal over the
    same intervals, None without one), 'slow' (its SLOW stretches, ``stretches``, longer than ``SLOW_SECONDS``)}.
    OUT OF BOUNDS: SLOW at some time, or its mean more than ``SLOW_BELOW`` below its normal."""
    report = job.get('speed')
    if not report:
        return None
    normals = Normals(con, exclude=job['id'])
    shape_name, games = report.get('shape'), report.get('games') or 0
    items = intervals(samples(con, job['id']))
    steps = sum(item['steps'] for item in items)
    seconds = sum(item['seconds'] for item in items)
    mean = steps / seconds if seconds else report.get('mean')
    expected_steps, judged = 0.0, 0.0
    for item in items:
        normal = normals.normal(shape_name, item['trainings'])
        if normal is not None:
            expected_steps += normal.per_game * games * item['seconds']
            judged += item['seconds']
    if not items:
        trainings = 0
        normal = normals.normal(shape_name, trainings)
        expected = None if normal is None else normal.per_game * games
    else:
        expected = expected_steps / judged if judged else None
    slow = [s for s in stretches(items, normals, shape_name, games) if s['end'] - s['start'] > SLOW_SECONDS]
    judged_mean = (sum(item['steps'] for item in items if normals.normal(shape_name, item['trainings']))
                   / judged) if judged else mean
    if expected is None:
        bounds = 'no normal'
    elif slow or (judged_mean is not None and judged_mean < (1 - SLOW_BELOW) * expected):
        bounds = 'out'
    else:
        bounds = 'in'
    beside = sorted({other for item in items for other in item['others']})
    return {'bounds': bounds, 'mean': mean, 'expected': expected, 'slow': slow, 'games': games, 'shape': shape_name,
            'seconds': seconds, 'beside': beside,
            'trainings': statistics.mean(item['trainings'] for item in items) if items else None}


# ------------------------------------------------------------------------------------------------ the texts

def clock(ts: Optional[float]) -> str:
    return '?' if ts is None else time.strftime('%H:%M', time.localtime(ts))


def per_game(value: Optional[float], games: Optional[int]) -> str:
    return '' if value is None or not games else f" ({value / games:.1f} a game)"


def sps(value: Optional[float]) -> str:
    return '?' if value is None else f"{value:,.0f}"


def speed_line(job: dict) -> Optional[str]:
    """A running training's speed (``triforce job list``, ``wait``): ``speed 957 steps/s mean (47.9 a game), last
    iteration 1,010, rollout 1,020; normal 1,150 steps/s, 57.5 a game beside 1 other training (median of 3 ended
    runs): ok``; None without a report."""
    report = job.get('speed')
    if not report:
        return None
    games = report.get('games')
    mark = job.get('speed_mark') or {}
    state = {'ok': 'ok', 'slow': 'SLOW', 'no-normal': 'no normal yet'}.get(mark.get('state'), 'not judged yet')
    normal = f"; normal {mark['normal']}" if mark.get('normal') else ''
    recent = '' if mark.get('rate') is None else f", last 10 minutes {sps(mark['rate'])}"
    return (f"speed {sps(report.get('mean'))} steps/s mean{per_game(report.get('mean'), games)}, last iteration "
            f"{sps(report.get('sps'))}, rollout {sps(report.get('rollout'))}{recent} [{report.get('shape')}]"
            f"{normal}: {state}")


def slow_line(con, job: dict) -> Optional[str]:
    """``SLOW: 31% below its normal (1,010 steps/s against 1,460) for 14 minutes since 09:12, beside #1376 ...; load
    27.1``; None unless the job's mark is SLOW."""
    mark = job.get('speed_mark') or {}
    if mark.get('state') != 'slow':
        return None
    minutes = (mark.get('ts', time.time()) - mark['since']) / 60
    load = '' if mark.get('load') is None else f"; load {mark['load']:.1f}"
    return (f"SLOW: {mark['below']:.0%} below its normal ({sps(mark.get('slow_rate'))} steps/s against "
            f"{sps(mark.get('normal_sps'))}) for {minutes:.0f} minutes since {clock(mark['since'])}, beside "
            f"{names(con, mark.get('others') or [])}{load}")


def final_line(con, job: dict) -> Optional[str]:
    """The one line a training's end prints last (the wrapper, ``triforce job wait``): ``#1371 lead-gen1-darknut-s1
    (train) speed: 957 steps/s mean (47.9 a game) [hitnet-768, 20 games, look-ahead 4:0.25], normal 1,150 (57.5 a
    game): IN BOUNDS`` or ``... OUT OF BOUNDS: 31% below for 42 minutes (09:12-09:54) beside ...``; None for a
    job that reports no speed."""
    found = verdict(con, job)
    if found is None:
        return None
    games = found['games']
    normal = ('no normal yet (fewer than 3 ended runs of this shape and no reference)' if found['expected'] is None
              else f"normal {sps(found['expected'])}{per_game(found['expected'], games)}")
    beside = (f"; beside it: {names(con, found['beside'])}" if found['beside'] else '; nothing else ran beside it')
    head = (f"{q.label(job)} {job.get('status') or ''} speed: {sps(found['mean'])} steps/s mean"
            f"{per_game(found['mean'], games)} [{found['shape']}], {normal}")
    if found['bounds'] == 'no normal':
        return f"{head}: NO NORMAL{beside}"
    if found['bounds'] == 'in':
        return f"{head}: IN BOUNDS{beside}"
    if found['slow']:
        worst = max(found['slow'], key=lambda s: s['end'] - s['start'])
        total = sum(s['end'] - s['start'] for s in found['slow'])
        load = '' if worst['load'] is None else f", load {worst['load']:.1f}"
        minutes = (worst['end'] - worst['start']) / 60
        return (f"{head}: OUT OF BOUNDS: {worst['below']:.0%} below its normal for {minutes:.0f} minutes "
                f"({clock(worst['start'])}-{clock(worst['end'])}; {total / 60:.0f} minutes slow in all), "
                f"beside {names(con, worst['others'])}{load}")
    below = 1 - found['mean'] / found['expected'] if found['expected'] else 0.0
    return f"{head}: OUT OF BOUNDS: its mean {below:.0%} below its normal{beside}"


# ------------------------------------------------------------------------------------------------ the watch

class Watch:
    """The wrapper's side (``job.run_job``): every ``SAMPLE_SECONDS`` a sample of the job's report, its mark into the
    row, and a notice in the job's log when it turns SLOW or back.  Runs on its own thread and connection; any
    failure of the queue's db is skipped (the job goes on)."""

    def __init__(self, job_id: int, say, directory: Optional[Path] = None, interval: Optional[float] = None):
        self.job_id = job_id
        self.say = say
        self.directory = directory
        self.interval = SAMPLE_SECONDS if interval is None else interval
        self.state: Optional[str] = None

    def tick(self, now: Optional[float] = None) -> Optional[dict]:
        """One sample and judgement; returns the mark."""
        con = q.connect(self.directory)
        try:
            with q.transaction(con):
                if sample(con, self.job_id, now) is None:
                    return None
                job = q.read_jobs(con, 'id = ?', (self.job_id,))[0]
                mark = judge(con, job, now)
                q.update(con, self.job_id, speed_mark=mark)
            job['speed_mark'] = mark
            if mark['state'] == 'slow' and self.state != 'slow':
                self.say(f"#{self.job_id} {slow_line(con, job)} (a notice: the job goes on)")
            elif mark['state'] != 'slow' and self.state == 'slow':
                self.say(f"#{self.job_id} speed back within bounds: {speed_line(job)}")
            self.state = mark['state']
            return mark
        finally:
            con.close()

    def run(self, stop):
        """Ticks until ``stop`` (a ``threading.Event``) is set."""
        while not stop.wait(self.interval):
            try:
                self.tick()
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    def final(self) -> Optional[str]:
        """The last sample, then ``final_line`` (None for a job that reported no speed)."""
        con = q.connect(self.directory)
        try:
            with q.transaction(con):
                sample(con, self.job_id)
            job = q.read_jobs(con, 'id = ?', (self.job_id,))[0]
            return final_line(con, job)
        finally:
            con.close()


# ------------------------------------------------------------------------------------------------ the backfill

RUN_LINE = re.compile(r'^\[(?:lab|train)\] \S+ run=\S+ dir=(\S+)')


def run_dir_of(log: Path) -> Optional[Path]:
    """The run folder a training's log names (``[lab] NAME run=ID dir=DIR ...``, ``[train] NAME run=ID dir=DIR``)."""
    try:
        with open(log, encoding='utf-8', errors='replace') as file:
            for line in file:
                found = RUN_LINE.match(line)
                if found:
                    return Path(found.group(1))
    except OSError:
        return None
    return None


def run_shape(run: dict) -> tuple[str, int, Optional[str]]:
    """(network, games, look-ahead) of a run's ``run.json`` (the lab's or ``triforce train``'s)."""
    envs = run.get('envs')
    games = sum(envs.values()) if isinstance(envs, dict) else (run.get('args') or {}).get('parallel')
    tried = ((run.get('manifest') or {}).get('training') or {}).get('try_others')
    if tried is None:
        text = (run.get('args') or {}).get('try_others')
        tried = None if not text else {'candidates': text.split(':')[0], 'share': text.split(':')[1] or 1}
    return network_name(run.get('model_config')), games, lookahead_text(tried)


def logs_of(job: dict, directory: Path) -> list[Path]:
    """The detached logs of ``job`` (``triforce job run --detach``'s ``<unit>.log``) that name it."""
    label = '-'.join(str(part) for part in (job.get('agent'), job.get('task'), job.get('subtask')) if part)
    stem = re.sub(r'[^A-Za-z0-9_.]+', '-', label)[:80]
    found = []
    for log in sorted(directory.glob(f'tfjob-{stem}-*.log')):
        try:
            with open(log, encoding='utf-8', errors='replace') as file:
                if any(f"#{job['id']} " in line for line in file):
                    found.append(log)
        except OSError:
            continue
    return found


def backfill(con, directory: Path, ids: Optional[list] = None, say=print) -> int:
    """Samples for trainings that ran before the record (``triforce job speed --backfill``): each training job
    without a live report whose detached log names its run folder gets its ``speed`` report and a sample every
    ``SAMPLE_SECONDS`` from the run's ``metrics.jsonl`` (``time`` and ``steps`` an iteration, ``charts/SPS``,
    ``charts/rollout_SPS``) within the job's start and end; the load average is unknown (None), the jobs beside it
    are the record's.  Run again, it adds what a training of older code ran since.  Returns how many jobs it
    filled."""
    filled = 0
    where = "kind = 'train' AND admitted_ts IS NOT NULL"
    for job in q.read_jobs(con, where):
        if ids and job['id'] not in ids:
            continue
        have = con.execute("SELECT max(ts) FROM speed_samples WHERE job = ?", (job['id'],)).fetchone()[0]
        if have is not None and not (job.get('speed') or {}).get('backfilled'):
            continue        # its wrapper records it live
        run_dir = next((found for found in map(run_dir_of, logs_of(job, directory)) if found), None)
        if run_dir is None or not (run_dir / 'metrics.jsonl').is_file() or not (run_dir / 'run.json').is_file():
            continue
        network, games, lookahead = run_shape(json.loads((run_dir / 'run.json').read_text(encoding='utf-8')))
        end = job.get('finished_ts') or time.time()
        rows = []
        with open(run_dir / 'metrics.jsonl', encoding='utf-8') as file:
            for line in file:
                metrics = json.loads(line)
                if job['admitted_ts'] <= metrics.get('time', 0) <= end:
                    rows.append(metrics)
        if len(rows) < 2:
            continue
        base = rows[0]['steps'] - rows[0].get('rollout/steps', 0)
        last_ts, prev, last = None, None, {}
        for metrics in rows:
            # The iteration's own steps a second: its rows over the time since the iteration before it.
            seconds = None if prev is None else metrics['time'] - prev['time']
            last = {'steps': metrics['steps'] - base, 'mean': metrics.get('charts/SPS'),
                    'sps': metrics.get('rollout/steps', 0) / seconds if seconds else None,
                    'rollout': metrics.get('charts/rollout_SPS')}
            prev = metrics
            if last_ts is not None and metrics['time'] - last_ts < SAMPLE_SECONDS and metrics is not rows[-1]:
                continue
            if have is None or metrics['time'] > have:
                sample(con, job['id'], metrics['time'], last, live=False)
            last_ts = metrics['time']
        q.update(con, job['id'], speed={
            **last, 'total': None, 'games': games, 'network': network, 'lookahead': lookahead,
            'shape': shape(network, games, lookahead), 'ts': rows[-1]['time'], 'backfilled': str(run_dir)})
        say(f"#{job['id']} {job.get('name')}: {len(rows)} iterations of {run_dir} -> samples")
        filled += 1
    return filled


def detached_dir() -> Path:
    return q.queue_dir() / 'detached'


def environment_note() -> str:
    """Where the numbers come from, for ``triforce job speed``'s header."""
    return f"the queue's db {q.queue_dir() / q.DB_NAME}" + (f" (${q.QUEUE_DIR_ENV} set)" if os.environ.get(
        q.QUEUE_DIR_ENV) else '')
