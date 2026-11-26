"""Each running job's CPU and its share of the machine's load, sampled from outside by the training GUI alone (the
owner, Nov 25: "Can you have each running task in the gui tell me how much cpu and load average it's taking up? It
doesn't need to update frequently, maybe every 2-5 sec."). A job is never asked anything and is never slowed: the
GUI's process reads what the kernel already keeps, every ``INTERVAL`` seconds.

- **Where a job lives.** A job's wrapper pid (its queue row's ``pid``) gives its cgroup once (``/proc/<pid>/cgroup``,
  the cgroup v2 line): a detached job's (``--detach``) is its systemd user unit ``tfjob-...service``, which holds the
  wrapper and everything its command started. A job started without ``--detach`` shares its caller's cgroup (a
  shell's, an agent's), so it is followed by its wrapper's process tree instead (``/proc/<pid>/task/*/children``).
- **CPU** (``Load.cores``): the cores it uses now, its cgroup's ``cpu.stat`` ``usage_usec`` between two samples over
  the time between them (a process tree: its live processes' ``utime + stime``); its share of the machine is that
  over the machine's threads (``os.cpu_count()``).
- **Load** (``Load.load``): what it adds to the load average. The kernel keeps no load average per job; this is the
  honest equivalent: its threads that are runnable now (running or waiting for a CPU: state R in
  ``/proc/<tid>/task/<tid>/stat`` for every thread of ``cgroup.threads``), smoothed as the kernel smooths its 1-minute
  average (``load = load * e + runnable * (1 - e)``, ``e = exp(-dt / 60)``). The kernel's own also counts threads in
  uninterruptible sleep (state D, mostly disk waits), which this leaves out.
- **Waiting** (``Load.waiting``): the share of the last 10 seconds in which some of its threads were ready to run and
  got no CPU (``cpu.pressure`` ``some avg10``, pressure stall information): a starved job, which its cores alone do
  not show. None for a process tree (no cgroup of its own).
- **The machine** (``Summary``): the running jobs' sums beside the machine's busy cores (``/proc/stat`` between the
  same two samples) and its 1-minute load average (a job that ended still adds its load, fading as the kernel's own
  average still holds it); what is left is outside the queue. While more than ``OUTSIDE_CORES`` cores are busy
  outside it for ``OUTSIDE_SECONDS``, ``warn`` is set, and the process outside every
  job that used the most CPU between two looks ``OUTSIDE_SCAN`` seconds apart is named (``top``; ``/proc/*/stat`` is
  read only while more than ``OUTSIDE_CORES`` are outside).

A unit that ends between two samples, a cgroup file that is missing, a pid that vanishes are all normal: the job's
last sample is kept (``Load.final``: its cgroup's total CPU time, when the last sample came within ``FINAL_SECONDS``
of its end) and nothing raises. Standard library only (the GUI imports it without the project's virtualenv).
"""

import math
import os
import re
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

INTERVAL = 3.0              # seconds between samples (the owner: "maybe every 2-5 sec")
LOAD_WINDOW = 60.0          # the smoothing's time constant: the kernel's 1-minute load average's
OUTSIDE_CORES = 3.0         # more cores than this busy outside the queue ...
OUTSIDE_SECONDS = 60.0      # ... for longer than this: the summary's warning
OUTSIDE_SCAN = 15.0         # while so, every process's CPU is read this often to name the busiest one outside
FINAL_SECONDS = 2 * INTERVAL + 2.0  # a job's last sample this close to its end gives its total CPU time
UNIT = re.compile(r'^tfjob-.*\.service$')
CLOCK_TICKS = os.sysconf('SC_CLK_TCK') if hasattr(os, 'sysconf') else 100
STATE_R = ord('R')


@dataclass(frozen=True)
class Load:
    """A job as last sampled: 'cores' (None before two samples), 'load' (its runnable threads, smoothed), 'runnable'
    (now), 'waiting' (``cpu.pressure`` some avg10, percent; None without), 'usage' (its cgroup's CPU seconds so far,
    None for a process tree), 'seen' (wall time of the sample), 'cgroup' (whether it has one of its own)."""
    cores: Optional[float]
    load: float
    runnable: int
    waiting: Optional[float]
    usage: Optional[float]
    seen: float
    cgroup: bool


@dataclass(frozen=True)
class Summary:
    """The machine beside its jobs: the running jobs' cores and load summed, the machine's busy cores (None before two
    samples) and 1-minute load average (None unread), its threads, what is outside the queue (None where a side is
    unknown), since when more than ``OUTSIDE_CORES`` have been (wall time, None while not), ``warn`` and ``top``
    (the busiest process outside every job: 'pid 123 (2.4 cores): python foo.py', '' while not looked for)."""
    jobs_cores: float
    jobs_load: float
    jobs: int
    machine_cores: Optional[float]
    machine_load: Optional[float]
    threads: int
    outside_cores: Optional[float]
    outside_load: Optional[float]
    outside_since: Optional[float]
    warn: bool
    top: str


def _read(path) -> Optional[bytes]:
    """A small kernel file's bytes, None when it can't be read (gone, missing, refused)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        return os.read(fd, 65536)
    except OSError:
        return None
    finally:
        os.close(fd)


def _ints(data: Optional[bytes]) -> list[int]:
    """The numbers of a list file (``cgroup.procs``, ``cgroup.threads``, ``children``)."""
    if not data:
        return []
    try:
        return [int(word) for word in data.split()]
    except ValueError:
        return []


def stat_fields(data: Optional[bytes]) -> Optional[list[bytes]]:
    """``/proc/.../stat``'s fields after the command (field 3, the state, first), None for no file. The command is
    in parentheses and may hold spaces or parentheses itself: everything after its last ``)``."""
    if not data:
        return None
    end = data.rfind(b')')
    return data[end + 2:].split() if end >= 0 else None


def usage_usec(cgroup: Path) -> Optional[int]:
    """A cgroup's CPU time so far (``cpu.stat`` ``usage_usec``), None without its file."""
    data = _read(cgroup / 'cpu.stat')
    if not data:
        return None
    for line in data.splitlines():
        if line.startswith(b'usage_usec '):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def pressure_some10(cgroup: Path) -> Optional[float]:
    """``cpu.pressure``'s ``some avg10`` (percent of the last 10 s some of its threads waited for a CPU), None
    without the file (no pressure stall information)."""
    data = _read(cgroup / 'cpu.pressure')
    if not data:
        return None
    for line in data.splitlines():
        if line.startswith(b'some '):
            for word in line.split()[1:]:
                if word.startswith(b'avg10='):
                    try:
                        return float(word[6:])
                    except ValueError:
                        return None
    return None


def cgroup_dirs(cgroup: Path) -> list[Path]:
    """A cgroup and the cgroups under it (a unit seldom has any)."""
    found, todo = [], [cgroup]
    while todo:
        path = todo.pop()
        found.append(path)
        try:
            with os.scandir(path) as entries:
                todo += [Path(entry.path) for entry in entries if entry.is_dir(follow_symlinks=False)]
        except OSError:
            continue
    return found


def runnable(proc: Path, tids) -> int:
    """How many of these threads are runnable now (state R: running or waiting for a CPU); a vanished one is not."""
    count = 0
    for tid in tids:
        data = _read(f"{proc}/{tid}/task/{tid}/stat")
        if data:
            end = data.rfind(b')')
            if end >= 0 and len(data) > end + 2 and data[end + 2] == STATE_R:
                count += 1
    return count


def process_ticks(proc: Path, pid: int) -> Optional[int]:
    """A process's CPU time so far in clock ticks (``utime + stime``, every thread of it), None once it is gone."""
    fields = stat_fields(_read(f"{proc}/{pid}/stat"))
    try:
        return int(fields[11]) + int(fields[12]) if fields else None
    except (IndexError, ValueError):
        return None


def machine_ticks(proc: Path) -> Optional[tuple[int, int]]:
    """(busy, total) clock ticks of every CPU so far (``/proc/stat``'s first line; idle and iowait are not busy)."""
    data = _read(proc / 'stat')
    if not data or not data.startswith(b'cpu '):
        return None
    try:
        fields = [int(value) for value in data.split(b'\n', 1)[0].split()[1:9]]
    except ValueError:
        return None
    if len(fields) < 5:
        return None
    total = sum(fields)
    return total - fields[3] - fields[4], total


def machine_load(proc: Path) -> Optional[float]:
    """The machine's 1-minute load average (``/proc/loadavg``), None unread."""
    data = _read(proc / 'loadavg')
    try:
        return float(data.split()[0]) if data else None
    except (IndexError, ValueError):
        return None


def command_of(proc: Path, pid: int, limit: int = 160) -> str:
    """A process's command line (its name in brackets without one: a kernel thread), cut at ``limit``."""
    data = _read(f"{proc}/{pid}/cmdline")
    text = data.replace(b'\0', b' ').strip().decode('utf-8', 'replace') if data else ''
    if not text:
        fields = _read(f"{proc}/{pid}/comm")
        text = f"[{fields.decode('utf-8', 'replace').strip()}]" if fields else '?'
    return text if len(text) <= limit else text[:limit - 1] + '…'


class Source:
    """Where a job's numbers come from: its cgroup, or (no cgroup of its own) its wrapper's process tree."""

    def __init__(self, proc: Path, cgroup: Optional[Path] = None, root: Optional[int] = None):
        self.proc, self.cgroup, self.root = proc, cgroup, root

    def tids(self) -> list[int]:
        if self.cgroup is not None:
            return [tid for path in cgroup_dirs(self.cgroup) for tid in _ints(_read(path / 'cgroup.threads'))]
        return [tid for pid in self.pids() for tid in self._tasks(pid)]

    def pids(self) -> list[int]:
        """Its processes now ([] once it is gone)."""
        if self.cgroup is not None:
            return [pid for path in cgroup_dirs(self.cgroup) for pid in _ints(_read(path / 'cgroup.procs'))]
        found, todo = [], [self.root]
        while todo:
            pid = todo.pop()
            tasks = self._tasks(pid)
            if not tasks:
                continue
            found.append(pid)
            for tid in tasks:
                todo += _ints(_read(f"{self.proc}/{pid}/task/{tid}/children"))
        return found

    def _tasks(self, pid: int) -> list[int]:
        try:
            return [int(name) for name in os.listdir(f"{self.proc}/{pid}/task")]
        except (OSError, ValueError):
            return []

    def usage(self) -> Optional[float]:
        """Its CPU seconds so far: the cgroup's ``usage_usec`` (None once it is gone); a tree's live processes'."""
        if self.cgroup is not None:
            value = usage_usec(self.cgroup)
            return None if value is None else value / 1e6
        pids = self.pids()
        if not pids:
            return None
        return sum(process_ticks(self.proc, pid) or 0 for pid in pids) / CLOCK_TICKS

    def waiting(self) -> Optional[float]:
        return pressure_some10(self.cgroup) if self.cgroup is not None else None


def source_of(pid: int, proc: Path, cgroup_root: Path) -> Optional[Source]:
    """The job's ``Source`` from its wrapper ``pid``: its systemd unit's cgroup (``tfjob-...service``, the cgroup v2
    line of ``/proc/<pid>/cgroup``), else its process tree; None while the pid can't be read."""
    data = _read(f"{proc}/{pid}/cgroup")
    if not data:
        return None
    for line in data.decode('utf-8', 'replace').splitlines():
        if line.startswith('0::'):
            path = line[3:].strip()
            if UNIT.match(path.rsplit('/', 1)[-1]):
                return Source(proc, cgroup=cgroup_root / path.lstrip('/'))
    return Source(proc, root=pid)


class JobLoads:
    """The running jobs' ``Load`` and the machine's ``Summary``, sampled at most every ``interval`` seconds however
    many callers ask (the window's timer and the web view's thread share one; a lock keeps them apart). ``proc``,
    ``cgroup_root``, ``clock`` (monotonic seconds), ``wall`` (time.time) and ``threads`` are the tests' to change."""

    def __init__(self, *, proc: Path = Path('/proc'), cgroup_root: Path = Path('/sys/fs/cgroup'),
                 clock=time.monotonic, wall=time.time, interval: float = INTERVAL, threads: Optional[int] = None,
                 host: Optional[str] = None):
        self.proc, self.cgroup_root = Path(proc), Path(cgroup_root)
        self.clock, self.wall, self.interval = clock, wall, interval
        self.threads = threads or os.cpu_count() or 1
        self.host = host if host is not None else socket.gethostname()
        self._lock = threading.Lock()
        self._sources: dict[int, Source] = {}
        self._loads: dict[int, Load] = {}
        self._last: dict[int, tuple[float, float]] = {}     # job -> (monotonic, usage) of its last sample
        self._machine: Optional[tuple[float, tuple[int, int]]] = None
        self._outside_ticks: dict[int, int] = {}
        self._fading: dict[int, tuple[float, float]] = {}   # an ended job -> (its load, monotonic) at its last sample
        self._at: Optional[float] = None
        self._summary: Optional[Summary] = None
        self.cost = 0.0             # CPU seconds this process spent sampling, all told (``time.thread_time``)
        self.samples = 0

    def get(self, job_id: int) -> Optional[Load]:
        """The job's last sample, None for one never sampled."""
        return self._loads.get(job_id)

    @property
    def summary(self) -> Optional[Summary]:
        return self._summary

    def sample(self, jobs: list[dict], force: bool = False) -> bool:
        """Samples the running jobs of ``jobs`` (queue rows of this host with a wrapper pid) when the last sample is
        ``interval`` old (a tenth early: two timers' ticks never skip one), or with ``force``; True when it did.
        Never raises for a job: what can't be read is left out of this sample."""
        with self._lock:
            tick = self.clock()
            if not force and self._at is not None and tick - self._at < self.interval * 0.9:
                return False
            started = time.thread_time()
            try:
                self._sample(jobs, tick)
            finally:
                self.cost += time.thread_time() - started
                self.samples += 1
            return True

    def _sample(self, jobs: list[dict], tick: float):
        wall = self.wall()
        running = {job['id']: job for job in jobs
                   if job.get('status') == 'running' and job.get('pid')
                   and (not job.get('host') or job.get('host') == self.host)}
        listed = {job['id'] for job in jobs}
        for job_id in list(self._loads):
            if job_id not in listed:            # the job left the list: forget it
                self._loads.pop(job_id, None)
                self._last.pop(job_id, None)
        for job_id in list(self._sources):
            if job_id not in running:
                self._end(job_id)
        for job_id, job in running.items():
            source = self._sources.get(job_id)
            if source is None:
                source = source_of(int(job['pid']), self.proc, self.cgroup_root)
                if source is None:
                    continue
                self._sources[job_id] = source
            self._sample_job(job_id, job, source, tick, wall)
        sources = [self._sources[job_id] for job_id in running if job_id in self._sources]
        self._summarize([self._loads[job_id] for job_id in running if job_id in self._sources], sources, tick, wall)
        self._at = tick

    def _end(self, job_id: int):
        """The job's unit ended (or it stopped running): no more samples; its load fades out of the jobs' sum as
        the kernel's 1-minute average forgets it (``_summarize``)."""
        self._sources.pop(job_id, None)
        last, load = self._last.pop(job_id, None), self._loads.get(job_id)
        if last is not None and load is not None:
            self._fading[job_id] = (load.load, last[0])

    def _sample_job(self, job_id: int, job: dict, source: Source, tick: float, wall: float):
        usage = source.usage()
        if usage is None:                       # the unit ended between two samples: its last sample stays
            self._end(job_id)
            return
        now_runnable = runnable(self.proc, source.tids())
        previous, last = self._loads.get(job_id), self._last.get(job_id)
        cores = None
        if last is not None and tick > last[0]:
            cores = max(0.0, (usage - last[1]) / (tick - last[0]))
        if previous is None:
            # First seen: as the kernel's average would be had it run this many threads since it started.
            admitted = job.get('admitted_ts')
            ran = max(0.0, wall - (wall if admitted is None else admitted))
            load = now_runnable * (1.0 - math.exp(-ran / LOAD_WINDOW))
        else:
            decay = math.exp(-max(0.0, tick - last[0]) / LOAD_WINDOW) if last is not None else 0.0
            load = previous.load * decay + now_runnable * (1.0 - decay)
        self._last[job_id] = (tick, usage)
        self._loads[job_id] = Load(cores=cores, load=load, runnable=now_runnable, waiting=source.waiting(),
                                   usage=usage if source.cgroup is not None else None, seen=wall,
                                   cgroup=source.cgroup is not None)

    def _summarize(self, loads: list[Load], sources: list[Source], tick: float, wall: float):
        """The machine's ``Summary`` beside ``loads``, the jobs sampled now.  A job that ended still adds its load,
        fading as the kernel's own 1-minute average still holds it, until it is under 0.01."""
        for job_id, (load, then) in list(self._fading.items()):
            if load * math.exp(-(tick - then) / LOAD_WINDOW) < 0.01:
                del self._fading[job_id]
        jobs_cores = sum(load.cores or 0.0 for load in loads)
        jobs_load = sum(load.load for load in loads) + sum(load * math.exp(-(tick - then) / LOAD_WINDOW)
                                                           for load, then in self._fading.values())
        ticks = machine_ticks(self.proc)
        machine_cores = None
        if ticks is not None and self._machine is not None:
            then, before = self._machine
            if tick > then and ticks[1] >= before[1]:
                machine_cores = max(0.0, (ticks[0] - before[0]) / CLOCK_TICKS / (tick - then))
                machine_cores = min(machine_cores, float(self.threads))
        if ticks is not None:
            self._machine = (tick, ticks)
        load1 = machine_load(self.proc)
        outside_cores = None if machine_cores is None else max(0.0, machine_cores - jobs_cores)
        outside_load = None if load1 is None else max(0.0, load1 - jobs_load)
        previous = self._summary
        since = previous.outside_since if previous is not None else None
        over = outside_cores is not None and outside_cores > OUTSIDE_CORES
        since = (since if since is not None else wall) if over else None
        last_scan = self._outside_ticks.get(-1)
        if over and (last_scan is None or tick - last_scan >= OUTSIDE_SCAN * 0.9):
            top = self._top_outside(sources, tick)
        else:
            top = (previous.top if previous is not None else '') if over else ''
        if not over:
            self._outside_ticks = {}
        self._summary = Summary(jobs_cores=jobs_cores, jobs_load=jobs_load, jobs=len(loads),
                                machine_cores=machine_cores, machine_load=load1, threads=self.threads,
                                outside_cores=outside_cores, outside_load=outside_load, outside_since=since,
                                warn=since is not None and wall - since >= OUTSIDE_SECONDS, top=top)

    def _top_outside(self, sources: list[Source], tick: float) -> str:
        """The process outside every job that used the most CPU since the last look ('' on the first look)."""
        inside = {pid for source in sources for pid in source.pids()}
        try:
            pids = [int(name) for name in os.listdir(self.proc) if name.isdigit()]
        except OSError:
            return ''
        before, now = self._outside_ticks, {}
        best, best_pid = 0, None
        for pid in pids:
            if pid in inside:
                continue
            ticks = process_ticks(self.proc, pid)
            if ticks is None:
                continue
            now[pid] = ticks
            used = ticks - before.get(pid, ticks)
            if used > best:
                best, best_pid = used, pid
        last_tick = before.get(-1)
        now[-1] = tick
        self._outside_ticks = now
        if best_pid is None or last_tick is None or tick <= last_tick:
            return self._summary.top if self._summary is not None else ''
        cores = best / CLOCK_TICKS / (tick - last_tick)
        return f"pid {best_pid} ({cores:.1f} cores): {command_of(self.proc, best_pid)}"


def final(load: Optional[Load], job: dict) -> Optional[tuple[float, float]]:
    """An ended job's (CPU seconds in all, mean cores over its run) from its cgroup's last sample, when that sample
    came within ``FINAL_SECONDS`` of its end; else None (a process tree's total isn't known, nor one not seen)."""
    if load is None or load.usage is None:
        return None
    started, ended = job.get('admitted_ts'), job.get('finished_ts')
    if started is None or ended is None or ended <= started or ended - load.seen > FINAL_SECONDS:
        return None
    return load.usage, load.usage / (ended - started)

