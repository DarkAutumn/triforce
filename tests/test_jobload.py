"""Each running job's CPU and load (``triforce.jobload``), sampled from a fake cgroup tree and a fake ``/proc`` (a tmp
directory): cores from two ``usage_usec`` samples, runnable threads counted and smoothed as the kernel smooths its
load average, the waiting share, a unit that vanishes, a job without a unit of its own, the machine's line; and the
sampling's cost against real processes."""

import math
import os
import subprocess
import sys
import time
from pathlib import Path

from triforce import jobload

APP = 'user.slice/user-1000.slice/user@1000.service/app.slice'
TICKS = jobload.CLOCK_TICKS


class Clock:
    """A clock the test moves (monotonic seconds, or wall time)."""

    def __init__(self, now: float):
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeMachine:
    """A cgroup v2 tree and a ``/proc`` under a tmp directory: units (``tfjob-...service``) with their processes and
    threads, a session scope for jobs without a unit, ``/proc/stat`` and ``/proc/loadavg``."""

    def __init__(self, root: Path):
        self.proc, self.cgroups = root / 'proc', root / 'cgroup'
        self.proc.mkdir(parents=True)
        self.cgroups.mkdir(parents=True)
        self.machine(busy=0, idle=0)
        self.loadavg(1.0)

    def unit(self, name: str, threads: dict[int, dict[int, str]], usage_usec: int = 0, pressure=None,
             cgroup: str | None = None) -> Path:
        """A unit's cgroup (``cgroup``: another path, e.g. a shell's scope) holding ``threads``: {pid: {tid:
        state}}; its cpu.stat's usage and its cpu.pressure's some avg10 (None: no file)."""
        path = cgroup or f"{APP}/{name}"
        directory = self.cgroups / path
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'cgroup.procs').write_text(''.join(f"{pid}\n" for pid in threads))
        (directory / 'cgroup.threads').write_text(''.join(f"{tid}\n" for tids in threads.values() for tid in tids))
        self.usage(directory, usage_usec)
        if pressure is not None:
            (directory / 'cpu.pressure').write_text(f"some avg10={pressure:.2f} avg60=1.00 avg300=1.00 total=5\n"
                                                    "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
        for pid, tids in threads.items():
            self.process(pid, tids, path)
        return directory

    def process(self, pid: int, tids: dict[int, str], cgroup: str, ticks: int = 0, children=()):
        (self.proc / str(pid) / 'task').mkdir(parents=True, exist_ok=True)
        (self.proc / str(pid) / 'cgroup').write_text(f"1:net_cls:/\n0::/{cgroup}\n")
        (self.proc / str(pid) / 'cmdline').write_bytes(f"python -m job{pid}\0--x\0".encode())
        self.ticks(pid, ticks)
        for tid, state in tids.items():
            self.state(pid, tid, state)
            (self.proc / str(pid) / 'task' / str(tid) / 'children').write_text(''.join(f"{c} " for c in children)
                                                                              if tid == pid else '')

    def stat(self, pid: int, state: str, ticks: int = 0) -> str:
        return (f"{pid} (python (a) b) {state} 1 {pid} {pid} 0 -1 4194304 10 0 0 0 {ticks} 0 0 0 20 0 1 0 "
                "100 2000 300\n")

    def state(self, pid: int, tid: int, state: str):
        """A thread's state, in both places it can be read (``/proc/<pid>/task/<tid>`` and ``/proc/<tid>/task/<tid>``,
        the one ``jobload`` reads)."""
        for owner in {pid, tid}:
            folder = self.proc / str(owner) / 'task' / str(tid)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'stat').write_text(self.stat(tid, state))

    def ticks(self, pid: int, ticks: int):
        (self.proc / str(pid) / 'stat').write_text(self.stat(pid, 'S', ticks))

    def usage(self, directory: Path, usec: int):
        (directory / 'cpu.stat').write_text(f"usage_usec {usec}\nuser_usec {usec}\nsystem_usec 0\n")

    def vanish(self, directory: Path, pids=()):
        """A unit that ended: its cgroup and its processes gone."""
        for path in sorted(directory.rglob('*'), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        directory.rmdir()
        for pid in pids:
            for path in sorted((self.proc / str(pid)).rglob('*'), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
            (self.proc / str(pid)).rmdir()

    def machine(self, busy: int, idle: int):
        """``/proc/stat``'s cpu line: ``busy`` ticks of user time, ``idle`` of idle."""
        (self.proc / 'stat').write_text(f"cpu  {busy} 0 0 {idle} 0 0 0 0 0 0\ncpu0 0 0 0 0 0 0 0 0 0 0\n")

    def loadavg(self, value: float):
        (self.proc / 'loadavg').write_text(f"{value:.2f} 1.00 1.00 2/900 4242\n")


def loads_of(machine: FakeMachine, clock: Clock, wall: Clock, threads: int = 20) -> jobload.JobLoads:
    return jobload.JobLoads(proc=machine.proc, cgroup_root=machine.cgroups, clock=clock, wall=wall, threads=threads,
                            host='here')


def running(job_id: int, pid: int, admitted: float, **fields) -> dict:
    return {'id': job_id, 'status': 'running', 'pid': pid, 'host': 'here', 'admitted_ts': admitted, **fields}


def test_cores_from_two_usage_samples(tmp_path):
    """A job's CPU is its cgroup's ``usage_usec`` between two samples over the time between them: none on the first
    sample, 2.5 cores after 7.5 core-seconds in 3 seconds; a job of another host or not running is not sampled."""
    machine = FakeMachine(tmp_path)
    unit = machine.unit('tfjob-a-1.service', {100: {100: 'S'}}, usage_usec=50_000_000)
    clock, wall = Clock(1000.0), Clock(5000.0)
    loads = loads_of(machine, clock, wall)
    jobs = [running(1, 100, 4000.0), running(2, 100, 4000.0, host='elsewhere'),
            {'id': 3, 'status': 'queued', 'pid': 100}]
    assert loads.sample(jobs)
    first = loads.get(1)
    assert first.cores is None and first.usage == 50.0 and first.cgroup and first.seen == 5000.0
    assert loads.get(2) is None and loads.get(3) is None
    assert not loads.sample(jobs)                                   # not due: one sample every INTERVAL seconds
    clock.now += 3.0
    wall.now += 3.0
    machine.usage(unit, 57_500_000)
    assert loads.sample(jobs)
    assert math.isclose(loads.get(1).cores, 2.5) and loads.get(1).usage == 57.5
    machine.usage(unit, 57_000_000)                                 # a counter that went back: never negative
    clock.now += 3.0
    loads.sample(jobs)
    assert loads.get(1).cores == 0.0


def test_runnable_threads_are_counted_and_smoothed_as_the_kernel_smooths_its_load(tmp_path):
    """A job's load is its runnable threads (state R) smoothed over a minute: ``load * e + n * (1 - e)``, ``e =
    exp(-dt / 60)``; first seen, as the kernel's average would be had it run as many since it was admitted (a job
    running for an hour: all of them; one just admitted: none yet).  Sleeping (S) and disk-waiting (D) threads are not
    runnable; a thread listed whose ``/proc`` entry is gone is not either."""
    machine = FakeMachine(tmp_path)
    states = {200: 'R', 201: 'R', 202: 'S', 203: 'D', 204: 'R'}
    machine.unit('tfjob-b-1.service', {200: states}, usage_usec=1)
    machine.unit('tfjob-c-1.service', {300: {300: 'R', 301: 'R'}}, usage_usec=1)
    clock, wall = Clock(10.0), Clock(10_000.0)
    loads = loads_of(machine, clock, wall)
    jobs = [running(1, 200, wall.now - 3600), running(2, 300, wall.now)]
    loads.sample(jobs)
    assert loads.get(1).runnable == 3 and math.isclose(loads.get(1).load, 3.0, rel_tol=1e-9)
    assert loads.get(2).runnable == 2 and loads.get(2).load == 0.0
    for tid in (200, 201, 204):
        machine.state(200, tid, 'S')
    machine.state(300, 301, 'S')
    clock.now += 3.0
    loads.sample(jobs)
    e = math.exp(-3.0 / 60)
    assert math.isclose(loads.get(1).load, 3.0 * e) and loads.get(1).runnable == 0
    assert math.isclose(loads.get(2).load, 1.0 * (1 - e)) and loads.get(2).runnable == 1
    clock.now += 4.0                                               # an uneven interval: its own decay
    loads.sample(jobs)
    assert math.isclose(loads.get(1).load, 3.0 * e * math.exp(-4.0 / 60))
    for path in (machine.proc / '204').rglob('*'):                  # a thread exits between two reads
        path.unlink() if path.is_file() else None
    clock.now += 3.0
    machine.state(200, 200, 'R')
    loads.sample(jobs)
    assert loads.get(1).runnable == 1


def test_waiting_is_the_cgroups_cpu_pressure(tmp_path):
    """Waiting is the cgroup's ``cpu.pressure`` ``some avg10``; a cgroup without the file has none."""
    machine = FakeMachine(tmp_path)
    machine.unit('tfjob-d-1.service', {400: {400: 'R'}}, pressure=38.25)
    machine.unit('tfjob-e-1.service', {500: {500: 'R'}})
    loads = loads_of(machine, Clock(0.0), Clock(100.0))
    loads.sample([running(1, 400, 0.0), running(2, 500, 0.0)])
    assert loads.get(1).waiting == 38.25 and loads.get(2).waiting is None


def test_a_unit_that_ends_keeps_its_last_sample_and_nothing_raises(tmp_path):
    """A unit that ends between two samples (its cgroup and its processes gone), a pid that vanished before it was
    ever read, a missing cgroup file: all normal.  The job's last sample stays (its total CPU time and mean cores
    over its run, when that sample came within ``FINAL_SECONDS`` of its end); a job that left the list is forgotten."""
    machine = FakeMachine(tmp_path)
    unit = machine.unit('tfjob-f-1.service', {600: {600: 'R', 601: 'R'}}, usage_usec=0)
    clock, wall = Clock(0.0), Clock(1000.0)
    loads = loads_of(machine, clock, wall)
    jobs = [running(1, 600, 400.0), running(2, 999, 900.0)]          # 999: no such process
    loads.sample(jobs)
    clock.now, wall.now = 3.0, 1003.0
    machine.usage(unit, 1_800_000_000)                               # 1800 CPU seconds over its run so far
    loads.sample(jobs)
    last = loads.get(1)
    assert last.usage == 1800.0 and loads.get(2) is None
    (unit / 'cgroup.threads').unlink()                               # a file missing: no threads, no error
    clock.now, wall.now = 6.0, 1006.0
    loads.sample(jobs)
    assert loads.get(1).runnable == 0
    machine.vanish(unit, pids=[600, 601])
    clock.now, wall.now = 9.0, 1009.0
    assert loads.sample(jobs)                                        # the row still says running: no error
    assert loads.get(1).seen == 1006.0 and loads.get(1).usage == 1800.0
    # its load fades out of the jobs' sum as the kernel's own 1-minute average still holds it
    assert loads.summary.jobs == 0 and loads.summary.jobs_cores == 0.0
    assert math.isclose(loads.summary.jobs_load, loads.get(1).load * math.exp(-3.0 / jobload.LOAD_WINDOW))
    ended = {**jobs[0], 'status': 'completed', 'finished_ts': 1008.0}
    total, mean = jobload.final(loads.get(1), ended)
    assert total == 1800.0 and math.isclose(mean, 1800.0 / 608.0)
    assert jobload.final(loads.get(1), {**ended, 'finished_ts': 1006.0 + jobload.FINAL_SECONDS + 1}) is None
    assert jobload.final(None, ended) is None
    clock.now += 3.0
    loads.sample([ended])
    assert loads.get(1).usage == 1800.0                              # still listed (under finished): kept
    clock.now += 3.0
    loads.sample([])
    assert loads.get(1) is None                                      # gone from the list: forgotten


def test_a_job_without_a_unit_of_its_own_is_followed_by_its_process_tree(tmp_path):
    """A job started without ``--detach`` shares its caller's cgroup (a shell's scope): its CPU is its wrapper's
    process tree's (``children``), its runnable threads that tree's; no waiting share, no total once it ends."""
    machine = FakeMachine(tmp_path)
    scope = f"{APP}/app-kitty@1.service"
    machine.unit('', {9: {9: 'R'}}, cgroup=scope)                    # the shell: in the scope, not in the job
    machine.process(700, {700: 'S'}, scope, ticks=10 * TICKS, children=(701,))
    machine.process(701, {701: 'R', 702: 'R'}, scope, ticks=20 * TICKS)
    clock, wall = Clock(0.0), Clock(500.0)
    loads = loads_of(machine, clock, wall)
    jobs = [running(1, 700, 0.0)]
    loads.sample(jobs)
    first = loads.get(1)
    assert not first.cgroup and first.usage is None and first.waiting is None and first.runnable == 2
    machine.ticks(701, 26 * TICKS)
    clock.now += 3.0
    loads.sample(jobs)
    assert math.isclose(loads.get(1).cores, 2.0)
    assert jobload.final(loads.get(1), {**jobs[0], 'status': 'completed', 'finished_ts': 503.0}) is None


def test_the_machine_beside_its_jobs_and_what_is_outside_the_queue(tmp_path):
    """The summary: the running jobs' cores and load summed beside the machine's busy cores (``/proc/stat``) and its
    load average; what is left is outside the queue.  More than ``OUTSIDE_CORES`` outside for ``OUTSIDE_SECONDS``
    warns, and names the busiest process outside every job."""
    machine = FakeMachine(tmp_path)
    unit = machine.unit('tfjob-g-1.service', {800: {800: 'R', 801: 'R'}}, usage_usec=0)
    machine.process(900, {900: 'R'}, f"{APP}/app-steam@1.service", ticks=0)     # outside the queue
    clock, wall = Clock(0.0), Clock(2000.0)
    loads = loads_of(machine, clock, wall)
    jobs = [running(1, 800, 0.0)]
    busy = idle = 0
    loads.sample(jobs)
    summary = loads.summary
    assert summary.machine_cores is None and summary.outside_cores is None and not summary.warn
    assert summary.jobs == 1 and math.isclose(summary.jobs_load, 2.0) and summary.machine_load == 1.0
    usage = 0
    for step in range(25):                                          # 2 cores in the job, 5 outside, 75 seconds
        clock.now += 3.0
        wall.now += 3.0
        usage += 6_000_000
        busy, idle = busy + 21 * TICKS, idle + 39 * TICKS             # 7 of 20 cores busy for 3 seconds
        machine.usage(unit, usage)
        machine.machine(busy, idle)
        machine.ticks(900, (step + 1) * 15 * TICKS)                  # 5 cores' worth
        machine.loadavg(9.5)
        loads.sample(jobs)
        summary = loads.summary
        assert math.isclose(summary.jobs_cores, 2.0) and math.isclose(summary.machine_cores, 7.0)
        assert math.isclose(summary.outside_cores, 5.0) and math.isclose(summary.outside_load, 9.5 - summary.jobs_load)
        assert summary.outside_since == 2003.0
        assert summary.warn is (wall.now - 2003.0 >= jobload.OUTSIDE_SECONDS)
        # every process's CPU is read only while over, every OUTSIDE_SCAN seconds: named from the second look on
        assert summary.top == ('pid 900 (5.0 cores): python -m job900 --x' if 3.0 * step >= jobload.OUTSIDE_SCAN * 0.9
                               else '')
    assert summary.warn and summary.threads == 20
    clock.now += 3.0                                                # the outside work stops: no warning
    busy, idle = busy + 6 * TICKS, idle + 54 * TICKS
    machine.machine(busy, idle)
    usage += 6_000_000
    machine.usage(unit, usage)
    loads.sample(jobs)
    assert not loads.summary.warn and loads.summary.outside_since is None and loads.summary.top == ''


def test_sampling_costs_well_under_one_percent_of_a_core(tmp_path):
    """Ten jobs of 40 threads each (real processes, their threads read from the real ``/proc``): one sample costs
    this process well under 1% of a core at one sample every ``INTERVAL`` seconds (CPU time, not wall time: the
    machine may be busy)."""
    code = ('import threading, time\n'
            '[threading.Thread(target=time.sleep, args=(120,), daemon=True).start() for _ in range(39)]\n'
            'print("ready", flush=True)\ntime.sleep(120)\n')
    children = [subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True) for _ in range(10)]
    try:
        for child in children:
            assert child.stdout.readline().strip() == 'ready'
        loads = jobload.JobLoads(cgroup_root=tmp_path, threads=20, host='here')
        for index, child in enumerate(children):
            directory = tmp_path / f"tfjob-cost-{index}.service"
            directory.mkdir()
            (directory / 'cgroup.procs').write_text(f"{child.pid}\n")
            (directory / 'cgroup.threads').write_text(''.join(f"{tid}\n" for tid in os.listdir(
                f"/proc/{child.pid}/task")))
            (directory / 'cpu.stat').write_text('usage_usec 1000\n')
            loads._sources[index] = jobload.Source(Path('/proc'), cgroup=directory)   # noqa: SLF001 - their units
        jobs = [running(index, child.pid, time.time()) for index, child in enumerate(children)]
        for _ in range(20):
            loads.sample(jobs, force=True)
        assert all(loads.get(index) is not None for index in range(10))
        per_sample = loads.cost / loads.samples
        assert per_sample < 0.01 * jobload.INTERVAL, per_sample
    finally:
        for child in children:
            child.kill()
            child.wait()
