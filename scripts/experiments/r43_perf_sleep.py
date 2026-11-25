"""Round 43 (Perf43): the env threads' time of running lab trainings, from /proc schedstat (round40.md "Where a
teacher's time goes"): running, waiting for a CPU (on the run queue), sleeping (waiting for the network's answer).
Every WINDOW seconds, for every training whose command line holds MATCH, one line to OUT:

    python scripts/experiments/r43_perf_sleep.py MATCH OUT [--window 10]

``epoch name/run envs running waiting sleeping load`` (shares of the window, the env threads' mean).  Only reads
/proc (no ptrace); the trainings are not touched.  ``r43_perf_read.py`` puts each window with the rollout it fell in.
"""
import argparse
import os
import time


def threads(pid: int) -> dict:
    out = {}
    for tid in os.listdir(f'/proc/{pid}/task'):
        try:
            comm = open(f'/proc/{pid}/task/{tid}/comm').read().strip()
            run, wait, _ = map(int, open(f'/proc/{pid}/task/{tid}/schedstat').read().split())
        except OSError:
            continue
        if comm.startswith('pool-'):
            out[tid] = (run, wait)
    return out


def trainings(match: str) -> dict[int, str]:
    out = {}
    for pid in map(int, filter(str.isdigit, os.listdir('/proc'))):
        try:
            cmd = open(f'/proc/{pid}/cmdline').read().split('\0')
        except OSError:
            continue
        line = ' '.join(cmd)
        if 'triforce.lab.train' in line and match in line and 'triforce.job' not in line:
            name = cmd[cmd.index('--name') + 1] if '--name' in cmd else '?'
            run = cmd[cmd.index('--run-id') + 1] if '--run-id' in cmd else '?'
            out[pid] = f'{name}/{run}'
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('match')
    parser.add_argument('out')
    parser.add_argument('--window', type=float, default=10.0)
    args = parser.parse_args()
    before: dict[int, dict] = {}
    while True:
        began = time.time()
        found = trainings(args.match)
        now_threads = {}
        for pid in found:
            try:
                now_threads[pid] = threads(pid)
            except OSError:
                continue
        load = open('/proc/loadavg').read().split()[0]
        with open(args.out, 'a') as file:
            for pid, current in now_threads.items():
                old = before.get(pid)
                if not old:
                    continue
                pairs = [(current[t][0] - old[t][0], current[t][1] - old[t][1]) for t in current if t in old]
                if not pairs:
                    continue
                n = len(pairs)
                run = sum(r for r, _ in pairs) / 1e9 / args.window / n
                wait = sum(w for _, w in pairs) / 1e9 / args.window / n
                file.write(f"{began:.0f} {found[pid]} {n} {run:.3f} {wait:.3f} {max(0.0, 1 - run - wait):.3f} "
                           f"{load}\n")
        before = now_threads
        time.sleep(max(0.0, args.window - (time.time() - began)))


if __name__ == '__main__':
    raise SystemExit(main())
