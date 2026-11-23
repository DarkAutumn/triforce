"""Record chain runs of the promoted team on the current core, for `tf bench --recordings` and the ladder corpus.

    triforce job run --agent A --task T --subtask S --desc '...' --timeout 30m -- \\
        .venv/bin/python scripts/record_bench_runs.py OUT_DIR [--runs 6] [--first-seed 500]

Plays ``--runs`` seeded chain runs on each of the six segments of the route (full, D2 from 2_7ds_0, after D2
from 0_3cc_0, D1 from 1_73s_0, after D1 from 0_37c_v2_0, D3 from 3_7cs_0), each recorded as a replay
(``triforce.replay``: the start state, every input and memory edit), and writes OUT_DIR/<segment>-<seed>.tfreplay.
`tf movie import` turns a recording into a corpus movie (README, "Emulator regression suite").  A job of the
machine's queue (``jobs.require_job``); the runs recorded of the total are its progress.
"""

import argparse
import sys
from pathlib import Path

from triforce import _native, jobs
from triforce.run import record_replay
from triforce.runner import Route, Runner

SEGMENTS = [('full', _native.DEFAULT_ROUTE, None), ('d2', _native.DEFAULT_ROUTE, '2_7ds_0'),
            ('post-d2', _native.DEFAULT_ROUTE, '0_3cc_0'), ('d1', _native.DEFAULT_ROUTE, '1_73s_0'),
            ('post-d1', _native.DEFAULT_ROUTE, '0_37c_v2_0'), ('d3', _native.DEFAULT_ROUTE, '3_7cs_0')]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('out')
    parser.add_argument('--runs', type=int, default=6)
    parser.add_argument('--first-seed', type=int, default=500)
    args = parser.parse_args(argv)
    jobs.require_job()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = Runner()
    total, done = len(SEGMENTS) * args.runs, 0
    for name, route_name, start in SEGMENTS:
        route = Route.load(route_name, start_from=start)
        for seed in range(args.first_seed, args.first_seed + args.runs):
            jobs.progress(done, total)
            result, data = record_replay(runner, route, seed, segment=name)
            done += 1
            (out / f'{name}-{seed}.tfreplay').write_bytes(data)
            print(f"{name} seed {seed}: {result.result}, {len(result.rooms)} rooms, "
                  f"{_native.Replay(data).frames} frames", flush=True)
    jobs.progress(total, total)
    return 0


if __name__ == '__main__':
    sys.exit(main())
