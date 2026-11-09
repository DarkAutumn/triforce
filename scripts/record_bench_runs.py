"""Record chain runs of the promoted team on the current core, for `tf bench --recordings`.

    .venv/bin/python scripts/record_bench_runs.py OUT_DIR [--runs 6] [--first-seed 500]

The corpus agent movies were recorded on the FCEUmm-era core and desync on zcore's hardware timing after a few
thousand frames, so they no longer show gameplay. This plays ``--runs`` seeded chain runs on each of four
segments of the route (full, D2 from 2_7ds_0, after D2 from 0_3cc_0, D1 from 1_73s_0) with every input
and poke recorded, and writes each as OUT_DIR/<segment>-<seed>.json: the origin state file, one controller byte
per frame and the pokes, plus the run's outcome.
"""

import argparse
import json
import sys
from pathlib import Path

from triforce import _native
from triforce.run import record_movie
from triforce.runner import Route, Runner

SEGMENTS = [('full', _native.DEFAULT_ROUTE, None), ('d2', _native.DEFAULT_ROUTE, '2_7ds_0'),
            ('post-d2', _native.DEFAULT_ROUTE, '0_3cc_0'), ('d1', _native.DEFAULT_ROUTE, '1_73s_0')]


def origin_path(origin: str) -> str:
    path = Path(origin)
    if not path.is_file():
        path = Path(_native.paths()['states_dir']) / (origin if origin.endswith('.state') else f'{origin}.state')
    return str(path.resolve())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('out')
    parser.add_argument('--runs', type=int, default=6)
    parser.add_argument('--first-seed', type=int, default=500)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = Runner()
    for name, route_name, start in SEGMENTS:
        route = Route.load(route_name, start_from=start)
        for seed in range(args.first_seed, args.first_seed + args.runs):
            result, movie = record_movie(runner, route, seed)
            last = result.rooms[-1] if result.rooms else None
            record = {'origin': origin_path(movie['origin']), 'buttons': list(movie['buttons']),
                      'pokes': [list(p) for p in movie['pokes']],
                      'meta': {'segment': name, 'route': route_name, 'start': start, 'seed': seed,
                               'result': result.result, 'rooms': len(result.rooms),
                               'last_room': f"{last.level}:{last.room}" if last else None}}
            (out / f'{name}-{seed}.json').write_text(json.dumps(record))
            print(f"{name} seed {seed}: {result.result}, {len(result.rooms)} rooms, {len(movie['buttons'])} frames",
                  flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
