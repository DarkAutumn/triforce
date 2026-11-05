"""Record agent chain movies in the FCEUmm oracle with the promoted specialists.

    TRIFORCE_MODELS_DIR=/home/leculver/git/triforce2/models \\
        .venv/bin/python scripts/record_corpus.py chain --route milestone1 --seed 0 --out /tmp/c
    ... record_corpus.py sweep --route milestone1 --seeds 0..399 --parallel 16 --out corpus/movies

Each run powers on the oracle, starts from the cached power_on_to_start base state, writes that position as a
temporary ``start.state`` and runs the route with ``Runner``.  The movie ``chain-<route>-<seed>.tfmovie``
is tagged ``agent``, the route, and ``power-on-to-d2`` when the run succeeds.
"""

import argparse
import json
import multiprocessing
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oracle_emulator import CORPUS, OracleEmulator    # pylint: disable=wrong-import-position

D2_TAG = 'power-on-to-d2'


def record_chain(route: str, seed: int, out: str, device: str = 'cpu') -> dict:
    """Plays one chain run in the oracle and writes its movie.  Returns a summary dict."""
    import torch    # pylint: disable=import-outside-toplevel
    from triforce.runner import Route, Runner    # pylint: disable=import-outside-toplevel
    from triforce.session import GameSession    # pylint: disable=import-outside-toplevel
    from triforce.start_states import write_savestate    # pylint: disable=import-outside-toplevel

    torch.set_num_threads(1)
    emu = OracleEmulator()
    try:
        emu.start_from_base('power_on_to_start')
        assert (emu.read(0x12), emu.read(0x10), emu.read(0xEB)) == (5, 0, 0x77), "power_on_to_start ended elsewhere"
        with tempfile.TemporaryDirectory() as tmp:
            start = os.path.join(tmp, 'start.state')
            write_savestate(start, emu.save_state())
            runner = Runner(session=GameSession(emulator=emu), models_dir=os.environ.get('TRIFORCE_MODELS_DIR'),
                            device=device)
            result = runner.run(Route.load(route, start_from=start), seed)
        name = f"chain-{route}-{seed}"
        tags = ['agent', route] + ([D2_TAG] if result.result == 'success' else [])
        last = result.rooms[-1] if result.rooms else None
        source = {'route': route, 'seed': seed, 'result': result.result, 'reason': result.reason,
                  'rooms': len(result.rooms), 'steps': result.steps,
                  'last_room': None if last is None else f"{last.level}:{last.room}"}
        os.makedirs(out, exist_ok=True)
        emu.save_movie(os.path.join(out, f"{name}.tfmovie"), name, tags, source)
        ram = emu.ram
        return {'name': name, 'frames': emu.frame, 'level': int(ram[0x10]), 'room': int(ram[0xEB]), **source}
    finally:
        emu.close()


def _worker(args):
    route, seed, out, device = args
    try:
        return record_chain(route, seed, out, device)
    except Exception as err:    # pylint: disable=broad-except
        return {'name': f"chain-{route}-{seed}", 'route': route, 'seed': seed, 'result': 'crash', 'reason': repr(err)}


def _seeds(text: str):
    lo, _, hi = text.partition('..')
    return range(int(lo), int(hi or lo) + 1)


def main(argv=None):
    """CLI."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    chain = sub.add_parser('chain')
    chain.add_argument('--route', required=True)
    chain.add_argument('--seed', type=int, required=True)
    chain.add_argument('--out', required=True)
    chain.add_argument('--device', default='cpu')
    sweep = sub.add_parser('sweep')
    sweep.add_argument('--route', required=True)
    sweep.add_argument('--seeds', required=True, help='A..B inclusive')
    sweep.add_argument('--parallel', type=int, default=16)
    sweep.add_argument('--out', required=True)
    sweep.add_argument('--device', default='cpu')
    sweep.add_argument('--log', default=None, help='JSONL of per-run summaries (appended)')
    args = parser.parse_args(argv)

    if args.command == 'chain':
        print(json.dumps(record_chain(args.route, args.seed, args.out, args.device)))
        return

    jobs = [(args.route, s, args.out, args.device) for s in _seeds(args.seeds)]
    ctx = multiprocessing.get_context('spawn')
    log = open(args.log, 'a', encoding='utf-8') if args.log else None    # pylint: disable=consider-using-with
    with ctx.Pool(args.parallel, maxtasksperchild=1) as pool:
        for summary in pool.imap_unordered(_worker, jobs):
            line = json.dumps(summary)
            print(line, flush=True)
            if log:
                log.write(line + '\n')
                log.flush()
    if log:
        log.close()


if __name__ == '__main__':
    main()
