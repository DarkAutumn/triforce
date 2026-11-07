"""Env-level emulator speed over every start state, by room family: the game path training pays for.

    .venv/bin/python scripts/env_speed.py [--steps 60] [--repeats 2] [--seed 0] [--json OUT]

Every state in triforce/states (both routes: overworld, D1, D2) is loaded with seeded RNG/room entry, then
plays ``--steps`` seeded random legal actions through ``Game.act`` with rendering off: the option executor,
look-ahead, interrupts and masks. Reported per family (the routes.yaml specialist owning the state's room) and
overall: real frames per second of ``act`` time (look-ahead frames are work but not counted as frames, so
they lower it), steps per second, and load time separately. Run it on two builds (PYTHONPATH) to A/B.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from triforce import _native

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from state_walk import legal_action  # noqa: E402  pylint: disable=wrong-import-position


def family_of() -> dict:
    """(level, room, in_cave) -> specialist family."""
    return {room: name for name, rooms in _native.families().items() for room in rooms}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--json', default=None)
    args = parser.parse_args(argv)
    families = family_of()
    stats = defaultdict(lambda: {'frames': 0, 'steps': 0, 'seconds': 0.0, 'load_seconds': 0.0, 'loads': 0})
    paths = sorted((ROOT / 'triforce' / 'states').glob('*.state'))
    for repeat in range(args.repeats):
        for i, path in enumerate(paths):
            seed = args.seed + 1000 * repeat + i
            game = _native.Game()
            game.render = False
            began = time.perf_counter()
            game.load(str(path), randomize_rng=True, fresh_room=True, seed=seed)
            loaded = time.perf_counter()
            family = families.get(tuple(game.state.full_location), 'other')
            s = stats[family]
            s['load_seconds'] += loaded - began
            s['loads'] += 1
            rng = np.random.default_rng(seed)
            frames0 = game.frame
            for _ in range(args.steps):
                mask = np.asarray(game.mask)
                if not mask.any() or game.state.game_over:
                    break
                action = legal_action(mask, rng)
                t = time.perf_counter()
                game.act(action)
                s['seconds'] += time.perf_counter() - t
                s['steps'] += 1
            s['frames'] += game.frame - frames0
    total = {k: sum(s[k] for s in stats.values()) for k in ('frames', 'steps', 'seconds', 'load_seconds', 'loads')}
    rows = sorted(stats.items(), key=lambda kv: -kv[1]['seconds'])
    print(f"{'family':18}{'frames/s':>10}{'steps/s':>9}{'frames':>9}{'steps':>7}{'act s':>7}{'load ms':>9}")
    for name, s in rows + [('ALL', total)]:
        print(f"{name:18}{s['frames'] / s['seconds']:10.0f}{s['steps'] / s['seconds']:9.0f}{s['frames']:9d}"
              f"{s['steps']:7d}{s['seconds']:7.1f}{1000 * s['load_seconds'] / max(1, s['loads']):9.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps({k: dict(v) for k, v in stats.items()} | {'ALL': total}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
