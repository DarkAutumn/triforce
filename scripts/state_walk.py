"""Walk game states through the full game path and digest every step, to prove an emulator change changes nothing.

    .venv/bin/python scripts/state_walk.py OUT.json [--steps 40] [--trajectories 300] [--seed 0]

Every start state in triforce/states and an evenly spaced sample of trajectory snapshots is loaded through
``Game.load`` (RNG randomization and room entry seeded), then plays ``--steps`` random legal actions drawn from a
seeded generator, through the option executor with look-ahead, interrupts and masks. Each step's RAM + PRG-RAM,
rendered RGB frame and change description are hashed; each state is played twice, rendering on and off, and the
RAM digests of the two must agree (rendering may not change emulation). OUT.json maps state -> per-step digests;
compare two runs with ``--compare A.json B.json``.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from triforce import _native

ROOT = Path(__file__).resolve().parents[1]


def legal_action(mask, rng) -> tuple[int, int, int]:
    """A random legal (kind, direction, duration) from a flat action mask."""
    nk, nd, nu = len(_native.ACTION_KINDS), len(_native.DIRECTIONS), len(_native.DURATIONS)
    kind = int(rng.choice(np.flatnonzero(mask[:nk])))
    dirs = np.flatnonzero(mask[nk + kind * nd: nk + (kind + 1) * nd])
    durs = np.flatnonzero(mask[nk + nk * nd + kind * nu: nk + nk * nd + (kind + 1) * nu])
    return kind, int(rng.choice(dirs)) if len(dirs) else 0, int(rng.choice(durs)) if len(durs) else 0


def walk(path: str, steps: int, seed: int, render: bool) -> list[str]:
    game = _native.Game()
    game.render = render
    game.load(path, randomize_rng=True, fresh_room=True, seed=seed)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        mask = np.asarray(game.mask)
        if not mask.any():
            break
        change = game.act(legal_action(mask, rng))
        h = hashlib.blake2b(bytes(game.ram()), digest_size=12)
        h.update(_native.Game.describe_change(change).encode())
        ram = h.hexdigest()
        pixels = ''
        if render:
            try:
                pixels = hashlib.blake2b(np.asarray(game.screen()).tobytes(), digest_size=12).hexdigest()
            except _native.RenderDisabledError:     # a step that ran no real frame has no picture
                pixels = '-'
        out.append(f"{ram}:{pixels}")
        if game.state.game_over:
            break
    return out


def states(trajectories: int) -> list[str]:
    paths = sorted(str(p) for p in (ROOT / 'triforce' / 'states').glob('*.state'))
    tracks = sorted(str(p) for p in (ROOT / 'trajectories').rglob('*.state'))
    if trajectories and tracks:
        step = max(1, len(tracks) // trajectories)
        paths += tracks[::step][:trajectories]
    return paths


def compare(a: str, b: str) -> int:
    x, y = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    bad = [k for k in sorted(set(x) | set(y)) if x.get(k) != y.get(k)]
    for k in bad[:10]:
        first = next((i for i, (p, q) in enumerate(zip(x.get(k, []), y.get(k, []))) if p != q), None)
        print(f"DIFFERS {k}: first differing step {first} (lengths {len(x.get(k, []))}/{len(y.get(k, []))})")
    print(f"{len(x)} vs {len(y)} states, {sum(len(v) for v in x.values())} steps: {len(bad)} differ")
    return 1 if bad else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('out', nargs='?')
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--trajectories', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--compare', nargs=2, metavar=('A', 'B'))
    args = parser.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    began, result, mismatched = time.time(), {}, []
    for i, path in enumerate(states(args.trajectories)):
        seed = args.seed + i
        on = walk(path, args.steps, seed, render=True)
        off = walk(path, args.steps, seed, render=False)
        if [d.split(':')[0] for d in on] != [d.split(':')[0] for d in off]:
            mismatched.append(path)
        result[str(Path(path).relative_to(ROOT))] = on
    Path(args.out).write_text(json.dumps(result, indent=0, sort_keys=True))
    steps = sum(len(v) for v in result.values())
    print(f"{len(result)} states, {steps} steps in {time.time() - began:.0f}s; render on/off mismatches: "
          f"{len(mismatched)} {mismatched[:3]}")
    return 1 if mismatched else 0


if __name__ == '__main__':
    sys.exit(main())
