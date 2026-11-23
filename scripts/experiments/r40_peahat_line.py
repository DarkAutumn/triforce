"""Round 40 (docs/experiments/round40.md, "0x1e's row and 0x1f's line"; the owner's idea, Nov 23): `d3first` leaves
0:0x1e E on the row where Link's head meets the rocks (Y $5D, tile row 4; game.yaml's d3first `span: [4, 4]`) and
crosses 0:0x1f W -> N by holding E along that row to X $80 and N through the wall gap: the Peahats miss him.

    python scripts/experiments/r40_peahat_line.py rows [--seeds 500]
    python scripts/experiments/r40_peahat_line.py states DIR... [--seeds 1]
    python scripts/experiments/r40_peahat_line.py back [--seeds 300]
    python scripts/experiments/r40_peahat_line.py trace OUT.txt [--seed 0]
    python scripts/experiments/r40_peahat_line.py demos OUT.npz [--runs 200]

``rows``: the rule frame by frame (raw buttons, no executor) from 0x1f's W entry with Link's Y set to each candidate
row (``Game.load`` of ``d3first_0_1fw_<seed % 3>`` with ``ram_overrides``; every load draws a natural RNG state and
spawn cycle from its seed, as every start does): from rows below $5D Link first holds N to $5D (N and E only), then
E to X $80, then N.  Prints hits, hearts lost, frames and successes per crossing, and the same with E held on the
entry row itself (``straight``; rows $60-$64 snap onto the rocks of row 6 and stall).
``states``: the same rule from saved 0x1f entry states (``eval-chain --harvest`` of the networks' 0x1e crossings:
the rows today's ow-tektite leaves by), each with its own RNG (the draw the network met: paired with the chain's
runs) or, with ``--seeds N``, N fresh draws a state; prints the rule per entry row.
``back``: step 116 (0x1f N -> W) down X $80 to the strip and W, against down to Y $85 and W.
``trace``: one clean crossing frame by frame (Link and the Peahats) into OUT.txt.
``demos``: demonstration rows (``triforce.demos``) for ``train --demos`` (ow-peahat): the rule as env actions
(``MOVE N`` to the strip, ``MOVE E`` to X $80, ``MOVE N``) from ``d3first_0_1fw_0..2`` over seeds 1000..; the
decisions of runs that leave N without losing health are kept.
"""

import argparse
import json
import sys
from pathlib import Path

ROOM_1F, ROOM_1E, ROOM_0F = 0x1F, 0x1E, 0x0F
LINK_X, LINK_Y, ROOM_ID, OBJ_ID = 0x70, 0x84, 0xEB, 0x34F
PEAHAT = 0x1A
STRIP_Y = 0x5D          # the row: Link's head against the rock rows 0-3 of 0x1e and 0x1f
GAP_X = 0x80            # 0x1f's N wall gap (game.yaml: asm:Tile_GetAtObjOffset false-wall exception)
STATES = ('d3first_0_1fw_0', 'd3first_0_1fw_1', 'd3first_0_1fw_2')
BACK_STATES = ('d3first_0_1fn_0', 'd3first_0_1fn_1', 'd3first_0_1fn_2')
ROWS = (0x55, 0x5D, 0x65, 0x6D, 0x75, 0x7D, 0x85, 0x8D)
LIMIT = 900


def game():
    # pylint: disable=import-outside-toplevel
    from triforce import _native
    return _native.Game(seed=1, route='d3first')


def peahats(g) -> list[tuple[int, int]]:
    return [(g.read(LINK_X + i), g.read(LINK_Y + i)) for i in range(1, 12) if g.read(OBJ_ID + i) == PEAHAT]


def line_button(x: int, y: int, row: int | None) -> str:
    """The rule: N to `row` while W of the gap (None: stay on the entry row), E to X $80, N."""
    if row is not None and y > row and x < GAP_X:
        return 'UP'
    return 'RIGHT' if x < GAP_X else 'UP'


def cross(g, state, seed: int, entry_y: int | None = None, row: int | None = STRIP_Y, trace=None, randomize=True):
    """One crossing of 0x1f from its W entry by the rule; (hits, hearts lost, frames, left N).  `randomize`: the
    load draws a natural RNG state and spawn cycle from `seed` (False: the state's own)."""
    overrides = {'link_y': entry_y} if entry_y is not None else None
    s = g.load(state, randomize_rng=randomize, ram_overrides=overrides, seed=seed)
    health = start = s.health
    hits = frames = 0
    while frames < LIMIT and g.read(ROOM_ID) == ROOM_1F:
        x, y = g.read(LINK_X), g.read(LINK_Y)
        button = line_button(x, y, row)
        if trace is not None:
            trace.append((frames, button, x, y, health, peahats(g)))
        s = g.press(button, 1)
        frames += 1
        hits += s.health < health
        health = s.health
    return hits, start - health, frames, g.read(ROOM_ID) == ROOM_0F


def summary(results) -> str:
    n = len(results)
    return (f"hits {sum(r[0] for r in results) / n:.3f}, hearts {sum(r[1] for r in results) / n:.3f}, "
            f"runs hit {sum(1 for r in results if r[0])}/{n}, frames {sum(r[2] for r in results) / n:.0f}, "
            f"left N {sum(r[3] for r in results)}/{n}")


def rows(seeds: int) -> None:
    g = game()
    print(f"0x1f W -> N, {seeds} seeds a row (states {', '.join(STATES)} by seed % 3)")
    for y in ROWS:
        rule = [cross(g, STATES[s % 3], s, y) for s in range(seeds)]
        print(f"Y ${y:02X} (tile row {(y - 61) // 8}) rule: {summary(rule)}", flush=True)
    for y in range(0x59, 0x66):
        straight = [cross(g, STATES[s % 3], s, y, row=None) for s in range(seeds)]
        print(f"Y ${y:02X} straight E: {summary(straight)}", flush=True)


def states(dirs: list[str], seeds: int) -> None:
    """The rule from harvested 0x1f W entry states, grouped by Link's entry row: with one seed each state keeps its
    own RNG (the draw the network met there: paired with its chain run), with more each load draws a fresh one."""
    # pylint: disable=import-outside-toplevel
    from collections import defaultdict
    g = game()
    paths = sorted(p for d in dirs for p in Path(d).rglob('*.state'))
    by_row = defaultdict(list)
    for path in paths:
        snap = path.read_bytes()
        g.load(snap, randomize_rng=False)
        if g.read(ROOM_ID) != ROOM_1F or g.read(LINK_X) > 8:
            continue
        y = g.read(LINK_Y)
        for k in range(seeds):
            by_row[y].append(cross(g, snap, k, randomize=seeds > 1))
    total = []
    for y in sorted(by_row):
        total += by_row[y]
        print(f"entry Y ${y:02X}: {len(by_row[y])} crossings, rule {summary(by_row[y])}")
    print(f"all: {summary(total)}")


def back(seeds: int) -> None:
    g = game()
    for row in (STRIP_Y, 0x85):
        results = []
        for seed in range(seeds):
            s = g.load(BACK_STATES[seed % 3], seed=seed)
            health = start = s.health
            hits = frames = 0
            while frames < LIMIT and g.read(ROOM_ID) == ROOM_1F:
                x, y = g.read(LINK_X), g.read(LINK_Y)
                s = g.press('DOWN' if y < row and x >= GAP_X - 8 else 'LEFT', 1)
                frames += 1
                hits += s.health < health
                health = s.health
            results.append((hits, start - health, frames, g.read(ROOM_ID) == ROOM_1E))
        print(f"0x1f N -> W, down X $80 to Y ${row:02X} and W: {summary(results).replace('left N', 'left W')}")


def trace(out: Path, seed: int) -> None:
    g = game()
    rows_ = []
    result = cross(g, STATES[seed % 3], seed, STRIP_Y, trace=rows_)
    lines = [f"# 0:0x1f W -> N from Y ${STRIP_Y:02X}, state {STATES[seed % 3]}, seed {seed}: hits {result[0]}, "
             f"frames {result[2]}, left N {result[3]}",
             "# frame button link_x link_y hearts peahats(x,y)..."]
    for frame, button, x, y, health, ps in rows_:
        lines.append(f"{frame:4d} {button:5s} {x:3d} ${y:02X} {health:4.1f} " + ' '.join(f"({px},{py})" for px, py in ps))
    out.write_text('\n'.join(lines) + '\n')
    print(f"{out}: {len(rows_)} frames, {result}")


def env_action(ram: bytes) -> str:
    """The rule as env actions: N to the strip, E to the gap's column (a MOVE E of about 1.33 px a frame, cut short
    near it: Link turns only on a multiple of 8, so a MOVE N from X $79-$7F walks on E to $80 and turns there,
    from $78 or below it would turn short of the gap), N out."""
    x, y = ram[LINK_X], ram[LINK_Y]
    if y > STRIP_Y and x < GAP_X:
        return 'MOVE N 4'
    gap = GAP_X - x
    if gap < 8:
        return 'MOVE N 32'
    for frames in (32, 16, 8, 4):
        if gap >= frames * 4 // 3 + 1:
            return f'MOVE E {frames}'
    return 'MOVE E 4'


def demos(out: Path, runs: int) -> None:
    # pylint: disable=import-outside-toplevel
    from triforce._native import Env
    from triforce.demos import DemoRecorder, action_indices, legal
    env = Env('ow-peahat', explicit_starts=True, strict_actions=False, seed=0)
    recorder = DemoRecorder()
    kept = played = 0
    lost = []
    for name in STATES:
        for seed in range(1000, 1000 + runs):
            obs, _ = env.reset(seed=seed, options={'start_state': name})
            health, info, steps = env.state.health, {}, 0
            for steps in range(1, 60):
                step = env_action(bytes(env.state.ram))
                if legal(obs['action_mask'], action_indices(step)):
                    recorder.add(obs, step)
                obs, _, terminated, truncated, info = env.step(step)
                if terminated or truncated:
                    break
            played += 1
            if info.get('outcome') == 'success' and env.state.health >= health:
                recorder.keep()
                kept += 1
            else:
                recorder.drop()
                lost.append((name, seed, info.get('ending'), health - env.state.health))
    rows_ = recorder.save(out)
    print(f"{out}: {rows_} rows from {kept} of {played} runs")
    for item in lost[:20]:
        print('  dropped', item)


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('rows', 'states', 'back', 'trace', 'demos'))
    parser.add_argument('paths', nargs='*', help="states: harvest dirs; trace: OUT.txt; demos: OUT.npz")
    parser.add_argument('--seeds', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runs', type=int, default=200)
    args = parser.parse_args(argv)
    if args.mode == 'rows':
        rows(args.seeds or 500)
    elif args.mode == 'states':
        states(args.paths, args.seeds or 1)
    elif args.mode == 'back':
        back(args.seeds or 300)
    elif args.mode == 'trace':
        trace(Path(args.paths[0]), args.seed)
    else:
        demos(Path(args.paths[0]), args.runs)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
