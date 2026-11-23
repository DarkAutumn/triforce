"""Round 26 (docs/experiments/round26.md, "7:0x0d, the Wallmaster room"): a rule that clears level 7 room 0x0d
through the environment's own actions, measured, and its demonstration rows for ``train --demos``.

    python scripts/experiments/r26_d7_0d_demos.py --measure [RUNS_PER_STATE] [--policy RUN_DIR]
    python scripts/experiments/r26_d7_0d_demos.py OUT.npz [RUNS_PER_STATE] [--policy RUN_DIR]

The room (docs/routes/d7-walks.md §7, zelda-asm/enemies.md "Wallmaster"): five Wallmasters that come out of a wall
only while Link stands against it (one per $60 frames, on a path fixed as it comes out: two squares out of the wall,
three along it toward where Link stood, two back in), three Bubbles (a touch takes the sword for 150 frames), a
block that moves once the five are dead and stairs in the NE corner.  The room's walkable ring IS the wall line;
only the inside of the C of blocks (X $20-$B0, Y $75-$9D), open to the W, is off it.

The rule, the button script ``r26_wallmaster_bot.py`` in the env's actions (``MOVE dir frames``, ``SWORD dir``,
``WAIT frames``):

* **lure** at ($20, $9D), the W wall a row below the door row: ``MOVE W`` there stops at the wall (on the door row
  it walks out through the bombed doorway).  The reset's own spot ($20, $8D) lures the first hand.
* **step off** as soon as a hand is out: ``MOVE E 8``, ``MOVE E 4`` to X $30-$37, ``MOVE S`` to Y $9D (the blocks
  stop Link there); the hand walks the wall line X $20 past Link's row a square away.
* **strike**: ``SWORD W`` (``BEAMS W`` at full health) while the hand is within `REACH_X` pixels W of Link and
  `REACH_Y` of his row.  It walks 0.375 pixels a frame, so the window is some 60 frames long: any decision in it
  hits, and the waits before it are as long as the hand's distance allows.
* with the sword taken by a Bubble, or while a hand's death sparkle lasts: off the wall, wait.
* **a dropped clock** (a Wallmaster's drop in one kill of ten) must not be taken while a hand lives: under the clock
  a hand that comes out stays in the wall where it starts, out of the sword's reach, and the room can't be cleared.
  It lies where the hand died, under the next swing and a Bubble's knock from the strike spot: Link goes deeper
  into the C (X `RETREAT_X`) and waits until it has gone (some 500 frames).
* all five dead: follow the objective's ``target`` in short steps to the tile E of the block, ``MOVE W 16`` beside
  it (the push; off its grid point the same action first takes Link on to the point), follow it to the stairs.

``--measure`` plays the rule from ``7_0dw_0..2`` with the reset's own hp and bomb draws and the family's staging
(``room_kills``: half the resets bank 1-4 kills) and prints endings, captures, clocks and hearts lost.  The default
writes the rows of the winning runs (``triforce.demos``): a SWORD row `SWING_REPEAT` times and the push
`PUSH_REPEAT` times (five and one decisions of a run); a WAIT, and a MOVE that ends at a wall, without its
duration.  With ``--policy`` (DAgger) a checkpoint plays at temperature 0.5 and the rule labels every state it
visits.

It plays the rule over many runs: a job of the machine's queue (``jobs.require_job``: ``triforce job run ...
--timeout 30m -- .venv/bin/python scripts/experiments/r26_d7_0d_demos.py ...``).
"""

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

from triforce._native import Env
from triforce.demos import DemoRecorder, action_indices, legal
from triforce.jobs import require_job

OBJ_ID, OBJ_X, OBJ_Y, OBJ_DIR, OBJ_STATE, OBJ_METASTATE = 0x34F, 0x70, 0x84, 0x98, 0xAC, 0x405   # slot 0 is Link
FACING_S, FACING_N = 4, 8
BLOCK_SLOT, BLOCK = 11, 0x68                    # the push block's object: state 0 until it is pushed
SWORD_BLOCKED_TIMER, SWORD_BLOCKED = 0x4C, 0x52E                                   # a Bubble's touch
WALLMASTER, ITEM, CLOCK = 0x27, 0x60, 0x21     # object types; a dropped item's kind is its ObjState
CLOCK_HELD = 0x66C                              # InvClock: every monster frozen until Link leaves the room
STATES = ('7_0dw_0', '7_0dw_1', '7_0dw_2')
MAX_STEPS = 600
WALL_X, SAFE_X, ROW = 0x20, 0x30, 0x9D          # the lure spot's X, the nearest X off the wall line, the strike row
RETREAT_X = 0x58                                # where Link waits while a clock lies by the wall
C_TOP, C_BOTTOM = 0x75, 0x9D                    # Link's Y inside the C of blocks (at $75 his head is in the blocks)
REACH_X, REACH_Y = 0x18, 12                     # a hand this far W of Link and this near his row is struck
PUSH = (0xD0, 0x8D)                             # where MOVE W pushes the block
SWING_REPEAT, PUSH_REPEAT = 10, 20              # row weights: a run's five swings and its one push among ~115 rows


def hands(ram: bytes) -> tuple[int, list[tuple[int, int]]]:
    """(Wallmasters alive, the positions of those out of the wall)."""
    alive, out = 0, []
    for slot in range(1, 12):
        if ram[OBJ_ID + slot] == WALLMASTER and ram[OBJ_METASTATE + slot] < 0x10:
            alive += 1
            if ram[OBJ_STATE + slot] == 1:
                out.append((ram[OBJ_X + slot], ram[OBJ_Y + slot]))
    return alive, out


def clock_lies(ram: bytes) -> bool:
    """A dropped clock lies in the room.  Taken (a touch, or the sword's blade over it) it freezes every monster
    for the rest of the visit, and a hand that then comes out stays in the wall where it starts, two squares out
    of the sword's reach: the room can no longer be cleared (BOMB is not offered in this family)."""
    return any(ram[OBJ_ID + slot] == ITEM and ram[OBJ_STATE + slot] == CLOCK for slot in range(1, 12))


def hand_dying(ram: bytes) -> bool:
    """A hand's death sparkle: its drop (perhaps a clock) shows when it ends, where the hand died, on the wall
    line beside the lure spot."""
    return any(ram[OBJ_ID + slot] == WALLMASTER and ram[OBJ_METASTATE + slot] >= 0x10 for slot in range(1, 12))


def wait(frames: float) -> str:
    """The longest WAIT that ends before `frames` have passed (at least WAIT 4)."""
    return f"WAIT {next((d for d in (32, 16, 8) if d <= frames), 4)}"


def until_reach(x: int, y: int, out: list[tuple[int, int]]) -> float:
    """Frames until the nearest hand walks into reach of a swing W from (x, y): out of the wall to X `WALL_X`,
    then along it, 0.375 pixels a frame (as long as it takes if none comes this way)."""
    return min((max(WALL_X - hx, 0) + max(abs(hy - y) - (REACH_Y - 1), 0)) / 0.375 for hx, hy in out)


def offered(obs: dict, step: str) -> bool:
    return legal(obs['action_mask'], action_indices(step))


def strike(obs: dict) -> str | None:
    """The swing W the mask offers: BEAMS at full health, else SWORD."""
    return next((step for step in ('SWORD W', 'BEAMS W') if offered(obs, step)), None)


def home(x: int, y: int) -> str | None:
    """The next MOVE toward the strike spot (X `SAFE_X`.., Y `ROW`) inside the C, None once there.  Off the C (a
    Bubble's knock along the ring) first back to its W side."""
    if x < SAFE_X:
        if y < C_TOP:
            return 'MOVE S 4'
        if y > C_BOTTOM:
            return 'MOVE N 4'
        return 'MOVE E 8' if x <= WALL_X + 4 else 'MOVE E 4'
    if y < C_TOP or y > C_BOTTOM:
        return 'MOVE W 16'                      # on the top or bottom corridor: to the W column
    if y != ROW:
        return 'MOVE S 16'                      # the blocks below stop Link on the row
    if x >= SAFE_X + 0x10:
        return 'MOVE W 8' if x >= SAFE_X + 0x18 else 'MOVE W 4'
    return None


def follow(obs: dict) -> str | None:
    """A MOVE down the objective's ``target`` (its first step N, S, W or E), None where the block has no
    direction.  Short steps: the block gives the first step and the whole way's length, not where the way turns
    (a long MOVE N past the turn into the bottom corridor's upper row walks back and forth).  E and W legs end at a
    wall, so they may be longer."""
    target = np.asarray(obs['target'])
    if not target[:4].any():
        return None
    direction = 'NSWE'[int(np.argmax(target[:4]))]
    tiles = round(float(target[4]) * 32)
    frames = 4 if tiles <= 1 else 16 if tiles >= 6 and direction in 'WE' else 8
    return f"MOVE {direction} {frames}"


def action(ram: bytes, obs: dict) -> tuple[str, int, bool]:
    """The rule's action, how many times its row is written, and whether its duration is left free."""
    x, y = ram[OBJ_X], ram[OBJ_Y]
    alive, out = hands(ram)
    if alive == 0:
        if ram[OBJ_ID + BLOCK_SLOT] == BLOCK and ram[OBJ_STATE + BLOCK_SLOT] & 3 == 0 and x == PUSH[0]:
            # Beside the block.  On its grid point MOVE W is the push (16 frames of it); within a square of the
            # point, walking toward it, MOVE W first takes Link on to the point (the game turns only there).
            facing = ram[OBJ_DIR]
            toward = y == PUSH[1] or (y > PUSH[1] and facing == FACING_N) or (y < PUSH[1] and facing == FACING_S)
            if abs(y - PUSH[1]) < 8 and toward and offered(obs, 'MOVE W 16'):
                return 'MOVE W 16', PUSH_REPEAT, False
            if abs(y - PUSH[1]) < 8:
                return ('MOVE N 4' if y > PUSH[1] else 'MOVE S 4'), 1, False
        step = follow(obs)
        if step is None:
            if ram[OBJ_STATE + BLOCK_SLOT] & 3 == 2 and x == PUSH[0] and y < PUSH[1]:
                return 'MOVE N 4', 1, False     # on the stairs' tile, a pixel or two short of them
            return 'WAIT 4', 1, True            # the last hand's death, or the block sliding: no target yet
        return step, 1, False
    blocked = ram[SWORD_BLOCKED_TIMER] != 0 or ram[SWORD_BLOCKED] != 0
    if clock_lies(ram):
        # Away from it, deep into the C, until it has gone (some 500 frames): a Bubble's knock from the strike
        # spot ends on the wall line, on the clock.
        if x < SAFE_X or not C_TOP <= y <= C_BOTTOM:
            step = home(x, y)
            return step, 1, step.startswith('MOVE S')
        if x < RETREAT_X:
            return 'MOVE E 16', 1, False
        return 'WAIT 16', 1, True
    dying = hand_dying(ram)
    if out or blocked or dying:
        step = home(x, y)
        if step is not None:
            return step, 1, step.startswith('MOVE S')
        if blocked or dying or not out:
            return 'WAIT 16', 1, True          # until the sword is back, or the dying hand's drop shows
        swing = strike(obs)
        if swing and any(abs(hy - y) < REACH_Y and x - REACH_X <= hx <= x + 4 for hx, hy in out):
            return swing, SWING_REPEAT, False
        return wait(until_reach(x, y, out)), 1, True
    # No hand out and the sword in hand: to the lure spot, and wait there for one.  MOVE W only on the strike row:
    # on the door row it leaves the room.
    if x == WALL_X and C_TOP <= y <= C_BOTTOM:
        return 'WAIT 4', 1, True
    if y == ROW:
        return 'MOVE W 16', 1, True
    step = home(x, y)
    return step, 1, step.startswith('MOVE S')


def sampler(run_dir: str, temperature: float = 0.5):
    """A policy that acts as play does (a draw at ``temperature``) from a checkpoint: obs -> action text."""
    # pylint: disable=import-outside-toplevel
    import torch
    from triforce import _native, registry
    network, _ = registry.load_model(run_dir)
    network.eval()
    kinds, directions, durations = (list(_native.ACTION_KINDS), list(_native.DIRECTIONS),
                                    [int(d) for d in _native.DURATIONS])
    rng = np.random.default_rng(0)

    def act(obs) -> str:
        with torch.no_grad():
            logits = network.joint_log_probs(obs)[0].numpy() / temperature
        p = np.exp(logits - logits.max())
        kind, rest = divmod(int(rng.choice(len(p), p=p / p.sum())), len(directions) * len(durations))
        direction, duration = divmod(rest, len(durations))
        text = kinds[kind]
        if text in _native.DIRECTIONAL_KINDS:
            text += f' {directions[direction]}'
        if text in _native.TIMED_KINDS:
            text += f' {durations[duration]}'
        return text
    return act


def play(runs: int, policy=None, recorder: DemoRecorder | None = None, verbose: bool = False) -> list[dict]:
    """``runs`` episodes per start state: the rule plays (seeds 1000..), or ``policy`` does and the rule labels
    (seeds 3000..).  One record per run; with a ``recorder`` the rows of the rule's winning runs, or of every run
    the policy played, are kept."""
    env = Env('wallmaster', explicit_starts=True, strict_actions=False, staged_kills=True, seed=0)
    seeds = range(1000, 1000 + runs) if policy is None else range(3000, 3000 + runs)
    records = []
    for name in STATES:
        for seed in seeds:
            obs, _ = env.reset(seed=seed, options={'start_state': name})
            health, info, steps, swings, touches, was_blocked = env.state.health, {}, 0, 0, 0, False
            clocks, was_lying, clock_taken = 0, False, False
            staged = env.banked_kills
            for _ in range(MAX_STEPS):
                ram = bytes(env.state.ram)
                step, repeat, free = action(ram, obs)
                blocked = ram[SWORD_BLOCKED_TIMER] != 0
                touches += blocked and not was_blocked
                was_blocked = blocked
                lying = clock_lies(ram)
                clocks += lying and not was_lying
                was_lying = lying
                clock_taken = clock_taken or (ram[CLOCK_HELD] != 0 and hands(ram)[0] > 0)
                swings += step.startswith(('SWORD', 'BEAMS'))
                if recorder is not None and offered(obs, step):
                    recorder.add(obs, step, repeat=repeat, any_duration=free)
                if verbose:
                    print(f"  {steps:3d} L({ram[OBJ_X]:02x},{ram[OBJ_Y]:02x}) {step:10s} hands {hands(ram)} "
                          f"blocked {ram[SWORD_BLOCKED_TIMER]}")
                obs, _, terminated, truncated, info = env.step(step if policy is None else policy(obs))
                steps += 1
                if terminated or truncated:
                    break
            record = {'state': name, 'seed': seed, 'ending': info.get('ending', 'max-steps'), 'steps': steps,
                      'frames': info.get('total_frames', 0), 'lost': health - env.state.health, 'swings': swings,
                      'staged': staged, 'bubbles': touches, 'clocks': clocks, 'clock_taken': clock_taken,
                      'left': hands(bytes(env.state.ram))[0]}
            records.append(record)
            if verbose:
                print(record)
            if recorder is not None:
                if policy is not None or record['ending'] == 'exit-correct':
                    recorder.keep()
                else:
                    recorder.drop()
    return records


def report(records: list[dict]):
    for name in (*STATES, 'all'):
        rows = [r for r in records if name in ('all', r['state'])]
        wins = [r for r in rows if r['ending'] == 'exit-correct']
        endings = collections.Counter(r['ending'] for r in rows)
        print(f"{name}: {len(wins)} of {len(rows)} exit-correct ({100 * len(wins) / max(len(rows), 1):.0f}%), "
              f"captured {endings.get('wallmastered', 0)}, endings {dict(endings)}, "
              f"hearts lost a run {sum(r['lost'] for r in rows) / max(len(rows), 1):.3f}, "
              f"winning runs: median {int(np.median([r['steps'] for r in wins])) if wins else 0} decisions, "
              f"{int(np.median([r['frames'] for r in wins])) if wins else 0} frames, "
              f"Bubble touches a run {sum(r['bubbles'] for r in rows) / max(len(rows), 1):.1f}, "
              f"runs with a clock dropped {sum(r['clocks'] > 0 for r in rows)}, "
              f"with one taken while a hand lived {sum(r['clock_taken'] for r in rows)}")
    staged = collections.Counter((r['staged'], r['ending'] == 'exit-correct') for r in records)
    print("by staged kills (won, played):",
          {k: (staged[(k, True)], staged[(k, True)] + staged[(k, False)]) for k in sorted({s for s, _ in staged})})


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out', nargs='?', help="the .npz file of demonstration rows")
    parser.add_argument('runs', nargs='?', type=int, default=100, help="runs per start state (default 100)")
    parser.add_argument('--measure', action='store_true',
                        help="play (the rule, or with --policy the checkpoint) and print the results; write nothing")
    parser.add_argument('--policy', default=None, metavar='RUN_DIR',
                        help="DAgger: this checkpoint plays (temperature 0.5), the rule labels")
    parser.add_argument('--verbose', action='store_true', help="print every decision")
    args = parser.parse_args(argv)
    require_job()
    policy = sampler(args.policy) if args.policy else None
    if args.measure:
        if args.out is not None:                        # `--measure 50`: the one positional is the run count
            args.runs = int(args.out)
        report(play(args.runs, policy, verbose=args.verbose))
        return 0
    if args.out is None:
        parser.error("an output file, or --measure")
    recorder = DemoRecorder()
    records = play(args.runs, policy, recorder, args.verbose)
    report(records)
    rows = recorder.save(Path(args.out))
    kept = len(records) if args.policy else sum(r['ending'] == 'exit-correct' for r in records)
    print(f"{args.out}: {rows} rows from {kept} of {len(records)} runs")
    return 0 if rows else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
