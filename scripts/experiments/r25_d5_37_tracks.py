"""Round 25 (docs/experiments/round25.md, "Rooms that bleed, Dungeon 5"): hand-made curriculum tracks and
demonstrations for level 5 room 0x37 entered from S and left N.

    python scripts/experiments/r25_d5_37_tracks.py OUT_DIR [TRACKS_PER_STATE]
    python scripts/experiments/r25_d5_37_tracks.py --demos OUT.npz [RUNS_PER_STATE] [--policy RUN_DIR]

The room's objective goes up the column X $70 (game.yaml ``path_costs``): Link crosses the water once on the ladder
and meets the pocket's Darknut from below, where a swing reaches it.  The explorer's own tracks walk X $78 and take
a hit on nearly every visit, so a model trained on them never sees the cheap way.  This script plays a fixed rule
(walk to X $70, north, SWORD N whenever a live monster stands just ahead, then back east to the door) from the
start states ``5_37s_0..2`` over seeds 1000.. and keeps, per state, the first ``TRACKS_PER_STATE`` (default 2) runs
that leave by the N door without losing health and with at least one swing.  A track is written as the curriculum
reads it: ``OUT_DIR/5_37s_<chain>~x70s<seed>/NNNN.state`` and ``trajectory.json``.  Put them in
``trajectories/darknut/`` in place of the explorer's ``5_37s_<chain>`` tracks.

``--demos`` writes demonstration rows (``triforce.demos``) for ``train --demos``, the fix that worked (round 25 §5;
the curriculum alone did not move the promoted policy off X $78): the rule plays ``RUNS_PER_STATE`` times (default
200) from each state in every spawn layout and the decisions of the runs that leave without losing health are kept;
with ``--policy`` (DAgger) a checkpoint plays and the rule labels every state it visits (one round of it made 0x37
worse, 0.29 against 0.17 hearts a pocket visit).  The round-25 recipe (``r75-d537-c5``):

    python scripts/experiments/r25_d5_37_tracks.py --demos rule.npz
    python -m triforce.train --specialist darknut --load <run> --iterations 60 --sil-coeff 0.1 \\
        --demos rule.npz --demo-coeff 0.05 --stop-success 1.01

It plays the rule over many runs: a job of the machine's queue (``jobs.require_job``: ``triforce job run ...
--timeout 30m -- .venv/bin/python scripts/experiments/r25_d5_37_tracks.py ...``).
"""

import argparse
import json
import sys
from pathlib import Path

from triforce._native import Env
from triforce.demos import DemoRecorder, action_indices, legal
from triforce.jobs import require_job

OBJ_ID, OBJ_X, OBJ_Y = 0x34F, 0x70, 0x84                # slot 0 is Link, 1..11 the monsters
OBJ_INVINCIBLE, OBJ_METASTATE = 0x4F0, 0x405            # flashing after a hit; a spawn cloud or a death
DARKNUT = 0x0B
COLUMN = 0x70
STATES = ('5_37s_0', '5_37s_1', '5_37s_2')
SEEDS = range(1000, 3000)
MAX_STEPS = 80
SWING_REPEAT = 40                                       # --demos: a swing's row weight (a swing is 1% of the rows)
FREE_REACH, FREE_WIDTH = 48, 24                         # --demos: a MOVE N's duration is cloned only this near a monster


def monster_ahead(ram: bytes, reach: int = 22, width: int = 12) -> bool:
    """A live monster that isn't flashing stands within ``reach`` pixels north of Link (and ``width`` to either side;
    the defaults are a swing's reach)."""
    x, y = ram[OBJ_X], ram[OBJ_Y]
    for slot in range(1, 12):
        kind = ram[OBJ_ID + slot]
        if kind and kind < 0x40 and not ram[OBJ_METASTATE + slot] and not ram[OBJ_INVINCIBLE + slot]:
            forward, sideways = y - ram[OBJ_Y + slot], ram[OBJ_X + slot] - x
            if -2 <= forward <= reach and abs(sideways) <= width:
                return True
    return False


def action(ram: bytes) -> str:
    """The rule: north off the door, west to the column, north along it (swinging at a monster ahead), then east
    to the N door's column and out."""
    x, y = ram[OBJ_X], ram[OBJ_Y]
    if y > 0xBD:
        return 'MOVE N 4'
    if x > COLUMN and y > 0x5D:
        return 'MOVE W 4'
    if y > 0x5D:
        return 'SWORD N' if monster_ahead(ram) else 'MOVE N 4'
    if x < 0x72:
        return 'MOVE E 4'
    return 'MOVE N 8'


def pocket_darknut(ram: bytes) -> bool:
    """The layout with a Darknut in the pocket below the water (the one the column meets)."""
    return any(ram[OBJ_ID + slot] == DARKNUT and 0x60 <= ram[OBJ_X + slot] <= 0x90
               and 0x88 <= ram[OBJ_Y + slot] <= 0x9A for slot in range(1, 12))


def craft(out: Path, per_state: int) -> list[Path]:
    env = Env('darknut', explicit_starts=True, strict_actions=False, seed=0)
    made = []
    for name in STATES:
        found = 0
        for seed in SEEDS:
            env.reset(seed=seed, options={'start_state': name})
            if not pocket_darknut(bytes(env.state.ram)):
                continue
            health = env.state.health
            points, actions, info = [env.snapshot()], [], {}
            for _ in range(MAX_STEPS):
                actions.append(action(bytes(env.state.ram)))
                _, _, terminated, truncated, info = env.step(actions[-1])
                if terminated or truncated:
                    break
                points.append(env.snapshot())
            kept = (info.get('outcome') == 'success' and env.state.health >= health
                    and any(step.startswith('SWORD') for step in actions))
            print(f"{name} seed {seed}: {len(actions)} steps, {info.get('ending')}, health {health} -> "
                  f"{env.state.health}{'' if kept else ' (dropped)'}")
            if not kept:
                continue
            track = out / f'{name}~x70s{seed}'
            track.mkdir(parents=True, exist_ok=True)
            for index, point in enumerate(points[:len(actions)]):
                (track / f'{index:04d}.state').write_bytes(point.state_file())
            (track / 'trajectory.json').write_text(json.dumps({
                'start_state': name, 'success': True, 'ending': info.get('ending'), 'steps': len(actions),
                'health': health, 'actions': actions,
                'explore': {'iterations': 0, 'env_steps': len(actions), 'cells': 0, 'seconds': 0.0, 'seed': seed},
                'crafted': 'round 25: X $70, one ladder crossing, SWORD N at a Darknut ahead'}, indent=1))
            made.append(track)
            found += 1
            if found >= per_state:
                break
        if found < per_state:
            print(f"{name}: only {found} of {per_state} tracks", file=sys.stderr)
    return made


def label(ram: bytes, obs: dict) -> tuple[str, bool]:
    """The rule's action for a demonstration row and whether its duration is left free: the swing is BEAMS where
    the mask offers that instead of SWORD (full health); a MOVE N past the door with no monster within
    ``FREE_REACH`` pixels ahead is cloned without its duration (``ANY_DURATION``): cloned in full, the rule's short
    steps spread to the family's Blue Darknut rooms; cloned without any durations, the policy overshot the column
    (round 25 §5)."""
    step = action(ram)
    if step.startswith('SWORD') and not legal(obs['action_mask'], action_indices(step)):
        step = step.replace('SWORD', 'BEAMS')
    free = step.startswith('MOVE N') and ram[OBJ_Y] <= 0xBD and not monster_ahead(ram, FREE_REACH, FREE_WIDTH)
    return step, free


def sampler(run_dir: str, temperature: float = 0.5):
    """A policy that acts as play does (a draw at ``temperature``) from a checkpoint: obs -> action text."""
    # pylint: disable=import-outside-toplevel
    import numpy as np
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


def demos(out: Path, runs: int, policy=None) -> tuple[int, int, int]:
    """Records demonstration rows (``triforce.demos``) from ``runs`` episodes per start state, every spawn layout and
    the reset's own hp and bombs draws.  Without ``policy`` the rule plays (seeds 1000..) and the decisions of the
    runs that leave by the N door without losing health are kept.  With one (DAgger: the policy plays, seeds
    3000..) every state it visits is kept, labelled with the rule's action (states the rule never reaches: the
    plain clone swung too late where its own path left the rule's, round 25 §5).  A swing row is written
    ``SWING_REPEAT`` times: a swing is 1% of the decisions, and cloned at that weight the policy never swung.
    Returns (rows, kept runs, runs played)."""
    env = Env('darknut', explicit_starts=True, strict_actions=False, seed=0)
    recorder = DemoRecorder()
    kept = played = 0
    seeds = range(1000, 1000 + runs) if policy is None else range(3000, 3000 + runs)
    for name in STATES:
        for seed in seeds:
            obs, _ = env.reset(seed=seed, options={'start_state': name})
            health, info = env.state.health, {}
            for _ in range(MAX_STEPS):
                ram = bytes(env.state.ram)
                step, free = label(ram, obs)
                if legal(obs['action_mask'], action_indices(step)):
                    recorder.add(obs, step, repeat=1 if step.startswith('MOVE') else SWING_REPEAT, any_duration=free)
                obs, _, terminated, truncated, info = env.step(step if policy is None else policy(obs))
                if terminated or truncated:
                    break
            played += 1
            if policy is not None or (info.get('outcome') == 'success' and env.state.health >= health):
                recorder.keep()
                kept += 1
            else:
                recorder.drop()
    return recorder.save(out), kept, played


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out', help="tracks: the output dir; --demos: the .npz file")
    parser.add_argument('count', nargs='?', type=int, default=None,
                        help="tracks per state (default 2); --demos: runs per state (default 200)")
    parser.add_argument('--demos', action='store_true', help="write demonstration rows instead of tracks")
    parser.add_argument('--policy', default=None, metavar='RUN_DIR',
                        help="--demos by DAgger: this checkpoint plays (temperature 0.5), the rule labels")
    args = parser.parse_args(argv)
    require_job()
    if args.demos:
        rows, kept, played = demos(Path(args.out), args.count or 200,
                                   sampler(args.policy) if args.policy else None)
        print(f"{args.out}: {rows} rows from {kept} of {played} runs")
        return 0
    if args.policy:
        parser.error("--policy goes with --demos")
    per_state = args.count or 2
    made = craft(Path(args.out), per_state)
    for track in made:
        print(track)
    return 0 if len(made) == per_state * len(STATES) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
