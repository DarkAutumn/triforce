"""Round 25 (docs/experiments/round25.md, "Rooms that bleed, Dungeon 5"): hand-made curriculum tracks for level 5
room 0x37 entered from S and left N.

    python scripts/experiments/r25_d5_37_tracks.py OUT_DIR [TRACKS_PER_STATE]

The room's objective goes up the column X $70 (game.yaml ``path_costs``): Link crosses the water once on the ladder
and meets the pocket's Darknut from below, where a swing reaches it.  The explorer's own tracks walk X $78 and take
a hit on nearly every visit, so a model trained on them never sees the cheap way.  This script plays a fixed rule
(walk to X $70, north, SWORD N whenever a live monster stands just ahead, then back east to the door) from the
start states ``5_37s_0..2`` over seeds 1000.. and keeps, per state, the first ``TRACKS_PER_STATE`` (default 2) runs
that leave by the N door without losing health and with at least one swing.  A track is written as the curriculum
reads it: ``OUT_DIR/5_37s_<chain>~x70s<seed>/NNNN.state`` and ``trajectory.json``.  Put them in
``trajectories/darknut/`` in place of the explorer's ``5_37s_<chain>`` tracks.
"""

import json
import sys
from pathlib import Path

from triforce._native import Env

OBJ_ID, OBJ_X, OBJ_Y = 0x34F, 0x70, 0x84                # slot 0 is Link, 1..11 the monsters
OBJ_INVINCIBLE, OBJ_METASTATE = 0x4F0, 0x405            # flashing after a hit; a spawn cloud or a death
DARKNUT = 0x0B
COLUMN = 0x70
STATES = ('5_37s_0', '5_37s_1', '5_37s_2')
SEEDS = range(1000, 3000)
MAX_STEPS = 80


def monster_ahead(ram: bytes) -> bool:
    """A live monster that isn't flashing stands within a swing north of Link."""
    x, y = ram[OBJ_X], ram[OBJ_Y]
    for slot in range(1, 12):
        kind = ram[OBJ_ID + slot]
        if kind and kind < 0x40 and not ram[OBJ_METASTATE + slot] and not ram[OBJ_INVINCIBLE + slot]:
            forward, sideways = y - ram[OBJ_Y + slot], ram[OBJ_X + slot] - x
            if -2 <= forward <= 22 and abs(sideways) <= 12:
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


def main(argv) -> int:
    if not argv or len(argv) > 2:
        print(__doc__, file=sys.stderr)
        return 2
    per_state = int(argv[1]) if len(argv) > 1 else 2
    made = craft(Path(argv[0]), per_state)
    for track in made:
        print(track)
    return 0 if len(made) == per_state * len(STATES) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
