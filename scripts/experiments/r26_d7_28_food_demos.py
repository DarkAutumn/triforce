"""Demonstration rows for 7:0x28, the hungry Goriya's room (docs/routes/d7-walks.md §10): walk up, give the food
the moment FOOD is offered, walk on through the N door.

    python scripts/experiments/r26_d7_28_food_demos.py OUT.npz [RUNS_PER_STATE]

Round 26: the goriya family's first training (`r76-goriya`, 488 iterations, the explorer's tracks as curriculum)
learned every Blue Goriya room and left this one at 1 of 600: 598 episodes ran out of steps without one FOOD.  The
action is new to the family and nothing in the room punishes waiting.  `train --demos FILE --demo-coeff C` clones
the rule's decisions during PPO (`triforce.demos`, as round 25 did for 5:0x37's column).

The rule: `FOOD N` where the mask offers it (else the first FOOD direction offered), otherwise `MOVE N`.  A FOOD
row is written `FOOD_REPEAT` times (it is one decision of a run); a MOVE N's duration is not cloned.  Only runs
that leave by the N door are kept."""

import sys
from pathlib import Path

from triforce._native import Env
from triforce.demos import DemoRecorder, action_indices, legal

STATES = ('7_28s_0', '7_28s_1', '7_28s_2')
MAX_STEPS = 120
FOOD_REPEAT = 40


def action(obs: dict) -> str:
    for direction in 'NWES':
        if legal(obs['action_mask'], action_indices(f'FOOD {direction}')):
            return f'FOOD {direction}'
    return 'MOVE N 8'


def demos(out: Path, runs: int) -> tuple[int, int, int]:
    """(rows, kept runs, runs played)."""
    env = Env('goriya', explicit_starts=True, strict_actions=False, seed=0)
    recorder = DemoRecorder()
    kept = played = 0
    for name in STATES:
        for seed in range(1000, 1000 + runs):
            obs, _ = env.reset(seed=seed, options={'start_state': name})
            info, fed = {}, False
            for _ in range(MAX_STEPS):
                step = action(obs)
                if legal(obs['action_mask'], action_indices(step)):
                    food = step.startswith('FOOD')
                    fed = fed or food
                    recorder.add(obs, step, repeat=FOOD_REPEAT if food else 1, any_duration=not food)
                obs, _, terminated, truncated, info = env.step(step)
                if terminated or truncated:
                    break
            played += 1
            if info.get('outcome') == 'success' and fed:
                recorder.keep()
                kept += 1
            else:
                recorder.drop()
    return recorder.save(out), kept, played


def main(argv) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    rows, kept, played = demos(Path(argv[0]), int(argv[1]) if len(argv) > 1 else 100)
    print(f"{rows} rows from {kept} of {played} runs -> {argv[0]}")
    return 0 if kept else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
