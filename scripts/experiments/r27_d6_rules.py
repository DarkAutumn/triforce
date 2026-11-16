"""Round 27 (docs/experiments/round27.md "Demonstrations"): three scripted rules of the stretch, played through the
family's own environment in its own actions, as demonstration rows for ``train --demos`` and as curriculum tracks.

    python scripts/experiments/r27_d6_rules.py play RULE [--runs N]              # the rule's results, nothing written
    python scripts/experiments/r27_d6_rules.py demos RULE OUT.npz [--runs N]     # rows of its clean runs
    python scripts/experiments/r27_d6_rules.py tracks RULE OUT_DIR [--per-state N]   # tracks of its clean runs

RULE (the walks behind each: docs/routes/d6-walks.md §2, §4.1 #7, §6):

- ``gravestone`` (ow-ghini, ``0_21s_<chain>``, 0:0x21 from S to its cave): the objective leads to the square under
  the one stone that moves; there the mask offers MOVE N and the rule plays ``MOVE N 32`` (the executor holds it
  through the slide and ends it on the stairs).
- ``bomb28`` (wizzrobe, ``6_28s_<chain>``, 6:0x28 from S out E through its bomb wall): the objective leads along the
  bottom row and up to the wall's spot; ``BOMB E`` the first time the mask offers it, then on through the hole.
- ``gohma`` (boss-gohma, ``6_1cs_<chain>``, 6:0x1c): the objective leads to the stand tile under Gohma; ``ARROW N``
  when the eye is fully open with frames to spare and Link's X − Gohma's X is inside the window; then the heart
  container and the N door by the objective.

Every rule walks by the observation's own ``target`` direction (the next step of the objective's path, the input
the model reads: docs/architecture.md "Observation"), ``MOVE <dir> 4`` a decision, and never fights.  A run is
**clean** when the episode ends in success and Link lost no more than ``--max-loss`` hearts (default: the rule's
own, below); only clean runs' decisions are written.  Rows: a plain walking MOVE is cloned without its duration
(``ANY_DURATION``: the rule's short steps are not the lesson, round 25 §5); the rule's own decision (the push, the
bomb, the arrow, a WAIT under Gohma) is cloned in full and written ``REPEAT`` times, since it is one decision among
dozens.  Each start state is played over seeds 1000.. with the reset's own hp and bombs draws (the family's
``ram_randomization``), so the rows cover what a run brings.

``tracks`` writes, per start state, the first N clean runs as the curriculum reads them:
``OUT_DIR/<state>~rule<seed>/NNNN.state`` and ``trajectory.json`` (as ``r25_d5_37_tracks.py``).
"""

import argparse
import collections
import json
import sys
from pathlib import Path

from triforce._native import Env
from triforce.demos import DemoRecorder, action_indices, legal

OBJ_X, OBJ_Y, OBJ_TYPE, OBJ_METASTATE = 0x70, 0x84, 0x34F, 0x405      # slot 0 is Link, 1..11 the monsters
GOHMA_TYPES = (0x33, 0x34)                                           # blue and red Gohma
GOHMA_EYE, GOHMA_EYE_TIMER = 0x46B, 0x444                            # Gohma_ObjEyeState (3: fully open), its timer
STAND_ROWS = (0xA5, 0xAD)                                            # Link's Y on the stand tile's row
CUR_OPENED_DOORS, DOOR_E = 0xEE, 0x01                                # the room's opened doors; 6:0x28's E hole
BOMB_STATES = (0xAC + 0x10, 0xAC + 0x11)                             # ObjState of the two bomb slots (0: no bomb)
BOMB_SPOT, BOTTOM_ROW = (0xD0, 0x8D), 0xBD                           # 6:0x28: the E wall's door spot; the walked row
DIRECTIONS = ('N', 'S', 'W', 'E')
SEEDS = 1000


def target_direction(obs) -> str | None:
    """The objective's next step from Link's tile (``target_N/S/W/E``), None on a target tile or without a path."""
    row = [float(v) for v in obs['target'][:4]]
    return DIRECTIONS[row.index(max(row))] if max(row) > 0.5 else None


def offered(obs, action: str) -> bool:
    """Whether the mask allows the action."""
    return legal(obs['action_mask'], action_indices(action))


def target_offset(obs) -> tuple[int, int]:
    """Pixels from Link to the target tile his path reaches (``target_dx``, ``target_dy``)."""
    return round(float(obs['target'][5]) * 256), round(float(obs['target'][6]) * 176)


def walk(obs) -> str:
    """One step along the objective's path; on the target's own tile (no next step) a step that closes what is left
    of the pixel offset (an item is taken by touching it, a stand square is a pixel position); else a WAIT."""
    direction = target_direction(obs)
    if direction is None:
        dx, dy = target_offset(obs)
        if abs(dx) >= 2:
            direction = 'E' if dx > 0 else 'W'
        elif abs(dy) >= 2:
            direction = 'S' if dy > 0 else 'N'
    if direction is not None and offered(obs, f'MOVE {direction} 4'):
        return f'MOVE {direction} 4'
    return 'WAIT 4'


def gravestone(ram: bytes, obs, memory: dict) -> tuple[str, bool]:
    """(action, whether it is the rule's own decision).  The push is played within ``PUSH_SLACK`` pixels of the
    stand square's column: Link turns up on the 8-pixel grid, and from further off he snaps to the next column."""
    if target_direction(obs) is None and offered(obs, 'MOVE N 32') and abs(target_offset(obs)[0]) <= PUSH_SLACK:
        return 'MOVE N 32', True
    return walk(obs), False


def bomb28(ram: bytes, obs, memory: dict) -> tuple[str, bool]:
    """The walked crossing (d6-walks.md §4.1 #7): along the bottom row to the column X $D0, up it to the door row,
    the first BOMB E the mask offers there (a `required` family: only where the bomb opens the route's wall), a WAIT
    on the spot until the hole is open, then through by the objective.  Knocked off the line, Link walks back onto
    it."""
    x, y = ram[OBJ_X], ram[OBJ_Y]
    if ram[CUR_OPENED_DOORS] & DOOR_E:
        return walk(obs), False
    at_spot = abs(x - BOMB_SPOT[0]) <= 4 and abs(y - BOMB_SPOT[1]) <= 4
    if memory.get('bombed') and (ram[BOMB_STATES[0]] or ram[BOMB_STATES[1]]):     # the bomb lies or blasts
        return ('WAIT 4', False) if at_spot else (walk(obs), False)
    if x >= BOMB_SPOT[0] - 4 and y <= BOMB_SPOT[1] + 4 and offered(obs, 'BOMB E'):
        memory['bombed'] = True
        return 'BOMB E', True
    if x < BOMB_SPOT[0] - 3:
        step = 'MOVE S 4' if y < BOTTOM_ROW - 5 else 'MOVE E 4'
    elif x > BOMB_SPOT[0] + 3:
        step = 'MOVE W 4'
    else:
        step = 'MOVE N 4' if y > BOMB_SPOT[1] + 3 else walk(obs)
    return (step, False) if offered(obs, step) else (walk(obs), False)


def gohma_slot(ram: bytes) -> int | None:
    """The slot of a living Gohma."""
    return next((slot for slot in range(1, 12)
                 if ram[OBJ_TYPE + slot] in GOHMA_TYPES and not ram[OBJ_METASTATE + slot]), None)


def gohma(ram: bytes, obs, memory: dict) -> tuple[str, bool]:
    """ARROW N on the stand row when the eye is fully open with more than ``EYE_SPARE`` frames left and Link's X −
    Gohma's X is in ``WINDOW``; else to the stand tile; waits there for the eye."""
    slot = gohma_slot(ram)
    if slot is None:
        return walk(obs), False
    dx = ram[OBJ_X] - ram[OBJ_X + slot]
    open_eye = ram[GOHMA_EYE + slot] == 3 and ram[GOHMA_EYE_TIMER + slot] > EYE_SPARE
    if (STAND_ROWS[0] <= ram[OBJ_Y] <= STAND_ROWS[1] and open_eye and WINDOW[0] <= dx <= WINDOW[1]
            and offered(obs, 'ARROW N')):
        return 'ARROW N', True
    return walk(obs), False


PUSH_SLACK = 3
EYE_SPARE = 22
WINDOW = (-9, 4)          # the walked kill window is [-10, +5] when the arrow arrives; Gohma drifts half a pixel a frame
RULES = {
    # name: (family, state stem, the rule, step limit, hearts a clean run may lose, the own decision's row weight)
    'gravestone': ('ow-ghini', '0_21s', gravestone, 150, 0.0, 20),
    'bomb28': ('wizzrobe', '6_28s', bomb28, 200, 0.5, 20),
    'gohma': ('boss-gohma', '6_1cs', gohma, 300, 1.0, 20),
}


def episode(env, name: str, seed: int, rule, limit: int, recorder: DemoRecorder | None = None, repeat: int = 1,
            points: list | None = None) -> dict:
    """Plays one episode by ``rule``; returns its result.  ``recorder`` takes the decisions (pending until the caller
    keeps or drops them); ``points`` the snapshot before every step."""
    obs, _ = env.reset(seed=seed, options={'start_state': name})
    health, containers, info, memory, actions, own = env.state.health, env.state.max_health, {}, {}, [], 0
    for _ in range(limit):
        step, mine = rule(bytes(env.state.ram), obs, memory)
        if recorder is not None:
            recorder.add(obs, step, repeat=repeat if mine else 1, any_duration=not mine)
        if points is not None:
            points.append(env.snapshot())
        actions.append(step)
        own += mine
        obs, _, terminated, truncated, info = env.step(step)
        if terminated or truncated:
            break
    # Hearts lost: a heart container taken on the way (Gohma's) adds a heart, which is no healing.
    lost = health - env.state.health + (env.state.max_health - containers)
    return {'success': info.get('outcome') == 'success', 'ending': info.get('ending'), 'steps': len(actions),
            'lost': lost, 'own': own, 'actions': actions, 'health': health, 'frames': info.get('total_frames')}


def states(stem: str) -> list[str]:
    return [f'{stem}_{chain}' for chain in (0, 1, 2)]


def play(name: str, runs: int, max_loss: float | None, out_demos: Path | None = None) -> int:
    """Plays the rule ``runs`` times a start state; prints the results; with ``out_demos`` writes the clean runs'
    rows.  Returns the number of clean runs."""
    family, stem, rule, limit, loss, repeat = RULES[name]
    loss = loss if max_loss is None else max_loss
    env = Env(family, explicit_starts=True, strict_actions=False, seed=0)
    recorder = DemoRecorder() if out_demos is not None else None
    clean_total = 0
    for state in states(stem):
        endings, lost, steps, own, clean = collections.Counter(), [], [], collections.Counter(), 0
        for seed in range(SEEDS, SEEDS + runs):
            result = episode(env, state, seed, rule, limit, recorder, repeat)
            endings[result['ending'] if result['ending'] else 'step-limit'] += 1
            own[result['own']] += 1
            good = result['success'] and result['lost'] <= loss
            if result['success']:
                lost.append(result['lost'])
                steps.append(result['steps'])
            clean += good
            if recorder is not None:
                recorder.keep() if good else recorder.drop()         # pylint: disable=expression-not-assigned
        clean_total += clean
        mean = (lambda v: sum(v) / len(v) if v else float('nan'))
        print(f"{name} {state}: {sum(n for e, n in endings.items() if e and 'exit-correct' in e or e == 'success')} "
              f"of {runs} by the ending, endings {dict(endings)}; successes lost {mean(lost):.2f} hearts "
              f"(max {max(lost, default=0):.1f}), {mean(steps):.0f} steps; own decisions a run {dict(own)}; "
              f"clean (success, at most {loss} hearts lost) {clean}")
    if recorder is not None:
        rows = recorder.save(out_demos)
        print(f"{out_demos}: {rows} rows from {clean_total} clean runs of {runs * 3}")
    return clean_total


def tracks(name: str, out: Path, per_state: int, max_loss: float | None) -> int:
    """Writes the first ``per_state`` clean runs of each start state as curriculum tracks; returns how many."""
    family, stem, rule, limit, loss, _ = RULES[name]
    loss = loss if max_loss is None else max_loss
    env = Env(family, explicit_starts=True, strict_actions=False, seed=0)
    made = 0
    for state in states(stem):
        found = 0
        for seed in range(SEEDS, SEEDS + 400):
            points = []
            result = episode(env, state, seed, rule, limit, points=points)
            if not (result['success'] and result['lost'] <= loss):
                continue
            track = out / f'{state}~rule{seed}'
            track.mkdir(parents=True, exist_ok=True)
            for index, point in enumerate(points[:result['steps']]):
                (track / f'{index:04d}.state').write_bytes(point.state_file())
            (track / 'trajectory.json').write_text(json.dumps({
                'start_state': state, 'success': True, 'ending': result['ending'], 'steps': result['steps'],
                'health': result['health'], 'actions': result['actions'],
                'explore': {'iterations': 0, 'env_steps': result['steps'], 'cells': 0, 'seconds': 0.0, 'seed': seed},
                'crafted': f'round 27: the scripted rule {name} (scripts/experiments/r27_d6_rules.py)'}, indent=1))
            print(f"{track}: {result['steps']} steps, {result['lost']} hearts lost")
            found += 1
            made += 1
            if found >= per_state:
                break
        if found < per_state:
            print(f"{state}: only {found} of {per_state} tracks", file=sys.stderr)
    return made


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('play', 'demos', 'tracks'):
        one = sub.add_parser(command)
        one.add_argument('rule', choices=sorted(RULES))
        if command != 'play':
            one.add_argument('out')
        one.add_argument('--max-loss', type=float, default=None, help="hearts a clean run may lose (default: the rule's)")
        if command == 'tracks':
            one.add_argument('--per-state', type=int, default=1)
        else:
            one.add_argument('--runs', type=int, default=40 if command == 'play' else 200,
                             help="episodes a start state")
    args = parser.parse_args(argv)
    if args.command == 'tracks':
        return 0 if tracks(args.rule, Path(args.out), args.per_state, args.max_loss) == 3 * args.per_state else 1
    clean = play(args.rule, args.runs, args.max_loss, Path(args.out) if args.command == 'demos' else None)
    return 0 if clean else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
