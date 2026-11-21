"""Phase X's states step: the entry state after a room neither the explorer nor a rule crosses, out of a chain run the
promoted models won (from the magic sword on `d3first`'s Link holds what `d2first`'s does but for the candle's colour
and the rupees, so d2first's models play its rooms).

    python scripts/experiments/r32c_models.py STEM --out DIR [--chains 0,1,2] [--runs 30] [--seed0 1000]
        [--goal-step N] [--bombs-quantiles Q0,Q1,...] [--parallel 4]

For each chain: `triforce eval-chain --route d3first --from STEM_<chain> --stop-at step:<the goal step> --runs RUNS
--replays DIR/replays/c<chain>` (the promoted runs of the checkout's `models/`, each in its act mode; a job in the
machine's line like every chain), then from the replays that reach the goal step's room the pre-entry frame of that
entry (the last frame before the game makes the room's monsters, as `eval-chain --harvest` and `entry_states` save
it: `--harvest` itself saves only where the acting family changes, and 6:0x38 -> 0x28 is wizzrobe's both) is cut
out (`ReplayPlayer`) and written as the goal step's entry state, hearts refilled (`entry_states.write_entry_state`).
The goal step is the next one after STEM's unless `--goal-step` names a later one (round 34: Dungeon 4's entrance,
`3_7cs --goal-step 31`, the states as runs arrive from Dungeon 3's start). The first won seed of a chain gives its
state; every run counts for the win rate.

`--bombs-quantiles` pools the arrivals of every chain played and writes one state a quantile, chain i the i-th: the
first arrival (by seed) holding the bombs at that quantile of the arrivals' bombs (nearest rank), so the states hold
what runs bring (round 34: `0.5,0.1,0.9`, chain 0 the median, the one segment chains start from)."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROUTE = 'd3first'

# zelda::room_entry: the game modes of a room transition and of a room entry, $12 and $11 (IsUpdatingMode).
TRANSITION_MODES = {0x02, 0x03, 0x06, 0x07, 0x0A, 0x10}
ENTER_MODES = {0x04, 0x09, 0x0B, 0x0C}
MODE, IS_UPDATING_MODE, LEVEL, HEART_VALUES, HEART_PARTIAL = 0x12, 0x11, 0x10, 0x66F, 0x670
BOMBS, KEYS = 0x658, 0x66E
WINDOW = 400                     # frames before the room event searched for the pre-entry frame


def is_pre_entry(ram) -> bool:
    mode = int(ram[MODE])
    return mode in TRANSITION_MODES or (mode in ENTER_MODES and ram[IS_UPDATING_MODE] == 0)


def route_goal(stem, goal_step=None):
    """(the route step `stem` names, the goal: the next step on the clip way, or the step `goal_step` after it) of
    d3first."""
    from triforce import entry_states                           # pylint: disable=import-outside-toplevel
    way = [s for s in entry_states.route_steps(ROUTE) if s.way in (None, 'clips')]
    at = next(i for i, s in enumerate(way) if s.filename(0) == f'{ROUTE}_{stem}_0.state')
    if goal_step is None:
        return way[at], way[at + 1]
    goal = next((s for s in way[at + 1:] if s.index == goal_step), None)
    if goal is None:
        raise ValueError(f"step {goal_step} is not on {ROUTE}'s way after {stem} (step {way[at].index})")
    return way[at], goal


def pick_by_bombs(arrivals, quantiles):
    """For each quantile, the first arrival (in the given order) holding the bombs at that quantile of the arrivals'
    bombs (nearest rank), each arrival used once (a later one with the same bombs, else the nearest count); None
    without arrivals.  `arrivals`: (key, bombs) pairs."""
    if not arrivals:
        return [None] * len(quantiles)
    ranked = sorted(bombs for _, bombs in arrivals)
    used, picks = set(), []
    for q in quantiles:
        want = ranked[round(q * (len(ranked) - 1))]
        free = [(i, a) for i, a in enumerate(arrivals) if i not in used]
        i, arrival = min(free, key=lambda f: (abs(f[1][1] - want), f[0]))
        used.add(i)
        picks.append(arrival)
    return picks


def play(stem, chain, goal, runs, seed0, out: Path, parallel=4) -> Path:
    """The chain runs of one chain from `stem`'s state to the goal step's room; their replay directory."""
    replays = out / 'replays' / f'c{chain}'
    replays.mkdir(parents=True, exist_ok=True)
    output = out / f'runs-c{chain}.jsonl'
    output.unlink(missing_ok=True)
    command = [sys.executable, '-m', 'triforce.cli', 'eval-chain', '--route', ROUTE, '--from', f'{stem}_{chain}',
               '--stop-at', f'step:{goal.index}', '--runs', str(runs), '--parallel', str(parallel),
               '--first-seed', str(seed0), '--replays', str(replays), '--output', str(output), '--no-log']
    print(' '.join(command[1:]), flush=True)
    subprocess.run(command, check=True)
    return replays


def cut(replay_path: Path, goal):
    """The pre-entry snapshots of the replay's entry into `goal`'s room (route step) as (machine state, Link's
    hearts, bombs, keys), the last frame first, or None when the run did not reach it."""
    from triforce import _native                                # pylint: disable=import-outside-toplevel
    data = replay_path.read_bytes()
    entries = [e for e in _native.Replay(data).events() if e.get('type') == 'room' and e.get('step') == goal.index]
    if not entries:
        return None
    entry = entries[0]
    player = _native.ReplayPlayer(data)
    start = max(0, entry['frame'] - WINDOW)
    player.seek(start)
    frames = []
    for frame in range(start, entry['frame'] + 1):
        ram = player.ram()
        if is_pre_entry(ram) and int(ram[LEVEL]) == goal.level:
            frames.append((player.save_state(), (int(ram[HEART_VALUES]) & 0x0F) + int(ram[HEART_PARTIAL]) / 256,
                           int(ram[BOMBS]), int(ram[KEYS])))
        if player.step(1) == 0:
            break
    return frames[::-1]


def manifest_entry(target: Path, chain, step, goal, hearts, seed, attempts, bombs, keys) -> dict:
    return {'file': target.name, 'chain': chain, 'route': ROUTE, 'route_step': goal.index,
            'from_step': step.index, 'specialist': goal.specialist, 'level': goal.level,
            'room': f'0x{goal.room:02x}', 'in_cave': goal.in_cave, 'entry': goal.entry,
            'exit': goal.exit, 'alternate': False, 'steps': 0, 'frames': 0, 'health': round(hearts, 2),
            'ending': 'exit-correct', 'env_steps': 0, 'attempts': attempts, 'seconds': 0.0,
            'played_by': f'triforce eval-chain (the promoted models, seed {seed}, {bombs} bombs, {keys} keys): '
                         f'scripts/experiments/r32c_models.py cut its entry into step {goal.index}'}


def write(game, candidates, target: Path, goal):
    """Writes the first candidate that loads into the goal's room; its (hearts, bombs, keys), or None."""
    from triforce import entry_states                           # pylint: disable=import-outside-toplevel
    from triforce._native import Snapshot                       # pylint: disable=import-outside-toplevel
    for machine, hearts, bombs, keys in candidates:
        try:
            entry_states.write_entry_state(game, Snapshot.from_state_file(machine), str(target), goal.location)
        except RuntimeError:
            continue
        return hearts, bombs, keys
    return None


def replay_files(replays: Path):
    return sorted(replays.glob('*.tfreplay'), key=lambda p: int(p.name.split('-')[0]))


def harvest(args) -> int:
    from triforce import entry_states                           # pylint: disable=import-outside-toplevel
    from triforce._native import Game                           # pylint: disable=import-outside-toplevel
    from triforce.paths import STATES_DIR                       # pylint: disable=import-outside-toplevel
    step, goal = route_goal(args.stem, args.goal_step)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    command = ' '.join(['python'] + sys.argv)
    harvested, missed = [], []
    game = Game(seed=0, states_dir=str(STATES_DIR))
    if args.bombs_quantiles:
        return harvest_by_bombs(args, step, goal, out, command, game)
    for chain in args.chains:
        files = replay_files(play(args.stem, chain, goal, args.runs, args.seed0, out, args.parallel))
        won, entry = 0, None
        for path in files:
            candidates = cut(path, goal)
            if candidates is None:
                continue
            won += 1
            if entry is not None:
                continue
            target = out / goal.filename(chain)
            written = write(game, candidates, target, goal)
            if written is None:
                continue
            entry = manifest_entry(target, chain, step, goal, written[0], int(path.name.split('-')[0]),
                                   files.index(path) + 1, *written[1:])
            entry_states._merge_manifest(str(out), [entry], command)     # pylint: disable=protected-access
            harvested.append(entry)
        if entry is None:
            missed.append(goal.filename(chain))
        print(json.dumps({'step': args.stem, 'chain': chain, 'runs': len(files), 'won': won,
                          'harvested': entry['file'] if entry else None}), flush=True)
    entry_states._write_build_record(str(out), command, harvested, missed)  # pylint: disable=protected-access
    return 0 if not missed else 1


def harvest_by_bombs(args, step, goal, out: Path, command: str, game) -> int:
    """`--bombs-quantiles`: every played chain's arrivals pooled; chain i's state the i-th quantile's pick."""
    from triforce import entry_states                           # pylint: disable=import-outside-toplevel
    arrivals, runs = [], 0
    for chain in args.chains:
        files = replay_files(play(args.stem, chain, goal, args.runs, args.seed0, out, args.parallel))
        runs += len(files)
        for path in files:
            candidates = cut(path, goal)
            if candidates:
                arrivals.append(((chain, int(path.name.split('-')[0]), path), candidates[0][2]))
    counts = {}
    for _, bombs in arrivals:
        counts[bombs] = counts.get(bombs, 0) + 1
    print(json.dumps({'step': args.stem, 'goal': goal.index, 'runs': runs, 'arrived': len(arrivals),
                      'bombs': dict(sorted(counts.items()))}), flush=True)
    harvested, missed = [], []
    for out_chain, pick in enumerate(pick_by_bombs(arrivals, args.bombs_quantiles)):
        target = out / goal.filename(out_chain)
        written = None
        if pick is not None:
            (_, seed, path), _ = pick
            written = write(game, cut(path, goal), target, goal)
        if written is None:
            missed.append(target.name)
            continue
        entry = manifest_entry(target, out_chain, step, goal, written[0], seed, len(arrivals), *written[1:])
        entry_states._merge_manifest(str(out), [entry], command)         # pylint: disable=protected-access
        harvested.append(entry)
        print(json.dumps({'file': target.name, 'seed': seed, 'quantile': args.bombs_quantiles[out_chain],
                          'hearts_in': round(written[0], 2), 'bombs': written[1], 'keys': written[2]}), flush=True)
    entry_states._write_build_record(str(out), command, harvested, missed)  # pylint: disable=protected-access
    return 0 if not missed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('stem', help="the step's state without route or chain (6_38s)")
    parser.add_argument('--out', required=True)
    parser.add_argument('--chains', default='0,1,2', type=lambda s: [int(c) for c in s.split(',')])
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--seed0', type=int, default=1000)
    parser.add_argument('--parallel', type=int, default=4)
    parser.add_argument('--goal-step', type=int, default=None,
                        help="the route step whose entry is cut (default: the step after STEM's)")
    parser.add_argument('--bombs-quantiles', default=None, type=lambda s: [float(q) for q in s.split(',')],
                        help="pool the chains' arrivals; chain i's state holds the bombs at the i-th quantile")
    return harvest(parser.parse_args())


if __name__ == '__main__':
    sys.exit(main())
