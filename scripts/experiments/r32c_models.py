"""Phase X's states step: the entry state after a room neither the explorer nor a rule crosses, out of a chain run the
promoted models won (from the magic sword on `d3first`'s Link holds what `d2first`'s does but for the candle's colour
and the rupees, so d2first's models play its rooms).

    python scripts/experiments/r32c_models.py STEM --out DIR [--chains 0,1,2] [--runs 30] [--seed0 1000]

For each chain: `triforce eval-chain --route d3first --from STEM_<chain> --stop-at step:<the next step> --runs RUNS
--replays DIR/replays/c<chain>` (the promoted runs of the checkout's `models/`, each in its act mode; a job in the
machine's line like every chain), then from the replays that reach the next step's room the pre-entry frame of that
entry (the last frame before the game makes the room's monsters, as `eval-chain --harvest` and `entry_states` save
it: `--harvest` itself saves only where the acting family changes, and 6:0x38 -> 0x28 is wizzrobe's both) is cut
out (`ReplayPlayer`) and written as the next step's entry state, hearts refilled (`entry_states.write_entry_state`).
The first won seed of a chain gives its state; every run counts for the win rate."""

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
WINDOW = 400                     # frames before the room event searched for the pre-entry frame


def is_pre_entry(ram) -> bool:
    mode = int(ram[MODE])
    return mode in TRANSITION_MODES or (mode in ENTER_MODES and ram[IS_UPDATING_MODE] == 0)


def route_goal(stem):
    """(the route step `stem` names, the next step on the clip way) of d3first."""
    from triforce import entry_states                           # pylint: disable=import-outside-toplevel
    way = [s for s in entry_states.route_steps(ROUTE) if s.way in (None, 'clips')]
    at = next(i for i, s in enumerate(way) if s.filename(0) == f'{ROUTE}_{stem}_0.state')
    return way[at], way[at + 1]


def play(stem, chain, goal, runs, seed0, out: Path) -> Path:
    """The chain runs of one chain from `stem`'s state to the next step's room; their replay directory."""
    replays = out / 'replays' / f'c{chain}'
    replays.mkdir(parents=True, exist_ok=True)
    output = out / f'runs-c{chain}.jsonl'
    output.unlink(missing_ok=True)
    command = [sys.executable, '-m', 'triforce.cli', 'eval-chain', '--route', ROUTE, '--from', f'{stem}_{chain}',
               '--stop-at', f'step:{goal.index}', '--runs', str(runs), '--parallel', '4', '--first-seed', str(seed0),
               '--replays', str(replays), '--output', str(output), '--no-log']
    print(' '.join(command[1:]), flush=True)
    subprocess.run(command, check=True)
    return replays


def cut(replay_path: Path, goal):
    """The pre-entry snapshots of the replay's entry into `goal`'s room (route step) with Link's hearts, the last
    frame first, or None when the run did not reach it."""
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
            frames.append((player.save_state(), (int(ram[HEART_VALUES]) & 0x0F) + int(ram[HEART_PARTIAL]) / 256))
        if player.step(1) == 0:
            break
    return frames[::-1]


def harvest(args) -> int:
    from triforce import entry_states                           # pylint: disable=import-outside-toplevel
    from triforce._native import Game, Snapshot                 # pylint: disable=import-outside-toplevel
    from triforce.paths import STATES_DIR                       # pylint: disable=import-outside-toplevel
    step, goal = route_goal(args.stem)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    command = ' '.join(['python'] + sys.argv)
    harvested, missed = [], []
    game = Game(seed=0, states_dir=str(STATES_DIR))
    for chain in args.chains:
        replays = play(args.stem, chain, goal, args.runs, args.seed0, out)
        files = sorted(replays.glob('*.tfreplay'), key=lambda p: int(p.name.split('-')[0]))
        won, entry = 0, None
        for path in files:
            candidates = cut(path, goal)
            if candidates is None:
                continue
            won += 1
            if entry is not None:
                continue
            target = out / goal.filename(chain)
            for machine, hearts in candidates:
                try:
                    entry_states.write_entry_state(game, Snapshot.from_state_file(machine), str(target), goal.location)
                except RuntimeError:
                    continue
                seed = int(path.name.split('-')[0])
                entry = {'file': target.name, 'chain': chain, 'route': ROUTE, 'route_step': goal.index,
                         'from_step': step.index, 'specialist': goal.specialist, 'level': goal.level,
                         'room': f'0x{goal.room:02x}', 'in_cave': goal.in_cave, 'entry': goal.entry,
                         'exit': goal.exit, 'alternate': False, 'steps': 0, 'frames': 0, 'health': round(hearts, 2),
                         'ending': 'exit-correct', 'env_steps': 0, 'attempts': files.index(path) + 1, 'seconds': 0.0,
                         'played_by': f'triforce eval-chain (the promoted models, seed {seed}): '
                                      f'scripts/experiments/r32c_models.py cut its entry into step {goal.index}'}
                entry_states._merge_manifest(str(out), [entry], command)     # pylint: disable=protected-access
                harvested.append(entry)
                break
        if entry is None:
            missed.append(goal.filename(chain))
        print(json.dumps({'step': args.stem, 'chain': chain, 'runs': len(files), 'won': won,
                          'harvested': entry['file'] if entry else None}), flush=True)
    entry_states._write_build_record(str(out), command, harvested, missed)  # pylint: disable=protected-access
    return 0 if not missed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('stem', help="the step's state without route or chain (6_38s)")
    parser.add_argument('--out', required=True)
    parser.add_argument('--chains', default='0,1,2', type=lambda s: [int(c) for c in s.split(',')])
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--seed0', type=int, default=1000)
    return harvest(parser.parse_args())


if __name__ == '__main__':
    sys.exit(main())
