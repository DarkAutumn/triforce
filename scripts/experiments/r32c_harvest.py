"""Phase X's states step: the entry state after a room the explorer does not clear, out of a run the walks' rule won
(`docs/routes/d3first.md` "For the agents that build d3first's states"; round 29's `--harvest`, docs/experiments/
round29.md §9).

    python scripts/experiments/r32c_harvest.py STEP --out DIR [--chains 0,1,2] [--seeds N] [--seed0 S]
    python -m triforce.entry_states install DIR

STEP is the walks' name of the room's state (`3_6bs`): its installed `d3first_<STEP>_<chain>.state` is loaded for
seeds S.. (the game's own randomness, `r27_walks.load`), the walks' actions for the step played live
(`r32_d3first.LEGS`, the walks' best variant where they found one, `BEST`: `r32_walks.Fighter`, `BombFighter`,
`ManhandlaFighter`, round 28's Gleeok fight), and of each chain the first won run's pre-entry frame into the route's
next step is written as that step's entry state (hearts refilled, the room fresh: `entry_states.write_entry_state`),
staged in DIR with its manifest entry (`played_by` names the rule and the seed).  Every seed is played: the line
per chain says how often the rule won.  Nothing in the checkout changes until `install`."""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from r32_d3first import BEST, LEGS, VARIANTS  # noqa: E402  pylint: disable=wrong-import-position
from r32_walks import PLAY, CAVE, CELLAR, Failed, Runner, hp  # noqa: E402  pylint: disable=wrong-import-position
from r27_walks import load  # noqa: E402  pylint: disable=wrong-import-position

ROUTE = 'd3first'


class Harvester(Runner):
    """A live `Runner` that keeps the pre-entry snapshot of the step's last action instead of writing it."""

    pre_entry = None

    def _save(self, save, snapshot):
        if save is not None:
            if snapshot is None:
                raise Failed('no pre-entry frame')
            self.pre_entry = snapshot


def step_actions(stem):
    """The walks' actions for `stem`, their best variant where they found one."""
    actions = dict(step for leg in LEGS.values() for step in leg)[stem]
    return VARIANTS[stem][BEST[stem]] if stem in BEST else actions


def route_goal(stem):
    """(the route step `stem` names, the next step on the clip way) of d3first."""
    from triforce import entry_states                           # pylint: disable=import-outside-toplevel
    way = [s for s in entry_states.route_steps(ROUTE) if s.way in (None, 'clips')]
    at = next(i for i, s in enumerate(way) if s.filename(0) == f'{ROUTE}_{stem}_0.state')
    return way[at], way[at + 1]


def harvest(args):
    from triforce import entry_states                           # pylint: disable=import-outside-toplevel
    from triforce.paths import STATES_DIR                       # pylint: disable=import-outside-toplevel
    step, goal = route_goal(args.step)
    actions = step_actions(args.step)
    rule = f"{args.step} {'(' + BEST[args.step] + ')' if args.step in BEST else ''}".strip()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    command = ' '.join(['python'] + sys.argv)
    harvested, missed = [], []
    for chain in args.chains:
        source = Path(STATES_DIR) / step.filename(chain)
        wins, why, lost, entry = 0, collections.Counter(), [], None
        for seed in range(args.seed0, args.seed0 + args.seeds):
            w = load(source, seed)
            w.until(lambda w: w.mode in (PLAY, CAVE, CELLAR), 'NONE', 2000)
            run = Harvester(w, stage=False)
            h0 = hp(w)
            try:
                run.play(actions, save_next=out / '.pre-entry')
            except Failed as e:
                why[str(e).split(' at ')[0][:60]] += 1
                continue
            if (w.level, w.room, w.mode == CAVE or w.mode == CELLAR) != goal.location.as_tuple():
                why[f'left into {w.level}:0x{w.room:02x}'] += 1
                continue
            wins += 1
            lost.append(h0 - hp(w))
            if entry is None:
                path = out / goal.filename(chain)
                entry_states.write_entry_state(w.game, run.pre_entry, str(path), goal.location)
                source_entry = entry_states.read_manifest(str(STATES_DIR)).get(source.name, {})
                entry = {'file': path.name, 'chain': chain, 'route': ROUTE, 'route_step': goal.index,
                         'from_step': step.index, 'specialist': goal.specialist, 'level': goal.level,
                         'room': f'0x{goal.room:02x}', 'in_cave': goal.in_cave, 'entry': goal.entry,
                         'exit': goal.exit, 'alternate': False, 'steps': 0, 'frames': run.frames(), 'health': h0,
                         'ending': 'exit-correct', 'env_steps': 0, 'attempts': seed - args.seed0 + 1,
                         'seconds': 0.0,
                         'played_by': f'scripts/experiments/r32c_harvest.py {rule} (seed {seed}, the walks\' rule)'}
                if source_entry.get('fight_skipped'):
                    entry['fight_skipped'] = True
                entry_states._merge_manifest(str(out), [entry], command)     # pylint: disable=protected-access
                harvested.append(entry)
        if entry is None:
            missed.append(step.filename(chain))
        print(json.dumps({'step': args.step, 'rule': rule, 'chain': chain, 'runs': args.seeds, 'won': wins,
                          'hearts_lost_in_wins': round(sum(lost) / len(lost), 2) if lost else None,
                          'failed': dict(why), 'harvested': entry['file'] if entry else None}), flush=True)
    entry_states._write_build_record(str(out), command, harvested, missed)  # pylint: disable=protected-access
    (out / '.pre-entry').unlink(missing_ok=True)
    return 0 if not missed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('step', help="the walks' state name of the room (r32_d3first.LEGS), e.g. 3_6bs")
    parser.add_argument('--out', required=True, help='staging directory (for entry_states install)')
    parser.add_argument('--chains', default='0,1,2')
    parser.add_argument('--seeds', type=int, default=10, help='runs a chain (every one is played)')
    parser.add_argument('--seed0', type=int, default=1000)
    args = parser.parse_args()
    args.chains = [int(c) for c in args.chains.split(',')]
    return harvest(args)


if __name__ == '__main__':
    sys.exit(main())
