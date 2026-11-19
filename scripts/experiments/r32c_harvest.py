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
per chain says how often the rule won.  Nothing in the checkout changes until `install`.

A room the walks fought with a rule of their second half (`FIGHTS`: `r32b_d5.py`'s Olive Garden and Digdogger,
`r32b_d7.py`'s 0x49, 0x1b, 0x1c and Aquamentus) has
no scripted way out of the room: the rule fights until the room is clear, and the explorer (`entry_states`'s, the
step's own task, the room as the fight left it: `fresh_rooms: false`) goes on from there to the step's exit; the
win rate is the rule's (clears of the seeds played), the state the first cleared run the explorer took out.

The leg after Dungeon 5 (the recorder to Level 3's landing, 0x74 ... 0x3c into Level 2: `tallies()`) has the walks'
rules of `r32b_leg.py` (one `Tally` function a room, monsters live) and `r32b_recorder.ride`, not `LEGS` steps: the
rule plays the room and the pre-entry frame of the next room (the scroll's or the cave's, a wrap exit's from the
engine's macro) is the next step's state."""

import argparse
import collections
import functools
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


# Where the harvest plays a step otherwise than the walks wrote it.  4:0x13: after the fight the fighter's floor graph
# lacks the room's middle and corners (every won fight failed there: d3first-walks-a.md §8 "a script fault"; here the
# heart's spot and ($78,$5D) both), so Link goes by buttons alone: down the W side to the bottom row, along it onto
# the heart container at ($D0,$BD), back along it and up the middle to the N door.  The fight is the walks' (round
# 28's GleeokFighter).
OVERRIDES = {'4_13w': [('clear', 'gleeok'), ('straight', ('y', 0xD0, 0xBD)), ('wait', 30), ('straight', (0x78, 0x5D)),
                       ('go', 'UP', ('x', 0x78))]}


def step_actions(stem):
    """The walks' actions for `stem`, their best variant where they found one (`OVERRIDES` first)."""
    if stem in OVERRIDES:
        return OVERRIDES[stem]
    actions = dict(step for leg in LEGS.values() for step in leg)[stem]
    return VARIANTS[stem][BEST[stem]] if stem in BEST else actions


def route_goal(stem):
    """(the route step `stem` names, the next step on the clip way) of d3first."""
    from triforce import entry_states                           # pylint: disable=import-outside-toplevel
    way = [s for s in entry_states.route_steps(ROUTE) if s.way in (None, 'clips')]
    at = next(i for i, s in enumerate(way) if s.filename(0) == f'{ROUTE}_{stem}_0.state')
    return way[at], way[at + 1]


def _garden(w):
    import r32b_d5                                              # pylint: disable=import-outside-toplevel
    return r32b_d5.BombFighter(w).fight(9000)


def _digdogger(w):
    import r32b_d5                                              # pylint: disable=import-outside-toplevel
    from r26_walks import DIGDOGGER_SMALL                       # pylint: disable=import-outside-toplevel
    w.select_b(r32b_d5.RECORDER_SLOT)
    w.press('LEFT', 6)
    w.press('B', 1)
    w.until(lambda w: any(o[1] == DIGDOGGER_SMALL for o in w.objects()), 'NONE', 400)
    return r32b_d5.DigFighter(w).fight(6000)


def _d7(targets, recorder=False, kills=None):
    """r32b_d7.py's fight of a Dungeon 7 room (`Fight`: round 28's fighter at `targets`), the recorder played facing
    right first where it says so, done at `kills` kills (None: the room clear)."""
    def fight(w):
        import r32b_d7                                          # pylint: disable=import-outside-toplevel
        from r26_walks import DIGDOGGER_SMALL                   # pylint: disable=import-outside-toplevel
        if recorder:
            w.select_b(r32b_d7.RECORDER_SLOT)
            w.press('RIGHT', 6)
            w.press('B', 1)
            w.until(lambda w: any(o[1] == DIGDOGGER_SMALL for o in w.objects()), 'NONE', 400)
        f = r32b_d7.Fight(w, targets)
        if kills is None:
            return f.fight(9000)
        start = len([o for o in w.objects() if o[1] in targets])
        for _ in range(6000):
            if w.hearts() <= 0:
                return 'dead'
            if start - len([o for o in w.objects() if o[1] in targets and o[5] < 0x10]) >= kills:
                return 'clear'
            f.frame()
        return 'time'
    return fight


def _fights():
    from r26_walks import AQUAMENTUS, DIGDOGGER_SMALL, GORIYA_BLUE, GORIYA_RED, KEESE_BLUE  # pylint: disable=C0415
    from r28_demos import GIBDO                                 # pylint: disable=import-outside-toplevel
    return {'5_05e': ('r32b_d5.py garden (BombFighter)', _garden),
            '5_24e': ('r32b_d5.py digdogger (the recorder, DigFighter)', _digdogger),
            '5_66w': ("r32b_d7.py's Fight (round 28's fighter): 0x66's 3 Gibdos, wood sword", _d7((GIBDO,))),
            '7_49s': ('r32b_d7.py 0x49 (Fight: Goriyas, Keese)', _d7((GORIYA_BLUE, GORIYA_RED, KEESE_BLUE))),
            '7_1bw': ('r32b_d7.py 0x1b (Fight: 6 Blue Goriyas)', _d7((GORIYA_BLUE,))),
            '7_1cw': ('r32b_d7.py 0x1c (the recorder, one small Digdogger)', _d7((DIGDOGGER_SMALL,), True, 1)),
            '7_2aw': ('r32b_d7.py 0x2a (Fight: Aquamentus)', _d7((AQUAMENTUS,)))}


# The walks' second-half fights: {state name: (the rule's name, the fight: returns 'clear' when the room is clear)}.
# 5:0x66 W -> N (Gibdos for the key and the N shutter): the explorer failed it with the wood sword (3 attempts, both
# chains; d2first's needed 5 with the white one), so round 28's fighter clears it and the explorer goes on.
FIGHTS = _fights()


# The leg after Dungeon 5 by the walks' rules of r32b_leg.py (`LEG`: state, rule, how the next room is entered), the
# names of the route's later visits for the leg's own (0x73 and 0x63 were crossed on the way to Level 4 first), and
# 0x0b's recorder out of Level 5 (r32b_recorder.py: facing right, played once: the whirlwind to Level 3's 0x74).
LEG_NAMES = {'0_73e': '0_73e_v2', '0_63s': '0_63s_v2'}


@functools.cache
def tallies() -> dict:
    """{state name: (the rule's name, rule(Tally) -> True when it left the room as the route does, how the next room
    is entered: 'scroll', 'cave', 'wrap' or 'recorder')}."""
    import r32b_leg                                             # pylint: disable=import-outside-toplevel
    from r32b_recorder import ride                              # pylint: disable=import-outside-toplevel
    rules = {LEG_NAMES.get(name, name): (f'r32b_leg.py {rule.__name__}', rule, how) for name, rule, how in r32b_leg.LEG}
    rules['0_0bc'] = ('r32b_recorder.py ride (facing right, once)', lambda t: ride(t.w, 'RIGHT', 1)['landed'] == 0x74,
                      'recorder')
    return rules


def explore_out(step, goal, cleared: Path, seed: int):
    """From the saved mid-room state `cleared` (the room as the fight left it), the explorer to the step's exit:
    the pre-entry snapshot of `goal`, or None."""
    from triforce import entry_states                           # pylint: disable=import-outside-toplevel
    from triforce._native import Explorer                       # pylint: disable=import-outside-toplevel
    from triforce.paths import STATES_DIR                       # pylint: disable=import-outside-toplevel
    config = entry_states.step_config(step, str(cleared), ROUTE).replace(fresh_rooms=False)
    for attempt in range(3):
        explorer = Explorer(config, seed=seed * 1000 + attempt, states_dir=str(STATES_DIR))
        result = explorer.explore(str(cleared), 3000, 25, 200, accept=None)
        if not result.success:
            continue
        _, pre_entry = explorer.replay(result, capture_pre_entry=True)
        if explorer.state.full_location != goal.location:
            pre_entry, _ = entry_states.follow_to(explorer.snapshot(), goal.location, str(STATES_DIR))
        if pre_entry is not None:
            return pre_entry
    return None


def harvest_fight(args, step, goal):
    from triforce import entry_states                           # pylint: disable=import-outside-toplevel
    from triforce.paths import STATES_DIR                       # pylint: disable=import-outside-toplevel
    rule, fight = FIGHTS[args.step]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    command = ' '.join(['python'] + sys.argv)
    harvested, missed = [], []
    for chain in args.chains:
        source = Path(STATES_DIR) / step.filename(chain)
        cleared = out / 'cleared' / f'{ROUTE}_{args.step}_{chain}.state'   # the step's name: the env plays its route
        cleared.parent.mkdir(exist_ok=True)
        clears, why, lost, entry = 0, collections.Counter(), [], None
        for seed in range(args.seed0, args.seed0 + args.seeds):
            w = load(source, seed)
            w.wait_play()
            h0 = hp(w)
            end = fight(w)
            if end != 'clear' or hp(w) <= 0:
                why[str(end)[:60]] += 1
                continue
            clears += 1
            lost.append(h0 - hp(w))
            if entry is not None:
                continue
            w.game.save_state_file(str(cleared))
            pre_entry = explore_out(step, goal, cleared, seed)
            if pre_entry is None:
                why['cleared, the explorer found no way out'] += 1
                continue
            path = out / goal.filename(chain)
            entry_states.write_entry_state(w.game, pre_entry, str(path), goal.location)
            entry = {'file': path.name, 'chain': chain, 'route': ROUTE, 'route_step': goal.index,
                     'from_step': step.index, 'specialist': goal.specialist, 'level': goal.level,
                     'room': f'0x{goal.room:02x}', 'in_cave': goal.in_cave, 'entry': goal.entry, 'exit': goal.exit,
                     'alternate': False, 'steps': 0, 'frames': 0, 'health': h0, 'ending': 'exit-correct',
                     'env_steps': 0, 'attempts': seed - args.seed0 + 1, 'seconds': 0.0,
                     'played_by': f'scripts/experiments/r32c_harvest.py: {rule} cleared the room (seed {seed}), '
                                   'the explorer took it out'}
            entry_states._merge_manifest(str(out), [entry], command)     # pylint: disable=protected-access
            harvested.append(entry)
        cleared.unlink(missing_ok=True)
        if entry is None:
            missed.append(step.filename(chain))
        print(json.dumps({'step': args.step, 'rule': rule, 'chain': chain, 'runs': args.seeds, 'won': clears,
                          'hearts_lost_in_wins': round(sum(lost) / len(lost), 2) if lost else None,
                          'failed': dict(why), 'harvested': entry['file'] if entry else None}), flush=True)
    entry_states._write_build_record(str(out), command, harvested, missed)  # pylint: disable=protected-access
    return 0 if not missed else 1


def harvest_tally(args, step, goal):
    """`tallies()`'s rule of the room from each seed; the first won run's pre-entry frame of the next room (the last
    pre-entry frame the rule's presses played, or for a wrap exit the engine macro's) is the next step's state."""
    from triforce import entry_states                           # pylint: disable=import-outside-toplevel
    from triforce.paths import STATES_DIR                       # pylint: disable=import-outside-toplevel
    from r27_walks import Tally, is_pre_entry                   # pylint: disable=import-outside-toplevel
    rule_name, rule, how = tallies()[args.step]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    command = ' '.join(['python'] + sys.argv)
    harvested, missed = [], []
    here = lambda w: (w.level, w.room, w.mode in (CAVE, CELLAR))                       # noqa: E731
    for chain in args.chains:
        source = Path(STATES_DIR) / step.filename(chain)
        wins, why, lost, entry = 0, collections.Counter(), [], None
        for seed in range(args.seed0, args.seed0 + args.seeds):
            w = load(source, seed)
            w.game.set_clips(True)
            w.game.capture_pre_entry = True
            pre = []

            def press(buttons='NONE', frames=1, _press=w.press, _w=w, _pre=pre):
                state = None
                for _ in range(frames):
                    if _w.mode != PLAY and is_pre_entry(_w):
                        _pre[:] = [_w.game.snapshot()]
                    state = _press(buttons, 1)
                return state
            w.press = press
            h0 = hp(w)
            t = Tally(w)
            ok = rule(t)
            if how == 'wrap' or not pre:
                pre[:] = [w.game.last_pre_entry]
            w.until(lambda w: here(w) == goal.location.as_tuple() and w.mode in (PLAY, CAVE, CELLAR)
                    and not is_pre_entry(w), 'NONE', 900)
            if hp(w) <= 0:
                why['dead'] += 1
                continue
            if not ok or here(w) != goal.location.as_tuple():
                why[f"{'ok' if ok else 'failed'}, in {w.level}:0x{w.room:02x}"] += 1
                continue
            if pre[-1] is None:
                why['no pre-entry frame'] += 1
                continue
            wins += 1
            lost.append(h0 - hp(w))
            if entry is None:
                path = out / goal.filename(chain)
                entry_states.write_entry_state(w.game, pre[-1], str(path), goal.location)
                entry = {'file': path.name, 'chain': chain, 'route': ROUTE, 'route_step': goal.index,
                         'from_step': step.index, 'specialist': goal.specialist, 'level': goal.level,
                         'room': f'0x{goal.room:02x}', 'in_cave': goal.in_cave, 'entry': goal.entry,
                         'exit': goal.exit, 'alternate': False, 'steps': 0, 'frames': t.frames, 'health': h0,
                         'ending': 'exit-correct', 'env_steps': 0, 'attempts': seed - args.seed0 + 1,
                         'seconds': 0.0,
                         'played_by': f"scripts/experiments/r32c_harvest.py {args.step}: {rule_name} (seed {seed}, "
                                      "the walks' rule)"}
                entry_states._merge_manifest(str(out), [entry], command)     # pylint: disable=protected-access
                harvested.append(entry)
        if entry is None:
            missed.append(step.filename(chain))
        print(json.dumps({'step': args.step, 'rule': rule_name, 'chain': chain, 'runs': args.seeds, 'won': wins,
                          'hearts_lost_in_wins': round(sum(lost) / len(lost), 2) if lost else None,
                          'failed': dict(why), 'harvested': entry['file'] if entry else None}), flush=True)
    entry_states._write_build_record(str(out), command, harvested, missed)  # pylint: disable=protected-access
    return 0 if not missed else 1


def harvest(args):
    from triforce import entry_states                           # pylint: disable=import-outside-toplevel
    from triforce.paths import STATES_DIR                       # pylint: disable=import-outside-toplevel
    step, goal = route_goal(args.step)
    if args.step in FIGHTS:
        return harvest_fight(args, step, goal)
    if args.step in tallies():
        return harvest_tally(args, step, goal)
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
