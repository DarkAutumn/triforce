"""Phase X's states step: d3first's start states over a stretch of its route, step by step, in one job.

    python scripts/experiments/r32c_states.py --first 0 --last 31 --rules 3_6bs,3_4bs,... --tag s1

For each step of the clip way from route step --first until the state of route step --last is written: the room's
crossing gives the next step's entry state, per chain (0-2) that has the step's own state.  A step named in --rules
(the walks' name of the room's state, `r32_d3first.LEGS`) is crossed by the walks' rule (`r32c_harvest.py`: the
first won run of each chain gives the next state, hearts refilled, every seed played for the win rate); any other
step by the explorer (`entry_states.build`, one step), and where the explorer fails for a chain, by the walks'
actions for the step if the walks have them (`r32_d3first.LEGS`, or the leg after Dungeon 5's rules of
`r32b_leg.py`: `r32c_harvest.tallies()`), else by round 28's or 29's rule of the room if there is one
(`r28_demos.py`, `r29_demos.py`, `r29_clip_demos.py` ROOM `--route d3first --harvest`: `demo_rooms()`); the state
then says so in its manifest entry's `played_by`.  `--rules` names those rooms as `STEM=ROOM` (`8_5es=darknut-5e`;
a bare STEM: its first room).  Each
step's states are staged under target/entry-states/r32c-<tag>/<step> and installed before the next step; a chain
with no state for a step stops there.  At the end `entry_states audit --route d3first`.  Never skips a fight."""

import argparse
import collections
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import r32c_harvest  # noqa: E402  pylint: disable=wrong-import-position
from r32_d3first import LEGS  # noqa: E402  pylint: disable=wrong-import-position

ROUTE = 'd3first'
WALKED = {name for leg in LEGS.values() for name, _ in leg} | set(r32c_harvest.tallies())


def stem(step) -> str:
    """The walks' name of a route step's state (`d3first_3_6bs_0.state` -> `3_6bs`)."""
    return step.filename(0)[len(ROUTE) + 1:-len('_0.state')]


def harvest(name, chains, out, seeds, log):
    """`r32c_harvest.py NAME --out OUT --chains CHAINS --seeds SEEDS`, in this process."""
    args = argparse.Namespace(step=name, out=str(out), chains=chains, seeds=seeds, seed0=1000)
    argv = sys.argv
    sys.argv = ['scripts/experiments/r32c_harvest.py', name, '--out', str(out), '--chains', ','.join(map(str, chains)),
                '--seeds', str(seeds)]
    try:
        r32c_harvest.harvest(args)
    except Exception as e:                                      # pylint: disable=broad-except
        log(f'  harvest {name} chains {chains}: ERROR {type(e).__name__}: {e}')
    finally:
        sys.argv = argv


def demo_rooms() -> dict:
    """{state name: [(module, ROOM)]}: round 28's and 29's rules (`r28_demos`, `r29_demos` with `r29_clip_demos`'s
    clips) of d3first's steps, the room's step found by its d2first step's room, entry, exit and way."""
    import r28_demos                                            # pylint: disable=import-outside-toplevel
    import r29_clip_demos                                       # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
    import r29_demos                                            # pylint: disable=import-outside-toplevel
    rooms = collections.defaultdict(list)
    for module in (r28_demos, r29_demos):
        for room, (_, index) in module.ROOMS.items():
            try:
                step, _ = r28_demos.route_step(index, ROUTE)
            except StopIteration:                               # a room d3first does not play
                continue
            if step.way != 'plain':                             # the stretch is built along the clip way
                rooms[stem(step)].append((module, room))
    return rooms


def play_demo(module, room, chains, out, runs, log):
    """`<module>.py ROOM --route d3first --chains CHAINS --runs RUNS --harvest OUT`, in this process."""
    script = f'scripts/experiments/{module.__name__}.py'
    args = argparse.Namespace(what=room, chains=','.join(map(str, chains)), runs=runs, hearts='own', tracks=None,
                              per_state=1, max_loss=None, harvest=Path(out), max_steps=3000, out=None, verbose=False,
                              route=ROUTE)
    argv = sys.argv
    sys.argv = [script, room, '--route', ROUTE, '--chains', args.chains, '--runs', str(runs), '--harvest', str(out)]
    try:
        module.play(args)
    except (Exception, SystemExit) as e:                        # pylint: disable=broad-except
        log(f'  {script} {room} chains {chains}: ERROR {type(e).__name__}: {e}')
    finally:
        sys.argv = argv


def staged(out, target, chains):
    return [c for c in chains if (Path(out) / target.filename(c)).exists()]


def main():
    from triforce import entry_states                           # pylint: disable=import-outside-toplevel
    from triforce.paths import STATES_DIR                       # pylint: disable=import-outside-toplevel
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--first', type=int, required=True, help='route step whose state the stretch starts from')
    parser.add_argument('--last', type=int, required=True, help='route step whose state ends the stretch')
    parser.add_argument('--rules', default='',
                        help="steps (the walks' names) crossed by the walks' rule, or STEM=ROOM by round 28's/29's")
    parser.add_argument('--chains', default='0,1,2')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--rule-seeds', type=int, default=20)
    parser.add_argument('--fallback-seeds', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--attempts', type=int, default=3)
    args = parser.parse_args()
    rules = dict((r.split('=') + [None])[:2] for r in args.rules.split(',') if r)
    demos = demo_rooms()

    def demo(name, room=None):
        """(module, ROOM) of round 28's/29's rule for `name` (`room`, or its first)."""
        return next((m, r) for m, r in demos[name] if room in (None, r))

    def log(text):
        print(f'{time.strftime("%H:%M:%S")} {text}', flush=True)

    way = [s for s in entry_states.route_steps(ROUTE) if s.way in (None, 'clips')]
    root = Path('target/entry-states') / f'r32c-{args.tag}'
    chains = [int(c) for c in args.chains.split(',')]
    at = next(i for i, s in enumerate(way) if s.index >= args.first)
    for step, target in zip(way[at:], way[at + 1:]):
        if target.index > args.last:
            break
        name = stem(step)
        if step.index > 0:
            chains = [c for c in chains if (Path(STATES_DIR) / step.filename(c)).exists()]
        if not chains:
            log(f'step {step.index} {name}: no chain has its state: stop')
            break
        out = root / f'{step.index:03d}-{name}'
        started = time.time()
        if name in rules:
            log(f'step {step.index} {name} -> {stem(target)}: the rule, chains {chains}')
            if rules[name] is None and (name in WALKED or name in r32c_harvest.FIGHTS or name not in demos):
                harvest(name, chains, out, args.rule_seeds, log)
            else:
                play_demo(*demo(name, rules[name]), chains, out, args.rule_seeds, log)
            how = 'rule'
        else:
            source = entry_states.START_STATE if step.index == 0 else \
                os.path.relpath(Path(STATES_DIR) / f'{ROUTE}_{name}_{{chain}}.state')    # no local path in the manifest
            command = (f'python -m triforce.entry_states build --route {ROUTE} --chains {",".join(map(str, chains))} '
                       f'--start-step {step.index} --last-step {target.index} --source {source} --out {out} '
                       f'--iterations {args.iterations} --patience {args.patience} --attempts {args.attempts}')
            entry_states.build(chains, str(out), args.iterations, args.patience, start_step=step.index, source=source,
                               command=command, route=ROUTE, attempts=args.attempts, last_step=target.index)
            how = 'explorer'
            missing = [c for c in chains if c not in staged(out, target, chains)]
            if missing and (name in WALKED or name in r32c_harvest.FIGHTS):
                log(f"  explorer failed chains {missing}: the walks' actions or fight for {name}")
                harvest(name, missing, out, args.fallback_seeds, log)
                how = f'explorer, walks for {missing}'
            elif missing and name in demos:
                module, room = demo(name)
                log(f"  explorer failed chains {missing}: {module.__name__} {room}")
                play_demo(module, room, missing, out, args.fallback_seeds, log)
                how = f'explorer, {module.__name__} {room} for {missing}'
        done = staged(out, target, chains)
        if done:
            entry_states.install(str(out), str(STATES_DIR), partial=True)
        log(f'step {step.index} {name} -> {stem(target)} ({how}): chains {done} of {chains}, '
            f'{time.time() - started:.0f}s')
        chains = done
    log('audit:')
    subprocess.run([sys.executable, '-m', 'triforce.entry_states', 'audit', '--route', ROUTE], check=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
