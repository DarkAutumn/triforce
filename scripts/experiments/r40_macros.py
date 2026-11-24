"""Round 40: the fresh generation's engine macros (routes.yaml `macros:`, docs/routes/d3first.md "The engine's
macros") played by a scripted rule over many seeds from `d3first`'s installed entry states, with the room's monsters
live: how often the fixed sequence lands and how often Link is hit (what a network choosing the moment must learn to
avoid). Run it from a checkout whose routes.yaml turns the macros on (`d3first: {whirlwinds: [0x0b, 0x22],
unaided_docks: true}`); with them off the whirlwinds are played by hand and the docks give way.

    python scripts/experiments/r40_macros.py ROOM [--runs N] [--waits 0,2,5,10,20] [--chains 0,1,2] [--hearts own|family|H]

    ww-0b     0:0x0b out of Level 5 (ow-leever, `0_0bc`): the whirlwind to 0x45.
    ww-22     0:0x22 out of Level 6 (ow-lynel, `0_22c`): the whirlwind to 0x3c.
    dock-55   0:0x55 off the raft (ow-zora, `0_55n_v2`): the dock warp unaided to 0x34.
    dock-3f   0:0x3f off the raft (ow-zora, `0_3fn`): the dock warp unaided to 0x31.

The rule, "walk there, play at once": on the whirlwind rooms RECORDER as soon as it is offered (Link comes out of the
level where the macro is offered); on a dock MOVE S off the dock, MOVE W to the clip's column, CLIP N, RECORDER. With
`--waits K`: K decisions of WAIT 8 first (on a dock: in the water, before the play), a moment chosen blind. Each
episode is the family's one-room episode from the state with the seed's natural random numbers and fresh spawns
(`env.reset(seed)`, seeds 1000 on); its ending is the env's: success = set down in the route's next room."""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import r29_demos                                                  # noqa: E402  pylint: disable=wrong-import-position
import r32p_rules                                                 # noqa: E402  pylint: disable=wrong-import-position

ROOMS = {'ww-0b': ('ow-leever', '0_0bc'), 'ww-22': ('ow-lynel', '0_22c'),
         'dock-55': ('ow-zora', '0_55n_v2'), 'dock-3f': ('ow-zora', '0_3fn')}
SEED0 = 1000


def rule(room, waits):
    """The rule's next action given the mask (`ok`), with `waits` WAITs before the play."""
    left = [waits]

    def play(ok):
        if left[0] > 0 and ok('WAIT 8'):
            left[0] -= 1
            return 'WAIT 8'
        return 'RECORDER'

    def step(_ram, ok):
        if ok('RECORDER'):
            return play(ok)
        if room.startswith('dock'):
            if ok('CLIP N'):
                return 'CLIP N'
            if ok('MOVE S 4') and not ok('MOVE W 4'):
                return 'MOVE S 4'                               # on the dock: the one step that keeps the warp
            if ok('MOVE W 4'):
                return 'MOVE W 4'
        return 'WAIT 4'                                         # the whirlwind out, or nothing offered
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('room', choices=list(ROOMS))
    parser.add_argument('--runs', type=int, default=100, help="runs a state and wait (default 100)")
    parser.add_argument('--waits', default='0', help="comma-separated WAIT counts before the play (default 0)")
    parser.add_argument('--chains', default='0,1,2')
    parser.add_argument('--hearts', default='own', help="own (the state's), family (its draws) or a number")
    parser.add_argument('--max-steps', type=int, default=60)
    args = parser.parse_args()
    from triforce import jobs                                  # pylint: disable=import-outside-toplevel
    from triforce.demos import action_indices, legal           # pylint: disable=import-outside-toplevel
    jobs.require_job()
    waits_list = [int(w) for w in args.waits.split(',')]
    chains = [int(c) for c in args.chains.split(',')]
    planned, played_runs = len(waits_list) * len(chains) * args.runs, 0
    family, stem = ROOMS[args.room]
    r32p_rules.ROOM = args.room
    for waits in waits_list:
        total = {'runs': 0, 'won': 0, 'endings': collections.Counter(), 'hit_runs': 0, 'hits': 0, 'hearts_lost': 0.0,
                 'not_offered': 0, 'deaths': 0}
        for chain in chains:
            state = f'd3first_{stem}_{chain}'
            env = r32p_rules.rule_env(family, state, args.hearts)
            for seed in range(SEED0, SEED0 + args.runs):
                obs, _ = env.reset(seed=seed, options={'start_state': state})
                act = rule(args.room, waits)
                played_runs += 1
                jobs.progress(played_runs, planned, 'runs')
                ram = bytes(env.state.ram)
                health = r29_demos.hearts_of(ram)
                hits, info, played = 0, {}, False
                for _ in range(args.max_steps):
                    mask = obs['action_mask']

                    def ok(text, mask=mask):
                        return legal(mask, action_indices(text))
                    choice = act(ram, ok)
                    if not ok(choice):
                        choice = 'WAIT 4'
                    played |= choice == 'RECORDER'
                    before = r29_demos.hearts_of(ram)
                    obs, _, terminated, truncated, info = env.step(choice)
                    ram = bytes(env.state.ram)
                    hits += r29_demos.hearts_of(ram) < before
                    if terminated or truncated:
                        break
                ending = info.get('ending', 'max-steps (the script\'s)')
                total['runs'] += 1
                total['won'] += info.get('outcome') == 'success'
                total['endings'][ending] += 1
                total['hit_runs'] += hits > 0
                total['hits'] += hits
                total['hearts_lost'] += max(0.0, health - r29_demos.hearts_of(ram))
                total['not_offered'] += not played
                total['deaths'] += 'death' in ending or env.state.game_over
        runs = total['runs']
        print(json.dumps({'room': args.room, 'family': family, 'waits': waits, 'runs': runs,
                          'success': round(total['won'] / runs, 4), 'hit_runs': round(total['hit_runs'] / runs, 4),
                          'hits_a_visit': round(total['hits'] / runs, 3),
                          'hearts_lost_a_visit': round(total['hearts_lost'] / runs, 3),
                          'deaths': total['deaths'], 'never_played': total['not_offered'],
                          'endings': dict(total['endings'].most_common())}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
