#!/usr/bin/env python3
"""Round 40 (Tracks41, docs/experiments/round40.md "The untracked states' tracks"): curriculum tracks of the states
Tracks40 left without one, recorded from the explorer's own reset of today's state files.

    python scripts/experiments/r41_tracks.py fighter --room 59|5b|5b-last --states S1,S2 --out ROOT
        [--seeds N] [--seed0 S] [--depth D] [--workers W]
    python scripts/experiments/r41_tracks.py trace --room 59|5b|5b-last --state S [--seed N]
    python scripts/experiments/r41_tracks.py table ROOT...

A track is what ``python -m triforce.explore`` writes (``env::explore::Explorer::write``): ``NNNN.state``, the session
before decision NNNN (0000 the reset), and ``trajectory.json`` (``start_state``, ``success``, ``ending``, ``steps``,
``health``, ``actions``, ``explore``) under ``ROOT/<family>/<state>~<how><seed>``; the curriculum loads every track
folder of a family as its own track and ties it to its seed state by ``start_state``.  Every run here starts as the
explorer's do: the family's ``d3first`` task (``lab.collect.task_yaml``) with no RAM randomization and the run-time
clips switch on (``env::explore::explore_config``), so a track's ``0000.state`` is the explorer's reset of its state
but for the random numbers (``r40_tracks.py audit`` checks it).

``fighter``: Darknut40's fighter (``r40_darknut_fighter.py``, the owner's rules) as its demonstrations are built
(``teach``: the look-ahead of ``--depth`` decisions near a Darknut, then the backtracking search where Link is still
hit), from seeds ``--seed0``.. until a visit leaves by the route's exit unhit; that visit, played again from the
reset, is the track (``<state>~fighter<seed>``).  A state none of whose ``--seeds`` visits is kept gets no track.
``--room 59``: 3:0x59 from N (the S door); ``5b``: 3:0x5b's first pass (one Red Darknut, its bombs, N);
``5b-last``: 3:0x5b's last pass (the fight, the room's bombs, the wall).
``trace`` prints one plain visit of the fighter a decision a line; ``table`` each track's steps, hits and hearts lost
(the report's table).
"""
import argparse
import json
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FAMILY = {'59': 'darknut', '5b': 'darknut', '5b-last': 'darknut'}
# The room entry each fighter room is (a task's room key), for a start's drop cycle.
ENTRY = {'59': '(3, 0x59, N)', '5b': '(3, 0x5b, S)', '5b-last': '(3, 0x5b, W)'}


def explore_env(family: str, layout='route', start: dict | None = None):
    """The family's ``d3first`` task as the explorer plays it: no RAM randomization, clips on, its resets in
    ``layout`` (round 45: ``route``, the route's planned spawn cycle, as ``triforce.explore --layout`` defaults to; a
    spawn cycle 0-8 for a track in another reachable layout).  ``start`` (round 45): what the reset writes beside, the
    values the route may arrive with at the door that the walked state holds only one of (``bombs``: the bag; ``drop``:
    ``(room key, drop cycle)``), as the task's own draws write them; a track start keeps them (a curriculum start
    draws none of a listed room's values)."""
    from triforce._native import Env  # pylint: disable=import-outside-toplevel
    from triforce.lab.collect import task_yaml  # pylint: disable=import-outside-toplevel
    from triforce.specialist import load_specialist  # pylint: disable=import-outside-toplevel
    draws = {}
    if start and start.get('bombs') is not None:
        draws.update(bombs=[start['bombs']] * 2, bombs_share=1.0)
    if start and start.get('drop') is not None:
        key, cycle = start['drop']
        draws['room_kill_cycle'] = {key: [cycle, cycle]}
    config = load_specialist(task_yaml(f'{family}@d3first')).replace(ram_randomization=draws)
    clips = config.to_dict()['clips']
    if clips['enabled']:
        config = config.replace(clips=dict(clips, off_share=0.0))
    env = Env(config, explicit_starts=True, strict_actions=False, seed=0)
    env.set_layout(layout)
    return env


def track_name(state: str, how: str, seed: int, layout, start: dict | None = None) -> str:
    """``<state>~<how><seed>``, then ``~bombs<N>`` and ``~drop<N>`` for a start's written bag and drop cycle, and
    ``~c<N>`` for a track in another reachable layout than the route's (round 45, as ``triforce.explore`` names it)."""
    start = start or {}
    return (f'{state}~{how}{seed}' + (f"~bombs{start['bombs']}" if start.get('bombs') is not None else '')
            + (f"~drop{start['drop'][1]}" if start.get('drop') is not None else '')
            + (f'~c{layout}' if isinstance(layout, int) else ''))


def write_track(env, root: Path, family: str, state: str, seed: int, how: str, actions: list, crafted: str,
                start: dict | None = None) -> dict:
    """``actions`` played again from the reset of ``state`` with ``seed``, a state file before each one, as the
    explorer writes a track; returns its trajectory.json (``success`` the replay's; ``layout`` the reset's, round
    45: ``Env.entry``, as ``Explorer::write`` records it; ``start`` the reset's written bag and drop cycle)."""
    env.reset(seed=seed, options={'start_state': state})
    begin = bytes(env.state.ram)
    start_health = env.state.health
    layout = env.entry
    entry = layout.get('entry') or {}
    if layout['asked'] == 'cycle' and entry.get('source') != 'rule':
        raise SystemExit(f"{state}: its door's reset rule holds no such spawn cycle ({entry})")
    track = root / family / track_name(state, how, seed, entry.get('spawn_cycle') if layout['asked'] == 'cycle'
                                       else None, start)
    if track.exists():
        raise SystemExit(f'{track} exists')
    track.mkdir(parents=True)
    info, played, hits, health = {}, [], 0, start_health
    for index, text in enumerate(actions):
        (track / f'{index:04d}.state').write_bytes(env.snapshot().state_file())
        _, _, terminated, truncated, info = env.step(text)
        played.append(text)
        hits += env.state.health < health
        health = env.state.health
        if terminated or truncated:
            break
    meta = {'start_state': state, 'success': info.get('outcome') == 'success', 'ending': info.get('ending'),
            'steps': len(played), 'health': env.state.health, 'actions': played,
            'explore': {'iterations': 0, 'env_steps': len(played), 'cells': 0, 'seconds': 0.0, 'seed': seed},
            'layout': layout, 'start': {'bombs': begin[0x658], 'drop_cycle': begin[0x52A], 'help_count': begin[0x50],
                                        'written': {k: (v[1] if k == 'drop' else v) for k, v in (start or {}).items()
                                                    if v is not None}},
            'crafted': crafted, 'hits': hits, 'hearts_lost': start_health - env.state.health}
    (track / 'trajectory.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    return {'track': str(track), **{k: meta[k] for k in ('success', 'ending', 'steps', 'health', 'hits',
                                                          'hearts_lost')},
            'spawn_cycle': entry.get('spawn_cycle'), 'planned': entry.get('planned')}


def _teach(task):
    import r40_darknut_fighter as fighter  # pylint: disable=import-outside-toplevel
    room, state, seed, depth, layout, start = task
    env, probe = explore_env(FAMILY[room], layout, start), explore_env(FAMILY[room], layout, start)
    way = []
    rec = fighter.teach(env, room, seed, state, probe, depth, way=way)
    return state, seed, rec, way


def fighter_tracks(room: str, states: list, root: Path, seeds: int, seed0: int, depth: int, workers: int,
                   layout='route', start: dict | None = None):
    from triforce import jobs  # pylint: disable=import-outside-toplevel
    jobs.require_job()
    left = list(states)
    written = []
    with multiprocessing.get_context('spawn').Pool(workers) as pool:
        for first in range(seed0, seed0 + seeds, workers):
            tasks = [(room, s, seed, depth, layout, start) for s in left
                     for seed in range(first, min(first + workers, seed0 + seeds))]
            if not tasks:
                break
            kept = {}
            for state, seed, rec, way in pool.imap(_teach, tasks):
                print(json.dumps({'state': state, 'seed': seed, 'ending': rec.get('ending'), 'success': rec['success'],
                                  'hits': rec['hits'], 'tries': rec.get('tries'), 'actions': len(way),
                                  'seconds': rec.get('seconds'), 'layout': rec.get('layout'),
                                  'kills': rec.get('kills'), 'drop_cycle': rec.get('drop_cycle')}), flush=True)
                if rec['success'] and rec['hits'] == 0 and state not in kept:
                    kept[state] = (seed, way)
            for state, (seed, way) in sorted(kept.items()):
                out = write_track(explore_env(FAMILY[room], layout, start), root, FAMILY[room], state, seed,
                                  'fighter', way,
                                  f"round 45 (Tracks45): Darknut40's fighter, room {room}, look-ahead {depth}, layout "
                                  f"{layout}, start {start or 'as the state holds it'} "
                                  f"(scripts/experiments/r41_tracks.py fighter)", start)
                print(json.dumps(out), flush=True)
                written.append(out)
                left.remove(state)
            jobs.progress(len(states) - len(left), len(states), 'states')
            if not left:
                break
    print(json.dumps({'room': room, 'layout': layout, 'start': start, 'tracks': len(written), 'without': left}))
    return 0 if not left else 1


def trace(room: str, state: str, seed: int):
    """One visit of the plain fighter (no look-ahead), a line a decision: Link, the bag, the room item, the
    Darknuts, the action."""
    import r40_darknut_fighter as fighter  # pylint: disable=import-outside-toplevel
    from r40_d3_69_line import mask_ok  # pylint: disable=import-outside-toplevel
    env = explore_env(FAMILY[room])
    obs, _ = env.reset(seed=seed, options={'start_state': state})
    rule = fighter.Fighter(room, fighter.Floor(env.state.render_grid()))
    for i in range(fighter.LIMIT):
        ram = bytes(env.state.ram)
        nuts = fighter.darknuts(ram)
        grid = env.state.render_grid() if not nuts else None
        step = rule.decide(ram, mask_ok(obs), grid)
        item = (ram[fighter.OBJ_X + fighter.ROOM_ITEM], ram[fighter.OBJ_Y + fighter.ROOM_ITEM])
        print(f"{i:4d} link {ram[fighter.OBJ_X]:3d},{ram[fighter.OBJ_Y]:3d} bag {ram[fighter.INV_BOMBS]} item {item} "
              f"nuts {len(nuts)} hp {env.state.health} {step}", flush=True)
        obs, _, terminated, truncated, info = env.step(step)
        if terminated or truncated:
            print(info.get('ending'), info.get('outcome'))
            break
    return 0


def table(roots: list):
    """Each track under the roots: its state, how it was made, its steps, the hits taken on it (the hearts falling
    from one state file to the next, and on the last step by its trajectory.json `health` where that is the end's:
    round 29's rule tracks, `r29_demos.write_track`, record the start's there) and the hearts lost."""
    from triforce import jobs  # pylint: disable=import-outside-toplevel
    from triforce._native import Game  # pylint: disable=import-outside-toplevel
    from triforce.paths import STATES_DIR  # pylint: disable=import-outside-toplevel
    jobs.require_job()
    game = Game(seed=0, states_dir=str(STATES_DIR))
    print('| family | state | track | steps | hits | hearts lost | ending |')
    print('|---|---|---|---:|---:|---:|---|')
    for root in roots:
        for meta_path in sorted(Path(root).glob('*/*/trajectory.json')):
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            health = [game.load(str(p), randomize_rng=False, controllable=True).health
                      for p in sorted(meta_path.parent.glob('*.state'))]
            if not str(meta.get('crafted', '')).startswith('round 29'):
                health.append(meta['health'])
            hits = sum(b < a for a, b in zip(health, health[1:]))
            print(f"| {meta_path.parent.parent.name} | {meta['start_state']} | `{meta_path.parent.name}` | "
                  f"{meta['steps']} | {hits} | {health[0] - health[-1]:g} | {meta.get('ending')} |", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('fighter')
    f.add_argument('--room', choices=sorted(FAMILY), required=True)
    f.add_argument('--states', required=True)
    f.add_argument('--out', required=True, type=Path, help='the trajectories root (<out>/<family>/<track>)')
    f.add_argument('--seeds', type=int, default=40)
    f.add_argument('--seed0', type=int, default=100000)
    f.add_argument('--depth', type=int, default=3)
    f.add_argument('--workers', type=int, default=4)
    f.add_argument('--layout', default='route', metavar='route|N',
                   help="the layout the resets serve (round 45; default route: the route's planned spawn cycle; N: "
                        "that reachable spawn cycle, tracks <state>~fighter<seed>~c<N>)")
    f.add_argument('--bombs', type=int, default=None,
                   help="the bag the reset writes (round 45; default the state's own; track ~bombs<N>)")
    f.add_argument('--drop-cycle', type=int, default=None,
                   help="the drop cycle $52A the reset writes for the room's entry (round 45; default the state's own; "
                        "track ~drop<N>)")
    t = sub.add_parser('trace')
    t.add_argument('--room', choices=sorted(FAMILY), required=True)
    t.add_argument('--state', required=True)
    t.add_argument('--seed', type=int, default=100000)
    b = sub.add_parser('table')
    b.add_argument('roots', nargs='+')
    args = parser.parse_args(argv)
    if args.command == 'trace':
        return trace(args.room, args.state, args.seed)
    if args.command == 'table':
        return table(args.roots)
    start = {'bombs': args.bombs, 'drop': None if args.drop_cycle is None else (ENTRY[args.room], args.drop_cycle)}
    return fighter_tracks(args.room, args.states.split(','), args.out, args.seeds, args.seed0, args.depth,
                          args.workers, int(args.layout) if args.layout.isdigit() else args.layout,
                          start if any(v is not None for v in start.values()) else None)


if __name__ == '__main__':
    sys.exit(main())
