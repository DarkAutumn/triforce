#!/usr/bin/env python3
"""Round 45 (Tracks45, docs/experiments/round40.md "Tracks in the route's layout"): `d3first`'s curriculum tracks
again from States44's walked states (one state per door, no redraw), in the route's layout, and the proofs.

    python scripts/experiments/r45_tracks.py table ROOT... [--json OUT]
    python scripts/experiments/r45_tracks.py replay ROOT [--points P] [--seeds S] [--families F,G] [--json OUT]
    python scripts/experiments/r45_tracks.py annotate ROOT...
    python scripts/experiments/r45_tracks.py compose --store DIR --add DIR [--add DIR] --walk WALK.json --out NEW
        [--keep GLOB]

A track is what ``python -m triforce.explore`` writes (``env::explore::Explorer::write``; the fighter's by
``r41_tracks.py fighter``): ``NNNN.state``, the session before decision NNNN (0000 the reset), and ``trajectory.json``,
whose ``layout`` (round 45) says what the room was entered with: the layout asked (``route``: the route's planned spawn
cycle ``$524``; ``cycle``: another reachable one), the RNG phase written, and ``entry`` (``$524`` with ``planned`` and
``reachable`` against ``triforce/entry_rule.d3first.json``, the chase flag and target, ``$4A``, the divider ``$26``,
the edge cell ``$525``; null for a cave or cellar start, where nothing reads ``$524``).

``table``: each track under the roots (``<root>/<family>/<track>``): its state, how it was made, its steps, its
layout, its ending, the hits taken on it (the hearts falling from one state file to the next).

``replay`` (the proof the lead asked for: a track explored in the route's layout replays to its end from its own saved
states under the training reset). A training's curriculum start loads a track's ``NNNN.state`` (mid-room, at a
decision: ``LoadOptions::controllable``) through ``SpecialistEnv::reset`` as every start: ``randomize_rng`` writes a
new RNG phase (``$18``-``$24``); the reset rule's draw (``$524``, the chase state, ``$4A``, ``$26``, ``$525``) and the
fresh-room and banked-kill writes act only on a pre-entry state, which no ``NNNN.state`` is; the task's
``ram_randomization`` (hearts, bombs, help count, a listed room's values) is written as at any start.  So a curriculum
start keeps the track's layout and every monster where the track had it, and redraws the random phase.  A curriculum
start is never a track's ``0000.state`` (a track's point moves a stride earlier per stage and the track is complete,
never drawn again, once it reaches 0; the room start is the seed state's own reset).  For each track and ``--points``
points spread over decisions 1 to its last (both always among them), a real curriculum env of the family's task
(``Env`` with a ``CurriculumSpec`` whose progress file pins the track's point there: ``stages`` = the track's length,
so the start is that state exactly) is reset with ``--seeds`` seeds; checked: (a) the RAM after the
reset equals the state file's but for the RNG registers (the layout and the monsters kept), with the explorer's
config (no ``ram_randomization``, as the track was made); (b) the track's remaining actions played from there reach
the track's own ending (``success``), the phase redrawn.  (b) is open-loop: a fight whose monsters walk by the
random numbers may go another way after a new phase; the rate by the distance from the end is the measure.

``compose``: a new trajectories folder for the lead to install: every track of ``--store`` (hard links; a track folder
is only ever created, never written over; nothing is chmod-ed) except the ``d3first_`` tracks of a door the walk
record (``triforce/states/d3first_walk.json``, States44) has rebuilt, whatever their chain (the walked state replaces
the door's three chains), then every track of each ``--add`` root (copies).  ``--keep GLOB`` (``family/track``) keeps
a store track the rule would drop.  It writes ``compose.json`` beside the folder: per family what was dropped and
added.

Every command that plays is a job of the machine's queue (``jobs.require_job``).
"""
import argparse
import fnmatch
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

RNG = range(0x18, 0x25)                                     # the RNG register, $18-$24 (zelda::rng)
CHAIN = re.compile(r'_\d+$')


def door_of(state: str) -> str:
    """``d3first_3_59n_1`` -> ``d3first_3_59n`` (the door; a visit suffix kept: ``d3first_0_1be_v2``)."""
    return CHAIN.sub('', state)


def tracks_under(root: Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).glob('*/*/trajectory.json'))


def explore_config(family: str):
    """The family's ``d3first`` task as the explorer plays it (``env::explore::explore_config``): no RAM
    randomization, the clips switch on in every episode."""
    from triforce.lab.collect import task_yaml  # pylint: disable=import-outside-toplevel
    from triforce.specialist import load_specialist  # pylint: disable=import-outside-toplevel
    config = load_specialist(task_yaml(f'{family}@d3first')).replace(ram_randomization={})
    clips = config.to_dict()['clips']
    if clips['enabled']:
        config = config.replace(clips=dict(clips, off_share=0.0))
    return config


def how_made(meta: dict, name: str) -> str:
    crafted = str(meta.get('crafted', ''))
    if '~fighter' in name:
        return 'fighter'
    if '~rule' in name:
        return 'rule'
    return 'explorer' if not crafted else crafted.split(':')[0]


def table(roots: list, out: Path | None):
    """Each track's row (``replay``'s and the report's table)."""
    from triforce import jobs  # pylint: disable=import-outside-toplevel
    from triforce._native import Game  # pylint: disable=import-outside-toplevel
    from triforce.paths import STATES_DIR  # pylint: disable=import-outside-toplevel
    jobs.require_job()
    game = Game(seed=0, states_dir=str(STATES_DIR))
    rows = []
    print('| family | state | track | made by | steps | $524 | route layout | chase | ending | hits | hearts lost |')
    print('|---|---|---|---|---:|---:|---|---:|---|---:|---:|')
    for root in roots:
        for track in tracks_under(root):
            meta = json.loads((track / 'trajectory.json').read_text(encoding='utf-8'))
            health = [game.load(str(p), randomize_rng=False, controllable=True).health
                      for p in sorted(track.glob('*.state'))] + [meta['health']]
            entry = (meta.get('layout') or {}).get('entry') or {}
            row = {'family': track.parent.name, 'state': meta['start_state'], 'track': track.name,
                   'how': how_made(meta, track.name), 'steps': meta['steps'], 'ending': meta.get('ending'),
                   'success': meta.get('success'), 'spawn_cycle': entry.get('spawn_cycle'),
                   'planned': entry.get('planned'), 'chase_flag': entry.get('chase_flag'),
                   'hits': sum(b < a for a, b in zip(health, health[1:])), 'hearts_lost': health[0] - health[-1]}
            rows.append(row)
            route = {True: 'yes', False: 'no', None: '-'}[row['planned']]
            print(f"| {row['family']} | {row['state']} | `{row['track']}` | {row['how']} | {row['steps']} | "
                  f"{'-' if row['spawn_cycle'] is None else row['spawn_cycle']} | {route} | "
                  f"{'-' if row['chase_flag'] is None else row['chase_flag']} | {row['ending']} | {row['hits']} | "
                  f"{row['hearts_lost']:g} |", flush=True)
    if out:
        out.write_text(json.dumps(rows, indent=1), encoding='utf-8')
    return 0


def points_of(length: int, count: int) -> list[int]:
    """``count`` decisions spread over 1 .. ``length - 1`` (a curriculum start's range), both ends among them."""
    if length - 1 <= count:
        return list(range(1, length))
    return sorted({1 + round(i * (length - 2) / (count - 1)) for i in range(count)})


def replay_point(config, track: Path, actions: list, length: int, point: int, seeds: list, files: list) -> dict:
    """A curriculum env pinned to ``point`` of ``track``, reset with each seed: the RAM against the file (but the
    RNG), and the remaining actions played open-loop to the end."""
    from triforce._native import CurriculumSpec, Env, Game  # pylint: disable=import-outside-toplevel
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'tracks'
        root.mkdir()
        os.symlink(track.resolve(), root / track.name)
        progress = Path(tmp) / 'curriculum.json'
        progress.write_text(json.dumps({track.name: {'length': length, 'point': point, 'attempts': 0, 'successes': 0,
                                                     'stages_passed': 0}}), encoding='utf-8')
        spec = CurriculumSpec(str(root), state_path=str(progress), fraction=1.0, stages=length, window=1_000_000,
                              threshold=1.0)
        env = Env(config, curriculum=spec, strict_actions=False, seed=0)
        saved = bytes(Game(seed=0).load(str(files[point]), randomize_rng=False, controllable=True).ram)
        out = {'point': point, 'left': length - point, 'kept': 0, 'won': 0, 'endings': Counter(), 'differ': Counter()}
        for seed in seeds:
            env.reset(seed=seed)
            ram = bytes(env.state.ram)
            differ = [a for a in range(len(ram)) if a not in RNG and ram[a] != saved[a]]
            out['kept'] += not differ
            out['differ'].update(f'${a:03X}' for a in differ)
            info = {}
            for text in actions[point:]:
                _, _, terminated, truncated, info = env.step(text)
                if terminated or truncated:
                    break
            out['endings'][info.get('ending')] += 1
            out['won'] += info.get('outcome') == 'success'
        out['endings'] = dict(out['endings'])
        out['differ'] = dict(out['differ'])
        return out


def replay(root: Path, points: int, seeds: int, families: list | None, out: Path | None):
    from triforce import jobs  # pylint: disable=import-outside-toplevel
    jobs.require_job()
    tracks = [t for t in tracks_under(root) if not families or t.parent.name in families]
    results, configs = [], {}
    for done, track in enumerate(tracks):
        meta = json.loads((track / 'trajectory.json').read_text(encoding='utf-8'))
        family = track.parent.name
        configs.setdefault(family, explore_config(family))
        files = sorted(track.glob('*.state'))
        actions = meta['actions']
        rows = [replay_point(configs[family], track, actions, len(files), k, list(range(seeds)), files)
                for k in points_of(len(files), points)]
        entry = (meta.get('layout') or {}).get('entry') or {}
        result = {'family': family, 'track': track.name, 'steps': len(files), 'recorded_success': meta.get('success'),
                  'spawn_cycle': entry.get('spawn_cycle'), 'planned': entry.get('planned'), 'points': rows}
        results.append(result)
        print(json.dumps({'track': f"{family}/{track.name}", 'steps': len(files), 'planned': entry.get('planned'),
                          'layout kept': f"{sum(r['kept'] for r in rows)}/{len(rows) * seeds}",
                          'won by distance from the end': {r['left']: f"{r['won']}/{seeds}" for r in rows}}),
              flush=True)
        jobs.progress(done + 1, len(tracks), 'tracks')
    resets = sum(len(r['points']) * seeds for r in results)
    kept = sum(p['kept'] for r in results for p in r['points'])
    last = [p for r in results for p in r['points'] if p['left'] == 1]
    print(json.dumps({'tracks': len(results), 'recorded success': sum(bool(r['recorded_success']) for r in results),
                      'resets': resets, 'layout kept (RAM = file but the RNG)': kept,
                      'won from the last decision': f"{sum(p['won'] for p in last)}/{len(last) * seeds}",
                      'won from the first point': f"{sum(r['points'][0]['won'] for r in results)}/"
                                                  f"{len(results) * seeds}"}))
    if out:
        out.write_text(json.dumps(results, indent=1), encoding='utf-8')
    return 0 if kept == resets else 1


def annotate(roots: list):
    """Writes ``start`` into each track's ``trajectory.json`` that lacks it (tracks explored before ``Explorer::write``
    recorded it, Nov 25): the same values (``env::explore::start_json``) read from the track's ``0000.state``, the
    explorer's reset itself (no ``ram_randomization``: the walked state's own bag, drop cycle and counters)."""
    from triforce._native import Game  # pylint: disable=import-outside-toplevel
    game = Game(seed=0)
    done = 0
    for root in roots:
        for track in tracks_under(root):
            path = track / 'trajectory.json'
            meta = json.loads(path.read_text(encoding='utf-8'))
            if meta.get('start') is not None:
                continue
            state = game.load(str(track / '0000.state'), randomize_rng=False, controllable=True)
            ram = bytes(state.ram)
            meta['start'] = {'bombs': ram[0x658], 'drop_cycle': ram[0x52A], 'help_count': ram[0x50],
                             'help_value': ram[0x51], 'kill_streak': ram[0x627], 'health': state.health,
                             'read_from': '0000.state'}
            path.write_text(json.dumps(meta, indent=1), encoding='utf-8')
            done += 1
    print(f'{done} tracks annotated')
    return 0


def compose(store: Path, adds: list, walk: Path, keep: list, out: Path):
    doors = {door_of(Path(name).stem) for name in json.loads(walk.read_text(encoding='utf-8'))['doors']}
    if out.exists():
        raise SystemExit(f'{out} exists')
    report = defaultdict(lambda: {'kept': 0, 'dropped': [], 'added': []})
    for track in tracks_under(store):
        family = track.parent.name
        meta = json.loads((track / 'trajectory.json').read_text(encoding='utf-8'))
        start = str(meta.get('start_state') or track.name)
        replaced = start.startswith('d3first_') and door_of(start) in doors
        if replaced and not any(fnmatch.fnmatchcase(f'{family}/{track.name}', g) for g in keep):
            report[family]['dropped'].append(track.name)
            continue
        target = out / family / track.name
        target.mkdir(parents=True)
        for file in track.iterdir():
            os.link(file, target / file.name)
        report[family]['kept'] += 1
    for root in adds:
        for track in tracks_under(root):
            family = track.parent.name
            target = out / family / track.name
            if target.exists():
                raise SystemExit(f'{family}/{track.name} is in the store and in {root}')
            shutil.copytree(track, target)
            report[family]['added'].append(track.name)
    summary = {f: {'kept': r['kept'], 'dropped': len(r['dropped']), 'added': len(r['added'])}
               for f, r in sorted(report.items())}
    files = sum(1 for _ in out.rglob('*') if _.is_file())
    record = {'store': str(store), 'adds': [str(a) for a in adds], 'walk': str(walk), 'doors': sorted(doors),
              'tracks': sum(r['kept'] + len(r['added']) for r in report.values()), 'files': files,
              'families': summary, 'detail': dict(report)}
    (out.parent / f'{out.name}.compose.json').write_text(json.dumps(record, indent=1), encoding='utf-8')
    print(json.dumps({k: record[k] for k in ('tracks', 'files', 'families')}, indent=1))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('table')
    t.add_argument('roots', nargs='+', type=Path)
    t.add_argument('--json', type=Path)
    r = sub.add_parser('replay')
    r.add_argument('root', type=Path)
    r.add_argument('--points', type=int, default=6)
    r.add_argument('--seeds', type=int, default=4)
    r.add_argument('--families', default=None)
    r.add_argument('--json', type=Path)
    a = sub.add_parser('annotate')
    a.add_argument('roots', nargs='+', type=Path)
    c = sub.add_parser('compose')
    c.add_argument('--store', required=True, type=Path)
    c.add_argument('--add', action='append', default=[], type=Path)
    c.add_argument('--walk', required=True, type=Path)
    c.add_argument('--keep', action='append', default=[])
    c.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == 'table':
        return table(args.roots, args.json)
    if args.command == 'annotate':
        return annotate(args.roots)
    if args.command == 'replay':
        return replay(args.root, args.points, args.seeds, args.families.split(',') if args.families else None,
                      args.json)
    return compose(args.store, args.add, args.walk, args.keep, args.out)


if __name__ == '__main__':
    sys.exit(main())
