#!/usr/bin/env python3
"""Round 40 (Tracks40): the fresh generation's curricula audited against its start states, and the folder proven.

    python scripts/experiments/r40_tracks.py audit --trajectories DIR [--trajectories DIR2 ...] --out FILE.json
    python scripts/experiments/r40_tracks.py table FILE.json [--route d3first|d2first] [--summary]
    python scripts/experiments/r40_tracks.py compose FILE.json --store DIR --add DIR [--add DIR] [--exclude GLOB]
        --out NEW_DIR
    python scripts/experiments/r40_tracks.py proof --trajectories DIR [--out FILE.json]

**What training checks.** A curriculum track is tied to its seed state by one thing only: its ``trajectory.json``
``start_state``, a state *name* (``triforce.inputs.check_curriculum``, ``track_starts``; the native loader
``env::curriculum::load_tracks`` reads the same name, and a curriculum reset takes that state's route entry). No hash or
bytes of the seed file are kept, so a track explored from an older file of the same name passes the check.

**What ``audit`` compares** (``audit``, a job): for every track under each ``--trajectories`` root, its first state
(``0000.state``: the explorer's episode right after the reset, ``env::explore::Explorer::write``) against the same
point explored from today's seed file of its ``start_state``: ``Explorer(task).explore(start_state, iterations=0)``,
the explorer's own reset and nothing more (``task``: the family's ``specialists/d3first/<family>.yaml`` for a
``d3first_`` state when there is one, else its own yaml). The track's state is loaded as ``r40_states.py read`` loads
a state (``Game.load``, ``randomize_rng=False``, ``controllable``, no buttons to game mode 5, 9 or 11; it is at that
frame already). Both are read: the room (level, room, cave), Link's position (``$70``/``$84``), the hearts (health,
containers, the partial heart), the inventory block ``$657``-``$67F`` (bombs ``$658``, keys ``$66E``, rupees, the
triforce, every item) and the kill counters (``$52A``, ``$50``, ``$627``). A track **belongs** when all of them are
equal; it is **stale** when any differs (it was recorded from another file of that name, or at other hearts on
purpose: a rule's tracks at the family's hearts); **no-state** when no seed file has its ``start_state`` (an orphan,
loaded by training all the same). The reset, not the bare file, is the reference: the session tops a bomb wall's
grant up on entering (``CheatLedger::settle``: 3:0x5b from S, 4:0x11, 5:0x65 hold 1 bomb more than their files) and a
cave's or a whirlwind landing's first decision is later than the file's first playable frame (Link's position). The
RNG and the monsters are not compared (every reset draws them).

Per task (every family on ``d3first``'s steps: ``specialists/d3first/<family>.yaml`` when there is one, else the
family's own yaml, the lab's ``task_yaml`` rule; and every family on ``d2first``'s steps by its own yaml), each seed
state with its tracks and verdicts, and for a seed without a belonging track what the task says
(``curriculum.untracked`` or nothing).

``proof``: every task of the audit loaded against ``--trajectories`` (its ``<family>`` folder) as ``triforce train``
does before its job waits (``train.main``: ``check_task`` with the curriculum's folder, ``recipe_manifest``, the
``recipe_flags``), then the native curriculum built from it (``Curriculum::new`` through ``TrainTask``, which a
training builds before its first episode) and ``curriculum_tracks``: what the run would record. Nothing is trained.

``compose`` builds a new trajectories folder from an audit: every track of ``--store`` (hard links: the store's files
are read-only and never written; a track folder is only ever created, never written over) except the ``d3first_``
tracks the audit found stale or without a state, then every track of each ``--add`` root the audit found belonging
(copies), but those ``--exclude`` names (``family/track`` globs) and those whose name the folder already has. Tracks of
bare (``d2first``) states are kept as the store has them whatever their verdict (round 40 changed no ``d2first``
state). It prints what it dropped and added and writes the list beside the audit (``compose.json``).
"""
import argparse
import fnmatch
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

from triforce.paths import STATES_DIR

MODE, PLAYABLE = 0x12, (5, 9, 11)
LINK_X, LINK_Y = 0x70, 0x84
INVENTORY = range(0x657, 0x680)
COUNTERS = {'cycle $52A': 0x52A, 'help $50': 0x50, 'streak $627': 0x627}
NAMED = {0x658: 'bombs', 0x66E: 'keys', 0x66F: 'containers', 0x671: 'triforce', 0x66D: 'rupees'}
ROUTES = ('d3first', 'd2first')


def tasks(route: str) -> list:
    """(family, task) of every family on ``route``'s steps, in route order: on d3first the route part's own yaml
    when there is one (``lab.collect.task_yaml``), else the family's."""
    from triforce import _native  # pylint: disable=import-outside-toplevel
    from triforce.lab.collect import task_yaml  # pylint: disable=import-outside-toplevel
    families = list(OrderedDict.fromkeys(step['specialist'] for step in _native.route_steps(route)))
    return [(f, task_yaml(f'{f}@{route}') if route == 'd3first' else f) for f in families]


def read_state(game, path: str) -> dict:
    """The fields the audit compares, at the state's first playable frame."""
    state = game.load(str(path), randomize_rng=False, controllable=True)
    for _ in range(600):
        if game.read(MODE) in PLAYABLE:
            break
        game.press('NONE', 1)
    state = game.state
    return {'mode': game.read(MODE), 'location': list(state.full_location), 'link': [game.read(LINK_X),
            game.read(LINK_Y)], 'health': state.health, 'inventory': [game.read(a) for a in INVENTORY],
            'counters': {k: game.read(a) for k, a in COUNTERS.items()}}


def explore_task(family: str, start: str):
    """The task a track of ``family`` from ``start`` is explored with: the family's d3first part's yaml for a
    ``d3first_`` state when there is one, else its own; None for neither."""
    root = Path(__file__).resolve().parents[2] / 'specialists'
    if start.startswith('d3first_') and (root / 'd3first' / f'{family}.yaml').is_file():
        return str(root / 'd3first' / f'{family}.yaml')
    return str(root / f'{family}.yaml') if (root / f'{family}.yaml').is_file() else None


def reset_state(explorers: dict, family: str, start: str):
    """The fields ``read_state`` reads, at the explorer's own reset from today's ``start`` (no iteration: the point a
    new track's ``0000.state`` is written from); a string saying why when there is none."""
    from triforce._native import Explorer  # pylint: disable=import-outside-toplevel
    from triforce.specialist import load_specialist  # pylint: disable=import-outside-toplevel
    task = explore_task(family, start)
    if task is None:
        return f'no task file for {family}'
    if task not in explorers:
        explorers[task] = Explorer(load_specialist(task), seed=0, states_dir=str(STATES_DIR))
    explorer = explorers[task]
    try:
        explorer.explore(start, iterations=0, patience=0)
    except (ValueError, RuntimeError) as error:
        return f'no reset of {start} under {Path(task).name}: {error}'
    state = explorer.state
    return {'mode': state.read(MODE), 'location': list(state.full_location),
            'link': [state.read(LINK_X), state.read(LINK_Y)], 'health': state.health,
            'inventory': [state.read(a) for a in INVENTORY],
            'counters': {k: state.read(a) for k, a in COUNTERS.items()}, 'task': short(task)}


def differences(track: dict, seed: dict) -> list:
    """What differs between a track's first state and its seed state, as 'field track->seed'."""
    out = []
    if track['mode'] not in PLAYABLE or seed['mode'] not in PLAYABLE:
        out.append(f"mode {track['mode']}->{seed['mode']}")
    if track['location'] != seed['location']:
        out.append(f"room {track['location']}->{seed['location']}")
    if track['link'] != seed['link']:
        out.append(f"link {tuple(track['link'])}->{tuple(seed['link'])}")
    if track['health'] != seed['health']:
        out.append(f"health {track['health']:g}->{seed['health']:g}")
    for address, a, b in zip(INVENTORY, track['inventory'], seed['inventory']):
        if a != b:
            out.append(f"{NAMED.get(address, f'${address:03X}')} {a}->{b}")
    for key, value in track['counters'].items():
        if value != seed['counters'][key]:
            out.append(f"{key} {value}->{seed['counters'][key]}")
    return out


def tracks_under(root: Path) -> list:
    """(family, track, start_state, meta) of every track under a trajectories root."""
    out = []
    for meta_path in sorted(root.glob('*/*/trajectory.json')):
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        start = str(meta.get('start_state') or meta_path.parent.name)
        start = start[:-len('.state')] if start.endswith('.state') else start
        out.append((meta_path.parent.parent.name, meta_path.parent.name, start, meta))
    return out


def audit(roots: list, out: Path):
    from triforce import jobs  # pylint: disable=import-outside-toplevel
    from triforce._native import Game  # pylint: disable=import-outside-toplevel
    from triforce.specialist import load_specialist  # pylint: disable=import-outside-toplevel
    from triforce.inputs import untracked  # pylint: disable=import-outside-toplevel
    jobs.require_job()
    game = Game(seed=0, states_dir=str(STATES_DIR))
    seeds_read: dict = {}
    explorers: dict = {}
    result = {'roots': [str(r) for r in roots], 'states_dir': str(STATES_DIR), 'tracks': {}, 'tasks': []}
    every = [(root, *t) for root in roots for t in tracks_under(root)]
    for done, (root, family, track, start, meta) in enumerate(every, 1):
        seed_path = STATES_DIR / f'{start}.state'
        row = {'root': str(root), 'family': family, 'track': track, 'start_state': start,
               'steps': meta.get('steps'), 'success': meta.get('success'), 'crafted': meta.get('crafted'),
               'mtime': (root / family / track / 'trajectory.json').stat().st_mtime}
        first = root / family / track / '0000.state'
        if not seed_path.is_file():
            row.update(verdict='no-state', diffs=[f'no {seed_path.name}'])
        elif not first.is_file():
            row.update(verdict='stale', diffs=['no 0000.state'])
        else:
            if (family, start) not in seeds_read:
                seeds_read[family, start] = reset_state(explorers, family, start)
            reference = seeds_read[family, start]
            if isinstance(reference, str):
                row.update(verdict='stale', diffs=[reference])
            else:
                diffs = differences(read_state(game, first), reference)
                row.update(verdict='stale' if diffs else 'belongs', diffs=diffs, task=reference['task'])
        result['tracks'][f'{root}/{family}/{track}'] = row
        jobs.progress(done, len(every), 'tracks')
    for route in ROUTES:
        for family, task in tasks(route):
            config = load_specialist(task)
            declared = untracked(config)
            seeds = sorted(Path(p).stem for p in config.resolve_start_states(None))
            if route == 'd2first':
                seeds = [s for s in seeds if not s.startswith('d3first_')]
            folder = Path(config.curriculum.trajectories) if config.curriculum.trajectories else None
            rows = []
            for seed in seeds:
                found = [r for r in result['tracks'].values()
                         if r['start_state'] == seed and r['family'] == config.name
                         and (folder is None or Path(r['root']) / r['family'] == folder)]
                rows.append({'state': seed, 'tracks': [{k: r[k] for k in ('root', 'track', 'verdict', 'diffs')}
                                                       for r in found],
                             'untracked': next((g for g in declared if fnmatch.fnmatchcase(seed, g)), None)})
            result['tasks'].append({'route': route, 'family': family, 'task': task, 'name': config.name,
                                    'curriculum': config.curriculum.enabled, 'untracked': declared, 'seeds': rows})
    out.write_text(json.dumps(result, indent=1), encoding='utf-8')
    verdicts = Counter(r['verdict'] for r in result['tracks'].values())
    print(f"{len(result['tracks'])} tracks: {dict(verdicts)}; {len(seeds_read)} seed states read; {out}")


def seed_verdict(row: dict, root: str) -> str:
    """A seed state's verdict in ``root``: belongs (a belonging track), stale (only stale ones), none."""
    mine = [t for t in row['tracks'] if t['root'] == root]
    if any(t['verdict'] == 'belongs' for t in mine):
        return 'belongs'
    return 'stale' if mine else 'none'


def short(task: str) -> str:
    """A task as the docs name it: ``specialists/d3first/zol.yaml``, or the family's name."""
    return 'specialists/' + task.split('/specialists/', 1)[1] if '/specialists/' in task else task


def table(path: Path, route: str, summary: bool, root: str = None):
    """The audit of ``root`` (default the first audited) as markdown: per task (``--summary``), or per room entry of
    the route's own states (``d3first_*`` on d3first, bare names on d2first), its chains side by side: the verdict,
    the tracks (``=``: the track named as its state, else its suffix) and a stale track's first differences."""
    data = json.loads(path.read_text(encoding='utf-8'))
    root = str(Path(root).resolve()) if root else data['roots'][0]
    tasks_ = [t for t in data['tasks'] if t['route'] == route]
    if summary:
        print('| family | task | curriculum | seed states | belongs | stale | none, untracked | none, nothing |')
        print('|---|---|---|---:|---:|---:|---:|---:|')
        totals = Counter()
        for t in tasks_:
            c = Counter()
            for row in t['seeds']:
                v = seed_verdict(row, root)
                c[v if v != 'none' else ('untracked' if row['untracked'] else 'nothing')] += 1
            c['seeds'] = len(t['seeds'])
            totals.update(c)
            print(f"| {t['family']} | `{short(t['task'])}` | {'on' if t['curriculum'] else 'off'} | {c['seeds']} | "
                  f"{c['belongs']} | {c['stale']} | {c['untracked']} | {c['nothing']} |")
        print(f"| **all** | | | {totals['seeds']} | {totals['belongs']} | {totals['stale']} | {totals['untracked']} | "
              f"{totals['nothing']} |")
        return
    print('| family | entry | chain 0 | chain 1 | chain 2 | without a track the task says |')
    print('|---|---|---|---|---|---|')
    for t in tasks_:
        entries: dict = OrderedDict()
        for row in t['seeds']:
            stem, _, chain = row['state'].rpartition('_')
            if not chain.isdigit():                     # `start`: no chain
                stem, chain = row['state'], '0'
            entries.setdefault(stem, {})[chain] = row
        for stem, chains in entries.items():
            if stem.startswith('d3first_') != (route == 'd3first'):
                continue
            cells, says = [], set()
            for chain in ('0', '1', '2'):
                row = chains.get(chain)
                if row is None:
                    cells.append('-')
                    continue
                verdict = seed_verdict(row, root)
                mine = [x for x in row['tracks'] if x['root'] == root]
                names = ' '.join(f"`{x['track'][len(row['state']):] or '='}`" for x in mine)
                stale = next((x for x in mine if x['verdict'] != 'belongs'), None)
                diffs = f" ({', '.join(stale['diffs'][:2])}{', ...' if len(stale['diffs']) > 2 else ''})" \
                    if stale and verdict != 'belongs' else ''
                cells.append(f"{verdict} {names}{diffs}".strip())
                if verdict != 'belongs':
                    says.add(f"untracked `{row['untracked']}`" if row['untracked'] else
                             'nothing (curriculum off)' if not t['curriculum'] else '**nothing: refused**')
            print(f"| {t['family']} | {stem[len('d3first_'):] if route == 'd3first' else stem} | {' | '.join(cells)} | "
                  f"{'; '.join(sorted(says)) or '-'} |")


def proof(root: Path, out: Path = None):
    """Every audited task loaded against ``root`` as ``triforce train`` loads it before training: ``root`` must be
    the folder the training's data rule picks (``train.data_sources``: this checkout's ``trajectories``).  Then every
    other task file under ``specialists/`` checked as ``tests/test_inputs.py``'s shipped-task test checks it
    (``check_task`` with the family's folder of ``root``)."""
    from triforce import jobs, registry, train  # pylint: disable=import-outside-toplevel
    from triforce._native import EnvPool, TrainTask  # pylint: disable=import-outside-toplevel
    from triforce.inputs import MissingInput, check_task  # pylint: disable=import-outside-toplevel
    from triforce.specialist import load_specialist, task_files  # pylint: disable=import-outside-toplevel
    import tempfile  # pylint: disable=import-outside-toplevel
    jobs.require_job()
    raised = []

    def capture(code, text, level='red'):
        raised.append({'code': code, 'text': text, 'level': level})
    train.flag = capture            # recipe_flags' jobs.flag, kept here instead of on this job's row
    rows, seen = [], set()
    for route in ROUTES:
        for family, task in tasks(route):
            if task in seen:
                continue
            seen.add(task)
            raised.clear()
            args = train.parse_args(['--specialist', task])
            train.validate_args(args)
            registry.use_team(args.team, args.models_dir)
            probe = train.Where(Path(), Path(), data=train.data_sources(args))
            if probe.folder('trajectories') is None or probe.folder('trajectories').resolve() != root.resolve():
                raise SystemExit(f"the training's trajectories are {probe.folder('trajectories')}, not {root}")
            sources = train.source_lines(args, probe.data)
            config = train.with_data(load_specialist(task), probe)
            folder = train.curriculum_dir(args, config, probe.folder('trajectories'))
            row = {'route': route, 'family': family, 'task': task, 'trajectories': sources['trajectories'],
                   'folder': folder}
            try:
                check_task(config, args.states_dir, folder)
                recipe = train.recipe_manifest(args, config, None, probe.folder('trajectories'))
                train.recipe_flags(args, config, recipe, 0, registry.models_root(args.models_dir))
                with tempfile.TemporaryDirectory() as tmp:
                    spec, _ = train.curriculum_spec(args, config, Path(tmp), None, probe.folder('trajectories'))
                    task_ = TrainTask(config, tmp, start_state=None, curriculum=spec, states_dir=args.states_dir)
                    pool = EnvPool(task_, num_envs=1, threads=1, seed=0)
                    del pool
                    state = Path(tmp) / 'curriculum.json'
                    loaded = len(json.loads(state.read_text(encoding='utf-8'))) if state.is_file() else None
                tracks = train.curriculum_tracks(folder)
                row.update(refused=None, flags=list(raised), curriculum_tracks=None if tracks is None else len(tracks),
                           loaded=loaded, manifest_curriculum=recipe['data'].get('curriculum'))
            except (MissingInput, SystemExit, ValueError, OSError) as error:
                row.update(refused=f"{type(error).__name__}: {error}", flags=list(raised))
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'manifest_curriculum'}), flush=True)
            jobs.progress(len(rows), None, 'tasks')
    names = {str(Path(t if t.endswith('.yaml') else f'specialists/{t}.yaml').resolve()) for t in seen}
    for path in task_files(Path(__file__).resolve().parents[2] / 'specialists'):
        if str(path) in names:
            continue
        config = load_specialist(str(path))
        folder = (str(config.curriculum.trajectories or root / config.name) if config.curriculum.enabled else None)
        row = {'route': 'other', 'family': config.name, 'task': short(str(path)), 'folder': folder, 'flags': []}
        try:
            check_task(config, None, folder)
            row['refused'] = None
        except MissingInput as error:
            row['refused'] = f"MissingInput: {error}"
        rows.append(row)
        print(json.dumps(row), flush=True)
    if out:
        out.write_text(json.dumps(rows, indent=1), encoding='utf-8')
    print(f"{len(rows)} tasks; refused {sum(1 for r in rows if r['refused'])}; "
          f"red flags {sum(1 for r in rows for f in r['flags'] if f['level'] == 'red')}")
    return 1 if any(r['refused'] for r in rows) else 0


def compose(audit_file: Path, store: Path, adds: list, excludes: list, out: Path):
    import os  # pylint: disable=import-outside-toplevel
    import shutil  # pylint: disable=import-outside-toplevel
    from triforce import jobs  # pylint: disable=import-outside-toplevel
    jobs.require_job()
    data = json.loads(audit_file.read_text(encoding='utf-8'))
    if out.exists():
        raise SystemExit(f"{out} exists: compose builds a new folder only")
    rows = {key: row for key, row in data['tracks'].items()}
    report = {'store': str(store), 'out': str(out), 'kept': 0, 'dropped': [], 'added': [], 'skipped': []}
    for key, row in sorted(rows.items()):
        if Path(row['root']) != store.resolve():
            continue
        name = f"{row['family']}/{row['track']}"
        if row['start_state'].startswith('d3first_') and row['verdict'] != 'belongs':
            report['dropped'].append({'track': name, 'verdict': row['verdict'], 'diffs': row['diffs']})
            continue
        source, target = store / name, out / name
        target.mkdir(parents=True)
        for path in sorted(source.iterdir()):
            os.link(path, target / path.name)
        report['kept'] += 1
    for add in adds:
        for key, row in sorted(rows.items()):
            if Path(row['root']) != add.resolve():
                continue
            name = f"{row['family']}/{row['track']}"
            if row['verdict'] != 'belongs' or any(fnmatch.fnmatchcase(name, g) for g in excludes):
                report['skipped'].append({'track': f"{add}/{name}", 'verdict': row['verdict'], 'diffs': row['diffs']})
                continue
            target = out / name
            if target.exists():
                report['skipped'].append({'track': f"{add}/{name}", 'verdict': 'name taken', 'diffs': []})
                continue
            shutil.copytree(add / name, target)
            report['added'].append({'track': name, 'from': str(add), 'start_state': row['start_state']})
    files = sum(1 for p in out.rglob('*') if p.is_file())
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    report.update(files=files, bytes=size, tracks=sum(1 for _ in out.glob('*/*/trajectory.json')))
    (audit_file.parent / 'compose.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f"{out}: {report['tracks']} tracks, {files} files, {size / 1e6:.1f} MB; kept {report['kept']} of the "
          f"store's, dropped {len(report['dropped'])}, added {len(report['added'])}, skipped {len(report['skipped'])}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit')
    a.add_argument('--trajectories', action='append', required=True, type=Path)
    a.add_argument('--out', required=True, type=Path)
    t = sub.add_parser('table')
    t.add_argument('file', type=Path)
    t.add_argument('--route', default='d3first', choices=ROUTES)
    t.add_argument('--root', default=None)
    t.add_argument('--summary', action='store_true')
    c = sub.add_parser('compose')
    c.add_argument('file', type=Path)
    c.add_argument('--store', required=True, type=Path)
    c.add_argument('--add', action='append', default=[], type=Path)
    c.add_argument('--exclude', action='append', default=[])
    c.add_argument('--out', required=True, type=Path)
    p = sub.add_parser('proof')
    p.add_argument('--trajectories', required=True, type=Path)
    p.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'audit':
        audit([r.resolve() for r in args.trajectories], args.out)
    elif args.command == 'table':
        table(args.file, args.route, args.summary, args.root)
    elif args.command == 'compose':
        compose(args.file, args.store.resolve(), args.add, args.exclude, args.out)
    else:
        return proof(args.trajectories, args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
