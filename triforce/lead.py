"""``triforce-lead``: the lead's operations on the shared store (``triforce.store``), apart from the public
``triforce store``.

Every operation here is a row of the store's audit log with entry point ``lead``:

- ``init``: make the store (root, index, README, layout).
- ``team new NAME (--from-models DIR | --from-manifest FILE | --base TEAM) [--slot SLOT=RUN ...]``: an immutable team.
- ``team set-current NAME``: the team of record (replaces ``triforce promote``).
- ``data install KIND NAME --from DIR``, ``data set-current KIND NAME``: named snapshots (copies).
- ``import-legacy --checkout DIR [--apply] [--manifests DIR]``: a copy of today's checkout into the store (dry run
  without ``--apply``; idempotent: changed sources are reported, never overwritten; never writes the source).
- ``delete run|team|data|job ...``: what nothing refers to any more.
- ``backup [--to DIR] [--apply]``: copies what is not yet in the archive (``/backup/triforce/store``).
"""

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

from . import paths, store
from .store import LEAD, Run

BACKUP_ENV = 'TRIFORCE_BACKUP'
DEFAULT_BACKUP = Path('/backup/triforce')
MANIFESTS = DEFAULT_BACKUP / 'models' / 'teams'
CURRENT_TEAM = '2.0.0.13'
# Legacy data folders of a checkout: (kind, path relative to the checkout); the trajectories' hand-made older copies
# are found by their glob.
LEGACY_DATA = (('harvested', 'triforce/states/harvested'), ('harvested-rooms', 'triforce/states/harvested-rooms'),
               ('trajectories', 'trajectories'), ('demos', 'target/demos'))
POINTERS = ('promoted', 'late')
RECORD_SUFFIXES = ('.jsonl', '.json', '.csv', '.tsv', '.md', '.txt', '.log')


# ---------------------------------------------------------------------------------------------------------------
# Teams from a models folder's pointers and from the archive's manifests

def pointer_slots(models: Path) -> dict:
    """slot key -> (family, run name) of every pointer file of a models folder (``promoted``,
    ``promoted.<route>``, ``late.<route>``)."""
    found = {}
    for family in sorted(p for p in Path(models).iterdir() if p.is_dir()):
        for pointer in sorted(family.iterdir()):
            head, _, route = pointer.name.partition('.')
            if not pointer.is_file() or head not in POINTERS or pointer.suffix == '.tmp' or \
                    (head == 'late' and not route):
                continue
            key = store.slot_key(family.name, route or None, head == 'late')
            found[key] = (family.name, pointer.read_text(encoding='utf-8').strip())
    return found


def team_from_models(models: Path) -> tuple[dict, list[str]]:
    """(slots, problems): every pointer of ``models`` resolved to the store run of the same family and name, with
    the act mode the folder gives it (its ``act`` file, ``registry.act_mode``)."""
    from . import registry  # pylint: disable=import-outside-toplevel
    slots, problems = {}, []
    for key, (family, name) in pointer_slots(models).items():
        run = Run(family, name)
        row = store.run_row(run)
        if row is None or row['status'] != store.SEALED:
            problems.append(f"{key}: {run.identity} is not a sealed run of the store")
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            act = registry.act_mode(Path(models) / family / name / store.CHECKPOINT)
        slots[key] = store.slot_of(run, act=act, hit_head=bool(row.get('hit_head')))
    return slots, problems


def team_from_manifest(manifest: dict) -> tuple[dict, list[str]]:
    """(slots, problems) of an archive manifest (``/backup/triforce/models/teams/<version>.json``): its route of
    record's networks, ``family@route`` keys as route slots and bare keys as the family's slot (they play that route
    as recorded), each resolved to the store run of that path and SHA-1, else any run with that SHA-1."""
    route = manifest.get('route_of_record') or next(iter(manifest['routes']))
    networks = manifest['routes'][route]['networks']
    slots, problems = {}, []
    con = store.connect()
    try:
        for key, network in sorted(networks.items()):
            sha1 = network.get('checkpoint_sha1')
            run = store.parse_run(network['run'])
            row = store.run_row(run)
            if row is None or row['sha1'] != sha1:
                other = con.execute("SELECT family, name FROM runs WHERE sha1 = ? AND status = 'sealed' "
                                    "ORDER BY family = ? DESC, name LIMIT 1", (sha1, run.family)).fetchone()
                if other is None:
                    problems.append(f"{key}: {network['run']} (sha1 {sha1}) is in no run of the store")
                    continue
                run = Run(other['family'], other['name'])
                row = store.run_row(run)
            slots[key] = store.slot_of(run, act=network.get('act'), hit_head=bool(row.get('hit_head')))
    finally:
        con.close()
    return slots, problems


# ---------------------------------------------------------------------------------------------------------------
# The import of a checkout

def _newest(path: Path) -> float:
    newest = path.stat().st_mtime
    for dirpath, _, filenames in os.walk(path, followlinks=True):
        for name in filenames:
            try:
                newest = max(newest, (Path(dirpath) / name).stat().st_mtime)
            except OSError:
                pass
    return newest


def _is_after_training(name: str) -> bool:
    return name in store.AFTER_TRAINING or (name.startswith('eval') and name.endswith('.json'))


def scan_models(models: Path) -> dict:
    """Classifies everything in a models folder: ``runs`` (each dir holding a ``ckpt.pt``, outside numbered
    checkpoints: family = its parent's path, which may be a group's or evidence folder's), each with the files filed
    by job instead (``after``: evaluations, ``act``, ``selection.json``) and the nested runs it excludes;
    ``pointers``; ``records`` (chain records, summaries, evidence tables: files with a record's suffix outside run
    dirs); ``legacy`` (everything else: no home, kept under ``legacy/``)."""
    models = Path(models)
    run_dirs = []
    for dirpath, dirnames, filenames in os.walk(models):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if d not in ('checkpoints', 'tb'))
        if here != models and store.CHECKPOINT in filenames:
            run_dirs.append(here)
    runs = []
    run_set = set(run_dirs)
    for here in run_dirs:
        nested = [str(d.relative_to(here)) for d in run_set if d != here and here in d.parents]
        after = sorted(p.name for p in here.iterdir() if p.is_file() and _is_after_training(p.name))
        runs.append({'family': str(here.parent.relative_to(models)), 'name': here.name, 'source': str(here),
                     'after': after, 'nested': nested})
    covered = lambda path: any(d == path or d in path.parents for d in run_set)  # noqa: E731
    pointers, records, legacy = [], [], []
    for dirpath, dirnames, filenames in os.walk(models):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not covered(here / d))
        for name in sorted(filenames):
            path = here / name
            relative = str(path.relative_to(models))
            head, _, route = name.partition('.')
            if here != models and here.parent == models and head in POINTERS and \
                    (head == 'promoted' or route) and not name.endswith('.tmp'):
                pointers.append(relative)
            elif name.endswith(RECORD_SUFFIXES):
                records.append(relative)
            else:
                legacy.append(relative)
    return {'runs': runs, 'pointers': pointers, 'records': records, 'legacy': legacy}


def plan_import(checkout: Path, manifests: Optional[Path] = MANIFESTS) -> dict:
    """What ``import_legacy`` would do for ``checkout`` (no write): the ROM, each data folder (its kind, a name by
    date and content, files, bytes), the models folder's classification (``scan_models``), the manifests found."""
    checkout = Path(checkout).resolve()
    plan = {'checkout': str(checkout), 'rom': None, 'data': [], 'models': None, 'manifests': []}
    rom = checkout / paths.CHECKOUT_ROM
    if rom.is_file():
        plan['rom'] = str(rom)
    sources = [(kind, checkout / relative, True) for kind, relative in LEGACY_DATA]
    sources += [('trajectories', path, False) for path in sorted(checkout.glob('trajectories.prev-*'))]
    for kind, source, current in sources:
        if not source.is_dir():
            continue
        sha256, files, size = store.tree_hash(source)
        name = f"{time.strftime('%y%m%d', time.localtime(_newest(source)))}-{sha256[:8]}"
        plan['data'].append({'kind': kind, 'source': str(source), 'name': name, 'sha256': sha256, 'files': files,
                             'bytes': size, 'current': current})
    models = checkout / 'models'
    if models.is_dir():
        plan['models'] = scan_models(models)
    if manifests is not None and Path(manifests).is_dir():
        plan['manifests'] = [str(p) for p in sorted(Path(manifests).glob('*.json'))]
    return plan


def _hit_head(path: Path) -> bool:
    try:
        return store._hit_head(path)  # pylint: disable=protected-access
    except Exception:  # pylint: disable=broad-except
        return False


def import_legacy(checkout: Path, apply: bool = False, manifests: Optional[Path] = MANIFESTS,
                  team: str = CURRENT_TEAM, out=print) -> dict:
    """Imports a COPY of ``checkout``'s shared things into the store (``plan_import``; times and modes kept, then
    read-only; never a hard link, so the checkout stays writable and a write there never changes the store): the ROM;
    each data folder as a named snapshot (the checkout's own folders made current when the kind has none); every run
    dir as a sealed run (its evaluations, ``act`` and ``selection.json`` filed by this job, indexed by the run); the
    loose records filed by this job; what has no home under ``legacy/import-<date>/`` (listed); the team ``team`` from
    the checkout's pointers (made current) and a team per archive manifest whose runs resolve.  Idempotent: what is
    already there with the same content is skipped; a run or data folder that changed in the checkout since its
    import is reported in ``changed`` (the reconciliation's list) and the store's copy is never overwritten.  Without
    ``apply`` only the plan and its counts are printed.  The source is never written."""
    from . import registry  # pylint: disable=import-outside-toplevel
    started = time.time()
    plan = plan_import(checkout, manifests)
    report = {'plan': plan, 'done': [], 'skipped': [], 'changed': [], 'problems': [], 'teams': {}, 'legacy': []}
    models = plan['models'] or {'runs': [], 'pointers': [], 'records': [], 'legacy': []}
    out(f"import of {plan['checkout']} into {store.root()} ({'apply' if apply else 'dry run'})")
    out(f"  rom: {plan['rom'] or 'none'}")
    for item in plan['data']:
        out(f"  data {item['kind']:15} {item['name']:18} {item['files']:7} files {item['bytes'] / 1e9:7.2f} GB  "
            f"{item['source']}{'  (current)' if item['current'] else ''}")
    out(f"  models: {len(models['runs'])} runs, {sum(len(r['after']) for r in models['runs'])} evaluation/act/"
        f"selection files, {len(models['pointers'])} pointers, {len(models['records'])} records, "
        f"{len(models['legacy'])} files without a home (legacy/)")
    out(f"  manifests: {len(plan['manifests'])}")
    if not apply:
        return report
    if not paths.store_exists():
        store.init()
    stamp = time.strftime('%y%m%d')
    # The ROM.
    if plan['rom']:
        target = store.root() / paths.STORE_ROM
        if target.exists():
            report['skipped'].append('rom')
        else:
            with store.writing('rom-install', {'source': plan['rom']}, LEAD):
                target.parent.mkdir(parents=True, exist_ok=True)
                store.link_tree(Path(plan['rom']), target, link=False)
                target.chmod(0o444)
            report['done'].append('rom')
    # The data snapshots.
    existing = {(row['kind'], row['sha256']): row['name'] for row in store.snapshots()}
    sources = {(row['kind'], row['source']) for row in store.snapshots()}
    for item in plan['data']:
        name = existing.get((item['kind'], item['sha256']))
        if name is None:
            if (item['kind'], str(Path(item['source']).resolve())) in sources:
                report['changed'].append(f"data {item['kind']} {item['source']}: new content, installed as "
                                         f"{item['name']} (the current snapshot is unchanged)")
            store.install_snapshot(item['kind'], item['name'], Path(item['source']),
                                   note=f"import of {item['source']}")
            name = item['name']
            report['done'].append(f"data {item['kind']}/{name}")
        else:
            report['skipped'].append(f"data {item['kind']}/{name}")
        if item['current'] and paths.current_name(item['kind']) is None:
            store.set_current_snapshot(item['kind'], name)
    # The runs.
    imported = 0
    for entry in models['runs']:
        run = Run(entry['family'], entry['name'])
        source = Path(entry['source'])
        row = store.run_row(run)
        if row is not None:
            why = store.source_changed(run, source, [*entry['after'], *entry['nested']])
            if why:
                report['changed'].append(f"run {run.identity}: {why} in {source} since its import (the store's "
                                         f"copy is left as it is)")
            report['skipped'].append(f"run {run.identity}")
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            act = registry.act_mode(source / store.CHECKPOINT)
        store.install_run(run, source, act, exclude=[*entry['after'], *entry['nested']],
                          note=f"import of {source}")
        if _hit_head(source / store.CHECKPOINT):
            with store.writing('run-hit-head', {'run': run.identity}, LEAD) as con:
                con.execute('UPDATE runs SET hit_head = 1 WHERE family = ? AND name = ?', (run.family, run.name))
        for name in entry['after']:
            kind = 'evaluate' if name.startswith('eval') else name.split('.')[0]
            result = _evaluation_counts(source / name) if kind == 'evaluate' else None
            store.file_output(source / name, kind, run=run.identity, name=f'{run.identity}/{name}', result=result,
                              entry=LEAD)
        imported += 1
    report['done'].append(f"{imported} runs")
    # Records and pointers (filed by this job), and what has no home (legacy/).
    models_dir = Path(plan['checkout']) / 'models'
    filed = {row['name'] for row in store.outputs(kind='record')}
    for relative in models['records'] + models['pointers']:
        if f'models/{relative}' in filed:
            continue
        store.file_output(models_dir / relative, 'record', name=f'models/{relative}', entry=LEAD)
    legacy_root = store.root() / 'legacy' / f'import-{stamp}'
    for relative in models['legacy']:
        target = legacy_root / 'models' / relative
        report['legacy'].append(f'models/{relative}')
        if target.exists():
            continue
        with store.writing('legacy', {'path': f'models/{relative}', 'into': str(target)}, LEAD):
            store.link_tree(models_dir / relative, target, link=False)
            target.chmod(0o444)
    if report['legacy']:
        listing = legacy_root / 'LIST.txt'
        if not listing.exists():
            listing.write_text('\n'.join(report['legacy']) + '\n', encoding='utf-8')
            listing.chmod(0o444)
    # The teams.
    known = {row['name'] for row in store.teams()}
    if plan['models'] is not None and team not in known:
        slots, problems = team_from_models(models_dir)
        report['problems'] += [f"team {team}: {p}" for p in problems]
        if not problems:
            store.create_team(team, slots, f"pointers of {models_dir} on {time.strftime('%Y-%m-%d')}",
                              entry=LEAD)
            known.add(team)
    if team in known:
        report['teams'][team] = 'from the pointers'
        if paths.current_team() is None:
            store.set_current_team(team)
    for manifest_path in plan['manifests']:
        manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        name = manifest.get('version') or Path(manifest_path).stem
        if name in known:
            report['teams'].setdefault(name, f'exists (manifest {manifest_path} not used)')
            continue
        slots, problems = team_from_manifest(manifest)
        if problems:
            report['teams'][name] = f"not made: {len(problems)} of {len(problems) + len(slots)} networks resolve " \
                                    f"to no run ({problems[0]}{', ...' if len(problems) > 1 else ''})"
            continue
        store.create_team(name, slots, f"manifest {manifest_path}", note=manifest.get('routes', {}).get(
            manifest.get('route_of_record'), {}).get('how'), entry=LEAD)
        known.add(name)
        report['teams'][name] = f'from {manifest_path}'
    report['seconds'] = round(time.time() - started, 1)
    report_path = store.job_folder('import-legacy') / f"import-report-{time.strftime('%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    store.file_output(report_path, 'record', entry=LEAD)
    out(f"  done in {report['seconds']} s: {', '.join(report['done']) or 'nothing'}; skipped "
        f"{len(report['skipped'])}; legacy {len(report['legacy'])}; teams {json.dumps(report['teams'], indent=1)}")
    for line in report['changed']:
        out(f"  changed in the checkout since its import: {line}")
    for problem in report['problems']:
        out(f"  problem: {problem}")
    out(f"  report: {report_path}")
    return report


def _evaluation_counts(path: Path) -> Optional[dict]:
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
        episodes = int(result['episodes'])
        return {'n': episodes, 'successes': round(result['success_rate'] * episodes)}
    except (OSError, ValueError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------------------------------------------
# Delete and back up

def delete(what: str, name: str, kind: Optional[str] = None) -> str:
    """Deletes a run (in no team), a team (not current), a snapshot (not current) or a job's filed files (its kept
    summary rows stay: the aggregate is kept for good)."""
    if what == 'run':
        run = store.parse_run(name)
        row = store.get_run(run)
        users = [t['name'] for t in store.teams()
                 if any(slot['run'] == run.identity for slot in store.read_team(t['name'])['slots'].values())]
        if users:
            raise store.StoreError(f"run {run.identity} plays in team(s) {', '.join(users)}")
        with store.writing('delete-run', {'run': run.identity, 'sha1': row['sha1']}, LEAD) as con:
            store.make_writable(run.path)
            shutil.rmtree(run.path)
            con.execute('DELETE FROM runs WHERE family = ? AND name = ?', (run.family, run.name))
        return f"deleted run {run.identity}"
    if what == 'team':
        if name == paths.current_team():
            raise store.StoreError(f"team {name} is the current one")
        with store.writing('delete-team', {'name': name}, LEAD) as con:
            path = paths.team_file(name)
            store.make_writable(path)
            path.unlink()
            con.execute('DELETE FROM teams WHERE name = ?', (name,))
        return f"deleted team {name}"
    if what == 'data':
        if name == paths.current_name(kind):
            raise store.StoreError(f"{kind} snapshot {name} is the current one")
        path = paths.data(kind, name)
        with store.writing('delete-data', {'kind': kind, 'name': name}, LEAD) as con:
            store.make_writable(path)
            shutil.rmtree(path)
            con.execute('DELETE FROM snapshots WHERE kind = ? AND name = ?', (kind, name))
        return f"deleted {kind} snapshot {name}"
    if what == 'job':
        con = store.connect()
        try:
            row = con.execute('SELECT folder FROM jobs WHERE job = ?', (name,)).fetchone()
        finally:
            con.close()
        if row is None:
            raise store.MissingError(f"no job {name} in the store")
        folder = store.root() / row['folder']
        with store.writing('delete-job-files', {'job': name, 'folder': row['folder']}, LEAD) as con:
            store.make_writable(folder)
            shutil.rmtree(folder, ignore_errors=True)
            con.execute('DELETE FROM outputs WHERE job = ?', (name,))
        return f"deleted the files of job {name} (its summary rows stay)"
    raise store.StoreError(f"delete what? run, team, data or job, not {what!r}")


def backup(to: Optional[Path] = None, apply: bool = False, out=print) -> list[str]:
    """Copies into the archive (``$TRIFORCE_BACKUP/store``, else ``/backup/triforce/store``) every sealed run, team,
    snapshot, job folder and the index that is not there yet (its own disk: real copies, made read-only)."""
    base = Path(to) if to else Path(os.environ.get(BACKUP_ENV) or DEFAULT_BACKUP) / 'store'
    items = [f"runs/{row['family']}/{row['name']}" for row in store.runs(status=store.SEALED)]
    items += [f"teams/{row['name']}.json" for row in store.teams()]
    items += [f"data/{row['kind']}/{row['name']}" for row in store.snapshots()]
    con = store.connect()
    try:
        items += [row['folder'] for row in con.execute('SELECT folder FROM jobs ORDER BY job')]
    finally:
        con.close()
    todo = [item for item in items if not (base / item).exists()]
    out(f"backup into {base}: {len(todo)} of {len(items)} to copy ({'apply' if apply else 'dry run'})")
    if not apply:
        return todo
    with store.writing('backup', {'to': str(base), 'items': len(todo)}, LEAD) as con:
        for item in todo:
            store.link_tree(store.root() / item, base / item, link=False)
            store.make_readonly(base / item)
        index = base / f"store.db.{time.strftime('%y%m%d-%H%M%S')}"
        con.execute(f"VACUUM INTO '{index}'")
    return todo


# ---------------------------------------------------------------------------------------------------------------
# The command

def main(argv=None) -> int:
    """``triforce-lead``."""
    parser = argparse.ArgumentParser(prog='triforce-lead', description="The lead's operations on the shared store "
                                     f"({paths.shared_root()}): each one is in the audit log.")
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init', help="make the store (root, index, README, layout.json)")
    team = sub.add_parser('team', help="make a team, set the current one").add_subparsers(dest='action',
                                                                                          required=True)
    new = team.add_parser('new', help="a new immutable team")
    new.add_argument('name')
    source = new.add_mutually_exclusive_group(required=True)
    source.add_argument('--from-models', type=Path, help="the pointers of an old-style models folder")
    source.add_argument('--from-manifest', type=Path, help="an archive manifest (/backup/triforce/models/teams)")
    source.add_argument('--base', help="an existing team, changed by --slot")
    new.add_argument('--slot', action='append', default=[], metavar='SLOT=RUN',
                     help="family, family@route or family@route:late = FAMILY/RUN (a sealed run)")
    new.add_argument('--note')
    current = team.add_parser('set-current', help="the team of record (replaces triforce promote)")
    current.add_argument('name')
    data = sub.add_parser('data', help="install a snapshot, set a kind's current one").add_subparsers(
        dest='action', required=True)
    install = data.add_parser('install')
    install.add_argument('kind', choices=paths.DATA_KINDS)
    install.add_argument('name')
    install.add_argument('--from', dest='source', type=Path, required=True)
    install.add_argument('--note')
    data_current = data.add_parser('set-current')
    data_current.add_argument('kind', choices=paths.DATA_KINDS)
    data_current.add_argument('name')
    imp = sub.add_parser('import-legacy', help="a checkout's ROM, data, models and pointers into the store (dry run "
                                               "without --apply)")
    imp.add_argument('--checkout', type=Path, default=Path('/home/leculver/git/triforce'))
    imp.add_argument('--apply', action='store_true')
    imp.add_argument('--manifests', type=Path, default=MANIFESTS)
    imp.add_argument('--team', default=CURRENT_TEAM, help="the name of the team the pointers make")
    remove = sub.add_parser('delete', help="a run in no team, a team or snapshot not current, a job's files")
    remove.add_argument('what', choices=('run', 'team', 'data', 'job'))
    remove.add_argument('name', help="FAMILY/RUN, TEAM, NAME (with --kind) or JOB")
    remove.add_argument('--kind', choices=paths.DATA_KINDS)
    back = sub.add_parser('backup', help="copy what the archive lacks (dry run without --apply)")
    back.add_argument('--to', type=Path)
    back.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            print(store.init())
        elif args.command == 'team' and args.action == 'new':
            problems = []
            if args.from_models:
                slots, problems = team_from_models(args.from_models)
                source_text = f"pointers of {args.from_models}"
            elif args.from_manifest:
                slots, problems = team_from_manifest(json.loads(args.from_manifest.read_text(encoding='utf-8')))
                source_text = f"manifest {args.from_manifest}"
            else:
                slots = dict(store.read_team(args.base)['slots'])
                source_text = f"team {args.base}"
            for item in args.slot:
                key, _, value = item.partition('=')
                slots[key] = store.slot_of(store.parse_run(value, store.split_slot(key)[0]))
                source_text += f" +{key}={slots[key]['run']}"
            if problems:
                raise store.StoreError('; '.join(problems))
            print(json.dumps(store.create_team(args.name, slots, source_text, args.note), indent=2))
        elif args.command == 'team':
            store.set_current_team(args.name)
            print(f"current team: {args.name}")
        elif args.command == 'data' and args.action == 'install':
            print(json.dumps(store.install_snapshot(args.kind, args.name, args.source, note=args.note), indent=2))
        elif args.command == 'data':
            print(store.set_current_snapshot(args.kind, args.name))
        elif args.command == 'import-legacy':
            report = import_legacy(args.checkout, args.apply, args.manifests, args.team)
            return 1 if report['problems'] else 0
        elif args.command == 'delete':
            if args.what == 'data' and not args.kind:
                parser.error("delete data NAME needs --kind")
            print(delete(args.what, args.name, args.kind))
        elif args.command == 'backup':
            backup(args.to, args.apply)
    except (store.StoreError, paths.MissingError) as error:
        print(f"triforce-lead: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
