"""The reconciliation of a checkout from before the store with the shared store (round 40; docs/data-layout.md
"The reconciliation of the main checkout"): what the checkout's own data folders hold that the store lacks, and
moving those folders ASIDE (never deleting them) so the checkout reads the store like every worktree.

    python -m triforce.reconcile compare [--checkout DIR] [--only AREA...] [--out REPORT.json]
    python -m triforce.reconcile aside [--checkout DIR] --report REPORT.json [--into DIR] [--accept-unbacked] [--apply]
    python -m triforce.reconcile back ASIDE_DIR [--apply]

``compare`` (a job of the machine's queue: it reads every byte of the folders and of the store's copies; ``--only``
splits it into pieces, and the hashes are kept in ``target/reconcile/sha1.db`` of the checkout it runs from, keyed by
path, size, mtime and inode, so a later full ``compare`` reads only what changed) compares by content (SHA-1), never
by name, each folder of ``FOLDERS`` that exists in the checkout with the store item that holds the same thing:

- the ROM with the store's ``rom/zelda.nes``;
- a data folder (``harvested``, ``harvested-rooms``, ``trajectories`` and its ``.prev-*`` copies, ``target/demos``)
  with every snapshot of its kind: ``identical`` to one (every path and content the same), else ``store has more``
  (every file's content is in some snapshot of the kind) or ``MAIN HAS FILES THE STORE LACKS`` (listed);
- ``models/`` as the lead's import classified it (``lead.scan_models``): every run dir with the store's run of that
  family and name (its files at their paths; its evaluations, ``act`` and ``selection.json`` with what the store filed
  for the run), the pointers and loose records with what the import filed as ``models/<path>``, the rest with
  ``legacy/import-*/models/<path>``; the pointers as slots against the store's current team (run and act mode).

Each file the store lacks is looked up in the archive (``/backup/triforce``, ``$TRIFORCE_BACKUP``): the models'
``files.tsv`` (SHA-1 of every kept file), the dated data copies ``data/<date>/<folder>`` and the store's own copy
``store/``.  The report (JSON) says per folder its files and bytes, the store item, the status, the files only the
checkout holds (with whether the archive has them) and what the archive holds of the folder.

``aside`` moves every folder of ``FOLDERS`` the checkout has into one dated folder beside it
(``<checkout>.aside-<yymmdd>``, same disk: a rename each, instant) with ``ASIDE.json`` naming them, after checking:
the report covers every folder and is newer than every file in them; nothing it lists as only the checkout's is
missing from the archive (``--accept-unbacked`` takes them as they are); the store has its ROM, a current team and a
current snapshot of every kind (what the checkout reads once its folders are gone); no live job of the queue works in
the checkout and no process holds a file of the folders open.  Without ``--apply`` it prints the checks and the
moves.  ``back ASIDE_DIR --apply`` puts every folder back where it was (refusing one whose place is taken).
"""

import argparse
import json
import os
import sqlite3
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from . import jobs, paths, store

# The checkout's own folders that the store replaces (relative path, area, data kind or 'rom' / 'models'), in the
# order they are compared; ``trajectories.prev-*`` (hand-made copies the import installed as snapshots) are found by
# glob.  ``logs/``, ``runs/``, ``replays/`` and the rest of ``target/`` stay: per checkout (docs/data-layout.md).
FOLDERS = (('roms', 'rom'), ('trajectories', 'trajectories'), ('triforce/states/harvested', 'harvested'),
           ('triforce/states/harvested-rooms', 'harvested-rooms'), ('target/demos', 'demos'), ('models', 'models'))
PREV_GLOB = 'trajectories.prev-*'
ASIDE_FILE = 'ASIDE.json'
IDENTICAL, MORE, LACKS = 'identical', 'store has more', 'MAIN HAS FILES THE STORE LACKS'
TABLE_LIMIT = 50            # files listed per folder in the printed table (the JSON report has every one)


def folders(checkout: Path) -> list[tuple[str, str]]:
    """(relative path, kind) of every folder of ``FOLDERS`` (and ``trajectories.prev-*``) the checkout has."""
    found = [(rel, kind) for rel, kind in FOLDERS if (checkout / rel).exists()]
    found += [(path.name, 'trajectories') for path in sorted(checkout.glob(PREV_GLOB)) if path.is_dir()]
    return found


class Hashes:
    """SHA-1 of files, kept in an SQLite file keyed by path, size, mtime and inode (a file rewritten is hashed again);
    ``read`` counts the bytes actually hashed (the job's progress)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path)
        self.con.execute('CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, mtime INTEGER, '
                         'ino INTEGER, sha1 TEXT)')
        self.read = self.pending = 0

    def sha1(self, path: Path) -> str:
        st = path.stat()
        row = self.con.execute('SELECT size, mtime, ino, sha1 FROM files WHERE path = ?', (str(path),)).fetchone()
        if row is not None and tuple(row[:3]) == (st.st_size, st.st_mtime_ns, st.st_ino):
            return row[3]
        digest = store.file_sha1(path)
        self.con.execute('INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?)',
                         (str(path), st.st_size, st.st_mtime_ns, st.st_ino, digest))
        self.read += st.st_size
        self.pending += 1
        if self.pending >= 2000:
            self.commit()
        return digest

    def commit(self):
        self.con.commit()
        self.pending = 0
        jobs.progress(round(self.read / 1e9, 2), None, 'GB hashed', force=True)

    def close(self):
        self.commit()
        self.con.close()


def regular_files(root: Path, exclude: Iterable[str] = ()) -> dict[str, Path]:
    """relative path -> path of every regular file under ``root``, links neither followed nor counted (the store's
    ``tree_hash`` rule); ``exclude``: relative folders left out (a run's nested runs)."""
    skip = {Path(name) for name in exclude}
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not (here / d).is_symlink() and (here / d).relative_to(root)
                             not in skip)
        for name in filenames:
            path = here / name
            if not path.is_symlink() and path.is_file():
                found[str(path.relative_to(root))] = path
    return found


def _size(files: dict) -> int:
    return sum(path.stat().st_size for path in files.values())


# ---------------------------------------------------------------------------------------------------------------
# The comparison

class Archive:
    """What ``/backup/triforce`` holds: the SHA-1s of every file kept under ``models/`` (its ``files.tsv``) and the
    dated copies of the checkout's data folders (``data/<date>/<name>``)."""

    def __init__(self, root: Path):
        self.root = root
        self.models = set()
        listing = root / 'models' / 'files.tsv'
        if listing.is_file():
            with open(listing, encoding='utf-8') as file:
                self.models = {line.split('\t', 1)[0] for line in file if line.strip()}

    def data_copies(self, name: str) -> list[Path]:
        return sorted(path for path in (self.root / 'data').glob(f'*/{name}') if path.is_dir())


def content(folder: Path, hashes: Hashes, sizes: Optional[set] = None) -> set:
    """SHA-1s of ``folder``'s files (only those of a size in ``sizes`` when given: no other can match)."""
    return {hashes.sha1(path) for path in regular_files(folder).values()
            if sizes is None or path.stat().st_size in sizes}


def compare_rom(checkout: Path, hashes: Hashes, archive: Archive) -> dict:
    main = checkout / paths.CHECKOUT_ROM
    ours = hashes.sha1(main)
    held = paths.shared_root() / paths.STORE_ROM
    kept = archive.root / 'store' / paths.STORE_ROM
    backed = kept.is_file() and hashes.sha1(kept) == ours
    same = held.is_file() and hashes.sha1(held) == ours
    return {'area': 'roms', 'relative': 'roms', 'kind': 'rom', 'files': 1, 'bytes': main.stat().st_size,
            'store': f"rom/zelda.nes{'' if same else ' (other content or missing)'}",
            'status': IDENTICAL if same else LACKS,
            'lacking': [] if same else [{'path': paths.CHECKOUT_ROM, 'sha1': ours, 'in_backup': backed}],
            'backup': f"store/rom/zelda.nes {'identical' if backed else 'missing or other'}"}


def compare_data(checkout: Path, relative: str, kind: str, hashes: Hashes, archive: Archive) -> dict:
    """A data folder against every snapshot of its kind (module doc)."""
    files = regular_files(checkout / relative)
    main = {rel: hashes.sha1(path) for rel, path in files.items()}
    sizes = {rel: path.stat().st_size for rel, path in files.items()}
    snapshots = store.snapshots(kind)
    identical = None
    for row in snapshots:
        theirs = regular_files(paths.data_root(kind) / row['name'])
        if set(theirs) == set(main) and all(theirs[rel].stat().st_size == sizes[rel] for rel in main) and \
                all(hashes.sha1(theirs[rel]) == main[rel] for rel in main):
            identical = row['name']
            break
    lacking = []
    if identical is None:
        held = set()
        for row in snapshots:
            held |= content(paths.data_root(kind) / row['name'], hashes, set(sizes.values()))
        lacking = sorted(rel for rel, sha in main.items() if sha not in held)
    copies = archive.data_copies(Path(relative).name)
    same_copies = [copy for copy in copies if {rel: hashes.sha1(path) for rel, path in regular_files(copy).items()}
                   == main]
    backed = set()
    if lacking:
        for copy in copies:
            backed |= content(copy, hashes, {sizes[rel] for rel in lacking})
    kept = archive.root / 'store' / 'data' / kind / identical if identical else None
    current = paths.current_name(kind)
    return {'area': relative, 'relative': relative, 'kind': kind, 'files': len(main), 'bytes': sum(sizes.values()),
            'store': (f"snapshot {identical}" + (' (current)' if identical == current else f" (current: {current})"))
            if identical else f"no snapshot identical (current: {current})",
            'status': IDENTICAL if identical else (LACKS if lacking else MORE),
            'lacking': [{'path': rel, 'sha1': main[rel], 'in_backup': main[rel] in backed} for rel in lacking],
            'backup': '; '.join([*(f"{copy.relative_to(archive.root)} identical" for copy in same_copies),
                                 *([f"{kept.relative_to(archive.root)} (store copy)"] if kept and kept.is_dir()
                                   else [])]) or 'nothing identical'}


def compare_models(checkout: Path, hashes: Hashes, archive: Archive, only: Optional[list] = None) -> dict:
    """``models/`` against the store's runs, filed outputs and legacy copies, and its pointers against the current
    team (module doc); ``only``: the run dirs whose family starts with one of these (a piece of the job)."""
    from . import lead  # pylint: disable=import-outside-toplevel
    models = checkout / 'models'
    scan = lead.scan_models(models)
    by_run, records = defaultdict(list), defaultdict(list)
    for row in store.outputs():
        if row['run']:
            by_run[row['run']].append(row)
        if row['kind'] == 'record' and row['name']:
            records[row['name']].append(row)
    by_sha1 = {row['sha1']: f"{row['family']}/{row['name']}" for row in store.runs() if row['sha1']}
    root = store.root()
    runs, others, lacking = [], [], []
    files_seen = bytes_seen = archived = 0
    for entry in scan['runs']:
        if only and not entry['family'].startswith(tuple(only)):
            continue
        run = store.Run(entry['family'], entry['name'])
        files = regular_files(Path(entry['source']), entry['nested'])
        main = {rel: hashes.sha1(path) for rel, path in files.items()}
        files_seen += len(main)
        bytes_seen += _size(files)
        archived += sum(1 for sha in main.values() if sha in archive.models)
        result = {'run': run.identity, 'files': len(main), 'status': IDENTICAL, 'differ': [], 'lacking': []}
        if store.run_row(run) is None:
            result.update(status='not in the store', lacking=sorted(main),
                          note=by_sha1.get(main.get(store.CHECKPOINT)))
        else:
            theirs = regular_files(run.path)
            filed = defaultdict(set)
            for out in by_run[run.identity]:
                if (root / out['path']).is_file():
                    filed[Path(out['name'] or out['path']).name].add(hashes.sha1(root / out['path']))
            for rel, sha in sorted(main.items()):
                size = files[rel].stat().st_size
                if rel in entry['after']:             # filed by job, indexed by the run
                    same = sha in filed.get(rel, ())
                else:
                    same = rel in theirs and theirs[rel].stat().st_size == size and hashes.sha1(theirs[rel]) == sha
                if same:
                    continue
                elsewhere = any(sha in shas for shas in filed.values()) or \
                    sha in content(run.path, hashes, {size})
                (result['differ'] if elsewhere else result['lacking']).append(rel)
            if result['lacking']:
                result['status'] = LACKS
            elif result['differ']:
                result['status'] = 'same content at other paths'
        lacking += [{'path': f"{run.identity}/{rel}", 'sha1': main[rel], 'in_backup': main[rel] in archive.models}
                    for rel in result['lacking']]
        runs.append(result)
    if not only:
        for kind, items in (('pointer', scan['pointers']), ('record', scan['records']), ('legacy', scan['legacy'])):
            for rel in items:
                path = models / rel
                sha = hashes.sha1(path)
                files_seen += 1
                bytes_seen += path.stat().st_size
                archived += sha in archive.models
                if kind == 'legacy':
                    held = [p for p in sorted(root.glob(f'legacy/import-*/models/{rel}')) if p.is_file()]
                else:
                    held = [root / out['path'] for out in records.get(f'models/{rel}', ())
                            if (root / out['path']).is_file()]
                same = any(hashes.sha1(p) == sha for p in held)
                others.append({'kind': kind, 'path': rel, 'status': IDENTICAL if same else
                               ('changed since the import' if held else 'not in the store')})
                if not same:
                    lacking.append({'path': rel, 'sha1': sha, 'in_backup': sha in archive.models})
    statuses = {r['status'] for r in runs} | {o['status'] for o in others}
    kept = sum(1 for r in runs if (archive.root / 'store' / 'runs' / r['run']).is_dir())
    return {'area': 'models' + (f" ({', '.join(f'{p}*' for p in only)})" if only else ''), 'relative': 'models',
            'kind': 'models',
            'files': files_seen, 'bytes': bytes_seen,
            'store': f"{len(runs)} runs as runs/<family>/<run>, evaluations and act files filed by job, records "
                     f"filed, the rest in legacy/; team {paths.current_team()}",
            'status': LACKS if lacking else (IDENTICAL if statuses <= {IDENTICAL} else MORE),
            'runs': runs, 'others': others, 'links': [] if only else lead.find_links(models),
            'slots': None if only else _pointer_slots(models), 'lacking': lacking,
            'backup': f"{kept} of {len(runs)} runs in store/runs; {archived} of {files_seen} files' content in "
                      f"models/files.tsv"}


def _pointer_slots(models: Path) -> dict:
    """The checkout's pointers as team slots against the store's current team: per slot the checkout's run and act
    mode and the team's, and whether they agree."""
    from . import lead, registry  # pylint: disable=import-outside-toplevel
    try:
        team = store.read_team(paths.CURRENT)
    except paths.MissingError as error:
        return {'team': None, 'error': str(error)}
    mine = lead.pointer_slots(models)
    rows = {}
    for key in sorted(set(mine) | set(team['slots'])):
        ours = theirs = act = None
        if key in mine:
            family, name = mine[key]
            ours = f'{family}/{name}'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ckpt = models / family / name / store.CHECKPOINT
                act = registry.act_mode(ckpt) if ckpt.is_file() else None
        slot = team['slots'].get(key)
        if slot:
            theirs = slot['run']
        rows[key] = {'checkout': ours, 'checkout_act': act, 'team': theirs, 'team_act': slot and slot['act'],
                     'same': ours == theirs and (slot is None or act == slot['act'])}
    return {'team': team['name'], 'agree': all(r['same'] for r in rows.values()),
            'differ': {key: row for key, row in rows.items() if not row['same']}, 'slots': len(rows)}


def compare(checkout: Path, only: Optional[list] = None, out=print) -> dict:
    """The comparison of ``checkout``'s folders (``only``: areas, ``models`` or ``models:<family prefix>``)."""
    checkout = Path(checkout).resolve()
    store.require_store()
    hashes = Hashes(paths.scratch('reconcile', 'sha1.db'))
    archive = Archive(paths.BACKUP_DIR)
    report = {'checkout': str(checkout), 'store': str(store.root()), 'backup': str(archive.root),
              'made': time.time(), 'complete': not only, 'areas': []}
    try:
        for relative, kind in folders(checkout):
            if only and not any(o == relative or o.split(':')[0] == relative for o in only):
                continue
            out(f"comparing {relative} ...")
            if kind == 'rom':
                area = compare_rom(checkout, hashes, archive)
            elif kind == 'models':
                prefix = [o.split(':', 1)[1] for o in only or () if o.startswith('models:')]
                area = compare_models(checkout, hashes, archive, prefix)
            else:
                area = compare_data(checkout, relative, kind, hashes, archive)
            report['areas'].append(area)
            hashes.commit()
    finally:
        hashes.close()
    report['hashed_bytes'] = hashes.read
    return report


def table(report: dict) -> str:
    """The report as a Markdown table, then every file only the checkout holds."""
    lines = ['| folder | files | size | store item | status | only here (in backup) | backup holds |',
             '|---|---|---|---|---|---|---|']
    for area in report['areas']:
        lacking = area['lacking']
        backed = sum(1 for item in lacking if item['in_backup'])
        lines.append(f"| `{area['area']}` | {area['files']:,} | {area['bytes'] / 1e9:.2f} GB | {area['store']} | "
                     f"{area['status']} | {len(lacking)} ({backed}) | {area['backup']} |")
    for area in report['areas']:
        if area['lacking']:
            lines += ['', f"Only in `{area['area']}` ({len(area['lacking'])}):"]
            lines += [f"- `{item['path']}` {item['sha1'][:12]} {'(in backup)' if item['in_backup'] else '(NOT in backup)'}"
                      for item in area['lacking'][:TABLE_LIMIT]]
            if len(area['lacking']) > TABLE_LIMIT:
                lines.append(f"- ... {len(area['lacking']) - TABLE_LIMIT} more (the JSON report lists every one)")
        slots = area.get('slots')
        if slots:
            lines += ['', f"Pointers against team {slots.get('team')}: {slots.get('slots')} slots, "
                      f"{'all agree' if slots.get('agree') else 'differ: ' + json.dumps(slots.get('differ'))}"]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------------------------------
# Aside and back

def live_jobs(checkout: Path) -> list[str]:
    """The queue's live jobs (queued or running, their wrapper holding its lock) whose checkout or working directory
    is in ``checkout``, but this process's own job."""
    from . import jobqueue as q  # pylint: disable=import-outside-toplevel
    directory = q.queue_dir()
    con = q.connect(directory)
    try:
        rows = q.read_jobs(con, "status IN ('queued', 'running')")
    finally:
        con.close()
    own = jobs.job_id()
    found = []
    for row in rows:
        if str(row['id']) == str(own) or q.lock_free(q.lock_path(directory, f"job-{row['id']}")):
            continue
        places = [Path(p) for p in (row.get('checkout'), row.get('cwd')) if p]
        if any(place == checkout or checkout in place.parents for place in places):
            found.append(f"#{row['id']} {row.get('name')} ({row['status']})")
    return found


def open_files(targets: list[Path]) -> list[str]:
    """Processes (but this one) with a file or working directory inside one of ``targets``."""
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            links = [proc / 'cwd', *(proc / 'fd').iterdir()]
        except OSError:
            continue
        for link in links:
            try:
                target = Path(os.readlink(link))
            except OSError:
                continue
            if any(target == t or t in target.parents for t in targets):
                try:
                    name = (proc / 'comm').read_text(encoding='utf-8').strip()
                except OSError:
                    name = '?'
                found.append(f"pid {proc.name} ({name}): {target}")
                break
    return found


def store_ready() -> list[str]:
    """What the store lacks for a checkout without folders: its ROM, a current team, a current snapshot per kind."""
    problems = []
    if not (paths.shared_root() / paths.STORE_ROM).is_file():
        problems.append(f"no ROM at {paths.shared_root() / paths.STORE_ROM}")
    if paths.current_team() is None:
        problems.append(f"no current team ({paths.teams_dir() / paths.CURRENT})")
    for kind in paths.DATA_KINDS:
        try:
            paths.data(kind)
        except paths.MissingError as error:
            problems.append(str(error))
    return problems


def _newest(folder: Path) -> tuple[float, Optional[Path]]:
    newest, which = folder.lstat().st_mtime, folder
    for dirpath, dirnames, filenames in os.walk(folder):
        here = Path(dirpath)
        for name in dirnames + filenames:
            mtime = (here / name).lstat().st_mtime
            if mtime > newest:
                newest, which = mtime, here / name
    return newest, which


def aside_checks(checkout: Path, report: dict, accept_unbacked: bool = False) -> list[str]:
    """Every reason not to move ``checkout``'s folders aside now (module doc), empty when there is none."""
    problems = []
    present = [rel for rel, _ in folders(checkout)]
    if Path(report['checkout']) != checkout:
        problems.append(f"the report is of {report['checkout']}, not {checkout}")
    covered = {area['relative'] for area in report['areas']}
    if not report.get('complete') or set(present) - covered:
        problems.append(f"the report does not cover {sorted(set(present) - covered) or 'everything'}: run "
                        f"`python -m triforce.reconcile compare` without --only")
    for rel in present:
        newest, which = _newest(checkout / rel)
        if newest > report['made']:
            problems.append(f"{which} changed after the comparison ({time.ctime(newest)}): compare again")
    unbacked = [f"{area['area']}: {item['path']}" for area in report['areas'] for item in area['lacking']
                if not item['in_backup']]
    if unbacked and not accept_unbacked:
        problems.append(f"{len(unbacked)} files only the checkout holds are in no backup (back them up, compare "
                        f"again, or --accept-unbacked): {', '.join(unbacked[:5])}{' ...' if len(unbacked) > 5 else ''}")
    problems += store_ready()
    problems += [f"a live job works in the checkout: {line}" for line in live_jobs(checkout)]
    problems += [f"a process holds a file of the folders: {line}" for line in
                 open_files([checkout / rel for rel in present])]
    return problems


def default_aside(checkout: Path) -> Path:
    return checkout.parent / f"{checkout.name}.aside-{time.strftime('%y%m%d')}"


def aside(checkout: Path, report_path: Path, into: Optional[Path] = None, accept_unbacked: bool = False,
          apply: bool = False, out=print) -> int:
    """Moves the checkout's folders into ``into`` (module doc); 0 when done (or, without ``apply``, possible)."""
    checkout = Path(checkout).resolve()
    into = Path(into or default_aside(checkout))
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    problems = aside_checks(checkout, report, accept_unbacked)
    if into.exists():
        problems.append(f"{into} exists")
    elif into.parent.stat().st_dev != checkout.stat().st_dev:
        problems.append(f"{into.parent} is on another disk than {checkout}: a move would copy")
    present = [rel for rel, _ in folders(checkout)]
    for rel in present:
        out(f"  {'move' if apply and not problems else 'would move'} {checkout / rel} -> {into / rel}")
    for problem in problems:
        out(f"  REFUSED: {problem}")
    if problems or not apply:
        return 1 if problems else 0
    into.mkdir()
    moved = []
    try:
        for rel in present:
            (into / rel).parent.mkdir(parents=True, exist_ok=True)
            os.rename(checkout / rel, into / rel)
            moved.append(rel)
    finally:
        (into / ASIDE_FILE).write_text(json.dumps({'checkout': str(checkout), 'moved': moved, 'when': time.time(),
                                                   'report': str(Path(report_path).resolve())}, indent=2) + '\n',
                                       encoding='utf-8')
    out(f"  moved {len(moved)} folders aside into {into} ({ASIDE_FILE}); put back with "
        f"`python -m triforce.reconcile back {into} --apply`")
    return 0


def back(into: Path, apply: bool = False, out=print) -> int:
    """Puts every folder ``aside`` moved into ``into`` back where it was; 0 when done (or possible)."""
    into = Path(into).resolve()
    record = json.loads((into / ASIDE_FILE).read_text(encoding='utf-8'))
    checkout = Path(record['checkout'])
    taken = [rel for rel in record['moved'] if (checkout / rel).exists() or (checkout / rel).is_symlink()]
    for rel in record['moved']:
        out(f"  {'put back' if apply and not taken else 'would put back'} {into / rel} -> {checkout / rel}")
    for rel in taken:
        out(f"  REFUSED: {checkout / rel} exists (moved there since; move it out of the way first)")
    if taken or not apply:
        return 1 if taken else 0
    for rel in record['moved']:
        os.rename(into / rel, checkout / rel)
    (into / ASIDE_FILE).unlink()
    for dirpath, _, _ in sorted(os.walk(into), key=lambda item: -len(item[0])):
        Path(dirpath).rmdir()
    out(f"  put back {len(record['moved'])} folders into {checkout}; {into} removed")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m triforce.reconcile', description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('compare', help="what the checkout's folders hold that the store lacks (a queue job)")
    c.add_argument('--checkout', type=Path, default=paths.checkout())
    c.add_argument('--only', nargs='+', default=None,
                   help="areas: roms, trajectories, trajectories.prev-rNN, triforce/states/harvested, "
                        "triforce/states/harvested-rooms, target/demos, models, models:<family prefix>")
    c.add_argument('--out', type=Path, default=None, help="the JSON report (default target/reconcile/report-*.json)")
    a = sub.add_parser('aside', help="move the checkout's folders into <checkout>.aside-<yymmdd>")
    a.add_argument('--checkout', type=Path, default=paths.checkout())
    a.add_argument('--report', type=Path, required=True, help="compare's JSON report (complete, newer than the files)")
    a.add_argument('--into', type=Path, default=None)
    a.add_argument('--accept-unbacked', action='store_true',
                   help="move even when files only the checkout holds are in no backup")
    a.add_argument('--apply', action='store_true')
    b = sub.add_parser('back', help="put the folders an aside moved back")
    b.add_argument('aside_dir', type=Path)
    b.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'compare':
        jobs.require_job()
        report = compare(args.checkout, args.only)
        target = args.out or paths.scratch('reconcile', f"report-{time.strftime('%y%m%d-%H%M%S')}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
        print(table(report))
        print(f"\nreport: {target} ({report['hashed_bytes'] / 1e9:.1f} GB hashed now)")
        return 0
    if args.command == 'aside':
        return aside(args.checkout, args.report, args.into, args.accept_unbacked, args.apply)
    return back(args.aside_dir, args.apply)


if __name__ == '__main__':
    sys.exit(main())
