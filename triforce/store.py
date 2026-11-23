"""The shared store (``docs/data-layout.md``): the one program that writes ``paths.shared_root()``.

Everything shared lives outside every checkout, under ``~/triforce-shared`` (``$TRIFORCE_SHARED``), and is written
only through this module (the public operations below, which our tools call) and ``triforce.lead`` (the lead's).
Built like the job queue (``triforce.jobqueue``): files, one kernel lock (``locks/store.lock``) around every write,
one SQLite index (``store.db``, WAL), no server.

::

    README.md, layout.json   the rules for a person; every location for tools that are not Python
    store.db, locks/         the index (snapshots, runs, teams, jobs, filed outputs, summaries) and the audit log
    rom/zelda.nes            read-only
    data/<kind>/<name>/      named snapshots, read-only; data/<kind>/current names the default
    runs/<family>/<run>/     network runs: open while their training writes, then sealed (read-only)
    teams/<name>.json        immutable teams; teams/current names the team of record
    jobs/<job>-<label>/      what a job filed (evaluations, chain records, statistics, replay packs, harvests)
    legacy/<import>/         what the import found no home for (listed in its report)

Rules (the owner's): shared things are never modified in place, only added (installed, sealed, named), and the
read-only ones are read-only on disk so an accident fails; missing is an error naming where it looked; every write
is a row of the audit log (when, job, agent, entry point, what).

Public operations (``entry='public'``): ``allocate_run`` / ``seal_run`` / ``resume_run`` / ``derive_run`` (the
trainings and ``select-checkpoint --apply``), ``job_folder`` / ``file_output`` (every tool's outputs, one kept summary
row each), and the read side (``runs``, ``teams``, ``snapshots``, ``read_team``, ``effective_team``, ``team_view``,
``write_view``).  The lead's (``triforce.lead``, ``entry='lead'``): install a snapshot, set a kind's current one,
make a team, set the current team, the import, delete, back up.
"""

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from . import jobs, paths

DB_NAME = 'store.db'
LOCK_NAME = 'store.lock'
TEAM_ENV = 'TRIFORCE_TEAM'          # the team a tool plays (``--team``), handed to its child processes
PUBLIC, LEAD = 'public', 'lead'
OPEN, SEALED = 'open', 'sealed'
CHECKPOINT = 'ckpt.pt'
# What a training writes after the fact into a run dir and the store files by job instead (rule 3): evaluations,
# the checkpoint choice, the act mode (the run's sealed default and a team's slot).
AFTER_TRAINING = ('act', 'selection.json')
SUMMARY_FIELDS = ('all.hearts_lost.mean', 'beams.rooms_with_beams.mean', 'beams.held_frames.mean',
                  'all.shots_hit.sum', 'all.kills_far.sum', 'all.frames.mean', 'endings', 'triforces', 'flags')

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, job TEXT, agent TEXT, entry TEXT,
    op TEXT, what TEXT, user TEXT, pid INTEGER, checkout TEXT);
CREATE TABLE IF NOT EXISTS snapshots (kind TEXT, name TEXT, created REAL, files INTEGER, bytes INTEGER,
    sha256 TEXT, source TEXT, linked INTEGER, note TEXT, PRIMARY KEY (kind, name));
CREATE TABLE IF NOT EXISTS currents (kind TEXT PRIMARY KEY, name TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS runs (family TEXT, name TEXT, status TEXT, created REAL, sealed REAL, owner_job TEXT,
    owner_agent TEXT, sha1 TEXT, act TEXT, parent TEXT, parent_sha1 TEXT, origin TEXT, team TEXT, data TEXT,
    note TEXT, hit_head INTEGER, PRIMARY KEY (family, name));
CREATE INDEX IF NOT EXISTS runs_sha1 ON runs (sha1);
CREATE TABLE IF NOT EXISTS teams (name TEXT PRIMARY KEY, created REAL, sha256 TEXT, source TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS jobs (job TEXT PRIMARY KEY, label TEXT, agent TEXT, created REAL, folder TEXT);
CREATE TABLE IF NOT EXISTS outputs (id INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT, kind TEXT, path TEXT,
    name TEXT, run TEXT, team TEXT, data TEXT, created REAL);
CREATE INDEX IF NOT EXISTS outputs_run ON outputs (run);
CREATE TABLE IF NOT EXISTS summaries (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, name TEXT,
    job TEXT, run TEXT, team TEXT, data TEXT, n INTEGER, successes INTEGER, rate REAL, se REAL, compact TEXT,
    output INTEGER);
CREATE INDEX IF NOT EXISTS summaries_run ON summaries (run);
CREATE INDEX IF NOT EXISTS summaries_team ON summaries (team);
"""


class StoreError(RuntimeError):
    """A store operation refused (a sealed run written, a name taken, a team that does not resolve, ...)."""


class MissingError(paths.MissingError):
    """A run, team or snapshot the store does not have (rule 4); the message names where it looked."""


# ---------------------------------------------------------------------------------------------------------------
# The index, the lock and the audit log

def root() -> Path:
    """The store's root (``paths.shared_root``)."""
    return paths.shared_root()


def require_store() -> Path:
    """The store's root when it has been made; ``MissingError`` otherwise."""
    if not paths.store_exists():
        raise MissingError(f"no shared store at {root()} (no {DB_NAME}; the lead makes it: `triforce store paths` "
                           f"prints where it is looked for)")
    return root()


def connect(create: bool = False) -> sqlite3.Connection:
    """The index (``store.db``); ``create`` makes the root and the schema (the lead's ``init``)."""
    base = root()
    if not create:
        require_store()
    (base / 'locks').mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(base / DB_NAME, timeout=60, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA busy_timeout=60000')
    con.executescript(SCHEMA)
    return con


def who() -> dict:
    """Who writes: the job (``$TRIFORCE_JOB_ID``) and its agent (the queue's row), the user, the process, the
    checkout."""
    job = jobs.job_id()
    agent = None
    if job is not None:
        try:
            from . import jobqueue  # pylint: disable=import-outside-toplevel
            con = jobqueue.connect()
            try:
                row = con.execute('SELECT agent FROM jobs WHERE id = ?', (job,)).fetchone()
            finally:
                con.close()
            agent = row['agent'] if row else None
        except (OSError, sqlite3.Error):
            agent = None
    return {'job': None if job is None else str(job), 'agent': agent or os.environ.get('TRIFORCE_AGENT'),
            'user': os.environ.get('USER'), 'pid': os.getpid(), 'checkout': str(paths.checkout())}


@contextlib.contextmanager
def writing(op: str, what: dict, entry: str = PUBLIC, create: bool = False):
    """One write of the store: the kernel lock, one ``BEGIN IMMEDIATE`` transaction and its audit row (``op``,
    ``what``, the entry point).  Yields the connection; an exception rolls the index back (files already made are
    the caller's to remove)."""
    con = connect(create=create)
    fd = os.open(root() / 'locks' / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        con.execute('BEGIN IMMEDIATE')
        try:
            yield con
            caller = who()
            con.execute('INSERT INTO audit (ts, job, agent, entry, op, what, user, pid, checkout) '
                        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (time.time(), caller['job'], caller['agent'], entry, op, json.dumps(what, sort_keys=True),
                         caller['user'], caller['pid'], caller['checkout']))
        except BaseException:
            con.execute('ROLLBACK')
            raise
        con.execute('COMMIT')
    finally:
        os.close(fd)
        con.close()


def audit_log(since: Optional[float] = None, job: Optional[str] = None, limit: Optional[int] = None) -> list[dict]:
    """The audit rows (oldest first), optionally since a time or of one job."""
    con = connect()
    try:
        sql, args = 'SELECT * FROM audit WHERE 1', []
        if since is not None:
            sql, args = sql + ' AND ts >= ?', [*args, since]
        if job is not None:
            sql, args = sql + ' AND job = ?', [*args, str(job)]
        rows = [dict(row) for row in con.execute(sql + ' ORDER BY id', args)]
    finally:
        con.close()
    for row in rows:
        row['what'] = json.loads(row['what'])
    return rows[-limit:] if limit else rows


# ---------------------------------------------------------------------------------------------------------------
# Files: read-only trees, hard-link copies, content hashes

def make_readonly(path: Path):
    """Files 0444, folders 0555 (deepest first), symlinks untouched: an accidental write fails."""
    path = Path(path)
    if path.is_symlink():
        return
    if path.is_file():
        path.chmod(0o444)
        return
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        for name in filenames:
            full = Path(dirpath) / name
            if not full.is_symlink():
                full.chmod(0o444)
        for name in dirnames:
            full = Path(dirpath) / name
            if not full.is_symlink():
                full.chmod(0o555)
    path.chmod(0o555)


def make_writable(path: Path):
    """Folders (and files) writable by their owner again: only for the lead's delete."""
    path = Path(path)
    if path.is_symlink() or not path.exists():
        return
    path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
    if path.is_dir():
        for dirpath, dirnames, filenames in os.walk(path):
            for name in dirnames + filenames:
                full = Path(dirpath) / name
                if not full.is_symlink():
                    full.chmod(stat.S_IMODE(full.stat().st_mode) | stat.S_IWUSR)


def link_tree(source: Path, target: Path, link: bool = True) -> tuple[int, int]:
    """Copies ``source`` (a file or a folder) to ``target`` (which must not exist): each file a hard link where
    ``link`` and the filesystem allow, else a copy; symlinks inside are copied as the files they point to (a
    snapshot holds content, not pointers).  Returns (files, bytes)."""
    source, target = Path(source), Path(target)
    if target.exists() or target.is_symlink():
        raise StoreError(f"{target} exists")
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        _link_file(source, target, link)
        return 1, target.stat().st_size
    files = size = 0
    for dirpath, dirnames, filenames in os.walk(source, followlinks=True):
        relative = Path(dirpath).relative_to(source)
        (target / relative).mkdir(parents=True, exist_ok=True)
        dirnames.sort()
        for name in sorted(filenames):
            src = Path(dirpath) / name
            if not src.exists():           # a dangling link: nothing to keep
                continue
            _link_file(src, target / relative / name, link)
            files += 1
            size += src.stat().st_size
    return files, size


def _link_file(source: Path, target: Path, link: bool):
    real = source.resolve()
    if link:
        try:
            os.link(real, target)
            return
        except OSError:
            pass
    shutil.copy2(real, target)


def tree_hash(path: Path) -> tuple[str, int, int]:
    """(sha256, files, bytes) of a folder's content: every file's relative path and bytes, in sorted order (the
    snapshot's identity, recorded once at install so no reader re-hashes it).  Symlinks count as the files they
    point to and dangling ones not at all, as ``link_tree`` copies them."""
    digest = hashlib.sha256()
    files = size = 0
    path = Path(path)
    for dirpath, dirnames, filenames in os.walk(path, followlinks=True):
        dirnames.sort()
        for name in sorted(filenames):
            full = Path(dirpath) / name
            if not full.exists():
                continue
            digest.update(str(full.relative_to(path)).encode() + b'\0')
            with open(full, 'rb') as file:
                for block in iter(lambda: file.read(1 << 20), b''):
                    digest.update(block)
            digest.update(b'\0')
            files += 1
            size += full.stat().st_size
    return digest.hexdigest(), files, size


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_text(path: Path, text: str):
    """``text`` into ``path`` through a temp file and a rename (a reader sees the old or the new one)."""
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def _relative(path: Path) -> str:
    return str(Path(path).relative_to(root()))


# ---------------------------------------------------------------------------------------------------------------
# Making the store, README and layout.json

README = Path(__file__).resolve().parent / 'store_readme.md'


def init(entry: str = LEAD) -> Path:
    """Makes the store's root, its index, README and layout (idempotent)."""
    base = root()
    base.mkdir(parents=True, exist_ok=True)
    with writing('init', {'root': str(base)}, entry, create=True):
        for name in ('rom', 'data', 'runs', 'teams', 'jobs', 'legacy'):
            (base / name).mkdir(exist_ok=True)
        for kind in paths.DATA_KINDS:
            (base / 'data' / kind).mkdir(exist_ok=True)
        write_layout()
        readme = base / 'README.md'
        if readme.exists():
            make_writable(readme)
        shutil.copyfile(README, readme)
        readme.chmod(0o444)
    return base


def write_layout():
    """``layout.json``: ``paths.layout()`` (written under the store's lock by every operation that changes a
    ``current``)."""
    _write_text(root() / 'layout.json', paths.layout_json())


# ---------------------------------------------------------------------------------------------------------------
# Data snapshots

def snapshots(kind: Optional[str] = None) -> list[dict]:
    """Installed snapshots (kind, name, files, bytes, sha256, source, current), by kind then name."""
    con = connect()
    try:
        rows = [dict(row) for row in con.execute(
            'SELECT * FROM snapshots' + (' WHERE kind = ?' if kind else '') + ' ORDER BY kind, name',
            (kind,) if kind else ())]
        currents = {row['kind']: row['name'] for row in con.execute('SELECT kind, name FROM currents')}
    finally:
        con.close()
    for row in rows:
        row['current'] = currents.get(row['kind']) == row['name']
    return rows


def snapshot(kind: str, name: str = paths.CURRENT) -> dict:
    """One snapshot's row, with its ``path`` (``current``: the kind's default); ``MissingError`` when absent."""
    path = paths.data(kind, name)
    con = connect()
    try:
        row = con.execute('SELECT * FROM snapshots WHERE kind = ? AND name = ?', (kind, path.name)).fetchone()
    finally:
        con.close()
    if row is None:
        raise MissingError(f"{kind} snapshot {path.name!r} has no row in {root() / DB_NAME}")
    return {**dict(row), 'path': str(path)}


def install_snapshot(kind: str, name: str, source: Path, link: bool = True, note: Optional[str] = None,
                     known: Optional[tuple[str, int, int]] = None, entry: str = LEAD) -> dict:
    """Installs ``source`` (a folder) as ``data/<kind>/<name>/``: hard links (``link``) or copies, read-only, its
    count, bytes and content hash recorded (``known``: the ``tree_hash`` of ``source`` already taken, used as is
    for hard links, which are the same files).  The lead's (``triforce-lead data install``)."""
    paths.data_root(kind)
    if not name or '/' in name or name == paths.CURRENT or name.startswith('.'):
        raise StoreError(f"bad snapshot name {name!r}")
    source = Path(source).resolve()
    if not source.is_dir():
        raise MissingError(f"{source} is not a folder")
    target = paths.data_root(kind) / name
    with writing('data-install', {'kind': kind, 'name': name, 'source': str(source), 'link': link}, entry) as con:
        if con.execute('SELECT 1 FROM snapshots WHERE kind = ? AND name = ?', (kind, name)).fetchone() or \
                target.exists():
            raise StoreError(f"{kind} snapshot {name!r} exists ({target}); a snapshot is never changed")
        try:
            link_tree(source, target, link)
            sha256, files, size = known if known is not None and link else tree_hash(target)
            make_readonly(target)
        except BaseException:
            if target.exists():
                make_writable(target)
                shutil.rmtree(target, ignore_errors=True)
            raise
        con.execute('INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (kind, name, time.time(), files, size, sha256, str(source), int(link), note))
    return {'kind': kind, 'name': name, 'files': files, 'bytes': size, 'sha256': sha256, 'path': str(target)}


def set_current_snapshot(kind: str, name: str, entry: str = LEAD) -> Path:
    """Makes snapshot ``name`` the kind's ``current`` (``data/<kind>/current``); the lead's."""
    path = paths.data(kind, name)
    with writing('data-set-current', {'kind': kind, 'name': name}, entry) as con:
        con.execute('INSERT OR REPLACE INTO currents VALUES (?, ?, ?)', (f'data:{kind}', name, time.time()))
        _write_text(paths.data_root(kind) / paths.CURRENT, name + '\n')
        write_layout()
    return path


# ---------------------------------------------------------------------------------------------------------------
# Runs

@dataclass
class Run:
    """A network run of the store: ``family`` (a family, or a group's path such as ``r38head-hit``), ``name``."""
    family: str
    name: str

    @property
    def identity(self) -> str:
        return f'{self.family}/{self.name}'

    @property
    def path(self) -> Path:
        return paths.run_dir(self.family, self.name)


def parse_run(text: str, family: Optional[str] = None) -> Run:
    """``family/run`` (a group's family may hold slashes: everything before the last one), or ``run`` with
    ``family``."""
    text = text.strip().strip('/')
    if '/' in text:
        head, _, tail = text.rpartition('/')
        return Run(head, tail)
    if family is None:
        raise StoreError(f"{text!r}: name a run as FAMILY/RUN")
    return Run(family, text)


def run_row(run: Run) -> Optional[dict]:
    """The index row of ``run``, None when the store has no such run."""
    con = connect()
    try:
        row = con.execute('SELECT * FROM runs WHERE family = ? AND name = ?', (run.family, run.name)).fetchone()
    finally:
        con.close()
    return None if row is None else dict(row)


def get_run(run: Run) -> dict:
    """``run_row`` with its ``path``; ``MissingError`` naming the index and the folder when absent."""
    row = run_row(run)
    if row is None:
        raise MissingError(f"no run {run.identity} in the store ({root() / DB_NAME}; folder {run.path})")
    return {**row, 'path': str(run.path)}


def runs(family: Optional[str] = None, status: Optional[str] = None) -> list[dict]:
    """The runs (family, name, status, sha1, act, parent, owner), by family then name."""
    con = connect()
    try:
        sql, args = 'SELECT * FROM runs WHERE 1', []
        if family:
            sql, args = sql + ' AND family = ?', [*args, family]
        if status:
            sql, args = sql + ' AND status = ?', [*args, status]
        return [dict(row) for row in con.execute(sql + ' ORDER BY family, name', args)]
    finally:
        con.close()


def _new_name(family: str, wanted: Optional[str], con) -> str:
    if wanted:
        if '/' in wanted or wanted.startswith('.') or wanted in (paths.CURRENT,):
            raise StoreError(f"bad run name {wanted!r}")
        if con.execute('SELECT 1 FROM runs WHERE family = ? AND name = ?', (family, wanted)).fetchone() or \
                paths.run_dir(family, wanted).exists():
            raise StoreError(f"run {family}/{wanted} exists in the store; a run is never written twice (choose "
                             f"another --run-id, or continue it: --load {family}/{wanted} makes a new run)")
        return wanted
    stamp = time.strftime('%Y%m%d-%H%M%S')
    for n in range(1000):
        name = f'{stamp}-{os.getpid() % 10000:04d}' + (f'-{n}' if n else '')
        if not con.execute('SELECT 1 FROM runs WHERE family = ? AND name = ?', (family, name)).fetchone() and \
                not paths.run_dir(family, name).exists():
            return name
    raise StoreError(f"no free run name in {family} for {stamp}")


def allocate_run(family: str, name: Optional[str] = None, parent: Optional[Run] = None, origin: str = 'train',
                 team: Optional[dict] = None, data: Optional[list] = None, entry: str = PUBLIC) -> Run:
    """A new open run ``runs/<family>/<name>/`` (``name`` unique in the family, else a sortable new one), owned by
    this job (``$TRIFORCE_JOB_ID``) and its agent; ``parent`` (a run this one starts from), ``team`` (the effective
    team it was trained against) and ``data`` (the snapshots it read) are recorded.  Its trainer writes it, then
    ``seal_run``."""
    parent_row = get_run(parent) if parent is not None else None
    caller = who()
    what = {'family': family, 'name': name, 'parent': parent and parent.identity, 'origin': origin}
    with writing('run-allocate', what, entry) as con:
        name = what['name'] = _new_name(family, name, con)
        run = Run(family, name)
        run.path.mkdir(parents=True)
        con.execute('INSERT INTO runs (family, name, status, created, owner_job, owner_agent, parent, parent_sha1, '
                    'origin, team, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (family, name, OPEN, time.time(), caller['job'], caller['agent'],
                     parent and parent.identity, parent_row and parent_row['sha1'], origin,
                     None if team is None else json.dumps(team_record(team)),
                     None if data is None else json.dumps(data)))
    return run


def resume_run(run: Run, entry: str = PUBLIC) -> Run:
    """Takes an open run (a training that was interrupted) for a new job of its owner (the same agent, or no agent
    known on either side); a sealed run is refused."""
    row = get_run(run)
    if row['status'] != OPEN:
        raise StoreError(f"run {run.identity} is sealed (read-only since {time.ctime(row['sealed'])}); continue it "
                         f"into a new run: --load {run.identity}")
    caller = who()
    if row['owner_agent'] and caller['agent'] and row['owner_agent'] != caller['agent']:
        raise StoreError(f"run {run.identity} is open for {row['owner_agent']} (job {row['owner_job']}), not "
                         f"{caller['agent']}")
    with writing('run-resume', {'run': run.identity, 'from_job': row['owner_job']}, entry) as con:
        con.execute('UPDATE runs SET owner_job = ? WHERE family = ? AND name = ?', (caller['job'], run.family,
                                                                                    run.name))
    return run


def default_act(run_path: Path) -> str:
    """The act mode a run plays by: its ``act`` file when written for its current ``ckpt.pt``, else the project's
    (``registry.act_mode``)."""
    from . import registry  # pylint: disable=import-outside-toplevel
    return registry.act_mode(Path(run_path) / CHECKPOINT)


def seal_run(run: Run, act: Optional[str] = None, entry: str = PUBLIC) -> dict:
    """Seals an open run: its checkpoint's SHA-1 and its default act mode (``act``, else ``default_act``) recorded,
    the files the store files elsewhere (``AFTER_TRAINING``) moved out of it into this job's folder, the folder
    read-only.  Never written again.  A run without a checkpoint is refused (it stays open)."""
    row = get_run(run)
    if row['status'] != OPEN:
        raise StoreError(f"run {run.identity} is already sealed")
    ckpt = run.path / CHECKPOINT
    if not ckpt.is_file():
        raise StoreError(f"run {run.identity} has no {CHECKPOINT} ({ckpt}); it stays open")
    mode = act or default_act(run.path)
    sha1 = file_sha1(ckpt)
    moved = _move_after_training(run)
    with writing('run-seal', {'run': run.identity, 'sha1': sha1, 'act': mode, 'filed': moved}, entry) as con:
        make_readonly(run.path)
        con.execute('UPDATE runs SET status = ?, sealed = ?, sha1 = ?, act = ? WHERE family = ? AND name = ?',
                    (SEALED, time.time(), sha1, mode, run.family, run.name))
    return {**get_run(run), 'filed': moved}


def _move_after_training(run: Run) -> list[str]:
    """Files an open run's evaluations, ``act`` and ``selection.json`` (rule 3) into this job's folder before it is
    sealed; returns their names."""
    moved = []
    for path in sorted(run.path.iterdir()):
        if path.is_file() and (path.name in AFTER_TRAINING or (path.name.startswith('eval') and
                                                                 path.suffix == '.json')):
            kind = 'evaluate' if path.name.startswith('eval') else path.name.split('.')[0]
            file_output(path, kind, run=run.identity, name=f'{run.family}/{run.name}/{path.name}', move=True)
            moved.append(path.name)
    return moved


def derive_run(parent: Run, checkpoint: Path, origin: str, family: Optional[str] = None,
               name: Optional[str] = None, act: Optional[str] = None, keep: Iterable[str] = ('run.json', 'version'),
               entry: str = PUBLIC) -> dict:
    """A new sealed run from ``checkpoint`` (a parent's ``ckpt.pt`` or one of its numbered checkpoints, or a
    converted file): ``select-checkpoint --apply`` (``origin='selection'``), ``convert-model`` (``'conversion'``).
    The parent's files named in ``keep`` come along (hard links); the parent is recorded."""
    get_run(parent)
    run = allocate_run(family or parent.family, name or f'{parent.name}-{origin[:3]}-{time.strftime("%H%M%S")}',
                       parent=parent, origin=origin, entry=entry)
    try:
        for item in keep:
            source = parent.path / item
            if source.exists():
                link_tree(source, run.path / item)
        link_tree(Path(checkpoint), run.path / CHECKPOINT)
        stamp = Path(checkpoint).parent / 'version'
        if stamp.is_file() and 'version' in keep:
            (run.path / 'version').unlink(missing_ok=True)
            link_tree(stamp, run.path / 'version')
    except BaseException:
        _drop_open(run)
        raise
    return seal_run(run, act=act, entry=entry)


def _drop_open(run: Run):
    """Removes an open run that failed while it was being made (its folder and its row)."""
    if run.path.exists():
        make_writable(run.path)
        shutil.rmtree(run.path, ignore_errors=True)
    with writing('run-drop', {'run': run.identity}, PUBLIC) as con:
        con.execute('DELETE FROM runs WHERE family = ? AND name = ? AND status = ?', (run.family, run.name, OPEN))


def install_run(run: Run, source: Path, act: str, exclude: Iterable[str] = (), link: bool = True,
                origin: str = 'import', note: Optional[str] = None, entry: str = LEAD) -> dict:
    """Installs a finished run dir from outside the store as a sealed run (the lead's import): its files hard-linked
    (``link``) or copied, except ``exclude`` (names relative to ``source``: the evaluations, ``act`` and
    ``selection.json`` the import files by job, nested runs), read-only; checkpoint SHA-1 and act mode recorded."""
    source = Path(source)
    excluded = {Path(name) for name in exclude}
    ckpt = source / CHECKPOINT
    if not ckpt.is_file():
        raise MissingError(f"{source} has no {CHECKPOINT}")
    sha1 = file_sha1(ckpt)
    with writing('run-install', {'run': run.identity, 'source': str(source), 'sha1': sha1, 'act': act,
                                 'origin': origin}, entry) as con:
        if con.execute('SELECT 1 FROM runs WHERE family = ? AND name = ?', (run.family, run.name)).fetchone() or \
                run.path.exists():
            raise StoreError(f"run {run.identity} exists in the store")
        try:
            run.path.mkdir(parents=True)
            for entry_path in sorted(source.iterdir()):
                relative = entry_path.relative_to(source)
                if relative in excluded:
                    continue
                if entry_path.is_dir() and not entry_path.is_symlink():
                    for dirpath, dirnames, filenames in os.walk(entry_path):
                        here = Path(dirpath)
                        dirnames[:] = sorted(d for d in dirnames if (here / d).relative_to(source) not in excluded)
                        (run.path / here.relative_to(source)).mkdir(parents=True, exist_ok=True)
                        for name in sorted(filenames):
                            if (here / name).exists():
                                _link_file(here / name, run.path / here.relative_to(source) / name, link)
                elif entry_path.exists():
                    _link_file(entry_path, run.path / relative, link)
            make_readonly(run.path)
        except BaseException:
            make_writable(run.path)
            shutil.rmtree(run.path, ignore_errors=True)
            raise
        con.execute('INSERT INTO runs (family, name, status, created, sealed, sha1, act, origin, note) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (run.family, run.name, SEALED, ckpt.stat().st_mtime, time.time(), sha1, act, origin,
                     note or str(source)))
    return get_run(run)


def run_of_path(path) -> Optional[Run]:
    """The store run a path lies in (a run folder, its ``ckpt.pt``, a numbered checkpoint), None outside
    ``runs/``."""
    try:
        relative = Path(path).resolve().relative_to(paths.runs_dir().resolve())
    except ValueError:
        return None
    parts = list(relative.parts)
    if parts and parts[-1] == CHECKPOINT:
        parts = parts[:-1]
    if 'checkpoints' in parts:
        parts = parts[:parts.index('checkpoints')]
    for cut in range(len(parts), 1, -1):
        candidate = Run('/'.join(parts[:cut - 1]), parts[cut - 1])
        if paths.store_exists() and run_row(candidate) is not None:
            return candidate
    return None


# ---------------------------------------------------------------------------------------------------------------
# Teams

SLOT_LATE = ':late'


def slot_key(family: str, route: Optional[str] = None, late: bool = False) -> str:
    """A team slot: ``family`` (today's ``promoted``), ``family@route`` (``promoted.<route>``), ``family@route:late``
    (``late.<route>``)."""
    if late and route is None:
        raise StoreError(f"{family}: a late slot names its route (family@route:late)")
    return family + (f'@{route}' if route else '') + (SLOT_LATE if late else '')


def split_slot(key: str) -> tuple[str, Optional[str], bool]:
    """(family, route, late) of a slot key."""
    late = key.endswith(SLOT_LATE)
    base = key[:-len(SLOT_LATE)] if late else key
    family, sep, route = base.partition('@')
    if not family or (sep and not route) or (late and not route):
        raise StoreError(f"bad slot {key!r}: family, family@route or family@route:late")
    return family, (route or None), late


def pointer_name(key: str) -> str:
    """The legacy pointer file a slot is: ``promoted``, ``promoted.<route>``, ``late.<route>``."""
    _, route, late = split_slot(key)
    return ('late' if late else 'promoted') + (f'.{route}' if route else '')


def _hit_head(ckpt: Path) -> bool:
    import torch  # pylint: disable=import-outside-toplevel
    saved = torch.load(ckpt, map_location='cpu', weights_only=True)
    return any('hit_head' in key for key in (saved.get('state_dict') or {}))


def slot_of(run: Run, act: Optional[str] = None, hit_head: Optional[bool] = None) -> dict:
    """A team slot playing sealed ``run``: its identity, checkpoint SHA-1, act mode (``act``, else the run's
    default) and whether its network has a hit head."""
    row = get_run(run)
    if row['status'] != SEALED:
        raise StoreError(f"run {run.identity} is open: only a sealed run plays in a team")
    return {'run': run.identity, 'sha1': row['sha1'], 'act': act or row['act'],
            'hit_head': _hit_head(run.path / CHECKPOINT) if hit_head is None else hit_head}


def teams() -> list[dict]:
    """The teams (name, created, sha256, source, current)."""
    con = connect()
    try:
        rows = [dict(row) for row in con.execute('SELECT * FROM teams ORDER BY created, name')]
    finally:
        con.close()
    current = paths.current_team()
    for row in rows:
        row['current'] = row['name'] == current
    return rows


def read_team(name: str = paths.CURRENT) -> dict:
    """A team's JSON (``current``: the team of record); ``MissingError`` naming the file otherwise."""
    resolved = paths.current_team() if name == paths.CURRENT else name
    if resolved is None:
        raise MissingError(f"no current team: {paths.teams_dir() / paths.CURRENT} is missing (`triforce store "
                           f"teams` lists them)")
    path = paths.team_file(resolved)
    if not path.is_file():
        raise MissingError(f"no team {resolved!r}: {path} is missing (`triforce store teams` lists them)")
    return json.loads(path.read_text(encoding='utf-8'))


def _team_sha256(slots: dict) -> str:
    return hashlib.sha256(json.dumps({key: [slot['run'], slot['sha1'], slot['act']]
                                      for key, slot in sorted(slots.items())}).encode()).hexdigest()


def create_team(name: str, slots: dict, source: str, note: Optional[str] = None, entry: str = LEAD) -> dict:
    """Writes ``teams/<name>.json`` (read-only, never changed): per slot key (``slot_key``) the sealed run's
    identity, checkpoint SHA-1, act mode and hit head (``slot_of``).  The lead's."""
    if not name or '/' in name or name == paths.CURRENT or '+' in name:
        raise StoreError(f"bad team name {name!r}")
    for key, slot in slots.items():
        split_slot(key)
        row = get_run(parse_run(slot['run']))
        if row['status'] != SEALED or row['sha1'] != slot['sha1']:
            raise StoreError(f"team {name}: slot {key} names {slot['run']} ({row['status']}, sha1 {row['sha1']}), "
                             f"not a sealed run with sha1 {slot['sha1']}")
    team = {'name': name, 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'source': source, 'note': note,
            'sha256': _team_sha256(slots), 'slots': dict(sorted(slots.items()))}
    path = paths.team_file(name)
    with writing('team-new', {'name': name, 'slots': len(slots), 'source': source, 'sha256': team['sha256']},
                 entry) as con:
        if path.exists() or con.execute('SELECT 1 FROM teams WHERE name = ?', (name,)).fetchone():
            raise StoreError(f"team {name} exists ({path}); a team is never changed: make a new one")
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_text(path, json.dumps(team, indent=2) + '\n')
        path.chmod(0o444)
        con.execute('INSERT INTO teams VALUES (?, ?, ?, ?, ?)', (name, time.time(), team['sha256'], source, note))
    return team


def set_current_team(name: str, entry: str = LEAD) -> dict:
    """Makes team ``name`` the team of record (``teams/current``; replaces ``triforce promote``).  The lead's."""
    team = read_team(name)
    with writing('team-set-current', {'name': name}, entry) as con:
        con.execute('INSERT OR REPLACE INTO currents VALUES (?, ?, ?)', ('team', name, time.time()))
        _write_text(paths.teams_dir() / paths.CURRENT, name + '\n')
        write_layout()
    return team


def parse_team_spec(spec: str) -> tuple[str, list[tuple[str, str]]]:
    """``NAME[+SLOT=RUN]...`` (``2.0.0.13+zol@d3first=r40-x-s401+darknut=darknut/r41-y``) -> (NAME, [(slot,
    run)]); NAME ``current`` is the team of record."""
    base, *rest = spec.split('+')
    overrides = []
    for part in rest:
        key, sep, value = part.partition('=')
        if not sep or not key or not value:
            raise StoreError(f"--team {spec}: each override is +SLOT=RUN (SLOT: family, family@route or "
                             f"family@route:late; RUN: RUN or FAMILY/RUN), not {part!r}")
        split_slot(key)
        overrides.append((key, value))
    return base or paths.CURRENT, overrides


def effective_team(spec: str) -> dict:
    """The team ``spec`` names (``parse_team_spec``) with its overrides applied: ``{'name': spec, 'base', 'base_sha256',
    'overrides', 'sha256', 'slots'}``.  An override's run is a sealed store run (``RUN`` alone: of the slot's family).
    This dict is what a run, a chain and an evaluation record as the team they played."""
    base_name, overrides = parse_team_spec(spec)
    base = read_team(base_name)
    slots = {key: dict(slot) for key, slot in base['slots'].items()}
    for key, value in overrides:
        family, _, _ = split_slot(key)
        slots[key] = slot_of(parse_run(value, family))
    name = base['name'] + ''.join(f'+{key}={slots[key]["run"]}' for key, _ in overrides)
    return {'name': name, 'base': base['name'], 'base_sha256': base['sha256'],
            'overrides': {key: slots[key]['run'] for key, _ in overrides}, 'sha256': _team_sha256(slots),
            'slots': slots}


def team_record(team: dict) -> dict:
    """What a result records of the team it played: name, base, overrides, content hash and per slot run, SHA-1 and
    act mode."""
    return {'name': team['name'], 'base': team.get('base', team['name']), 'overrides': team.get('overrides', {}),
            'sha256': team['sha256'],
            'slots': {key: {k: slot[k] for k in ('run', 'sha1', 'act')} for key, slot in team['slots'].items()}}


# ---------------------------------------------------------------------------------------------------------------
# Views: an old-style models folder of links for a team

def write_view(team: dict, into: Path, evaluations: bool = True, readonly: bool = True) -> Path:
    """Writes the old-style models folder of ``team`` into ``into`` (which must not exist or be empty):
    ``<family>/<pointer>`` files (``promoted``, ``promoted.<route>``, ``late.<route>``) naming each slot's run, and
    ``<family>/<run>/`` per run: links to every file of the sealed run, an ``act`` file (``<mode> <sha1>``, the
    slot's), and (``evaluations``) links to the evaluations filed for the run (the latest of each name).  The folder
    is made read-only (``readonly``): a tool that would write into a run through it fails.  What ``--team`` plays,
    and ``triforce store view`` (old-version trees)."""
    into = Path(into)
    if into.exists() and any(into.iterdir()):
        raise StoreError(f"{into} exists and is not empty")
    into.mkdir(parents=True, exist_ok=True)
    filed = _filed_evaluations() if evaluations else {}
    acts: dict = {}
    for key, slot in sorted(team['slots'].items()):
        family, _, _ = split_slot(key)
        run = parse_run(slot['run'])
        folder = into / family / run.name
        mode = f"{slot['act']} {slot['sha1']}\n"
        if acts.setdefault(str(folder), mode) != mode:
            raise StoreError(f"team {team['name']}: run {run.identity} plays in two slots of {family} with two act "
                             f"modes; an old-style folder holds one act file a run")
        if not folder.exists():
            folder.mkdir(parents=True)
            for entry in sorted(run.path.iterdir()):
                if entry.name != 'act':
                    (folder / entry.name).symlink_to(entry)
            for name, target in filed.get(run.identity, {}).items():
                if not (folder / name).exists():
                    (folder / name).symlink_to(target)
            (folder / 'act').write_text(mode, encoding='utf-8')
        (into / family / pointer_name(key)).write_text(run.name + '\n', encoding='utf-8')
    (into / 'TEAM.json').write_text(json.dumps(team_record(team), indent=2) + '\n', encoding='utf-8')
    if readonly:
        make_readonly(into)
    return into


def _filed_evaluations() -> dict:
    """run identity -> {file name: path} of the evaluations filed for it (the latest of each name)."""
    found: dict = {}
    try:
        con = connect()
    except MissingError:
        return found
    try:
        for row in con.execute("SELECT run, name, path FROM outputs WHERE kind = 'evaluate' AND run IS NOT NULL "
                               "ORDER BY id"):
            found.setdefault(row['run'], {})[Path(row['name'] or row['path']).name] = root() / row['path']
    finally:
        con.close()
    return found


_TEAMS: dict = {}
_VIEWS: dict = {}


def team_of(spec: str) -> dict:
    """``effective_team(spec)``, resolved once per process (a run, a chain, an evaluation records this one)."""
    if spec not in _TEAMS:
        _TEAMS[spec] = effective_team(spec)
    return _TEAMS[spec]


def team_view(spec: str) -> Path:
    """This process's view (``write_view``) of team ``spec`` (``team_of``): made once in a temporary folder, removed
    at exit."""
    if spec not in _VIEWS:
        base = Path(os.environ.get('XDG_RUNTIME_DIR') or tempfile.gettempdir())
        folder = Path(tempfile.mkdtemp(prefix=f'triforce-team-{os.getpid()}-', dir=base))
        write_view(team_of(spec), folder / 'models')
        _VIEWS[spec] = folder / 'models'
        import atexit  # pylint: disable=import-outside-toplevel

        def remove(folder=folder):
            make_writable(folder)
            shutil.rmtree(folder, ignore_errors=True)
        atexit.register(remove)
    return _VIEWS[spec]


# ---------------------------------------------------------------------------------------------------------------
# Jobs: what a job files

def _safe(text: str) -> str:
    return ''.join(c if c.isalnum() or c in '-_.' else '-' for c in text).strip('-') or 'job'


def job_folder(label: Optional[str] = None, entry: str = PUBLIC) -> Path:
    """This job's folder ``jobs/<job id>-<label>/`` (``$TRIFORCE_JOB_ID``, ``$TRIFORCE_JOB_LABEL``), made on the
    first call; a process outside a job (the owner at a terminal) gets ``jobs/local-<time>-<pid>-<label>``.  Tools
    write their outputs in it while the job runs and file them (``file_output``)."""
    caller = who()
    job = caller['job'] or f"local-{time.strftime('%y%m%d-%H%M%S')}-{os.getpid()}"
    con = connect()
    try:
        row = con.execute('SELECT folder FROM jobs WHERE job = ?', (job,)).fetchone()
    finally:
        con.close()
    if row is not None:
        return root() / row['folder']
    label = _safe(jobs.label(label or 'tool'))
    folder = paths.job_dir(job, label)
    with writing('job-folder', {'job': job, 'folder': _relative(folder)}, entry) as con:
        existing = con.execute('SELECT folder FROM jobs WHERE job = ?', (job,)).fetchone()
        if existing is not None:
            return root() / existing['folder']
        folder.mkdir(parents=True, exist_ok=True)
        con.execute('INSERT INTO jobs VALUES (?, ?, ?, ?, ?)', (job, label, caller['agent'], time.time(),
                                                                _relative(folder)))
    return folder


def _get(stats: dict, dotted: str):
    value = stats
    for part in dotted.split('.'):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def summary_row(summary: Optional[dict], result: Optional[dict] = None) -> dict:
    """The kept columns of a filed output: from a statistics dict (``triforce.run_stats``: ``rate`` and the compact
    fields ``SUMMARY_FIELDS``), else from a tool's own ``result`` (``n``, ``successes``)."""
    if summary:
        rate = summary.get('rate') or {}
        return {'n': rate.get('n'), 'successes': rate.get('successes'), 'rate': rate.get('rate'),
                'se': rate.get('se'), 'compact': {field: _get(summary, field) for field in SUMMARY_FIELDS}}
    result = result or {}
    n, successes = result.get('n'), result.get('successes')
    rate = successes / n if n and successes is not None else None
    se = (rate * (1 - rate) / n) ** 0.5 if rate is not None else None
    return {'n': n, 'successes': successes, 'rate': rate, 'se': se, 'compact': None}


def file_output(path, kind: str, *, run: Optional[str] = None, team: Optional[str] = None,
                data: Optional[list] = None, summary: Optional[dict] = None, result: Optional[dict] = None,
                name: Optional[str] = None, move: bool = False, entry: str = PUBLIC) -> Path:
    """Files ``path`` (a file or a folder: an evaluation, a chain's records and its ``.meta.json``, statistics, a
    replay pack, a harvest) for this job: a path inside the job's folder is indexed where it is; one outside is copied
    (``move``: moved) into it, so the tool's own file stays as it was.  The filed copy is read-only.  ``run``
    (``family/run``) and ``team`` (the effective team's name) index it; ``kind`` 'chain', 'evaluate' or 'train' also
    leaves one kept summary row (``summary_row``: the statistics dict ``summary``, else ``result``'s ``n`` /
    ``successes``)."""
    source = Path(path)
    if not source.exists():
        raise MissingError(f"nothing to file at {source}")
    folder = job_folder()
    try:
        source.resolve().relative_to(folder.resolve())
        target = source.resolve()
    except ValueError:
        target = folder / kind / (name or source.name)
        if target.exists():
            target = target.with_name(f'{target.stem}-{time.strftime("%H%M%S")}{target.suffix}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(source), target)
        else:
            link_tree(source, target, link=False)
    row = summary_row(summary, result) if kind in ('chain', 'evaluate', 'train') else None
    caller = who()
    with writing('file', {'kind': kind, 'path': _relative(target), 'run': run, 'team': team}, entry) as con:
        make_readonly(target)
        cursor = con.execute('INSERT INTO outputs (job, kind, path, name, run, team, data, created) '
                             'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                             (caller['job'], kind, _relative(target), name or source.name, run, team,
                              None if data is None else json.dumps(data), time.time()))
        if row is not None:
            con.execute('INSERT INTO summaries (ts, kind, name, job, run, team, data, n, successes, rate, se, '
                        'compact, output) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (time.time(), kind, (summary or {}).get('name') or name or source.name, caller['job'], run,
                         team, None if data is None else json.dumps(data), row['n'], row['successes'], row['rate'],
                         row['se'], None if row['compact'] is None else json.dumps(row['compact']),
                         cursor.lastrowid))
    return target


def record_summary(kind: str, name: str, *, run: Optional[str] = None, team: Optional[str] = None,
                   data: Optional[list] = None, summary: Optional[dict] = None, result: Optional[dict] = None,
                   entry: str = PUBLIC):
    """A kept summary row with no file of its own (a training: its files are its sealed run)."""
    row = summary_row(summary, result)
    caller = who()
    with writing('summary', {'kind': kind, 'name': name, 'run': run, 'team': team}, entry) as con:
        con.execute('INSERT INTO summaries (ts, kind, name, job, run, team, data, n, successes, rate, se, compact) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (time.time(), kind, name, caller['job'], run, team, None if data is None else json.dumps(data),
                     row['n'], row['successes'], row['rate'], row['se'],
                     None if row['compact'] is None else json.dumps(row['compact'])))


def outputs(job: Optional[str] = None, run: Optional[str] = None, kind: Optional[str] = None) -> list[dict]:
    """Filed outputs, oldest first, by job, run or kind."""
    con = connect()
    try:
        sql, args = 'SELECT * FROM outputs WHERE 1', []
        for column, value in (('job', job), ('run', run), ('kind', kind)):
            if value is not None:
                sql, args = sql + f' AND {column} = ?', [*args, str(value)]
        return [dict(row) for row in con.execute(sql + ' ORDER BY id', args)]
    finally:
        con.close()


def summaries(run: Optional[str] = None, team: Optional[str] = None, kind: Optional[str] = None) -> list[dict]:
    """The kept summary rows (oldest first) of a run or a team (its name, or a name and its overrides' prefix)."""
    con = connect()
    try:
        sql, args = 'SELECT * FROM summaries WHERE 1', []
        if run is not None:
            sql, args = sql + ' AND run = ?', [*args, run]
        if team is not None:
            sql, args = sql + ' AND (team = ? OR team LIKE ?)', [*args, team, f'{team}+%']
        if kind is not None:
            sql, args = sql + ' AND kind = ?', [*args, kind]
        rows = [dict(row) for row in con.execute(sql + ' ORDER BY id', args)]
    finally:
        con.close()
    for row in rows:
        row['compact'] = None if row['compact'] is None else json.loads(row['compact'])
    return rows


def output_dir(what: str) -> Path:
    """A folder of this job's folder for a tool's output named ``what`` (an eval-chain's records, a harvest)."""
    folder = job_folder() / what
    folder.mkdir(parents=True, exist_ok=True)
    return folder
