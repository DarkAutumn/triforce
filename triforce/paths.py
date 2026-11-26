"""Every location the project reads or writes outside git: each is one function's answer (``docs/data-layout.md``).

Two kinds of place:

- **The checkout** (``checkout()``): the tracked files, and ``scratch()`` (``<checkout>/target``), per checkout and
  never an input to another run.  ``STATES_DIR`` (the tracked start states, ``$TRIFORCE_STATES_DIR``),
  ``SPECIALISTS_DIR`` and the corpus are tracked.
- **The shared store** (``shared_root()``: ``$TRIFORCE_SHARED``, else ``~/triforce-shared``), outside every checkout,
  written only through ``triforce.store`` (and the lead's ``triforce.lead``): ``rom()``, the named data snapshots
  (``data(kind, name)``), the network runs (``run_dir``), the teams (``team_file``), what each job filed
  (``job_dir``).  ``layout()`` lists them all; the store writes it as ``layout.json`` for tools that are not Python.
  The Rust side resolves the ROM by the same rule (the ``paths`` crate).

The transition (the owner's staged cutover): until the lead's reconciliation, a checkout that still has today's
folders (``roms/``, ``trajectories/``, ``triforce/states/harvested``, ``models/``, or the links ``triforce job
setup`` used to make) keeps reading them exactly as before (``legacy_data``); a checkout without them reads the
store: the snapshot its ``triforce/data_pins.yaml`` pins for the kind (``pins``; the file is committed, so every
worktree of a branch reads the same snapshot), else the store's current one.  Missing is an error: ``MissingError``
names every place looked in.
"""

import json
import os
from pathlib import Path
from typing import Optional

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent
STATES_DIR = Path(os.environ.get('TRIFORCE_STATES_DIR', PACKAGE_DIR / 'states'))
SPECIALISTS_DIR = ROOT / 'specialists'
CORPUS_MOVIES_DIR = ROOT / 'corpus' / 'movies'
BASE_MOVIE = CORPUS_MOVIES_DIR / 'power_on_to_start.tfmovie'
# The project's archive (`/backup/triforce/README.md`): `$TRIFORCE_BACKUP`, else /backup/triforce.
BACKUP_DIR = Path(os.environ.get('TRIFORCE_BACKUP', '/backup/triforce'))

SHARED_ENV = 'TRIFORCE_SHARED'
ROM_ENV = 'TRIFORCE_ROM'
STORE_ROM = 'rom/zelda.nes'
CHECKOUT_ROM = 'roms/zelda.nes'
CURRENT = 'current'                 # the name of a kind's default snapshot, and of the team of record
# The data kinds (named snapshots, ``data/<kind>/<name>/``) and where each lived in a checkout before the store.
DATA_KINDS = ('harvested', 'harvested-rooms', 'trajectories', 'demos')
CHECKOUT_DATA = {'harvested': 'triforce/states/harvested', 'harvested-rooms': 'triforce/states/harvested-rooms',
                 'trajectories': 'trajectories', 'demos': 'target/demos'}
_LEGACY_DATA = {'harvested': STATES_DIR / 'harvested', 'harvested-rooms': STATES_DIR / 'harvested-rooms',
                'trajectories': ROOT / 'trajectories', 'demos': ROOT / 'target' / 'demos'}
# A checkout's pins: ``kind: snapshot`` lines (``#`` comments, blank lines), read before the store's current (the
# Rust twin, ``paths::pins``, parses the same lines).  No file, no pins: the branch reads the store's current.
PINS_FILE = 'triforce/data_pins.yaml'

# Kept names: the checkout's folders as they were (today's readers and writers, until reconciliation).
HARVESTED_DIR = _LEGACY_DATA['harvested']
TRAJECTORIES_DIR = _LEGACY_DATA['trajectories']


class MissingError(FileNotFoundError):
    """A shared thing that is not there (rule 4: missing is an error); the message names where it was looked for."""


def checkout() -> Path:
    """This checkout (the folder holding the ``triforce`` package)."""
    return ROOT


def scratch(*parts: str) -> Path:
    """``<checkout>/target[/parts]``: this checkout's scratch (build output and tools' working files)."""
    return ROOT.joinpath('target', *parts)


def shared_root() -> Path:
    """The shared store: ``$TRIFORCE_SHARED``, else ``~/triforce-shared``."""
    value = os.environ.get(SHARED_ENV, '').strip()
    return Path(value) if value else Path.home() / 'triforce-shared'


def store_exists() -> bool:
    """Whether the store has been made (the lead's import or ``triforce-lead init``): its index is there."""
    return (shared_root() / 'store.db').is_file()


def rom_candidates() -> list[Path]:
    """Where the ROM is looked for, in order: ``$TRIFORCE_ROM`` alone when set, else the store's ``rom/zelda.nes``,
    then the checkout's ``roms/zelda.nes`` (the ``paths`` crate's rule)."""
    explicit = os.environ.get(ROM_ENV, '').strip()
    if explicit:
        return [Path(explicit)]
    return [shared_root() / STORE_ROM, ROOT / CHECKOUT_ROM]


def rom() -> Path:
    """The ROM file (``rom_candidates``' first that exists); ``MissingError`` naming every place otherwise."""
    candidates = rom_candidates()
    for path in candidates:
        if path.is_file():
            return path
    raise MissingError(f"ROM not found: looked in {', '.join(map(str, candidates))} (set {ROM_ENV}, or ask the lead "
                       f"to install the store's {STORE_ROM})")


def _check_kind(kind: str):
    if kind not in DATA_KINDS:
        raise ValueError(f"unknown data kind {kind!r}; the kinds are {', '.join(DATA_KINDS)}")


def data_root(kind: str) -> Path:
    """``<store>/data/<kind>``: the kind's snapshots and its ``current`` file."""
    _check_kind(kind)
    return shared_root() / 'data' / kind


def current_name(kind: str) -> Optional[str]:
    """The name of ``kind``'s current snapshot (``data/<kind>/current``), None when none is set."""
    return _read_name(data_root(kind) / CURRENT)


def data(kind: str, name: str = CURRENT) -> Path:
    """The folder of ``kind``'s snapshot ``name`` (``current``: the one the lead set as the default); read-only.
    ``MissingError`` when it is not installed."""
    root = data_root(kind)
    resolved = current_name(kind) if name == CURRENT else name
    if resolved is None:
        raise MissingError(f"no current {kind} snapshot: {root / CURRENT} is missing (the store at {shared_root()} "
                           f"has none installed; `triforce store data` lists them)")
    path = root / resolved
    if not path.is_dir():
        raise MissingError(f"no {kind} snapshot {resolved!r}: {path} is missing (`triforce store data {kind}` lists "
                           f"them)")
    return path


def legacy_data(kind: str) -> Path:
    """Where ``kind`` lived in this checkout before the store (``triforce/states/harvested``, ``.../harvested-rooms``,
    ``trajectories``, ``target/demos``); read while it exists, until the lead's reconciliation."""
    _check_kind(kind)
    return _LEGACY_DATA[kind]


def pins(checkout: Optional[Path] = None) -> dict:
    """kind -> snapshot name the checkout pins (``<checkout>/triforce/data_pins.yaml``, default this one; {} without
    the file).  ``ValueError`` naming the line for anything but ``kind: name`` lines of known kinds, once each."""
    path = Path(checkout or ROOT) / PINS_FILE
    if not path.is_file():
        return {}
    found = {}
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        text = line.split('#', 1)[0].strip()
        if not text:
            continue
        kind, sep, name = (part.strip() for part in text.partition(':'))
        if not sep or kind not in DATA_KINDS or not name or kind in found or ' ' in name or name == CURRENT:
            raise ValueError(f"{path}:{number}: {line.strip()!r}: a pin is 'KIND: SNAPSHOT' with KIND one of "
                             f"{', '.join(DATA_KINDS)}, once each, and a snapshot name")
        found[kind] = name
    return found


def data_source(kind: str, name: Optional[str] = None) -> tuple[Path, dict]:
    """(folder, record) of the ``kind`` a tool reads: snapshot ``name`` from the store when one is named; else this
    checkout's own folder when it still has one (the transition: today's command lines read what they read); else the
    snapshot this checkout pins (``pins``); else the store's current snapshot.  The record says which (``{'kind',
    'snapshot', 'path'}``, with ``'pinned': True`` for a pin, or ``{'kind', 'legacy', 'path'}``), for the run's
    provenance.  ``MissingError`` when there is none of them, or the pinned snapshot is not in the store."""
    if name is None and legacy_data(kind).exists():
        return legacy_data(kind), {'kind': kind, 'legacy': True, 'path': str(legacy_data(kind))}
    pinned = pins().get(kind) if name is None else None
    try:
        path = data(kind, name or pinned or CURRENT)
    except MissingError as error:
        if pinned:
            raise MissingError(f"{error}; it is pinned by {ROOT / PINS_FILE}") from error
        if name is None:
            raise MissingError(f"{error}; this checkout has no {legacy_data(kind)} either") from error
        raise
    record = {'kind': kind, 'snapshot': path.name, 'path': str(path)}
    return path, {**record, 'pinned': True} if pinned else record


def data_dir(kind: str, legacy: Optional[Path] = None, checkout: Optional[Path] = None) -> Path:
    """The folder of ``kind`` every reader uses (the Rust twin is ``paths::data_dir``): ``legacy`` (default this
    checkout's ``legacy_data(kind)``; a caller's own states dir passes ``<states>/harvested``) while it exists, else
    the snapshot ``checkout`` (default this one) pins, else the store's current snapshot, else ``legacy`` again for
    the reader's refusal (``data_missing`` says where it looked)."""
    legacy = Path(legacy) if legacy is not None else legacy_data(kind)
    if legacy.exists():
        return legacy
    try:
        return data(kind, pins(checkout).get(kind, CURRENT))
    except MissingError:
        return legacy


def data_missing(kind: str, legacy: Optional[Path] = None, checkout: Optional[Path] = None) -> str:
    """What a reader of ``kind`` says when ``data_dir`` found nothing: both places it looked."""
    legacy = Path(legacy) if legacy is not None else legacy_data(kind)
    pinned = pins(checkout).get(kind)
    if pinned:
        return (f"{legacy} is missing and the store at {shared_root()} has no {kind} snapshot {pinned!r} (pinned by "
                f"{Path(checkout or ROOT) / PINS_FILE})")
    return (f"{legacy} is missing and the store at {shared_root()} has no current {kind} snapshot "
            f"({data_root(kind) / CURRENT})")


def runs_dir() -> Path:
    """``<store>/runs``: every network run, ``<family>/<run>/``."""
    return shared_root() / 'runs'


def run_dir(family: str, run: str) -> Path:
    """``<store>/runs/<family>/<run>``."""
    return runs_dir() / family / run


def teams_dir() -> Path:
    """``<store>/teams``: ``<name>.json`` per team and ``current`` (the team of record's name)."""
    return shared_root() / 'teams'


def team_file(name: str) -> Path:
    """``<store>/teams/<name>.json``."""
    return teams_dir() / f'{name}.json'


def current_team() -> Optional[str]:
    """The name of the team of record (``teams/current``), None when none is set."""
    return _read_name(teams_dir() / CURRENT)


def jobs_dir() -> Path:
    """``<store>/jobs``: what each job filed, ``<job id>-<label>/``."""
    return shared_root() / 'jobs'


def job_dir(job: str, label: str) -> Path:
    """``<store>/jobs/<job>-<label>``."""
    return jobs_dir() / f'{job}-{label}'


def legacy_models_dir() -> Path:
    """``<checkout>/models``: the models folder before the store (``registry.models_root``'s default while it exists)."""
    return ROOT / 'models'


def logs_dir() -> Path:
    """This checkout's event logs (``$TRIFORCE_LOGS_DIR``, else ``<checkout>/logs``: the training GUI's input)."""
    value = os.environ.get('TRIFORCE_LOGS_DIR')
    return Path(value) if value else ROOT / 'logs'


def layout() -> dict:
    """Every location, as ``triforce store paths`` prints it and the store writes ``layout.json``."""
    root = shared_root()
    try:
        found_rom = str(rom())
    except MissingError:
        found_rom = None
    return {
        'shared': str(root), 'store_exists': store_exists(), 'readme': str(root / 'README.md'),
        'layout': str(root / 'layout.json'), 'index': str(root / 'store.db'),
        'rom': found_rom, 'store_rom': str(root / STORE_ROM),
        'data': {kind: {'root': str(root / 'data' / kind), 'current': current_name(kind) if root.is_dir() else None}
                 for kind in DATA_KINDS},
        'runs': str(runs_dir()), 'teams': str(teams_dir()),
        'current_team': current_team() if root.is_dir() else None, 'jobs': str(jobs_dir()),
        'checkout': str(ROOT), 'scratch': str(scratch()), 'states': str(STATES_DIR), 'logs': str(logs_dir()),
        'legacy': {'models': str(legacy_models_dir()), 'rom': str(ROOT / CHECKOUT_ROM),
                   **{kind: str(path) for kind, path in _LEGACY_DATA.items()}},
    }


def _read_name(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding='utf-8').strip() or None
    except OSError:
        return None


def state_path(name: str) -> Path:
    """``STATES_DIR/<name>.state`` (``name`` may already end in ``.state`` or be a path)."""
    path = Path(name)
    if path.suffix == '.state' and (path.is_absolute() or path.parent != Path('.')):
        return path
    return STATES_DIR / (name if name.endswith('.state') else f'{name}.state')


def layout_json() -> str:
    """``layout()`` as the JSON text of ``layout.json``."""
    return json.dumps(layout(), indent=2) + '\n'
