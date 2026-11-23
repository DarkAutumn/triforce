"""Which sources the built module was built from (round 40: a run of Python code over a module built from older Rust
is a recipe nobody wrote down, and nothing checked it).

``triforce job build`` (and every verb that builds: ``test --build``, ``setup``) writes ``triforce/_native.build.json``
after a successful build: the fingerprint of the sources the module is built from (``fingerprint``: the git index
entries of ``SOURCES``, the working tree's changes to them, and their untracked files), the module file's SHA-1 and
the time.  ``problem`` compares it with the checkout as it is now: no module, no stamp (built by hand or by
``uv run``), or sources that changed since, each is a problem; the queued tools refuse to start on one
(``jobs.announce``).  A module rebuilt from the same sources (``uv run`` after a landing) is no problem.

Standard library only (it runs before the first build)."""

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Optional

PACKAGE = Path(__file__).resolve().parent
SOURCES = ('crates', 'Cargo.toml', 'Cargo.lock', 'pyproject.toml')
STAMP_NAME = '_native.build.json'


def module_path(root: Path) -> Optional[Path]:
    """The built extension module of the checkout at ``root`` (``triforce/_native*.so``), None before a build."""
    found = sorted((Path(root) / 'triforce').glob('_native*.so'))
    return found[0] if found else None


def stamp_path(root: Path) -> Path:
    return Path(root) / 'triforce' / STAMP_NAME


def _git(root: Path, *args) -> bytes:
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True, check=True, timeout=60).stdout


def fingerprint(root: Path) -> str:
    """SHA-256 of what the module is built from at ``root``: the index entries of ``SOURCES``, the working tree's
    diff against the index, and the untracked files there (name and content)."""
    digest = hashlib.sha256()
    digest.update(_git(root, 'ls-files', '-s', '--', *SOURCES))
    digest.update(_git(root, 'diff', '--binary', '--', *SOURCES))
    for name in sorted(_git(root, 'ls-files', '-z', '--others', '--exclude-standard', '--', *SOURCES).split(b'\0')):
        if name:
            digest.update(name)
            digest.update((Path(root) / name.decode()).read_bytes())
    return digest.hexdigest()


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_stamp(root: Path) -> int:
    """After a successful build at ``root``: records its sources' fingerprint and the module's SHA-1 (a job step: 0, or
    1 with the reason on stderr when there is no module to stamp)."""
    module = module_path(root)
    if module is None:
        print(f"triforce job: no built module in {Path(root) / 'triforce'} to stamp", flush=True)
        return 1
    stamp = {'sources': fingerprint(root), 'module': module.name, 'module_sha1': file_sha1(module),
             'built': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    path = stamp_path(root)
    tmp = path.with_name(f'.{path.name}.tmp')
    tmp.write_text(json.dumps(stamp, indent=1) + '\n', encoding='utf-8')
    tmp.replace(path)
    return 0


def read_stamp(root: Path) -> Optional[dict]:
    try:
        return json.loads(stamp_path(root).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def problem(root: Path = PACKAGE.parent) -> Optional[str]:
    """Why the module at ``root`` may not be the build of its sources (None: it is)."""
    if module_path(root) is None:
        return f"no built module in {Path(root) / 'triforce'}"
    stamp = read_stamp(root)
    if stamp is None:
        return (f"{stamp_path(root)} is missing: the module was not built by `triforce job build`, so which sources "
                f"it holds is unknown")
    try:
        now = fingerprint(root)
    except (OSError, subprocess.SubprocessError) as error:
        return f"the sources of {root} can't be read ({error})"
    if now != stamp.get('sources'):
        return f"{', '.join(SOURCES)} changed since the module was built ({stamp.get('built')})"
    return None
