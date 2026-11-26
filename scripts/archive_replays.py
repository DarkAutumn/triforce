"""Keeps every replay that shows how a version of the networks played (the owner, Nov 22: "I want to go back to
previous runs and say 'see this was a problem'").

    triforce job run --agent A --task T --subtask S --desc '...' --timeout 20m -- \\
        .venv/bin/python scripts/archive_replays.py [PATH ...] [--git-history] [--kind KIND --label LABEL]
                                                    [--repo replays] [--backup DIR]

A job of the machine's queue (``jobs.require_job``: it reads, compresses and writes many replays); its progress is
the replays done of those found, then the 7z archives written.

Each replay (a file, or every ``*.tfreplay`` under a directory, and with ``--git-history`` every version of every
replay ever committed) is named by what it is, from its own META:

    <triforce version>_<route>_<kind>_<networks>_<frames>f[_<ending>][_<label>]

``kind`` is ``golden`` (a race stitched from segment winners), ``single-run`` (one unbroken run from power on or
the start) or ``chain-<start state>`` (a segment); ``networks`` is ``single-network`` when one network plays every
slot, else ``<n>-networks``. Golden and single runs go into the repository's ``replays/`` as plain ``.tfreplay``
files (checked in, uncompressed so they play as they are: the owner, Nov 23). Every replay goes into ``--backup``
(default ``/backup/triforce/replays``, laid out in ``/backup/triforce/README.md``) by version and kind:

    <version>/golden/<name>.tfreplay.bz2        one file each, bzip2
    <version>/single-run/<name>.tfreplay.bz2
    <version>/<KIND>/<name>.tfreplay.bz2        with --kind (e.g. runs-250 and --label best): backup only
    <version>/chain/<group>.7z                  segments, one 7z per directory they were found in

``--keep-names`` names each replay by its own file name instead (the owner's takes: ``--kind owner --keep-names``,
from ``scripts/owner_takes.py add``, so a take is found in the backup under the name it has in ``replays/owner/``).

Identical files are kept once (by SHA-1); ``index.jsonl`` in the backup records each replay's name, SHA-1, META
summary, where it is kept (``path``, and ``member`` inside a 7z) and every path it was found at. Run it again at
any time: what is already kept is skipped. ``bunzip2 -k FILE`` or ``7z x ARCHIVE`` restores a replay that
``tf replay verify`` and ``play-zelda.py`` read.
"""
import argparse
import bz2
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from triforce import _native, jobs

ROOT = Path(__file__).resolve().parents[1]
SINGLE_FILE_KINDS = ('golden', 'single-run')


def describe(data: bytes) -> dict:
    """What the replay is, from its META (and the frame count)."""
    replay = _native.Replay(data)
    meta = replay.meta
    source = str(meta.get('source', ''))
    golden = str(meta.get('golden', '')) == 'True'
    # From power on, or from the route's own start state ('start'), to the end with no stop: one unbroken run.
    whole = (('power-on' in source or "'name': 'start'" in source)
             and str(meta.get('stop_at', 'None')) in ('None', ''))
    if golden:
        kind = 'golden'
    elif whole:
        kind = 'single-run'
    else:
        start = None if 'power-on' in source else re.search(r"'name': '([^']+)'", source)
        kind = f"chain-{start.group(1) if start else 'power-on'}"
    hashes = {m.get('hash') for m in meta.get('models') or [] if m.get('hash')}
    networks = 'single-network' if len(hashes) == 1 else f'{len(hashes)}-networks'
    ending = str(replay.ending or 'none')
    return {'version': str(meta.get('triforce') or 'unknown'), 'route': str(meta.get('route') or 'unknown'),
            'kind': kind, 'networks': networks, 'frames': int(replay.frames), 'ending': ending,
            'code': str(meta.get('code', '')), 'stop_at': str(meta.get('stop_at', ''))}


def name_of(info: dict, label: str | None = None) -> str:
    """The replay's name, without an extension."""
    tail = '' if info['kind'] in SINGLE_FILE_KINDS and info['ending'] == 'win' else f"_{info['ending']}"
    return (f"{info['version']}_{info['route']}_{info['kind']}_{info['networks']}_{info['frames']}f{tail}"
            + (f'_{label}' if label else ''))


def group_of(where: str) -> str:
    """The 7z a segment goes into: the directory it was found in, named from its checkout on
    (``~/git/triforce-r38-hit/target/r38/m6/replays-d7-ppo`` -> ``r38-hit_r38-m6-replays-d7-ppo``)."""
    if where.startswith('git:'):
        return 'git-history'
    parts = Path(where).parent.parts
    checkout = next((i for i, part in enumerate(parts) if part == 'triforce' or part.startswith('triforce-')), None)
    if checkout is None:
        return re.sub(r'[^A-Za-z0-9._-]+', '-', '-'.join(parts[-3:])).strip('-') or 'other'
    owner = 'main' if parts[checkout] == 'triforce' else parts[checkout][len('triforce-'):]
    rest = [p for p in parts[checkout + 1:] if p != 'target']
    return re.sub(r'[^A-Za-z0-9._-]+', '-', f"{owner}_{'-'.join(rest) or 'root'}")


def seven_zip() -> str:
    found = shutil.which('7z') or shutil.which('7za')
    if not found:
        raise SystemExit("7z (p7zip) is needed to keep segments in the backup")
    return found


def add_to_7z(archive: Path, members: list[tuple[str, bytes]]) -> None:
    """Adds ``members`` (file name, bytes) to ``archive``, creating it if it is not there."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for member, data in members:
            (Path(tmp) / member).write_bytes(data)
        subprocess.run([seven_zip(), 'a', '-t7z', '-mx=9', '-bd', '-bso0', '-bsp0', str(archive.resolve()),
                        *[member for member, _ in members]], cwd=tmp, check=True)


def git_versions() -> list[tuple[bytes, str]]:
    """Every distinct blob of every ``*.tfreplay`` path in the repository's history (``<commit>:<path>``)."""
    out = subprocess.run(['git', '-C', str(ROOT), 'log', '--all', '--format=%H', '--name-only', '--diff-filter=AM',
                          '--', '*.tfreplay', '*.tfreplay.bz2'], capture_output=True, text=True, check=True).stdout
    seen, found, commit = set(), [], None
    for line in out.splitlines():
        if re.fullmatch(r'[0-9a-f]{40}', line):
            commit = line
        elif line.endswith(('.tfreplay', '.tfreplay.bz2')) and commit:
            blob = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', f'{commit}:{line}'],
                                  capture_output=True, text=True).stdout.strip()
            if blob and blob not in seen:
                seen.add(blob)
                data = subprocess.run(['git', '-C', str(ROOT), 'cat-file', 'blob', blob],
                                      capture_output=True, check=True).stdout
                found.append((bz2.decompress(data) if line.endswith('.bz2') else data, f'git:{commit[:8]}:{line}'))
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('paths', nargs='*', help="replay files or directories (searched for *.tfreplay)")
    parser.add_argument('--git-history', action='store_true', help="also every committed version of every replay")
    parser.add_argument('--kind', help="keep these replays under <version>/KIND/ in the backup only, one bzip2 file "
                                       "each, whatever their META says (e.g. runs-250: the 250-run test's picks)")
    parser.add_argument('--label', help="appended to each name (e.g. best, worst, stuck-0x35)")
    parser.add_argument('--keep-names', action='store_true',
                        help="name each replay by its own file name, not its META (the owner's takes)")
    parser.add_argument('--repo', default=str(ROOT / 'replays'), help="where golden and single runs are checked in")
    parser.add_argument('--backup', default='/backup/triforce/replays', help="where every replay is kept")
    args = parser.parse_args(argv)
    if args.kind is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.kind):
        parser.error(f"--kind {args.kind!r}: letters, digits, '.', '_' and '-' only")
    if args.label is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.label):
        parser.error(f"--label {args.label!r}: letters, digits, '.', '_' and '-' only")
    jobs.require_job()
    repo, backup = Path(args.repo), Path(args.backup)
    backup.mkdir(parents=True, exist_ok=True)
    index_path = backup / 'index.jsonl'
    index = {}
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            entry = json.loads(line)
            index[entry['sha1']] = entry
    names = {e['name'] for e in index.values()}
    items = []
    for path in args.paths:
        p = Path(path)
        files = sorted(p.rglob('*.tfreplay')) if p.is_dir() else [p]
        items += [(f.read_bytes(), str(f.resolve())) for f in files]
    if args.git_history:
        items += git_versions()
    added = kept = 0
    segments = defaultdict(list)            # 7z path (relative to the backup) -> [(member, bytes)]
    for done, (data, where) in enumerate(items):
        jobs.progress(done, len(items), 'replays')
        sha = hashlib.sha1(data).hexdigest()
        if sha in index:
            if where not in index[sha]['found']:
                index[sha]['found'].append(where)
            kept += 1
            continue
        try:
            info = describe(data)
        except Exception as error:  # pylint: disable=broad-exception-caught
            print(f"skip {where}: {type(error).__name__}: {error}", file=sys.stderr)
            continue
        name = Path(where).stem if args.keep_names and not where.startswith('git:') else name_of(info, args.label)
        if name in names:
            name = f'{name}_{sha[:8]}'
        names.add(name)
        entry = {'sha1': sha, 'name': name, **info}
        if args.kind or info['kind'] in SINGLE_FILE_KINDS:
            folder = f"{info['version']}/{args.kind or info['kind']}"
            entry['path'] = f'{folder}/{name}.tfreplay.bz2'
            (backup / folder).mkdir(parents=True, exist_ok=True)
            (backup / entry['path']).write_bytes(bz2.compress(data, 9))
            if not args.kind:
                repo.mkdir(parents=True, exist_ok=True)
                (repo / f'{name}.tfreplay').write_bytes(data)
        else:
            entry['path'] = f"{info['version']}/chain/{group_of(where)}.7z"
            entry['member'] = f'{name}.tfreplay'
            segments[entry['path']].append((entry['member'], data))
        index[sha] = {**entry, 'found': [where]}
        added += 1
        print(f"kept {entry['path']}{':' + entry['member'] if 'member' in entry else ''}  <- {where}")
    jobs.progress(len(items), len(items), 'replays')
    for done, (archive, members) in enumerate(segments.items(), 1):
        add_to_7z(backup / archive, members)
        jobs.progress(done, len(segments), 'archives')
    index_path.write_text(''.join(json.dumps(e) + '\n' for e in sorted(index.values(), key=lambda e: e['name'])))
    print(f"{added} new, {kept} already kept; {len(index)} in {index_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
