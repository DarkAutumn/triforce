"""Keeps every replay that shows how a version of the networks played (the owner, Nov 22: "I want to go back to
previous runs and say 'see this was a problem'").

    .venv/bin/python scripts/archive_replays.py [PATH ...] [--git-history] [--repo replays] [--backup DIR]

Each replay (a file, or every ``*.tfreplay`` under a directory, and with ``--git-history`` every version of every
replay ever committed) is named by what it is, from its own META:

    <triforce version>_<route>_<kind>_<networks>_<frames>f[_<ending>].tfreplay.bz2

``kind`` is ``golden`` (a race stitched from segment winners), ``single-run`` (one unbroken run from power on or
the start) or ``chain-<start state>`` (a segment); ``networks`` is ``single-network`` when one network plays every
slot, else ``<n>-networks``. Golden and single runs go bzip2-compressed into the repository's ``replays/``
(checked in: about 100 KB each); every replay goes into ``--backup`` (default ``/backup/triforce/replays``),
under ``<kind>/``. Identical files are kept once (by SHA-1); ``index.jsonl`` in the backup records each file's
name, its SHA-1, META summary and every path it was found at. Run it again at any time: what is already kept is
skipped. ``bunzip2 -k FILE`` restores a replay ``tf replay verify`` and ``play-zelda.py`` read.
"""
import argparse
import bz2
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from triforce import _native

ROOT = Path(__file__).resolve().parents[1]


def describe(data: bytes) -> dict:
    """What the replay is, from its META (and the frame count)."""
    replay = _native.Replay(data)
    meta = replay.meta
    source = str(meta.get('source', ''))
    golden = str(meta.get('golden', '')) == 'True'
    whole = 'power-on' in source and str(meta.get('stop_at', 'None')) in ('None', '')
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


def name_of(info: dict) -> str:
    tail = '' if info['kind'] in ('golden', 'single-run') and info['ending'] == 'win' else f"_{info['ending']}"
    return (f"{info['version']}_{info['route']}_{info['kind']}_{info['networks']}_{info['frames']}f{tail}"
            '.tfreplay.bz2')


def git_versions() -> list[tuple[bytes, str]]:
    """Every distinct blob of every ``*.tfreplay`` path in the repository's history (``<commit>:<path>``)."""
    out = subprocess.run(['git', '-C', str(ROOT), 'log', '--all', '--format=%H', '--name-only', '--diff-filter=AM',
                          '--', '*.tfreplay'], capture_output=True, text=True, check=True).stdout
    seen, found, commit = set(), [], None
    for line in out.splitlines():
        if re.fullmatch(r'[0-9a-f]{40}', line):
            commit = line
        elif line.endswith('.tfreplay') and commit:
            blob = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', f'{commit}:{line}'],
                                  capture_output=True, text=True).stdout.strip()
            if blob and blob not in seen:
                seen.add(blob)
                data = subprocess.run(['git', '-C', str(ROOT), 'cat-file', 'blob', blob],
                                      capture_output=True, check=True).stdout
                found.append((data, f'git:{commit[:8]}:{line}'))
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('paths', nargs='*', help="replay files or directories (searched for *.tfreplay)")
    parser.add_argument('--git-history', action='store_true', help="also every committed version of every replay")
    parser.add_argument('--repo', default=str(ROOT / 'replays'), help="where golden and single runs are checked in")
    parser.add_argument('--backup', default='/backup/triforce/replays', help="where every replay is kept")
    args = parser.parse_args(argv)
    repo, backup = Path(args.repo), Path(args.backup)
    repo.mkdir(parents=True, exist_ok=True)
    backup.mkdir(parents=True, exist_ok=True)
    index_path = backup / 'index.jsonl'
    index = {}
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            entry = json.loads(line)
            index[entry['sha1']] = entry
    items = []
    for path in args.paths:
        p = Path(path)
        files = sorted(p.rglob('*.tfreplay')) if p.is_dir() else [p]
        items += [(f.read_bytes(), str(f.resolve())) for f in files]
    if args.git_history:
        items += git_versions()
    added = kept = 0
    for data, where in items:
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
        name = name_of(info)
        names = {e['name'] for e in index.values()}
        if name in names:
            name = name.replace('.tfreplay.bz2', f'_{sha[:8]}.tfreplay.bz2')
        packed = bz2.compress(data, 9)
        folder = backup / ('golden' if info['kind'] == 'golden' else 'single-run' if info['kind'] == 'single-run'
                           else 'chain')
        folder.mkdir(exist_ok=True)
        (folder / name).write_bytes(packed)
        if info['kind'] in ('golden', 'single-run'):
            (repo / name).write_bytes(packed)
        index[sha] = {'sha1': sha, 'name': name, **info, 'found': [where]}
        added += 1
        print(f"kept {name}  <- {where}")
    index_path.write_text(''.join(json.dumps(e) + '\n' for e in sorted(index.values(), key=lambda e: e['name'])))
    print(f"{added} new, {kept} already kept; {len(index)} in {index_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
