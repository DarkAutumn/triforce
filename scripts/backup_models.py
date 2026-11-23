"""The networks half of the backup (``/backup/triforce``; its layout and why: ``/backup/triforce/README.md``).

    .venv/bin/python scripts/backup_models.py store MODELS_DIR --source NAME [--link]
    .venv/bin/python scripts/backup_models.py manifest VERSION CHAIN.meta.json [--how TEXT] [--confidence LEVEL]
    .venv/bin/python scripts/backup_models.py team VERSION OUT_DIR [--route ROUTE]
    .venv/bin/python scripts/backup_models.py index

``store`` keeps a checkout's ``models/``: every run dir (a directory holding ``ckpt.pt``, with its
``checkpoints/``) once, under ``models/runs/<path as under models/>`` (a run of the same path with another
``ckpt.pt`` becomes ``<path>@<sha1[:8]>``), and the rest (pointer files, chain records, evidence) under
``models/sources/NAME/``, with ``runs.txt`` mapping each run path of that source to where it is kept. A file whose
content is already kept is hard-linked, never copied twice (``--link``: hard-link from MODELS_DIR itself, when it is
on the backup's disk). ``runs.jsonl`` lists every run dir: its checkpoint's SHA-1, its ``version`` file, size and
every ``<source>:<path>`` it was found as.

``manifest`` writes the team a version played into ``models/teams/VERSION.json`` from a chain's provenance (the
``.meta.json`` next to a chain's records: every network's key, checkpoint and SHA-1); run it at every landing.
``team`` rebuilds a models directory that plays VERSION's team on ROUTE (each key's run copied out of the backup,
``promoted`` / ``promoted.<route>`` written; a family's late run is its bare key, which plays where ``late.<route>``
did), for ``--models-dir`` with the version's tag checked out. ``index`` rewrites ``INDEX.md``: what is kept, and
which versions' teams are complete.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP = Path(os.environ.get('TRIFORCE_BACKUP', '/backup/triforce'))


def sha1_of(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def version_key(version: str) -> list:
    return [int(x) if x.isdigit() else x for x in re.split(r'[.]', version)]


def run_path(checkpoint: str) -> str:
    """A provenance checkpoint (``family/run/ckpt.pt``, or an absolute path inside some ``models/``) -> run path."""
    parts = [p for p in Path(checkpoint).parts if p != 'ckpt.pt']
    if 'models' in parts:
        parts = parts[len(parts) - parts[::-1].index('models'):]
    return '/'.join(parts)


class Store:
    """``models/`` of the backup: ``runs/``, ``sources/``, ``teams/``, ``runs.jsonl`` and ``files.tsv``
    (SHA-1, size and one kept path of every file content, for hard links)."""

    def __init__(self, backup: Path):
        self.root = backup / 'models'
        self.runs_index = self.root / 'runs.jsonl'
        self.files_index = self.root / 'files.tsv'
        self.runs = {}
        if self.runs_index.exists():
            for line in self.runs_index.read_text().splitlines():
                entry = json.loads(line)
                self.runs[entry['path']] = entry
        self.files = {}
        if self.files_index.exists():
            for line in self.files_index.read_text().splitlines():
                sha, size, path = line.split('\t', 2)
                self.files[sha] = (int(size), path)

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_index.write_text(''.join(json.dumps(self.runs[p]) + '\n' for p in sorted(self.runs)))
        self.files_index.write_text(''.join(f'{sha}\t{size}\t{path}\n'
                                            for sha, (size, path) in sorted(self.files.items())))

    def by_sha1(self) -> dict:
        found = defaultdict(list)
        for entry in self.runs.values():
            found[entry['ckpt_sha1']].append(entry['path'])
        return found

    def put(self, source: Path, dest_rel: str, sha: str, link: bool) -> str:
        """Keeps ``source`` (content ``sha``) at ``models/<dest_rel>``: 'kept', 'same' (already there) or
        'differs' (something else is there; nothing written)."""
        dest = self.root / dest_rel
        known = self.files.get(sha)
        known_path = self.root / known[1] if known else None
        if known_path is not None and not known_path.exists():
            known_path = None
        if dest.exists():
            if known_path is not None and os.path.samefile(dest, known_path):
                return 'same'
            return 'same' if sha1_of(dest) == sha else 'differs'
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if known_path is not None:
                os.link(known_path, dest)
            elif link:
                os.link(source, dest)
            else:
                shutil.copy2(source, dest)
        except OSError:                  # too many links to one inode, or another disk: a copy
            shutil.copy2(source, dest)
            known_path = None            # the copy is what later files link to
        if known_path is None:
            self.files[sha] = (dest.stat().st_size, dest_rel)
        return 'kept'


def walk(models: Path):
    """(run dirs, other files) under ``models``, as paths relative to it; symlinked run dirs count under their link."""
    run_dirs, others = [], []
    for dirpath, dirnames, filenames in os.walk(models, followlinks=True):
        here = Path(dirpath)
        if 'ckpt.pt' in filenames:
            run_dirs.append(here.relative_to(models))
            dirnames[:] = []
            continue
        others += [(here / f).relative_to(models) for f in filenames if (here / f).is_file()]
    return run_dirs, others


def store(args, hashes: dict | None = None) -> int:
    """``hashes``: path -> SHA-1 already computed (a migration's ``sha1sum`` of everything)."""
    models, source = Path(args.models_dir).resolve(), args.source
    kept = Store(Path(args.backup))
    hashed = hashes or {}

    def digest(path: Path) -> str:
        return hashed.get(str(path)) or sha1_of(path)

    run_dirs, others = walk(models)
    mapping, differs, new_runs = [], 0, 0
    for rel in sorted(run_dirs):
        ckpt_sha = digest(models / rel / 'ckpt.pt')
        path = str(rel)
        same = [p for p in (path, *[q for q in kept.runs if q.startswith(path + '@')])
                if p in kept.runs and kept.runs[p]['ckpt_sha1'] == ckpt_sha]
        dest = same[0] if same else path if path not in kept.runs else f'{path}@{ckpt_sha[:8]}'
        new_runs += not same
        size = 0
        for dirpath, _, filenames in os.walk(models / rel, followlinks=True):
            for name in filenames:
                file = Path(dirpath) / name
                inner = file.relative_to(models / rel)
                size += file.stat().st_size
                if kept.put(file, f'runs/{dest}/{inner}', digest(file), args.link) == 'differs':
                    differs += 1
                    kept.put(file, f'sources/{source}/{rel}/{inner}', digest(file), args.link)
        version = models / rel / 'version'
        entry = kept.runs.setdefault(dest, {'path': dest, 'ckpt_sha1': ckpt_sha,
                                            'version': version.read_text().strip() if version.is_file() else None,
                                            'bytes': size, 'found': []})
        if f'{source}:{path}' not in entry['found']:
            entry['found'].append(f'{source}:{path}')
        mapping.append(f'{path}\t{dest}\n')
    for rel in others:
        kept.put(models / rel, f'sources/{source}/{rel}', digest(models / rel), args.link)
    runs_txt = kept.root / 'sources' / source / 'runs.txt'
    runs_txt.parent.mkdir(parents=True, exist_ok=True)
    runs_txt.write_text(''.join(mapping))
    kept.save()
    print(f"{source}: {len(run_dirs)} run dirs ({new_runs} new), {len(others)} other files, {differs} files differing "
          f"from the kept run (under sources/{source}/)")
    return 0


def manifest(args) -> int:
    meta = json.loads(Path(args.meta).read_text())
    provenance = meta.get('provenance', meta)
    route = provenance.get('route')
    networks = {m['name']: {'run': run_path(m['checkpoint']), 'checkpoint_sha1': m.get('checkpoint_sha1'),
                            'hash': m.get('hash'), 'act': m.get('act'), 'trained_version': m.get('version')}
                for m in provenance.get('models') or []}
    write_team(Path(args.backup), args.version, route, networks,
               {'file': str(Path(args.meta).resolve()), 'kind': 'chain-meta', 'code': provenance.get('code')},
               args.how or 'from the provenance of a chain the version played (manifest)', args.confidence)
    return 0


def write_team(backup: Path, version: str, route: str, networks: dict, evidence: dict, how: str, confidence: str,
               extra: dict | None = None) -> Path:
    path = backup / 'models' / 'teams' / f'{version}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    team = json.loads(path.read_text()) if path.exists() else {'version': version, 'routes': {}}
    tag = f'v{version}'
    commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--verify', '-q', f'{tag}^{{commit}}'],
                            capture_output=True, text=True).stdout.strip()
    team['tag'], team['commit'] = (tag, commit) if commit else (None, None)
    team.setdefault('route_of_record', route)
    team['routes'][route] = {'networks': networks, 'evidence': evidence, 'how': how, 'confidence': confidence}
    team.update(extra or {})
    path.write_text(json.dumps(team, indent=1) + '\n')
    print(f"wrote {path} ({route}: {len(networks)} networks)")
    return path


def completeness(kept: Store, networks: dict) -> tuple[list, list]:
    """(present: [(key, kept run path)], missing: [key -> run]) by checkpoint SHA-1."""
    by_sha = kept.by_sha1()
    present, missing = [], []
    for key, network in sorted(networks.items()):
        paths = by_sha.get(network.get('checkpoint_sha1') or '')
        if paths:
            preferred = [p for p in paths if p == network['run']] or [p for p in paths if p.startswith(network['run'])]
            present.append((key, (preferred or paths)[0]))
        else:
            missing.append(f"{key}={network['run']}")
    return present, missing


def team(args) -> int:
    kept = Store(Path(args.backup))
    record = json.loads((kept.root / 'teams' / f'{args.version}.json').read_text())
    route = args.route or record.get('route_of_record') or next(iter(record['routes']))
    networks = record['routes'][route]['networks']
    present, missing = completeness(kept, networks)
    if missing:
        print(f"{args.version} on {route}: not every network is kept: {', '.join(missing)}", file=sys.stderr)
        return 1
    out = Path(args.out_dir)
    for key, stored in present:
        family = key.split('@')[0]
        run = networks[key]['run']
        if run.split('/')[0] != family:
            print(f"{key}: run {run} is not under its family; copy it by hand", file=sys.stderr)
            return 1
        if not (out / run).exists():
            shutil.copytree(kept.root / 'runs' / stored, out / run)
        pointer = 'promoted' if '@' not in key else f"promoted.{key.split('@')[1]}"
        (out / family / pointer).write_text(run.split('/', 1)[1] + '\n')
    acts = sorted({str(n.get('act')) for n in networks.values()})
    print(f"{out}: {args.version}'s {len(present)} networks on {route} (act {', '.join(acts)})")
    return 0


def index(args) -> int:
    backup = Path(args.backup)
    kept = Store(backup)
    teams = sorted((json.loads(p.read_text()) for p in (kept.root / 'teams').glob('*.json')),
                   key=lambda t: version_key(t['version']))
    replays = Counter()
    replay_index = backup / 'replays' / 'index.jsonl'
    if replay_index.exists():
        for line in replay_index.read_text().splitlines():
            entry = json.loads(line)
            replays[(entry['version'], entry['path'].split('/')[1])] += 1
    lines = ['# What /backup/triforce keeps', '',
             f"Written by `scripts/backup_models.py index` on {datetime.date.today()}; the layout and how to use it: "
             "`README.md`.", '',
             '## Can a version\'s 250-run test run again without retraining?', '',
             "Yes when every network of the version's team (its route of record) is kept, matched by checkpoint SHA-1 "
             "(`models/teams/<version>.json`; rebuild a models dir with `scripts/backup_models.py team VERSION DIR`). "
             "The one-network student is the single network a golden or single run of that version played.", '',
             '| version | route | networks | team complete | missing runs | one-network student kept | team from |',
             '|---|---|---|---|---|---|---|']
    by_sha = kept.by_sha1()
    for record in teams:
        routes = record['routes']
        for route in sorted(routes, key=lambda r: r != record.get('route_of_record')):
            entry = routes[route]
            _, missing = completeness(kept, entry['networks'])
            students = [s for s in record.get('one_network') or [] if s.get('route', route) == route]
            student = ', '.join(f"{s['run']} ({'yes' if by_sha.get(s.get('checkpoint_sha1') or '') else 'no'})"
                                for s in students) or 'none played'
            lines.append(f"| {record['version']} | {route} | {len(entry['networks'])} | "
                         f"{'**yes**' if not missing else 'no'} | {', '.join(missing) or '-'} | {student} | "
                         f"{entry['confidence']}: {entry['how']} |")
    missing_versions = record_gaps(teams)
    if missing_versions:
        lines += ['', f"Tagged versions with no team on record (nothing of theirs played a chain or replay that "
                      f"names its networks): {', '.join(missing_versions)}."]
    lines += ['', '## Runs', '',
              f"{len(kept.runs)} run dirs under `models/runs/` ({len(kept.files)} distinct files, "
              f"{sum(s for s, _ in kept.files.values()) / 1e9:.1f} GB), listed in `models/runs.jsonl`; by family:", '']
    families = Counter(p.split('/')[0] for p in kept.runs)
    lines += [', '.join(f'{family} {n}' for family, n in sorted(families.items())), '',
              '## Sources (what each checkout\'s models/ held besides runs)', '']
    sources = sorted(p.name for p in (kept.root / 'sources').iterdir()) if (kept.root / 'sources').exists() else []
    lines += [f"- `models/sources/{s}/`" for s in sources]
    lines += ['', '## Replays (`replays/<version>/<kind>/`)', '', '| version | kind | replays |', '|---|---|---|']
    lines += [f'| {v} | {k} | {n} |' for (v, k), n in sorted(replays.items(), key=lambda x: (version_key(x[0][0]), x[0][1]))]
    data = backup / 'worktree-data'
    lines += ['', '## Worktree data (`worktree-data/<date>_<branch>/`)', '']
    lines += [f'- `{p.name}`' for p in sorted(data.iterdir())] if data.exists() else []
    (backup / 'INDEX.md').write_text('\n'.join(lines) + '\n')
    print(f"wrote {backup / 'INDEX.md'}")
    return 0


def record_gaps(teams: list) -> list:
    tags = subprocess.run(['git', '-C', str(ROOT), 'tag', '--list', 'v*'], capture_output=True, text=True).stdout
    have = {t['version'] for t in teams}
    return sorted((t[1:] for t in tags.split() if re.fullmatch(r'v[\d.]+', t) and t[1:] not in have), key=version_key)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('--backup', default=str(BACKUP), help="the backup's root (default /backup/triforce)")
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('store', help="keep a checkout's models/")
    p.add_argument('models_dir')
    p.add_argument('--source', required=True, help="what it is: <date>_<branch> (sources/NAME/)")
    p.add_argument('--link', action='store_true', help="hard-link new files from MODELS_DIR (same disk)")
    p = sub.add_parser('manifest', help="record a version's team from a chain's .meta.json")
    p.add_argument('version')
    p.add_argument('meta')
    p.add_argument('--how')
    p.add_argument('--confidence', default='high', choices=('high', 'medium', 'low'))
    p = sub.add_parser('team', help="rebuild a models dir that plays a version's team")
    p.add_argument('version')
    p.add_argument('out_dir')
    p.add_argument('--route')
    sub.add_parser('index', help="rewrite INDEX.md")
    args = parser.parse_args(argv)
    return {'store': store, 'manifest': manifest, 'team': team, 'index': index}[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
