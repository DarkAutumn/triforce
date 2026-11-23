"""Run lineage (round 40; decisions 2026-11-23 "The next generation's teachers are trained fresh" and "A version's
first three numbers each mean a fresh generation"): which network a training run started from, kept so a
generation's recipe can be replayed and a fresh generation proven fresh.

A run records its parent in ``run.json`` (``triforce train``, the lab's trainer): ``parent`` is the checkpoint the
weights came from by family, run id, iterations and the checkpoint file's SHA-1, never a filesystem path, and
``null`` for a run trained from scratch; the lab also names its teachers that way (``teacher_ids``).  Runs from
before round 40 name their start only by a path (``source``, ``args.load``) or a family's name (its promoted run then,
no run id), which this module reads as well as it can.

``triforce lineage [--route R]`` prints each promoted pointer's chain (``promoted``, ``promoted.<route>``,
``late.<route>``) back to its origin, looking for run dirs in the models dir and the backup's ``models/runs``."""

import argparse
import json
from pathlib import Path
from typing import Optional

from . import registry
from .paths import BACKUP_DIR

BACKUP_RUNS = BACKUP_DIR / 'models' / 'runs'
POINTERS = (registry.PROMOTED_NAME, f'{registry.PROMOTED_NAME}.', f'{registry.LATE_NAME}.')


def identity(checkpoint: dict, path) -> dict:
    """The checkpoint at ``path`` (``checkpoint``: its loaded dict) as a run's parent or teacher: its family and run
    (as saved in it), its iterations and its file's SHA-1."""
    return {'family': checkpoint.get('specialist'), 'run': checkpoint.get('run_id'),
            'iterations': checkpoint.get('iterations'), 'sha1': registry.checkpoint_sha1(path)}


def path_identity(path) -> dict:
    """``identity`` of the checkpoint file or run dir at ``path``."""
    return identity(registry.load_checkpoint(path, check_schema=False), path)


class Lineage:
    """Run dirs by (family, run) in the models dir, then the backup (``runs/<family>/<run>[@sha]``)."""

    def __init__(self, models_dir=None, backup: Optional[Path] = BACKUP_RUNS):
        self.models = registry.models_root(models_dir)
        self.backup = backup if backup is not None and backup.is_dir() else None
        self.by_sha1 = {}
        index = self.backup.parent / 'runs.jsonl' if self.backup is not None else None
        if index is not None and index.is_file():
            for line in index.read_text(encoding='utf-8').splitlines():
                entry = json.loads(line)
                self.by_sha1.setdefault(entry['ckpt_sha1'], entry['path'])

    def run_dir(self, family: str, run: str) -> Optional[Path]:
        """The kept dir of ``family``/``run`` holding a ``run.json`` (or only a checkpoint: a network built, not
        trained), or None."""
        candidates = [self.models / family / run]
        if self.backup is not None:
            candidates.append(self.backup / family / run)
            candidates += sorted((self.backup / family).glob(f'{run}@*')) if (self.backup / family).is_dir() else []
        return (next((d for d in candidates if (d / 'run.json').is_file()), None)
                or next((d for d in candidates if (d / registry.CHECKPOINT_NAME).is_file()), None))

    def parent(self, family: str, run: str) -> tuple:
        """(kind, value) of ``family``/``run``'s parent: ('scratch', None), ('run', (family, run)), or ('lost', why)."""
        directory = self.run_dir(family, run)
        if directory is None:
            return 'lost', 'run dir not kept'
        if not (directory / 'run.json').is_file():
            # Round 38's hit-head networks were built from a policy and a hit network, not trained: the policy's.
            policy = registry.load_checkpoint(directory, check_schema=False).get('policy')
            parsed = _parsed(str(policy)) if policy else None
            return ('run', parsed) if parsed else ('lost', 'no run.json and no policy it was built from')
        record = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
        if 'parent' in record:                          # round 40 on: an identity, null from scratch
            parent = record['parent']
            if parent is None:
                return 'scratch', None
            kept = self.by_sha1.get(parent.get('sha1'))
            if kept is not None:                        # the backup's path for that very file
                kept_family, kept_run = kept.rsplit('/', 1)
                return 'run', (kept_family, kept_run.split('@')[0])
            return 'run', (parent['family'], parent['run'])
        args = record.get('args') or {}
        source = record.get('source') or args.get('load') or args.get('warm_start') or args.get('init')
        if not source:
            return 'scratch', None
        parsed = _parsed(str(source))
        if parsed is None:
            return 'lost', f'loaded {source!r}: a promoted run then, no run id'
        return 'run', parsed


def _parsed(source: str) -> Optional[tuple]:
    """(family, run) of a checkpoint path or run dir (``.../models/<family>/<run>[/checkpoints/itN][/ckpt.pt]``);
    None for a bare name (a family's promoted run, resolved when the run started)."""
    parts = [p for p in source.rstrip('/').split('/') if p]
    if parts and parts[-1].endswith('.pt'):
        parts = parts[:-1]
    if len(parts) >= 2 and parts[-2] == registry.SNAPSHOTS_NAME:
        parts = parts[:-2]
    if len(parts) < 2:
        return None
    if 'models' in parts[:-1]:
        after = parts[len(parts) - 1 - parts[::-1].index('models') + 1:]
        if len(after) >= 2:
            return '/'.join(after[:-1]), after[-1].split('@')[0]
    return parts[-2], parts[-1].split('@')[0]


def chain(lineage: Lineage, family: str, run: str) -> tuple:
    """([(family, run), ...] from the run to its origin, origin): 'scratch', or 'lost: <why>'."""
    links, seen = [(family, run)], {(family, run)}
    while True:
        kind, value = lineage.parent(*links[-1])
        if kind == 'scratch':
            return links, 'scratch'
        if kind == 'lost':
            return links, f'lost: {value}'
        if value == links[-1]:
            return links, 'lost: continued in its own run dir, its first parent overwritten'
        if value in seen:
            return links, f'lost: a loop at {value[0]}/{value[1]}'
        seen.add(value)
        links.append(value)


def pointers(models_dir=None, route: Optional[str] = None) -> list:
    """(family, pointer file name, run) of every promoted pointer in the models dir (on ``route``: ``promoted`` and
    the route's own ``promoted.<route>`` and ``late.<route>``)."""
    found = []
    root = registry.models_root(models_dir)
    for family in sorted(p for p in root.iterdir() if p.is_dir()):
        for pointer in sorted(family.iterdir()):
            name = pointer.name
            if not pointer.is_file() or pointer.suffix == '.tmp':
                continue
            if name != registry.PROMOTED_NAME and not any(name.startswith(p) for p in POINTERS[1:]):
                continue
            if route is not None and name not in (registry.PROMOTED_NAME, f'{registry.PROMOTED_NAME}.{route}',
                                                  f'{registry.LATE_NAME}.{route}'):
                continue
            found.append((family.name, name, pointer.read_text(encoding='utf-8').strip()))
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='triforce lineage', description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('--models-dir', default=None)
    parser.add_argument('--route', default=None, help="only promoted and this route's promoted.<route>, late.<route>")
    parser.add_argument('--backup', type=Path, default=BACKUP_RUNS, help=f"kept run dirs (default {BACKUP_RUNS})")
    parser.add_argument('--json', action='store_true', help="one JSON line a pointer")
    args = parser.parse_args(argv)
    lineage = Lineage(args.models_dir, args.backup)
    counts = {'pointers': 0, 'scratch': 0, 'traced': 0, 'lost': 0, 'deepest': 0}
    for family, name, run in pointers(args.models_dir, args.route):
        links, origin = chain(lineage, family, run)
        counts['pointers'] += 1
        depth = len(links) - 1 + (origin != 'scratch')
        counts['deepest'] = max(counts['deepest'], depth)
        counts['scratch' if len(links) == 1 and origin == 'scratch' else
               'traced' if origin == 'scratch' else 'lost'] += 1
        if args.json:
            print(json.dumps({'family': family, 'pointer': name, 'chain': [f'{f}/{r}' for f, r in links],
                              'origin': origin}))
        else:
            text = ' <- '.join(r if f == family else f'{f}/{r}' for f, r in links)
            print(f"{family:20} {name:18} {text} <- {'(scratch)' if origin == 'scratch' else '? (' + origin + ')'}")
    print(f"{counts['pointers']} pointers: {counts['scratch']} trained from scratch, "
          f"{counts['traced'] + counts['lost']} continuations ({counts['traced']} traced to a from-scratch origin, "
          f"{counts['lost']} with the origin lost), up to {counts['deepest']} deep")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
