"""A run's manifest (round 40, the lead's design): one resolved description of everything a run consumed, built
before it starts and written under the key ``manifest`` into a training's ``run.json`` and a chain's / evaluation's
``<output>.meta.json``.  ``diff(a, b)`` lists every difference between two manifests; ``triforce manifest diff``
groups them (``group``).

What it records, each by content and never by an absolute path alone (``label`` writes a location relative to the
checkout, the main checkout or the home directory):

- ``code``: the commit, ``git describe``, the tracked files that differ from HEAD (``dirty``), the branch, the
  version, and the built module (``triforce.build``'s stamp: the fingerprint of the sources it was built from, the
  module's SHA-1, ``problem`` when it is not the build of those sources);
- ``rom``: the ROM's SHA-1;
- ``task`` (a training, an evaluation): the family, the yaml (label and SHA-256) and the effective config after
  defaults (and its SHA-256);
- ``data``: every data set by where it came from, how many files and a content hash (``tree``): the seed states per
  ``start_states`` glob, the harvested states (with the fraction drawn), the curriculum's tracks (with the declared
  ``untracked``), demonstration files; a chain's start state;
- ``networks``: per policy key the network it loads: family, run, checkpoint SHA-1, iterations, architecture,
  whether it has a hit head, its act mode, and where it was read from (``models``: today the legacy models dir,
  recorded as ``legacy``);
- ``arguments`` (the command line, paths labelled), ``seeds`` and ``job`` (id, agent, task, subtask).

Hashing is by content: zol@d3first's 3,396 harvested states (38 MB) take 0.17 s, its curriculum (3,336 files)
0.29 s, wizzrobe's (17,410 files, 200 MB) 2.1 s with a warm page cache (Nov 23)."""

import hashlib
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from . import __version__, jobs
from . import build as module_build
from .paths import ROOT

SCHEMA = 1
GROUPS = ('code', 'task', 'data', 'networks', 'arguments')       # what ``group`` sorts a difference into
RUN_FIELDS = ('seeds', 'job', 'kind', 'schema', 'output')         # differ from run to run of one recipe


# ------------------------------------------------------------------------------------------------- locations

@lru_cache(maxsize=1)
def checkouts() -> tuple:
    """Every checkout of this repository (``git worktree list``), the deepest first, this one among them."""
    try:
        text = subprocess.run(['git', '-C', str(ROOT), 'worktree', 'list', '--porcelain'], capture_output=True,
                              text=True, check=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        text = ''
    roots = {ROOT} | {Path(line[len('worktree '):]) for line in text.splitlines() if line.startswith('worktree ')}
    return tuple(sorted(roots, key=lambda p: -len(p.parts)))


def label(path) -> Optional[str]:
    """``path`` as a manifest writes it: relative to the checkout of this repository it lies in (the main one or a
    worktree: ``triforce/states/harvested/zol``, so two arms in two worktrees compare), else ``~/<rel>`` under the
    home directory, else as given (None stays None)."""
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    for base in checkouts():
        if resolved.is_relative_to(base):
            return str(resolved.relative_to(base))
    if resolved.is_relative_to(Path.home()):
        return '~/' + str(resolved.relative_to(Path.home()))
    return str(path)


# ---------------------------------------------------------------------------------------------------- digests

def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def tree(paths: Iterable, base=None) -> dict:
    """``{'count', 'sha256'}`` of files: SHA-256 over each one's name (relative to ``base``, else its file name) and
    the SHA-256 of its content, in name order."""
    named = sorted((str(Path(p).relative_to(base)) if base is not None else Path(p).name, Path(p)) for p in paths)
    digest = hashlib.sha256()
    for name, path in named:
        digest.update(name.encode())
        digest.update(bytes.fromhex(file_sha256(path)))
    return {'count': len(named), 'sha256': digest.hexdigest()}


# ------------------------------------------------------------------------------------------------- the parts

def _git(*args) -> Optional[str]:
    try:
        out = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def code() -> dict:
    """The code that runs: commit, describe, the tracked files that differ from HEAD, branch, version, module."""
    dirty = _git('diff', '--name-only', 'HEAD')
    stamp = module_build.read_stamp(ROOT) or {}
    return {'commit': _git('rev-parse', 'HEAD'), 'describe': _git('describe', '--always', '--dirty'),
            'dirty': sorted(dirty.splitlines()) if dirty else [], 'branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
            'version': __version__,
            'module': {'sources': stamp.get('sources'), 'sha1': stamp.get('module_sha1'), 'built': stamp.get('built'),
                       'problem': module_build.problem(ROOT)}}


def rom() -> dict:
    from . import _native  # pylint: disable=import-outside-toplevel
    path = Path(_native.paths()['rom'])
    digest = hashlib.sha1(path.read_bytes()).hexdigest()
    return {'sha1': digest, 'file': label(path)}


def job() -> Optional[dict]:
    """The queue's job this process runs in (None outside one)."""
    job_id = jobs.job_id()
    if not job_id:
        return None
    from . import jobqueue  # pylint: disable=import-outside-toplevel
    con = jobqueue.connect()
    try:
        rows = jobqueue.read_jobs(con, 'id = ?', (job_id,))
    finally:
        con.close()
    row = rows[0] if rows else {}
    return {'id': job_id, 'agent': row.get('agent'), 'task': row.get('task'), 'subtask': row.get('subtask')}


def task(config, path=None) -> dict:
    """The task: family, yaml (label, SHA-256), its ``extends`` chain (``[{'yaml', 'sha256'}]``, the base first, and
    the keys this file sets over it, ``overrides``) and the effective config after defaults (its SHA-256)."""
    import yaml as pyyaml  # pylint: disable=import-outside-toplevel
    effective = config.to_dict()
    text = json.dumps(effective, sort_keys=True)
    file = Path(path) if path is not None and str(path).endswith('.yaml') else None
    if file is None:
        from .paths import SPECIALISTS_DIR  # pylint: disable=import-outside-toplevel
        file = SPECIALISTS_DIR / f'{config.name}.yaml'
    out = {'family': config.name, 'yaml': label(file), 'yaml_sha256': file_sha256(file) if file.is_file() else None,
           'effective': effective, 'effective_sha256': hashlib.sha256(text.encode()).hexdigest()}
    own = pyyaml.safe_load(file.read_text(encoding='utf-8')) if file.is_file() else None
    if isinstance(own, dict) and 'extends' in own:
        chain, current = [], file
        while True:
            values = pyyaml.safe_load(current.read_text(encoding='utf-8'))
            if not isinstance(values, dict) or 'extends' not in values or len(chain) > 16:
                break
            current = current.parent / values['extends']
            chain.insert(0, {'yaml': label(current), 'sha256': file_sha256(current)})
        out['extends'] = chain
        out['overrides'] = {key: value for key, value in own.items() if key != 'extends'}
    return out


def data(config, states_dir=None, trajectories=None, start_state=None, demos=(), demo_groups=None) -> dict:
    """The data sets a training of ``config`` draws: seed states per glob, harvested states, curriculum tracks,
    demonstration files (``start_state``: a pinned start, the only state drawn; ``demo_groups``: one group a
    demonstration file, each entry's ``group``, when the rows are drawn by group: ``demos.recipe``)."""
    from . import _native  # pylint: disable=import-outside-toplevel
    from .inputs import untracked  # pylint: disable=import-outside-toplevel
    base = Path(states_dir or _native.paths()['states_dir'])
    out: dict = {'states_dir': label(base)}
    if start_state is not None:
        path = Path(start_state) if str(start_state).endswith('.state') else base / f'{start_state}.state'
        out['start_state'] = {'file': label(path), **tree([path])}
    else:
        patterns = {pattern: tree(config.replace(start_states=[pattern]).resolve_start_states(states_dir))
                    for pattern in config.start_states}
        out['start_states'] = {'patterns': patterns, **tree(config.resolve_start_states(states_dir))}
        out['harvested'] = {'dir': label(config.harvested_dir(states_dir)), 'fraction': config.harvested_fraction,
                            **tree(config.resolve_harvested_states(states_dir))}
        if trajectories is not None:
            folder = Path(trajectories)
            files = [p for p in folder.rglob('*') if p.is_file()]
            out['curriculum'] = {'dir': label(folder), 'tracks': len(list(folder.glob('*/trajectory.json'))),
                                 'untracked': untracked(config), **tree(files, folder)}
    if demos:
        out['demos'] = demo_entries(demos, demo_groups)
    return out


def demo_entries(demos, groups=None) -> list[dict]:
    """``data.demos``: each demonstration file's label and hash, and its group when the rows are drawn by group (no
    ``group`` key otherwise: a recipe without groups reads as before)."""
    return [{'file': label(p), 'sha256': file_sha256(p), **({} if groups is None else {'group': groups[i]})}
            for i, p in enumerate(demos)]


def demo_groups(data_: dict) -> Optional[list[str]]:
    """The demonstration files' groups a manifest's ``data`` records (``demo_entries``), None when it records none."""
    entries = data_.get('demos') or []
    return [e['group'] for e in entries] if entries and 'group' in entries[0] else None


def network(path, act: Optional[str] = None) -> dict:
    """A checkpoint file or run dir as the manifest names it (``act``: the act mode it plays by, None: unknown)."""
    from . import registry  # pylint: disable=import-outside-toplevel
    from .provenance import has_hit_head  # pylint: disable=import-outside-toplevel
    saved = registry.load_checkpoint(path, check_schema=False)
    config = saved.get('model_config') or {}
    return {'family': saved.get('specialist'), 'run': saved.get('run_id'), 'iterations': saved.get('iterations'),
            'checkpoint_sha1': registry.checkpoint_sha1(path), 'arch': config.get('arch') or 'zelda',
            'hit_head': has_hit_head(saved), 'act': act, 'file': label(path)}


def models(models_dir=None) -> dict:
    """Where the networks were read from: the store's effective team (``registry.team_in_use``: name, overrides,
    content hash, every slot), else the legacy models dir (``registry.models_root``)."""
    from . import registry, store  # pylint: disable=import-outside-toplevel
    team = registry.team_in_use(models_dir)
    why = registry.models_source(models_dir)
    if team is not None:
        return {'kind': 'store', 'team': store.team_record(team), 'why': why}
    return {'kind': 'legacy', 'dir': label(registry.models_root(models_dir)), 'why': why}


def arguments(values: dict) -> dict:
    """A command's parsed arguments with every existing path labelled (``label``)."""
    def one(value):
        if isinstance(value, (list, tuple)):
            return [one(v) for v in value]
        if isinstance(value, str) and os.sep in value and Path(value).expanduser().exists():
            return label(value)
        return value
    return {key: one(value) for key, value in sorted(values.items())}


def build(kind: str, *, arguments_: dict, seeds: Optional[dict] = None, **parts) -> dict:
    """The manifest of a run of ``kind`` (``train``, ``chain``, ``evaluate``): ``code``, ``rom`` and ``job`` of
    this process, the given ``parts`` (``task``, ``data``, ``networks``, ``models``, ...), the arguments and seeds."""
    return {'schema': SCHEMA, 'kind': kind, 'code': code(), 'rom': rom(), **parts,
            'arguments': arguments(arguments_), 'seeds': seeds or {}, 'job': job()}


def team(policies, act: Optional[str] = None) -> dict:
    """A chain's networks (``runner.Policies``) as the manifest names them, per policy key: family, run, iterations,
    checkpoint SHA-1, architecture, hit head, the act mode it plays by (``act``: one for every network, None: each
    one's own) and its file; and per key the SHA-256 of its task config (``configs``)."""
    networks = {}
    for name, mode in zip(policies.names, policies.modes(act)):
        described = policies.described.get(name, {})
        networks[name] = {'family': described.get('family'), 'run': described.get('run'),
                          'iterations': described.get('iterations'),
                          'checkpoint_sha1': described.get('checkpoint_sha1'),
                          'arch': (described.get('architecture') or {}).get('name'),
                          'hit_head': 'hit' in (described.get('outputs') or ()), 'act': mode,
                          'file': label(policies.checkpoints.get(name))}
    configs = {name: hashlib.sha256(config.to_json().encode()).hexdigest()
               for name, config in zip(policies.names, policies.configs)}
    return {'networks': networks, 'configs': configs}


def chain(args, route, policies, act: Optional[str] = None) -> dict:
    """An ``eval-chain``'s manifest: its team (``team``), its start state and game data, its arguments and seeds."""
    from .paths import state_path  # pylint: disable=import-outside-toplevel
    from .provenance import data_hash  # pylint: disable=import-outside-toplevel
    played = team(policies, act)
    data = {'game_data': data_hash()}
    if route.power_on is not None:
        data['power_on'] = route.power_on
    elif route.checkpoint is None:
        path = state_path(route.start_state)
        data['start_state'] = {'file': label(path), **tree([path])}
    return build('chain', arguments_=vars(args), seeds={'first_seed': args.first_seed, 'runs': args.runs},
                 task={'route': route.name, 'configs': played['configs']}, data=data,
                 networks=played['networks'], models=models(args.models_dir))


def same_team(mine: dict, reference: dict, excepted=()) -> tuple[list, list]:
    """(differences, unknown) of two chains' teams: every difference of ``networks`` and ``task.configs`` but those
    of the ``excepted`` keys (a family names its own key and its route keys, ``zol`` also ``zol@d3first``); a value
    a record from before manifests never recorded is unknown."""
    differences, unknown = [], []
    for entry in diff({'networks': mine.get('networks'), 'configs': (mine.get('task') or {}).get('configs')},
                      {'networks': reference.get('networks'),
                       'configs': (reference.get('task') or {}).get('configs')}):
        key = entry['path'].split('.')[1] if '.' in entry['path'] else entry['path']
        if key in excepted or key.split('@')[0] in excepted:
            continue
        legacy = reference.get('schema') == 0 and entry['b'] is None or mine.get('schema') == 0 and entry['a'] is None
        (unknown if legacy else differences).append(entry)
    return differences, unknown


# ------------------------------------------------------------------------------------------------------- diff

def _flat(value, prefix: str = '') -> dict:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(_flat(item, f'{prefix}.{key}' if prefix else str(key)))
        return out          # an empty section is no difference from an absent one
    return {prefix: value}


def diff(a: dict, b: dict) -> list[dict]:
    """Every difference between two manifests: ``{'path': 'data.harvested.count', 'a': ..., 'b': ...}`` (a path
    one side lacks has None there), in path order."""
    flat_a, flat_b = _flat(a or {}), _flat(b or {})
    return [{'path': path, 'a': flat_a.get(path), 'b': flat_b.get(path)}
            for path in sorted(flat_a.keys() | flat_b.keys()) if flat_a.get(path) != flat_b.get(path)]


def group(path: str) -> str:
    """The group a difference's path belongs to (``GROUPS``; ``rom`` is data, the run's own fields ``run``)."""
    head = path.split('.', 1)[0]
    if head == 'rom' or head == 'models':
        return 'data' if head == 'rom' else 'networks'
    return head if head in GROUPS else 'run'


# ---------------------------------------------------------------------------------------- reading what exists

# What a record made before manifests cannot say (``legacy_train``, ``legacy_chain``).
UNKNOWN_TRAIN = ('code', 'rom', 'task.effective', 'task.yaml_sha256', 'data.start_states', 'data.harvested.sha256',
                 'data.curriculum.sha256', 'networks.warm_start.checkpoint_sha1', 'job')
UNKNOWN_CHAIN = ('code.commit', 'code.dirty', 'code.module', 'rom', 'data.start_state.sha256', 'job')


def legacy_train(record: dict) -> dict:
    """What a training's ``run.json`` from before manifests says, in a manifest's shape (``schema`` 0), with the
    parts it cannot say listed in ``unknown``."""
    args = dict(record.get('args') or {})
    harvested = record.get('harvested') or {}
    tracks = record.get('curriculum_tracks')
    specialist = str(args.get('specialist') or record.get('specialist'))
    out = {'schema': 0, 'kind': 'train',
           'task': {'family': record.get('specialist'),
                    'yaml': label(specialist) if specialist.endswith('.yaml') else f'specialists/{specialist}.yaml'},
           'data': {'harvested': {'dir': label(harvested.get('dir')), 'fraction': harvested.get('fraction'),
                                  'count': harvested.get('states')}},
           'networks': {}, 'arguments': arguments(args), 'seeds': {'seed': args.get('seed')},
           'unknown': list(UNKNOWN_TRAIN)}
    if record.get('curriculum'):
        out['data']['curriculum'] = {'dir': label(record['curriculum']), 'tracks': len(tracks or {})}
    if record.get('source'):
        out['networks']['warm_start'] = {'file': label(record['source']),
                                         'start_iteration': record.get('start_iteration')}
    return out


def legacy_chain(provenance: dict) -> dict:
    """What a chain's ``.meta.json`` provenance from before manifests says, in a manifest's shape (``schema`` 0)."""
    networks = {m['name']: {'family': m.get('family'), 'run': m.get('run'), 'checkpoint_sha1': m.get('checkpoint_sha1'),
                            'act': m.get('act'), 'file': m.get('checkpoint')}
                for m in provenance.get('models') or []}
    configs = {m['name']: m.get('config_sha256') for m in provenance.get('models') or []}
    rest = {k: v for k, v in provenance.items() if k not in ('models', 'version', 'code', 'data', 'missing', 'route')}
    return {'schema': 0, 'kind': 'chain', 'code': {'describe': provenance.get('code'),
                                                   'version': provenance.get('version')},
            'task': {'route': provenance.get('route'), 'configs': configs},
            'data': {'game_data': provenance.get('data'), 'start_state': {'file': provenance.get('start_from')}},
            'networks': networks, 'missing': provenance.get('missing'), 'arguments': rest,
            'unknown': list(UNKNOWN_CHAIN)}


def read(source) -> dict:
    """The manifest of a run dir (its ``run.json``), a ``run.json``, a chain output or its ``.meta.json``, an
    evaluation's ``.meta.json``, or a statistics file (``.stats.json``); a record from before manifests is read by
    ``legacy_train`` / ``legacy_chain``.  Raises ``ValueError`` for anything else."""
    path = Path(source)
    if path.is_dir():
        path = path / 'run.json'
    elif not path.name.endswith('.json') and Path(f'{path}.meta.json').is_file():
        path = Path(f'{path}.meta.json')
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError(f"{source}: no run.json, .meta.json or .stats.json to read ({error})") from None
    if record.get('manifest'):
        return record['manifest']
    if 'args' in record and 'specialist' in record:
        return legacy_train(record)
    if 'provenance' in record:
        return legacy_chain(record['provenance'])
    raise ValueError(f"{path}: holds no manifest and is no training's run.json or chain's .meta.json")


# What may differ between two runs of one recipe (``same_recipe``): the run's own fields, its seeds, ids, outputs and
# where it logs, and when its module was built.
RUN_PATHS = ('schema', 'kind', 'seeds', 'job', 'output', 'code.module.built', 'arguments.seed', 'arguments.run_id',
             'arguments.events', 'arguments.logs_dir', 'arguments.models_dir', 'arguments.output',
             'arguments.same_recipe_as', 'arguments.same_team_as', 'arguments.except_', 'arguments.first_seed')


def matches(path: str, key: str) -> bool:
    """``key`` names ``path``: its dotted segments appear in ``path``'s, in order and next to each other
    (``sil_coeff`` names ``arguments.sil_coeff``, ``harvested`` every ``data.harvested.*``)."""
    parts, wanted = path.split('.'), key.split('.')
    return any(parts[i:i + len(wanted)] == wanted for i in range(len(parts) - len(wanted) + 1))


def same_recipe(mine: dict, reference: dict, excepted=()) -> tuple[list, list]:
    """(differences, unknown) of ``mine`` against ``reference``: every difference but the run's own (``RUN_PATHS``)
    and the ``excepted`` keys (``matches``); a difference where a record from before manifests (``schema`` 0) has
    nothing to say (the part it lists in ``unknown``, or a field it never recorded) is unknown, not a difference."""
    differences, unknown = [], []
    for entry in diff({k: v for k, v in mine.items() if k != 'unknown'},
                      {k: v for k, v in reference.items() if k != 'unknown'}):
        path = entry['path']
        if any(path == p or path.startswith(p + '.') for p in RUN_PATHS) or any(matches(path, k) for k in excepted):
            continue
        silent = [side for side, value in (('mine', mine), ('reference', reference)) if value.get('schema') == 0 and (
            any(path == p or path.startswith(p + '.') for p in value.get('unknown') or ())
            or entry['a' if side == 'mine' else 'b'] is None)]
        (unknown if silent else differences).append(entry)
    return differences, unknown


def short(value) -> str:
    """A value on one line, at most 90 characters."""
    text = json.dumps(value, sort_keys=True, default=str) if not isinstance(value, str) else value
    return text if len(text) <= 90 else text[:87] + '...'


def report(a: dict, b: dict, names=('A', 'B')) -> list[str]:
    """``triforce manifest diff``'s lines: every difference by group (``GROUPS``, then ``run``), one a line, then
    what each side cannot say (a record from before manifests)."""
    lines = []
    differences = diff({k: v for k, v in a.items() if k != 'unknown'}, {k: v for k, v in b.items() if k != 'unknown'})
    for name in (*GROUPS, 'run'):
        mine = [d for d in differences if group(d['path']) == name]
        if mine:
            lines.append(f"{name} ({len(mine)}):")
            lines += [f"  {d['path']}: {short(d['a'])} -> {short(d['b'])}" for d in mine]
    for side, manifest in zip(names, (a, b)):
        if manifest.get('unknown'):
            lines.append(f"unknown in {side} (a record from before manifests): {', '.join(manifest['unknown'])}")
    return lines or ['the same']


def main(argv=None) -> int:
    """``triforce manifest diff A B [--json]`` / ``triforce manifest show X``: A, B, X are run dirs, ``run.json``
    files, chain outputs (their ``.meta.json``), evaluations' ``.meta.json`` or statistics files."""
    import argparse  # pylint: disable=import-outside-toplevel
    parser = argparse.ArgumentParser(prog='triforce manifest', description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='verb', required=True)
    compare = sub.add_parser('diff', help="every difference between two runs' manifests, by group")
    compare.add_argument('a')
    compare.add_argument('b')
    compare.add_argument('--json', action='store_true', help="the differences as a JSON list")
    show = sub.add_parser('show', help="one run's manifest as JSON")
    show.add_argument('source')
    args = parser.parse_args(argv)
    try:
        if args.verb == 'show':
            print(json.dumps(read(args.source), indent=1, default=str))
            return 0
        a, b = read(args.a), read(args.b)
    except ValueError as error:
        print(f"triforce manifest: error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(diff(a, b), indent=1, default=str))
    else:
        print(f"A: {args.a}\nB: {args.b}")
        print('\n'.join(report(a, b)))
    return 0
