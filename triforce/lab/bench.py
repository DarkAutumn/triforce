"""The yardstick for experiments (docs/lab.md "The yardstick"): a fixed, cheap suite to compare two networks in
under an hour of the queue, and the manifest of the full tier.

    python -m triforce.lab.bench manifest [--states-per-family 24] [--episodes 5]   # writes bench.json (once)
    python -m triforce.lab.bench run --team --out team.json                          # today's promoted networks
    python -m triforce.lab.bench run --network CKPT --keys zol stalfos-keese-gel --out student.json
    python -m triforce.lab.bench compare team.json student.json
    python -m triforce.lab.bench speed --key zol --out speed.json                    # decisions/s per acting path
    python -m triforce.lab.bench rooms --harvest DIR --chains H-d3.jsonl ... --out triforce/lab/bench-rooms.json

The cheap tier (``bench.json`` beside this file, committed): per policy key of today's team (a family, or
``family@route`` for a route's own run), room entries as the chains bring them (the harvested pre-entry states
under ``<states>/harvested/<family>/``: a family's rooms by entry, the same number of states from each, chosen by a
fixed hash order), else the family's seed start states (``source: seed``: no harvested entries; coverage missing).
Every state is played ``episodes`` times on fixed seeds in the project's act mode, clips on.  Per key: success,
hearts lost per episode, hits taken per room visit, decisions per episode, bombs used per episode; one JSON per
network, every episode kept (state, seed) so ``compare`` pairs them: per key and in total, the paired difference
and its standard error.  The cheap tier plays a harvested entry with the family's ``ram_randomization`` (hearts,
bombs and help drawn at reset, as training does), and its entries are handoffs only (a chain harvests a room entry
where the family changes): it never sees a room entered from a room of the same family, and a key's 120 episodes
weigh as much in the total as any other key's.  Round 38 found a student 1.2 points above the team on it whose
chains lost 17-40 points a dungeon, in exactly those rooms.

The rooms tier (``bench-rooms.json``, ``rooms``): every room entry of the route where the team takes damage, as the
team's chains bring it (an every-room harvest, ``eval-chain --harvest DIR --harvest-every-room``), played as it
arrived (``arrived``: no ``ram_randomization``, the hearts and bombs of the chain), one row per room entry
(``name``: the key, the room, the entry side); ``compare`` lists the rows a network loses on (``losing``).  Its
states live under ``<states>/harvested-rooms/<family>/`` (untracked, like ``harvested``: copy them from main).

The full tier (``bench.json`` ``full``): the segment chains of the route as the landings run them
(``triforce eval-batch TAG --chains TAG --route ROUTE --runs 1000 --act model``, a candidate's networks given with
``--checkpoint``), and the route from the start (the ``full`` segment).
"""

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .. import __version__, _native, registry
from ..eval_batch import route_segments
from ..evaluate import evaluate_specialist
from ..events import progress_job, write_json_atomic
from ..jobs import add_job_arguments, begin_job, work
from ..specialist import load_specialist

MANIFEST = Path(__file__).with_name('bench.json')
FORMAT = 1
ACT = registry.DEFAULT_ACT
ROUTE = 'd3first'
HARVESTED = 'harvested'
ROOMS = 'harvested-rooms'
ROOMS_MANIFEST = Path(__file__).with_name('bench-rooms.json')
_ENTRY = re.compile(r'^(.*)_h\d+_\d+$')


def team_keys(root=None) -> list[str]:
    """Every policy key of today's team: each family with a promoted run, and ``family@route`` for each of its
    ``promoted.<route>`` on a route with its own runs."""
    keys = []
    for family in sorted(_native.families()):
        try:
            registry.promoted_run(family, root)
        except FileNotFoundError:
            continue
        keys.append(family)
        for route in registry.route_promotions(family, root):
            if registry.own_models_until(route) is not None:
                keys.append(registry.route_policy(family, route))
    return keys


def _order(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def choose_states(family: str, count: int, states_dir: Path, route: str = ROUTE) -> tuple[list[str], str]:
    """(state names relative to ``states_dir``, source) of ``family``: harvested entries spread evenly over its
    room entries (the route's own, ``<route>_*``, where it has them), else its seed start states."""
    directory = states_dir / HARVESTED / family
    files = sorted(p.stem for p in directory.glob('*.state')) if directory.is_dir() else []
    source = HARVESTED
    if not files:
        config = load_specialist(family)
        files = sorted(Path(p).stem for p in config.resolve_start_states(str(states_dir)))
        source = 'seed'
    routed = [name for name in files if name.startswith(f'{route}_')]
    files = routed or files     # the route's own states where it has them (the others are older routes')
    groups = defaultdict(list)
    for name in files:
        match = _ENTRY.match(name)
        groups[match.group(1) if match else name.rsplit('_', 1)[0]].append(name)
    for names in groups.values():
        names.sort(key=_order)
    chosen, depth = [], 0
    while len(chosen) < count and any(len(names) > depth for names in groups.values()):
        for entry in sorted(groups):
            if depth < len(groups[entry]) and len(chosen) < count:
                chosen.append(groups[entry][depth])
        depth += 1
    prefix = f'{HARVESTED}/{family}/' if source == HARVESTED else ''
    return sorted(prefix + name for name in chosen), source


def make_manifest(states_per_family: int, episodes: int, seed: int, root=None, states_dir=None,
                  route: str = ROUTE) -> dict:
    """The cheap tier's manifest from the team in ``root`` and the states (with their harvests) in ``states_dir``,
    preferring ``route``'s own entries; regenerate it when a route's states or harvests change (``manifest --route R
    --states-dir DIR --models-dir DIR``)."""
    states_dir = Path(states_dir or _native.paths()['states_dir'])
    entries = []
    for key in team_keys(root):
        family = registry.split_policy(key)[0]
        states, source = choose_states(family, states_per_family, states_dir, route)
        entries.append({'key': key, 'family': family, 'source': source, 'states': states})
    segments = route_segments(route).segments.split()
    return {'format': FORMAT, 'act': ACT, 'episodes': episodes, 'seed': seed, 'clips': 'on',
            'states_per_family': states_per_family, 'entries': entries,
            'missing_coverage': sorted({e['family'] for e in entries if e['source'] != HARVESTED}),
            'full': {'route': route, 'runs': 1000, 'act': 'model', 'segments': segments,
                     'command': f'triforce eval-batch TAG --chains TAG --route {route} --runs 1000 --act model '
                                '[--checkpoint KEY=CKPT ...]'}}


def load_manifest(path=None) -> dict:
    return json.loads(Path(path or MANIFEST).read_text(encoding='utf-8'))


def _episode(record: dict) -> dict:
    visits = record.get('visits') or []
    return {'state': Path(str(record['start_state'])).stem,
            'seed': record['seed'], 'success': bool(record['success']),
            'hearts_lost': float(record['start_health']) - float(record['health']),
            'hits': sum(int(v.get('hits', 0)) for v in visits), 'visits': len(visits),
            'decisions': int(record['steps']), 'bombs': int((record.get('bombs') or {}).get('used', 0))}


def summarize(episodes: list[dict]) -> dict:
    n = len(episodes)
    visits = sum(e['visits'] for e in episodes)
    return {'episodes': n,
            'success': sum(e['success'] for e in episodes) / n if n else 0.0,
            'hearts_lost': sum(e['hearts_lost'] for e in episodes) / n if n else 0.0,
            'hits_per_visit': sum(e['hits'] for e in episodes) / visits if visits else 0.0,
            'decisions': sum(e['decisions'] for e in episodes) / n if n else 0.0,
            'bombs': sum(e['bombs'] for e in episodes) / n if n else 0.0}


def entry_task(entry: dict, states_dir=None):
    """(config, states dir) that play a manifest entry's states: a harvested entry's from its family's folder under
    the entry's source (``harvested``, ``harvested-rooms``; start states resolve by bare name in one folder), seed
    states from the states dir; an ``arrived`` entry with the family's ``ram_randomization`` off."""
    base = Path(states_dir or _native.paths()['states_dir'])
    names = [Path(state).name for state in entry['states']]
    if entry['source'] != 'seed':
        base = base / entry['source'] / entry['family']
    config = load_specialist(entry['family']).replace(start_states=names)
    if entry.get('arrived'):
        config = config.replace(ram_randomization={})
    return config, str(base)


def run_bench(networks: dict[str, str], manifest: dict, parallel: int, states_dir=None, act: Optional[str] = None,
              log=print, progress=None) -> dict:
    """Plays every manifest entry whose key ``networks`` names (key -> checkpoint), one row per entry (its
    ``name``, else its key); ``progress`` (an ``events.JobProgress``) hears the episodes played and their success
    after each entry."""
    rows, total_seconds, decisions = {}, 0.0, 0
    for entry in manifest['entries']:
        key = entry['key']
        if key not in networks:
            continue
        label = entry.get('name', key)
        config, directory = entry_task(entry, states_dir)
        began = time.time()
        result = evaluate_specialist(config, networks[key], episodes_per_state=manifest['episodes'],
                                     parallel=parallel, seed=manifest['seed'], act=act or manifest['act'],
                                     states_dir=directory, clips=manifest['clips'])
        seconds = time.time() - began
        episodes = [_episode(r) for r in result['records']]
        rows[label] = {**summarize(episodes), 'checkpoint': str(networks[key]), 'model_hash': result['model_hash'],
                       'seconds': round(seconds, 1), 'source': entry['source'], 'records': episodes}
        total_seconds += seconds
        decisions += sum(e['decisions'] for e in episodes)
        row = rows[label]
        log(f"[bench] {label}: success {row['success']:.3f} hearts lost {row['hearts_lost']:.2f} "
            f"hits/visit {row['hits_per_visit']:.3f} decisions {row['decisions']:.0f} ({seconds:.0f}s)")
        if progress is not None:
            played = [e for r in rows.values() for e in r['records']]
            progress.update(len(played), summarize(played)['success'], key=label)
    every = [e for row in rows.values() for e in row['records']]
    return {'format': FORMAT, 'version': __version__, 'act': act or manifest['act'], 'manifest': _manifest_hash(manifest),
            'parallel': parallel, 'seconds': round(total_seconds, 1),
            'decisions_per_second': decisions / total_seconds if total_seconds else 0.0,
            'total': summarize(every), 'keys': rows}


def _manifest_hash(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]


METRICS = ('success', 'hearts_lost', 'hits', 'decisions', 'bombs')
FIELDS = ('state', 'seed', 'success', 'hearts_lost', 'hits', 'visits', 'decisions', 'bombs')


def compact(result: dict) -> dict:
    """``result`` with every key's episodes as one table (``fields``, ``rows``): what ``run`` writes."""
    keys = {key: {**row, 'records': {'fields': list(FIELDS),
                                     'rows': [[e[f] for f in FIELDS] for e in records(row)]}}
            for key, row in result['keys'].items()}
    return {**result, 'keys': keys}


def records(row: dict) -> list[dict]:
    """A key's episodes as dicts, from either form (``compact``)."""
    kept = row['records']
    return kept if isinstance(kept, list) else [dict(zip(kept['fields'], values)) for values in kept['rows']]


def paired(a: list[dict], b: list[dict]) -> dict:
    """Per metric, the mean of b - a over the episodes both played (by state and seed) and its standard error."""
    index = {(e['state'], e['seed']): e for e in a}
    pairs = [(index[(e['state'], e['seed'])], e) for e in b if (e['state'], e['seed']) in index]
    out = {'pairs': len(pairs)}
    for metric in METRICS:
        diffs = [float(y[metric]) - float(x[metric]) for x, y in pairs]
        n = len(diffs)
        mean = sum(diffs) / n if n else 0.0
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
        out[metric] = (mean, math.sqrt(var / n) if n else 0.0)
    return out


LOSING_POINTS = 0.05       # a row losing this much success, beyond two standard errors, is listed under the table


def compare(base: dict, new: dict) -> dict:
    """Per row and in total, ``new - base`` paired by episode; ``losing``: the rows whose success falls by
    ``LOSING_POINTS`` or more and by more than two standard errors, worst first (a total averages them away)."""
    keys = [k for k in base['keys'] if k in new['keys']]
    rows = {k: paired(records(base['keys'][k]), records(new['keys'][k])) for k in keys}
    every_a = [dict(e, state=f"{k}/{e['state']}") for k in keys for e in records(base['keys'][k])]
    every_b = [dict(e, state=f"{k}/{e['state']}") for k in keys for e in records(new['keys'][k])]
    losing = sorted((k for k, r in rows.items()
                     if r['success'][0] <= -LOSING_POINTS and r['success'][0] < -2 * r['success'][1]),
                    key=lambda k: rows[k]['success'][0])
    return {'keys': rows, 'total': paired(every_a, every_b), 'losing': losing}


def format_compare(result: dict) -> str:
    lines = ['| key | pairs | success | hearts lost | hits | decisions | bombs |', '|---|---:|---:|---:|---:|---:|---:|']
    for key, row in [*result['keys'].items(), ('**total**', result['total'])]:
        cells = [f"{row[m][0]:+.3f} ± {row[m][1]:.3f}" for m in METRICS]
        lines.append(f"| {key} | {row['pairs']} | " + ' | '.join(cells) + ' |')
    losing = result.get('losing', [])
    lines.append('')
    lines.append(f"**Losing** ({len(losing)} rows, success down {LOSING_POINTS:.0%} or more beyond 2 SE): "
                 + ('; '.join(f"{k} {result['keys'][k]['success'][0]:+.3f} ± {result['keys'][k]['success'][1]:.3f}"
                              for k in losing) or 'none'))
    return '\n'.join(lines)


def _harvest_runs(chains: list[str]) -> dict:
    """(seed, room index) -> that room's record, from the JSON-lines outputs of the harvesting chains."""
    out = {}
    for path in chains:
        with open(path, encoding='utf-8') as lines:
            for line in lines:
                run = json.loads(line)
                for room in run['rooms']:
                    out[(int(run['seed']), int(room['index']))] = room
    return out


def make_rooms_manifest(harvest: Path, chains: list[str], states_per_room: int, episodes: int, seed: int,
                        min_hearts: float, states_dir=None, route: str = ROUTE) -> dict:
    """The rooms tier from an every-room harvest of the team (``harvest``: ``<family>/<stem>_h<seed>_<n>.state``)
    and the chains that harvested it (their records tell what each harvested visit cost the team): one entry per
    room entry (stem) whose visits lost ``min_hearts`` or more a visit or any run, keyed by the policy that plays
    its route step, ``states_per_room`` states by the fixed hash order.  The chosen states are copied under
    ``<states>/harvested-rooms/<family>/``."""
    states_dir = Path(states_dir or _native.paths()['states_dir'])
    visits = _harvest_runs(chains)
    steps: dict = {}
    for step in _native.route_steps(route):
        steps.setdefault((step['level'], step['room'], bool(step['in_cave']), step['entry'], step['visit']),
                         step['index'])
    entries = []
    for folder in sorted(p for p in harvest.iterdir() if p.is_dir()):
        family = folder.name
        groups = defaultdict(list)
        for path in folder.glob(f'{route}_*.state'):
            match = _ENTRY.match(path.stem)
            if match:
                groups[match.group(1)].append(path)
        for stem, paths in sorted(groups.items()):
            rooms = [visits.get(tuple(int(x) for x in re.findall(r'_h(\d+)_(\d+)$', p.stem)[0])) for p in paths]
            rooms = [r for r in rooms if r is not None]
            if not rooms:
                continue
            lost = sum(max(0.0, r['health_in'] - r['health_out']) for r in rooms) / len(rooms)
            deaths = sum(r['outcome'] == 'death' for r in rooms)
            if lost < min_hearts and not deaths:
                continue
            parsed = _native.state_name(paths[0].stem)
            step = steps.get(tuple(parsed)) if parsed is not None else None
            key = _native.step_policy(route, step, family) if step is not None else family
            chosen = sorted(paths, key=lambda p: _order(p.stem))[:states_per_room]
            target = states_dir / ROOMS / family
            target.mkdir(parents=True, exist_ok=True)
            for path in chosen:
                shutil.copyfile(path, target / path.name)
            entries.append({'key': key, 'family': family, 'source': ROOMS, 'arrived': True,
                            'name': f"{key} {stem.removeprefix(f'{route}_')}",
                            'team_visits': len(rooms), 'team_hearts_lost': round(lost, 3), 'team_deaths': deaths,
                            'states': sorted(f'{ROOMS}/{family}/{p.stem}' for p in chosen)})
    return {'format': FORMAT, 'act': ACT, 'episodes': episodes, 'seed': seed, 'clips': 'on', 'route': route,
            'states_per_room': states_per_room, 'min_hearts': min_hearts, 'entries': entries}


def speed(key: str, manifest: dict, parallel: int, devices: list[str], states_dir=None, root=None,
          checkpoint=None) -> dict:
    """Decisions per second of one key's entry played by its network (or ``checkpoint``) through each acting path:
    the Rust actor (``rust``) and the same network served by torch (``cpu``, ``cuda``, ``cuda-half``: autocast
    float16; ``nets.serve``)."""
    from ..nets import serve  # pylint: disable=import-outside-toplevel
    checkpoint = checkpoint or registry.policy_checkpoint(key, root)
    entry = next(e for e in manifest['entries'] if e['key'] == key)
    config, directory = entry_task(entry, states_dir)
    out = {}
    for path in devices:
        network = registry.load_model(checkpoint)[0]
        if path == 'rust':
            policy = network.to_native()
        else:
            device, _, half = path.partition('-')
            policy = serve.Server(device=device, half=half == 'half').policy(network)
        began = time.time()
        result = _play_policy(config, policy, manifest, parallel, directory)
        seconds = time.time() - began
        decisions = sum(r['steps'] for r in result)
        out[path] = {'decisions': decisions, 'seconds': round(seconds, 2), 'decisions_per_second': decisions / seconds,
                     'success': sum(r['success'] for r in result) / len(result)}
    return {'key': key, 'checkpoint': str(checkpoint), 'parallel': parallel, 'paths': out}


def _play_policy(config, policy, manifest: dict, parallel: int, states_dir) -> list[dict]:
    from ..clips import episode_jobs  # pylint: disable=import-outside-toplevel
    from ..evaluate import _play  # pylint: disable=import-outside-toplevel
    jobs = episode_jobs(config, manifest['episodes'], manifest['seed'], manifest['clips'],
                        None if states_dir is None else str(states_dir))
    task = _native.EvalTask(config, jobs, states_dir=None if states_dir is None else str(states_dir))
    with _native.EnvPool(task, num_envs=max(1, min(parallel, len(jobs))), threads=parallel, seed=0) as pool:
        return _play(pool, policy, manifest['act'], len(jobs), None, None, False)


def main(argv=None) -> int:
    # pylint: disable=too-many-locals
    parser = argparse.ArgumentParser(prog='python -m triforce.lab.bench', description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    m = sub.add_parser('manifest')
    m.add_argument('--states-per-family', type=int, default=24)
    m.add_argument('--episodes', type=int, default=5)
    m.add_argument('--seed', type=int, default=360000)
    m.add_argument('--out', default=str(MANIFEST))
    m.add_argument('--route', default=ROUTE, help="the route whose own room entries (<route>_*) are preferred and "
                                                  "whose segments make the full tier")
    r = sub.add_parser('run')
    r.add_argument('--team', action='store_true', help="every key's network of today (policy_checkpoint)")
    r.add_argument('--network', default=None, help="one checkpoint for every key in --keys")
    r.add_argument('--keys', nargs='*', default=None)
    r.add_argument('--checkpoint', nargs='*', default=[], metavar='KEY=CKPT')
    r.add_argument('--out', required=True)
    r.add_argument('--act', default=None,
                   help="another act mode than the manifest's (e.g. the planner beside a result: "
                        "temperature:0.5+relax:16+plan:16:1; served networks refuse foresight)")
    s = sub.add_parser('speed')
    s.add_argument('--key', default='zol')
    s.add_argument('--paths', nargs='+', default=['rust', 'cpu', 'cuda'],
                   choices=['rust', 'cpu', 'cuda', 'cuda-half'])
    s.add_argument('--network', default=None, help="a checkpoint to time instead of the key's network")
    s.add_argument('--out', required=True)
    for p in (m, r, s):
        p.add_argument('--models-dir', default=None)
        p.add_argument('--states-dir', default=None)
    for p in (r, s):
        p.add_argument('--parallel', type=int, default=10)
        p.add_argument('--manifest', default=str(MANIFEST))
        p.add_argument('--logs-dir', default=None, help="the log folder's root (default logs/, $TRIFORCE_LOGS_DIR)")
        add_job_arguments(p)
    c = sub.add_parser('compare')
    c.add_argument('base')
    c.add_argument('new')
    o = sub.add_parser('rooms', help="the rooms tier's manifest from an every-room harvest of the team")
    o.add_argument('--harvest', required=True, help="eval-chain --harvest DIR --harvest-every-room's DIR")
    o.add_argument('--chains', nargs='+', required=True, help="the JSON-lines outputs of those harvesting chains")
    o.add_argument('--states-per-room', type=int, default=12)
    o.add_argument('--episodes', type=int, default=5)
    o.add_argument('--seed', type=int, default=380000)
    o.add_argument('--min-hearts', type=float, default=0.1,
                   help="a room entry joins when the team lost this many hearts a visit there (or a run)")
    o.add_argument('--states-dir', default=None)
    o.add_argument('--out', default=str(ROOMS_MANIFEST))
    args = parser.parse_args(argv)
    if args.command == 'manifest':
        manifest = make_manifest(args.states_per_family, args.episodes, args.seed, args.models_dir, args.states_dir,
                                 args.route)
        write_json_atomic(args.out, manifest)
        print(f"[bench] {len(manifest['entries'])} keys, "
              f"{sum(len(e['states']) for e in manifest['entries']) * manifest['episodes']} episodes; "
              f"no harvested entries: {manifest['missing_coverage']}")
        return 0
    if args.command == 'rooms':
        manifest = make_rooms_manifest(Path(args.harvest), args.chains, args.states_per_room, args.episodes,
                                       args.seed, args.min_hearts, args.states_dir)
        write_json_atomic(args.out, manifest)
        print(f"[bench] rooms: {len(manifest['entries'])} room entries, "
              f"{sum(len(e['states']) for e in manifest['entries']) * manifest['episodes']} episodes")
        return 0
    if args.command == 'compare':
        result = compare(json.loads(Path(args.base).read_text()), json.loads(Path(args.new).read_text()))
        print(format_compare(result))
        return 0
    begin_job(args)
    manifest = load_manifest(args.manifest)
    if args.command == 'speed':
        with progress_job('lab-speed', 'eval', len(args.paths), unit='acting paths', root=args.logs_dir,
                          work=work(eval_episodes=100)) as progress:
            result = speed(args.key, manifest, args.parallel, args.paths, args.states_dir, args.models_dir,
                           args.network)
            progress.update(len(args.paths))
        write_json_atomic(args.out, result)
        print(json.dumps(result, indent=2))
        return 0
    networks: dict[str, str] = {}
    if args.team:
        for entry in manifest['entries']:
            networks[entry['key']] = str(registry.policy_checkpoint(entry['key'], args.models_dir))
    if args.network:
        for key in args.keys or [e['key'] for e in manifest['entries']]:
            networks[key] = args.network
    networks.update(item.split('=', 1) for item in args.checkpoint)
    episodes = sum(len(e['states']) * manifest['episodes'] for e in manifest['entries'] if e['key'] in networks)
    with progress_job('lab-bench', 'eval', episodes, root=args.logs_dir,
                      work=work(eval_episodes=episodes)) as progress:
        result = run_bench(networks, manifest, args.parallel, args.states_dir, act=args.act, progress=progress)
    write_json_atomic(args.out, compact(result))
    print(json.dumps(result['total'], indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
