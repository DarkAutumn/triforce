"""Stage 2's recording (rounds 36-38; round 40's generation, docs/lab.md "Stage 2: one network from its teachers"):
teacher-labelled data for every policy key a student learns, with the hit labels its hit head trains on.

    python scripts/experiments/r36_merge_record.py --out DIR --group walkers --teacher-group walkers=NAME/RUN \\
        --try-others 8:1.0 [--sources resets entries rooms] [--decisions 20480] [--student CKPT] [--dry-run]
    python scripts/experiments/r36_merge_record.py --out DIR --keys zol@d3first stalfos-keese-gel@d3first \\
        --teacher zol=zol/RUN --teacher stalfos-keese-gel=stalfos-keese-gel/RUN --state-route d3first

Recording is a job of the machine's queue (``jobs.require_job``: ``triforce job run ... --timeout 3h --
.venv/bin/python scripts/experiments/r36_merge_record.py ...``; ``--dry-run`` runs anywhere: the plan, every teacher
resolved and checked); the decisions recorded of the total are its progress (``events.progress_job``).

**The keys (the student's tasks).** ``--keys KEY[=YAML]...`` (a family, or ``family@route``; ``=YAML`` the task file,
else ``lab.collect.task_yaml``: ``specialists/<route>/<family>.yaml`` where there is one, the route's own task and so
its observation switches) and ``--group NAME...`` (every member of a teacher of ``triforce/groups.yaml``,
``lab.groups``; its route is the default ``--state-route``). Neither: every key of today's team (round 38).

**The teachers.** ``--teacher KEY=NET`` names a key's teacher (KEY a policy key, or a family: every requested key of
it) and ``--teacher-group GROUP=NET`` one teacher for every requested member of a group. NET is a store run
``FAMILY/RUN``, a run dir, a ``ckpt.pt``, or a team slot ``family@route[:late]`` (``train.resolve_warm_start``). Once
one teacher is named, every key needs one (a missing teacher is an error, never today's team's in its place). None
named: each key's network of the team in use on ``--state-route`` (``registry.policy_checkpoint``: a bare key's
``late.<route>`` over its ``promoted``), as round 38 recorded. A teacher (and a ``--student``) that reads a block whose
observation switch is off in the task it records on is refused (``lab.recipe.network_reads``: it would read zeros).

Per key and source the key's teacher acting on ``--parallel`` envs (``lab.collect.MultiCollector``, ``--actor
teacher``; with ``--student CKPT`` the student acts and the teacher labels its decisions: a correction round), every
row labelled with the teacher's joint log-probs and value and, with ``--try-others K:P``, the look-ahead's verdicts on
K other candidates on a share P of the decisions (the hit labels of ``--aux hit=hit:C``); the arrays are
``lab.record``'s (``record.shard_arrays``), in ``<out>/<key>/<source>/shard-NNNN.npz`` and ``meta.json``:

- ``resets``: the key's task (``--part-yaml DIR``: ``<key>.yaml`` there instead, round 36's old runs' tasks), its own
  hp and item randomisation, on ``--state-route``'s states alone (``lab.collect.parse_families``);
- ``entries``: the family's harvested pre-entry states as the chains bring them (the route's own, ``<route>_*``),
  less every state the bench plays, no randomisation, on the key's task; each state goes to the key that plays its
  route step (``_native.step_policy``: ``family@d3first`` before the magic sword), or with named teachers, when that
  key is not recorded, to the recorded key of its family (a group teacher plays its family's whole route);
- ``rooms`` (``--harvest DIR``): the same from an every-room harvest (``eval-chain --harvest DIR
  --harvest-every-room``, ``DIR/<family>/``), less both bench tiers' states. With ``--student`` and the student's own
  harvest it is a correction round on the rooms the student brings itself to, at the hearts it brings.

Seeds ``--seed`` + 1000 · the key's index (+ 500 for entries, + 700 for rooms): disjoint from the bench's
(360000 + i). The index is the key's place in the team's keys (round 38's), or in the requested keys (``--group``'s
members first, then ``--keys``) when there are any.

``meta.json``: the key, source, rows, seed, the start state of every episode (``start_states_used``), the teacher
(``teacher``: what was named, ``teacher_network``: run, checkpoint SHA-1, architecture, hit head, file), the student
likewise, the task (``task``: yaml and effective config with their hashes, ``manifest.task``), the observation
switches it recorded with (``obs``), the data it drew (``data``: seed and harvested states by content,
``manifest.data``), ``try_others``, and the code and ROM (``manifest.code``, ``manifest.rom``). With ``--state-route`` a
source with an episode started outside the route's namespace fails before its ``meta.json`` is written
(``foreign_starts``; round 38 found 1,280 such episodes, ``d2first``'s harvest)."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from triforce import _native, manifest, paths, registry
from triforce.events import progress_job
from triforce.inputs import MissingInput
from triforce.jobs import require_job, work
from triforce.lab import groups
from triforce.lab.bench import HARVESTED, ROOMS_MANIFEST, ROUTE, load_manifest, team_keys
from triforce.lab.collect import Family, MultiCollector, Teachers, parse_families, task_yaml
from triforce.lab.recipe import SWITCH_KEYS, network_reads
from triforce.lab.record import shard_arrays
from triforce.lab.train import parse_try_others
from triforce.specialist import load_specialist
from triforce.train import resolve_warm_start

STEPS_PER_ENV = 512


def foreign_starts(used, route: str) -> list:
    """The start states of ``used`` (names or labels) outside ``route``'s start-state namespace, sorted."""
    namespace = _native.route_namespace(route)
    return sorted(s for s in used if _native.state_namespace(s) != namespace)


def step_index() -> dict:
    """(level, room, in_cave, entry, visit) -> the route's step indices with that room entry."""
    index: dict = {}
    for step in _native.route_steps(ROUTE):
        entry = step['entry']
        index.setdefault((step['level'], step['room'], bool(step['in_cave']), entry, step['visit']), []).append(
            step['index'])
    return index


def entry_states(keys: list[str], harvested: Path, route_only_entries: bool = False,
                 whole_route: bool = False) -> dict:
    """key -> state names (bare) in ``harvested/<family>/`` it plays, less both bench tiers';
    ``route_only_entries``: only the route's own (``d3first_*``), never another route's harvest in their place.  A
    state goes to the key that plays its route step, else to the family's bare key; ``whole_route`` (named teachers:
    a key's teacher plays its family's whole route): else to the family's recorded key of the route."""
    bench = {Path(s).name for m in (load_manifest(), load_manifest(ROOMS_MANIFEST))
             for e in m['entries'] for s in e['states']}
    steps = step_index()
    result: dict = {k: [] for k in keys}
    for family in sorted({registry.split_policy(k)[0] for k in keys}):
        folder = harvested / family
        names = sorted(p.stem for p in folder.glob('*.state')) if folder.is_dir() else []
        routed = [n for n in names if n.startswith(f'{ROUTE}_')]
        for name in routed if route_only_entries else (routed or names):
            if name in bench:
                continue
            key = family
            parsed = _native.state_name(name)
            if routed and parsed is not None:
                level, room, cave, entry, visit = parsed
                found = steps.get((level, room, cave, entry, visit))
                if found:
                    key = _native.step_policy(ROUTE, found[0], family)
            if key not in result:
                key = family
            if key not in result and whole_route:
                key = registry.route_policy(family, ROUTE)
            if key in result:
                result[key].append(name)
    return result


def requested_keys(items: list[str], group_names: list[str]) -> tuple[list[str], dict, list]:
    """(keys in order, key -> task yaml named with ``=YAML``, the groups) of ``--keys`` and ``--group``."""
    keys, yamls, chosen = [], {}, []
    for name in group_names:
        group = groups.get(name)
        chosen.append(group)
        keys += [k for k in group.members if k not in keys]
    for item in items:
        key, _, path = item.partition('=')
        if key not in keys:
            keys.append(key)
        if path:
            yamls[key] = path
    return keys, yamls, chosen


def named_teachers(keys: list[str], teacher: list[str], teacher_group: list[str]) -> dict:
    """key -> what names its teacher (``--teacher KEY=NET``: KEY a requested key or a family, every requested key of
    it; ``--teacher-group GROUP=NET``: every requested member of the group).  A key named twice, or a name that
    reaches no requested key, is an error; once one is named, every key needs one."""
    named: dict = {}

    def put(key, value, what):
        if key in named and named[key] != value:
            raise SystemExit(f"{what}: {key} already has the teacher {named[key]}")
        named[key] = value

    for item in teacher:
        name, sep, value = item.partition('=')
        if not sep or not name or not value:
            raise SystemExit(f"--teacher {item!r}: KEY=NET (a policy key or a family; a store run FAMILY/RUN, a run "
                             "dir, a checkpoint or a slot family@route[:late])")
        hit = [k for k in keys if k == name or registry.split_policy(k)[0] == name]
        if not hit:
            raise SystemExit(f"--teacher {item}: {name} is none of the recorded keys {keys}")
        for key in hit:
            put(key, value, f'--teacher {item}')
    for item in teacher_group:
        name, sep, value = item.partition('=')
        if not sep or not name or not value:
            raise SystemExit(f"--teacher-group {item!r}: GROUP=NET")
        hit = [k for k in groups.get(name).members if k in keys]
        if not hit:
            raise SystemExit(f"--teacher-group {item}: no member of {name} is recorded ({keys})")
        for key in hit:
            put(key, value, f'--teacher-group {item}')
    missing = [k for k in keys if k not in named]
    if named and missing:
        raise SystemExit(f"no teacher named for {missing}: name one for every key (--teacher KEY=NET, "
                         f"--teacher-group GROUP=NET), never today's team's in its place")
    return named


def resolve(value: str, root) -> Path:
    """The checkpoint ``value`` names (``train.resolve_warm_start``)."""
    try:
        return Path(resolve_warm_start(value, root))
    except MissingInput as error:
        raise SystemExit(str(error)) from None


def switches_off(network, config) -> list[str]:
    """The observation switches whose blocks ``network`` reads (``lab.recipe.network_reads``) and ``config`` leaves
    off (the blocks would be zeros), as ``obs.<switch> (<keys>)``."""
    reads = network_reads(network)
    obs = config.to_dict()['obs']
    return [f"obs.{switch} ({', '.join(k for k in keys if k in reads)})" for switch, keys in SWITCH_KEYS.items()
            if not obs.get(switch) and any(k in reads for k in keys)]


def record(key: str, config, states_dir, out: Path, rollouts: int, seed: int, teachers: Teachers, envs: int,
           student=None, tried=None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    collector = MultiCollector([Family(key, config, envs)], out / 'run', STEPS_PER_ENV,
                               0.99, 0.95, seed=seed, states_dir=states_dir, teachers=teachers,
                               actor='teacher' if student is None else 'student', tried=tried)
    rows = 0
    used: dict = {}
    try:
        network = teachers.networks[key] if student is None else student
        for shard in range(rollouts):
            collector.start(network)
            collector.take()
            for event in collector.episodes:
                name = str(event['info'].get('start_state', '?'))
                used[name] = used.get(name, 0) + 1
            collector.ready(network, teachers.device)
            batch = collector.buffer.flatten('cpu')
            np.savez_compressed(out / f'shard-{shard:04d}.npz', **shard_arrays(batch))
            rows += batch['masks'].shape[0]
        episodes = len(collector.episodes)
    finally:
        collector.close()
    return {'key': key, 'rows': rows, 'seconds': round(time.time() - began, 1), 'last_rollout_episodes': episodes,
            'seed': seed, 'start_states_used': dict(sorted(used.items()))}


def main(argv=None):
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--keys', nargs='+', action='extend', default=[], metavar='KEY[=YAML]')
    parser.add_argument('--group', nargs='+', action='extend', default=[], metavar='NAME',
                        help="every member of a teacher of triforce/groups.yaml (lab.groups)")
    parser.add_argument('--teacher', nargs='+', action='extend', default=[], metavar='KEY=NET',
                        help="a key's (or a family's) teacher: a store run FAMILY/RUN, a run dir, a checkpoint, a slot")
    parser.add_argument('--teacher-group', nargs='+', action='extend', default=[], metavar='GROUP=NET',
                        help="one teacher for every recorded member of GROUP")
    parser.add_argument('--sources', nargs='*', default=['resets', 'entries'])
    parser.add_argument('--part-yaml', default=None,
                        help="dir of <key>.yaml (round 36's old runs' tasks: <family>@d3first.yaml; a bare key's "
                             "own task, <family>.yaml, when the dir has one); default: the key's task "
                             "(lab.collect.task_yaml: specialists/<route>/<family>.yaml where there is one)")
    parser.add_argument('--decisions', type=int, default=20480)
    parser.add_argument('--parallel', type=int, default=10)
    parser.add_argument('--seed', type=int, default=36_100_000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--student', default=None,
                        help="a network that acts instead of the teachers (a correction round: the teachers label "
                             "its decisions); NET as for --teacher")
    parser.add_argument('--try-others', default=None, metavar='K:P',
                        help="the look-ahead's verdicts on K other candidates on a share P of the decisions (the hit "
                             "head's labels; lab.train's)")
    parser.add_argument('--state-route', default=None, metavar='ROUTE',
                        help="resets on this route's states alone (lab.collect.route_only), entries only its harvest "
                             "(default: the --group's route)")
    parser.add_argument('--harvest', default=None, metavar='DIR',
                        help="an every-room harvest (DIR/<family>/*.state): the 'rooms' source's states")
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    states_dir = Path(_native.paths()['states_dir'])
    root = registry.models_root(None)
    keys, yamls, chosen = requested_keys(args.keys, args.group)
    routes = sorted({g.route for g in chosen})
    if args.state_route is None and len(routes) == 1:
        args.state_route = routes[0]
    if any(args.state_route != r for r in routes):
        raise SystemExit(f"--state-route {args.state_route}: the groups play {routes}")
    explicit = bool(keys)
    every = team_keys() if not explicit else keys
    keys = keys or every
    named = named_teachers(keys, args.teacher, args.teacher_group)
    if 'rooms' in args.sources and not args.harvest:
        parser.error("the 'rooms' source needs --harvest DIR")
    tried = parse_try_others(args.try_others)
    harvested = paths.data_dir(HARVESTED, states_dir / HARVESTED)
    entries = entry_states(keys, harvested, args.state_route is not None, bool(named)) \
        if 'entries' in args.sources else {}
    rooms = entry_states(keys, Path(args.harvest), True, bool(named)) if args.harvest else {}
    rollouts = max(1, round(args.decisions / (STEPS_PER_ENV * args.parallel)))
    teacher_files = {key: resolve(value, root) for key, value in named.items()} if named else {
        key: registry.policy_checkpoint(key, root, args.state_route) for key in keys}
    student_file = resolve(args.student, root) if args.student else None
    plan = []
    for i, key in enumerate(every):
        if key not in keys:
            continue
        family, part = registry.split_policy(key)
        if key in yamls:
            yaml = yamls[key]
        elif args.part_yaml and (part or (Path(args.part_yaml) / f'{key}.yaml').is_file()):
            yaml = str(Path(args.part_yaml) / f'{key}.yaml')
        else:
            yaml = task_yaml(key)
        task = load_specialist(yaml)
        if 'resets' in args.sources:
            base = parse_families([f'{key}={yaml}'], args.parallel, state_route=args.state_route,
                                  curriculum=False)[0].config
            plan.append((key, 'resets', base, yaml, None, args.seed + 1000 * i))
        for source, found, folder, offset in (('entries', entries, harvested, 500),
                                              ('rooms', rooms, Path(args.harvest or '.'), 700)):
            if source in args.sources and found.get(key):
                config = task.replace(start_states=found[key], ram_randomization={}, harvested_fraction=0.0)
                plan.append((key, source, config, yaml, str(folder / family), args.seed + 1000 * i + offset))
    loaded: dict = {}

    def network_at(path: Path):
        real = str(path.resolve())
        if real not in loaded:
            loaded[real] = registry.load_model(path)[0]
        return loaded[real]

    refused = []
    for key, source, config, _, _, _ in plan:
        for role, path in (('teacher', teacher_files[key]), ('student', student_file)):
            off = switches_off(network_at(path), config) if path is not None else []
            if off:
                refused.append(f"{key} {source}: the {role} {path} reads {', '.join(off)}, off in the task")
    if refused:
        raise SystemExit("networks that would read zeros (turn the switch on in the task, or record with another "
                         "network):\n  " + '\n  '.join(refused))
    for key, source, config, yaml, _, _ in plan:
        obs = config.to_dict()['obs']
        print(f"[plan] {key} {source}: {len(config.start_states)} start states, task {yaml}, teacher "
              f"{named.get(key, '(team)')} = {teacher_files[key]}, obs "
              f"{','.join(k for k, v in sorted(obs.items()) if v) or 'none'}", flush=True)
    if args.dry_run:
        return 0
    require_job()
    from triforce.nets.serve import learner_process  # pylint: disable=import-outside-toplevel
    learner_process()           # its labelling uses the GPU beside the actors' server: no CUDA graphs there
    out = Path(args.out)
    per_item = rollouts * STEPS_PER_ENV * args.parallel
    total = len(plan) * per_item
    code, rom = manifest.code(), manifest.rom()
    with progress_job('r36-merge-record', 'eval', total, unit='decisions',
                      work=work(train_steps=total)) as progress:
        student = network_at(student_file).to(args.device).eval() if student_file is not None else None
        student_network = manifest.network(student_file) if student_file is not None else None
        for done, (key, source, config, yaml, sdir, seed) in enumerate(plan, 1):
            target = out / key / source
            if not (target / 'meta.json').is_file():
                teachers = Teachers([key], args.device, root, {key: str(teacher_files[key])})
                meta = record(key, config, sdir, target, rollouts, seed, teachers, args.parallel, student, tried)
                foreign = foreign_starts(meta['start_states_used'], args.state_route) if args.state_route else []
                if foreign:
                    raise SystemExit(f"{key} {source}: {len(foreign)} start states outside {args.state_route}'s "
                                     f"namespace (no meta.json written): {foreign[:5]}")
                meta.update(source=source, states=len(config.start_states),
                            teacher=named.get(key, str(teacher_files[key])),
                            teacher_network=manifest.network(teacher_files[key]),
                            student=args.student, student_network=student_network,
                            try_others=args.try_others, state_route=args.state_route,
                            obs=config.to_dict()['obs'], task=manifest.task(config, yaml),
                            data=manifest.data(config, sdir), code=code, rom=rom)
                (target / 'meta.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
                print(f"[record] {key} {source}: {meta['rows']} rows in {meta['seconds']}s", flush=True)
                del teachers
            progress.update(done * per_item, key=f"{key} {source}")
    return 0


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    sys.exit(main())
