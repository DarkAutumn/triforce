"""Round 36, "One network by imitation": teacher-labelled data for every policy key of today's team.

    python scripts/experiments/r36_merge_record.py --out target/merge/data --decisions 20480 [--keys ...] [--dry-run]

Per key (51: 34 families, 17 ``family@d3first``) and source, the key's teacher (``registry.policy_checkpoint``)
acting on 10 envs (``lab.collect.MultiCollector``, ``--actor teacher``), every row labelled with its joint
log-probs and value, written like ``lab.record`` (``<out>/<key>/<source>/shard-NNNN.npz``, ``meta.json``):

- ``resets``: the key's training task (the family's yaml, or a route part's own yaml, ``--part-yaml``: the
  ``run.json`` ``specialist`` of the old run), its own hp and item randomisation;
- ``entries``: the family's harvested pre-entry states as the chains bring them (the route's own, ``d3first_*``,
  where the family has them, as the bench chooses), less every state the bench plays, no randomisation; each state
  goes to the key that plays its route step (``_native.step_policy``: ``family@d3first`` before the magic sword);
- ``rooms`` (``--harvest DIR``): the same from an every-room harvest (``eval-chain --harvest DIR
  --harvest-every-room``, ``DIR/<family>/``), with the room entries a chain makes from a room of the same family,
  which the handoff harvest never saves, less both tiers' states (``bench.json``, ``bench-rooms.json``). With
  ``--student`` and the student's own harvest it is a correction round on the rooms the student brings itself to,
  at the hearts it brings.

Seeds ``--seed`` + 1000 · key index (+ 500 for entries, + 700 for rooms): disjoint from the bench's (360000 + i).

``meta.json`` lists the start state of every episode a source played (``start_states_used``). With ``--state-route``
a source with an episode started outside the route's namespace fails before its ``meta.json`` is written
(``foreign_starts``; round 38 found 1,280 such episodes, ``d2first``'s harvest, before the start-state pool kept to
its namespaces)."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from triforce import _native, registry
from triforce.events import progress_job
from triforce.jobs import add_job_arguments, begin_job, work
from triforce.lab.bench import HARVESTED, ROOMS_MANIFEST, ROUTE, load_manifest, team_keys
from triforce.lab.collect import Family, MultiCollector, Teachers, route_only, task_yaml
from triforce.specialist import load_specialist

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


def entry_states(keys: list[str], harvested: Path, route_only_entries: bool = False) -> dict:
    """key -> state names (bare) in ``harvested/<family>/`` it plays, less both bench tiers';
    ``route_only_entries``: only the route's own (``d3first_*``), never another route's harvest in their place."""
    bench = {Path(s).name for manifest in (load_manifest(), load_manifest(ROOMS_MANIFEST))
             for e in manifest['entries'] for s in e['states']}
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
            if key in result:
                result[key].append(name)
    return result


def record(key: str, config, states_dir, out: Path, rollouts: int, seed: int, teachers: Teachers, envs: int,
           student=None):
    out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    collector = MultiCollector([Family(key, config, envs)], out / 'run', STEPS_PER_ENV, 0.99, 0.95, seed=seed,
                               states_dir=states_dir, teachers=teachers,
                               actor='teacher' if student is None else 'student')
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
            arrays = {f'obs/{k}': v.numpy() for k, v in batch['obs'].items()}
            arrays.update(action_mask=batch['masks'].numpy(), action=batch['actions'].numpy(),
                          teacher_joint=batch['teacher_joint'].numpy(),
                          teacher_value=batch['teacher_value'].numpy())
            np.savez_compressed(out / f'shard-{shard:04d}.npz', **arrays)
            rows += batch['masks'].shape[0]
        episodes = len(collector.episodes)
    finally:
        collector.close()
    return {'key': key, 'rows': rows, 'seconds': round(time.time() - began, 1), 'last_rollout_episodes': episodes,
            'teacher': teachers.paths[key], 'seed': seed, 'start_states_used': dict(sorted(used.items()))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--keys', nargs='*', default=None)
    parser.add_argument('--sources', nargs='*', default=['resets', 'entries'])
    parser.add_argument('--part-yaml', default=None,
                        help="dir of <family>@d3first.yaml (round 36's old runs' tasks); default: the lab's task "
                             "of the key (specialists/<route>/<family>.yaml, the corrected route's)")
    parser.add_argument('--decisions', type=int, default=20480)
    parser.add_argument('--parallel', type=int, default=10)
    parser.add_argument('--seed', type=int, default=36_100_000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--student', default=None,
                        help="a checkpoint that acts instead of the teachers (DAgger: the teachers label its states)")
    parser.add_argument('--state-route', default=None, metavar='ROUTE',
                        help="resets on this route's states alone (lab.collect.route_only), entries only its harvest")
    parser.add_argument('--harvest', default=None, metavar='DIR',
                        help="an every-room harvest (DIR/<family>/*.state): the 'rooms' source's states")
    parser.add_argument('--dry-run', action='store_true')
    add_job_arguments(parser)
    args = parser.parse_args(argv)
    begin_job(args)
    states_dir = Path(_native.paths()['states_dir'])
    every = team_keys()
    keys = args.keys or every
    entries = entry_states(every, states_dir / HARVESTED, args.state_route is not None)
    if 'rooms' in args.sources and not args.harvest:
        parser.error("the 'rooms' source needs --harvest DIR")
    rooms = entry_states(every, Path(args.harvest), True) if args.harvest else {}
    rollouts = max(1, round(args.decisions / (STEPS_PER_ENV * args.parallel)))
    plan = []
    for i, key in enumerate(every):
        if key not in keys:
            continue
        family, part = registry.split_policy(key)
        if part and args.part_yaml:
            base = load_specialist(str(Path(args.part_yaml) / f'{key}.yaml'))
        else:
            base = load_specialist(task_yaml(key))
        if args.state_route:
            base = route_only(base, args.state_route)
        if 'resets' in args.sources:
            plan.append((key, 'resets', base, None, args.seed + 1000 * i))
        if 'entries' in args.sources and entries[key]:
            config = load_specialist(family).replace(start_states=entries[key], ram_randomization={})
            plan.append((key, 'entries', config, str(states_dir / HARVESTED / family), args.seed + 1000 * i + 500))
        if 'rooms' in args.sources and rooms[key]:
            config = load_specialist(family).replace(start_states=rooms[key], ram_randomization={})
            plan.append((key, 'rooms', config, str(Path(args.harvest) / family), args.seed + 1000 * i + 700))
    for key, source, config, _, _ in plan:
        print(f"[plan] {key} {source}: {len(config.start_states)} start states", flush=True)
    if args.dry_run:
        return 0
    out = Path(args.out)
    per_item = rollouts * STEPS_PER_ENV * args.parallel
    total = len(plan) * per_item
    with progress_job('r36-merge-record', 'eval', total, unit='decisions',
                      work=work(train_steps=total)) as progress:
        student = None
        if args.student:
            student = registry.load_model(args.student)[0].to(args.device).eval()
        for done, (key, source, config, sdir, seed) in enumerate(plan, 1):
            target = out / key / source
            if not (target / 'meta.json').is_file():
                teachers = Teachers([key], args.device)
                meta = record(key, config, sdir, target, rollouts, seed, teachers, args.parallel, student)
                meta.update(source=source, states=len(config.start_states))
                foreign = foreign_starts(meta['start_states_used'], args.state_route) if args.state_route else []
                if foreign:
                    raise SystemExit(f"{key} {source}: {len(foreign)} start states outside {args.state_route}'s "
                                     f"namespace (no meta.json written): {foreign[:5]}")
                (target / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
                print(f"[record] {key} {source}: {meta['rows']} rows in {meta['seconds']}s", flush=True)
                del teachers
            progress.update(done * per_item, key=f"{key} {source}")
    return 0


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    sys.exit(main())
