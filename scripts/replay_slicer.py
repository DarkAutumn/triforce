"""Cuts recorded runs down to their room visits (the owner, Nov 28): one replay a visit, or one replay a run of
several visits back to back, each starting from the machine state where the visit starts, stripped to what
watching needs (``crates/replay`` ``rewrite``: no decisions, targets, masks, rewards or visit records).

    triforce job run --agent A --task T --desc '...' --timeout 30m -- \\
        .venv/bin/python scripts/replay_slicer.py PATH... --out DIR [--wins-only | --losses-only]
            [--rooms [N | ROOM,...]] [--bosses [N | NAME,...]] [--families [N | NAME,...]]
            [--visits SEED:ROOM:vN,... [--visits-into NAME]] [--speed slowest|slow|fastest|all[,...]] [--top N]
            [--seeds SEED,...] [--compile] [--parallel N] [--no-verify]

PATH: replays, or folders of them (``*.tfreplay``; a chain's runs ``<seed>-<ending>.tfreplay``).  Two passes:

1. No emulation: every replay's ROOM events and VISIT records give its room visits: the room (``3x4d``, the
   overworld ``ox55``, a cave ``ox77cave``), the visit (``v2``: the second time the run enters that room), the
   family that played it, its frames (seconds = frames / 60.0988, as the chain records) and outcome.  The
   statistics of each room+visit are over the selected runs (``--wins-only``: the wins): the 10% mark, median, 90%
   mark (linearly interpolated quantiles, numpy's default and ``time-wasters.md``'s; ``run_stats.py`` uses the
   nearest rank), mean, and "lost" = mean - 10% mark; a room+visit in fewer than 80% of the runs is a detour and
   has no bucket.  A visit's bucket: ``slowest`` above its room+visit's 90% mark, ``slow`` above the median up to
   the 90% mark, ``fastest`` below the 10% mark (``all``: every visit).  A 0-frame record (the milestone a won run
   ends on) is no visit; a record whose frames don't end at its own event is warned about; a file that is a slice
   itself (META kind ``clip``), has no seed or can't be read is skipped with a line.  Then every slice to write is
   decided: two files of different replays under one name (the same seed or file name in two folders), or a
   ``--visits`` seed in two replays given, stop the run.
2. The replays with something to write, each walked once in the emulator (``--parallel`` processes): the machine
   at each chosen visit's first and last frame (``replay_cut_states``), then each slice written
   (``replay_rewrite``) and verified bit-exactly (``replay_verify``; ``--no-verify`` skips it).  A replay or file
   that fails is reported and left out of the index; the run ends with exit 1.

What is written, into ``--out``:

- ``--rooms``: ``rooms/<room>/v<N>/`` (every room+visit; ``N``: the N that lose the most, by "lost"; or rooms
  named, ``9x42`` every visit, ``9x42v2`` one);
- ``--bosses``: ``bosses/<boss>/`` (aquamentus, dodongo, manhandla, gleeok, digdogger, gohma, ganon...: every
  boss, the N losing the most, the lost of its room+visits added up, or names);
- ``--families``: ``families/<family>/`` (the same by family);
- ``--visits``: ``<visits-into>/`` (default ``visits``), the visits named, whatever their bucket;
- each visit in the ``--speed`` bucket (default ``slowest``; several: a folder each), ``--top N`` at most per
  folder (slowest first; the fastest bucket fastest first), only runs of ``--seeds`` when given (the statistics
  are still over every selected run); a file a visit, ``<seed>-<room>-v<N>-<seconds>s.tfreplay``
  (``00000148-9x42-v1-215.3s.tfreplay``), and ``index.md`` listing them with their room+visit's marks;
- ``--compile``: instead, a replay a run and bucket of the run's visits (those of the groups given; none given:
  every room+visit) in the run's order, each after the first loaded where the one before ends (a LOAD event):
  ``<run>.<bucket>.tfreplay`` (``00000817-win.slow.tfreplay``) and ``<run>.<bucket>.md`` beside it (not with
  ``--visits`` or ``--top``).

Every file verifies (``tf replay verify``) and plays in ``play-zelda.py`` (a folder of them).
"""
import argparse
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

from triforce import _native, jobs

FPS = 60.0988
SUFFIX = '.tfreplay'
SPEEDS = ('slowest', 'slow', 'fastest', 'all')
DETOUR = 0.8        # a room+visit in fewer of the selected runs than this has no bucket


def room_tag(level: int, room: int, cave: bool) -> str:
    """The owner's notation: ``3x4d``, the overworld ``ox55``, a cave or cellar ``ox77cave``."""
    return f"{'o' if level == 0 else level}x{room:02x}{'cave' if cave else ''}"


def boss_name(family: str) -> Optional[str]:
    """The boss a family fights (``boss-aquamentus`` -> aquamentus, ``ganon``), or None."""
    if family.startswith('boss-'):
        return family[len('boss-'):]
    return family if family == 'ganon' else None


@dataclass
class Visit:
    """One room visit of a run: input frames ``entry..exit``."""
    path: str
    seed: int
    run_ending: str
    level: int
    room: int
    cave: bool
    number: int
    family: str
    entry: int
    exit: int
    outcome: str
    bucket: Optional[str] = None
    marks: dict = field(default_factory=dict)

    @property
    def tag(self) -> str:
        return room_tag(self.level, self.room, self.cave)

    @property
    def key(self) -> tuple:
        return (self.tag, self.number)

    @property
    def seconds(self) -> float:
        return (self.exit - self.entry) / FPS

    def name(self) -> str:
        return f"{self.seed:08d}-{self.tag}-v{self.number}-{self.seconds:.1f}s{SUFFIX}"

    def about(self) -> dict:
        """META ``clip`` of its slice."""
        return {'source': Path(self.path).name, 'seed': self.seed, 'room': self.tag, 'visit': self.number,
                'family': self.family, 'seconds': round(self.seconds, 3), 'outcome': self.outcome,
                'bucket': self.bucket, 'marks': self.marks}


def read_visits(path: str) -> dict:
    """Pass 1 for one replay: its seed, ending, frames and room visits (ROOM events paired with VISIT records; a
    visit the replay ends in without a record runs to its last frame; a 0-frame record, the milestone a won run
    ends on, is no visit).  ``warnings``: a visit whose record's frames don't end at the record's own event.
    ``skip``: why the file is no run to slice (unreadable, a slice itself (META kind ``clip``), no seed)."""
    try:
        replay = _native.Replay(Path(path).read_bytes())
        meta = replay.meta
    except (OSError, _native.ReplayError) as e:
        return {'path': path, 'skip': f"unreadable: {e}"}
    if meta.get('kind') == 'clip':
        return {'path': path, 'skip': "a slice (META kind clip), not a run"}
    seed = meta.get('seed')
    digits = re.match(r'\d+', Path(path).name)
    if seed is None and digits is None:
        return {'path': path, 'skip': "no seed (META seed, or a file name starting with it)"}
    seed = int(seed if seed is not None else digits.group(0))
    models = meta.get('models') or []
    entered: dict = defaultdict(int)
    visits, warnings, open_ = [], [], None

    def close(at: int, family: Optional[str], outcome: str):
        level, room, cave, frame, policy = open_
        if at <= frame:
            return
        if family is None:
            model = models[policy] if policy is not None and policy < len(models) else {}
            family = model.get('family') or model.get('name') or '-'
        entered[(level, room, cave)] += 1
        visits.append(Visit(path, seed, replay.ending, level, room, cave, entered[(level, room, cave)],
                            family, frame, at, outcome))

    for event in replay.events():
        if event['type'] == 'room':
            if open_ is not None:       # entered without the last one's record: it ends here
                close(event['frame'], None, 'unrecorded')
            open_ = (event['level'], event['room'], bool(event['in_cave']), event['frame'], event['policy'])
        elif event['type'] == 'visit' and open_ is not None:
            record = event['record']
            end = open_[3] + int(record['frames'])
            if end != event['frame']:
                warnings.append(f"{room_tag(*open_[:3])} at frame {open_[3]}: the record's {record['frames']} frames "
                                f"end at {end}, its event is at {event['frame']}")
            close(end, record.get('specialist'), str(record.get('outcome')))
            open_ = None
    if open_ is not None:
        close(replay.frames, None, replay.ending)
    return {'path': path, 'seed': seed, 'ending': replay.ending, 'frames': replay.frames, 'visits': visits,
            'warnings': warnings}


def quantile(values: list, q: float) -> float:
    """The ``q`` quantile of sorted ``values``, linearly interpolated (numpy's default; ``time-wasters.md``'s
    figures).  ``run_stats.py``'s room statistics use the nearest rank instead, so a "90%" there can differ from
    the slicer's mark by up to one visit's step."""
    at = q * (len(values) - 1)
    low = int(at)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (at - low)


def room_stats(runs: list) -> dict:
    """Per room+visit over ``runs``: its runs, marks (``fast`` the 10% mark, ``median``, ``slow`` the 90% mark),
    ``mean``, ``lost`` (mean - fast), ``family`` and ``detour`` (in fewer than 80% of the runs)."""
    times, families = defaultdict(list), defaultdict(lambda: defaultdict(int))
    for run in runs:
        for visit in run['visits']:
            times[visit.key].append(visit.seconds)
            families[visit.key][visit.family] += 1
    stats = {}
    for key, values in times.items():
        values.sort()
        mean = sum(values) / len(values)
        fast, slow = quantile(values, 0.1), quantile(values, 0.9)
        stats[key] = {'runs': len(values), 'fast': fast, 'median': quantile(values, 0.5), 'slow': slow, 'mean': mean,
                      'lost': mean - fast, 'family': max(families[key], key=families[key].get),
                      'detour': len(values) < DETOUR * len(runs)}
    return stats


def bucket(visit: Visit, stats: dict) -> Optional[str]:
    """The visit's bucket against its room+visit's marks (None: a detour, or between the 10% mark and the median)."""
    s = stats[visit.key]
    if s['detour']:
        return None
    if visit.seconds > s['slow']:
        return 'slowest'
    if visit.seconds > s['median']:
        return 'slow'
    return 'fastest' if visit.seconds < s['fast'] else None


def marks(stats: dict, key: tuple) -> dict:
    s = stats[key]
    return {name: round(s[name], 2) for name in ('fast', 'median', 'slow', 'mean', 'lost')}


def parse_rooms(text: str) -> list:
    """``9x42`` (every visit) or ``9x42v2``: (tag, visit or None)."""
    specs = []
    for part in (p.strip().lower() for p in text.split(',') if p.strip()):
        match = re.fullmatch(r'([o\d]x[0-9a-f]{2}(?:cave)?)(?:-?v(\d+))?', part)
        if not match:
            raise SystemExit(f"--rooms {part}: a room is 3x4d, ox55 or ox77cave, a visit 9x42v2")
        specs.append((match.group(1), int(match.group(2)) if match.group(2) else None))
    return specs


def ranked(groups: dict, stats: dict) -> list:
    """Group names, the one losing the most first: a group's lost is its room+visits' lost added up (detours
    left out)."""
    def lost(name):
        return sum(stats[key]['lost'] for key in groups[name] if not stats[key]['detour'])
    return sorted(groups, key=lost, reverse=True)


def choose(arg: str, groups: dict, stats: dict, what: str) -> list:
    """``--bosses`` / ``--families``: 'all', a count (the N losing the most) or names."""
    if arg == 'all':
        return sorted(groups)
    if arg.isdigit():
        return ranked(groups, stats)[:int(arg)]
    names = [n.strip() for n in arg.split(',') if n.strip()]
    unknown = [n for n in names if n not in groups]
    if unknown:
        raise SystemExit(f"--{what} {','.join(unknown)}: none in the selected runs ({', '.join(sorted(groups))})")
    return names


@dataclass
class Output:
    """One file pass 2 writes: ``visits`` of one replay (one: a slice; several: a compilation)."""
    dest: str
    visits: list
    ending: Optional[dict] = None


def plan(args, runs: list, stats: dict) -> tuple:
    """The outputs, and the index lines a folder (folder -> lines)."""
    speeds = [s.strip() for s in args.speed.split(',') if s.strip()]
    for s in speeds:
        if s not in SPEEDS:
            raise SystemExit(f"--speed {s}: one of {', '.join(SPEEDS)}")
    seeds = None if args.seeds is None else {int(s) for s in args.seeds.split(',')}
    out = Path(args.out)
    visits = [v for run in runs for v in run['visits'] if v.exit > v.entry]
    for v in visits:
        v.bucket = bucket(v, stats)
        v.marks = marks(stats, v.key)
    wanted = [v for v in visits if seeds is None or v.seed in seeds]

    def in_speed(v, speed):
        return speed == 'all' or v.bucket == speed

    # (tree, folder name, predicate) per group.
    groups = []
    if args.rooms is not None:
        keys = [k for k in stats if not stats[k]['detour']]
        if args.rooms == 'all':
            chosen = sorted(keys)
        elif args.rooms.isdigit():
            chosen = sorted(keys, key=lambda k: stats[k]['lost'], reverse=True)[:int(args.rooms)]
        else:
            specs = parse_rooms(args.rooms)
            chosen = sorted(k for k in stats if any(k[0] == t and n in (None, k[1]) for t, n in specs))
        groups += [('rooms', f"{tag}/v{number}", lambda v, k=(tag, number): v.key == k) for tag, number in chosen]
    for flag, tree, name_of in (('bosses', 'bosses', lambda f: boss_name(f)), ('families', 'families', lambda f: f)):
        arg = getattr(args, flag)
        if arg is None:
            continue
        members = defaultdict(set)
        for key, s in stats.items():
            name = name_of(s['family'])
            if name is not None:
                members[name].add(key)
        groups += [(tree, name, lambda v, ks=frozenset(members[name]): v.key in ks)
                   for name in choose(arg, members, stats, flag)]

    outputs, index = [], defaultdict(list)
    if args.compile:
        runs_by_path = defaultdict(list)
        for v in wanted:
            if not groups or any(pick(v) for _, _, pick in groups):
                runs_by_path[v.path].append(v)
        for path, mine in runs_by_path.items():
            for speed in speeds:
                chosen = [v for v in mine if in_speed(v, speed)]
                if not chosen:
                    continue
                stem = Path(path).name[:-len(SUFFIX)] if path.endswith(SUFFIX) else Path(path).name
                dest = out / f"{stem}.{speed}{SUFFIX}"
                ending = {'kind': 'other', 'result': 'compilation',
                          'reason': f"{len(chosen)} {speed} visits of {Path(path).name}"}
                outputs.append(Output(str(dest), chosen, ending))
                index[str(dest.with_suffix('.md'))] = chosen
        return refuse_collisions(outputs), index

    for tree, folder, pick in groups:
        for speed in speeds:
            chosen = [v for v in wanted if pick(v) and in_speed(v, speed)]
            chosen.sort(key=lambda v: v.seconds, reverse=speed != 'fastest')
            if args.top:
                chosen = chosen[:args.top]
            where = out / tree / folder / (speed if len(speeds) > 1 else '')
            for v in chosen:
                outputs.append(Output(str(where / v.name()), [v], slice_ending(v)))
            index[str(where)] += chosen
    if args.visits:
        for spec in (s.strip() for s in args.visits.split(',') if s.strip()):
            match = re.fullmatch(r'(\d+):([o\d]x[0-9a-f]{2}(?:cave)?):?v(\d+)', spec.lower())
            if not match:
                raise SystemExit(f"--visits {spec}: SEED:ROOM:vN (148:9x42:v1)")
            seed, key = int(match.group(1)), (match.group(2), int(match.group(3)))
            found = [v for v in visits if v.seed == seed and v.key == key]
            if not found:
                raise SystemExit(f"--visits {spec}: no such visit in the selected runs")
            if len({v.path for v in found}) > 1:
                raise SystemExit(f"--visits {spec}: seed {seed} is in more than one replay given: "
                                 f"{', '.join(sorted({v.path for v in found}))}; give one of them alone")
            where = out / args.visits_into
            outputs.append(Output(str(where / found[0].name()), found, slice_ending(found[0])))
            index[str(where)].append(found[0])
    return refuse_collisions(outputs), index


def refuse_collisions(outputs: list) -> list:
    """``outputs``, unless two of different replays would write the same file (the same seed or file name from two
    folders): then the run stops, naming them."""
    sources = defaultdict(set)
    for o in outputs:
        sources[o.dest].add(o.visits[0].path)
    clashes = {dest: paths for dest, paths in sources.items() if len(paths) > 1}
    if clashes:
        dest, paths = next(iter(sorted(clashes.items())))
        raise SystemExit(f"{len(clashes)} files would be written from two replays, e.g. {dest} from "
                         f"{', '.join(sorted(paths))}: slice those folders apart (another --out)")
    return outputs


def slice_ending(v: Visit) -> dict:
    """A one-visit slice's ending: how the visit ended."""
    died = 'death' in v.outcome or 'died' in v.outcome
    return {'kind': 'death' if died else 'other', 'result': v.outcome,
            'reason': f"{v.tag} v{v.number}: {v.outcome} after {v.seconds:.1f} s"}


def write(path: str, outputs: list, verify: bool) -> list:
    """Pass 2 for one replay: the machine at every cut in one walk, then each output written (and verified).  A
    failure is reported as the output's ``error`` (every output's when the walk fails), never raised."""
    try:
        data = Path(path).read_bytes()
        frames = sorted({f for o in outputs for v in o.visits for f in (v.entry, v.exit)})
        cuts = _native.replay_cut_states(data, frames)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return [{'dest': o.dest, 'error': f"{path}: {e}"} for o in outputs]
    done = []
    for o in outputs:
        try:
            windows = [{'from': v.entry, 'to': v.exit, 'start': cuts[v.entry][0], 'end': cuts[v.exit][1],
                        'about': v.about()} for v in o.visits]
            sliced = _native.replay_rewrite(data, watch=True, windows=windows, ending=o.ending)
            report = _native.replay_verify(sliced) if verify else None
            dest = Path(o.dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(f'{dest.name}.{os.getpid()}.tmp')
            tmp.write_bytes(sliced)
            os.replace(tmp, dest)
            done.append({'dest': o.dest, 'bytes': len(sliced), 'frames': _native.Replay(sliced).frames,
                         'verified': report is not None})
        except Exception as e:  # pylint: disable=broad-exception-caught
            done.append({'dest': o.dest, 'error': f"{path}: {e}"})
    return done


def write_index(index: dict, compile_: bool, out: Path, written: Optional[set] = None):
    """``index.md`` a folder (a compilation: ``<run>.<bucket>.md``, with where each visit starts in it): its visits
    with their room+visit's marks; with ``written`` (the files pass 2 wrote), only those."""
    for where, visits in index.items():
        path = Path(where) if compile_ else Path(where) / 'index.md'
        if written is not None:
            if compile_:
                visits = visits if str(path.with_suffix(SUFFIX)) in written else []
            else:
                visits = [v for v in visits if str(Path(where) / v.name()) in written]
        if not visits:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ['| File | Seed | Room | Visit | Family | Seconds | Bucket | 10% | Median | 90% | Mean | Outcome |',
                 '|---|---:|---|---:|---|---:|---|---:|---:|---:|---:|---|']
        total = 0.0
        for v in visits:
            m = v.marks
            name = v.name() if not compile_ else f"{v.tag} v{v.number} at {total:.1f} s"
            lines.append(f"| {name} | {v.seed} | {v.tag} | {v.number} | {v.family} | {v.seconds:.2f} | "
                         f"{v.bucket or '-'} | {m['fast']} | {m['median']} | {m['slow']} | {m['mean']} | {v.outcome} |")
            total += v.seconds
        title = path.stem if compile_ else str(Path(where).relative_to(out)) if Path(where) != out else out.name
        head = (f"# {title}\n\n{len(visits)} visits, {total:.1f} s in all; marks: the room+visit's over the "
                f"selected runs (seconds, linearly interpolated quantiles).\n\n")
        path.write_text(head + '\n'.join(lines) + '\n')


def replay_paths(paths: list) -> list:
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(str(f) for f in p.iterdir() if f.suffix == SUFFIX and f.is_file())
        elif p.is_file():
            files.append(str(p))
        else:
            raise SystemExit(f"{p}: not a replay or a folder of them")
    return files


def wanted_by_name(path: str, args) -> bool:
    """A chain's run named ``<seed>-<ending>.tfreplay`` is passed over by its name when its ending isn't wanted."""
    match = re.fullmatch(r'\d+-(\w+)' + re.escape(SUFFIX), Path(path).name)
    if not match:
        return True
    return not ((args.wins_only and match.group(1) != 'win') or (args.losses_only and match.group(1) == 'win'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('paths', nargs='+', metavar='PATH')
    parser.add_argument('--out', required=True)
    side = parser.add_mutually_exclusive_group()
    side.add_argument('--wins-only', action='store_true')
    side.add_argument('--losses-only', action='store_true')
    parser.add_argument('--rooms', nargs='?', const='all', default=None, metavar='N|ROOMS')
    parser.add_argument('--bosses', nargs='?', const='all', default=None, metavar='N|NAMES')
    parser.add_argument('--families', nargs='?', const='all', default=None, metavar='N|NAMES')
    parser.add_argument('--visits', default=None, metavar='SEED:ROOM:vN,...')
    parser.add_argument('--visits-into', default='visits', metavar='NAME')
    parser.add_argument('--speed', default='slowest', metavar='|'.join(SPEEDS))
    parser.add_argument('--top', type=int, default=None, metavar='N')
    parser.add_argument('--seeds', default=None, metavar='SEED,...')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--parallel', type=int, default=8, metavar='N')
    parser.add_argument('--no-verify', action='store_true')
    args = parser.parse_args(argv)
    if not args.compile and args.rooms is None and args.bosses is None and args.families is None \
            and args.visits is None:
        parser.error("nothing to write: --rooms, --bosses, --families, --visits or --compile")
    if args.compile and (args.visits is not None or args.top is not None):
        parser.error("--compile writes a run's visits by bucket: not with --visits or --top")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    jobs.require_job()
    began = time.monotonic()
    files = [f for f in replay_paths(args.paths) if wanted_by_name(f, args)]

    with ProcessPoolExecutor(max_workers=max(1, args.parallel), mp_context=get_context('spawn')) as pool:
        runs = []
        for count, run in enumerate(pool.map(read_visits, files, chunksize=8), 1):
            jobs.progress(count, len(files), 'replays read')
            if 'skip' in run:
                print(f"skipped {run['path']}: {run['skip']}", flush=True)
                continue
            for warning in run['warnings']:
                print(f"warning: {run['path']}: {warning}", flush=True)
            if (args.wins_only and run['ending'] != 'win') or (args.losses_only and run['ending'] == 'win'):
                continue
            runs.append(run)
        if not runs:
            raise SystemExit("no run selected")
        stats = room_stats(runs)
        outputs, index = plan(args, runs, stats)
        read = time.monotonic() - began
        print(f"pass 1: {len(runs)} runs of {len(files)} replays, {sum(len(r['visits']) for r in runs)} visits, "
              f"{len(stats)} room+visits; {len(outputs)} files to write ({read:.1f} s)", flush=True)

        by_path = defaultdict(list)
        for o in outputs:
            by_path[o.visits[0].path].append(o)
        results, done = [], 0
        futures = [pool.submit(write, path, mine, not args.no_verify) for path, mine in sorted(by_path.items())]
        for future in futures:
            results += future.result()
            done += 1
            jobs.progress(done, len(futures), 'replays sliced')
    written = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    write_index(index, args.compile, Path(args.out), {w['dest'] for w in written})

    trees = defaultdict(lambda: [0, 0])
    root = Path(args.out)
    for w in written:
        part = Path(w['dest']).relative_to(root).parts
        tree = trees[part[0] if len(part) > 1 else '.']
        tree[0] += 1
        tree[1] += w['bytes']
    for tree, (count, size) in sorted(trees.items()):
        print(f"  {tree}: {count} files, {size / 1e6:.1f} MB")
    for f in failed:
        print(f"FAILED {f['dest']}: {f['error']}")
    verified = sum(w['verified'] for w in written)
    print(f"pass 2: {len(written)} files from {len(by_path)} replays, {verified} verified bit-exactly, "
          f"{sum(w['bytes'] for w in written) / 1e6:.1f} MB, {len(failed)} failed; "
          f"{time.monotonic() - began:.1f} s in all")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
