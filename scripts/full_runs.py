"""Unbroken runs of the whole game from the start (no golden stitching, no segments): the best of them, and the
250-run test of a build.

    .venv/bin/python scripts/full_runs.py run [--runs 100] [--first-seed 0] [--models-dir DIR] [--out DIR]
    .venv/bin/python scripts/full_runs.py report OUT_DIR
    .venv/bin/python scripts/full_runs.py runs250 [--models-dir DIR] [--team NAME] [--out DIR] [--docs DIR]
    .venv/bin/python scripts/full_runs.py stats OUT_DIR [--team NAME] [--docs DIR] [--no-archive]
    .venv/bin/python scripts/full_runs.py analyze JSON [JSON ...] [--markdown FILE] [--rooms N]

``run`` plays ``--runs`` runs of ``d3first`` from the start with every run recorded as a replay
(``OUT_DIR/replays/``), writes the chain records (``OUT_DIR/runs.jsonl``), then reports.

Every subcommand but ``analyze`` is a job of the machine's queue (``triforce job run -- .venv/bin/python
scripts/full_runs.py ...``): ``run`` and ``runs250`` announce their chain's work (``jobs.announce``), so the queue
estimates them, and the chain inside reports its runs as the job's progress; ``report`` and ``stats`` crunch and
archive replays (``jobs.require_job``: ``--timeout`` states their duration).

``report`` reads ``OUT_DIR``: the wall time, how many runs took each dungeon's triforce and ended the game, and the
**best run** (the owner, Nov 23): the furthest route step reached (a room record's ``step`` is the route step the run
stood on, so walking out of a room and back, or into a room off the route, adds nothing), then the fewest cheated
bombs and keys, then the fewest cheated rupees, then the fewest frames.  The best run's replay is archived
(``scripts/archive_replays.py``: ``replays/`` and ``/backup/triforce/replays``) and its path printed; ``best.json``
in OUT_DIR keeps the report.

``runs250`` is the test every versioned build gets (the owner, Nov 23): 250 runs of ``d3first`` from console power
on (``--power-on``: each seed its own file-select delay), every one recorded (``triforce job run --express`` puts
it in the express place whatever the history estimates).  The team is frozen first:
the family pointers of ``--models-dir`` (default ``<repo>/models``) are copied into ``OUT_DIR/models`` (their run
folders linked), so a promotion during the test changes nothing.  Then ``stats``.

``stats`` reads such an ``OUT_DIR`` and keeps (``kept``): the **best** run (the ranking above), the **worst** (the
reverse: least far; ties the most cheated, then the most frames) and up to 3 runs that ended **stuck** (any end but
a win, a death or an error), each in a different room (the rooms that stuck most often first, the lowest seed of
each).  Their replays are copied to ``OUT_DIR/kept/`` and archived to the backup (``archive_replays.py --kind
runs-250 --label <reasons>``: never the repository's ``replays/``).  It writes the long-term record
``<docs>/<version>[_<team>].json`` (default ``docs/runs250/``, committed; ``OUT_DIR/runs250.json`` too): the
version, commit, team (each family's run and checkpoint SHA-1), seeds, wall time and decisions a second, the runs
that took each triforce, the endings by kind, deaths and stucks per room (room, route step, family), every room
visit's hits and hearts lost per room, the cheated items and ending of each run, the median frames to each triforce
and the kept replays.  ``--team`` names a team other than main's (``2.0.0.11_one-d1w8.json``).

``analyze`` compares one or more of those records (the first is the baseline): per-dungeon rates and endings with
their binomial standard errors and each record's difference from the baseline, and per room (room and route step)
the deaths and stucks among the runs that reached it, hits and hearts lost per visit, for the ``--rooms`` rooms
that lost most runs.  It prints Markdown (``--markdown FILE`` also writes it) for check-ins.
"""
import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTE = 'd3first'
ORDER = [3, 4, 1, 2, 5, 7, 6, 8, 9]          # d3first's dungeons, in route order
RUNS250 = 250
SCHEMA = 1                                   # the runs-250 record's layout
STUCK_KEPT = 3                               # stuck runs kept, each in a different room
DOCS = ROOT / 'docs' / 'runs250'
# A run's end (``ending``): win, death, error, else stuck by its reason's kind (crates/env/src/chain.rs).
STUCK_KINDS = ('bounce-limit', 'wrong-exit-limit', 'retreat-limit', 'dock-warp-missed')
RUN_FIELDS = ('seed', 'ending', 'room', 'step', 'furthest_step', 'triforces', 'cheated_bombs', 'cheated_keys',
              'cheated_rupees', 'frames')
# A room at a route step (step None: off the route): its visits, the runs that entered it there, the runs that died or
# stuck there, the visits' hits taken (actions that lost health), hearts lost (``hearts_lost``) and frames.
ROOM_FIELDS = ('room', 'step', 'family', 'visits', 'runs', 'deaths', 'stucks', 'hits', 'hearts_lost', 'frames')


def score(record: dict, steps: list) -> dict:
    """One run's place in the ranking (``best``) and what it took: a dungeon's triforce counts once the run left the
    route step whose exit is that triforce correctly (not by entering the next dungeon: a run can die between)."""
    rooms = record.get('rooms') or []
    won = record.get('result') == 'success'
    furthest = len(steps) if won else max(((room.get('step') or 0) for room in rooms), default=0)
    taken = sorted(triforce_frames(record, steps), key=ORDER.index)
    return {'seed': record['seed'], 'furthest_step': furthest, 'won': won,
            'cheated_bombs_keys': sum((room.get('cheated_bombs') or 0) + (room.get('cheated_keys') or 0)
                                      for room in rooms),
            'cheated_rupees': sum(room.get('cheated_rupees') or 0 for room in rooms),
            'frames': record.get('frames'), 'reason': record.get('reason'), 'triforces': taken}


def triforce_frames(record: dict, steps: list) -> dict:
    """{level: frames from the run's start to leaving its triforce step correctly}; a win took Dungeon 9's at its
    last frame."""
    by_step = {i: step['level'] for i, step in enumerate(steps) if step.get('exit') == 'triforce'}
    frames, taken = 0, {}
    for room in record.get('rooms') or []:
        frames += room.get('frames') or 0
        if room.get('step') in by_step and room.get('outcome') == 'exit-correct':
            taken.setdefault(by_step[room['step']], frames)
    if record.get('result') == 'success':
        taken.setdefault(9, record.get('frames') or frames)
    return taken


def rank(s: dict) -> tuple:
    """The best run first: furthest, then fewest bombs and keys cheated, then fewest rupees cheated, then fewest
    frames."""
    return (-s['furthest_step'], s['cheated_bombs_keys'], s['cheated_rupees'], s['frames'] or 0)


def best(scores: list) -> dict:
    """Furthest, then fewest bombs and keys cheated, then fewest rupees cheated, then fewest frames."""
    return min(scores, key=rank)


def worst(scores: list) -> dict:
    """The reverse of ``best``: least far, then most cheated, then most frames."""
    return max(scores, key=rank)


def report(out: Path) -> dict:
    from triforce import _native  # pylint: disable=import-outside-toplevel
    steps = _native.route_steps(ROUTE)
    records = read_runs(out)
    scores = [score(r, steps) for r in records]
    top = best(scores)
    taken = Counter(level for s in scores for level in s['triforces'])
    replay = replay_of(out, top['seed'])
    timing = json.loads((out / 'timing.json').read_text()) if (out / 'timing.json').exists() else {}
    result = {'runs': len(scores), 'seconds': timing.get('seconds'),
              'triforces': {f'D{level}': taken[level] for level in ORDER},
              'won': sum(s['won'] for s in scores), 'best': top,
              'best_step': (str(steps[top['furthest_step']]) if top['furthest_step'] < len(steps) else 'the end'),
              'best_replay': str(replay) if replay else None}
    (out / 'best.json').write_text(json.dumps(result, indent=2) + '\n')
    if replay:
        subprocess.run([sys.executable, str(ROOT / 'scripts' / 'archive_replays.py'), str(replay)], check=False)
    print(json.dumps(result, indent=2))
    return result


def read_runs(out: Path) -> list:
    return [json.loads(line) for line in (out / 'runs.jsonl').read_text().splitlines() if line.strip()]


def replay_of(out: Path, seed: int):
    return next(iter(sorted((out / 'replays').glob(f"{seed:08d}-*.tfreplay"))), None)


def chain_command(out: Path, runs: int, first_seed: int, parallel: int, models_dir=None) -> list:
    """``triforce eval-chain`` of ``d3first`` with every run recorded (``--models-dir`` is the ``triforce``
    command's own option: before ``eval-chain``)."""
    return [sys.executable, '-m', 'triforce.cli', *(['--models-dir', str(models_dir)] if models_dir else []),
            'eval-chain', '--route', ROUTE, '--clips', 'on', '--act', 'model', '--runs', str(runs),
            '--first-seed', str(first_seed), '--parallel', str(parallel), '--replays', str(out / 'replays'),
            '--output', str(out / 'runs.jsonl')]


def announce_chain(runs: int, name: str):
    """The chain's work for the queue (``jobs.announce``: the wrapper's dry call ends here)."""
    sys.path.insert(0, str(ROOT))
    from triforce.jobs import announce, work  # pylint: disable=import-outside-toplevel
    from triforce.runner import steps_played  # pylint: disable=import-outside-toplevel
    announce('chain', work(chain_steps=runs * steps_played(ROUTE)), name)


def run(args) -> int:
    announce_chain(args.runs, 'full-runs')
    out = Path(args.out)
    (out / 'replays').mkdir(parents=True, exist_ok=True)
    command = chain_command(out, args.runs, args.first_seed, args.parallel, args.models_dir)
    if args.power_on:
        command.append('--power-on')      # from console power on (a random file-select delay a seed), not the start state
    start = time.time()
    code = subprocess.run(command, cwd=ROOT, check=False).returncode
    (out / 'timing.json').write_text(json.dumps({'seconds': round(time.time() - start, 1), 'runs': args.runs}) + '\n')
    if code not in (0, 1):
        return code
    report(out)
    return 0


# ---------------------------------------------------------------------------------------------- the 250-run test

def freeze_models(source: Path, target: Path) -> Path:
    """A models root at ``target`` that plays what ``source``'s pointers name now: every family folder's pointer
    files (``promoted``, ``promoted.<route>``, ``late.<route>``) copied, its run folders linked."""
    for family in sorted(p for p in source.iterdir() if p.is_dir()):
        pointers = [p for p in family.iterdir() if p.is_file() and p.name.split('.')[0] in ('promoted', 'late')]
        if not pointers:
            continue
        (target / family.name).mkdir(parents=True, exist_ok=True)
        for pointer in pointers:
            shutil.copy2(pointer, target / family.name / pointer.name)
        for child in family.iterdir():
            link = target / family.name / child.name
            if child.is_dir() and not link.exists():
                link.symlink_to(child.resolve(), target_is_directory=True)
    return target


def runs250(args) -> int:
    out = Path(args.out)
    if (out / 'runs.jsonl').exists():
        print(f"{out / 'runs.jsonl'} exists; `full_runs.py stats {out}` reports it, or choose a new --out",
              file=sys.stderr)
        return 2
    announce_chain(args.runs, 'runs250')
    (out / 'replays').mkdir(parents=True, exist_ok=True)
    models = freeze_models(Path(args.models_dir).resolve(), out / 'models')
    command = chain_command(out, args.runs, args.first_seed, args.parallel, models) + [
        '--power-on', '--events', str(out / 'events.jsonl')]
    start, commit = time.time(), git_head()      # the code that plays (stats may run later, on a newer commit)
    code = subprocess.run(command, cwd=ROOT, check=False).returncode
    (out / 'timing.json').write_text(json.dumps({'seconds': round(time.time() - start, 1), 'runs': args.runs,
                                                 'commit': commit}) + '\n')
    if code not in (0, 1):
        return code
    stats(out, args.team, Path(args.docs), archive=not args.no_archive)
    return 0


def ending(record: dict) -> str:
    """win, death, error, else the stuck kind: bounce-limit, wrong-exit-limit, retreat-limit, dock-warp-missed,
    no-room-progress, stalled, room-step-cap, run-step-cap, off-route (left the route into an untagged room) or
    other."""
    reason = record.get('reason') or ''
    if record.get('result') == 'success':
        return 'win'
    if record.get('result') == 'error':
        return 'error'
    if reason.startswith('died in'):
        return 'death'
    head = reason.split(':', 1)[0]
    if head in STUCK_KINDS:
        return head
    if reason.startswith('run exceeded'):
        return 'run-step-cap'
    for marker, kind in ((' no-room-progress:', 'no-room-progress'), (' stalled:', 'stalled'),
                         (' exceeded ', 'room-step-cap'), (' left the route into untagged ', 'off-route')):
        if marker in reason:
            return kind
    return 'other'


def room_key(room: dict) -> str:
    """``3:0x4d``, a cave ``0:0x77:cave``."""
    return f"{room['level']}:{room['room']}" + (':cave' if room.get('in_cave') else '')


def hearts_lost(room: dict) -> float:
    """Health lost in a visit: what it came in with plus the dropped hearts it healed, less what it left with (a
    fairy's or a potion's refill is not counted, so a visit that used one shows less)."""
    return max(0.0, (room.get('health_in') or 0) + (room.get('hearts_healed') or 0) - (room.get('health_out') or 0))


def chain_seconds(out: Path):
    """The chain's own wall seconds (its ``chain_start`` to its ``chain`` event: the wait in the line left out)."""
    path = out / 'events.jsonl'
    if not path.exists():
        return None
    times = {}
    for line in path.read_text().splitlines():
        if line.strip():
            event = json.loads(line)
            if event.get('event') in ('chain_start', 'chain'):
                times[event['event']] = event.get('ts')
    if times.get('chain_start') is None or times.get('chain') is None:
        return None
    return round(times['chain'] - times['chain_start'], 1)


def git_head() -> str:
    try:
        return subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def keep(records: list, scores: list) -> list:
    """The kept runs: best, worst and up to ``STUCK_KEPT`` stuck ones in different rooms (the rooms that stuck most
    first, the lowest seed of each); [{seed, reasons}] in that order, a run kept for two reasons once."""
    kept = {}
    for reason, s in (('best', best(scores)), ('worst', worst(scores))):
        kept.setdefault(s['seed'], []).append(reason)
    stuck = defaultdict(list)
    for record in records:
        if ending(record) not in ('win', 'death', 'error') and record.get('rooms'):
            stuck[room_key(record['rooms'][-1])].append(record['seed'])
    for room, seeds in sorted(stuck.items(), key=lambda item: (-len(item[1]), min(item[1])))[:STUCK_KEPT]:
        kept.setdefault(min(seeds), []).append(f"stuck-{room.replace(':', '-')}")
    return [{'seed': seed, 'reasons': reasons} for seed, reasons in kept.items()]


def summarize(records: list, provenance: dict, steps: list, seconds=None, total_seconds=None) -> dict:
    """The runs-250 record of ``records`` (one chain's runs) and its ``provenance`` (``<output>.meta.json``)."""
    # pylint: disable=too-many-locals
    scores = {r['seed']: score(r, steps) for r in records}
    endings = Counter(ending(r) for r in records)
    rooms = defaultdict(lambda: {'visits': 0, 'runs': set(), 'deaths': 0, 'stucks': 0, 'hits': 0,
                                 'hearts_lost': 0.0, 'frames': 0, 'families': Counter()})
    deaths, stucks = Counter(), Counter()
    triforce_at = defaultdict(list)
    rows = []
    for record in records:
        for room in record.get('rooms') or []:
            entry = rooms[(room_key(room), room.get('step'))]
            entry['visits'] += 1
            entry['runs'].add(record['seed'])
            entry['hits'] += room.get('hits') or 0
            entry['hearts_lost'] += hearts_lost(room)
            entry['frames'] += room.get('frames') or 0
            entry['families'][room.get('specialist')] += 1
        kind = ending(record)
        last = (record.get('rooms') or [None])[-1]
        if last is not None and kind not in ('win', 'error'):
            where = (room_key(last), last.get('step'), last.get('specialist'))
            if kind == 'death':
                deaths[where] += 1
                rooms[where[:2]]['deaths'] += 1
            else:
                stucks[(*where, kind)] += 1
                rooms[where[:2]]['stucks'] += 1
        for level, frames in triforce_frames(record, steps).items():
            triforce_at[level].append(frames)
        s = scores[record['seed']]
        cheated = {item: sum(room.get(f'cheated_{item}') or 0 for room in record.get('rooms') or [])
                   for item in ('bombs', 'keys', 'rupees')}
        rows.append([record['seed'], kind, None if last is None else room_key(last),
                     None if last is None else last.get('step'), s['furthest_step'],
                     ''.join(str(level) for level in s['triforces']), cheated['bombs'], cheated['keys'],
                     cheated['rupees'], record.get('frames')])
    decisions = sum(r.get('steps') or 0 for r in records)
    batches = provenance.get('batches') or [{}]
    prov = provenance.get('provenance') or {}
    stuck_kinds = {k: n for k, n in sorted(endings.items()) if k not in ('win', 'death', 'error')}
    furthest = [s['furthest_step'] for s in scores.values()]
    return {
        'schema': SCHEMA, 'test': 'runs-250', 'version': prov.get('version'), 'commit': git_head(),
        'code': prov.get('code'), 'data': prov.get('data'), 'route': prov.get('route', ROUTE),
        'power_on': prov.get('power_on'), 'clips': prov.get('clips'), 'max_steps': prov.get('max_steps'),
        'route_steps': len(steps),
        'team': [{key: model.get(key) for key in ('name', 'checkpoint', 'checkpoint_sha1', 'hash', 'act', 'version')}
                 for model in prov.get('models') or []],
        'missing': prov.get('missing'),
        'seeds': {'first': batches[0].get('first_seed'), 'runs': len(records)},
        'wall_seconds': seconds, 'total_seconds': total_seconds, 'decisions': decisions,
        'decisions_per_second': round(decisions / seconds, 1) if seconds else None,
        'frames': sum(r.get('frames') or 0 for r in records),
        'triforces': {f'D{level}': len(triforce_at[level]) for level in ORDER},
        'won': endings['win'],
        'endings': {'win': endings['win'], 'death': endings['death'], 'stuck': sum(stuck_kinds.values()),
                    'error': endings['error']},
        'stuck_kinds': stuck_kinds,
        'furthest_step': {'median': statistics.median(furthest) if furthest else None,
                          'max': max(furthest, default=None)},
        'triforce_frames_median': {f'D{level}': (round(statistics.median(triforce_at[level]))
                                                 if triforce_at[level] else None) for level in ORDER},
        'deaths': [{'room': room, 'step': step, 'family': family, 'runs': n}
                   for (room, step, family), n in sorted(deaths.items(), key=lambda item: (-item[1], str(item[0])))],
        'stucks': [{'room': room, 'step': step, 'family': family, 'kind': kind, 'runs': n}
                   for (room, step, family, kind), n in sorted(stucks.items(),
                                                               key=lambda item: (-item[1], str(item[0])))],
        'room_fields': list(ROOM_FIELDS),
        'rooms': [[room, step, entry['families'].most_common(1)[0][0], entry['visits'], len(entry['runs']),
                   entry['deaths'], entry['stucks'], entry['hits'], round(entry['hearts_lost'], 1), entry['frames']]
                  for (room, step), entry in sorted(rooms.items(),
                                                    key=lambda item: (item[0][1] is None, item[0][1] or 0,
                                                                      item[0][0]))],
        'run_fields': list(RUN_FIELDS),
        'runs': rows,
        'kept': [{**k, 'ending': scores[k['seed']]['reason'], 'furthest_step': scores[k['seed']]['furthest_step'],
                  'frames': scores[k['seed']]['frames'], 'cheated_bombs_keys': scores[k['seed']]['cheated_bombs_keys'],
                  'cheated_rupees': scores[k['seed']]['cheated_rupees']}
                 for k in keep(records, list(scores.values()))],
    }


def dump(record: dict) -> str:
    """The record as JSON, one top-level key a line and each list item on its own line (small, diffable)."""
    lines = []
    for key, value in record.items():
        if isinstance(value, list) and value and isinstance(value[0], (dict, list)):
            items = ',\n'.join(f'  {json.dumps(item)}' for item in value)
            lines.append(f'{json.dumps(key)}: [\n{items}\n ]')
        else:
            lines.append(f'{json.dumps(key)}: {json.dumps(value)}')
    return '{\n ' + ',\n '.join(lines) + '\n}\n'


def record_name(version: str, team) -> str:
    return f"{version}{'_' + team if team else ''}.json"


def stats(out: Path, team=None, docs: Path = DOCS, archive: bool = True) -> dict:
    from triforce import _native  # pylint: disable=import-outside-toplevel
    steps = _native.route_steps(ROUTE)
    meta = json.loads((out / 'runs.jsonl.meta.json').read_text())
    timing = json.loads((out / 'timing.json').read_text()) if (out / 'timing.json').exists() else {}
    record = summarize(read_runs(out), meta, steps, chain_seconds(out), timing.get('seconds'))
    record['commit'] = timing.get('commit') or record['commit']
    plan = out / 'training.json'                 # the chain's plan: the place it took in the job line
    record['lane'] = json.loads(plan.read_text()).get('lane') if plan.exists() else None
    record['team_name'] = team or 'main'
    kept_dir = out / 'kept'
    kept_dir.mkdir(exist_ok=True)
    for kept in record['kept']:
        replay = replay_of(out, kept['seed'])
        if replay is None:
            kept['replay'] = None
            continue
        label = '_'.join(kept['reasons'])           # archive_replays.py --label: [A-Za-z0-9][A-Za-z0-9._-]*
        copy = kept_dir / f"{label}_{replay.name}"
        shutil.copy2(replay, copy)
        kept['replay'] = copy.name
        if archive:
            done = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'archive_replays.py'), '--kind', 'runs-250',
                                   '--label', label, str(copy)], check=False).returncode
            if done:
                print(f"full_runs: archive_replays.py failed ({done}) for {copy}", file=sys.stderr)
    text = dump(record)
    (out / 'runs250.json').write_text(text)
    docs.mkdir(parents=True, exist_ok=True)
    path = docs / record_name(record['version'], team)
    path.write_text(text)
    print(markdown([record], [path.name]))
    print(f"record: {path}; kept replays: {kept_dir}")
    return record


# ---------------------------------------------------------------------------------------------- analysis

def rate(n: int, total: int) -> tuple:
    """(p, binomial standard error)."""
    if not total:
        return None, None
    p = n / total
    return p, math.sqrt(p * (1 - p) / total)


def pct(p) -> str:
    return '–' if p is None else f'{100 * p:.1f}'


def with_se(n: int, total: int) -> str:
    p, se = rate(n, total)
    return '–' if p is None else f'{100 * p:.1f} ± {100 * se:.1f}'


def diff(base: tuple, other: tuple) -> str:
    """``other - base`` in points with its standard error and z (from two (n, total))."""
    (p1, se1), (p2, se2) = rate(*base), rate(*other)
    if p1 is None or p2 is None:
        return '–'
    se = math.hypot(se1, se2)
    z = '' if se == 0 else f' (z {(p2 - p1) / se:+.1f})'
    return f'{100 * (p2 - p1):+.1f} ± {100 * se:.1f}{z}'


def table(header: list, rows: list) -> list:
    return ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|',
            *('| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows)]


def markdown(records: list, names: list, top_rooms: int = 12) -> str:
    """The check-in summary of one or more runs-250 records (the first is the baseline of the differences)."""
    # pylint: disable=too-many-locals
    labels = [f"{r['version']} {r.get('team_name') or 'main'}" for r in records]
    lines = [f"### Runs-250: {' vs '.join(labels)}", '']
    rows = []
    for label, name, r in zip(labels, names, records):
        n = r['seeds']['runs']
        median = r['triforce_frames_median']
        rows.append([label, name, n, with_se(r['won'], n), with_se(r['endings']['death'], n),
                     with_se(r['endings']['stuck'], n), r['furthest_step']['median'],
                     ', '.join(f'{k} {v}' for k, v in r['stuck_kinds'].items()) or '–',
                     median.get('D9') or '–', r.get('wall_seconds') or '–', r.get('decisions_per_second') or '–'])
    lines += table(['build', 'record', 'runs', 'won %', 'died %', 'stuck %', 'median step', 'stuck kinds',
                    'median frames to the end', 'wall s', 'decisions/s'], rows) + ['']
    header = ['dungeon'] + [f'{label} %' for label in labels] + [f'{label} − {labels[0]}' for label in labels[1:]]
    rows = []
    for level in ORDER:
        key = f'D{level}'
        counts = [(r['triforces'][key], r['seeds']['runs']) for r in records]
        rows.append([key, *(with_se(*c) for c in counts), *(diff(counts[0], c) for c in counts[1:])])
    lines += ['Runs that took each triforce (± binomial standard error, points):', ''] + table(header, rows) + ['']
    rows = [[f'D{level}', *(r['triforce_frames_median'][f'D{level}'] or '–' for r in records)] for level in ORDER]
    lines += ["Median frames to each triforce (the chain's count, from the game's start after the file select; a "
              "replay adds the power-on boot and the file select):", ''] + table(['dungeon', *labels], rows) + ['']
    # Rooms: the ones that lost the most runs (deaths + stucks) in any record, by route step.
    by_room = [{(e['room'], e['step']): e for e in (dict(zip(r['room_fields'], row)) for row in r['rooms'])}
               for r in records]
    losses = Counter()
    for rooms in by_room:
        for key, e in rooms.items():
            losses[key] = max(losses[key], e['deaths'] + e['stucks'])
    chosen = sorted([k for k, n in losses.most_common() if n][:top_rooms], key=lambda k: (k[1] is None, k[1] or 0))
    header = ['room (step)', 'family'] + [f'{label}: lost/reached, hits/visit, hearts/visit' for label in labels] + [
        f'lost % {label} − {labels[0]}' for label in labels[1:]]
    rows = []
    for key in chosen:
        cells, counts, family = [], [], None
        for rooms in by_room:
            e = rooms.get(key)
            if e is None:
                cells.append('not reached')
                counts.append((0, 0))
                continue
            family = family or e['family']
            lost = e['deaths'] + e['stucks']
            counts.append((lost, e['runs']))
            cells.append(f"{lost}/{e['runs']} ({pct(lost / e['runs'])}%; {e['deaths']} died, {e['stucks']} stuck), "
                         f"{e['hits'] / e['visits']:.2f}, {e['hearts_lost'] / e['visits']:.2f}")
        rows.append([f"{key[0]} ({'off route' if key[1] is None else key[1]})", family, *cells,
                     *(diff(counts[0], c) for c in counts[1:])])
    lines += [f'The {len(chosen)} rooms that lost the most runs (a run lost there died or stuck; reached: runs that '
              'entered it at that route step):', ''] + table(header, rows) + ['']
    for label, r in zip(labels, records):
        kept = '; '.join(f"seed {k['seed']} {'+'.join(k['reasons'])} (step {k['furthest_step']}, {k['frames']} "
                         f"frames, {k['cheated_bombs_keys']} bombs/keys and {k['cheated_rupees']} rupees cheated: "
                         f"{k['ending']})" for k in r['kept'])
        lines.append(f'- {label} kept: {kept}.')
    return '\n'.join(lines) + '\n'


def analyze(paths: list, out=None, top_rooms: int = 12) -> str:
    records = [json.loads(Path(p).read_text()) for p in paths]
    text = markdown(records, [Path(p).name for p in paths], top_rooms)
    if out:
        Path(out).write_text(text)
    print(text)
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p_run = sub.add_parser('run', help="play the runs, then report")
    p_run.add_argument('--runs', type=int, default=100)
    p_run.add_argument('--first-seed', type=int, default=0)
    p_run.add_argument('--parallel', type=int, default=10)
    p_run.add_argument('--models-dir', default=None, help="a team other than main's (e.g. one network in every slot)")
    p_run.add_argument('--power-on', action='store_true',
                       help="start each run from console power on (the golden races' start) instead of the start state")
    p_run.add_argument('--out', default=str(ROOT / 'target' / 'full-runs' / time.strftime('%y%m%d-%H%M%S')))
    p_report = sub.add_parser('report', help="report an earlier run's folder")
    p_report.add_argument('out')
    p_250 = sub.add_parser('runs250', help="the 250-run test of a build from power on, then stats")
    p_250.add_argument('--runs', type=int, default=RUNS250)
    p_250.add_argument('--first-seed', type=int, default=0)
    p_250.add_argument('--parallel', type=int, default=10)
    p_250.add_argument('--models-dir', default=str(ROOT / 'models'),
                       help="the models root whose pointers are frozen and played (default <repo>/models)")
    p_250.add_argument('--team', default=None, help="a name for a team other than main's (the record's file name)")
    p_250.add_argument('--out', default=str(ROOT / 'target' / 'runs250' / time.strftime('%y%m%d-%H%M%S')))
    p_250.add_argument('--docs', default=str(DOCS), help="where the long-term record goes (default docs/runs250)")
    p_250.add_argument('--no-archive', action='store_true', help="don't archive the kept replays to the backup")
    p_stats = sub.add_parser('stats', help="the runs-250 record and kept replays of a runs250 folder")
    p_stats.add_argument('out')
    p_stats.add_argument('--team', default=None)
    p_stats.add_argument('--docs', default=str(DOCS))
    p_stats.add_argument('--no-archive', action='store_true')
    p_analyze = sub.add_parser('analyze', help="compare runs-250 records (the first is the baseline)")
    p_analyze.add_argument('records', nargs='+')
    p_analyze.add_argument('--markdown', default=None, help="also write the Markdown summary here")
    p_analyze.add_argument('--rooms', type=int, default=12, help="rooms in the per-room table")
    args = parser.parse_args(argv)
    if args.command == 'run':
        return run(args)
    if args.command == 'runs250':
        return runs250(args)
    if args.command == 'analyze':
        analyze(args.records, args.markdown, args.rooms)
        return 0
    sys.path.insert(0, str(ROOT))
    from triforce.jobs import require_job  # pylint: disable=import-outside-toplevel
    require_job()               # crunching and archiving many replays: a job (--timeout states how long)
    if args.command == 'stats':
        stats(Path(args.out), args.team, Path(args.docs), archive=not args.no_archive)
        return 0
    report(Path(args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
