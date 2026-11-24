"""Judging a teacher of the fresh generation in chains, and setting any measurement beside the baseline (round 40).

    triforce eval-batch NAME --teacher GROUP=FAMILY/RUN [--teacher ...] [--base TEAM] [--runs N] [--plan]
    triforce judge plan GROUP [GROUP...] [--route d3first]
    triforce judge compare TAG [--base r40-2015-1k] [--chains-dir DIR] [--base-dir DIR] [--output FILE]

**Where a teacher plays** (``teacher_plan``).  A route's room statistics cut it into scopes
(``room_stats.ROUTE_SCOPES``): per dungeon the leg from the previous triforce through the dungeon (``to-dN``: the file
of the segment that starts after the previous dungeon, ``full`` for the first) and the dungeon from its own entrance
(``dN``), both cut at the dungeon's exit.  A teacher is judged on a scope when one of its families plays a route step
inside it (``_native.route_steps``: each step's ``specialist``), on that scope's own stretch only: its chain stops at
the cut (``--stop-at step:N``), so a group that plays only in Dungeons 3 to 5 is not chained through Dungeon 9.  The
leg and the entrance are judged separately: a group that plays only on the overworld is not chained from a dungeon's
entrance, where none of its families plays.

**The batch** (``eval_batch --teacher``): the base team (``--base``, default the team of record) with the teacher's
run in every slot of its families (``store.group_slots``: ``BASE+@GROUP=FAMILY/RUN``), one chain per judged scope
written as ``<chains-dir>/<NAME>-<file>.jsonl`` (the names ``room-stats``, ``triforce stats`` and this module read)
and filed by the job in the store, then the yardstick report and room statistics, then ``compare`` against the
baseline (``<chains-dir>/<NAME>-judge.md``).  ``--plan`` prints the scopes, the team spec and the chain commands and
runs nothing.

**The comparison** (``compare``): for every dungeon whose scopes the measurement ``TAG`` has (any batch: a judged
teacher's, a ``--chains`` tag's), per start (from the previous triforce, from the entrance): the base's rate over all
its runs and on the measurement's seeds, the measurement's, their difference paired by seed with its standard error
(unpaired when no seed is shared), the three rooms where most of the measurement's runs ended (deaths, stuck and other
endings, per run, the base's beside), hearts lost a run, and the route's tracked goals per run where the records carry
them (``run_stats.goal_run_values``; 2.0.0.15's records predate them).  A team that is the base in most slots and a
teacher in some plays the base's runs elsewhere, so its paired difference is the teacher's effect.  Files are found in
``--chains-dir`` (default ``target/r56-chains``), else through the store's index of filed chain records (``kind
'chain'``, named ``<TAG>-<file>.jsonl``; their ``.meta.json`` filed as ``chain-meta``), so the baseline of round 40
(``BASELINE``, job 615's) is read from the store and no worktree."""

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import room_stats
from .room_stats import Scope

BASELINE = 'r40-2015-1k'        # 2.0.0.15's team, 1000 runs a segment (round40.md "The baseline ..."; store job 615)
MOST = 3                        # rooms shown per scope


# ------------------------------------------------------------------------------------------------ where it plays

@dataclass(frozen=True)
class Judged:
    """One scope a teacher is judged on: the scope, its first and cut step, the families of the teacher playing in
    it."""
    dungeon: str
    start_kind: str                     # 'leg' (from the previous triforce) or 'entrance'
    scope: Scope
    start_step: int
    cut_step: Optional[int]
    families: tuple

    def chain_args(self, runs: int, route: str) -> list[str]:
        """The ``eval-chain`` arguments of this scope's chain: from its start, to its cut."""
        return ['--runs', str(runs), '--act', 'model', '--route', route,
                *(['--from', self.scope.start] if self.scope.start != 'start' else []),
                *(['--stop-at', f'step:{self.cut_step}'] if self.cut_step is not None else [])]


def dungeon_scopes(route: str) -> list[tuple[str, Scope, Scope]]:
    """(dungeon, the leg's scope ``to-dN``, the entrance's scope ``dN``) of every dungeon of ``route``, in route
    order."""
    table = {scope.name: scope for scope in room_stats.route_scopes(route)}
    pairs = [(name[len('to-'):], scope, table[name[len('to-'):]]) for name, scope in table.items()
             if scope.kind == 'segment' and name[len('to-'):] in table]
    return pairs


def scope_steps(plan: list[dict], scope: Scope) -> range:
    """The route steps a scope plays: from its start's step to the step before its cut (to the end without one)."""
    start = room_stats.state_step(plan, scope.start)
    end = room_stats.state_step(plan, scope.cut) if scope.cut else len(plan)
    return range(start, end)


def teacher_plan(families, route: str, plan: Optional[list[dict]] = None) -> list[Judged]:
    """Every scope of ``route`` in which one of ``families`` plays a step, leg before entrance, in route order."""
    if plan is None:
        from ._native import route_steps  # pylint: disable=import-outside-toplevel
        plan = list(route_steps(route))
    families, judged = set(families), []
    for dungeon, leg, entrance in dungeon_scopes(route):
        for kind, scope in (('leg', leg), ('entrance', entrance)):
            steps = scope_steps(plan, scope)
            playing = tuple(sorted({plan[i]['specialist'] for i in steps} & families))
            if playing:
                cut = room_stats.state_step(plan, scope.cut) if scope.cut else None
                judged.append(Judged(dungeon, kind, scope, steps.start, cut, playing))
    return judged


def teacher_families(teachers: list[str]) -> tuple[str, dict[str, tuple]]:
    """(route, group -> families) of ``GROUP=FAMILY/RUN`` items; one route for all (ValueError otherwise)."""
    from . import store  # pylint: disable=import-outside-toplevel
    known, found, routes = store.groups(), {}, set()
    for item in teachers:
        group, sep, run = item.partition('=')
        if not sep or not group or '/' not in run:
            raise ValueError(f"--teacher {item}: GROUP=FAMILY/RUN")
        if group not in known:
            raise ValueError(f"--teacher {item}: no group {group!r} (groups: {', '.join(sorted(known))})")
        routes.add(known[group][0])
        found[group] = known[group][1]
    if len(routes) != 1:
        raise ValueError(f"--teacher: the groups play on several routes {sorted(routes)}")
    return routes.pop(), found


def teacher_spec(base: str, teachers: list[str]) -> str:
    """The team spec of a base with each teacher in its group's slots: ``BASE+@GROUP=FAMILY/RUN...``."""
    from .store import GROUP_SLOT  # pylint: disable=import-outside-toplevel
    return base + ''.join(f'+{GROUP_SLOT}{item}' for item in teachers)


def plan_text(judged: list[Judged]) -> list[str]:
    """The plan's table: a row per judged scope."""
    lines = ['| dungeon | start | scope | chain file | from | to | families of the teacher playing there |',
             '|---|---|---|---|---|---|---|']
    for item in judged:
        start = 'previous triforce' if item.start_kind == 'leg' else 'its entrance'
        stop = f'step {item.cut_step} ({item.scope.cut})' if item.cut_step is not None else 'the end of the game'
        lines.append(f"| {item.dungeon} | {start} | {item.scope.name} | {item.scope.file} | step {item.start_step} "
                     f"({item.scope.start}) | {stop} | {', '.join(item.families)} |")
    return lines


# --------------------------------------------------------------------------------------------- finding the files

def _store_files(tag: str) -> tuple[dict[str, Path], dict[str, Path]]:
    """(file -> chain path, file -> meta path) of a tag's chain records filed in the store (the latest of each)."""
    from . import paths, store  # pylint: disable=import-outside-toplevel
    if not paths.store_exists():
        return {}, {}
    chains, metas, prefix = {}, {}, f'{tag}-'
    for row in store.outputs(kind='chain'):
        name = row['name'] or Path(row['path']).name
        if name.startswith(prefix) and name.endswith('.jsonl'):
            chains[name[len(prefix):-len('.jsonl')]] = store.root() / row['path']
    for row in store.outputs(kind='chain-meta'):
        name = row['name'] or Path(row['path']).name
        if name.startswith(prefix) and name.endswith('.jsonl.meta.json'):
            metas[name[len(prefix):-len('.jsonl.meta.json')]] = store.root() / row['path']
    return chains, metas


def find_files(tag: str, chains_dir: Optional[Path]) -> dict[str, tuple[Path, Optional[Path]]]:
    """File name (``full``, ``post-d3``, ``d4``...) -> (chain records, their ``.meta.json`` or None) of ``tag``:
    ``<chains_dir>/<tag>-<file>.jsonl`` where it exists, else the store's filed copy."""
    found: dict = {}
    if chains_dir is not None:
        for path in sorted(Path(chains_dir).glob(f'{tag}-*.jsonl')):
            meta = path.with_name(path.name + '.meta.json')
            found[path.name[len(tag) + 1:-len('.jsonl')]] = (path, meta if meta.is_file() else None)
    chains, metas = _store_files(tag)
    for name, path in chains.items():
        found.setdefault(name, (path, metas.get(name)))
    return found


# ---------------------------------------------------------------------------------------------- per-run results

@dataclass
class RunResult:
    """One run within one scope."""
    seed: int
    reached: bool
    hearts_lost: float
    ended_in: Optional[str]             # room key where it ended inside the scope, None if it did not
    goals: dict


def scope_runs(runs: list[dict], plan: list[dict], scope: Scope) -> dict[int, RunResult]:
    """Seed -> ``RunResult`` of each run within ``scope`` (``room_stats.cut_run``)."""
    from .run_stats import goal_run_values  # pylint: disable=import-outside-toplevel
    start = room_stats.state_step(plan, scope.start)
    cut = room_stats.state_step(plan, scope.cut) if scope.cut else None
    out = {}
    for run in runs:
        visits, reached, ended = room_stats.cut_run(plan, run, start, cut)
        records = [record for record, _ in visits]
        lost = sum(max(0.0, float(r.get('health_in') or 0) - float(r.get('health_out') or 0)) for r in records)
        where = room_stats.room_key(*room_stats._location(records[-1])) if ended is not None and records else None  # pylint: disable=protected-access
        out[int(run['seed'])] = RunResult(int(run['seed']), bool(reached), lost, where, goal_run_values(records))
    return out


def mean_se(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    """Mean and its standard error (None without values; se None for one value)."""
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var / len(values))


def difference(arm: dict[int, RunResult], base: dict[int, RunResult], value) -> dict:
    """The arm's mean of ``value`` minus the base's: paired by seed where seeds are shared (``paired`` True, ``n``
    the pairs), else of the two means with both standard errors."""
    shared = sorted(set(arm) & set(base))
    if shared:
        mean, se = mean_se([value(arm[s]) - value(base[s]) for s in shared])
        return {'diff': mean, 'se': se, 'paired': True, 'n': len(shared)}
    (a, sa), (b, sb) = mean_se([value(r) for r in arm.values()]), mean_se([value(r) for r in base.values()])
    if a is None or b is None:
        return {'diff': None, 'se': None, 'paired': False, 'n': 0}
    se = None if sa is None or sb is None else math.sqrt(sa ** 2 + sb ** 2)
    return {'diff': a - b, 'se': se, 'paired': False, 'n': len(arm)}


def _rate(results: dict[int, RunResult], seeds=None) -> dict:
    chosen = [r for s, r in results.items() if seeds is None or s in seeds]
    n = len(chosen)
    rate = sum(r.reached for r in chosen) / n if n else None
    return {'n': n, 'rate': rate, 'se': math.sqrt(rate * (1 - rate) / n) if n else None}


def _endings(results: dict[int, RunResult]) -> dict[str, float]:
    """Room key -> share of the runs that ended there."""
    counts = Counter(r.ended_in for r in results.values() if r.ended_in)
    return {room: count / len(results) for room, count in counts.most_common()} if results else {}


GOAL_KEYS = ('goal_crossed_kills', 'goal_wraps_missed', 'goal_planned_hits', 'goal_carrier_bombs')


def _goals(results: dict[int, RunResult]) -> Optional[dict]:
    """Per goal measure its mean a run, None when the records carry none."""
    out = {}
    for key in GOAL_KEYS:
        values = [r.goals.get(key) for r in results.values() if r.goals.get(key) is not None]
        if values:
            out[key] = mean_se(values)[0]
    return out or None


def scope_comparison(arm: dict[int, RunResult], base: Optional[dict[int, RunResult]]) -> dict:
    """The numbers of one scope: rates, the difference, the rooms runs ended in, hearts lost, goals."""
    hearts = mean_se([r.hearts_lost for r in arm.values()])
    out = {'arm': _rate(arm), 'arm_hearts_lost': {'mean': hearts[0], 'se': hearts[1]}, 'arm_ends': _endings(arm),
           'arm_goals': _goals(arm), 'base': None}
    if base:
        base_hearts = mean_se([r.hearts_lost for r in base.values()])
        out.update(base=_rate(base), base_on_seeds=_rate(base, set(arm)),
                   base_hearts_lost={'mean': base_hearts[0], 'se': base_hearts[1]}, base_ends=_endings(base),
                   base_goals=_goals(base), rate_diff=difference(arm, base, lambda r: float(r.reached)),
                   hearts_diff=difference(arm, base, lambda r: r.hearts_lost))
    return out


# ------------------------------------------------------------------------------------------------------ the text

_NAMES: Optional[dict] = None


def owner_room(key: str) -> str:
    """``3:0x4d`` -> ``3x4d``, ``0:0x1f`` -> ``ox1f`` (the owner's notation), with game.yaml's name where it has
    one."""
    global _NAMES  # pylint: disable=global-statement
    if _NAMES is None:
        from ._native import game_rooms  # pylint: disable=import-outside-toplevel
        _NAMES = {room: info['name'].rstrip('*') for room, info in game_rooms().items() if info.get('name')}
    level, room, cave = room_stats.parse_room_key(key)
    text = f"{'o' if level == 0 else level}x{room:02x}" + (' cave' if cave else '')
    name = _NAMES.get((level, room, cave))
    return f"{text} {name}" if name else text


def _pct(value) -> str:
    return '-' if value is None else f"{100 * value:.1f}%"


def _diff(diff: Optional[dict], scale: float = 100.0, unit: str = '') -> str:
    if not diff or diff['diff'] is None:
        return '-'
    se = '' if diff['se'] is None else f" ± {scale * diff['se']:.{1 if scale == 100 else 2}f}"
    how = f"paired, {diff['n']}" if diff['paired'] else 'unpaired'
    return f"{scale * diff['diff']:+.{1 if scale == 100 else 2}f}{se}{unit} ({how})"


def table(rows: list[dict]) -> list[str]:
    """The comparison's Markdown table, a row per dungeon and start."""
    goals = any(row['numbers'].get('arm_goals') for row in rows)
    head = ('| Dungeon | Start | Base (all runs) | Base on these seeds | Measured | Difference | Rooms where most '
            'runs end (base) | Hearts lost a run (base) |' + (' Goals a run (base) |' if goals else ''))
    lines = [head, '|' + '---|' * (8 + goals)]
    for row in rows:
        n = row['numbers']
        base, arm = n.get('base'), n['arm']
        ends = ', '.join(f"{owner_room(room)} {_pct(share)} ({_pct((n.get('base_ends') or {}).get(room, 0.0))})"
                         for room, share in list(n['arm_ends'].items())[:MOST]) or 'none'
        hearts = n['arm_hearts_lost']['mean']
        base_hearts = (n.get('base_hearts_lost') or {}).get('mean')
        cells = [row['dungeon'], 'previous triforce' if row['start'] == 'leg' else 'entrance',
                 f"{_pct(base['rate'])} of {base['n']}" if base else 'no record',
                 _pct(n['base_on_seeds']['rate']) if base else '-', f"{_pct(arm['rate'])} of {arm['n']}",
                 _diff(n.get('rate_diff')), ends,
                 f"{hearts:.2f} ({'-' if base_hearts is None else f'{base_hearts:.2f}'}); "
                 f"{_diff(n.get('hearts_diff'), 1.0)}" if hearts is not None else '-']
        if goals:
            mine, theirs = n.get('arm_goals') or {}, n.get('base_goals') or {}
            cells.append(', '.join(f"{key[len('goal_'):]} {value:.2f} ({'-' if key not in theirs else f'{theirs[key]:.2f}'})"
                                   for key, value in mine.items()) or '-')
        lines.append('| ' + ' | '.join(cells) + ' |')
    return lines


# ------------------------------------------------------------------------------------------------------ compare

def _read(path: Path, meta: Optional[Path]) -> tuple[list[dict], dict]:
    runs = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    provenance = json.loads(meta.read_text(encoding='utf-8')).get('provenance', {}) if meta else {}
    return runs, provenance


def _team_name(provenance: dict) -> Optional[str]:
    team = provenance.get('team')
    return team.get('name') if isinstance(team, dict) else None


def compare(tag: str, base: str = BASELINE, chains_dir: Optional[Path] = None, base_dir: Optional[Path] = None,
            route: Optional[str] = None) -> dict:
    """The comparison of measurement ``tag`` with ``base`` (``find_files`` each; ``base_dir`` the base's folder,
    default ``chains_dir`` then the store): a row per dungeon and start the measurement has."""
    from ._native import route_steps  # pylint: disable=import-outside-toplevel
    arm_files = find_files(tag, chains_dir)
    if not arm_files:
        raise ValueError(f"no chain records {tag}-*.jsonl in {chains_dir} or filed in the store")
    base_files = find_files(base, base_dir if base_dir is not None else chains_dir)
    if not base_files:
        raise ValueError(f"no chain records {base}-*.jsonl in {base_dir or chains_dir} or filed in the store")
    loaded: dict = {}

    def load(files, name):
        if name not in files:
            return None
        key = files[name][0]
        if key not in loaded:
            loaded[key] = _read(*files[name])
        return loaded[key]

    if route is None:
        first = load(arm_files, next(iter(arm_files)))
        route = first[1].get('route') or next((r.get('route') for r in first[0] if r.get('route')), None)
        if route is None:
            raise ValueError(f"{tag}: no route recorded (pass --route)")
    plan = list(route_steps(route))
    rows, teams = [], {}
    for dungeon, leg, entrance in dungeon_scopes(route):
        for start, scope in (('leg', leg), ('entrance', entrance)):
            arm = load(arm_files, scope.file)
            if arm is None:
                continue
            other = load(base_files, scope.file)
            teams.setdefault('measured', _team_name(arm[1]))
            if other is not None:
                teams.setdefault('base', _team_name(other[1]))
            numbers = scope_comparison(scope_runs(arm[0], plan, scope),
                                       scope_runs(other[0], plan, scope) if other is not None else None)
            rows.append({'dungeon': dungeon, 'start': start, 'scope': scope.name, 'file': scope.file,
                         'numbers': numbers, 'arm_file': str(arm_files[scope.file][0]),
                         'base_file': str(base_files[scope.file][0]) if scope.file in base_files else None})
    return {'tag': tag, 'base': base, 'route': route, 'teams': teams, 'rows': rows}


def report(result: dict) -> str:
    """The comparison as Markdown."""
    teams = result['teams']
    lines = [f"# {result['tag']} beside {result['base']} ({result['route']})", '',
             f"Measured: team {teams.get('measured') or '(not recorded)'}; base: team "
             f"{teams.get('base') or '(not recorded)'}.  Rates are the runs that reached the scope's cut (the "
             "dungeon's exit); the difference is paired by seed (the measurement's seeds in the base's runs).  Rooms: "
             "the share of runs that ended there (death, stuck or another ending), the base's in brackets.", '']
    if not result['rows']:
        lines.append(f"{result['tag']} has no file of any dungeon's scope.")
    else:
        lines += table(result['rows'])
        if not any(row['numbers'].get('arm_goals') for row in result['rows']):
            lines += ['', "The records carry no tracked goals (`goal_*`, round 40): none are shown."]
    return '\n'.join(lines)


def main(argv=None) -> int:
    """``triforce judge plan|compare``."""
    parser = argparse.ArgumentParser(prog='triforce judge', description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    plan_parser = sub.add_parser('plan', help="the scopes a group's families play in (no run)")
    plan_parser.add_argument('groups', nargs='+')
    plan_parser.add_argument('--route', default=None, help="default the groups' route")
    cmp = sub.add_parser('compare', help="a measurement beside the baseline, per dungeon (Markdown)")
    cmp.add_argument('tag')
    cmp.add_argument('--base', default=BASELINE, help=f"the base measurement's tag (default {BASELINE})")
    cmp.add_argument('--chains-dir', type=Path, default=room_stats.CHAINS_DIR,
                     help="where the measurement's files are (else the store's filed copies)")
    cmp.add_argument('--base-dir', type=Path, default=None, help="where the base's files are (default --chains-dir, "
                                                                 "then the store)")
    cmp.add_argument('--route', default=None, help="default the measurement's recorded route")
    cmp.add_argument('--output', type=Path, default=None, help="also write the numbers as JSON")
    cmp.add_argument('--markdown', type=Path, default=None, help="also write the Markdown to this file")
    args = parser.parse_args(argv)
    try:
        if args.command == 'plan':
            from . import store  # pylint: disable=import-outside-toplevel
            known = store.groups()
            for group in args.groups:
                if group not in known:
                    raise ValueError(f"no group {group!r} (groups: {', '.join(sorted(known))})")
                route, families = known[group]
                print(f"## {group} on {args.route or route}: {', '.join(families)}\n")
                print('\n'.join(plan_text(teacher_plan(families, args.route or route))) + '\n')
            return 0
        result = compare(args.tag, args.base, args.chains_dir, args.base_dir, args.route)
    except (ValueError, room_stats.RoomStatsError) as error:
        print(f"triforce judge: error: {error}", file=sys.stderr)
        return 2
    text = report(result)
    print(text)
    if args.output:
        from .events import write_json_atomic  # pylint: disable=import-outside-toplevel
        write_json_atomic(args.output, result)
    if args.markdown:
        args.markdown.write_text(text + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
