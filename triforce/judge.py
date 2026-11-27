"""Judging a teacher of the fresh generation in chains, and setting any measurement beside the baseline (round 40).

    triforce eval-batch NAME --teacher GROUP=FAMILY/RUN[@itN|@latest] [--base TEAM] [--runs N] [--plan]
    triforce judge plan GROUP [GROUP...] [--route d3first]
    triforce judge compare TAG [--base r40-2015-1k] [--chains-dir DIR] [--base-dir DIR] [--output FILE]
    triforce judge checkpoints GROUP=FAMILY/RUN [--its 114,228|all] [--dungeons dN,...] [--scopes S,...] --runs N
                               [--plan | --submit] [--agent A] [--tag TAG] [--chains-dir DIR]

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
(``BASELINE``, job 615's) is read from the store and no worktree.

**Several checkpoints, small** (``checkpoints``, round 41): a teacher's numbered checkpoints (each a sealed run of its
own, ``store.checkpoint_run``, made by its first chain's job) on the same scopes and the same seeds, one chain a
checkpoint and scope so each fits the queue's express place: ``--plan`` prints one ``triforce job run --express
--timeout 10m --detach`` command a chain not yet played (``eval-batch <tag>-it<NNNNNN> --teacher
GROUP=FAMILY/RUN@itN --scopes S --no-report``), ``--submit`` queues them; run without either it reads the chains
played (in ``--chains-dir`` or filed in the store), skips nothing it has and lists the rest, and writes one table by
iteration (steps, each scope's rate ± its standard error, the paired difference with the baseline on those seeds,
hearts lost a run, the rooms where most runs end, the tracked goals) as ``<tag>-checkpoints.md`` / ``.json`` beside
the chains, filed in the store once every piece is in.  A comparison of a team holding such a run names its
iteration and steps (``played_checkpoints``)."""

import argparse
import hashlib
import json
import math
import os
import shlex
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


def select_scopes(judged: list[Judged], dungeons: Optional[str] = None, scopes: Optional[str] = None) -> list[Judged]:
    """``judged`` cut to ``dungeons`` (``d3,d4``) and ``scopes`` (``to-d3,d3``: the plan's scope names); ValueError
    naming what the group plays for one it does not."""
    if dungeons:
        wanted = [name for name in dungeons.split(',') if name]
        played = list(dict.fromkeys(item.dungeon for item in judged))
        if not wanted or set(wanted) - set(played):
            raise ValueError(f"--dungeons {dungeons}: the group plays in {', '.join(played)}")
        judged = [item for item in judged if item.dungeon in wanted]
    if scopes:
        wanted = [name for name in scopes.split(',') if name]
        played = [item.scope.name for item in judged]
        if not wanted or set(wanted) - set(played):
            raise ValueError(f"--scopes {scopes}: the group is judged on {', '.join(played)}")
        judged = [item for item in judged if item.scope.name in wanted]
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


def _ends(n: dict) -> str:
    return ', '.join(f"{owner_room(room)} {_pct(share)} ({_pct((n.get('base_ends') or {}).get(room, 0.0))})"
                     for room, share in list(n['arm_ends'].items())[:MOST]) or 'none'


def _hearts(n: dict) -> str:
    hearts = n['arm_hearts_lost']['mean']
    base_hearts = (n.get('base_hearts_lost') or {}).get('mean')
    return (f"{hearts:.2f} ({'-' if base_hearts is None else f'{base_hearts:.2f}'}); "
            f"{_diff(n.get('hearts_diff'), 1.0)}" if hearts is not None else '-')


def _goal_text(n: dict) -> str:
    mine, theirs = n.get('arm_goals') or {}, n.get('base_goals') or {}
    return ', '.join(f"{key[len('goal_'):]} {value:.2f} ({'-' if key not in theirs else f'{theirs[key]:.2f}'})"
                     for key, value in mine.items()) or '-'


def checkpoint_label(checkpoints: dict) -> Optional[str]:
    """``it 228, 3.74M steps`` when the measured team played one numbered checkpoint (``played_checkpoints``), the
    runs' labels joined when several."""
    labels = [f"it {c['iteration']}" + (f", {c['steps'] / 1e6:.2f}M steps" if c.get('steps') is not None else '')
              for c in checkpoints.values()]
    return '; '.join(labels) or None


def table(rows: list[dict], measured: Optional[str] = None) -> list[str]:
    """The comparison's Markdown table, a row per dungeon and start (``measured``: what the Measured column played,
    ``checkpoint_label``)."""
    goals = any(row['numbers'].get('arm_goals') for row in rows)
    head = (f"| Dungeon | Start | Base (all runs) | Base on these seeds | Measured{f' ({measured})' if measured else ''} "
            "| Difference | Rooms where most runs end (base) | Hearts lost a run (base) |"
            + (' Goals a run (base) |' if goals else ''))
    lines = [head, '|' + '---|' * (8 + goals)]
    for row in rows:
        n = row['numbers']
        base, arm = n.get('base'), n['arm']
        cells = [row['dungeon'], 'previous triforce' if row['start'] == 'leg' else 'entrance',
                 f"{_pct(base['rate'])} of {base['n']}" if base else 'no record',
                 _pct(n['base_on_seeds']['rate']) if base else '-', f"{_pct(arm['rate'])} of {arm['n']}",
                 _diff(n.get('rate_diff')), _ends(n), _hearts(n)]
        if goals:
            cells.append(_goal_text(n))
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


def played_checkpoints(provenance: dict) -> dict[str, dict]:
    """Run -> its numbered checkpoint (``store.checkpoint_info``: parent, iteration, steps, SHA-1) for each run the
    measured team put in a slot that ``store.checkpoint_run`` made from one."""
    from . import paths, store  # pylint: disable=import-outside-toplevel
    team = provenance.get('team')
    runs = sorted(set((team.get('overrides') or {}).values())) if isinstance(team, dict) else []
    out = {}
    for run in runs if paths.store_exists() else []:
        info = store.checkpoint_info(store.parse_run(run))
        if info:
            out[run] = {key: info.get(key) for key in ('parent', 'iteration', 'steps', 'checkpoint_sha1')}
    return out


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
    rows, teams, checkpoints = [], {}, {}
    for dungeon, leg, entrance in dungeon_scopes(route):
        for start, scope in (('leg', leg), ('entrance', entrance)):
            arm = load(arm_files, scope.file)
            if arm is None:
                continue
            other = load(base_files, scope.file)
            teams.setdefault('measured', _team_name(arm[1]))
            checkpoints.update(played_checkpoints(arm[1]))
            if other is not None:
                teams.setdefault('base', _team_name(other[1]))
            numbers = scope_comparison(scope_runs(arm[0], plan, scope),
                                       scope_runs(other[0], plan, scope) if other is not None else None)
            rows.append({'dungeon': dungeon, 'start': start, 'scope': scope.name, 'file': scope.file,
                         'numbers': numbers, 'arm_file': str(arm_files[scope.file][0]),
                         'base_file': str(base_files[scope.file][0]) if scope.file in base_files else None})
    return {'tag': tag, 'base': base, 'route': route, 'teams': teams, 'checkpoints': checkpoints, 'rows': rows}


def checkpoint_lines(checkpoints: dict) -> list[str]:
    """A line per run played from a numbered checkpoint: which, of which run, its steps and SHA-1."""
    return [f"- `{run}`: iteration {c['iteration']} of `{c['parent']}`"
            + (f", {c['steps']:,} steps" if c.get('steps') is not None else '')
            + f", checkpoint sha1 {(c.get('checkpoint_sha1') or '-')[:12]}" for run, c in sorted(checkpoints.items())]


def report(result: dict) -> str:
    """The comparison as Markdown."""
    teams = result['teams']
    checkpoints = result.get('checkpoints') or {}
    lines = [f"# {result['tag']} beside {result['base']} ({result['route']})", '',
             f"Measured: team {teams.get('measured') or '(not recorded)'}; base: team "
             f"{teams.get('base') or '(not recorded)'}.  Rates are the runs that reached the scope's cut (the "
             "dungeon's exit); the difference is paired by seed (the measurement's seeds in the base's runs).  Rooms: "
             "the share of runs that ended there (death, stuck or another ending), the base's in brackets.", '']
    if checkpoints:
        lines += ['Played from numbered checkpoints:', *checkpoint_lines(checkpoints), '']
    if not result['rows']:
        lines.append(f"{result['tag']} has no file of any dungeon's scope.")
    else:
        lines += table(result['rows'], checkpoint_label(checkpoints))
        if not any(row['numbers'].get('arm_goals') for row in result['rows']):
            lines += ['', "The records carry no tracked goals (`goal_*`, round 40): none are shown."]
    return '\n'.join(lines)


# ------------------------------------------------------------------------------------------ several checkpoints

@dataclass
class Piece:
    """One chain of ``judge checkpoints``: a checkpoint's iteration on one judged scope, in batch ``tag``."""
    iteration: int
    tag: str
    judged: Judged
    done: bool


@dataclass
class CheckpointsPlan:
    """What ``judge checkpoints`` judges: the teacher's group and run, the route, the pieces in order, and the control
    (``control``: the tag of the base team's own chains on these scopes and seeds, with ``with_`` in their slots too;
    ``control_team`` its team spec; ``control_missing`` its scopes not played yet)."""
    group: str
    parent: object                      # store.Run
    route: str
    runs: int
    tag: str
    chains_dir: Path
    base: Optional[str]
    pieces: list
    with_: list
    control: str
    control_team: str
    control_missing: list
    pending: list                       # --with checkpoints whose run is not made yet (store.checkpoint_run records)


def _safe_tag(text: str) -> str:
    return ''.join(c if c.isalnum() or c in '-_.' else '-' for c in text)


# The longest default tag `judge checkpoints` makes: a chain's file adds `-it<NNNNNN>-<scope file>.jsonl.meta.json`
# (about 40 bytes), a batch's folder in the store the same tag, and a file name may hold 255 bytes (Nov 27: ten
# `--with` runs in the control's tag, OSError 36, job #2289).
TAG_LIMIT = 100


def _digest8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _short_tag(text: str) -> str:
    """``text`` as a tag (``_safe_tag``) of at most ``TAG_LIMIT`` characters: a longer one keeps its start and ends in
    8 hex characters of its whole text's hash, so it stays stable and distinct."""
    text = _safe_tag(text)
    return text if len(text) <= TAG_LIMIT else f"{text[:TAG_LIMIT - 9]}-{_digest8(text)}"


def control_tag(team: str, resolved: list[str], runs: int) -> str:
    """The control's default tag: ``control-<team>[-w<8 hex>]-r<runs>``, the ``w`` part the hash of the sorted
    ``--with`` runs (their names are in the control's team spec, ``--team`` of its chains and their meta)."""
    others = f"-w{_digest8(chr(10).join(sorted(resolved)))}" if resolved else ''
    return _short_tag(f"control-{team}{others}-r{runs}")


def checkpoints_plan(teacher: str, its: Optional[str], runs: int, dungeons: Optional[str] = None,
                     scopes: Optional[str] = None, tag: Optional[str] = None, chains_dir: Optional[Path] = None,
                     base: Optional[str] = None, with_: Optional[list] = None,
                     control: Optional[str] = None) -> CheckpointsPlan:
    """The pieces of judging ``GROUP=FAMILY/RUN``'s numbered checkpoints (``its``: ``114,228`` or ``all``: every
    completely written one) on the group's scopes (``select_scopes``), ``runs`` a chain on seeds 0..runs-1: one chain
    a checkpoint and scope, in batch ``<tag>-it<NNNNNN>`` (``tag`` default ``<family>-<run>-r<runs>``, at most
    ``TAG_LIMIT`` characters), done when its records are in ``chains_dir`` or filed in the store (``find_files``).
    ``with_``: other groups' runs (``GROUP=FAMILY/RUN[@itN]``) in their slots too, for every piece and the control.  The
    control: the base team (with ``with_``) without the teacher on the same scopes and seeds, tag ``control`` (default
    ``control_tag``).  ValueError for a wrong argument."""
    from . import store  # pylint: disable=import-outside-toplevel
    route, groups = teacher_families([teacher])
    group = next(iter(groups))
    text, named = store.split_checkpoint(teacher.partition('=')[2])
    if named is not None:
        raise ValueError(f"{teacher}: name the run (GROUP=FAMILY/RUN); --its chooses its checkpoints")
    with_ = list(with_ or [])
    if with_:
        with_route, with_groups = teacher_families(with_)
        if with_route != route or group in with_groups:
            raise ValueError(f"--with {' '.join(with_)}: other groups of {route} than the teacher's ({group})")
    parent = store.parse_run(text)
    found = store.numbered_checkpoints(parent, store.get_run(parent)['status'] == store.SEALED)
    complete = [c['iteration'] for c in found if c['complete']]
    if its in (None, 'all'):
        chosen = complete
    else:
        try:
            chosen = [int(item.removeprefix('it')) for item in its.split(',') if item]
        except ValueError:
            raise ValueError(f"--its {its}: iterations (114,228 or it114,it228) or all") from None
        unknown = [it for it in chosen if it not in complete]
        if unknown:
            writing = [c['iteration'] for c in found if not c['complete']]
            raise ValueError(f"--its {its}: {parent.identity} has no completely written numbered checkpoint "
                             f"{', '.join(map(str, unknown))} (complete: {', '.join(map(str, complete)) or 'none'}"
                             + (f"; being written: {', '.join(map(str, writing))}" if writing else '') + ")")
    if not chosen:
        raise ValueError(f"{parent.identity} has no completely written numbered checkpoint yet")
    judged = select_scopes(teacher_plan(groups[group], route), dungeons, scopes)
    if not judged:
        raise ValueError(f"no family of {group} plays a step of {route}")
    chains_dir = Path(chains_dir or room_stats.CHAINS_DIR).resolve()
    tag = tag or _short_tag(f"{parent.family.replace('/', '-')}-{parent.name}-r{runs}")
    pieces = []
    for iteration in chosen:
        piece_tag = f"{tag}-it{iteration:06d}"
        have = find_files(piece_tag, chains_dir)
        pieces += [Piece(iteration, piece_tag, item, item.scope.file in have) for item in judged]
    resolved, pending = [], []
    for item in with_:
        name, _, run = item.partition('=')
        text, iteration = store.split_checkpoint(run)
        if iteration is not None:
            record = store.checkpoint_run(store.parse_run(text), iteration, make=False)
            if not record['exists']:
                pending.append(record)
            run = record['run']
        resolved.append(f"{name}={run}")
    team = store.read_team(base or 'current')['name']
    control_team = teacher_spec(team, resolved)
    control = control or control_tag(team, resolved, runs)
    have = find_files(control, chains_dir)
    control_missing = [item for item in judged if item.scope.file not in have]
    return CheckpointsPlan(group, parent, route, runs, tag, chains_dir, base, pieces, with_, control, control_team,
                           control_missing, pending)


def _job(agent: str, task: str, subtask: str, desc: str, timeout: str) -> list[str]:
    # The program beside this interpreter, not a bare name: `--submit` from a shell without the venv on its path
    # found no `triforce` (the lead's watcher, Nov 24).  The description says its meaningful part within the queue's
    # limit (`_others`, never the whole team spec: 381 characters with ten `--with`s, Nov 27); `fit_description` is
    # the last guard (`triforce job run` refuses a longer one).
    from .job import fit_description  # pylint: disable=import-outside-toplevel
    return [_triforce(), 'job', 'run', '--agent', agent, '--task', task, '--subtask', subtask,
            '--desc', fit_description(desc), '--express', '--timeout', timeout, '--detach', '--']


def _others(plan: CheckpointsPlan) -> str:
    """The ``--with`` runs in a job's description: their count, not their names (the command carries them)."""
    count = len(plan.with_)
    return '' if not count else f" with {count} other teacher{'s' if count > 1 else ''}' runs"


def _triforce() -> str:
    exe = Path(sys.executable).with_name('triforce')
    return str(exe) if exe.is_file() else 'triforce'


def piece_command(plan: CheckpointsPlan, piece: Piece, agent: str, task: str, timeout: str = '10m') -> list[str]:
    """The queue's command for one piece: an express job (``--timeout``, detached) of ``eval-batch`` with the teacher
    at that checkpoint on that scope alone (``--with`` the others), no report (``judge checkpoints`` collects)."""
    name = f"{plan.group}-it{piece.iteration:06d}-{piece.judged.scope.name}"
    return [*_job(agent, task, name, f"{plan.parent.identity} it {piece.iteration} on {piece.judged.scope.name}"
                                     f"{_others(plan)}, {plan.runs} runs", timeout),
            _triforce(), 'eval-batch', piece.tag,
            '--teacher', f"{plan.group}={plan.parent.identity}@it{piece.iteration}",
            *[arg for item in plan.with_ for arg in ('--with', item)],
            *(['--base', plan.base] if plan.base else []), '--scopes', piece.judged.scope.name,
            '--runs', str(plan.runs), '--no-report', '--chains-dir', str(plan.chains_dir)]


def control_command(plan: CheckpointsPlan, judged: Judged, agent: str, task: str, timeout: str = '10m') -> list[str]:
    """The queue's command for one scope of the control: the base team (with ``--with``'s runs) on that scope and
    the same seeds, as ``<control>-<file>.jsonl``."""
    output = plan.chains_dir / f"{plan.control}-{judged.scope.file}.jsonl"
    args = shlex.join([*judged.chain_args(plan.runs, plan.route), '--output', str(output)])
    base = plan.control_team.split('+', 1)[0]
    return [*_job(agent, task, f"control-{judged.scope.name}", f"the control of {plan.group}'s checkpoints: "
                  f"{base}{_others(plan)} on {judged.scope.name}, {plan.runs} runs", timeout),
            _triforce(), 'eval-batch', plan.control, '--team', plan.control_team,
            '--chain', f"{judged.scope.file}: {args}"]


def checkpoints_table(plan: CheckpointsPlan, base: str = BASELINE, base_dir: Optional[Path] = None) -> dict:
    """The finished pieces beside ``base`` (today's team on its own route) and the control (the base team on this
    code's route, ``plan.control``) on the same seeds: a row per checkpoint and scope (``scope_comparison`` with
    each), with the checkpoint's steps (its run's ``run.json``, ``store.checkpoint_info``)."""
    from . import store  # pylint: disable=import-outside-toplevel
    from ._native import route_steps  # pylint: disable=import-outside-toplevel
    steps_of = list(route_steps(plan.route))
    bases = {'base': find_files(base, base_dir if base_dir is not None else plan.chains_dir),
             'control': find_files(plan.control, plan.chains_dir)}
    cut: dict = {}

    def runs_of(which: str, scope: Scope):
        if (which, scope.file) not in cut:
            files = bases[which]
            cut[which, scope.file] = (scope_runs(_read(*files[scope.file])[0], steps_of, scope)
                                      if scope.file in files else None)
        return cut[which, scope.file]

    rows = []
    for piece in plan.pieces:
        if not piece.done:
            continue
        scope = piece.judged.scope
        path, meta = find_files(piece.tag, plan.chains_dir)[scope.file]
        runs, provenance = _read(path, meta)
        arm = scope_runs(runs, steps_of, scope)
        snapshot = store.checkpoint_run_name(plan.parent, piece.iteration)
        info = store.checkpoint_info(snapshot) or {}
        rows.append({'iteration': piece.iteration, 'steps': info.get('steps'),
                     'run': snapshot.identity if info else None,
                     'checkpoint_sha1': info.get('checkpoint_sha1'), 'dungeon': piece.judged.dungeon,
                     'start': piece.judged.start_kind, 'scope': scope.name, 'file': str(path),
                     'team': _team_name(provenance),
                     'numbers': scope_comparison(arm, runs_of('base', scope)),
                     'control_numbers': scope_comparison(arm, runs_of('control', scope)),
                     'rooms': darknut_rooms(runs, steps_of), 'curriculum': curriculum_points(info)})
    control_rooms = []
    for judged in dict((j.scope.file, j) for j in (p.judged for p in plan.pieces)).values():
        if judged.scope.file in bases['control']:
            rooms = darknut_rooms(_read(*bases['control'][judged.scope.file])[0], steps_of)
            if rooms is not None:
                control_rooms.append({'iteration': 'control', 'start': judged.start_kind, 'rooms': rooms})
    return {'teacher': f"{plan.group}={plan.parent.identity}", 'with': plan.with_, 'route': plan.route,
            'runs': plan.runs, 'tag': plan.tag, 'base': base, 'control': plan.control,
            'control_team': plan.control_team, 'rows': rows, 'control_rooms': control_rooms,
            'missing': [f"{p.tag}-{p.judged.scope.file}" for p in plan.pieces if not p.done]
            + [f"{plan.control}-{j.scope.file}" for j in plan.control_missing]}


def checkpoints_report(result: dict) -> str:
    """``judge checkpoints``' table as Markdown: a row per checkpoint and start, beside today's team on its own route
    (the baseline) and on the generation's route (the control)."""
    goals = any(row['numbers'].get('arm_goals') for row in result['rows'])
    lines = [f"# {result['teacher']}: numbered checkpoints ({result['route']}, {result['runs']} runs a chain)", '',
             f"Each checkpoint plays the teacher's slots (`+@GROUP=`"
             + (f", with {', '.join(result['with'])}" if result['with'] else '')
             + ", the rest of the base team as it is) on seeds 0-" + str(result['runs'] - 1) + ". Rates are the runs "
             "that reached the scope's cut, ± their standard error. Two bases on the same seeds, each difference "
             f"paired by seed: **today's team on its own route** (`{result['base']}`: 2.0.0.15's code and route) and "
             f"**today's team on the generation's route** (`{result['control']}`: `{result['control_team']}` on this "
             "code). Hearts lost a run, rooms where most runs end (their share) and goals: the generation's route's "
             "in brackets, the hearts' paired difference with it.", '',
             "| it | steps | dungeon | start | rate | today's team, own route, these seeds | difference | today's team, "
             "generation's route, these seeds | difference | hearts lost a run | rooms where most runs end |"
             + (' goals a run |' if goals else ''),
             '|' + '---|' * (11 + goals)]
    for row in result['rows']:
        n, c = row['numbers'], row['control_numbers']
        arm = n['arm']
        cells = [str(row['iteration']), '-' if row['steps'] is None else f"{row['steps'] / 1e6:.2f}M", row['dungeon'],
                 'previous triforce' if row['start'] == 'leg' else 'entrance',
                 f"{_pct(arm['rate'])} ± {100 * (arm['se'] or 0):.1f} of {arm['n']}",
                 _pct(n['base_on_seeds']['rate']) if n.get('base') else 'no record', _diff(n.get('rate_diff')),
                 _pct(c['base_on_seeds']['rate']) if c.get('base') else 'not played', _diff(c.get('rate_diff')),
                 _hearts(c), _ends(c)]
        if goals:
            cells.append(_goal_text(c))
        lines.append('| ' + ' | '.join(cells) + ' |')
    rooms = [row for row in result['rows'] if row.get('rooms')] + list(result.get('control_rooms') or [])
    if rooms:
        lines += ['', "Dungeon 3's Darknut rooms (counts over the runs; the control is today's team on the "
                      "generation's route; the curriculum as the training's `tasks/*/curriculum.json` read when the "
                      "checkpoint's run was made):", '', *darknut_rooms_lines(rooms)]
    if result['missing']:
        lines += ['', f"Not yet played ({len(result['missing'])}): {', '.join(result['missing'])}."]
    elif not goals:
        lines += ['', "The records carry no tracked goals (`goal_*`, round 40): none are shown."]
    return '\n'.join(lines)


def checkpoints_main(args) -> int:
    """``triforce judge checkpoints``: with ``--plan`` the pieces and a queue command for each one not played (the
    control's too); ``--submit`` queues those (express, detached; a ``--with`` checkpoint's run made first);
    otherwise the table of the pieces played, written beside the chains (``<tag>-checkpoints.md`` / ``.json``;
    ``-vs-<base>`` added beside another ``--base``) and, once every piece is in, filed in the store."""
    import subprocess  # pylint: disable=import-outside-toplevel
    plan = checkpoints_plan(args.teacher, args.its, args.runs, args.dungeons, args.scopes, args.tag,
                            args.chains_dir, args.team_base, args.with_, args.control)
    missing = [piece for piece in plan.pieces if not piece.done]
    its = sorted({piece.iteration for piece in plan.pieces})
    scopes = list(dict.fromkeys(piece.judged.scope.name for piece in plan.pieces))
    print(f"# {plan.group}={plan.parent.identity}: checkpoints {', '.join(map(str, its))} on "
          f"{', '.join(scopes)}, {plan.runs} runs a chain; tag {plan.tag}-it<NNNNNN>; chains in {plan.chains_dir}")
    print(f"{len(plan.pieces) - len(missing)} of {len(plan.pieces)} chains played; the control {plan.control} "
          f"({plan.control_team}): {len(scopes) - len(plan.control_missing)} of {len(scopes)}")
    if args.plan or args.submit:
        for record in plan.pending:
            if args.submit:
                from . import store  # pylint: disable=import-outside-toplevel
                record = store.checkpoint_run(store.parse_run(record['parent']), record['iteration'])
            print(f"--with: {record['run']} (iteration {record['iteration']} of {record['parent']}) "
                  + ('made now' if record['made'] else 'is made by --submit or the first piece played'))
        commands = [control_command(plan, judged, args.agent, args.task, args.timeout)
                    for judged in plan.control_missing]
        commands += [piece_command(plan, piece, args.agent, args.task, args.timeout) for piece in missing]
        for command in commands:
            print(shlex.join(command))
            if args.submit:
                subprocess.run(command, check=True)
        if commands:
            print("\nthen `triforce job wait UNIT...` and again without --plan: the table reads the chains played "
                  "(here or filed in the store) and lists the rest")
        return 0
    result = checkpoints_table(plan, args.base, args.base_dir)
    text = checkpoints_report(result)
    print('\n' + text)
    name = f"{plan.tag}-checkpoints" + ('' if args.base == BASELINE else f"-vs-{args.base}")
    markdown, numbers = plan.chains_dir / f"{name}.md", plan.chains_dir / f"{name}.json"
    from .events import write_json_atomic  # pylint: disable=import-outside-toplevel
    plan.chains_dir.mkdir(parents=True, exist_ok=True)
    markdown.write_text(text + '\n', encoding='utf-8')
    write_json_atomic(numbers, result)
    if result['missing']:
        print(f"\n{len(result['missing'])} chains not played; queue them with --plan (or --submit); nothing filed yet")
        return 0
    from . import store  # pylint: disable=import-outside-toplevel
    for path in (markdown, numbers):
        print(f"filed {store.file_output(path, 'judge', name=path.name)}")
    return 0


# The walkers' Dungeon 3 Darknut rooms in a chain (round 41, the lead's list for the generation's first checkpoints):
# 3:0x5b's first pass at drop cycle 0 (the fight for the first Darknut's bombs), 3:0x59 from N by the bombs Link
# brings (4 or 8 by the route), 3:0x69 in and out.
def darknut_rooms(runs: list[dict], plan: list[dict]) -> Optional[dict]:
    """Those rooms' figures over ``runs`` (chain records), found on the route by room (``plan``: ``route_steps``);
    None when the route has no such rooms or the runs play none of them."""
    def steps_of(room):
        return [i for i, s in enumerate(plan) if (s['level'], s['room'], s['in_cave']) == (3, room, False)]
    first_5b, first_59, passes_69 = steps_of(0x5b)[:1], steps_of(0x59)[:1], steps_of(0x69)[:2]
    if not first_5b or not first_59 or len(passes_69) < 2:
        return None
    visits: dict = {}
    for run in runs:
        for v in run.get('rooms') or []:
            visits.setdefault(v.get('step'), []).append(v)
    if not any(visits.get(step) for step in (first_5b[0], first_59[0], *passes_69)):
        return None
    exits = lambda vs: sum(1 for v in vs if v.get('outcome') == 'exit-correct')  # noqa: E731
    hits = lambda vs: (sum(v.get('hits') or 0 for v in vs) / len(vs)) if vs else None  # noqa: E731
    pass_5b = visits.get(first_5b[0], [])
    at_0 = [v for v in pass_5b if v.get('goal_cycle_in') == 0]
    by_bombs: dict = {}
    for v in visits.get(first_59[0], []):
        by_bombs.setdefault(v.get('bombs_in'), []).append(v)
    return {
        '5b_first': {'visits': len(pass_5b), 'cycle0': len(at_0), 'exits': exits(at_0),
                     'killed': sum(1 for v in at_0 if v.get('goal_carrier_kills')),
                     'dropped': sum(v.get('goal_carrier_bombs') or 0 for v in at_0),
                     'taken': sum(v.get('goal_carrier_bombs_taken') or 0 for v in at_0), 'hits': hits(at_0)},
        '59_n': {str(bombs): {'visits': len(vs), 'exits': exits(vs)}
                 for bombs, vs in sorted(by_bombs.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        '69': [{'visits': len(visits.get(step, [])), 'exits': exits(visits.get(step, [])),
                'hits': hits(visits.get(step, []))} for step in passes_69]}


def curriculum_points(info: dict, marker: str = '3_59n') -> Optional[str]:
    """``point/length`` of each seed state named ``marker`` in a checkpoint run's recorded curricula
    (``store.parent_curricula``), with the iteration they were read at; None when none was recorded."""
    curriculum = (info or {}).get('curriculum') or {}
    tracks = [f"{track['point']}/{track['length']}" for states in (curriculum.get('tasks') or {}).values()
              for state, track in sorted(states.items()) if marker in state]
    return f"{', '.join(tracks)} (read at it {curriculum.get('iteration')})" if tracks else None


def darknut_rooms_lines(rows: list[dict]) -> list[str]:
    """The table of ``darknut_rooms`` per checkpoint and start (and the control's, ``it`` 'control')."""
    def hit(value):
        return '-' if value is None else f"{value:.2f}"
    lines = ['| it | start | 3x5b first pass at cycle 0: visits (of all), exits, Darknut killed, bombs dropped / '
             'taken, hits a visit | 3x59 from N by bombs brought: visits / exits | 3x69 in: visits, exits, hits a '
             'visit | 3x69 out: visits, exits, hits a visit | curriculum 3_59n point/length |', '|---|---|---|---|---|---|---|']
    for row in rows:
        d = row['rooms']
        first = d['5b_first']
        bombs = ', '.join(f"{b}: {x['visits']} / {x['exits']}" for b, x in d['59_n'].items()) or 'none'
        passes = [f"{p['visits']}, {p['exits']}, {hit(p['hits'])}" for p in d['69']]
        lines.append(f"| {row['iteration']} | {'previous triforce' if row['start'] == 'leg' else 'entrance'} | "
                     f"{first['cycle0']} ({first['visits']}), {first['exits']}, {first['killed']}, "
                     f"{first['dropped']} / {first['taken']}, {hit(first['hits'])} | {bombs} | {passes[0]} | "
                     f"{passes[1]} | {row.get('curriculum') or '-'} |")
    return lines


def main(argv=None) -> int:
    """``triforce judge plan|compare|checkpoints``."""
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
    ck = sub.add_parser('checkpoints', help="a teacher's numbered checkpoints judged small, one chain a checkpoint "
                                            "and scope, in one table by iteration")
    ck.add_argument('teacher', metavar='GROUP=FAMILY/RUN')
    ck.add_argument('--its', default='all', metavar='114,228,...|all',
                    help="the iterations (default all: every completely written numbered checkpoint)")
    ck.add_argument('--dungeons', default=None, metavar='dN,...', help="the dungeons' scopes (default every one)")
    ck.add_argument('--scopes', default=None, metavar='SCOPE,...', help="only these scopes (to-d3, d3, ...)")
    ck.add_argument('--runs', type=int, required=True, help="runs a chain (seeds 0..N-1, the same for every one)")
    ck.add_argument('--tag', default=None, help="the chains' prefix (default <family>-<run>-r<runs>)")
    ck.add_argument('--chains-dir', type=Path, default=room_stats.CHAINS_DIR,
                    help="where the chains are written and read (else the store's filed copies)")
    ck.add_argument('--team-base', default=None, help="the team the teacher joins (eval-batch --base; default "
                                                      "current)")
    ck.add_argument('--base', default=BASELINE, help=f"the base measurement's tag: today's team on its own route "
                                                     f"(default {BASELINE})")
    ck.add_argument('--base-dir', type=Path, default=None, help="where the base's files are (default the store)")
    ck.add_argument('--with', dest='with_', action='append', default=[], metavar='GROUP=FAMILY/RUN[@itN]',
                    help="another group's run in its slots too, in every chain and the control (repeatable): a "
                         "teacher judged beside the walkers' best checkpoint")
    ck.add_argument('--control', default=None, help="the control's tag: today's team (with --with) on this code's "
                                                    "route, same scopes and seeds (default control-<team>[-...]-r<N>)")
    ck.add_argument('--plan', action='store_true', help="the pieces and a `triforce job run --express` command for "
                                                       "each not played; run nothing")
    ck.add_argument('--submit', action='store_true', help="queue those commands (express, detached)")
    ck.add_argument('--agent', default=os.environ.get('TRIFORCE_AGENT') or 'AGENT', help="the jobs' --agent")
    ck.add_argument('--task', default='judge-checkpoints', help="the jobs' --task")
    ck.add_argument('--timeout', default='10m', help="each job's --timeout (the express place: 10m at most)")
    args = parser.parse_args(argv)
    try:
        if args.command == 'checkpoints':
            if args.runs < 1:
                raise ValueError("--runs must be >= 1")
            return checkpoints_main(args)
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
    except (ValueError, room_stats.RoomStatsError, *paths_errors()) as error:
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


def paths_errors() -> tuple:
    """The store's refusals a command reports as an error (a missing run, a run still open where it must not be)."""
    from . import paths, store  # pylint: disable=import-outside-toplevel
    return (paths.MissingError, store.StoreError)


if __name__ == '__main__':
    sys.exit(main())
