"""Per-room statistics of the route by scope: a full run, each segment, each dungeon alone, and the room alone.

    triforce room-stats TAG [--noclips [TAG2]] [--chains-dir DIR] [--output FILE] [--route NAME] [--models-dir DIR]

A room's death rate means little on its own: 5% of the isolated episodes started in Gleeok's room is another number
than a 5% chance of dying there on a run from the game start.  This module reads the chain files of one measurement
(``scripts/experiments/r56_chains.sh``: ``<chains-dir>/<TAG>-<file>.jsonl``, one run per line with a ``RoomRecord``
per room visit, ``crates/env/src/chain.rs``) and the single-room evaluations of the models that played them, and
writes one JSON file, ``<chains-dir>/<TAG>-room-stats.json`` (``FORMAT``), with the same per-room fields in every
scope (``SCOPES``):

- ``full``: the ``full`` file (runs from the game start to the D4 triforce).  ``death_rate`` = runs that died in the
  room / runs started: the chance, on a full run, of dying there.
- ``to-d2``, ``to-d1``, ``to-d3``, ``to-d4`` (segments: the route cut at the dungeon exits): game start -> out of
  Dungeon 2 (0:0x3c), 0:0x3c -> out of Dungeon 1 (0:0x37), 0:0x37 -> out of Dungeon 3 (0:0x74), 0:0x74 -> the D4
  triforce.  Each from the file whose runs start at the segment's start (``full``, ``post-d2``, ``post-d1``,
  ``post-d3``), cut at the first entry into the next segment's start step: the runs play on to the route's end, the
  segment is the stretch before the cut.  ``death_rate`` = runs that died in the room within the segment / runs
  started on it.  A room on several segments (overworld 0x38) has an entry in each.
- ``d2``, ``d1``, ``d3``, ``d4``: the dungeon files (from the dungeon's entrance), cut like the segments.
- ``room``: the single-room evaluations (``triforce.evaluate``, ``visited_rooms``) of the models the chains played
  (each in its chain act mode), merged per room weighted by visits.  Here a run is an isolated episode that entered
  the room: ``death_rate`` = such episodes that died in it / such episodes.

A chain room entry (``chain_entry``): ``runs`` (of the scope), ``entered`` (runs that entered the room),
``visits``, ``reach_rate`` (entered / runs: survival into the room), ``deaths`` / ``death_rate``, ``stuck`` /
``stuck_rate``, ``endings`` (every way a run ended in the room: ``death``, ``stuck``, ``retreat-limit``, ``missed``,
``error``, ...) and ``end_rate``, ``death_rate_entered`` (deaths / entered), ``through_rate`` ((entered - ended) /
runs: survival out of it), per visit ``hearts_lost`` (``max(0, health_in - health_out)``), ``hearts_gained``,
``hearts_change``, ``hearts_healed``, ``hits`` (actions that lost health: the records' ``hits``, round 22; None in
files from before it), ``steps``, ``frames``, ``clips`` (``clip_done``; None in files from before clips), ``bombs``
(placed) and ``cheated_bombs``, then ``wrong_exits`` (visits that left by a wrong exit), ``outcomes`` (visits by
outcome), ``specialists`` (visits by specialist) and ``steps`` (the same fields per route step through the room:
``"44"``, or ``"off"`` for visits off the route).  A visit's route step is the record's ``step`` (round 22: the step
Link played on entering the room); for older records it is inferred (``assign_steps``: the route's steps matched
along the run).  The route is this code's (``--route``, default the chains'); visits the route doesn't explain (a
route changed since the chains ran) count as ``off``.  A room scope entry has the same names: ``runs`` =
``entered`` = episodes that entered the room, ``reach_rate``, ``frames``, ``wrong_exits`` and ``steps`` None,
``hits`` from the evaluation; its ``hearts_lost``/``hearts_gained`` are the evaluation's (summed within the visit,
not net).

The clips switch (round 22: CLIP offered and the clip way through each routes.yaml ``stretch``, or masked and the
plain way).  Every room entry, in every scope, has ``by_clips``: ``{'on': {...}, 'off': {...}}``, the same fields
(without the room's location) over the visits played with the switch on / off, only the states it has; the entry
itself stays the combined numbers.  A chain visit's switch is its record's ``clips_allowed`` (the switch when the
visit began), else its run's ``clips``, else the chain file's (``.meta.json`` ``clips``), else on (files from before
the switch played with clips allowed).  In a chain scope a switch state's ``runs`` are the runs that played a visit
in it, so the switch's ``death_rate`` is per run that played with it; the scope has ``by_clips`` too (``runs``,
``arrived``, ``arrival_rate``, ``endings``, ``off_route_visits``; arrivals and endings in the switch of the run's
last visit).  The room scope splits by each evaluation's ``by_clips[switch]['visited_rooms']``; an evaluation from
before the switch counts as on.  The switch-off runs come from the same file (``--clips-off-from``) or from a second
tag: ``--noclips`` reads ``<TAG>-noclips-*.jsonl`` too (``CLIPS=off TAG=<TAG>-noclips r56_chains.sh``; ``--noclips
TAG2`` another tag), each scope from both tags' files together (a file both link to counts once), and writes
``<TAG>-both-room-stats.json``.  Files of format 1 (no ``by_clips``) are still read: their visits played on.

Rooms are keyed ``L:0xRR`` (``:cave`` for a cave or cellar), as ``eval.json``.  The file also records its
provenance: ``tag``, ``tags`` (with ``noclips``, the second), ``version`` (this code's), ``route``, ``team`` (the
models of the chains: run and act mode), ``promoted`` (each family's promoted run in the models dir when the file was
written) and per scope its ``source`` (the first chain file: its runs, start state, act modes, clips switch, code,
data and version) and ``files`` (every chain file read), or ``sources`` (the evaluations read, with their clips
mode).  ``newest`` finds the newest file in a chains dir (``zelda-map.py``'s default).
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import __version__, registry
from .events import write_json_atomic
from .paths import ROOT

FORMAT = 2                              # 2: by_clips (round 22); 1 is still read (every visit then played on)
READ_FORMATS = (1, 2)
CHAINS_DIR = ROOT / 'target' / 'r56-chains'
SUFFIX = '-room-stats.json'
NOCLIPS_SUFFIX = '-noclips'             # the switch-off measurement of a tag (r56_chains.sh CLIPS=off)
BOTH_SUFFIX = '-both'                   # a room-stats file of a tag and its switch-off tag together
DETAILED_FORMAT = 2                     # eval.json with visited_rooms (triforce.evaluate.FORMAT)
COUNT_KEYS = ('visits', 'episodes', 'deaths')
KIND_MEAN_KEYS = ('drops_per_visit', 'pickups_per_visit')
# Exits that leave a room playing on; any other last outcome of a failed run is where the run ended.
PLAYING_OUTCOMES = ('exit-correct', 'exit-wrong', 'retreat', 'milestone', 'in-progress')
STEP_BACK, STEP_AHEAD = 2, 8            # assign_steps: how far from the route cursor a visit may match a step
DIGITS = 5
SWITCHES = ('on', 'off')               # the run-time clips switch (triforce.clips.name)
BOTH = 'both'                          # load_evals_by_clips: every visit, both switch states together


@dataclass(frozen=True)
class Scope:
    """One scope of the chain statistics: runs of ``<TAG>-<file>.jsonl`` from ``start`` (a state label, or
    ``start``: the game start), cut before the first entry into the route step of ``cut`` (None: to the end)."""
    name: str
    kind: str                           # full, segment or dungeon
    label: str
    file: str
    start: str
    cut: Optional[str]


SCOPES = (
    Scope('full', 'full', 'Full run: start → the D4 triforce', 'full', 'start', None),
    Scope('to-d2', 'segment', 'To D2: start → out of Dungeon 2 (0:0x3c)', 'full', 'start', '0_3cc_0'),
    Scope('to-d1', 'segment', 'To D1: 0:0x3c → out of Dungeon 1 (0:0x37)', 'post-d2', '0_3cc_0', '0_37c_0'),
    Scope('to-d3', 'segment', 'To D3: 0:0x37 → out of Dungeon 3 (0:0x74)', 'post-d1', '0_37c_0', '0_74c_0'),
    Scope('to-d4', 'segment', 'To D4: 0:0x74 → the D4 triforce', 'post-d3', '0_74c_0', None),
    Scope('d2', 'dungeon', 'Dungeon 2: 2:0x7d → out (0:0x3c)', 'd2', '2_7ds_0', '0_3cc_0'),
    Scope('d1', 'dungeon', 'Dungeon 1: 1:0x73 → out (0:0x37)', 'd1', '1_73s_0', '0_37c_0'),
    Scope('d3', 'dungeon', 'Dungeon 3: 3:0x7c → out (0:0x74)', 'd3', '3_7cs_0', '0_74c_0'),
    Scope('d4', 'dungeon', 'Dungeon 4: 4:0x71 → the D4 triforce', 'd4', '4_71s_0', None),
)
ROOM_SCOPE = 'room'
ROOM_LABEL = 'Room only: isolated single-room evaluations'


class RoomStatsError(Exception):
    """A chain file that doesn't fit its scope (another start, several routes)."""


def room_key(level: int, room: int, in_cave: bool) -> str:
    """(1, 0x44, False) -> '1:0x44', as ``eval.json`` keys rooms (``triforce.evaluate``)."""
    return f"{level}:0x{room:02x}" + (':cave' if in_cave else '')


def parse_room_key(key: str) -> tuple[int, int, bool]:
    """'0:0x2c:cave' -> (0, 0x2c, True)."""
    parts = key.split(':')
    return int(parts[0]), int(parts[1], 16), len(parts) > 2 and parts[2] == 'cave'


def _location(record: dict) -> tuple[int, int, bool]:
    return int(record['level']), int(str(record['room']), 16), bool(record['in_cave'])


def _round(value):
    return None if value is None else round(value, DIGITS)


# ----------------------------------------------------------------------------------------------- route steps

def state_step(plan: list[dict], state: str) -> int:
    """The route step a chain started from ``state`` begins at (``crates/env/src/chain.rs`` ``start_step``): the
    first step whose ``<level>_<room><side>_`` prefixes the label (side: ``cave`` in a cave, ``c`` out of one, else
    the entry); ``start`` is step 0.  Raises RoomStatsError for a state no step has."""
    if state == 'start':
        return 0
    label = state.rsplit('/', 1)[-1]
    for step in plan:
        side = 'cave' if step['in_cave'] else 'c' if step['entry'] == 'cave' else step['entry'].lower()
        if label.startswith(f"{step['level']}_{step['room']:02x}{side}_"):
            return step['index']
    raise RoomStatsError(f"no route step starts at {state}")


def assign_steps(plan: list[dict], records: list[dict], start: int) -> list[Optional[int]]:
    """The route step of each visit (None: off the route, or the milestone marker).  A record that names its step
    (``step``, round 22: the step Link played on entering the room) keeps it, unless that step isn't in this route's
    plan for the record's room (a route changed since the chains ran: off the route).  For older records a cursor
    follows the route: a visit matches the step of its room nearest the cursor, from ``STEP_BACK`` steps behind
    (walking back a room, a retreat) to ``STEP_AHEAD`` ahead (a wrong exit that skips route rooms: the game's
    progress, not the route's order, picks the next objective); a correct exit moves the cursor to the step after the
    visit's."""
    locations = [(s['level'], s['room'], s['in_cave']) for s in plan]
    cursor, steps = start, []
    for record in records:
        if record['outcome'] == 'milestone':
            steps.append(None)
            continue
        location = _location(record)
        if 'step' in record:
            step = record['step']
            step = step if step is not None and 0 <= step < len(plan) and locations[step] == location else None
        else:
            near = [k for k in range(max(0, cursor - STEP_BACK), min(len(plan), cursor + STEP_AHEAD + 1))
                    if locations[k] == location]
            step = min(near, key=lambda k: (abs(k - cursor), k > cursor)) if near else None
        steps.append(step)
        if step is not None and record['outcome'] == 'exit-correct':
            cursor = step + 1
    return steps


def cut_run(plan: list[dict], run: dict, start: int, cut: Optional[int]) -> tuple[list[tuple[dict, Optional[int]]],
                                                                                 bool, Optional[str]]:
    """(the run's (visit, step) pairs before the cut, whether it reached the cut (or, without one, the route's end),
    how it ended inside the scope: None if it didn't end there, else the outcome of its last visit (``other`` for a
    failed run whose last visit left the room))."""
    records = run.get('rooms') or []
    steps = assign_steps(plan, records, start)
    for index, step in enumerate(steps):
        if cut is not None and step == cut and index > 0:
            return [(r, s) for r, s in zip(records[:index], steps[:index]) if r['outcome'] != 'milestone'], True, None
    played = [(r, s) for r, s in zip(records, steps) if r['outcome'] != 'milestone']
    if cut is None and run.get('result') == 'success':
        return played, True, None
    if run.get('result') == 'success' or not played:
        return played, False, None
    outcome = played[-1][0]['outcome']
    return played, False, 'other' if outcome in PLAYING_OUTCOMES else outcome


# ------------------------------------------------------------------------------------------------- chain stats

class _Visits:
    """Sums over one room's (or one room step's) visits within a scope."""

    def __init__(self):
        self.visits = 0
        self.entered: set[int] = set()
        self.endings = Counter()
        self.outcomes = Counter()
        self.specialists = Counter()
        self.sums = Counter()
        self.clips = False
        self.hit_visits = 0                 # visits whose record counts hits (round 22)

    def add(self, run: int, record: dict):
        self.visits += 1
        self.entered.add(run)
        self.outcomes[record['outcome']] += 1
        self.specialists[record.get('specialist') or '-'] += 1
        health_in, health_out = float(record['health_in']), float(record['health_out'])
        self.sums['hearts_lost'] += max(0.0, health_in - health_out)
        self.sums['hearts_gained'] += max(0.0, health_out - health_in)
        self.sums['hearts_change'] += health_out - health_in
        self.sums['hearts_healed'] += float(record.get('hearts_healed') or 0.0)
        self.sums['steps'] += record.get('steps') or 0
        self.sums['frames'] += record.get('frames') or 0
        self.sums['bombs'] += record.get('bombs_used') or 0
        self.sums['cheated_bombs'] += record.get('cheated_bombs') or 0
        if 'clip_done' in record:
            self.clips = True
            self.sums['clips'] += record['clip_done'] or 0
        if record.get('hits') is not None:
            self.hit_visits += 1
            self.sums['hits'] += record['hits']

    def entry(self, runs: int) -> dict:
        entered, ended = len(self.entered), sum(self.endings.values())
        visits = self.visits

        def per_visit(name):
            return _round(self.sums[name] / visits) if visits else None
        return {
            'runs': runs, 'entered': entered, 'visits': visits,
            'reach_rate': _round(entered / runs) if runs else None,
            'deaths': self.endings['death'], 'death_rate': _round(self.endings['death'] / runs) if runs else None,
            'stuck': self.endings['stuck'], 'stuck_rate': _round(self.endings['stuck'] / runs) if runs else None,
            'endings': dict(self.endings), 'end_rate': _round(ended / runs) if runs else None,
            'death_rate_entered': _round(self.endings['death'] / entered) if entered else None,
            'through_rate': _round((entered - ended) / runs) if runs else None,
            'hearts_lost_per_visit': per_visit('hearts_lost'),
            'hearts_gained_per_visit': per_visit('hearts_gained'),
            'hearts_change_per_visit': per_visit('hearts_change'),
            'hearts_healed_per_visit': per_visit('hearts_healed'),
            'hits_per_visit': _round(self.sums['hits'] / self.hit_visits) if self.hit_visits else None,
            'steps_per_visit': per_visit('steps'),
            'frames_per_visit': per_visit('frames'),
            'clips_per_visit': per_visit('clips') if self.clips else None,
            'bombs_per_visit': per_visit('bombs'),
            'cheated_bombs_per_visit': per_visit('cheated_bombs'),
            'wrong_exits': self.outcomes['exit-wrong'],
            'outcomes': dict(self.outcomes),
            'specialists': dict(self.specialists),
        }


def visit_switch(record: dict, run: dict) -> str:
    """The clips switch a visit played (``triforce.clips.name``: 'on' / 'off'): the record's ``clips_allowed`` (the
    switch when the visit began), else the run's ``clips`` (the switch it started with), else on (files from before
    the switch, which played with clips allowed)."""
    from .clips import name  # pylint: disable=import-outside-toplevel  # _native: only when needed
    allowed = record.get('clips_allowed')
    return name(run.get('clips') if allowed is None else allowed)


def _room_entries(rooms: dict, steps: dict, count: int, plan: list[dict]) -> dict:
    """Room key -> ``_Visits.entry`` with its ``steps`` (route order, 'off' last), from ``rooms`` (key -> _Visits)
    and ``steps`` ((key, step) -> _Visits)."""
    by_room = defaultdict(dict)                 # room key -> {step: _Visits}, steps in route order, 'off' last
    for (key, step), visits in sorted(steps.items(), key=lambda item: (item[0][1] is None, item[0][1] or 0)):
        by_room[key][step] = visits
    first = {key: next(iter(room_steps)) for key, room_steps in by_room.items()}
    entries = {}
    for key in sorted(rooms, key=lambda k: len(plan) if first[k] is None else first[k]):    # route order
        entries[key] = {**rooms[key].entry(count),
                        'steps': {('off' if step is None else str(step)): {'step': step, **visits.entry(count)}
                                  for step, visits in by_room[key].items()}}
    return entries


def chain_scope(runs: list[dict], plan: list[dict], scope: Scope) -> dict:
    """One chain scope: ``runs`` (started), ``arrived`` / ``arrival_rate`` (runs that reached the cut, or the
    route's end), ``endings`` (how the others ended), ``off_route_visits`` and ``rooms``: room key -> ``chain_entry``
    with its ``steps``.  Every room entry also has ``by_clips``: the same entry (``steps`` included) over the visits
    of each switch state (``visit_switch``; only those it has), and so has the scope (``runs``, ``arrived``,
    ``arrival_rate``, ``endings``, ``off_route_visits``).  A switch state's ``runs`` are the scope's runs that played
    a visit in it (a run without visits counts in its start switch), so a run that flips (``--clips-off-from``)
    counts in both; a run's arrival and its ending count in the switch of its last visit."""
    start = state_step(plan, scope.start)
    cut = state_step(plan, scope.cut) if scope.cut else None
    # Keyed by group: None (every visit) or a switch state.
    rooms, steps = defaultdict(_Visits), defaultdict(_Visits)
    played = {switch: set() for switch in SWITCHES}
    arrived, endings, off_route = Counter(), defaultdict(Counter), Counter()
    for index, run in enumerate(runs):
        visits, reached, ended = cut_run(plan, run, start, cut)
        switches = [visit_switch(record, run) for record, _ in visits]
        last = switches[-1] if switches else visit_switch({}, run)
        played[last].add(index)
        for (record, step), switch in zip(visits, switches):
            played[switch].add(index)
            key = room_key(*_location(record))
            for group in (None, switch):
                rooms[group, key].add(index, record)
                steps[group, key, step].add(index, record)
                off_route[group] += step is None
        for group in (None, last):
            arrived[group] += reached
            if ended is not None:
                record, step = visits[-1]
                key = room_key(*_location(record))
                rooms[group, key].endings[ended] += 1
                steps[group, key, step].endings[ended] += 1
                endings[group][ended] += 1
    counts = {None: len(runs), **{switch: len(played[switch]) for switch in SWITCHES}}
    groups = {group: _room_entries({k[1]: v for k, v in rooms.items() if k[0] == group},
                                   {k[1:]: v for k, v in steps.items() if k[0] == group}, counts[group], plan)
              for group in counts}
    entries = {}
    for key, entry in groups[None].items():
        level, room, in_cave = parse_room_key(key)
        entries[key] = {'level': level, 'room': room, 'in_cave': in_cave, **entry,
                        'by_clips': {switch: groups[switch][key] for switch in SWITCHES if key in groups[switch]}}

    def totals(group):
        count = counts[group]
        return {'runs': count, 'arrived': arrived[group], 'arrival_rate': _round(arrived[group] / count) if count
                else None, 'endings': dict(endings[group]), 'off_route_visits': off_route[group]}
    return {'name': scope.name, 'kind': scope.kind, 'label': scope.label, **totals(None), 'start': scope.start,
            'start_step': start, 'cut': scope.cut, 'cut_step': cut, 'rooms': entries,
            'by_clips': {switch: totals(switch) for switch in SWITCHES if counts[switch]}}


def read_chain(path: Path) -> tuple[list[dict], dict]:
    """(runs, provenance) of one eval-chain output and its ``.meta.json`` (``{}`` without one)."""
    runs = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    meta = path.with_name(path.name + '.meta.json')
    provenance = json.loads(meta.read_text(encoding='utf-8')).get('provenance', {}) if meta.is_file() else {}
    return runs, provenance


def chain_source(path: Path, runs: list[dict], provenance: dict) -> dict:
    """What a scope was measured from: the file's name and runs, its start, act modes, the clips switch its runs
    started with and where it turned off (None in files from before the switch), code, data and version."""
    models = provenance.get('models') or []
    return {'file': path.name, 'runs': len(runs), 'start_from': provenance.get('start_from'),
            'route': provenance.get('route'), 'act': dict(Counter(m.get('act') for m in models)),
            'clips': provenance.get('clips'), 'clips_off_from': provenance.get('clips_off_from'),
            'code': provenance.get('code'), 'data': provenance.get('data'), 'version': provenance.get('version')}


def chain_team(provenance: dict) -> dict[str, dict]:
    """The models a chain played: specialist -> {'run', 'act', 'checkpoint_sha1'} (checkpoints are recorded as
    ``<specialist>/<run>/ckpt.pt``)."""
    team = {}
    for model in provenance.get('models') or []:
        parts = str(model.get('checkpoint') or '').split('/')
        if len(parts) >= 3:
            team[model['name']] = {'run': parts[-2], 'act': model.get('act'),
                                   'checkpoint_sha1': model.get('checkpoint_sha1')}
    return team


# -------------------------------------------------------------------------------------------- room-only stats

def read_eval(run_dir: Path, mode: Optional[str] = None
              ) -> tuple[str, Optional[dict[str, dict]], Optional[str], str, Optional[str]]:
    """(status, visited_rooms by switch, act mode, why, clips mode) of a run's evaluation of its current weights in
    ``mode`` (default its act mode; ``triforce.evaluate.read_evaluation``: ``eval.json`` or ``eval-<mode>.json``).
    status: 'detailed', 'no eval' (none in that act mode for these weights; ``why`` says what was found),
    'unreadable' or 'old format'.  The rooms are ``{'both': visited_rooms, 'on': ..., 'off': ...}``: the
    evaluation's ``by_clips[switch]['visited_rooms']``, or for one from before the switch (no ``by_clips``; clips
    mode None) every room under ``on``."""
    from .evaluate import read_evaluation  # pylint: disable=import-outside-toplevel  # torch: only when needed
    try:
        path, data = read_evaluation(run_dir, mode)
    except (FileNotFoundError, LookupError) as error:      # no ckpt.pt, or no evaluation of it in its act mode
        return 'no eval', None, None, str(error), None
    except (OSError, ValueError) as error:
        return 'unreadable', None, None, str(error), None
    mode = registry.eval_mode(data)
    if data.get('format') != DETAILED_FORMAT or not isinstance(data.get('visited_rooms'), dict):
        return 'old format', None, mode, path.name, data.get('clips')
    by_clips = data.get('by_clips')
    if isinstance(by_clips, dict):
        split = {switch: (by_clips.get(switch) or {}).get('visited_rooms') or {} for switch in SWITCHES}
    else:
        split = {'on': data['visited_rooms'], 'off': {}}
    return 'detailed', {BOTH: data['visited_rooms'], **split}, mode, path.name, data.get('clips')


def mean_keys() -> tuple[str, ...]:
    """The per-visit means of ``visited_rooms`` merged over evaluations."""
    from ._native import CHEAT_ITEMS, CLIP_STATS  # pylint: disable=import-outside-toplevel
    return ('steps_per_visit', 'health_lost_per_visit', 'health_gained_per_visit', 'health_change_per_visit',
            'hits_per_visit', 'no_hit_rate', 'hits_dealt_per_visit', 'keys_per_visit', 'bombs_used_per_visit',
            *(f'cheated_{item}_per_visit' for item in CHEAT_ITEMS), 'boss_stuns_per_visit', 'boss_eats_per_visit',
            *(f'clip_{stat}_per_visit' for stat in CLIP_STATS))


def merge_room_stats(entries: list[tuple[str, dict]]) -> dict:
    """One room's ``visited_rooms`` stats from several specialists' evals -> one, weighted by visits: counts are
    summed, per-visit means are sum(mean x visits) / total visits, the death rate is recomputed.  ``specialists``
    lists (specialist, visits) of the evals that contributed."""
    entries = [(name, stats) for name, stats in entries if stats.get('visits')]
    if not entries:
        return {}
    first = entries[0][1]
    merged = {'level': first['level'], 'room': first['room'], 'in_cave': bool(first.get('in_cave')),
              'specialists': [(name, stats['visits']) for name, stats in entries]}
    for key in COUNT_KEYS:
        merged[key] = sum(stats.get(key) or 0 for _, stats in entries)
    visits = merged['visits']
    merged['death_rate'] = merged['deaths'] / visits
    for key in mean_keys():
        weighted = [(stats[key], stats['visits']) for _, stats in entries if stats.get(key) is not None]
        weight = sum(v for _, v in weighted)
        merged[key] = sum(value * v for value, v in weighted) / weight if weight else None
    for key in KIND_MEAN_KEYS:
        totals = Counter()
        for _, stats in entries:
            for kind, mean in (stats.get(key) or {}).items():
                totals[kind] += mean * stats['visits']
        merged[key] = {kind: total / visits for kind, total in totals.items() if total}
    exits = Counter()
    for _, stats in entries:
        exits.update(stats.get('exits') or {})
    merged['exits'] = dict(exits)
    return merged


def load_evals_by_clips(models_dir: Path, run_id: Optional[str] = None, team: Optional[dict[str, dict]] = None
                        ) -> tuple[dict[str, dict[tuple, dict]], list[dict]]:
    """The merged per-room stats of every promoted specialist's detailed eval (of its current weights, in its act
    mode) by switch: ``{'both': every episode, 'on': ..., 'off': ...}``, each (level, room, in_cave) ->
    ``merge_room_stats`` (``read_eval``: an evaluation from before the switch counts as on), and one source entry
    per specialist: {'specialist', 'run', 'status', 'mode', 'file', 'clips' (the evaluation's clips mode), 'rooms',
    'visits'}.  With ``run_id``, a specialist's ``<run_id>`` evaluation is used instead when it has one (e.g. a round
    trained with ``--no-promote``).  With ``team`` (``chain_team``) its runs are read instead, each in its act mode
    there."""
    from .specialist import list_specialists  # pylint: disable=import-outside-toplevel
    per_room, sources = {group: defaultdict(list) for group in (BOTH, *SWITCHES)}, []
    root = Path(models_dir)
    if team is not None:
        chosen = [(name, member['run'], member.get('act')) for name, member in sorted(team.items())]
    else:
        current = set(list_specialists())
        chosen = []
        for directory in sorted(root.iterdir()) if root.is_dir() else ():
            # Current specialists only: models/ keeps the directories of retired families.
            if directory.name in current and (directory / registry.PROMOTED_NAME).is_file():
                chosen.append((directory.name, registry.promoted_run(directory.name, root), None))
    for name, run, mode in chosen:
        directory = root / name
        read = None
        if run_id and (directory / run_id / registry.CHECKPOINT_NAME).is_file():
            read = read_eval(directory / run_id)
            if read[0] == 'detailed':
                run = run_id
            else:
                read = None
        status, rooms, mode, file, clips = read or read_eval(directory / run, mode)
        source = {'specialist': name, 'run': run, 'status': status, 'mode': mode, 'file': file, 'clips': clips,
                  'rooms': 0, 'visits': 0}
        for group, group_rooms in (rooms or {}).items():
            for stats in group_rooms.values():
                if stats.get('visits'):
                    key = (int(stats['level']), int(stats['room']), bool(stats.get('in_cave')))
                    per_room[group][key].append((name, stats))
                    if group == BOTH:
                        source['rooms'] += 1
                        source['visits'] += stats['visits']
        sources.append(source)
    merged = {group: {key: merge_room_stats(entries) for key, entries in rooms.items()}
              for group, rooms in per_room.items()}
    return merged, sources


def load_evals(models_dir: Path, run_id: Optional[str] = None,
               team: Optional[dict[str, dict]] = None) -> tuple[dict[tuple, dict], list[dict]]:
    """``load_evals_by_clips``' merged stats over every episode (both switch states), and its sources."""
    merged, sources = load_evals_by_clips(models_dir, run_id, team)
    return merged[BOTH], sources


def eval_entry(merged: dict) -> dict:
    """A ``merge_room_stats`` room under the chain entry's names: a run is an episode that entered the room."""
    episodes = merged.get('episodes') or merged['visits']
    exits = merged.get('exits') or {}
    endings = {name: exits[exit] for name, exit in (('death', 'death'), ('stuck', 'timeout'), ('other', 'other'))
               if exits.get(exit)}
    ended = sum(endings.values())
    return {
        'level': int(merged['level']), 'room': int(merged['room']), 'in_cave': bool(merged['in_cave']),
        'runs': episodes, 'entered': episodes, 'visits': merged['visits'], 'reach_rate': None,
        'deaths': merged['deaths'], 'death_rate': _round(merged['deaths'] / episodes),
        'stuck': endings.get('stuck', 0), 'stuck_rate': _round(endings.get('stuck', 0) / episodes),
        'endings': endings, 'end_rate': _round(ended / episodes),
        'death_rate_entered': _round(merged['deaths'] / episodes),
        'through_rate': _round((episodes - ended) / episodes),
        'hearts_lost_per_visit': _round(merged.get('health_lost_per_visit')),
        'hearts_gained_per_visit': _round(merged.get('health_gained_per_visit')),
        'hearts_change_per_visit': _round(merged.get('health_change_per_visit')),
        'hearts_healed_per_visit': None,
        'hits_per_visit': _round(merged.get('hits_per_visit')),
        'steps_per_visit': _round(merged.get('steps_per_visit')),
        'frames_per_visit': None,
        'clips_per_visit': _round(merged.get('clip_done_per_visit')),
        'bombs_per_visit': _round(merged.get('bombs_used_per_visit')),
        'cheated_bombs_per_visit': _round(merged.get('cheated_bombs_per_visit')),
        'wrong_exits': None,
        'outcomes': dict(exits),
        'specialists': dict(merged.get('specialists') or []),
        'steps': None,
    }


def _switch_entry(entry: dict) -> dict:
    """A room entry inside ``by_clips``: the entry without the room's location."""
    return {name: value for name, value in entry.items() if name not in ('level', 'room', 'in_cave')}


def room_entries(merged: dict[str, dict[tuple, dict]]) -> dict[str, dict]:
    """Room key -> ``eval_entry`` over every episode with ``by_clips`` (the entry of each switch state the room has
    episodes in), from ``load_evals_by_clips``' merged stats."""
    rooms = {}
    for key, stats in sorted(merged[BOTH].items()):
        if stats:
            rooms[room_key(*key)] = {**eval_entry(stats), 'by_clips': {
                switch: _switch_entry(eval_entry(merged[switch][key])) for switch in SWITCHES
                if merged[switch].get(key)}}
    return rooms


def room_scope(models_dir: Path, team: Optional[dict[str, dict]]) -> dict:
    """The ``room`` scope: the evaluations of ``team`` (``chain_team``; None: the promoted models), every room split
    by switch (``room_entries``)."""
    merged, sources = load_evals_by_clips(models_dir, team=team)
    return {'name': ROOM_SCOPE, 'kind': ROOM_SCOPE, 'label': ROOM_LABEL, 'runs': None,
            'team': 'chains' if team is not None else 'promoted', 'sources': sources, 'rooms': room_entries(merged)}


# ---------------------------------------------------------------------------------------------------- the file

def promoted_runs(models_dir: Path) -> dict[str, str]:
    """Each current family's promoted run (absent: none promoted)."""
    from .specialist import list_specialists  # pylint: disable=import-outside-toplevel
    root = Path(models_dir)
    return {name: registry.promoted_run(name, root) for name in list_specialists()
            if (root / name / registry.PROMOTED_NAME).is_file()}


def noclips_tag(tag: str) -> str:
    """The tag of ``tag``'s measurement with the clips switch off (``CLIPS=off TAG=<tag>-noclips``,
    ``scripts/experiments/r56_chains.sh``)."""
    return f'{tag}{NOCLIPS_SUFFIX}'


def _start_switch(runs: list[dict], provenance: dict) -> list[dict]:
    """``runs`` with the file's switch (``provenance['clips']``) where a run doesn't record its own."""
    clips = provenance.get('clips')
    if not isinstance(clips, bool):
        return runs
    return [run if run.get('clips') is not None else {**run, 'clips': clips} for run in runs]


def build(tag: str, chains_dir: Path = CHAINS_DIR, models_dir=None, route: Optional[str] = None,
          noclips: Optional[str] = None) -> dict:
    """The room-stats file of ``<chains_dir>/<tag>-*.jsonl``: every scope whose chain file exists, and the room
    scope of the chains' team (of the promoted models if no chain file names its models).  With ``noclips`` (a
    second tag, ``noclips_tag``: the same measurement with the switch off) each scope reads both tags' files
    together (a file both tags link to once): the combined numbers are over both, ``by_clips`` splits them.  Raises
    RoomStatsError when no chain file (or none of ``noclips``) exists or a file starts elsewhere than its scope."""
    from ._native import route_steps  # pylint: disable=import-outside-toplevel
    chains_dir, models_dir = Path(chains_dir), registry.models_root(models_dir)
    tags = [tag, *([noclips] if noclips else [])]
    loaded, scopes, team, plans, found = {}, {}, None, {}, set()
    for scope in SCOPES:
        paths = {}                          # resolved -> the path of the first tag that has it
        for name in tags:
            path = chains_dir / f'{name}-{scope.file}.jsonl'
            if path.is_file() and path.resolve() not in paths:
                paths[path.resolve()] = path
                found.add(name)
        if not paths:
            continue
        runs, sources, routes = [], [], set()
        for resolved, path in paths.items():
            if resolved not in loaded:
                loaded[resolved] = read_chain(path)
            file_runs, provenance = loaded[resolved]
            start = provenance.get('start_from') or 'start'
            if provenance and start.rsplit('/', 1)[-1] != scope.start:
                raise RoomStatsError(f"{path.name} starts from {start}, scope {scope.name} from {scope.start}")
            routes |= {run.get('route') for run in file_runs}
            routes |= {provenance['route']} if provenance.get('route') else set()
            runs.extend(_start_switch(file_runs, provenance))
            sources.append(chain_source(path, file_runs, provenance))
            if team is None and chain_team(provenance):
                team = chain_team(provenance)
        files = ', '.join(path.name for path in paths.values())
        if len(routes) > 1:
            raise RoomStatsError(f"{files}: runs of several routes {sorted(routes, key=str)}")
        name = route or (routes.pop() if routes else None)
        if name is None:
            raise RoomStatsError(f"{files}: no route recorded (pass --route)")
        if name not in plans:
            plans[name] = list(route_steps(name))
        scopes[scope.name] = {**chain_scope(runs, plans[name], scope), 'route': name, 'source': sources[0],
                              'files': sources}
    for name in tags:
        if name not in found:
            raise RoomStatsError(f"no chain files {name}-*.jsonl in {chains_dir}")
    scopes[ROOM_SCOPE] = room_scope(models_dir, team)
    return {
        'format': FORMAT, 'tag': tag, 'tags': tags, 'noclips': noclips,
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'version': __version__,
        'route': sorted(plans)[0] if len(plans) == 1 else sorted(plans),
        'team': team, 'promoted': promoted_runs(models_dir),
        'segments': [{'name': s.name, 'kind': s.kind, 'label': s.label, 'file': s.file, 'start': s.start,
                      'cut': s.cut} for s in SCOPES],
        'scopes': scopes,
    }


def output_path(tag: str, chains_dir: Path = CHAINS_DIR, noclips: Optional[str] = None) -> Path:
    """``<chains_dir>/<tag>-room-stats.json``; ``<tag>-both-room-stats.json`` read with a ``noclips`` tag."""
    return Path(chains_dir) / f"{tag}{BOTH_SUFFIX if noclips else ''}{SUFFIX}"


def newest(chains_dir: Path = CHAINS_DIR) -> Optional[Path]:
    """The most recently written ``*-room-stats.json`` in ``chains_dir`` (None without one)."""
    files = sorted(Path(chains_dir).glob(f'*{SUFFIX}'), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def read(path: Path) -> dict:
    """A room-stats file (format 1, from before the clips switch: no ``by_clips``, or 2); raises ValueError for
    another format."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('format') not in READ_FORMATS or not isinstance(data.get('scopes'), dict):
        raise ValueError(f"{path}: not a room-stats file of format {' or '.join(map(str, READ_FORMATS))}")
    return data


def summary(data: dict, worst: int = 3) -> str:
    """One line per scope: runs, arrivals, the rooms with the highest death rate."""
    tags = ' + '.join(data.get('tags') or [data['tag']])
    lines = [f"room stats {tags} (format {data['format']}, route {data['route']})"]
    for name, scope in data['scopes'].items():
        rooms = sorted(scope['rooms'].items(), key=lambda item: -(item[1]['death_rate'] or 0))[:worst]
        deaths = ', '.join(f"{key} {entry['death_rate']:.1%}" for key, entry in rooms if entry['death_rate'])
        if name == ROOM_SCOPE:
            read_ = sum(s['status'] == 'detailed' for s in scope['sources'])
            lines.append(f"  {name:6} {read_}/{len(scope['sources'])} evaluations, {len(scope['rooms'])} rooms; "
                         f"deaths per episode: {deaths or 'none'}")
        else:
            split = scope.get('by_clips') or {}
            switches = ('; ' + ', '.join(f"clips {switch} {totals['runs']} runs, arrived "
                                         f"{totals['arrival_rate'] or 0:.1%}" for switch, totals in split.items())
                        if 'off' in split else '')
            files = ', '.join(source['file'] for source in scope.get('files') or [scope['source']])
            lines.append(f"  {name:6} {scope['runs']:5} runs ({files}), arrived {scope['arrival_rate']:.1%}"
                         f"{switches}, off-route visits {scope['off_route_visits']}; "
                         f"deaths per run: {deaths or 'none'}")
    return '\n'.join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='triforce room-stats', description=__doc__.split('\n\n')[0])
    parser.add_argument('tag', help="the chain files' tag: <chains-dir>/<TAG>-full.jsonl etc.")
    parser.add_argument('--noclips', nargs='?', const=True, default=None, metavar='TAG',
                        help="also read this tag's chain files, the same measurement with the clips switch off "
                             "(default <TAG>-noclips: CLIPS=off TAG=<TAG>-noclips r56_chains.sh); writes "
                             "<TAG>-both-room-stats.json")
    parser.add_argument('--chains-dir', type=Path, default=CHAINS_DIR, help="default <repo>/target/r56-chains")
    parser.add_argument('--output', type=Path, default=None,
                        help="default <chains-dir>/<TAG>-room-stats.json (<TAG>-both-room-stats.json with --noclips)")
    parser.add_argument('--route', default=None, help="route the visits are matched to (default the chains')")
    parser.add_argument('--models-dir', default=None,
                        help="models root (default $TRIFORCE_MODELS_DIR or <repo>/models)")
    args = parser.parse_args(argv)
    noclips = noclips_tag(args.tag) if args.noclips is True else args.noclips
    try:
        data = build(args.tag, args.chains_dir, args.models_dir, args.route, noclips)
    except RoomStatsError as error:
        print(f"triforce room-stats: error: {error}", file=sys.stderr)
        return 2
    output = args.output or output_path(args.tag, args.chains_dir, noclips)
    write_json_atomic(output, data, indent=1)
    print(summary(data))
    print(f"wrote {output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
