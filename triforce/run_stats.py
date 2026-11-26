"""Run statistics (round 40, the owner: "every training and every big evaluation should leave a robust, standard set of
statistics [...] in one place"): one file for every chain, evaluation and training, computed from its records alone,
kept beside the output and registered in the store, and read back by ``triforce stats list|show|compare|ingest``.

Files.  ``<output>.stats.json`` beside a chain's jsonl and an evaluation's json, ``<run dir>/train.stats.json`` in a
training's run dir; each is also put into the store (``register``; ``store_path``, a placeholder until the shared
state's one place is decided), keyed ``<job id>-<name>`` with the queue job's agent, task, subtask, description,
commit and command.  Written when ``eval-chain`` ends (``cli.ChainOutput.finish``: a batch's chains, a round's, the
250 test's; a batch moves its chain's file with the output, ``move_chain``), when an evaluation is written
(``evaluate.write_result``) and when ``triforce train`` ends (again when the run's ``eval.json`` is written).

What a chain's file holds (``chain_stats``; every mean with its ``n`` and ``se``, so a reader can form a standard
error; numbers the records lack are null, never zero):

- ``rate`` (n, successes, rate, se); ``endings`` (wins, deaths, stuck, errors; stuck runs by reason, deaths and stuck
  runs by room, error reasons); ``triforces`` (runs that left each dungeon's triforce step by its exit) and
  ``game_ended``; ``furthest_step``.
- ``all``, ``by_step``, ``by_room``, ``by_family``: visits, runs, exits, wrong exits, deaths, stuck, other endings;
  per visit: hits, hearts lost, health in and out, frames (with median and 90th percentile), linger (longest quiet
  stretch, terminal tail), bombs used, keys used, cheated bombs, keys and rupees, shots, shots lined up, lined-up
  decisions and those not fired, kills far, near and other, shots that hit, beams lost, frames holding the beams and
  the visit frame they were first lost (``crates/zelda/src/beams.rs``).
- ``beams``: rooms cleared holding the beams and the frames to the entry of the room that took them (from each
  visit's health, any record), the frame they were first lost (round 40's counters), runs that never lost them,
  where they were lost first.
- ``time``: frames from the start to entering each route step (n, mean, median, 90th percentile) and to the end.
- ``runs``: the per-run values, aligned with ``seeds`` (``result`` one letter a run: w win, d death, s stuck, e
  error), for pairing two files by seed (``compare``): frames, hearts at the end, hearts lost, hits, shots, shots
  that hit, kills far and near, the beams measures, ``enter`` (frames to entering each step, null where not entered)
  and ``end``.
- ``goals`` (round 40, ``crates/zelda/src/goals.rs``): by route step, what the step expected of a visit from the
  route's data (a screen wrap, no hit where a path is planned, the carriers) beside what happened: visits, those the
  route crossed (no FIGHT objective) and the kills the drop cycle counted in them (carriers aside), WorldKillCycle on
  entering, wraps taken, visits without a hit, carriers aimed at on entering and at all, killed, their bombs dropped
  and taken; ``runs`` adds per run the kills where the route crossed, expected wraps not taken, hits in rooms with a
  planned path and carriers' bombs taken (``compare`` pairs them).
- ``flags`` (``{level: red|note, code, text}``: ``chain_flags``), ``manifest`` (the meta's, else null),
  ``provenance`` (route, start, stop, clips, the team), ``records`` (the counters the records carry, runs asked and
  played, the seeds).

An evaluation's (``eval_stats``) has the same groups over its episodes' room visits, the rate over episodes and by
start state, ``runs`` keyed ``<start state>#<seed>``; a training's (``train_stats``) the last iterations' training
figures, the steps to each early-stop threshold, steps a second, its harvested states and curriculum, and its
evaluation's headline when ``eval.json`` exists."""

import argparse
import json
import math
import os
import re
import shlex
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

SCHEMA = 1
SUFFIX = '.stats.json'
TRAIN_NAME = 'train.stats.json'
FPS = 60.0988                   # NES frames a second
RATE_EXTREME_RUNS = 100         # a rate of exactly 0% or 100% over this many runs or more is noted
STUCK_SHARE = 0.05              # a stuck share above this (and STUCK_MIN runs) is red
STUCK_MIN = 5
CANARY_MIN = 20                 # a count is a canary only once this many runs could have produced it
THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.99)
TRAIN_WINDOW = 20               # the iterations a training's final figures are averaged over


# ------------------------------------------------------------------------------------------------- accumulators

def _round(x):
    return None if x is None else round(x, 4)


class Mean:
    """Mean and standard error of the mean from running sums (values None are skipped)."""
    __slots__ = ('n', 'sum', 'sq')

    def __init__(self):
        self.n, self.sum, self.sq = 0, 0.0, 0.0

    def add(self, x):
        if x is None:
            return
        self.n += 1
        self.sum += x
        self.sq += x * x

    def out(self) -> Optional[dict]:
        if not self.n:
            return None
        mean = self.sum / self.n
        se = (math.sqrt(max(0.0, (self.sq - self.n * mean * mean) / (self.n - 1)) / self.n)
              if self.n > 1 else None)
        return {'n': self.n, 'sum': _round(self.sum), 'sq': _round(self.sq), 'mean': _round(mean), 'se': _round(se)}


def mean_se(values: list) -> tuple:
    """(mean, standard error) of ``values`` (se None under two values)."""
    n = len(values)
    if not n:
        return None, None
    m = sum(values) / n
    return m, (math.sqrt(sum((x - m) ** 2 for x in values) / (n - 1) / n) if n > 1 else None)


def quantile(sorted_values: list, q: float):
    """The ``q`` quantile (nearest rank) of sorted values."""
    if not sorted_values:
        return None
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def distribution(values: list) -> Optional[dict]:
    """n, mean, se, median and 90th percentile."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    m, se = mean_se(values)
    return {'n': len(values), 'sum': sum(values), 'sq': _round(sum(float(v) * v for v in values)), 'mean': _round(m),
            'se': _round(se), 'median': quantile(values, 0.5), 'p90': quantile(values, 0.9)}


# A visit's per-visit figures (``visit_figures``), in the order the reader shows them.
VISIT_MEANS = ('hits', 'hearts_lost', 'health_in', 'health_out', 'linger_max_frames', 'linger_tail_frames',
               'bombs_used', 'keys_used', 'cheated_bombs', 'cheated_keys', 'cheated_rupees', 'shots', 'shots_lined',
               'lined', 'lined_unfired', 'kills_far', 'kills_near', 'kills_other', 'kill_distance', 'shots_hit',
               'beams_lost', 'held_frames', 'beams_lost_visit', 'aimless_frames')
BEAM_FIELDS = ('shots', 'shots_lined', 'lined', 'lined_unfired', 'kills_far', 'kills_near', 'kills_other',
               'kill_distance', 'beams_lost')


def round40(visit: dict) -> bool:
    """The visit carries round 40's counters (shots that hit, the beams held)."""
    return visit.get('beam_held_frames') is not None


def visit_figures(visit: dict, kind: str = 'chain') -> dict:
    """A chain room record's or an evaluation visit's figures (None where the record lacks them)."""
    linger = visit.get('linger') or {}
    new = round40(visit)
    keys = (max(0, visit['keys_in'] - visit['keys_out'])
            if visit.get('keys_in') is not None and visit.get('keys_out') is not None else None)
    figures = {
        'hits': visit.get('hits'),
        'hearts_lost': visit.get('health_lost') if kind == 'evaluate' else visit.get('beam_hearts_lost'),
        'health_in': visit.get('health_in'), 'health_out': visit.get('health_out'),
        'linger_max_frames': linger.get('max_frames'), 'linger_tail_frames': linger.get('tail_frames'),
        'bombs_used': visit.get('bombs_used'), 'keys_used': keys,
        'cheated_bombs': visit.get('cheated_bombs'), 'cheated_keys': visit.get('cheated_keys'),
        'cheated_rupees': visit.get('cheated_rupees'),
        **{field: visit.get(f'beam_{field}') for field in BEAM_FIELDS},
        'shots_hit': visit.get('beam_shots_hit') if new else None,
        'held_frames': visit.get('beam_held_frames') if new else None,
        'beams_lost_visit': (visit.get('beam_lost_frame') is not None) if new else None,
        'lost_frame': visit.get('beam_lost_frame') if new else None,
        'frames': visit.get('frames'),
        'aimless_frames': visit.get('aimless_frames'),
    }
    return figures


class Group:
    """The visits of one route step, room, family or the whole file."""

    def __init__(self):
        self.visits = 0
        self.runs: set = set()
        self.outcomes: Counter = Counter()
        self.means = {name: Mean() for name in VISIT_MEANS}
        self.frames: list = []
        self.lost_frames: list = []
        self.families: Counter = Counter()
        self.rooms: Counter = Counter()

    def add(self, run_key, outcome: str, figures: dict, family: str, room: str):
        self.visits += 1
        self.runs.add(run_key)
        self.outcomes[outcome] += 1
        self.families[family] += 1
        self.rooms[room] += 1
        for name, mean in self.means.items():
            value = figures.get(name)
            mean.add(float(value) if value is not None else None)
        if figures.get('frames') is not None:
            self.frames.append(figures['frames'])
        if figures.get('lost_frame') is not None:
            self.lost_frames.append(figures['lost_frame'])

    def out(self) -> dict:
        exits = self.outcomes.get('exit-correct', 0)
        known = {'exit-correct', 'exit-wrong', 'death', 'stuck'}
        group = {'visits': self.visits, 'runs': len(self.runs), 'exits': exits,
                 'wrong_exits': self.outcomes.get('exit-wrong', 0), 'deaths': self.outcomes.get('death', 0),
                 'stuck': self.outcomes.get('stuck', 0),
                 'other_endings': {k: v for k, v in sorted(self.outcomes.items()) if k not in known},
                 'frames': distribution(self.frames),
                 'lost_frame': distribution(self.lost_frames)}
        group.update({name: mean.out() for name, mean in self.means.items()})
        if len(self.families) == 1:
            group['family'] = next(iter(self.families))
        elif self.families:
            group['families'] = dict(self.families.most_common())
        if len(self.rooms) == 1:
            group['room'] = next(iter(self.rooms))
        return group


class Goals:
    """The route's goals over one step's visits (round 40, ``crates/zelda/src/goals.rs``: a visit's ``goal_*``
    fields): what the step expected, from the route's data, beside what happened.  Only visits carrying the fields
    count."""

    def __init__(self):
        self.visits = 0
        self.expected: Counter = Counter()
        self.crossed = 0
        self.crossed_kills: list = []        # deaths that moved the drop cycle, carriers aside, a crossed visit
        self.cycle_in: Counter = Counter()
        self.wraps: list = []                # 1 a visit that took a wrap, 0 not (where one was expected)
        self.no_hit: list = []               # 1 a visit without a hit (where a planned path was expected)
        self.carrier_in = self.carrier_seen = self.carrier_killed = 0
        self.carrier_bombs = self.carrier_taken = 0
        # Round 46, a crossed visit carrying the fields: 1 it killed / it left no monster alive (of a room that had
        # one), the detour (tiles) and the frames it walked away from the way out with no bomb reason.
        self.crossed_killed: list = []
        self.crossed_cleared: list = []
        self.crossed_detour: list = []
        self.crossed_away_free: list = []

    def add(self, v: dict):
        self.visits += 1
        for name in ('wrap', 'no_hit', 'carriers'):
            self.expected[name] += v.get(f'goal_{name}_expected') or 0
        if not v.get('goal_fought'):
            self.crossed += 1
            self.crossed_kills.append((v.get('goal_cycle_kills') or 0) - (v.get('goal_carrier_kills') or 0))
            if v.get('goal_alive_max') is not None:
                self.crossed_killed.append(1 if v.get('kills') else 0)
                if v['goal_alive_max']:
                    self.crossed_cleared.append(0 if v['goal_alive_out'] else 1)
                if v['goal_way_known']:
                    self.crossed_detour.append(v['goal_detour'])
                self.crossed_away_free.append(v['goal_away_free_frames'])
        self.cycle_in[v.get('goal_cycle_in')] += 1
        if v.get('goal_wrap_expected'):
            self.wraps.append(1 if v.get('goal_wraps') else 0)
        if v.get('goal_no_hit_expected') and v.get('hits') is not None:
            self.no_hit.append(0 if v['hits'] else 1)
        self.carrier_in += v.get('goal_carrier_in') or 0
        self.carrier_seen += v.get('goal_carrier_seen') or 0
        self.carrier_killed += 1 if v.get('goal_carrier_kills') else 0
        self.carrier_bombs += v.get('goal_carrier_bombs') or 0
        self.carrier_taken += v.get('goal_carrier_bombs_taken') or 0

    def out(self) -> dict:
        share = lambda xs: rate_block(sum(xs), len(xs)) if xs else None  # noqa: E731
        return {'visits': self.visits, 'expects': [k for k, n in self.expected.items() if n],
                'crossed': self.crossed, 'crossed_kills': distribution(self.crossed_kills),
                'crossed_killed': share(self.crossed_killed), 'crossed_cleared': share(self.crossed_cleared),
                'crossed_detour': distribution(self.crossed_detour),
                'crossed_away_free': distribution(self.crossed_away_free),
                'cycle_in': {str(k): n for k, n in sorted(self.cycle_in.items(), key=lambda kv: (kv[0] is None, kv[0]))},
                'wrap_taken': share(self.wraps), 'no_hit': share(self.no_hit),
                'carrier': {'in': self.carrier_in, 'seen': self.carrier_seen, 'killed': self.carrier_killed,
                            'bombs': self.carrier_bombs, 'taken': self.carrier_taken}}


def goal_run_values(played: list) -> dict:
    """One run's goal measures (None for records without the fields): kills where the route crossed (carriers
    aside), expected wraps not taken, hits in rooms with a planned path, carriers' bombs taken."""
    if not played or any(v.get('goal_fought') is None for v in played):
        return {'goal_crossed_kills': None, 'goal_wraps_missed': None, 'goal_planned_hits': None,
                'goal_carrier_bombs': None}
    return {
        'goal_crossed_kills': sum((v['goal_cycle_kills'] - v['goal_carrier_kills']) for v in played
                                  if not v['goal_fought']),
        'goal_wraps_missed': sum(1 for v in played if v['goal_wrap_expected'] and not v['goal_wraps']),
        'goal_planned_hits': sum(v.get('hits') or 0 for v in played if v['goal_no_hit_expected']),
        'goal_carrier_bombs': sum(v['goal_carrier_bombs_taken'] for v in played),
    }


def room_key(level, room, in_cave=False) -> str:
    """``3:0x5b`` (``:cave`` in a cave); ``room`` an int or the records' ``0x5b``."""
    text = room if isinstance(room, str) else f"0x{room:02x}"
    return f"{level}:{text}" + (':cave' if in_cave else '')


# ------------------------------------------------------------------------------------------------- run measures

def result_letter(run: dict) -> str:
    """w win, d death, s stuck (any other end), e error (``runner.ChainTally``'s endings)."""
    if run['result'] == 'success':
        return 'w'
    if run['result'] == 'error':
        return 'e'
    return 'd' if run.get('reason', '').startswith('died in') else 's'


def stuck_reason(reason: str) -> str:
    """A stuck run's reason without its rooms and counts: ``bounce-limit``, ``wrong-exit-limit``, ``max-steps``,
    ``left-route``, ``masked-action``, else its words before the first parenthesis."""
    if re.match(r'^[a-z-]+-limit\b', reason):
        return reason.split(':')[0].split()[0]
    if 'exceeded' in reason and 'steps' in reason:
        return 'max-steps'
    if 'left the route' in reason:
        return 'left-route'
    if 'chose masked action' in reason:
        return 'masked-action'
    return reason.split('(')[0].strip()[:60] or 'unknown'


def beams_held(rooms: list) -> tuple:
    """The lead's room-level measure (any record): (records left still holding the beams before the first left
    without them, frames to the entry of that room, its room, lost at all).  Holding: within half a heart of full
    (asm:SwordShot_Make).  A chain's records with its milestone (the arrival of a won run counts as one more, as in
    the lead's measure of Nov 23)."""
    t = 0
    for i, v in enumerate(rooms):
        if v.get('health_out') is not None and not v['health_out'] > v['max_health'] - 0.5:
            return i, t, room_key(v['level'], v['room'], v.get('in_cave')), True
        t += v.get('frames') or 0
    return len(rooms), t, None, False


def beams_lost_at(rooms: list) -> Optional[tuple]:
    """Round 40's counters: (run frame at which the beams were first lost or None, frames held in the run); None for
    records without them."""
    if not rooms or not all(round40(v) for v in rooms):
        return None
    t, held = 0, 0
    lost = None
    for v in rooms:
        held += v['beam_held_frames']
        if lost is None and v.get('beam_lost_frame') is not None:
            lost = t + v['beam_lost_frame']
        t += v.get('frames') or 0
    return lost, held


def _sum(rooms: list, field: str, need: bool = True):
    """Sum of a visit field; None when ``need`` and a visit lacks it."""
    values = [v.get(field) for v in rooms]
    if need and any(x is None for x in values):
        return None
    return sum(x or 0 for x in values)


# ------------------------------------------------------------------------------------------------------- chains

def chain_stats(runs: list, meta: Optional[dict] = None, steps: Optional[list] = None,
                own_until: Optional[int] = None, source: Optional[str] = None, segment: Optional[str] = None,
                overrides: Optional[list] = None) -> dict:
    """The statistics of a chain's records (``runs``: the jsonl's dicts), its ``.meta.json`` (``meta``), the route's
    steps (``_native.route_steps``) and ``own_models_until`` (for the canaries); ``segment``: the route segment its
    start state begins (``eval_batch.segment_starts``), ``overrides``: its ``--checkpoint`` overrides as ``KEY = run``
    (``checkpoint_overrides``), both for the one-line ``description``.  Pure: reads nothing."""
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    runs = sorted(runs, key=lambda r: (r.get('seed') is None, r.get('seed')))
    provenance = (meta or {}).get('provenance') or {}
    route = provenance.get('route') or (runs[0].get('route') if runs else None)
    groups_all = Group()
    by_step: dict = defaultdict(Group)
    by_room: dict = defaultdict(Group)
    by_family: dict = defaultdict(Group)
    endings = Counter()
    stuck_reasons, death_rooms, stuck_rooms, error_reasons = Counter(), Counter(), Counter(), Counter()
    triforce_steps = {s['index']: s['level'] for s in steps or () if s.get('exit') == 'triforce'}
    triforces, game_ended = Counter(), 0
    furthest = None
    per_run = defaultdict(list)
    enter: dict = defaultdict(dict)
    networks: dict = defaultdict(Counter)
    beam_first_room = Counter()
    by_goal: dict = defaultdict(Goals)
    seeds = []
    for index, run in enumerate(runs):
        seed = run.get('seed')
        seeds.append(seed)
        key = seed if seed is not None else f'#{index}'
        letter = result_letter(run)
        endings[letter] += 1
        rooms = run.get('rooms') or []
        played = [v for v in rooms if v.get('outcome') != 'milestone']
        last = played[-1] if played else None
        if letter == 'd' and last is not None:
            death_rooms[room_key(last['level'], last['room'], last.get('in_cave'))] += 1
        elif letter == 's':
            stuck_reasons[stuck_reason(run.get('reason', ''))] += 1
            if last is not None:
                stuck_rooms[room_key(last['level'], last['room'], last.get('in_cave'))] += 1
        elif letter == 'e':
            error_reasons[run.get('reason', '')[:120]] += 1
        if run.get('reason', '').startswith('the game ended'):
            game_ended += 1
        for level in {triforce_steps[v['step']] for v in played
                      if v.get('step') in triforce_steps and v.get('outcome') == 'exit-correct'}:
            triforces[level] += 1
        t = 0
        for v in played:
            room = room_key(v['level'], v['room'], v.get('in_cave'))
            figures = visit_figures(v)
            family = v.get('specialist', '?')
            outcome = v.get('outcome', '?')
            groups_all.add(key, outcome, figures, family, room)
            by_room[room].add(key, outcome, figures, family, room)
            by_family[family].add(key, outcome, figures, family, room)
            step = v.get('step')
            if step is not None:
                by_step[step].add(key, outcome, figures, family, room)
                networks[step][v.get('policy') or family] += 1
                if v.get('goal_fought') is not None:
                    by_goal[step].add(v)
                enter[step].setdefault(index, t)
                furthest = step if furthest is None else max(furthest, step)
            t += v.get('frames') or 0
        rooms_held, frames_held, lost_room, lost = beams_held(rooms)
        if lost:
            beam_first_room[lost_room] += 1
        exact = beams_lost_at(played)
        per_run['frames'].append(run.get('frames'))
        per_run['furthest'].append(max((v['step'] for v in played if v.get('step') is not None), default=None))
        per_run['end'].append(run.get('frames') if letter == 'w' else None)
        per_run['hearts_end'].append(rooms[-1].get('health_out') if rooms else None)
        per_run['hearts_lost'].append(_sum(played, 'beam_hearts_lost'))
        per_run['hits'].append(_sum(played, 'hits'))
        per_run['aimless_frames'].append(_sum(played, 'aimless_frames'))
        per_run['shots'].append(_sum(played, 'beam_shots'))
        per_run['shots_hit'].append(_sum(played, 'beam_shots_hit') if exact is not None else None)
        per_run['kills_far'].append(_sum(played, 'beam_kills_far'))
        per_run['kills_near'].append(_sum(played, 'beam_kills_near'))
        per_run['beam_rooms'].append(rooms_held)
        per_run['beam_frames'].append(frames_held)
        per_run['beam_lost_at'].append(None if exact is None else exact[0])
        per_run['beam_held'].append(None if exact is None else exact[1])
        for name, value in goal_run_values(played).items():
            per_run[name].append(value)
    n = len(runs)
    wins = endings['w']
    rate = wins / n if n else None
    names = {}
    for step in by_step:
        g = by_step[step]
        names[step] = (next(iter(g.rooms)) if g.rooms else '?', g.families.most_common(1)[0][0] if g.families else '?')
    enter_rows = {str(step): [enter[step].get(i) for i in range(n)] for step in sorted(enter)}
    counters = ('round40' if any(round40(v) for r in runs for v in r.get('rooms') or ())
                else 'round38' if any('beam_shots' in v for r in runs for v in r.get('rooms') or ()) else 'older')
    batches = (meta or {}).get('batches') or []
    exact_lost = [x for x in per_run['beam_lost_at'] if x is not None]
    has_exact = any(x is not None for x in per_run['beam_held'])
    stats = {
        'schema': SCHEMA, 'kind': 'chain', 'name': chain_name(source) if source else None,
        'source': source, 'written': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'manifest': (meta or {}).get('manifest'),
        'provenance': {**{k: provenance.get(k) for k in ('start_from', 'power_on', 'stop_at', 'stop_step', 'clips',
                                                         'clips_off_from', 'version', 'code', 'data')},
                       'route': route, 'segment': segment, 'overrides': list(overrides or ()),
                       'team': {m['name']: {'run': m.get('run'), 'act': m.get('act'),
                                            'checkpoint_sha1': (m.get('checkpoint_sha1') or '')[:12] or None}
                                for m in provenance.get('models') or ()},
                       'missing': provenance.get('missing')},
        'records': {'counters': counters, 'runs': n,
                    'asked': sum(b.get('runs', 0) for b in batches) if batches else None,
                    'unfinished_batches': sum(not b.get('finished', True) for b in batches),
                    'seeds': seed_facts(seeds)},
        'rate': rate_block(wins, n),
        'endings': {'wins': wins, 'deaths': endings['d'], 'stuck': endings['s'], 'errors': endings['e'],
                    'stuck_reasons': dict(stuck_reasons.most_common()),
                    'death_rooms': dict(death_rooms.most_common()), 'stuck_rooms': dict(stuck_rooms.most_common()),
                    'error_reasons': dict(error_reasons.most_common(10))},
        'triforces': {str(level): count for level, count in sorted(triforces.items())},
        'game_ended': game_ended,
        'furthest_step': furthest,
        'all': groups_all.out(),
        'by_step': {str(step): {**by_step[step].out(), 'room': names[step][0], 'family': names[step][1],
                                'enter': distribution(list(enter[step].values())),
                                **network_of(networks[step], provenance)}
                    for step in sorted(by_step)},
        'by_room': {room: g.out() for room, g in sorted(by_room.items())},
        'by_family': {family: g.out() for family, g in sorted(by_family.items())},
        'beams': {
            'rooms_with_beams': mean_of(per_run['beam_rooms']),
            'frames_with_beams': mean_of(per_run['beam_frames']),
            'never_lost': n - sum(beam_first_room.values()),
            'first_lost_room': dict(beam_first_room.most_common(12)),
            'lost_at': distribution(exact_lost) if has_exact else None,
            'held_frames': mean_of(per_run['beam_held']) if has_exact else None,
        },
        'goals': {str(step): {**by_goal[step].out(), 'room': names[step][0], 'family': names[step][1]}
                  for step in sorted(by_goal)},
        'time': {**{str(step): {**(distribution(list(enter[step].values())) or {}), 'room': names[step][0],
                                'family': names[step][1]} for step in sorted(enter)},
                 'end': distribution(per_run['end'])},
        'runs': {'seeds': seeds, 'result': ''.join(result_letter(r) for r in runs), **dict(per_run),
                 'enter': enter_rows},
    }
    stats['flags'] = chain_flags(stats, steps, own_until, runs)
    stats['description'] = describe(stats)
    return stats


def network_of(played: Counter, provenance: dict) -> dict:
    """The network that played a step most (its policy key) and that key's run and checkpoint SHA-1 in the team."""
    key = played.most_common(1)[0][0] if played else None
    model = next((m for m in provenance.get('models') or () if m['name'] == key), {})
    return {'network': key, 'network_run': model.get('run'),
            'network_sha1': (model.get('checkpoint_sha1') or '')[:12] or None}


def mean_of(values: Iterable) -> Optional[dict]:
    """``Mean`` over values (None skipped)."""
    m = Mean()
    for v in values:
        m.add(None if v is None else float(v))
    return m.out()


def rate_block(successes: int, n: int) -> dict:
    """n, successes, rate and its binomial standard error."""
    rate = successes / n if n else None
    return {'n': n, 'successes': successes, 'rate': _round(rate),
            'se': _round(math.sqrt(rate * (1 - rate) / n)) if n else None}


def seed_facts(seeds: list) -> dict:
    """First, last, whether they are the standard 0..N-1 (pairing with any standard chain), duplicates."""
    known = [s for s in seeds if s is not None]
    dup = [s for s, c in Counter(known).items() if c > 1]
    return {'first': min(known) if known else None, 'last': max(known) if known else None,
            'standard': sorted(known) == list(range(len(seeds))) and len(known) == len(seeds),
            'duplicates': sorted(dup)[:10]}


def flag(code: str, text: str, level: str = 'red') -> dict:
    """The flag's shape (shared with ``triforce.jobs.flag``)."""
    return {'level': level, 'code': code, 'text': text}


def common_flags(stats: dict, unit: str = 'runs', stuck: bool = True) -> list:
    """The flags a chain and an evaluation share: errors, extreme rate, stuck share (``stuck``: a chain's; an
    evaluation's other endings, a wrong exit, no damage, the step cap, are its task's ordinary failures), runs asked,
    seeds, old records."""
    flags = []
    rate, endings, records = stats['rate'], stats['endings'], stats['records']
    n = rate['n']
    if not n:
        return [flag('no-runs', f"the file holds no {unit}: check that the job played anything")]
    if endings['errors']:
        first = next(iter(endings['error_reasons']), '')
        flags.append(flag('run-error', f"{endings['errors']} of {n} {unit} ended in an error (first: {first}); "
                                       "read the reason before using any number here"))
    if n >= RATE_EXTREME_RUNS and rate['successes'] in (0, n):
        flags.append(flag('rate-extreme', f"the rate is exactly {100 * rate['rate']:.0f}% over {n} {unit}: check the "
                                          "start and stop (a segment that cannot fail or cannot succeed)", 'note'))
    if stuck and endings['stuck'] >= STUCK_MIN and endings['stuck'] > STUCK_SHARE * n:
        reasons = ', '.join(f"{k} {v}" for k, v in list(endings['stuck_reasons'].items())[:3])
        flags.append(flag('stuck-share', f"{endings['stuck']} of {n} {unit} ended stuck ({reasons}), above "
                                         f"{STUCK_SHARE:.0%}: check the rooms in endings.stuck_rooms for a route or "
                                         "model fault"))
    asked = records.get('asked')
    if asked is not None and n < asked:
        flags.append(flag('runs-short', f"{n} {unit} recorded of {asked} asked: the job stopped early or lost runs"))
    if records.get('unfinished_batches'):
        flags.append(flag('runs-short', f"{records['unfinished_batches']} batch(es) of the file never finished"))
    seeds = records.get('seeds') or {}
    if seeds.get('duplicates'):
        flags.append(flag('seeds-duplicate', f"seeds {seeds['duplicates']} appear more than once: the file was "
                                             "appended to twice; pairing by seed is wrong"))
    elif seeds and not seeds.get('standard', True):
        flags.append(flag('seeds-nonstandard', f"the seeds are {seeds.get('first')}..{seeds.get('last')}, not the "
                                               f"standard 0..{n - 1}: it pairs only with files on the same seeds",
                          'note'))
    if records.get('counters') != 'round40':
        flags.append(flag('records-old', f"the records predate round 40's counters ({records.get('counters')}): "
                                         "shots that hit and the frames holding the beams are null", 'note'))
    return flags


def chain_flags(stats: dict, steps: Optional[list], own_until: Optional[int], runs: list) -> list:
    """The result canaries of a chain (module doc; docs/architecture.md "Run statistics")."""
    # pylint: disable=too-many-locals
    flags = common_flags(stats)
    if not steps or not runs:
        return flags
    prov = stats['provenance']
    route = prov.get('route')
    visited = {int(k): g for k, g in stats['by_step'].items()}
    if not visited:
        flags.append(flag('no-route-steps', "no visit carries a route step: the records' step field is missing"))
        return flags
    moved = sorted(k for k, g in visited.items()
                   if k >= len(steps) or g.get('room') != room_key(steps[k]['level'], steps[k]['room'],
                                                                   steps[k]['in_cave']))
    if moved:
        flags.append(flag('route-renumbered', f"the records' route steps are not this checkout's (step {moved[0]} is "
                                              f"{visited[moved[0]].get('room')} in them): the records were played on "
                                              "another route version, so the route canaries were not run", 'note'))
        return flags
    low, high = min(visited), max(visited)
    if prov.get('stop_step') is not None:
        high = min(high, prov['stop_step'] - 1)     # the stop step is reached, not played (the run's milestone)
    clips = prov.get('clips')
    mixed = prov.get('clips_off_from') is not None or clips is None

    def compatible(step: dict) -> bool:
        way = step.get('way')
        if way is None:
            return True
        if mixed:
            return False
        return way == ('clips' if clips else 'plain')
    span = [s for s in steps if low <= s['index'] <= high and compatible(s)]
    for before, after in zip(span, span[1:]):
        left = (visited.get(before['index']) or {}).get('exits', 0)
        if left >= CANARY_MIN and after['index'] not in visited:
            flags.append(flag('step-never-entered',
                              f"route step {after['index']} ({room_key(after['level'], after['room'], after['in_cave'])}"
                              f" {after['specialist']}) was never entered though {left} visits left step "
                              f"{before['index']} by its exit: check the route data and the records' step field"))
    families = Counter(s['specialist'] for s in span)
    played = stats['by_family']
    for family, count in families.items():
        if family not in played and family != '-':
            flags.append(flag('family-never-played', f"{family} has {count} route step(s) in the steps this chain "
                                                     f"played ({low}-{high}) and not one visit: check the team and "
                                                     "the route's specialists"))
    team = prov.get('team') or {}
    decisions = Counter()
    for run in runs:
        for v in run.get('rooms') or ():
            if v.get('outcome') != 'milestone':
                decisions[v.get('policy') or v.get('specialist')] += (v.get('decisions') or {}).get('n', 0)
    expected = set()
    for s in span:
        own = f"{s['specialist']}@{route}"
        expected.add(own if own_until is not None and s['index'] < own_until and own in team else s['specialist'])
    for key in sorted(expected & set(team)):
        if decisions[key] == 0 and played.get(key.split('@')[0], {}).get('visits', 0) >= CANARY_MIN:
            flags.append(flag('network-no-decisions', f"the team lists {key} for steps this chain played and it made "
                                                      "no decision: another network played its rooms; check the "
                                                      "team and the --checkpoint overrides"))
    return flags


def describe(stats: dict) -> str:
    """What the file IS, in the words a person scanning ``stats list`` needs: a chain's segment (or start), end, runs
    and its ``--checkpoint`` overrides (the arm); an evaluation's family, run, act mode and episodes; a training's
    family, run and iterations."""
    prov = stats.get('provenance') or {}
    if stats['kind'] == 'chain':
        start = prov.get('start_from')
        if start and prov.get('route') and start.startswith(f"{prov['route']}_"):
            start = start[len(prov['route']) + 1:]
        start = start or ('power on' if prov.get('power_on') else 'the start')
        where = prov.get('segment') or start
        end = (f"step {prov['stop_step']}" if prov.get('stop_step') is not None
               else prov.get('stop_at') or 'the end')
        arm = ', '.join(prov.get('overrides') or ()) or 'no --checkpoint override'
        return f"{where} ({start} to {end}), {stats['rate']['n']} runs, {arm}"
    if stats['kind'] == 'evaluate':
        run = Path(prov.get('checkpoint') or 'run/ckpt.pt').parent.name
        return f"evaluate {prov.get('specialist')} {run} ({prov.get('act')}), {stats['rate']['n']} episodes"
    return (f"train {prov.get('specialist')} {prov.get('run_id')}, {stats['records']['iterations']} iterations to "
            f"step {stats['records']['steps']}")


def checkpoint_overrides(command: Iterable, output=None) -> list:
    """``KEY = run`` for each ``--checkpoint KEY=PATH`` of a command (a batch's: of the ``--chain`` entry naming
    ``output``'s file)."""
    tokens = [str(t) for t in command or ()]
    if output is not None:
        name = Path(output).name
        spec = next((t for t in tokens if name in t and ' ' in t), None)
        if spec is not None:
            head = spec.split(' ', 1)[0]
            tokens = shlex.split(spec.split(':', 1)[1] if head.endswith(':') else spec)
    out = []
    for i, token in enumerate(tokens):
        value = (tokens[i + 1] if token == '--checkpoint' and i + 1 < len(tokens)
                 else token.split('=', 1)[1] if token.startswith('--checkpoint=') else None)
        if value and '=' in value:
            key, path = value.split('=', 1)
            path = Path(path)
            out.append(f"{key} = {path.parent.name if path.suffix == '.pt' else path.name}")
    return out


def source_job(path) -> Optional[dict]:
    """The queue job that wrote the chain output ``path``: the latest whose command names its file."""
    # pylint: disable=import-outside-toplevel
    from .jobqueue import connect, read_jobs
    path = Path(path).resolve()
    try:
        con = connect()
        try:
            rows = read_jobs(con, 'command LIKE ?', (f'%{path.name}%',))
        finally:
            con.close()
    except Exception:  # pylint: disable=broad-except
        return None
    named = [r for r in rows if 'ingest' not in (r.get('command') or ())
             and any(str(path) in str(t) or f'/{path.name}' in str(t) or str(t) == path.name
                     for t in r.get('command') or ())]
    return named[-1] if named else None


def chain_name(source) -> str:
    """A chain file's name: its file name without ``.jsonl`` (and a batch's ``.tmp``)."""
    name = Path(source).name
    for suffix in ('.tmp', '.jsonl'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def read_jsonl(path) -> list:
    """A chain file's runs."""
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path) -> Optional[dict]:
    """A JSON file, None when it does not exist."""
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def route_context(route: Optional[str]) -> tuple:
    """(route steps, own_models_until) of a route this checkout knows, else (None, None)."""
    # pylint: disable=import-outside-toplevel
    if not route:
        return None, None
    from ._native import route_steps, routes
    from .registry import own_models_until
    if route not in routes():
        return None, None
    return route_steps(route), own_models_until(route)


def compute_chain(path, overrides: Optional[list] = None) -> dict:
    """The statistics of the chain output ``path`` (its runs and ``.meta.json``; ``overrides`` as ``chain_stats``)."""
    # pylint: disable=import-outside-toplevel
    from .eval_batch import bare_state, segment_starts
    path = Path(path)
    meta = read_json(f'{path}.meta.json')
    runs = read_jsonl(path)
    provenance = (meta or {}).get('provenance') or {}
    route = provenance.get('route') or (runs[0].get('route') if runs else None)
    steps, until = route_context(route)
    start = provenance.get('start_from')
    segment = (segment_starts(route).get(bare_state(start)) if start else None) if steps else None
    if segment is None and steps and not start and not provenance.get('power_on'):
        segment = 'full'
    return chain_stats(runs, meta, steps, until, source=str(path.resolve()), segment=segment, overrides=overrides)


def write_chain(path) -> Optional[Path]:
    """Writes ``<path>.stats.json`` and registers it; a failure is printed, never raised (the chain's records stand
    and ``triforce stats ingest`` computes it again)."""
    try:
        stats = compute_chain(path, checkpoint_overrides(sys.argv[1:]))
        register(stats)
        return write(stats, stats_path(path))
    except Exception as error:  # pylint: disable=broad-except
        print(f"triforce: statistics of {path} not written: {type(error).__name__}: {error} "
              f"(`triforce stats ingest {path}` writes them)", file=sys.stderr)
        return None


def move_chain(tmp, output):
    """A batch's chain moved from ``tmp`` to ``output`` (``eval_batch``): its statistics follow, naming the final
    file, and are registered again under the same key."""
    old = stats_path(tmp)
    if not old.is_file():
        return
    stats = json.loads(old.read_text(encoding='utf-8'))
    stats['source'] = str(Path(output).resolve())
    stats['name'] = chain_name(output)
    register(stats)
    write(stats, stats_path(output))
    old.unlink()


# -------------------------------------------------------------------------------------------------- evaluations

def eval_letter(record: dict) -> str:
    if record.get('success'):
        return 'w'
    ending = str(record.get('ending') or record.get('outcome') or '')
    if ending == 'error':
        return 'e'
    return 'd' if ending in ('death', 'died') or record.get('health', 1) <= 0 else 's'


def eval_stats(result: dict, source: Optional[str] = None, meta: Optional[dict] = None) -> dict:
    """The statistics of an evaluation result (``evaluate.evaluate_specialist``'s dict, ``format`` 2 with every
    episode's ``visits``).  Pure."""
    # pylint: disable=too-many-locals
    records = sorted(result.get('records') or (), key=lambda r: (r.get('start_state', ''), r.get('seed', 0)))
    family = result.get('specialist')
    groups_all, by_room = Group(), defaultdict(Group)
    by_state: dict = defaultdict(lambda: [0, 0])
    endings, death_rooms, stuck_reasons, error_reasons = Counter(), Counter(), Counter(), Counter()
    per_run = defaultdict(list)
    keys = []
    for record in records:
        key = f"{record.get('start_state')}#{record.get('seed')}"
        keys.append(key)
        letter = eval_letter(record)
        endings[letter] += 1
        by_state[record.get('start_state')][0] += 1
        by_state[record.get('start_state')][1] += bool(record.get('success'))
        visits = record.get('visits') or []
        if letter == 'd' and visits:
            death_rooms[room_key(visits[-1]['level'], visits[-1]['room'], visits[-1].get('in_cave'))] += 1
        elif letter == 's':
            stuck_reasons[str(record.get('ending') or record.get('outcome'))] += 1
        elif letter == 'e':
            error_reasons[str(record.get('outcome'))[:120]] += 1
        for i, v in enumerate(visits):
            room = room_key(v['level'], v['room'], v.get('in_cave'))
            outcome = ('exit-correct' if record.get('success') else {'d': 'death', 's': 'stuck', 'e': 'error'}[letter]
                       ) if i == len(visits) - 1 else 'exit'
            figures = visit_figures(v, 'evaluate')
            groups_all.add(key, outcome, figures, family, room)
            by_room[room].add(key, outcome, figures, family, room)
        rooms_held, frames_held, _, _ = beams_held(visits)
        exact = beams_lost_at(visits)
        per_run['frames'].append(record.get('frames'))
        per_run['hearts_end'].append(record.get('health'))
        per_run['hearts_lost'].append(_sum(visits, 'health_lost') if visits else None)
        per_run['hits'].append(_sum(visits, 'hits') if visits else None)
        per_run['aimless_frames'].append(_sum(visits, 'aimless_frames') if visits else None)
        per_run['shots'].append(_sum(visits, 'beam_shots') if visits else None)
        per_run['shots_hit'].append(_sum(visits, 'beam_shots_hit') if exact is not None else None)
        per_run['kills_far'].append(_sum(visits, 'beam_kills_far') if visits else None)
        per_run['kills_near'].append(_sum(visits, 'beam_kills_near') if visits else None)
        per_run['beam_rooms'].append(rooms_held)
        per_run['beam_frames'].append(frames_held)
        per_run['beam_lost_at'].append(None if exact is None else exact[0])
        per_run['beam_held'].append(None if exact is None else exact[1])
    n = len(records)
    has_visits = any(r.get('visits') for r in records)
    counters = ('round40' if any(round40(v) for r in records for v in r.get('visits') or ())
                else 'round38' if any('beam_shots' in v for r in records for v in r.get('visits') or ())
                else 'older')
    stats = {
        'schema': SCHEMA, 'kind': 'evaluate', 'name': None, 'source': source,
        'written': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'manifest': (meta or {}).get('manifest') or result.get('manifest'),
        'provenance': {k: result.get(k) for k in ('specialist', 'checkpoint', 'checkpoint_sha1', 'act', 'clips',
                                                  'episodes_per_state', 'seed', 'version', 'code', 'data')},
        'records': {'counters': counters, 'runs': n, 'asked': None, 'unfinished_batches': 0, 'seeds': {},
                    'visits': has_visits},
        'rate': rate_block(endings['w'], n),
        'endings': {'wins': endings['w'], 'deaths': endings['d'], 'stuck': endings['s'], 'errors': endings['e'],
                    'stuck_reasons': dict(stuck_reasons.most_common()), 'death_rooms': dict(death_rooms.most_common()),
                    'stuck_rooms': {}, 'error_reasons': dict(error_reasons.most_common(10))},
        'by_state': {state: rate_block(s, k) for state, (k, s) in sorted(by_state.items())},
        'all': groups_all.out(),
        'by_room': {room: g.out() for room, g in sorted(by_room.items())},
        'by_family': {family: groups_all.out()} if family else {},
        'beams': {'rooms_with_beams': mean_of(per_run['beam_rooms']),
                  'frames_with_beams': mean_of(per_run['beam_frames']),
                  'held_frames': mean_of(per_run['beam_held'])},
        'runs': {'seeds': keys, 'result': ''.join(eval_letter(r) for r in records), **dict(per_run)},
    }
    flags = common_flags(stats, 'episodes', stuck=False)
    if not has_visits and n:
        flags.append(flag('no-visits', "the evaluation's records have no room visits (format 1): per-room "
                                       "statistics are empty; evaluate again with this code", 'note'))
    config_rooms = (result.get('config') or {}).get('rooms') or ()
    seen = {room_key(v['level'], v['room'], v.get('in_cave')) for r in records for v in r.get('visits') or ()}
    if has_visits and n >= CANARY_MIN:
        for text in config_rooms:
            m = re.match(r'\((\d+), (0x[0-9a-f]+)(, cave)?\)', text)
            if m and room_key(int(m.group(1)), m.group(2), bool(m.group(3))) not in seen:
                flags.append(flag('family-room-never-played', f"{family}'s room {text} was never played in "
                                                              f"{n} episodes: no start state reaches it; check "
                                                              "its start_states"))
    stats['flags'] = flags
    stats['description'] = describe(stats)
    return stats


def eval_name(result: dict, path) -> str:
    """``<specialist>-<run>-<eval file stem>``."""
    checkpoint = Path(result.get('checkpoint') or 'run/ckpt.pt')
    return f"{result.get('specialist')}-{checkpoint.parent.name}-{Path(path).stem}"


def write_eval(result: dict, path) -> Optional[Path]:
    """Writes ``<path>.stats.json`` of an evaluation written to ``path`` and registers it; the run dir's training
    statistics take it in when ``path`` is its ``eval.json``.  A failure is printed, never raised."""
    try:
        stats = eval_stats(result, str(Path(path).resolve()), read_json(f'{path}.meta.json'))
        stats['name'] = eval_name(result, path)
        register(stats)
        written = write(stats, stats_path(path))
        if Path(path).name == 'eval.json' and (Path(path).parent / TRAIN_NAME).is_file():
            write_train(Path(path).parent)
        return written
    except Exception as error:  # pylint: disable=broad-except
        print(f"triforce: statistics of {path} not written: {type(error).__name__}: {error}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------------------------------- trainings

def train_stats(run: dict, metrics: list, summary: Optional[dict], evaluation: Optional[dict],
                source: Optional[str] = None) -> dict:
    """The statistics of a training: ``run`` (its ``run.json``), ``metrics`` (``metrics.jsonl``'s rows), its
    ``train_summary.json`` and its evaluation's statistics (``eval.json.stats.json``) when they exist.  Pure."""
    args = run.get('args') or {}
    rate_key = 'balanced_success_rate' if any('balanced_success_rate' in m for m in metrics) else 'success_rate'
    reached = {}
    for threshold in sorted({*THRESHOLDS, *([args['stop_success']] if args.get('stop_success') else [])}):
        hit = next((m for m in metrics if (m.get(rate_key) or 0) >= threshold), None)
        reached[str(threshold)] = None if hit is None else {'iteration': hit.get('iteration'), 'steps': hit.get('steps')}
    window = metrics[-TRAIN_WINDOW:]
    final = {key: mean_of(m.get(key) for m in window)
             for key in ('success_rate', 'balanced_success_rate', 'health_change', 'health_change_total',
                         'hearts_at_exit', 'hearts/taken', 'hearts/dropped', 'linger/max_frames',
                         'linger/tail_frames', 'episode_frames', 'episode_steps', 'episodes', 'harvested/success_rate',
                         'charts/SPS')}
    harvested = run.get('harvested') or {}
    drawn = sum(m.get('harvested/episodes') or 0 for m in metrics)
    tracks = run.get('curriculum_tracks')
    evaluation_block = None if evaluation is None else {
        'source': evaluation.get('source'), 'rate': evaluation.get('rate'),
        'hits': (evaluation.get('all') or {}).get('hits'), 'hearts_lost': (evaluation.get('all') or {}).get('hearts_lost'),
        'frames': (evaluation.get('all') or {}).get('frames'), 'flags': evaluation.get('flags')}
    stats = {
        'schema': SCHEMA, 'kind': 'train', 'name': f"{run.get('specialist')}-{run.get('run_id')}-train",
        'source': source, 'written': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'manifest': run.get('manifest'),
        'provenance': {'specialist': run.get('specialist'), 'run_id': run.get('run_id'), 'source': run.get('source'),
                       'start_iteration': run.get('start_iteration'), 'start_steps': run.get('start_steps'),
                       'parameters': run.get('parameters'), 'stop_success': args.get('stop_success'),
                       'iterations_asked': args.get('iterations'), 'start_state': args.get('start_state')},
        'records': {'iterations': len(metrics), 'last_iteration': metrics[-1].get('iteration') if metrics else None,
                    'steps': metrics[-1].get('steps') if metrics else None},
        'summary': summary,
        'steps_per_second': (summary or {}).get('steps_per_second'),
        'rate_metric': rate_key,
        'final': {'iterations': len(window), **final},
        'reached': reached,
        'harvested': {**harvested, 'episodes_drawn': drawn},
        'curriculum': {'dir': run.get('curriculum'), 'tracks': None if tracks is None else len(tracks)},
        'evaluation': evaluation_block,
    }
    flags = []
    if not metrics:
        flags.append(flag('no-iterations', "metrics.jsonl holds no iteration: the training never trained"))
    if (harvested.get('fraction') or 0) > 0 and not args.get('start_state') and metrics and drawn == 0:
        flags.append(flag('harvested-none', f"harvested_fraction is {harvested['fraction']:.0%} and no episode started "
                                            f"from a harvested state ({harvested.get('states')} states in "
                                            f"{harvested.get('dir')}): check the worktree's triforce/states/harvested"))
    if run.get('curriculum') and tracks is not None and len(tracks) == 0:
        flags.append(flag('curriculum-empty', f"the curriculum is on ({run.get('curriculum')}) and has no track: "
                                              "check the trajectories directory"))
    if (summary or {}).get('status') == 'interrupted':
        flags.append(flag('interrupted', "the training was interrupted: its figures stop early", 'note'))
    if evaluation is not None:
        flags.extend({**f, 'code': f"eval-{f['code']}"} for f in evaluation.get('flags') or ())
    stats['flags'] = flags
    stats['description'] = describe(stats)
    return stats


def compute_train(run_dir) -> dict:
    """A run dir's training statistics (its files)."""
    run_dir = Path(run_dir)
    metrics_path = run_dir / 'metrics.jsonl'
    metrics = read_jsonl(metrics_path) if metrics_path.is_file() else []
    return train_stats(read_json(run_dir / 'run.json') or {}, metrics, read_json(run_dir / 'train_summary.json'),
                       read_json(stats_path(run_dir / 'eval.json')), source=str(run_dir.resolve()))


def write_train(run_dir) -> Optional[Path]:
    """Writes ``<run dir>/train.stats.json`` and registers it; a failure is printed, never raised."""
    try:
        stats = compute_train(run_dir)
        register(stats)
        return write(stats, Path(run_dir) / TRAIN_NAME)
    except Exception as error:  # pylint: disable=broad-except
        print(f"triforce: statistics of {run_dir} not written: {type(error).__name__}: {error}", file=sys.stderr)
        return None


# ------------------------------------------------------------------------------------------------- files, store

def stats_path(output) -> Path:
    """``<output>.stats.json``."""
    return Path(f'{output}{SUFFIX}')


def write(stats: dict, path) -> Path:
    """Writes a statistics file atomically (compact JSON)."""
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(stats, separators=(',', ':')) + '\n', encoding='utf-8')
    os.replace(tmp, path)
    return path


def store_path() -> Path:
    """Where registered statistics are kept.  A placeholder (Nov 23: the shared state's one place, and the program
    every write goes through, are being decided with the owner): ``$XDG_STATE_HOME/triforce/stats`` (the queue's
    folder, ``jobqueue.queue_dir``)."""
    from .jobqueue import queue_dir  # pylint: disable=import-outside-toplevel
    return queue_dir() / 'stats'


def job_facts(job_id: Optional[int]) -> Optional[dict]:
    """The queue row of ``job_id``: agent, task, subtask, description, commit, branch, checkout, command."""
    if job_id is None:
        return None
    # pylint: disable=import-outside-toplevel
    from .jobqueue import connect, read_jobs
    try:
        con = connect()
        try:
            rows = read_jobs(con, 'id = ?', (job_id,))
        finally:
            con.close()
    except Exception:  # pylint: disable=broad-except
        rows = []
    if not rows:
        return {'id': job_id}
    row = rows[0]
    return {'id': job_id, 'agent': row.get('agent'), 'task': row.get('task'), 'subtask': row.get('subtask'),
            'description': row.get('description'), 'commit': row.get('commit_id'), 'branch': row.get('branch'),
            'checkout': row.get('checkout'), 'command': row.get('command')}


def register(stats: dict, agent: Optional[str] = None, task: Optional[str] = None) -> Path:
    """Puts ``stats`` into the store (``store_path``) as ``<job id>-<name>.stats.json``, its ``job`` block filled
    from this process's queue job (``agent`` and ``task``: the data's owners when they are not the job's, for an
    ingest).  The one way a file goes into the store; returns its path there."""
    from .jobs import job_id  # pylint: disable=import-outside-toplevel
    job = job_facts(job_id()) or {'id': None}
    if agent or task:
        job = {**job, 'ingested_by': {k: job.get(k) for k in ('agent', 'task', 'subtask')},
               'agent': agent or job.get('agent'), 'task': task or job.get('task')}
    stats['job'] = job
    store = store_path()
    store.mkdir(parents=True, exist_ok=True)
    base = f"{job['id'] if job['id'] is not None else 'nojob'}-{safe(stats.get('name') or 'stats')}"
    key, n = base, 1
    while (store / f'{key}{SUFFIX}').is_file():
        held = (read_json(store / f'{key}{SUFFIX}') or {}).get('source')
        if held in (stats.get('source'), f"{stats.get('source')}.tmp"):
            break               # the same output again (an append, a batch's chain moved from its .tmp)
        n += 1
        key = f'{base}-{n}'
    stats['id'] = key
    return write(stats, store / f'{key}{SUFFIX}')


def safe(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9._@-]+', '_', text)


def stored() -> list:
    """Every registered file's statistics (store order: newest last)."""
    store = store_path()
    if not store.is_dir():
        return []
    out = []
    for path in store.glob(f'*{SUFFIX}'):
        try:
            out.append(json.loads(path.read_text(encoding='utf-8')))
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda s: s.get('written') or '')


def load(ref: str) -> dict:
    """A statistics file by store id or path (a chain's jsonl, an evaluation's json or a run dir: its statistics
    file beside it)."""
    path = Path(ref)
    for candidate in (path / TRAIN_NAME, stats_path(path), path):
        if candidate.is_file() and candidate.name.endswith(SUFFIX):
            return json.loads(candidate.read_text(encoding='utf-8'))
    store = store_path()
    exact = store / f'{ref}{SUFFIX}'
    if exact.is_file():
        return json.loads(exact.read_text(encoding='utf-8'))
    matches = sorted(store.glob(f'*{ref}*{SUFFIX}')) if store.is_dir() else []
    if len(matches) == 1:
        return json.loads(matches[0].read_text(encoding='utf-8'))
    if matches:
        raise SystemExit(f"triforce stats: {ref} names {len(matches)} files: "
                         + ', '.join(m.name[:-len(SUFFIX)] for m in matches[:8]))
    raise SystemExit(f"triforce stats: no statistics for {ref} (a store id, a statistics file, or an output with "
                     f"its {SUFFIX} beside it; `triforce stats ingest FILE` computes one)")


# ------------------------------------------------------------------------------------------------------ reading

def fmt(x, digits: int = 2) -> str:
    if x is None:
        return '-'
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def mean_text(m: Optional[dict], digits: int = 2) -> str:
    if not m:
        return '-'
    return fmt(m['mean'], digits) + (f" ± {fmt(m['se'], digits)}" if m.get('se') is not None else '')


def table(header: list, rows: list) -> list:
    return ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' if i == 0 else '---:' for i in range(len(header)))
            + '|', *('| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows)]


def group_row(label: str, g: dict) -> list:
    v = g['visits'] or 1
    return [label, g.get('family') or ', '.join(list((g.get('families') or {}))[:3]), g['visits'], g['exits'],
            g['deaths'], g['stuck'], mean_text(g.get('hits')), mean_text(g.get('hearts_lost')),
            mean_text(g.get('health_in')),
            fmt((g.get('frames') or {}).get('mean'), 0), (g.get('frames') or {}).get('median', '-'),
            (g.get('frames') or {}).get('p90', '-'), mean_text(g.get('shots')),
            fmt((g.get('shots_hit') or {}).get('sum'), 0), fmt((g.get('kills_far') or {}).get('sum'), 0),
            fmt((g.get('kills_near') or {}).get('sum'), 0), fmt((g.get('beams_lost') or {}).get('sum', 0) / v
                                                                if g.get('beams_lost') else None),
            mean_text(g.get('aimless_frames'), 0)]


GROUP_HEADER = ['', 'family', 'visits', 'exits', 'deaths', 'stuck', 'hits/visit', 'hearts lost/visit', 'health in',
                'frames', 'median', 'p90', 'shots/visit', 'shots hit', 'far kills', 'near kills', 'beams lost/visit',
                'aimless frames/visit']


def show(stats: dict, rooms: int = 20, family: Optional[str] = None) -> str:
    """A statistics file as markdown."""
    # pylint: disable=too-many-locals
    job = stats.get('job') or {}
    lines = [f"# {stats.get('id') or stats.get('name')} ({stats['kind']})", '',
             f"source {stats.get('source')}; written {stats.get('written')}; job {job.get('id')} "
             f"{job.get('agent')}/{job.get('task')}/{job.get('subtask')} at {job.get('commit')}"
             + (f"; \"{job.get('description')}\"" if job.get('description') else ''), '']
    for f in stats.get('flags') or ():
        lines.append(f"- **{f['level'].upper()}** `{f['code']}`: {f['text']}")
    if stats['kind'] == 'train':
        return '\n'.join(lines + show_train(stats))
    rate, endings = stats['rate'], stats['endings']
    lines += ['', f"**rate** {rate['successes']} of {rate['n']} = {100 * (rate['rate'] or 0):.1f}% ± "
              f"{100 * (rate['se'] or 0):.1f}; deaths {endings['deaths']}, stuck {endings['stuck']} "
              f"({', '.join(f'{k} {v}' for k, v in endings['stuck_reasons'].items()) or '-'}), errors "
              f"{endings['errors']}; records {stats['records'].get('counters')}"]
    if stats.get('triforces'):
        lines.append('triforces taken: ' + ', '.join(f"level {k} {v}" for k, v in stats['triforces'].items())
                     + (f"; game ended {stats['game_ended']}" if stats.get('game_ended') else ''))
    beams = stats.get('beams') or {}
    lines.append(f"beams: rooms cleared holding them {mean_text(beams.get('rooms_with_beams'))}, frames to the entry "
                 f"of the room that took them {mean_text(beams.get('frames_with_beams'), 0)}"
                 + (f", frames held (counters) {mean_text(beams.get('held_frames'), 0)}" if beams.get('held_frames')
                    else '') + (f", never lost {beams['never_lost']}" if 'never_lost' in beams else ''))
    if stats.get('by_step') and not family:
        lines += ['', '## Route steps', '']
        rows = []
        for step, g in stats['by_step'].items():
            t = (stats.get('time') or {}).get(step) or {}
            rows.append([f"{step} {g.get('room')}", *group_row('', g)[1:], fmt(t.get('mean'), 0)])
        lines += table(['step', *GROUP_HEADER[1:], 'frames to enter'], rows)
        end = (stats.get('time') or {}).get('end')
        if end:
            lines.append(f"\nframes to the end (winners): {fmt(end['mean'], 0)} (median {end['median']}, "
                         f"p90 {end['p90']}, {end['mean'] / FPS:.0f} s)")
    if stats.get('goals') and not family:
        lines += ['', "## Route goals (what each step expected, from the route's data, and what happened)", '']
        lines += goal_table(stats['goals'])
    lines += ['', '## Families', '']
    lines += table(GROUP_HEADER, [group_row(name, g) for name, g in stats['by_family'].items()
                                  if family in (None, name)])
    room_rows = [(name, g) for name, g in stats['by_room'].items()
                 if family is None or g.get('family') == family or family in (g.get('families') or {})]
    room_rows.sort(key=lambda item: (-item[1]['deaths'], -item[1]['visits']))
    lines += ['', f"## Rooms (the {min(rooms, len(room_rows))} with most deaths)", '']
    lines += table(GROUP_HEADER, [group_row(name, g) for name, g in room_rows[:rooms]])
    if stats.get('by_state'):
        lines += ['', '## Start states', '']
        lines += table(['state', 'episodes', 'rate'], [[k, v['n'], f"{100 * v['rate']:.1f}% ± {100 * (v['se'] or 0):.1f}"]
                                                       for k, v in stats['by_state'].items()])
    return '\n'.join(lines)


def goal_cells(g: Optional[dict]) -> list:
    """A step's goal figures as table cells (``-`` for a file without the step)."""
    if not g:
        return ['-'] * len(GOAL_HEADER)
    share = lambda b: f"{b['successes']}/{b['n']}" if b else '-'  # noqa: E731
    c = g['carrier']
    cycle = ' '.join(f"{k}:{n}" for k, n in g['cycle_in'].items())
    return [g['visits'], g['crossed'], mean_text(g.get('crossed_kills')), share(g.get('crossed_killed')),
            share(g.get('crossed_cleared')), mean_text(g.get('crossed_detour')), mean_text(g.get('crossed_away_free')),
            cycle, share(g.get('wrap_taken')), share(g.get('no_hit')), f"{c['in']}/{c['seen']}", c['killed'],
            f"{c['taken']}/{c['bombs']}"]


GOAL_HEADER = ['visits', 'crossed', 'kills/crossed visit', 'crossed: killed', 'crossed: cleared',
               'crossed: detour tiles', 'crossed: frames away w/o bomb reason', 'cycle in (value:visits)', 'wrap taken',
               'no hit', 'carrier in/aimed at', 'carrier killed', 'carrier bombs taken/dropped']


def goal_table(goals: dict, other: Optional[dict] = None) -> list:
    """The route goals by step (one file, or two side by side: each cell ``base | arm``)."""
    rows = []
    steps = sorted(set(goals) | set(other or {}), key=int)
    for step in steps:
        g, o = goals.get(step), (other or {}).get(step)
        head = g or o
        label = f"{step} {head['room']}"
        expects = ', '.join(sorted(set(head['expects']) | set((o or {}).get('expects') or ()))) or '-'
        cells = goal_cells(g)
        if other is not None:
            cells = [f"{a} \\| {b}" for a, b in zip(cells, goal_cells(o))]
        rows.append([label, head.get('family', ''), expects, *cells])
    return table(['step', 'family', 'expects', *GOAL_HEADER], rows)


def show_train(stats: dict) -> list:
    final = stats['final']
    lines = ['', f"{stats['records']['iterations']} iterations to step {stats['records']['steps']}; "
                 f"{fmt(stats.get('steps_per_second'), 0)} steps a second; status "
                 f"{(stats.get('summary') or {}).get('status')}", '',
             f"## The last {final['iterations']} iterations", '']
    lines += table(['metric', 'mean ± se'], [[k, mean_text(v, 3)] for k, v in final.items()
                                              if isinstance(v, dict)])
    lines += ['', f"## Steps to each threshold of {stats['rate_metric']}", '']
    lines += table(['threshold', 'iteration', 'steps'],
                   [[k, (v or {}).get('iteration', '-'), (v or {}).get('steps', '-')] for k, v in stats['reached'].items()])
    h = stats['harvested']
    lines += ['', f"harvested: {h.get('states')} states, fraction {h.get('fraction')}, episodes drawn "
                  f"{h.get('episodes_drawn')}; curriculum {stats['curriculum']}"]
    if stats.get('evaluation'):
        e = stats['evaluation']
        lines.append(f"evaluation: {e['rate']['successes']} of {e['rate']['n']}, hits/visit {mean_text(e.get('hits'))}, "
                     f"hearts lost/visit {mean_text(e.get('hearts_lost'))}")
    return lines


def paired(base: dict, arm: dict, field: str, keep=None) -> Optional[tuple]:
    """(pairs, base mean, arm mean, mean difference, its se) of a per-run field over the seeds both files have
    (``win``: 1 a won run, else 0; ``keep(i_base, i_arm)``: which pairs count)."""
    bi = {s: i for i, s in enumerate(base['runs']['seeds'])}
    ai = {s: i for i, s in enumerate(arm['runs']['seeds'])}
    if field == 'win':
        bv, av = ([1.0 if c == 'w' else 0.0 for c in s['runs']['result']] for s in (base, arm))
    else:
        bv, av = base['runs'].get(field), arm['runs'].get(field)
    if bv is None or av is None:
        return None
    pairs = [(bv[bi[s]], av[ai[s]]) for s in bi if s in ai and (keep is None or keep(bi[s], ai[s]))]
    pairs = [(b, a) for b, a in pairs if b is not None and a is not None]
    if not pairs:
        return None
    d, se = mean_se([a - b for b, a in pairs])
    return len(pairs), sum(b for b, _ in pairs) / len(pairs), sum(a for _, a in pairs) / len(pairs), d, se


def pm(d, se, digits=2, scale=1.0) -> str:
    if d is None:
        return '-'
    return f"{d * scale:+.{digits}f}" + (f" ± {se * scale:.{digits}f}" if se is not None else '')


def time_pairs(base: dict, arm: dict, step: str) -> Optional[list]:
    """(base, arm) frames to entering ``step`` (``end``: to the end) over the seeds both entered it in; None when
    a file has no route time (an evaluation)."""
    bv = (base['runs'].get('enter') or {}).get(step) if step != 'end' else base['runs'].get('end')
    av = (arm['runs'].get('enter') or {}).get(step) if step != 'end' else arm['runs'].get('end')
    if bv is None or av is None:
        return None
    bi = {s: i for i, s in enumerate(base['runs']['seeds'])}
    ai = {s: i for i, s in enumerate(arm['runs']['seeds'])}
    pairs = [(bv[bi[s]], av[ai[s]]) for s in bi if s in ai]
    return [(b, a) for b, a in pairs if b is not None and a is not None]


def verdict(d, se, higher_is_better: bool) -> str:
    """``better``, ``worse`` or ``within noise`` (within 2 standard errors; one pair or none is noise)."""
    if d is None:
        return 'not in both files'
    if se is None or abs(d) <= 2 * se:
        return 'within noise'
    return 'better' if (d > 0) == higher_is_better else 'worse'


def summary_lines(base: dict, arm: dict, min_pairs: int) -> list:
    """One line a measure in the owner's order: the rate, hearts, beams held, shots that hit and kills from range,
    time to the last step both files entered with ``min_pairs`` runs or more."""
    def line(label: str, field: str, higher: bool, digits: int = 2, scale: float = 1.0, units: tuple = ('', '')):
        p = paired(base, arm, field)
        if p is None:
            return f"- {label}: not in both files"
        return (f"- {label}: {fmt(p[1] * scale, digits)}{units[0]} -> {fmt(p[2] * scale, digits)}{units[0]}, "
                f"{pm(p[3], p[4], digits, scale)}{units[1]} ({p[0]} pairs): **{verdict(p[3], p[4], higher)}**")
    out = [line('rate', 'win', True, 1, 100, ('%', ' points')),
           line('hearts lost a run', 'hearts_lost', False),
           line('beams held, rooms cleared holding them', 'beam_rooms', True)]
    if paired(base, arm, 'beam_held') is not None:
        out.append(line('beams held, frames holding them', 'beam_held', True, 0))
    out += [line('shots that hit a run', 'shots_hit', True), line('kills from range a run', 'kills_far', True)]
    steps = sorted(set(base['runs'].get('enter') or {}) & set(arm['runs'].get('enter') or {}), key=int)
    last = next((s for s in reversed(steps) if len(time_pairs(base, arm, s) or ()) >= min_pairs), None)
    if last is None:
        out.append("- time: no step both files entered in enough runs")
    else:
        pairs = time_pairs(base, arm, last)
        d, se = mean_se([a - b for b, a in pairs])
        room = (base.get('by_step', {}).get(last) or {}).get('room', '?')
        out.append(f"- time to enter step {last} ({room}), the last both entered in {min_pairs}+ runs: "
                   f"{pm(d, se, 0)} frames ({d / FPS:+.1f} s, {len(pairs)} pairs): **{verdict(d, se, False)}**")
    return out


def compare(base: dict, arms: list, min_pairs: int = 20) -> str:
    """Arms against a base, paired by seed, in the owner's order (AGENTS.md "What decides, in order"): the rate,
    then hearts, beams held, shots that hit and kills from range, time to each step; one table a metric."""
    # pylint: disable=too-many-locals
    if any(s['kind'] == 'train' for s in (base, *arms)):
        raise SystemExit("triforce stats compare: a training has no runs to pair by seed; compare its evaluations "
                         "or chains (`triforce stats show` reads a training's file)")
    names = [base.get('id') or base.get('name')] + [a.get('id') or a.get('name') for a in arms]
    lines = [f"# {names[0]} against {', '.join(names[1:])}", '']
    for s, name in zip([base, *arms], names):
        for f in s.get('flags') or ():
            if f['level'] == 'red':
                lines.append(f"- **RED** {name} `{f['code']}`: {f['text']}")
    lines += ['', "## Summary (the owner's order; a difference beyond 2 standard errors is better or worse)", '']
    for arm, name in zip(arms, names[1:]):
        lines += [f"**{name}** against **{names[0]}**:", *summary_lines(base, arm, min_pairs), '']
    lines += ['', '## The rate (paired by seed)', '']
    rows = [[names[0], base['rate']['n'], base['rate']['successes'], f"{100 * base['rate']['rate']:.1f}%", '', '']]
    for arm, name in zip(arms, names[1:]):
        p = paired(base, arm, 'win')
        rows.append([name, arm['rate']['n'], arm['rate']['successes'], f"{100 * arm['rate']['rate']:.1f}%",
                     p[0] if p else 0, pm(p[3], p[4], 1, 100) if p else '-'])
    lines += table(['arm', 'runs', 'successes', 'rate', 'paired runs', 'difference (points)'], rows)

    def metric(title: str, field: str, digits: int = 2, keep=None, note: str = ''):
        out = ['', f"## {title}", ''] + ([note, ''] if note else [])
        rows = []
        for arm, name in zip(arms, names[1:]):
            p = paired(base, arm, field, keep(base, arm) if keep else None)
            if p:
                rows.append([name, p[0], fmt(p[1], digits), fmt(p[2], digits), pm(p[3], p[4], digits)])
            else:
                rows.append([name, 0, '-', '-', '- (not in both files)'])
        return out + table(['arm', 'pairs', 'base', 'arm', 'difference'], rows)

    both_won = lambda b, a: lambda i, j: b['runs']['result'][i] == 'w' and a['runs']['result'][j] == 'w'  # noqa
    lines += metric('Hearts lost a run', 'hearts_lost')
    lines += metric('Hearts at the end (runs won in both)', 'hearts_end', keep=both_won)
    lines += metric('Hits taken a run', 'hits')
    lines += metric('Aimless frames a run (round 38\'s calm stretches beyond the walk their ground needed)',
                    'aimless_frames', 0)
    lines += metric('Beams held: rooms cleared holding them', 'beam_rooms',
                    note='Rooms left still within half a heart of full before the first left without (any record).')
    lines += metric('Beams held: frames to the entry of the room that took them', 'beam_frames', 0)
    lines += metric('Beams held: frames holding them (round 40 counters)', 'beam_held', 0)
    lines += metric('Shots that hit a run (round 40 counters)', 'shots_hit')
    lines += metric('Kills from range a run', 'kills_far')
    lines += metric('Shots fired a run', 'shots')
    lines += metric('Kills where the route crosses a run (carriers aside; the drop cycle\'s count)',
                    'goal_crossed_kills')
    lines += metric('Expected screen wraps not taken a run', 'goal_wraps_missed')
    lines += metric('Hits in rooms with a planned path a run', 'goal_planned_hits')
    lines += metric("Carriers' bombs taken a run", 'goal_carrier_bombs')
    for arm, name in zip(arms, names[1:]):
        if base.get('goals') or arm.get('goals'):
            lines += ['', f"## Route goals by step ({names[0]} | {name}; not paired)", '']
            lines += goal_table(base.get('goals') or {}, arm.get('goals') or {})
    lines += ['', '## Shooting, all visits', '']
    rows = []
    for s, name in zip([base, *arms], names):
        g = s['all']
        shots = (g.get('shots') or {}).get('sum')
        hit = (g.get('shots_hit') or {}).get('sum')
        far, near = (g.get('kills_far') or {}).get('sum'), (g.get('kills_near') or {}).get('sum')
        rows.append([name, g['visits'], fmt(shots, 0), fmt(hit, 0),
                     f"{100 * hit / shots:.1f}%" if shots and hit is not None else '-', fmt(far, 0), fmt(near, 0),
                     f"{100 * far / (far + near):.1f}%" if far is not None and near is not None and far + near else '-'])
    lines += table(['file', 'visits', 'shots', 'shots that hit', 'hit share', 'far kills', 'near kills', 'far share'],
                   rows)
    for arm, name in zip(arms, names[1:]):
        lines += ['', f"## Time: frames from the start to ENTERING each step ({name}; runs that entered it in both, "
                      "paired by seed)", '']
        rows = []
        steps = sorted(set(base['runs'].get('enter') or {}) & set(arm['runs'].get('enter') or {}), key=int)
        for step in [*steps, 'end']:
            pairs = time_pairs(base, arm, step)
            if pairs is None or len(pairs) < min_pairs:
                continue        # an evaluation has no route time
            d, se = mean_se([a - b for b, a in pairs])
            where = 'the end' if step == 'end' else (base.get('by_step', {}).get(step) or {}).get('room', '?')
            family = '' if step == 'end' else (base.get('by_step', {}).get(step) or {}).get('family', '')
            rows.append([step, f"{where} {family}".strip(), len(pairs), fmt(sum(b for b, _ in pairs) / len(pairs), 0),
                         fmt(sum(a for _, a in pairs) / len(pairs), 0), pm(d, se, 0), f"{d / FPS:+.1f}"])
        lines += table(['step', 'room (family)', 'runs both', 'base', 'arm', 'difference', 'seconds'], rows)
    return '\n'.join(lines)


def parse_since(text: Optional[str]) -> Optional[float]:
    """``24h``, ``30m``, ``2d`` -> seconds."""
    if not text:
        return None
    m = re.fullmatch(r'(\d+(?:\.\d+)?)([smhd])', text.strip())
    if not m:
        raise SystemExit(f"triforce stats: --since {text}: write e.g. 30m, 24h, 2d")
    return float(m.group(1)) * {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}[m.group(2)]


def list_text(rows: list) -> str:
    lines = table(['id', 'written', 'agent', 'task', 'what', 'rate', 'red', 'notes'], [
        [s.get('id'), (s.get('written') or '')[:16], (s.get('job') or {}).get('agent'),
         (s.get('job') or {}).get('task'), s.get('description') or s['kind'],
         f"{100 * s['rate']['rate']:.1f}%" if (s.get('rate') or {}).get('rate') is not None else '-',
         sum(f['level'] == 'red' for f in s.get('flags') or ()), sum(f['level'] == 'note' for f in s.get('flags') or ())]
        for s in rows])
    return '\n'.join(lines)


def ingest(paths: list, agent: Optional[str], task: Optional[str]) -> list:
    """Computes the statistics of existing outputs (a chain's jsonl, an evaluation's json, a run dir) and registers
    them; writes nothing beside them (other agents' trees are read only).  A chain's ``--checkpoint`` overrides,
    and its owner when ``agent`` / ``task`` are not given, come from the queue job that wrote it (``source_job``)."""
    out = []
    for ref in paths:
        path = Path(ref)
        owner = (agent, task)
        if path.is_dir():
            stats = compute_train(path)
        elif path.suffix == '.jsonl':
            job = source_job(path) or {}
            stats = compute_chain(path, checkpoint_overrides(job.get('command'), path))
            stats['provenance']['source_job'] = {k: job.get(k) for k in ('id', 'agent', 'task', 'subtask',
                                                                          'description')} if job else None
            owner = (agent or job.get('agent'), task or job.get('task'))
        else:
            result = json.loads(path.read_text(encoding='utf-8'))
            stats = eval_stats(result, str(path.resolve()), read_json(f'{path}.meta.json'))
            stats['name'] = eval_name(result, path)
        register(stats, *owner)
        out.append(stats)
    return out


def main(argv=None) -> int:
    """``triforce stats list|show|compare|ingest``."""
    parser = argparse.ArgumentParser(prog='triforce stats', description=__doc__.split('\n\n', maxsplit=1)[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('list', help="the registered statistics files")
    p.add_argument('--agent')
    p.add_argument('--task')
    p.add_argument('--since', help="e.g. 24h, 30m, 2d")
    p = sub.add_parser('show', help="one file's tables (markdown)")
    p.add_argument('ref', metavar='ID|PATH')
    p.add_argument('--rooms', type=int, default=20, help="rooms shown, those with the most deaths first")
    p.add_argument('--family', help="only this family's rooms")
    p = sub.add_parser('compare', help="arms against a base, paired by seed, in the owner's order (markdown)")
    p.add_argument('base', metavar='BASE')
    p.add_argument('arms', nargs='+', metavar='ARM')
    p.add_argument('--min-pairs', type=int, default=20, help="steps with fewer runs entered in both are left out")
    p = sub.add_parser('ingest', help="compute and register the statistics of existing outputs (read only)")
    p.add_argument('paths', nargs='+', metavar='FILE')
    p.add_argument('--agent', help="the data's owner (default the job's agent)")
    p.add_argument('--task', help="the data's task (default the job's task)")
    args = parser.parse_args(argv)
    if args.command == 'list':
        since = parse_since(args.since)
        rows = [s for s in stored()
                if (args.agent is None or (s.get('job') or {}).get('agent') == args.agent)
                and (args.task is None or (s.get('job') or {}).get('task') == args.task)
                and (since is None or _age(s) <= since)]
        print(list_text(rows))
    elif args.command == 'show':
        print(show(load(args.ref), args.rooms, args.family))
    elif args.command == 'compare':
        print(compare(load(args.base), [load(a) for a in args.arms], args.min_pairs))
    else:
        from .jobs import require_job  # pylint: disable=import-outside-toplevel
        require_job()
        started = time.monotonic()
        for stats in ingest(args.paths, args.agent, args.task):
            red = sum(f['level'] == 'red' for f in stats['flags'])
            print(f"{stats['id']}: {stats['kind']} {stats.get('source')} ({red} red flag(s)) -> "
                  f"{store_path() / (stats['id'] + SUFFIX)}")
        print(f"({time.monotonic() - started:.1f} s)")
    return 0


def _age(stats: dict) -> float:
    try:
        written = time.mktime(time.strptime(stats['written'][:19], '%Y-%m-%dT%H:%M:%S'))
    except (KeyError, ValueError):
        return float('inf')
    return time.time() - written


if __name__ == '__main__':
    sys.exit(main())
