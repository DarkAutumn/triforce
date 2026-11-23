"""What a heart is worth at every route step: V(step, hearts), fitted from chain runs, for the reward's heart terms.

    python -m triforce.heart_value CHAIN.jsonl... [--route ROUTE] [--out triforce/heart_value.json] [--measured-on REV]

A fit is a job of the machine's queue (``jobs.require_job``; it reads every chain file given): ``triforce job run
--agent A --task T --subtask S --desc '...' --timeout 20m -- python -m triforce.heart_value ...``; the chain files
read of the total are its progress.

V(k, h) is the chance that a run entering route step k's room with h hearts is alive when it takes the next
triforce (which refills him): 0 at no hearts, non-decreasing in h, flat where nothing downstream can kill him before
the refill.  A specialist whose ``reward.heart_value`` turns a term on prices hearts by it (``env::heart_value``):
the success terminal's health part by V of the step Link leaves into, and every change of hearts in a room by the
change of V at the room's step.

The fit, per way of the route (``clips`` from the visits with the run-time clips switch on, ``plain`` from those
with it off; a step on one way only has no table on the other), backwards from each triforce:

- Each step's visit is a transition of hearts: in at ``health_in``, out at ``health_out`` or dead.  Per half-heart
  arrival bin the empirical transitions are mixed with ``PRIOR_VISITS`` pseudo-visits of a prior that assumes the
  damage a room deals does not depend on the hearts Link brings: a Kaplan-Meier estimate of the damage (half
  hearts, a gain negative) over every visit of the step, a death at h censoring it at "at least h".  Rare
  arrivals (a room usually entered at full health) follow the prior; common ones their own data.
- V(k, h) = the expected V(next step, hearts out) over that transition (V = 1 after a triforce step, and after the
  route's last step: round 29's ending), then made
  non-decreasing in h (weighted pool-adjacent-violators).

Refit after every round that changes the team or the route (the env refuses a table whose steps no longer match the
route, but only for families that turn a term on).  Chains measured from a segment start enter its first room at
the start state's hearts only: give the fit the full route's runs as well as the segment runs.  Fit on the longest
route: a table prices its own route and every route whose steps are its first ones (``HeartValues::prices``; round
27: fitted on the scouting route ``d6`` from round 24's chains and Dungeon 6's, it priced ``d2first`` as well, whose
rows it reproduced, since V at a step depends only on the steps up to the next triforce; at 1.7.0.0 ``d6`` merged
into ``d2first`` and the table refit on it from both lines' sources is the two tables' rows, identical; round 28: on
``d8``, from those, Dungeon 7's and Dungeon 8's chains, pricing ``d2first`` and ``d8``; round 29: on ``d9``, from
those and the Dungeon 9 team's chains, rows 0-296 unchanged, pricing all three; its third round refitted level 9
from the second team's ``post-d8`` and ``d9`` chains, which give the leg's steps their first visits; at 1.9.0.0
``d8`` and ``d9`` merged into ``d2first``, the table names ``d2first``, and the refit on it from the same 30 sources is
byte-identical).

Chains measured on an older form of the route (``--measured-on REV``: the git revision whose routes.yaml numbered
their steps) are read onto the route as it stands: the two step lists are aligned on (room, entry, exit, way), and a
visit of an old step that has no counterpart is left out (round 31, phase IX: Dungeon 1's key rooms and 9:0x65 left
the route, its new steps have no visits yet; a step whose rules changed but not its room and sides, 6:0x29 now taking
its key, keeps the visits of its old play until the new team's chains refit it).  A row whose step and every step
after it up to the next triforce have their old counterparts in the same order comes out as before.
"""

import argparse
import collections
import difflib
import json
import os
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple

from . import jobs
from .events import write_json_atomic

TABLE_PATH = os.path.join(os.path.dirname(__file__), 'heart_value.json')
# Pseudo-visits of the damage prior mixed into every arrival bin.
PRIOR_VISITS = 20.0
WAYS = ('clips', 'plain')


def room_key(step: dict) -> str:
    """'1:0x44', '0:0x77:cave' (``env::config::room_key``)."""
    return f"{step['level']}:0x{step['room']:02x}" + (':cave' if step['in_cave'] else '')


def _identity(step: dict) -> Tuple[str, str, str, Optional[str]]:
    """What aligns a step of an older route with one of the route as it stands."""
    return room_key(step), step['entry'], step['exit'], step['way']


def route_at(revision: str, route: str) -> List[dict]:
    """The steps of ``route`` as triforce/routes.yaml at git ``revision`` numbered them (each with its ``level``,
    ``room``, ``in_cave``, ``entry``, ``exit`` and ``way``; a ``stretch:`` block is its clip way's steps, then its
    plain way's, as ``zelda::game_map`` numbers them; an ``include:`` is its block's elements)."""
    import yaml  # pylint: disable=import-outside-toplevel
    text = subprocess.run(['git', 'show', f'{revision}:triforce/routes.yaml'], check=True, capture_output=True,
                          text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).stdout
    routes = yaml.safe_load(text)

    def elements(items: List[dict]) -> List[dict]:
        steps = []
        for item in items:
            if 'include' in item:
                steps += elements(routes['blocks'][item['include']])
            elif 'stretch' in item:
                steps += [dict(s, way=way) for way in ('clips', 'plain') for s in item[way]]
            else:
                steps.append(dict(item, way=None))
        return steps

    def flat(name: str) -> List[dict]:
        body = routes[name]
        if isinstance(body, list):
            return elements(body)
        return (flat(body['extends']) if 'extends' in body else []) + elements(body['steps'])

    return [{'level': s['level'], 'room': int(str(s['room']), 16), 'in_cave': bool(s.get('in_cave')),
             'entry': s['entry'], 'exit': s['exit'], 'way': s['way']} for s in flat(route)]


def step_map(old: List[dict], new: List[dict]) -> Dict[int, int]:
    """Old step index -> the index of the same step (room, entry, exit, way) on ``new``, by the longest common
    subsequence of the two lists; an old step with no counterpart is left out."""
    matcher = difflib.SequenceMatcher(None, [_identity(s) for s in old], [_identity(s) for s in new], autojunk=False)
    return {a + i: b + i for a, b, size in matcher.get_matching_blocks() for i in range(size)}


def _damage_prior(visits: List[dict]):
    """Kaplan-Meier pmf of the damage a visit takes (half hearts; negative = healed) and the mass beyond the last
    observed survivor; a death at h is censored at "at least 2h"."""
    exact = [int(round(2 * (v['health_in'] - v['health_out']))) for v in visits if v['outcome'] != 'death']
    censored = [max(1, int(round(2 * v['health_in']))) for v in visits if v['outcome'] == 'death']
    if not visits:
        return {0: 1.0}, 0.0
    survive, pmf = 1.0, {}
    for x in sorted(set(exact)):
        at_risk = sum(1 for e in exact if e >= x) + sum(1 for c in censored if c > x)
        if at_risk == 0:
            break
        p = survive * sum(1 for e in exact if e == x) / at_risk
        pmf[x] = p
        survive -= p
    return pmf, max(survive, 0.0)


def _isotonic(values: List[float], weights: List[float]) -> List[float]:
    """Weighted pool-adjacent-violators: the closest non-decreasing sequence."""
    blocks: List[List[float]] = []
    for value, weight in zip(values, weights):
        blocks.append([value, weight, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            total = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / total if total > 0 else (v1 + v2) / 2, total, c1 + c2])
    out: List[float] = []
    for value, _, count in blocks:
        out.extend([value] * int(count))
    return out


def _on_way(step: dict, way: str) -> bool:
    return step['way'] is None or step['way'] == way


def fit_way(steps: List[dict], visits: Dict[int, List[dict]], way: str,
             max_hearts_seen: Optional[Dict[int, int]] = None) -> Dict[int, dict]:
    """step index -> {max_hearts, values (V at 0, 0.5, ... max hearts), visits} for the steps on ``way``.  A step
    with no visits on this way takes its max hearts from ``max_hearts_seen`` (either way's visits), else the step
    before's.  The steps up to a triforce none of which has a visit on this way (round 26: a stretch of the route
    no chain has played yet) get no curve at all rather than the prior's "nothing here costs a heart": a family
    that prices hearts by the table then falls back to the plain health terms there (``env::heart_value``)."""
    on_way = [k for k, s in enumerate(steps) if _on_way(s, way)]
    max_hearts: Dict[int, int] = {}
    last = 3
    for k in on_way:
        counts = collections.Counter(v['max_health'] for v in visits.get(k, []))
        if counts:
            last = counts.most_common(1)[0][0]
        elif max_hearts_seen and k in max_hearts_seen:
            last = max_hearts_seen[k]
        max_hearts[k] = last
    table: Dict[int, dict] = {}
    following: Optional[List[float]] = None
    played = False
    for k in reversed(on_way):
        if steps[k]['exit'] == 'triforce' or k == on_way[-1]:
            following = None
            stretch = [j for j in on_way if j <= k]
            previous = next((j for j in reversed(stretch[:-1]) if steps[j]['exit'] == 'triforce'), -1)
            played = any(visits.get(j) for j in stretch if j > previous)
        if not played:
            continue

        def after(half_hearts: int, nxt=following) -> float:
            if nxt is None:
                return 1.0
            return nxt[min(max(half_hearts, 0), len(nxt) - 1)]

        top = 2 * max_hearts[k]
        here = visits.get(k, [])
        pmf, _ = _damage_prior(here)
        bins: Dict[int, List[dict]] = collections.defaultdict(list)
        for v in here:
            bins[min(max(int(round(2 * v['health_in'])), 1), top)].append(v)
        values, weights = [], []
        for i in range(1, top + 1):
            prior = sum(p * after(i - x) for x, p in pmf.items() if x < i)
            seen = bins.get(i, [])
            observed = sum(after(int(round(2 * v['health_out']))) for v in seen if v['outcome'] != 'death')
            values.append((observed + PRIOR_VISITS * prior) / (len(seen) + PRIOR_VISITS))
            weights.append(len(seen) + PRIOR_VISITS)
        curve = [0.0] + _isotonic(values, weights)
        table[k] = {'max_hearts': max_hearts[k], 'values': curve, 'visits': len(here)}
        following = curve
    return table


def read_visits(paths: Iterable[str], steps_of: Optional[Dict[int, int]] = None) -> Dict[str, Dict[int, List[dict]]]:
    """way -> route step -> room records of the chain jsonl files (by each visit's run-time clips switch).  With
    ``steps_of`` (``step_map``: the chains' step -> the route's) each record is filed under its step on the route as
    it stands, and a record of a step that is no longer on it is left out."""
    by_way: Dict[str, Dict[int, List[dict]]] = {way: collections.defaultdict(list) for way in WAYS}
    paths = list(paths)
    for done, path in enumerate(paths, 1):
        with open(path, encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                for room in json.loads(line)['rooms']:
                    if room.get('step') is None:
                        continue
                    if 'clips_allowed' not in room:
                        raise ValueError(f"{path}: room records without clips_allowed (round 22 on); refit from newer chains")
                    step = room['step'] if steps_of is None else steps_of.get(room['step'])
                    if step is None:
                        continue
                    way = 'clips' if room['clips_allowed'] else 'plain'
                    by_way[way][step].append(room)
        jobs.progress(done, len(paths), 'files')
    return by_way


def build_table(steps: List[dict], paths: List[str], route: str, measured_on: Optional[List[dict]] = None) -> dict:
    """The table ``env::heart_value`` reads: per route step its identity (checked against the route) and V per way.
    ``measured_on``: the steps of the route the chains were measured on, when it is not ``steps`` (``route_at``)."""
    visits = read_visits(paths, None if measured_on is None else step_map(measured_on, steps))
    seen: Dict[int, int] = {}
    for k in range(len(steps)):
        counts = collections.Counter(v['max_health'] for way in WAYS for v in visits[way].get(k, []))
        if counts:
            seen[k] = counts.most_common(1)[0][0]
    fitted = {way: fit_way(steps, visits[way], way, seen) for way in WAYS}
    rows = []
    for k, step in enumerate(steps):
        row = {'step': k, 'room': room_key(step), 'entry': step['entry'], 'exit': step['exit'], 'visit': step['visit'],
               'way': step['way'], 'specialist': step['specialist']}
        for way in WAYS:
            entry = fitted[way].get(k)
            row[way] = None if entry is None else {
                'max_hearts': entry['max_hearts'], 'visits': entry['visits'],
                'values': [round(v, 4) for v in entry['values']]}
        rows.append(row)
    return {'route': route, 'prior_visits': PRIOR_VISITS, 'sources': [os.path.basename(p) for p in paths],
            'steps': rows}


def main(argv=None):
    """Command line."""
    parser = argparse.ArgumentParser(prog='python -m triforce.heart_value', description=__doc__.split('\n\n')[0])
    parser.add_argument('chains', nargs='+', help="chain run jsonl files (clips on and off; the full route too)")
    parser.add_argument('--route', default='d2first')
    parser.add_argument('--out', default=None,
                        help="default triforce/heart_value.json; a route with a start-state namespace NS (phase X) "
                             "triforce/heart_value.NS.json, the table its states' trainings read")
    parser.add_argument('--measured-on', metavar='REV',
                        help="the git revision whose routes.yaml numbered the chains' steps (default: the route as it is)")
    args = parser.parse_args(argv)
    jobs.require_job()
    from . import _native
    namespace = _native.route_namespace(args.route)
    if args.out is None:
        args.out = TABLE_PATH if namespace is None else f'{os.path.splitext(TABLE_PATH)[0]}.{namespace}.json'
    steps = _native.route_steps(args.route)
    measured_on = None if args.measured_on is None else route_at(args.measured_on, args.route)
    table = build_table(steps, args.chains, args.route, measured_on)
    write_json_atomic(args.out, table, indent=1)
    for row in table['steps']:
        curves = []
        for way in WAYS:
            entry = row[way]
            if entry is not None:
                values = entry['values']
                curves.append(f"{way} n={entry['visits']:5} " + ' '.join(f"{v:.2f}" for v in values[2::2]))
        print(f"{row['step']:3} {row['room']:12} {row['entry']:>5}->{row['exit']:8} {row['specialist']:18} "
              + ' | '.join(curves))
    return 0


if __name__ == '__main__':
    main()
