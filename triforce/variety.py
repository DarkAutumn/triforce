"""Risk and variety of play from the per-visit decision records (round 23).

Every evaluation room visit (``eval.json`` ``records[i]['visits'][j]['decisions']``) and every chain room record
(``RoomRecord.decisions``) carries ``decisions`` (``crates/zelda/src/decisions.rs``): ``n`` decisions, ``low1`` /
``low5`` (actions chosen with probability under 1% / 5% under the unmodified policy, whatever the act mode),
``low1_hit`` / ``low5_hit`` / ``low1_death`` / ``low5_death`` (those followed by a hit, a death, within 4
decisions), ``surprisal`` (sum of -ln p), ``act_entropy`` (sum of the entropies of the distributions the actions were
drawn from: 0 for argmax) and ``actions`` (two hex digits per decision, the joint action).  Records written before
round 23 have none; every function here skips them.

``risk`` sums the counts of a group of visits (or runs) and reports them per unit; ``variety`` compares the action
sequences and frame counts of a group of plays from the same start: ``distinct`` sequences, ``distinct_rate``
(distinct / plays), ``diverge_all`` (the first decision at which not every play agrees: the common prefix of all),
``diverge_mean`` (the first differing decision of two plays, averaged over every pair), and the spread of frames
(``frames_mean``, ``frames_sd``, ``frames_p10``, ``frames_p90``).
"""

import math
from os.path import commonprefix
from typing import Iterable, Optional

# The per-visit decision means an evaluation's ``visited_rooms`` keep (``risk`` per visit), merged by room-stats.
VISIT_RISK_KEYS = ('decisions_per_visit', 'low1_per_visit', 'low5_per_visit', 'low1_hit_per_visit',
                   'low5_hit_per_visit', 'low1_death_per_visit', 'low5_death_per_visit', 'act_entropy_per_decision')
COUNTS = ('n', 'low1', 'low5', 'low1_hit', 'low5_hit', 'low1_death', 'low5_death')
SUMS = ('surprisal', 'act_entropy')


def decisions_of(record: dict) -> Optional[dict]:
    """A visit's or room record's ``decisions`` (None when it has none: an older record, or no decision taken)."""
    decisions = record.get('decisions')
    return decisions if decisions and decisions.get('n') is not None else None


def risk(groups: Iterable[Iterable[dict]], unit: str) -> dict:
    """The decision counts of ``groups`` (each one unit: an episode, a run, a visit; a group is the ``decisions``
    dicts of its visits) summed and per unit: ``<unit>s`` counted (units with decision records), ``decisions``,
    ``low1`` .. ``low5_death`` totals, ``<count>_per_<unit>``, ``low1_hit_rate`` / ``low5_hit_rate`` /
    ``low1_death_rate`` / ``low5_death_rate`` (share of the low decisions followed by a hit or death),
    ``act_entropy`` and ``surprisal`` per decision.  {} when no unit has records."""
    totals = dict.fromkeys(COUNTS + SUMS, 0.0)
    units = 0
    for group in groups:
        seen = False
        for decisions in group:
            if decisions is None:
                continue
            seen = True
            for key in COUNTS + SUMS:
                totals[key] += decisions.get(key) or 0
        units += seen
    if not units:
        return {}
    n = totals['n']
    result = {f'{unit}s': units, 'decisions': int(n)}
    for key in COUNTS[1:]:
        result[key] = int(totals[key])
        result[f'{key}_per_{unit}'] = totals[key] / units
    for low in ('low1', 'low5'):
        for what in ('hit', 'death'):
            result[f'{low}_{what}_rate'] = totals[f'{low}_{what}'] / totals[low] if totals[low] else None
    result['decisions_per_' + unit] = n / units
    result['act_entropy_per_decision'] = totals['act_entropy'] / n if n else None
    result['surprisal_per_decision'] = totals['surprisal'] / n if n else None
    return result


def _decisions(hexes: str) -> int:
    return len(hexes) // 2


def mean_pair_prefix(sequences: list[str]) -> Optional[float]:
    """The common prefix (in decisions) of two of ``sequences`` averaged over every pair: with the sequences sorted,
    a pair's common prefix is the smallest of the adjacent ones between them, so the sum over pairs is the sum of
    the minima of every run of adjacent prefixes (a monotonic stack, linear)."""
    count = len(sequences)
    if count < 2:
        return None
    ordered = sorted(sequences)
    adjacent = [_decisions(commonprefix([a, b])) for a, b in zip(ordered, ordered[1:])]
    total, stack, running = 0, [], 0          # stack of (value, how many runs ending here it is the minimum of)
    for value in adjacent:
        width = 1
        while stack and stack[-1][0] >= value:
            top, top_width = stack.pop()
            running -= top * top_width
            width += top_width
        stack.append((value, width))
        running += value * width
        total += running
    return total / (count * (count - 1) / 2)


def _quantile(ordered: list[float], q: float) -> float:
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def variety(sequences: list[Optional[str]], frames: list[float]) -> dict:
    """How differently ``len(frames)`` plays from one start went: their action sequences (hex, None where a play
    has no records) and frame counts.  See the module doc for the fields."""
    result = {}
    known = [s for s in sequences if s is not None]
    if len(known) >= 2:
        result.update(distinct=len(set(known)), distinct_rate=len(set(known)) / len(known),
                      diverge_all=_decisions(commonprefix(known)), diverge_mean=mean_pair_prefix(known))
    if frames:
        ordered = sorted(float(f) for f in frames)
        mean = sum(ordered) / len(ordered)
        result.update(frames_mean=mean,
                      frames_sd=math.sqrt(sum((f - mean) ** 2 for f in ordered) / (len(ordered) - 1))
                      if len(ordered) > 1 else 0.0,
                      frames_p10=_quantile(ordered, 0.1), frames_p90=_quantile(ordered, 0.9))
    return result


def sequence(visits: Iterable[dict]) -> Optional[str]:
    """The actions of consecutive visits as one sequence (None when any visit has no records)."""
    parts = []
    for visit in visits:
        decisions = decisions_of(visit)
        if decisions is None:
            return None
        parts.append(decisions.get('actions') or '')
    return ''.join(parts)
