"""Round 36, "One network by imitation": a student's bench results against the 51 teachers, slot by slot.

    python scripts/experiments/r36_merge_report.py --student-early S.json --student-late L.json --out R.json

A *slot* is one of the 51 teachers on the rooms it plays on the route:

- a family without a route part (17): the bench's entry of the family, its teacher the family's run
  (``triforce/lab/reference.json``);
- a route part ``family@d3first`` (17): the bench's entry of the family (harvested ``d3first_*`` states, the
  wood-sword rooms), its teacher the part's run (``reference.json``'s ``family@d3first`` row);
- the default run of a paired family (17): the rooms it plays after the magic sword (``r36_merge_pairs.py``'s
  manifest, ``late17-team.json``), its teacher the default run.

The student plays the family keys only (one network: a family's two bench keys hold the same states and seeds,
so they would play the same episodes).  Per slot: the paired differences (student - teacher) with standard errors
(``lab.bench.paired``) and the share of the teacher's success kept; overall: the mean over slots, and pooled."""

import argparse
import json
import math
import sys
from pathlib import Path

from triforce.lab.bench import paired, records, summarize

ROUTE = 'd3first'


def slots(reference: dict, late_team: dict) -> list[tuple[str, str, str, dict]]:
    """(slot name, family key the student plays, which student file, teacher row)."""
    out = []
    keys = reference['keys']
    families = sorted({k.split('@')[0] for k in keys})
    for family in families:
        part = f'{family}@{ROUTE}'
        if part in keys:
            out.append((part, family, 'early', keys[part]))
            if family in late_team['keys']:
                out.append((family, family, 'late', late_team['keys'][family]))
        else:
            out.append((family, family, 'early', keys[family]))
    return out


def report(reference: dict, late_team: dict, early: dict, late: dict) -> dict:
    rows = {}
    pooled_teacher, pooled_student = [], []
    for slot, key, which, teacher in slots(reference, late_team):
        student = (early if which == 'early' else late)['keys'].get(key)
        if student is None:
            continue
        a, b = records(teacher), records(student)
        diff = paired(a, b)
        index = {(e['state'], e['seed']) for e in b}
        ta = [e for e in a if (e['state'], e['seed']) in index]
        t, s = summarize(ta), summarize(b)
        rows[slot] = {'rooms': which, 'episodes': diff['pairs'],
                      'teacher': {m: t[m] for m in ('success', 'hearts_lost', 'hits_per_visit', 'decisions')},
                      'student': {m: s[m] for m in ('success', 'hearts_lost', 'hits_per_visit', 'decisions')},
                      'diff': {m: diff[m] for m in ('success', 'hearts_lost', 'hits', 'decisions')},
                      'kept': s['success'] / t['success'] if t['success'] > 0 else None}
        pooled_teacher += ta
        pooled_student += b
    n = len(rows)
    mean_t = sum(r['teacher']['success'] for r in rows.values()) / n
    mean_s = sum(r['student']['success'] for r in rows.values()) / n
    by_drop = sorted(rows, key=lambda k: rows[k]['diff']['success'][0])
    worst = by_drop[:10]
    w_t = sum(rows[k]['teacher']['success'] for k in worst) / len(worst)
    w_s = sum(rows[k]['student']['success'] for k in worst) / len(worst)
    diffs = [r['diff']['success'][0] for r in rows.values()]
    ses = [r['diff']['success'][1] for r in rows.values()]
    return {
        'slots': n,
        'mean_over_slots': {'teacher_success': mean_t, 'student_success': mean_s, 'kept': mean_s / mean_t,
                            'diff': sum(diffs) / n, 'diff_se': math.sqrt(sum(x * x for x in ses)) / n,
                            'teacher_hearts': sum(r['teacher']['hearts_lost'] for r in rows.values()) / n,
                            'student_hearts': sum(r['student']['hearts_lost'] for r in rows.values()) / n,
                            'teacher_hits_per_visit': sum(r['teacher']['hits_per_visit'] for r in rows.values()) / n,
                            'student_hits_per_visit': sum(r['student']['hits_per_visit'] for r in rows.values()) / n},
        'pooled': {'teacher': summarize(pooled_teacher), 'student': summarize(pooled_student),
                   'diff': paired(pooled_teacher, pooled_student)},
        'worst10': {'slots': worst, 'teacher_success': w_t, 'student_success': w_s, 'kept': w_s / w_t if w_t else None},
        'rows': rows}


def table(result: dict) -> str:
    lines = ['| slot | rooms | n | teacher success | student success | diff | hearts diff | hits diff |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for slot, r in sorted(result['rows'].items(), key=lambda kv: kv[1]['diff']['success'][0]):
        d = r['diff']
        lines.append(f"| {slot} | {r['rooms']} | {r['episodes']} | {r['teacher']['success']:.3f} | "
                     f"{r['student']['success']:.3f} | {d['success'][0]:+.3f} ± {d['success'][1]:.3f} | "
                     f"{d['hearts_lost'][0]:+.2f} ± {d['hearts_lost'][1]:.2f} | {d['hits'][0]:+.2f} ± {d['hits'][1]:.2f} |")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--reference', default='triforce/lab/reference.json')
    parser.add_argument('--late-team', default='target/merge/bench/late17-team.json')
    parser.add_argument('--student-early', required=True)
    parser.add_argument('--student-late', required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--table', action='store_true')
    args = parser.parse_args(argv)
    load = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    result = report(load(args.reference), load(args.late_team), load(args.student_early), load(args.student_late))
    summary = {k: v for k, v in result.items() if k != 'rows'}
    print(json.dumps(summary, indent=1))
    if args.table:
        print(table(result))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
