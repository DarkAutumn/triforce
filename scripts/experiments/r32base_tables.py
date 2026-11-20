"""Round 32's d3first baseline tables (docs/experiments/round32.md): the chain segments (and the full run to Level 5's
entrance), every route step's room in the chains, the d2first chains of the same rooms (the white sword), the
single-room evaluations against the same states on d2first, and the golden path's segments.

    .venv/bin/python scripts/experiments/r32base_tables.py [--part first|rest] [--tag TAG]
        [--chains-dir target/r56-chains] [--d2first-chains <main>/target/r56-chains/r29land-1k]
        [--d2first-steps FILE] [--rooms DIR] [--models DIR] [--golden]

`--part first`: steps 0-97 (tag r32base); `rest`: Level 5's entrance to the ending (tag r32base2).
`--d2first-steps`: d2first's route steps as the d2first chains numbered them (a JSON list of
`_native.route_steps('d2first')` from the checkout that ran them: their `step` fields index it).  `--golden`: only
the golden path's table (every triforce-to-triforce segment, both parts' chains in `--chains-dir`).  Prints markdown.
"""

import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from triforce import _native

ROUTE = 'd3first'
PARTS = {
    'first': {
        'tag': 'r32base',
        'segments': (('full', 'start', 0, 97), ('d3', '3_7cs_0', 6, 24), ('post-d3', '0_74c_0', 24, 31),
                     ('d4', '4_71s_0', 31, 53), ('post-d4', '0_45c_0', 53, 61), ('d1', '1_73s_0', 61, 79),
                     ('post-d1', '0_37c_v2_0', 79, 97)),
        'planned': ('d3', 'd4', 'full'),
        'milestones': ((6, "Level 3's entrance (3:0x7c)"), (23, "Dungeon 3's triforce room (3:0x3d)"),
                       (24, 'out of Dungeon 3 (0:0x74)'), (31, "Level 4's entrance (4:0x71)"),
                       (52, "Dungeon 4's triforce room (4:0x03)"), (53, 'out of Dungeon 4 (0:0x45)'),
                       (61, "Level 1's entrance (1:0x73)"), (78, "Dungeon 1's triforce room (1:0x36)"),
                       (79, 'out of Dungeon 1 (0:0x37)'), (97, "Level 5's entrance (5:0x76)")),
        'rooms': (0, 97),
    },
    'rest': {
        'tag': 'r32base2',
        'segments': (('d5', '5_76s_0', 97, 120), ('post-d5', '0_0bc_0', 120, 145), ('d2', '2_7ds_0', 145, 155),
                     ('post-d2', '0_3cc_0', 155, 174), ('d7', '7_79s_0', 174, 194), ('post-d7', '0_42c_0', 194, 211),
                     ('d6', '6_79s_0', 211, 227), ('post-d6', '0_22c_0', 227, 233), ('d8', '8_7es_0', 233, 243),
                     ('post-d8', '0_6dc_0', 243, 258), ('d9', '9_76s_0', 258, None)),
        'planned': ('d5', 'd2', 'd7', 'd6', 'd8', 'd9'),
        'milestones': (),
        'rooms': (97, 304),
    },
}
# The golden path's unit: from one triforce to the next, legs included (tag, leg segment, tag, dungeon segment).
# The first: the full run's steps 0-5 are its leg (Level 3's entrance at step 6).
GOLDEN = (("the start → Dungeon 3", 'r32base', 'full', 'r32base', 'd3'),
          ('Dungeon 3 → Dungeon 4', 'r32base', 'post-d3', 'r32base', 'd4'),
          ('Dungeon 4 → Dungeon 1', 'r32base', 'post-d4', 'r32base', 'd1'),
          ('Dungeon 1 → Dungeon 5', 'r32base', 'post-d1', 'r32base2', 'd5'),
          ('Dungeon 5 → Dungeon 2', 'r32base2', 'post-d5', 'r32base2', 'd2'),
          ('Dungeon 2 → Dungeon 7', 'r32base2', 'post-d2', 'r32base2', 'd7'),
          ('Dungeon 7 → Dungeon 6', 'r32base2', 'post-d7', 'r32base2', 'd6'),
          ('Dungeon 6 → Dungeon 8', 'r32base2', 'post-d6', 'r32base2', 'd8'),
          ('Dungeon 8 → the ending', 'r32base2', 'post-d8', 'r32base2', 'd9'))
SEGMENTS = PARTS['first']['segments']
D2_SEGMENT_FILES = ('d2', 'post-d2', 'd1', 'post-d1', 'd3', 'post-d3', 'd4', 'post-d4', 'd5', 'post-d5', 'd7',
                    'post-d7', 'd6', 'post-d6', 'd8', 'post-d8', 'd9')


def load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def ci(k, n):
    """95% half-width of a rate in points."""
    if n == 0:
        return 0.0
    p = k / n
    return 196 * math.sqrt(p * (1 - p) / n)


def on_route(rooms):
    return [r for r in rooms if r.get('step') is not None and r['outcome'] != 'milestone']


def end_step(run):
    """The route step a failed run was lost in (its last on-route room) and how (death, stuck, ...)."""
    rooms = [r for r in run['rooms'] if r['outcome'] != 'milestone']
    last = rooms[-1] if rooms else None
    steps = [r['step'] for r in rooms if r.get('step') is not None]
    step = steps[-1] if steps else None
    how = last['outcome'] if last else run['reason']
    off = last is not None and last.get('step') is None
    return step, how, off


def max_step(run):
    steps = [r['step'] for r in run['rooms'] if r.get('step') is not None and r['outcome'] != 'milestone']
    return max(steps) if steps else -1


def step_name(step):
    s = STEPS[step]
    room = f"{s['level']}:0x{s['room']:02x}" + (' cave' if s['in_cave'] else '')
    return room, f"{s['entry']}→{s['exit']}", s['specialist']


def segment_table(runs_by_seg, label):
    print(f"\n### {label}\n")
    print('| Segment | From → stop | Runs | Arrived | Deaths | Stuck | Other ends | Hearts at the stop (of max) |')
    print('|---|---|---|---|---|---|---|---|')
    for name, start, first, stop in SEGMENTS:
        runs = runs_by_seg.get(name)
        if not runs:
            continue
        n = len(runs)
        ok = [r for r in runs if r['result'] == 'success']
        ends = Counter(end_step(r)[1] for r in runs if r['result'] != 'success')
        deaths = ends.pop('death', 0)
        stuck = ends.pop('stuck', 0)
        other = sum(ends.values())
        hearts = [r['rooms'][-1]['health_out'] for r in ok if r['rooms']]
        mx = [r['rooms'][-1]['max_health'] for r in ok if r['rooms']]
        h = f"{sum(hearts) / len(hearts):.2f} of {sum(mx) / len(mx):.1f}" if hearts else '-'
        print(f"| {name} | {start} (step {first}) → {'the ending' if stop is None else f'step {stop}'} | {n} | "
              f"**{100 * len(ok) / n:.1f}%** "
              f"± {ci(len(ok), n):.1f} | {deaths} | {stuck} | {other}"
              f"{' (' + ', '.join(f'{k} {v}' for k, v in ends.items()) + ')' if ends else ''} | {h} |")


def milestones_table(full_runs, plan_runs):
    print("\n### The full run from the game start: its milestones\n")
    print('| Reached | Models alone (1000) | Planner (300) |')
    print('|---|---|---|')
    for step, label in PART['milestones']:
        cells = []
        for runs in (full_runs, plan_runs):
            if not runs:
                cells.append('-')
                continue
            k = sum(1 for r in runs if r['result'] == 'success' or max_step(r) >= step)
            cells.append(f"{100 * k / len(runs):.1f}% ({k})")
        print(f"| step {step}: {label} | {cells[0]} | {cells[1]} |")


def room_stats(runs):
    """Per route step: visits, entering runs, runs lost there (by how), hearts in/lost, steps, bombs, cheats."""
    stats = defaultdict(lambda: defaultdict(float))
    for run in runs:
        seen = set()
        for r in on_route(run['rooms']):
            s = stats[r['step']]
            s['visits'] += 1
            if r['step'] not in seen:
                seen.add(r['step'])
                s['runs'] += 1
                s['hin'] += r['health_in']
                s['max'] += r['max_health']
            s['lost'] += r['health_in'] - r['health_out']
            s['steps'] += r['steps']
            s['bombs'] += r['bombs_used']
            s['wrong'] += r['outcome'] == 'exit-wrong'
            for k in ('cheated_bombs', 'cheated_keys', 'cheated_rupees'):
                s[k] += r[k]
        if run['result'] != 'success':
            step, how, off = end_step(run)
            if step is not None:
                stats[step]['end_' + ('offroute-' if off else '') + how] += 1
                stats[step]['ended'] += 1
    return stats


def ends_text(s):
    parts = [f"{k[4:]} {int(v)}" for k, v in sorted(s.items()) if k.startswith('end_')]
    return ', '.join(parts)


def d2first_index(d2steps):
    """(level, room, in_cave, entry, exit) -> d2first step numbers, and (level, room, in_cave) -> step numbers."""
    exact, room = defaultdict(list), defaultdict(list)
    for s in d2steps:
        exact[(s['level'], s['room'], s['in_cave'], s['entry'], s['exit'])].append(s['index'])
        room[(s['level'], s['room'], s['in_cave'])].append(s['index'])
    return exact, room


def merge(stats, steps):
    out = defaultdict(float)
    for step in steps:
        for k, v in stats.get(step, {}).items():
            out[k] += v
    return out


def d2_cell(stats, index, key):
    """A d2first column's cell for the step ``key`` (level, room, in_cave, entry, exit): lost per 1000 entering, hearts
    in, lost a visit over d2first's steps of that room and sides (`room` = by another entry; `new` = never played)."""
    exact, by_room = index
    steps, kind = exact.get(key), ''
    if not steps:
        steps = by_room.get(key[:3])
        kind = 'room ' if steps else ''
    s = merge(stats, steps) if steps else None
    if s and s['runs']:
        return (f"{kind}{1000 * s['ended'] / s['runs']:.0f}, {s['hin'] / s['runs']:.1f}/"
                f"{s['max'] / s['runs']:.0f}, {s['lost'] / s['visits']:.2f}")
    return 'new'


def trim_to_dungeon(run):
    """A run of a chain with no stop, cut where it first leaves the dungeon it started in (scored a success there)."""
    level = run['rooms'][0]['level'] if run['rooms'] else 0
    for i, r in enumerate(run['rooms']):
        if level and r['level'] == 0 and r['outcome'] != 'milestone':
            return dict(run, rooms=run['rooms'][:i], result='success')
    return run


def rooms_table(seg_stats, full_stats, d2_stats, d2steps, plan_stats, ix=None):
    """``ix``: (stats, steps) of d2first's phase IX chains (Dungeons 6, 8, 9 as d3first plays them), or None."""
    d2index = d2first_index(d2steps)
    ixindex = d2first_index(ix[1]) if ix else None
    first, last = PART['rooms']
    with_full = bool(PART['milestones'])
    print(f"\n### Every room of steps {first}-{last - 1} (models alone: its segment's 1000 runs; "
          f"{'the full run' + chr(39) + 's; ' if with_full else ''}d2first's)\n")
    print("Segment columns from the segment that holds the step (its start state's hearts)"
          f"{'; `full` from the game start (hearts carried)' if with_full else ''}. Hearts in: mean on entering / max. "
          "Lost: mean hearts lost a visit (a death loses all). Ends: runs lost in the room (death, stuck; `offroute-` "
          "after a wrong exit). Cheats: totals over the segment's runs (bombs/keys/rupees). d2first: r29land-1k's "
          "segments in the same room by the same entry and exit (`room` = the room by another entry; `new` = d2first "
          "never plays it): lost per 1000 entering, hearts in, lost a visit"
          f"{'; IX: the same from round 31' + chr(39) + 's d2first chains of Dungeons 6, 8, 9 after phase IX (r31-1k, v1.9.0.0 team, each cut where it leaves the dungeon)' if ix else ''}. "
          f"Planner: runs lost there of the planner's {'/'.join(PART['planned'])} runs that entered.\n")
    print('| Step | Room | In→out | Family | Entered | Lost (per 1000) | Ends | Hearts in | Lost a visit | Steps '
          '| Bombs | Wrong exits | Cheats b/k/r |' + (' Full: entered, lost (per 1000), hearts in |' if with_full else '')
          + ' d2first: lost/1000, hearts in, lost a visit |' + (' d2first IX |' if ix else '')
          + ' Planner: lost/entered |')
    print('|' + '---|' * (15 + with_full + bool(ix)))
    for step in range(first, last):
        s = seg_stats.get(step)
        room, io, fam = step_name(step)
        st = STEPS[step]
        key = (st['level'], st['room'], st['in_cave'], st['entry'], st['exit'])
        d2txt = d2_cell(d2_stats, d2index, key)
        if ix:
            d2txt += f" | {d2_cell(ix[0], ixindex, key) if st['level'] in (6, 8, 9) else ''}"
        f = full_stats.get(step)
        ftxt = (f"{int(f['runs'])}, {int(f['ended'])} ({1000 * f['ended'] / f['runs']:.0f}), "
                f"{f['hin'] / f['runs']:.1f}" if f and f['runs'] else '-')
        fcell = f" {ftxt} |" if with_full else ''
        p = plan_stats.get(step)
        ptxt = f"{int(p['ended'])}/{int(p['runs'])}" if p and p['runs'] else '-'
        if not s or not s['runs']:
            print(f"| {step} | {room} | {io} | {fam} | - | | | | | | | | |{fcell} {d2txt} | {ptxt} |")
            continue
        n, v = s['runs'], s['visits']
        print(f"| {step} | {room} | {io} | {fam} | {int(n)} | {int(s['ended'])} ({1000 * s['ended'] / n:.0f}) | "
              f"{ends_text(s)} | {s['hin'] / n:.2f}/{s['max'] / n:.0f} | {s['lost'] / v:.2f} | {s['steps'] / v:.0f} | "
              f"{s['bombs'] / v:.2f} | {int(s['wrong'])} | {int(s['cheated_bombs'])}/{int(s['cheated_keys'])}/"
              f"{int(s['cheated_rupees'])} |{fcell} {d2txt} | {ptxt} |")


def segment_bounds(tag, name):
    """(first step, stop step or None) of the segment ``name`` of the part whose tag is ``tag``."""
    part = next(p for p in PARTS.values() if p['tag'] == tag)
    return next((first, stop) for n, _, first, stop in part['segments'] if n == name)


def arrival_at(run, step):
    """Link's hearts on reaching route step ``step`` (a run stopped there: its last room's hearts out; else the first
    record of that step: its hearts in), or None if he never got there."""
    if run['result'] == 'success' and run['rooms'] and max_step(run) < step:
        return run['rooms'][-1]['health_out']
    for r in on_route(run['rooms']):
        if r['step'] == step:
            return r['health_in']
    return None


def golden_table(chains_dir):
    """Every triforce-to-triforce segment: the leg's arrivals at the dungeon's entrance (rate, hearts), the dungeon's
    rate from its start state (full hearts), and the chance one run wins both: the dungeon's rate scaled per arrival
    by V(entrance, hearts) / V(entrance, full) of a heart_value fit (deaths only) on the two chains."""
    from triforce.heart_value import build_table  # pylint: disable=import-outside-toplevel
    print("\n### The golden path's segments (triforce to triforce, legs included; models alone)\n")
    print("Leg: runs reaching the dungeon's entrance (of 1000) and their hearts there. Dungeon: its segment from its "
          "start state at full hearts. V: a heart_value fit on the leg's and the dungeon's chains (death only, a "
          "damage prior where no run entered at those hearts): the chance of living to the triforce from the "
          "entrance at the leg's mean hearts against at full. One run wins it: the mean over the leg's runs of "
          "the dungeon's rate × V(their hearts) / V(full) (0 for a leg run that never arrived); leg × dungeon: the "
          "same at full hearts.\n")
    print("| Segment | Leg arrives | Hearts there (of max) | Dungeon (full hearts) | V at those hearts / at full | "
          "One run wins it | Leg × dungeon |")
    print('|---|---|---|---|---|---|---|')
    d = Path(chains_dir)
    for title, ltag, leg, dtag, dun in GOLDEN:
        lpath, dpath = d / f"{ltag}-1k-{leg}.jsonl", d / f"{dtag}-1k-{dun}.jsonl"
        if not (lpath.is_file() and dpath.is_file()):
            print(f"| {title} | - | - | - | - | - | - |")
            continue
        legs, duns = load(lpath), load(dpath)
        entrance, _ = segment_bounds(dtag, dun)
        hearts = [h for h in (arrival_at(r, entrance) for r in legs) if h is not None]
        rate = sum(r['result'] == 'success' for r in duns) / len(duns)
        table = build_table(STEPS, [str(lpath), str(dpath)], ROUTE)
        row = table['steps'][entrance]['clips']
        if not hearts or row is None:
            print(f"| {title} | {len(hearts)} | - | {100 * rate:.1f}% | - | 0.0% | 0.0% |")
            continue
        values, top = row['values'], 2 * row['max_hearts']
        full = values[top]

        def v(h, values=values, top=top):
            return values[min(max(int(round(2 * h)), 0), top)]
        mean_h = sum(hearts) / len(hearts)
        ratio = (lambda h: v(h) / full) if full > 0 else (lambda h: 0.0)
        wins = sum(rate * ratio(h) for h in hearts) / len(legs)
        print(f"| {title} | {100 * len(hearts) / len(legs):.1f}% | {mean_h:.2f} of {row['max_hearts']} | "
              f"{100 * rate:.1f}% | {v(mean_h):.2f} / {full:.2f} | **{100 * wins:.1f}%** | "
              f"{100 * len(hearts) / len(legs) * rate:.1f}% |")


def eval_by_state(path, clips_on_only=True):
    """state -> (episodes, successes, mean health change) from an evaluation's records (clips on only)."""
    with open(path, encoding='utf-8') as f:
        e = json.load(f)
    out = defaultdict(lambda: [0, 0, 0.0, 0])
    for r in e['records']:
        if clips_on_only and r.get('clips_allowed') is False:
            continue
        o = out[r['start_state']]
        o[0] += 1
        o[1] += bool(r['success'])
        o[2] += r['health'] - r['start_health']
        o[3] += r['outcome'] == 'death' or r.get('ending') == 'death'
    return e, out


def singles_table(rooms_dir, models_dir):
    print("\n### Single rooms: every family's d3first states (100 episodes a state, clips on)\n")
    print("d2first: the same family's promoted run on the same state name on d2first (its own `eval.json`, the "
          "records with clips on), white sword and d2first's containers; `-` where d2first has no such state.\n")
    print('| Family | Run | d3first states | Success | Health change | Deaths | d2first same states: success, '
          'health change |')
    print('|---|---|---|---|---|---|---|')
    per_state = []
    for path in sorted(glob.glob(os.path.join(rooms_dir, '*.json'))):
        fam = Path(path).stem
        e, st = eval_by_state(path)
        run = Path(e['checkpoint']).name if not e['checkpoint'].endswith('.pt') else Path(e['checkpoint']).parent.name
        d2path = Path(models_dir) / fam / run / 'eval.json'
        d2 = eval_by_state(d2path)[1] if d2path.is_file() else {}
        n = sum(v[0] for v in st.values())
        k = sum(v[1] for v in st.values())
        hc = sum(v[2] for v in st.values()) / max(n, 1)
        dd = sum(v[3] for v in st.values())
        bare = {name: name[len('d3first_'):] for name in st}
        d2n = sum(d2[b][0] for b in bare.values() if b in d2)
        d2k = sum(d2[b][1] for b in bare.values() if b in d2)
        d2h = sum(d2[b][2] for b in bare.values() if b in d2) / max(d2n, 1)
        d2txt = f"{100 * d2k / d2n:.0f}%, {d2h:+.2f} ({d2n} episodes)" if d2n else '-'
        print(f"| {fam} | {run} | {len(st)} | {100 * k / max(n, 1):.1f}% | {hc:+.2f} | {dd} | {d2txt} |")
        for name, v in st.items():
            b = bare[name]
            d2v = d2.get(b)
            per_state.append((fam, name, v, d2v))
    print("\n#### Per state (the states below 90% or losing more than a heart)\n")
    print('| Family | State | Success | Health change | Deaths | d2first same state: success, health change |')
    print('|---|---|---|---|---|---|')
    for fam, name, v, d2v in sorted(per_state, key=lambda x: (x[2][1] / max(x[2][0], 1), x[1])):
        rate = v[1] / max(v[0], 1)
        hc = v[2] / max(v[0], 1)
        if rate >= 0.9 and hc > -1.0:
            continue
        d2txt = f"{100 * d2v[1] / d2v[0]:.0f}%, {d2v[2] / d2v[0]:+.2f}" if d2v and d2v[0] else '-'
        print(f"| {fam} | {name[len('d3first_'):]} | {100 * rate:.0f}% | {hc:+.2f} | {v[3]} | {d2txt} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=sorted(PARTS), default='first')
    parser.add_argument('--tag', default=None, help="the chains' tag (default the part's: r32base, r32base2)")
    parser.add_argument('--chains-dir', default='target/r56-chains')
    parser.add_argument('--d2first-chains', default='/home/leculver/git/triforce/target/r56-chains/r29land-1k')
    parser.add_argument('--d2first-steps', default='target/r32-base/d2first-main-steps.json')
    parser.add_argument('--rooms', default='target/r32-base/rooms', help="single-room results (part first only)")
    parser.add_argument('--models', default='models')
    parser.add_argument('--golden', action='store_true', help="only the golden path's segments")
    parser.add_argument('--d2first-ix-chains',
                        default='/home/leculver/git/triforce-r31-steps/target/r31-chains/r31-1k',
                        help="d2first's Dungeon 6, 8, 9 chains after phase IX (part rest; PREFIX-d6.jsonl ...)")
    parser.add_argument('--d2first-ix-steps', default='tests/fixtures/d2first_steps.json',
                        help="d2first's steps as the phase IX chains numbered them")
    args = parser.parse_args()
    global STEPS, PART, SEGMENTS  # pylint: disable=global-statement
    STEPS = _native.route_steps(ROUTE)
    if args.golden:
        golden_table(args.chains_dir)
        return
    PART = PARTS[args.part]
    SEGMENTS = PART['segments']
    tag = args.tag or PART['tag']
    d = Path(args.chains_dir)
    model = {name: load(d / f"{tag}-1k-{name}.jsonl") for name, *_ in SEGMENTS
             if (d / f"{tag}-1k-{name}.jsonl").is_file()}
    plan = {name: load(d / f"{tag}-plan-{name}.jsonl") for name in PART['planned']
            if (d / f"{tag}-plan-{name}.jsonl").is_file()}
    segment_table(model, 'Segments, models alone (today\'s team, act `model`, clips on, 1000 runs)')
    segment_table(plan, 'Segments, the planner (`temperature:0.5+relax:16+plan:16:1`, 300 runs)')
    if PART['milestones']:
        milestones_table(model.get('full'), plan.get('full'))
    seg_runs = [r for name, runs in model.items() if name != 'full' for r in runs]
    with open(args.d2first_steps, encoding='utf-8') as f:
        d2steps = json.load(f)
    d2runs = [r for name in D2_SEGMENT_FILES if os.path.isfile(f"{args.d2first_chains}-{name}.jsonl")
              for r in load(f"{args.d2first_chains}-{name}.jsonl")]
    plan_runs = [r for runs in plan.values() for r in runs]
    ix = None
    if args.part == 'rest' and os.path.isfile(f"{args.d2first_ix_chains}-d6.jsonl"):
        with open(args.d2first_ix_steps, encoding='utf-8') as f:
            ixsteps = json.load(f)
        ixruns = [trim_to_dungeon(r) for name in ('d6', 'd8', 'd9')
                  if os.path.isfile(f"{args.d2first_ix_chains}-{name}.jsonl")
                  for r in load(f"{args.d2first_ix_chains}-{name}.jsonl")]
        ix = (room_stats(ixruns), ixsteps)
    rooms_table(room_stats(seg_runs), room_stats(model.get('full', [])), room_stats(d2runs), d2steps,
                room_stats(plan_runs), ix)
    if args.part == 'first' and os.path.isdir(args.rooms):
        singles_table(args.rooms, args.models)


STEPS = []
PART = PARTS['first']

if __name__ == '__main__':
    main()
