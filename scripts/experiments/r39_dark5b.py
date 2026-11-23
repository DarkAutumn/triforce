"""Round 39, 3:0x5b with a full bag (backlog Track R; round38.md "The owner's three causes"): the single-room cases
of d3first's step 19 and the paired tables of the darknut@d3first continuation.

    python scripts/experiments/r39_dark5b.py states OUT_DIR
    python scripts/experiments/r39_dark5b.py tasks OUT_DIR
    python scripts/experiments/r39_dark5b.py evals BASE.json ARM.json [...]
    python scripts/experiments/r39_dark5b.py chains BASE.jsonl ARM.jsonl [--rooms L:0xRR ...]

``states``: the owner's case as start states, written to OUT_DIR (every state of ``triforce/states`` linked beside
them, so OUT_DIR serves as ``TRIFORCE_STATES_DIR``): 0x5b entered from W (``d3first_3_5bw_0-2``), all three
Darknuts banked (the room's bombs lying at ($80, $90)), 8 bombs, hearts as the family draws them; Link walked round
the bombs (E 2, N 2, then E along row y 93) and stopped at one of six points E of them, or at the bomb spot's row
(S 2, then W 0-2), where the golden replay paced. Named ``d3first_3_5bw_<10-39>`` (route step 19 by the name).

``tasks``: two evaluation yamls of darknut@d3first in OUT_DIR: ``lying.yaml`` (those states as they are) and
``full.yaml`` (``d3first_3_5bw_0-2`` with 8 bombs and the Darknuts alive: the crossing).

``evals``: per evaluation, success, the wall's bomb laid (``bombs_used`` of the 0x5b visit), frames and aimless
frames a visit (mean / median / p90), steps, hearts lost; paired by (state, seed) against the first.

``chains``: per chain record, the segment's arrival (``eval_batch.D3FIRST_TARGETS`` by the file's segment suffix)
and per room the darknut visits, deaths, hearts lost and hits a visit, frames a visit; paired by seed.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

YAML = ROOT / 'specialists' / 'd3first' / 'darknut.yaml'
BASES = ('d3first_3_5bw_0', 'd3first_3_5bw_1', 'd3first_3_5bw_2')
# After E 2, N 2 (to row y 93): E k more; then optionally S 2 (to the spot's row, y ~136) and W j.
STOPS = [(k, False, 0) for k in (3, 4, 5, 6, 7, 8)] + [(8, True, j) for j in (0, 1, 2, 3)]


def build_states(out: Path):
    import yaml
    from triforce import _native
    from triforce.paths import STATES_DIR
    from triforce.specialist import load_specialist

    out.mkdir(parents=True, exist_ok=True)
    for state in STATES_DIR.iterdir():
        link = out / state.name
        if not link.exists():
            link.symlink_to(state.resolve())
    spec = yaml.safe_load(YAML.read_text(encoding='utf-8'))
    ram = spec['ram_randomization']
    ram.update({'bombs': [8, 8], 'bombs_share': 1.0, 'room_kills': {'(3, 0x5b)': [3, 3]}, 'room_kills_share': 1.0})
    for key in ('room_bombs', 'room_bombs_share'):
        ram.pop(key, None)
    builder = out.parent / 'r39-build-5b.yaml'
    builder.write_text(yaml.safe_dump(spec, sort_keys=False), encoding='utf-8')
    config = load_specialist(str(builder))
    index = 10
    for b, base in enumerate(BASES):
        env = _native.Env(config, start_state=base, staged_kills=True, strict_actions=False)
        for s, (east, south, west) in enumerate(STOPS):
            env.reset(seed=100 * b + s)
            moves = ['MOVE E 16'] * 2 + ['MOVE N 16'] * 2 + ['MOVE E 16'] * east
            moves += (['MOVE S 16'] * 2 + ['MOVE W 16'] * west) if south else []
            for move in moves:
                env.step(move)
            state = env.state
            treasure = state.treasure
            assert treasure and (treasure['x'], treasure['y']) == (0x80, 0x90), f"{base} {s}: the bombs lying"
            assert state.bombs == 8 and not state.enemies and env.route == ('d3first', 19), (base, s)
            name = f'd3first_3_5bw_{index}.state'
            (out / name).unlink(missing_ok=True)
            (out / name).write_bytes(bytes(env.snapshot().state_file()))
            print(f"{name}: {base}, Link ({state.link['x']}, {state.link['y']}), hp {state.health}, bombs 8")
            index += 1
    builder.unlink()


def write_tasks(out: Path):
    import yaml
    spec = yaml.safe_load(YAML.read_text(encoding='utf-8'))
    common = {'rooms': ['(3, 0x5b)'], 'harvested_fraction': 0.0, 'curriculum': {'enabled': False}}
    ram = dict(spec['ram_randomization'])
    for key in ('room_kills', 'room_kills_share', 'room_bombs', 'room_bombs_share'):
        ram.pop(key, None)
    lying = {**spec, **common, 'start_states': ['3_5bw_[1-3][0-9]'], 'ram_randomization': {}}
    full = {**spec, **common, 'start_states': ['3_5bw_[0-2]'],
            'ram_randomization': {**ram, 'bombs': [8, 8], 'bombs_share': 1.0}}
    out.mkdir(parents=True, exist_ok=True)
    for name, task in (('lying', lying), ('full', full)):
        task['notes'] = f"round 39, 3:0x5b single-room ({name}): scripts/experiments/r39_dark5b.py"
        (out / f'{name}.yaml').write_text(yaml.safe_dump(task, sort_keys=False), encoding='utf-8')
        print(out / f'{name}.yaml')


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def _paired(base: dict, arm: dict, value) -> str:
    keys = sorted(set(base) & set(arm))
    d = [value(arm[k]) - value(base[k]) for k in keys]
    if len(d) < 2:
        return ''
    return f"{statistics.mean(d):+.3f} ± {statistics.pstdev(d) / math.sqrt(len(d)):.3f}"


def _visit(record):
    for visit in record.get('visits') or []:
        if visit['level'] == 3 and visit['room'] in ('0x5b', 91) and not visit.get('in_cave'):
            return visit
    return None


def evals(paths):
    rows = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        records = {(r.get('start_state'), r.get('seed')): r for r in data['records']}
        rows.append((path, records))
    base = rows[0][1]
    for path, records in rows:
        visits = [_visit(r) or {} for r in records.values()]
        n = len(records)
        ok = sum(r.get('outcome') == 'exit-correct' or r.get('success') for r in records.values())
        laid = sum((v.get('bombs_used') or 0) > 0 for v in visits)
        frames = [v.get('frames', 0) for v in visits]
        aimless = [v.get('aimless_frames', 0) for v in visits]
        hearts = [v.get('health_in', 0) - v.get('health_out', 0) for v in visits]
        steps = [r.get('steps', 0) for r in records.values()]
        print(f"{Path(path).name}: {n} episodes, success {ok / n:.1%}, bomb laid {laid / n:.1%}, "
              f"frames {statistics.mean(frames):.0f} / {_pct(frames, .5)} / {_pct(frames, .9)}, "
              f"aimless {statistics.mean(aimless):.0f} / {_pct(aimless, .5)} / {_pct(aimless, .9)}, "
              f"steps {statistics.mean(steps):.1f}, hearts lost {statistics.mean(hearts):.3f}")
        if records is not base:
            print(f"   paired: frames {_paired(base, records, lambda r: (_visit(r) or {}).get('frames', 0))}, "
                  f"aimless {_paired(base, records, lambda r: (_visit(r) or {}).get('aimless_frames', 0))}, "
                  f"success {_paired(base, records, lambda r: float(r.get('outcome') == 'exit-correct' or bool(r.get('success'))))}")


def chains(paths, rooms):
    from triforce.eval_batch import D3FIRST_TARGETS, arrival
    runs = []
    for path in paths:
        segment = Path(path).stem.rsplit('-', 1)[-1]
        target = D3FIRST_TARGETS.get(segment)
        records = {}
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            records[r['seed']] = r
        runs.append((path, target, records))
    base = runs[0][2]
    for path, target, records in runs:
        def arrived(r, target=target):
            return float(bool(arrival(r, target)) if target else r['result'] == 'success')
        n = len(records)
        rate = sum(arrived(r) for r in records.values()) / n
        line = f"{Path(path).name}: {n} runs, arrival {rate:.1%}"
        if records is not base:
            keys = sorted(set(base) & set(records))
            only_b = sum(arrived(base[k]) > arrived(records[k]) for k in keys)
            only_a = sum(arrived(records[k]) > arrived(base[k]) for k in keys)
            line += f" (paired {_paired(base, records, arrived)}; base only {only_b} / arm only {only_a})"
        print(line)
        stats = {}
        for r in records.values():
            for v in r['rooms']:
                if not v['specialist'].startswith('darknut'):
                    continue
                key = f"{v['level']}:{v['room']}@{v.get('step')}"
                if rooms and f"{v['level']}:{v['room']}" not in rooms:
                    continue
                s = stats.setdefault(key, {'visits': 0, 'deaths': 0, 'hearts': 0.0, 'hits': 0, 'frames': 0,
                                           'aimless': 0, 'stuck': 0, 'bombs': 0})
                s['visits'] += 1
                s['deaths'] += v['outcome'] == 'death'
                s['stuck'] += v['outcome'] not in ('death', 'exit-correct')
                s['hearts'] += max(0.0, v['health_in'] - v['health_out'])
                s['hits'] += v.get('hits') or 0
                s['frames'] += v['frames']
                s['aimless'] += v.get('aimless_frames') or 0
                s['bombs'] += v.get('bombs_used') or 0
        for key in sorted(stats, key=lambda k: (k.split('@')[0], int(k.split('@')[1]) if k.split('@')[1] != 'None' else -1)):
            s = stats[key]
            n_v = s['visits']
            print(f"   {key}: {n_v} visits, {s['deaths']} deaths, {s['stuck']} other, hearts {s['hearts'] / n_v:.3f}, "
                  f"hits {s['hits'] / n_v:.3f}, frames {s['frames'] / n_v:.0f}, aimless {s['aimless'] / n_v:.0f}, "
                  f"bombs {s['bombs'] / n_v:.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('states').add_argument('out', type=Path)
    sub.add_parser('tasks').add_argument('out', type=Path)
    sub.add_parser('evals').add_argument('paths', nargs='+')
    c = sub.add_parser('chains')
    c.add_argument('paths', nargs='+')
    c.add_argument('--rooms', nargs='*', default=[])
    args = parser.parse_args(argv)
    if args.command == 'states':
        build_states(args.out)
    elif args.command == 'tasks':
        write_tasks(args.out)
    elif args.command == 'evals':
        evals(args.paths)
    else:
        chains(args.paths, args.rooms)


if __name__ == '__main__':
    main()
