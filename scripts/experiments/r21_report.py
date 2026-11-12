"""Round 21 (docs/experiments/round21.md): the gauntlet's tables from each run's eval-<act>.json and metrics.jsonl
(`scripts/experiments/r21_gauntlet.sh`).  Prints markdown.

    python scripts/experiments/r21_report.py [SEED...]"""
import json
import statistics
import sys
from pathlib import Path

ARMS = ['none', 'kind-jump', 'kind-walk', 'long-jump', 'long-walk', 'edge-jump', 'edge-walk']
ACTS = ['sample', 'argmax']
ENDINGS = ['exit-correct', 'exit-wrong', 'death', 'max-steps']
ROOMS = {89: '0x59', 90: '0x5a', 91: '0x5b'}


def spec(arm):
    return f'gauntlet-{arm}'


def run_dir(arm, seed):
    return Path('models') / spec(arm) / f'r21-s{seed}'


def stats(records):
    n = len(records)
    if not n:
        return None
    clips = [r.get('clips', {}) for r in records]
    endings = {e: sum(r['ending'] == e for r in records) / n for e in ENDINGS}
    endings['other'] = 1 - sum(endings.values())
    return {
        'n': n,
        'success': sum(r['success'] for r in records) / n,
        'dh': sum(r['health'] - r['start_health'] for r in records) / n,
        'steps': sum(r['steps'] for r in records) / n,
        'frames': sum(r['frames'] for r in records) / n,
        'clips': sum(c.get('done', 0) for c in clips) / n,
        'clip_share': sum(c.get('done', 0) > 0 for c in clips) / n,
        'aborted': sum(c.get('aborted', 0) for c in clips) / n,
        'endings': endings,
        'other_kinds': sorted({r['ending'] for r in records} - set(ENDINGS)),
    }


def pct(v):
    return f'{100 * v:.1f}%'


def load(arm, seed, act):
    path = run_dir(arm, seed) / f'eval-{act}.json'
    return json.loads(path.read_text())['records'] if path.exists() else None


def main_table(seeds):
    print('| arm | seed | act | success | Δhearts | steps | frames | clips/ep | ep. with clip | aborted/ep | '
          'exit-correct | exit-wrong | death | max-steps | other |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for act in ACTS:
        for arm in ARMS:
            for seed in seeds:
                records = load(arm, seed, act)
                if records is None:
                    continue
                s = stats(records)
                e = s['endings']
                other = pct(e['other']) + (f" ({', '.join(s['other_kinds'])})" if s['other_kinds'] else '')
                print(f"| {arm} | {seed} | {act} | {pct(s['success'])} | {s['dh']:+.2f} | {s['steps']:.0f} | "
                      f"{s['frames']:.0f} | {s['clips']:.2f} | {pct(s['clip_share'])} | {s['aborted']:.3f} | "
                      f"{pct(e['exit-correct'])} | {pct(e['exit-wrong'])} | {pct(e['death'])} | "
                      f"{pct(e['max-steps'])} | {other} |")


def split_table(seeds, act):
    groups = {
        'cross W→E': lambda r: r['entry'] == 'W' and r['target'] == 'E',
        'cross E→W': lambda r: r['entry'] == 'E' and r['target'] == 'W',
        'south W→S': lambda r: r['entry'] == 'W' and r['target'] == 'S',
        'south E→S': lambda r: r['entry'] == 'E' and r['target'] == 'S',
    }
    print(f'Success / Δhearts / clips per episode / mean steps / mean frames, act {act}:\n')
    print('| arm | seed | ' + ' | '.join(groups) + ' | crossing | south |')
    print('|---|---|' + '---|' * (len(groups) + 2))
    for arm in ARMS:
        for seed in seeds:
            records = load(arm, seed, act)
            if records is None:
                continue
            cells = []
            for test in list(groups.values()) + [lambda r: r['target'] != 'S', lambda r: r['target'] == 'S']:
                s = stats([r for r in records if test(r)])
                cells.append(f"{pct(s['success'])} / {s['dh']:+.2f} / {s['clips']:.2f} / {s['steps']:.0f} / "
                             f"{s['frames']:.0f}")
            print(f'| {arm} | {seed} | ' + ' | '.join(cells) + ' |')
    print()


def room_table(seeds, act):
    print(f'Per room, crossing and south: success / Δhearts / clips per episode, act {act}:\n')
    heads = [f'{name} {kind}' for name in ROOMS.values() for kind in ('cross', 'south')]
    print('| arm | seed | ' + ' | '.join(heads) + ' |')
    print('|---|---|' + '---|' * len(heads))
    for arm in ARMS:
        for seed in seeds:
            records = load(arm, seed, act)
            if records is None:
                continue
            cells = []
            for room in ROOMS:
                for south in (False, True):
                    s = stats([r for r in records if r['room'][1] == room and (r['target'] == 'S') == south])
                    cells.append(f"{pct(s['success'])} / {s['dh']:+.2f} / {s['clips']:.2f}")
            print(f'| {arm} | {seed} | ' + ' | '.join(cells) + ' |')
    print()


def state_table(seeds, act):
    """Success per (state, target) for the crossing tasks: where a wrap is on the entry row and where not."""
    print(f'Crossing success per start state, act {act} (* = the skip is available on the entry row):\n')
    wrap_rows = {'0_59w_1', '0_59e_0', '0_59e_1', '0_59e_2', '0_5aw_0', '0_5aw_1', '0_5aw_2', '0_5ae_1', '0_5ae_2',
                 '0_5bw_0', '0_5bw_1', '0_5be_0', '0_5be_1', '0_5be_2'}
    states = [f'0_{room}{side}_{c}' for room in ('59', '5a', '5b') for side in 'we' for c in range(3)]
    print('| arm | seed | ' + ' | '.join(s + ('*' if s in wrap_rows else '') for s in states) + ' |')
    print('|---|---|' + '---|' * len(states))
    for arm in ARMS:
        for seed in seeds:
            records = load(arm, seed, act)
            if records is None:
                continue
            cells = []
            for state in states:
                s = stats([r for r in records if r['start_state'] == state and r['target'] != 'S'])
                cells.append(f"{100 * s['success']:.0f}" if s else '-')
            print(f'| {arm} | {seed} | ' + ' | '.join(cells) + ' |')
    print()


def wrap_table(seeds):
    """Which way the crossings wrap (the right way is toward the entry edge: W after entering from W), how many
    crossings left through the entry edge (exit-wrong) with no wrap done, and the wraps in the south tasks."""
    print('Wrap directions from the evaluation records (crossing tasks: right = toward the entry edge, wrong = '
          'toward the target edge; per episode). `entry-edge exits` = crossings that ended exit-wrong through the '
          'edge they came in by, `0 wraps` of them without any wrap. 0_5bw_0 E: crossings from 0_5bw_0, whose wrap '
          'lands in the lower (wrong) E opening: success, exit-wrong through E / through W.\n')
    print('| arm | seed | act | right wraps/crossing | wrong wraps/crossing | crossings with ≥ 2 wraps | '
          'entry-edge exits (0 wraps) | wraps/south ep. | 0_5bw_0 E: success, exit E / W |')
    print('|---|---|---|---|---|---|---|---|---|')
    for act in ACTS:
        for arm in ARMS:
            for seed in seeds:
                records = load(arm, seed, act)
                if records is None:
                    continue
                crossing = [r for r in records if r['target'] != 'S']
                south = [r for r in records if r['target'] == 'S']

                def done(r, d):
                    return r.get('clips', {}).get(f'done_{d.lower()}', 0)
                right = sum(done(r, r['entry']) for r in crossing) / len(crossing)
                wrong = sum(done(r, r['target']) for r in crossing) / len(crossing)
                multi = sum(r.get('clips', {}).get('done', 0) >= 2 for r in crossing)
                back = [r for r in crossing if r['ending'] == 'exit-wrong' and r.get('visits')
                        and r['visits'][-1]['exit'] == r['entry'].lower()]
                back0 = sum(r.get('clips', {}).get('done', 0) == 0 for r in back)
                south_wraps = sum(r.get('clips', {}).get('done', 0) for r in south) / len(south)
                lower = [r for r in crossing if r['start_state'] == '0_5bw_0']
                exit_e = sum(r['ending'] == 'exit-wrong' and r['visits'][-1]['exit'] == 'e' for r in lower)
                exit_w = sum(r['ending'] == 'exit-wrong' and r['visits'][-1]['exit'] == 'w' for r in lower)
                print(f"| {arm} | {seed} | {act} | {right:.2f} | {wrong:.3f} | {multi}/{len(crossing)} | "
                      f"{len(back)} ({back0}) | {south_wraps:.2f} | "
                      f"{pct(sum(r['success'] for r in lower) / len(lower))}, {exit_e} / {exit_w} |")
    print()



def rolling(values, n=5):
    return [statistics.mean(values[max(0, i - n + 1):i + 1]) for i in range(len(values))]


def learning_table(seeds):
    print('Training (metrics.jsonl; success = rolling mean of 5 iterations of the training success rate, '
          'clips = `clips/done` per finished episode):\n')
    print('| arm | seed | steps | 50% at | 80% at | 90% at | final success (last 10 it.) | clips/ep first 5 it. | '
          'clips ≥ 0.5 from | clips peak (at) | clips within 0.1 of final from | clips last 10 it. | '
          'aborted last 10 it. | wall time |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for arm in ARMS:
        for seed in seeds:
            path = run_dir(arm, seed) / 'metrics.jsonl'
            if not path.exists():
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            success = rolling([r['success_rate'] for r in rows])
            clips = rolling([r.get('clips/done', 0.0) for r in rows])
            cells = []
            for threshold in (0.5, 0.8, 0.9):
                hit = next((rows[i]['steps'] for i, v in enumerate(success) if v >= threshold), None)
                cells.append(f'{hit / 1e3:.0f}k' if hit else 'not reached')
            rise = next((rows[i]['steps'] for i, v in enumerate(clips) if v >= 0.5), None)
            peak = max(range(len(clips)), key=lambda i: clips[i])
            final = statistics.mean(r.get('clips/done', 0) for r in rows[-10:])
            settle = next((rows[i]['steps'] for i in range(len(clips))
                           if all(abs(v - final) <= 0.1 for v in clips[i:])), None)
            summary = run_dir(arm, seed) / 'train_summary.json'
            wall = json.loads(summary.read_text()).get('wall_seconds') if summary.exists() else None
            last = rows[-10:]
            print(f"| {arm} | {seed} | {rows[-1]['steps'] / 1e6:.2f}M | {' | '.join(cells)} | "
                  f"{pct(statistics.mean(r['success_rate'] for r in last))} | "
                  f"{statistics.mean(r.get('clips/done', 0) for r in rows[:5]):.2f} | "
                  f"{f'{rise / 1e3:.0f}k' if rise else 'never'} | {clips[peak]:.2f} ({rows[peak]['steps'] / 1e3:.0f}k) | "
                  f"{f'{settle / 1e3:.0f}k' if settle else 'not settled'} | {final:.2f} | "
                  f"{statistics.mean(r.get('clips/aborted', 0) for r in last):.3f} | "
                  f"{f'{wall / 60:.0f} min' if wall else '-'} |")
    print()


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1]
    print('## Final evaluation\n')
    main_table(seeds)
    print()
    for act in ACTS:
        print('## By task type\n')
        split_table(seeds, act)
        room_table(seeds, act)
        state_table(seeds, act)
    wrap_table(seeds)
    print('## Learning\n')
    learning_table(seeds)


if __name__ == '__main__':
    main()
