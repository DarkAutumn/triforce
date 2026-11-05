"""Old-stack (stable-retro/FCEUmm) training throughput from logs/*/<specialist>.jsonl progress events.

    .venv/bin/python scripts/old_stack_sps.py      # table on stdout, per-run list in target/perf/old_stack_runs.json

Per run: specialist, parallel envs, device, median 'sps' over the run's settled progress events (dropping the
first 10% so startup doesn't count); how many runs shared the machine at the time is unknown (rounds often ran
two at once). Prints a table grouped by (parallel, specialist). See docs/performance.md."""
import glob
import json
import os
import statistics
from collections import defaultdict

runs = []
for path in sorted(glob.glob('logs/*/*.jsonl')):
    start, sps = None, []
    for line in open(path, encoding='utf-8'):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get('event') == 'start' and 'parallel' in e:
            if start and len(sps) >= 10:
                runs.append((start, sps))
            start, sps = e, []
        elif e.get('event') == 'progress' and start and e.get('sps'):
            sps.append(e['sps'])
    if start and len(sps) >= 10:
        runs.append((start, sps))

groups = defaultdict(list)
for start, sps in runs:
    settled = sps[len(sps) // 10:]
    groups[(start['parallel'], start['specialist'])].append((statistics.median(settled), start['run_id'],
                                                              start['steps_per_update'], start['device'],
                                                              start['time'][:10]))
print(f"{'parallel':>8} {'specialist':18} {'runs':>4} {'median sps':>10} {'range':>14}")
for (par, spec), rows in sorted(groups.items()):
    meds = [r[0] for r in rows]
    print(f"{par:>8} {spec:18} {len(rows):>4} {statistics.median(meds):10.1f} {min(meds):7.1f}-{max(meds):<6.1f}")
os.makedirs('target/perf', exist_ok=True)
json.dump([{'parallel': s['parallel'], 'specialist': s['specialist'], 'run_id': s['run_id'], 'device': s['device'],
            'steps_per_update': s['steps_per_update'], 'date': s['time'][:10],
            'median_sps': statistics.median(v[len(v) // 10:]), 'events': len(v)} for s, v in runs],
          open('target/perf/old_stack_runs.json', 'w', encoding='utf-8'), indent=1)
