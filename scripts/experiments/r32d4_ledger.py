"""Round 32: Dungeon 4's per-room ledger on d3first (route steps 31-52) for d4 chains side by side: entered, lost
(per 1000 entering), hearts in, hearts lost a visit, bombs used, cheated bombs; the segment's arrivals and Gleeok.

    .venv/bin/python scripts/experiments/r32d4_ledger.py LABEL=CHAIN.jsonl ...    (markdown on stdout)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r32base_tables as t                                         # noqa: E402

arms = [(a.split('=', 1)[0], t.load(a.split('=', 1)[1])) for a in sys.argv[1:]]
t.STEPS = t._native.route_steps(t.ROUTE)                           # pylint: disable=protected-access
stats = [(label, t.room_stats(runs), runs) for label, runs in arms]
print('| Arm | Runs | Arrived (0:0x45) | Gleeok entered | Gleeok won |')
print('|---|---|---|---|---|')
for label, s, runs in stats:
    n = len(runs)
    ok = sum(r['result'] == 'success' for r in runs)
    g = s.get(51, {})
    entered = int(g.get('runs', 0))
    won = entered - int(g.get('ended', 0))
    print(f"| {label} | {n} | **{100 * ok / n:.1f}%** ± {t.ci(ok, n) / 1.96:.1f} ({ok}) | {entered} | {won} |")
print()
head = ' | '.join(f'{label}: entered, lost/1000, hearts in, lost a visit, bombs, cheated b' for label, _, _ in stats)
print(f'| Step | Room | In→out | Family | {head} |')
print('|---|---|---|---|' + '---|' * len(stats))
for step in range(31, 53):
    room, way, family = t.step_name(step)
    cells = []
    for _, s, _ in stats:
        x = s.get(step)
        if not x or not x.get('runs'):
            cells.append('-')
            continue
        n = x['runs']
        cells.append(f"{int(n)}, {1000 * x.get('ended', 0) / n:.0f}, {x['hin'] / n:.2f}/{x['max'] / n:.0f}, "
                     f"{x['lost'] / x['visits']:.2f}, {x['bombs'] / x['visits']:.2f}, {int(x['cheated_bombs'])}")
    print(f"| {step} | {room} | {way} | {family} | {' | '.join(cells)} |")
