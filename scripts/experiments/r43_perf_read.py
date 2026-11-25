"""Round 43 (Perf43): reads the speed runs of ``scripts/experiments/r43_perf.sh`` (round40.md "Where a teacher's time
goes"): per run (and per serving mode, for a run that alternated them: ``charts/serve_graphs``) its median rollout
steps a second (in all and a game) over the iterations after the first ``--skip``, the process's CPU a decision, the
served network's batches, the env threads' shares (``r43_perf_sleep.py``'s windows that fell inside those rollouts)
and the load average over its iterations (``target/r43-perf/load.log``).

    .venv/bin/python scripts/experiments/r43_perf_read.py [MODELS_DIR...] [--skip 2]
"""
import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
NAN = float('nan')


def loads(path: Path) -> list[tuple[float, float]]:
    out = []
    if path.is_file():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                out.append((float(parts[0]), float(parts[1])))
    return out


def sleeps(path: Path) -> dict[str, list[tuple]]:
    """name/run -> [(epoch, running, waiting, sleeping)] (each window's end is epoch + its length)."""
    out: dict[str, list] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 7:
                out.setdefault(parts[1], []).append((float(parts[0]), *map(float, parts[3:6])))
    return out


def med(values) -> float:
    values = [v for v in values if v == v]
    return statistics.median(values) if values else NAN


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='*', default=[str(HERE / 'target/r43-perf/models')])
    parser.add_argument('--skip', type=int, default=2, help="iterations left out at the start")
    parser.add_argument('--load', default=str(HERE / 'target/r43-perf/load.log'))
    parser.add_argument('--sleep', default=str(HERE / 'target/r43-perf/sleep.log'))
    parser.add_argument('--window', type=float, default=10.0)
    args = parser.parse_args()
    load, slept = loads(Path(args.load)), sleeps(Path(args.sleep))
    rows = []
    for models in args.models:
        for metrics in sorted(Path(models).glob('r43-perf-*/r*/metrics.jsonl')):
            run = json.loads((metrics.parent / 'run.json').read_text())
            games = sum(run['envs'].values())
            name = f"{metrics.parent.parent.name}/{metrics.parent.name}"
            its = [json.loads(line) for line in metrics.read_text().splitlines()]
            kept = [m for m in its if m['iteration'] > args.skip] or its
            modes = sorted({m.get('charts/serve_graphs') for m in kept}, key=str)
            for mode in modes:
                group = [m for m in kept if m.get('charts/serve_graphs') == mode]
                # Each iteration's rollout ended about its update's seconds before its line; it began rollout_seconds
                # before that.
                spans = [(m['time'] - m.get('charts/optimize_seconds', 0) - m.get('charts/recompute_seconds', 0)
                          - m.get('charts/rollout_seconds', 0),
                          m['time'] - m.get('charts/optimize_seconds', 0) - m.get('charts/recompute_seconds', 0))
                         for m in group]
                inside = [w for w in slept.get(name, []) if any(a <= w[0] - args.window and w[0] <= b
                                                                for a, b in spans)]
                during = [value for t, value in load if any(a - 30 <= t <= b + 30 for a, b in spans)]
                roll = med(m.get('charts/rollout_SPS', NAN) for m in group)
                cpu = med(m['charts/cpu_seconds'] / m['rollout/steps'] * 1000 for m in group
                          if 'charts/cpu_seconds' in m)
                rows.append((name.removeprefix('r43-perf-'), {None: '-', 1.0: 'graphs', 0.0: 'eager'}.get(mode, mode),
                             games, len(group), roll, roll / games, cpu,
                             med(m.get('charts/serve_rows_per_batch', NAN) for m in group),
                             med(m.get('charts/serve_busy', NAN) for m in group),
                             med(w[1] for w in inside), med(w[2] for w in inside), med(w[3] for w in inside),
                             len(inside), statistics.mean(during) if during else NAN, group[0]['time']))
    print("| run | serving | games | iterations | rollout steps/s | a game | CPU ms a decision | rows a batch | "
          "server busy | env running | env waiting for a CPU | env sleeping | windows | load |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda r: r[-1]):
        (name, mode, games, n, roll, per, cpu, rows_b, busy, running, waiting, sleeping, windows, ld, _) = row
        print(f"| {name} | {mode} | {games} | {n} | {roll:,.0f} | {per:.1f} | {cpu:.2f} | {rows_b:.1f} | "
              f"{busy:.2f} | {running:.2f} | {waiting:.2f} | {sleeping:.2f} | {windows} | {ld:.1f} |")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
