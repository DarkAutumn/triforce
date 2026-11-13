#!/usr/bin/env python3
"""Round 23: the act-mode pilot's tables (docs/experiments/round23.md), from what r23_act_pilot.sh wrote.

    .venv/bin/python scripts/experiments/r23_act_report.py [--dir target/r23-pilot] [--json OUT]
        [--modes m1,m2,...] [--baseline model]

Single rooms (``<dir>/rooms/<mode>/<family>-<mode>.json``, ``triforce evaluate`` results): per mode over the
families measured in it, and per family per mode: success with its standard error, hearts (mean health change per
episode, SE), deaths, loops (``stuck``, ``no-damage``, ``no-room-progress`` and ``max-steps`` endings), frames per
episode, moves under 1% and 5% per episode and the share of them followed by a hit or a death within 4 decisions,
the acting entropy per decision, and the variety from the same start state (mean over states of the share of
distinct action sequences, the mean first diverging decision of two episodes, the frames' standard deviation).
``model`` (the baseline) is each family's own act mode, read from its ``act`` file: the sample or argmax result.

Chains (``<dir>/chains/r23-<mode>-<segment>.jsonl``, ``triforce eval-chain``): per segment and mode the arrivals
(``triforce.eval_batch.arrival``: the next segment's start room; the D4 triforce for ``d4`` and ``full``) with the
SE, how the other runs ended before arriving (death, stuck, other), hearts at the arrival, frames to it, and the
same move counts and variety (the segment's start room, every run from the same state) up to the arrival.
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from triforce import registry  # noqa: E402  pylint: disable=wrong-import-position
from triforce.eval_batch import TARGETS, arrival  # noqa: E402  pylint: disable=wrong-import-position
from triforce.variety import decisions_of, risk, sequence, variety  # noqa: E402  pylint: disable=wrong-import-position

LOOPS = ('stuck', 'no-damage', 'no-room-progress', 'max-steps')
RAW = ('decisions', 'low1', 'low5', 'low1_hit', 'low5_hit', 'low1_death', 'low5_death', 'hit', 'death')
SEGMENTS = ('d2', 'd3', 'd4', 'full')


def se_rate(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n) if n else float('nan')


def mean_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if not n:
        return float('nan'), float('nan')
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return mean, sd / math.sqrt(n)


def room_summary(result: dict) -> dict:
    """One evaluation's row."""
    records = result['records']
    n = len(records)
    endings = Counter(r['ending'] for r in records)
    hearts = [r['health'] - r['start_health'] for r in records]
    success = sum(r['success'] for r in records) / n
    states = [s['variety'] for s in result['by_state'].values() if s.get('variety')]

    def state_mean(key):
        values = [v[key] for v in states if v.get(key) is not None]
        return sum(values) / len(values) if values else None
    r = result.get('risk') or {}
    raw = {key: r.get(key) or 0 for key in RAW}
    return {
        'episodes': n, 'success': success, 'success_se': se_rate(success, n),
        'hearts': mean_se(hearts)[0], 'hearts_se': mean_se(hearts)[1],
        'deaths': endings.get('death', 0), 'loops': sum(endings.get(k, 0) for k in LOOPS),
        'endings': dict(endings), 'frames': result['mean_frames'],
        'low1': r.get('low1_per_episode'), 'low5': r.get('low5_per_episode'),
        'low1_hit': r.get('low1_hit_rate'), 'low5_hit': r.get('low5_hit_rate'),
        'low1_death': r.get('low1_death_rate'), 'low5_death': r.get('low5_death_rate'),
        'entropy': r.get('act_entropy_per_decision'),
        'hit_base': r.get('hit_rate'), 'death_base': r.get('death_rate'),
        'distinct': state_mean('distinct_rate'), 'diverge': state_mean('diverge_mean'),
        'frames_sd': state_mean('frames_sd'), 'raw': raw,
    }


def read_rooms(directory: Path, modes: list[str]) -> dict:
    """{mode: {family: row}}."""
    rooms = {}
    for mode in modes:
        folder = directory / 'rooms' / mode
        for path in sorted(folder.glob('*.json')) if folder.is_dir() else ():
            result = json.loads(path.read_text(encoding='utf-8'))
            if result.get('act') != mode:
                continue
            rooms.setdefault(mode, {})[result['specialist']] = room_summary(result)
    return rooms


def own_modes(models_dir) -> dict:
    """Each family's act mode (its ``act`` file)."""
    from triforce.specialist import list_specialists  # pylint: disable=import-outside-toplevel
    modes = {}
    for name in list_specialists():
        try:
            modes[name] = registry.act_mode(registry.promoted_checkpoint(name, models_dir))
        except FileNotFoundError:
            pass
    return modes


def add_baseline(rooms: dict, own: dict) -> None:
    """``model``: each family's sample or argmax row, by its act file."""
    baseline = {family: rooms[mode][family] for family, mode in own.items() if family in rooms.get(mode, {})}
    if baseline:
        rooms['model'] = baseline


def mode_totals(rows: dict) -> dict:
    """A mode over its families: episodes, mean success (families weighted alike), failures, deaths, loops, hearts
    and the per-episode means weighted by episodes."""
    n = sum(r['episodes'] for r in rows.values())

    def weighted(key):
        values = [(r[key], r['episodes']) for r in rows.values() if r.get(key) is not None]
        return sum(v * w for v, w in values) / sum(w for _, w in values) if values else None
    failures = sum(round((1 - r['success']) * r['episodes']) for r in rows.values())
    raw = {key: sum(r['raw'][key] for r in rows.values()) for key in RAW}
    based = all(r['hit_base'] is not None for r in rows.values())

    def share(part, whole):
        return raw[part] / raw[whole] if raw[whole] else None
    return {'families': len(rows), 'episodes': n, 'success': weighted('success'),
            'success_se': math.sqrt(sum((r['success_se'] * r['episodes']) ** 2 for r in rows.values())) / n,
            'failures': failures, 'deaths': sum(r['deaths'] for r in rows.values()),
            'loops': sum(r['loops'] for r in rows.values()), 'hearts': weighted('hearts'),
            'frames': weighted('frames'), 'low1': raw['low1'] / n, 'low5': raw['low5'] / n,
            'low1_hit': share('low1_hit', 'low1'), 'low5_hit': share('low5_hit', 'low5'),
            'low1_death': share('low1_death', 'low1'), 'low5_death': share('low5_death', 'low5'),
            'hit_base': share('hit', 'decisions') if based else None,
            'death_base': share('death', 'decisions') if based else None,
            'entropy': weighted('entropy'),
            'distinct': weighted('distinct'), 'diverge': weighted('diverge'), 'frames_sd': weighted('frames_sd')}


def segment_summary(runs: list[dict], segment: str) -> dict:
    """One chain segment file's row."""
    target = TARGETS.get(segment)
    arrived, endings, hearts, frames, groups, sequences, first_frames = 0, Counter(), [], [], [], [], []
    for run in runs:
        record = arrival(run, target)
        rooms = run['rooms']
        upto = rooms[:rooms.index(record) + 1] if record is not None and target is not None else rooms
        if record is not None:
            arrived += 1
            hearts.append(float(record['health_in'] if target is not None else record['health_out']))
            frames.append(sum(r.get('frames') or 0 for r in (upto[:-1] if target is not None else upto)))
        else:
            last = rooms[-1] if rooms else {}
            outcome = last.get('outcome', 'error')
            endings['death' if outcome == 'death' else 'stuck' if outcome == 'stuck' else outcome] += 1
        groups.append([decisions_of(r) for r in (upto[:-1] if record is not None and target is not None else upto)])
        if rooms:
            sequences.append(sequence(rooms[:1]))
            first_frames.append(rooms[0].get('frames') or 0)
    n = len(runs)
    rate = arrived / n if n else 0.0
    r = risk(groups, 'run')
    v = variety(sequences, first_frames)
    return {'runs': n, 'arrived': arrived, 'arrival': rate, 'arrival_se': se_rate(rate, n),
            'deaths': endings.get('death', 0), 'stuck': endings.get('stuck', 0),
            'other': sum(c for k, c in endings.items() if k not in ('death', 'stuck')), 'endings': dict(endings),
            'hearts': mean_se(hearts)[0], 'frames': mean_se(frames)[0], 'frames_sd': variety([], frames).get('frames_sd'),
            'low1': r.get('low1_per_run'), 'low5': r.get('low5_per_run'), 'low1_hit': r.get('low1_hit_rate'),
            'low5_hit': r.get('low5_hit_rate'), 'low1_death': r.get('low1_death_rate'),
            'low5_death': r.get('low5_death_rate'), 'hit_base': r.get('hit_rate'),
            'death_base': r.get('death_rate'), 'entropy': r.get('act_entropy_per_decision'),
            'distinct_first_room': v.get('distinct_rate'), 'diverge_first_room': v.get('diverge_mean')}


def read_chains(directory: Path, modes: list[str]) -> dict:
    """{segment: {mode: row}}."""
    chains = {}
    for mode in modes + ['model']:
        for segment in SEGMENTS:
            path = directory / 'chains' / f'r23-{mode}-{segment}.jsonl'
            if path.is_file():
                runs = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
                chains.setdefault(segment, {})[mode] = segment_summary(runs, segment)
    return chains


def fmt(value, spec='.1f', scale=1.0, none='–'):
    return none if value is None or (isinstance(value, float) and math.isnan(value)) else format(value * scale, spec)


def pct(value, se=None):
    if value is None:
        return '–'
    return f"{value * 100:.1f}" + (f" ± {se * 100:.1f}" if se is not None else '')


RISK_HEAD = ('<1% | then hit / death % | <5% | then hit / death % | any move: then hit / death % | entropy',
             '---:|---|---:|---|---|---:')


def risk_cells(row):
    return (f"{fmt(row['low1'], '.2f')} | {pct(row['low1_hit'])} / {pct(row['low1_death'])} | "
            f"{fmt(row['low5'], '.2f')} | {pct(row['low5_hit'])} / {pct(row['low5_death'])} | "
            f"{pct(row['hit_base'])} / {pct(row['death_base'])} | {fmt(row['entropy'], '.2f')}")


def report(rooms: dict, chains: dict, modes: list[str]) -> str:
    lines = []
    order = [m for m in ['model'] + modes if m in rooms]
    if rooms:
        lines += ['## Single rooms, per mode (families weighted by episodes; moves per episode)', '',
                  '| mode | families | episodes | success % | failures | deaths | loops | hearts/ep | frames/ep | '
                  f'{RISK_HEAD[0]} | distinct % | diverge | frames sd |',
                  f'|---|---:|---:|---:|---:|---:|---:|---:|---:|{RISK_HEAD[1]}|---:|---:|---:|']
        for mode in order:
            t = mode_totals(rooms[mode])
            lines.append(
                f"| {mode} | {t['families']} | {t['episodes']} | {pct(t['success'], t['success_se'])} | "
                f"{t['failures']} | {t['deaths']} | {t['loops']} | {fmt(t['hearts'], '+.3f')} | {fmt(t['frames'], '.0f')} | "
                f"{risk_cells(t)} | {pct(t['distinct'])} | {fmt(t['diverge'], '.0f')} | "
                f"{fmt(t['frames_sd'], '.0f')} |")
        families = sorted({f for rows in rooms.values() for f in rows})
        lines += ['', '## Single rooms, per family', '']
        for family in families:
            lines += [f'### {family}', '',
                      f'| mode | success % | hearts/ep | deaths | loops | frames | {RISK_HEAD[0]} | distinct % | diverge | '
                      'frames sd |',
                      f'|---|---:|---:|---:|---:|---:|{RISK_HEAD[1]}|---:|---:|---:|']
            for mode in order:
                row = rooms[mode].get(family)
                if row is None:
                    continue
                lines.append(
                    f"| {mode} | {pct(row['success'], row['success_se'])} | {fmt(row['hearts'], '+.2f')} ± "
                    f"{fmt(row['hearts_se'], '.2f')} | {row['deaths']} | {row['loops']} | {fmt(row['frames'], '.0f')} | "
                    f"{risk_cells(row)} | {pct(row['distinct'])} | {fmt(row['diverge'], '.0f')} | "
                    f"{fmt(row['frames_sd'], '.0f')} |")
            lines.append('')
    for segment in SEGMENTS:
        if segment not in chains:
            continue
        lines += [f'## Chains: {segment}', '',
                  '| mode | runs | arrived % | deaths | stuck | other | hearts at arrival | frames | frames sd | '
                  f'{RISK_HEAD[0]} | distinct 1st room % | diverge 1st room |',
                  f'|---|---:|---:|---:|---:|---:|---:|---:|---:|{RISK_HEAD[1]}|---:|---:|']
        for mode in ['model'] + modes:
            row = chains[segment].get(mode)
            if row is None:
                continue
            lines.append(
                f"| {mode} | {row['runs']} | {pct(row['arrival'], row['arrival_se'])} | {row['deaths']} | "
                f"{row['stuck']} | {row['other']} | {fmt(row['hearts'], '.2f')} | {fmt(row['frames'], '.0f')} | "
                f"{fmt(row['frames_sd'], '.0f')} | {risk_cells(row)} | {pct(row['distinct_first_room'])} | "
                f"{fmt(row['diverge_first_room'], '.0f')} |")
        lines.append('')
    return '\n'.join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--dir', type=Path, default=Path('target/r23-pilot'))
    parser.add_argument('--modes', default=None, help="comma-separated (default: every rooms/ folder)")
    parser.add_argument('--models-dir', default=None)
    parser.add_argument('--json', type=Path, default=None, help="also write the numbers here")
    args = parser.parse_args(argv)
    if args.modes:
        modes = [registry.parse_act(m) for m in args.modes.split(',') if m]
    else:
        folder = args.dir / 'rooms'
        modes = sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
        chain_modes = {p.name[len('r23-'):].rsplit('-', 1)[0] for p in (args.dir / 'chains').glob('r23-*.jsonl')} \
            if (args.dir / 'chains').is_dir() else set()
        modes += sorted(m for m in chain_modes if m not in modes and m != 'model')
    rooms = read_rooms(args.dir, modes)
    add_baseline(rooms, own_modes(args.models_dir))
    chains = read_chains(args.dir, modes)
    print(report(rooms, chains, modes))
    if args.json:
        args.json.write_text(json.dumps({'rooms': rooms, 'chains': chains}, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
