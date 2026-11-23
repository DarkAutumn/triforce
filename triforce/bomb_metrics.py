"""The owner's two bomb metrics (Nov 23; docs/architecture.md "Bomb windows", crates/zelda/src/bomb_windows.rs) summed
over chain records: per run, per batch, and per room, so the rooms where the chances arise stand out.

    .venv/bin/python -m triforce.bomb_metrics RUNS.jsonl [RUNS.jsonl ...] [--rooms 12] [--json OUT]
    .venv/bin/python -m triforce.bomb_metrics --replays REPLAY [REPLAY ...] [--rooms 12] [--json OUT]

Chain jsonl carries the counts made live; ``--replays`` scores replays offline (``score_replay``: the same Rust
counter over the replay's frames, its ROOM events and the facts of its targets stream), the way an old version's
runs are scored by today's rules.

**Metric A, the 10th-kill bomb.** Opportunities = ``tenth_armed`` (the 10th kill by a bomb) + ``tenth_missed`` (a bomb
placed in the window that ended killing nothing and opening no wall) + ``tenth_other`` (the 10th kill by another
weapon); "bomb" = armed + missed. Every count is reported for all windows and without the full-bag ones (``*_full_*``:
the window opened with the bag full), so the comparison stays like for like once the observation stops lighting
with a full bag. Also: the forced bomb drops that fell, were taken, wasted (expired or left), lost before falling.

**Metric B, the bomb carrier.** Opportunities = ``carrier_hits`` (the next death a carrier's) + ``carrier_miss_kill``
(another monster's) + ``carrier_miss_leave`` (Link left first); the bombs the hits dropped (the 41% roll) and those
taken.

Not opportunities, reported beside: Metric A's streaks lost (``tenth_lost``) and paused windows ended without a bomb
(``tenth_paused_out``), Metric B's windows whose conditions ended (``carrier_closed``), cycle moves no slot accounted
for (``carrier_unattributed``, expected 0), and the windows open when the run ended (``open_at_end``).
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from . import _native

FIELDS = tuple(_native.BOMB_WINDOW_FIELDS)


def room_key(room: dict) -> str:
    """'3:0x5b', '0:0x77:cave'."""
    return f"{room['level']}:{room['room']}" + (':cave' if room.get('in_cave') else '')


def totals(rooms) -> dict:
    """The fields summed over room records (None, an older record's, counts 0)."""
    out = dict.fromkeys(FIELDS, 0)
    for room in rooms:
        for name in FIELDS:
            out[name] += room.get(name) or 0
    return out


def derived(t: dict) -> dict:
    """Metric A's and B's headline numbers from summed fields."""
    a_all = t['tenth_armed'] + t['tenth_missed'] + t['tenth_other']
    a_full = t['tenth_full_armed'] + t['tenth_full_missed'] + t['tenth_full_other']
    return {
        'tenth': {
            'opportunities': a_all, 'armed': t['tenth_armed'], 'missed': t['tenth_missed'], 'other': t['tenth_other'],
            'full_bag': a_full,
            'not_full': {'opportunities': a_all - a_full, 'armed': t['tenth_armed'] - t['tenth_full_armed'],
                         'missed': t['tenth_missed'] - t['tenth_full_missed'],
                         'other': t['tenth_other'] - t['tenth_full_other']},
            'lost': t['tenth_lost'], 'paused_out': t['tenth_paused_out'],
            'forced': {'fell': t['tenth_forced_fell'], 'taken': t['tenth_forced_taken'],
                       'wasted': t['tenth_forced_wasted'], 'lost': t['tenth_forced_lost']},
        },
        'carrier': {
            'opportunities': t['carrier_hits'] + t['carrier_miss_kill'] + t['carrier_miss_leave'],
            'hits': t['carrier_hits'], 'miss_kill': t['carrier_miss_kill'], 'miss_leave': t['carrier_miss_leave'],
            'bombs': t['carrier_bombs'], 'bombs_taken': t['carrier_bombs_taken'], 'closed': t['carrier_closed'],
            'unattributed': t['carrier_unattributed'],
        },
    }


def summarize(records: list) -> dict:
    """The batch's metrics: totals, per run (``run_fields`` rows) and per room (``room_fields`` rows, most
    opportunities first). Records without the fields (older than Nov 23) are counted as runs with none."""
    batch = totals(room for record in records for room in record.get('rooms') or [])
    by_room = defaultdict(lambda: dict.fromkeys(FIELDS, 0))
    families = defaultdict(Counter)
    for record in records:
        for room in record.get('rooms') or []:
            key = room_key(room)
            for name in FIELDS:
                by_room[key][name] += room.get(name) or 0
            families[key][room.get('specialist')] += 1
    runs = []
    for record in records:
        d = derived(totals(record.get('rooms') or []))
        a, b = d['tenth'], d['carrier']
        runs.append([record.get('seed'), a['opportunities'], a['armed'] + a['missed'], a['full_bag'],
                     b['opportunities'], b['hits'], b['bombs']])
    rooms = []
    for key, t in by_room.items():
        d = derived(t)
        a, b = d['tenth'], d['carrier']
        if a['opportunities'] or b['opportunities']:
            rooms.append([key, families[key].most_common(1)[0][0], a['opportunities'], a['armed'], a['missed'],
                          a['other'], a['full_bag'], b['opportunities'], b['hits'], b['miss_kill'], b['miss_leave'],
                          b['bombs']])
    rooms.sort(key=lambda r: (-(r[2] + r[7]), r[0]))
    return {
        'runs': len(records),
        **derived(batch),
        'open_at_end': {'tenth': sum(bool(r.get('tenth_open_at_end')) for r in records),
                        'carrier': sum(bool(r.get('carrier_open_at_end')) for r in records)},
        'run_fields': ['seed', 'tenth_opportunities', 'tenth_bomb', 'tenth_full_bag', 'carrier_opportunities',
                       'carrier_hits', 'carrier_bombs'],
        'per_run': runs,
        'room_fields': ['room', 'family', 'tenth_opportunities', 'tenth_armed', 'tenth_missed', 'tenth_other',
                        'tenth_full_bag', 'carrier_opportunities', 'carrier_hits', 'carrier_miss_kill',
                        'carrier_miss_leave', 'carrier_bombs'],
        'rooms': rooms,
    }


def share(n: int, total: int) -> str:
    return f"{n} ({100 * n / total:.1f}%)" if total else str(n)


def markdown(s: dict, top_rooms: int = 12) -> str:
    """A check-in's table of the batch's metrics, with and without the full-bag windows, and the top rooms."""
    a, b, n = s['tenth'], s['carrier'], s['runs']
    nf = a['not_full']
    lines = [f"Bomb metrics over {n} runs (the owner, Nov 23):", '',
             '| Metric A: the 10th-kill bomb | all windows | without a full bag |', '|---|---|---|']
    for label, key in (('opportunities', 'opportunities'), ('bomb, armed', 'armed'), ('bomb, missed', 'missed'),
                       ('other weapon', 'other')):
        all_n, nf_n = a[key], nf[key]
        lines.append(f"| {label} | {share(all_n, a['opportunities']) if key != 'opportunities' else all_n} | "
                     f"{share(nf_n, nf['opportunities']) if key != 'opportunities' else nf_n} |")
    lines += [f"| full-bag windows | {share(a['full_bag'], a['opportunities'])} | |", '',
              f"Not counted: {a['lost']} streaks lost before the 10th kill, {a['paused_out']} paused windows ended "
              f"without a bomb, {s['open_at_end']['tenth']} open at the run's end. Forced bomb drops: "
              f"{a['forced']['fell']} fell, {a['forced']['taken']} taken, {a['forced']['wasted']} wasted, "
              f"{a['forced']['lost']} lost before falling.", '',
              '| Metric B: the bomb carrier | count |', '|---|---|',
              f"| opportunities | {b['opportunities']} |",
              f"| hits (a carrier died next) | {share(b['hits'], b['opportunities'])} |",
              f"| misses by kill | {share(b['miss_kill'], b['opportunities'])} |",
              f"| misses by leaving | {share(b['miss_leave'], b['opportunities'])} |",
              f"| bombs dropped by hits (taken) | {b['bombs']} ({b['bombs_taken']}) |", '',
              f"Not counted: {b['closed']} windows closed by their conditions, {s['open_at_end']['carrier']} open at "
              f"the run's end, {b['unattributed']} cycle moves unattributed.", '',
              f"Rooms with the most opportunities (top {top_rooms}):", '',
              '| room | family | A opp | A armed | A missed | A other | A full | B opp | B hits | B miss kill '
              '| B miss leave | B bombs |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    lines += ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in s['rooms'][:top_rooms]]
    return '\n'.join(lines) + '\n'


def read_records(paths) -> list:
    return [json.loads(line) for path in paths for line in Path(path).read_text().splitlines() if line.strip()]


RULE_REASONS = ('never', 'tenth', 'required', 'reserve')     # a BOMB block by the bomb rule (not the game's own)


def replay_facts(replay) -> list:
    """The env's per-decision facts from a replay's targets stream (``_native.Replay``: ``targets()``, the objective
    when it changed; ``masks()``, the action mask and the rules that blocked kinds, when it changed):
    ``[(frame, kill_all, bomb_rule, bomb_never)]`` at every frame either changed. ``bomb_rule``: no bomb-rule reason
    blocks BOMB (BOMB off only for the game's own limits, a bomb in the air, an empty bag, still counts as allowed: the
    live chain's ``BombUse::allows``); ``bomb_never``: the mode is ``never``. An older replay has neither: no facts
    (Metric A then opens on RAM alone, Metric B reads BOMB off and no FIGHT)."""
    def bomb(m):
        reasons = m['blocked'].get('BOMB', ())
        return (not any(r in RULE_REASONS for r in reasons), 'never' in reasons)
    changes = sorted([(t['frame'], 0, t['objective'] == 'FIGHT') for t in replay.targets()]
                     + [(m['frame'], 1, bomb(m)) for m in replay.masks()])
    facts, current = [], [False, (False, False)]
    for frame, which, value in changes:
        current[which] = value
        row = (frame, current[0], *current[1])
        if facts and facts[-1][0] == frame:
            facts[-1] = row
        else:
            facts.append(row)
    return facts


def score_replay(path, states_dir=None) -> dict:
    """One replay scored offline as a chain run record: ``{'seed', 'rooms': [...], 'tenth_open_at_end',
    'carrier_open_at_end', 'held'}`` (rooms in ROOM-event order, as the chain's records)."""
    data = Path(path).read_bytes()
    replay = _native.Replay(data)
    scored = _native.replay_bomb_windows(data, replay_facts(replay), states_dir)
    return {'seed': replay.meta.get('seed'), **scored}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('runs', nargs='+', help="chain jsonl files (with --replays: replay files)")
    parser.add_argument('--replays', action='store_true', help="score replays offline instead of reading jsonl")
    parser.add_argument('--rooms', type=int, default=12)
    parser.add_argument('--json', default=None, help="also write the summary here")
    args = parser.parse_args(argv)
    records = [score_replay(p) for p in args.runs] if args.replays else read_records(args.runs)
    summary = summarize(records)
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1) + '\n')
    sys.stdout.write(markdown(summary, args.rooms))
    return 0


if __name__ == '__main__':
    sys.exit(main())
