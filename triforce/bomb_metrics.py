"""The owner's two bomb metrics (Nov 23; docs/architecture.md "Bomb windows", crates/zelda/src/bomb_windows.rs): a
calculation over replays, nothing more. Each replay gives its list of **windows** (``_native.replay_bomb_windows``:
the RAM frame by frame, the cheated bombs from its EDITS, and the facts its targets stream records); every count is a
sum over those lists, per run, per room and per batch, always with the full-bag split.

    .venv/bin/python -m triforce.bomb_metrics REPLAY [REPLAY ...] [--out bomb_windows.jsonl] [--json SUMMARY]
    .venv/bin/python -m triforce.bomb_metrics --windows bomb_windows.jsonl [--json SUMMARY] [--rooms 12]

A window: ``metric`` (``tenth``: A, the 10th-kill bomb; ``carrier``: B, the bomb carrier), ``open_frame``,
``open_room``, ``close_frame``, ``close_room`` (credit goes to the room where it closed), ``outcome``, ``killed_type``
and ``killed_slot`` (what closed it), ``full_bag``, ``drop`` (the bombs ``armed`` and ``hit`` brought), and the run's
``seed`` and ``replay``.

**Metric A** opportunities: ``armed`` (the 10th kill by a bomb) + ``missed`` (a bomb of the window ended killing
nothing and opening no wall, or was in the air when Link left) + ``other`` (the 10th kill by another weapon, whatever
it killed). Its full bag is the bag on the frame before the resolving action (the bomb placed, the other weapon's kill
landing): with a full bag a bomb 10th kill goes 8 → 7 → 8, nothing gained and the 5 rupees lost.
**Metric B** opportunities: ``hit`` (the next death a carrier's) + ``miss`` (another's) + ``left`` (Link left with an
eligible carrier alive); its full bag is judged at the close.
Not opportunities, reported beside: ``lost`` (harm first), ``paused-out``, ``closed`` (B's conditions ended),
``suspended`` (Dodongo's room, outside both metrics), ``open-at-end``.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from . import _native

TENTH = ('armed', 'missed', 'other')
CARRIER = ('hit', 'miss', 'left')
UNCOUNTED = ('lost', 'paused-out', 'closed', 'suspended', 'open-at-end')
RULE_REASONS = ('never', 'tenth', 'required', 'reserve')     # a BOMB block by the bomb rule (not the game's own)


def replay_facts(replay) -> list:
    """The env's per-decision facts from a replay's targets stream (``_native.Replay``: ``targets()``, the objective
    when it changed; ``masks()``, the action mask and the rules that blocked kinds, when it changed):
    ``[(frame, kill_all, bomb_rule, bomb_never)]`` at every frame either changed. ``bomb_rule``: no bomb-rule reason
    blocks BOMB (off only for the game's own limits, a bomb in the air, an empty bag, still counts as allowed);
    ``bomb_never``: the mode is ``never``. A replay without the stream has no facts: Metric A then opens on RAM alone,
    Metric B reads BOMB off and no FIGHT."""
    def bomb(m):
        reasons = m['blocked'].get('BOMB', ())
        return (not any(r in RULE_REASONS for r in reasons), 'never' in reasons)
    targets = replay.targets() if hasattr(replay, 'targets') else []
    masks = replay.masks() if hasattr(replay, 'masks') else []
    changes = sorted([(t['frame'], 0, t['objective'] == 'FIGHT') for t in targets]
                     + [(m['frame'], 1, bomb(m)) for m in masks], key=lambda c: (c[0], c[1]))
    facts, current = [], [False, (False, False)]
    for frame, which, value in changes:
        current[which] = value
        row = (frame, current[0], *current[1])
        if facts and facts[-1][0] == frame:
            facts[-1] = row
        else:
            facts.append(row)
    return facts


def score_replay(path, states_dir=None) -> list:
    """One replay's windows, each with the run's ``seed`` and ``replay`` file."""
    data = Path(path).read_bytes()
    replay = _native.Replay(data)
    seed = replay.meta.get('seed')
    return [{**w, 'seed': seed, 'replay': str(path)}
            for w in _native.replay_bomb_windows(data, replay_facts(replay), states_dir)]


def counts(windows) -> dict:
    """Metric A's and B's counts over ``windows``, all of them and without the full-bag ones."""
    def tally(metric, outcomes, full=None):
        c = Counter(w['outcome'] for w in windows if w['metric'] == metric and (full is None or w['full_bag'] == full))
        return {'opportunities': sum(c[o] for o in outcomes), **{o: c[o] for o in outcomes},
                **{o: c[o] for o in UNCOUNTED if c[o]}}
    drops = Counter((w['outcome'], w['drop']) for w in windows if w.get('drop'))
    return {
        'tenth': {**tally('tenth', TENTH), 'not_full': tally('tenth', TENTH, False),
                  'forced': {f: drops['armed', f] for f in ('pending', 'fell', 'taken', 'wasted', 'lost')}},
        'carrier': {**tally('carrier', CARRIER), 'not_full': tally('carrier', CARRIER, False),
                    'bombs': {f: drops['hit', f] for f in ('none', 'fell', 'taken', 'wasted')},
                    'unattributed': sum(w['metric'] == 'carrier' and w['outcome'] in ('hit', 'miss')
                                        and w['killed_type'] is None for w in windows)},
    }


def summarize(windows: list) -> dict:
    """The batch's counts, per run and per room (``close_room``; most opportunities first)."""
    by_run, by_room = defaultdict(list), defaultdict(list)
    for w in windows:
        by_run[w['seed']].append(w)
        by_room[w['close_room']].append(w)
    rooms = []
    for room, ws in by_room.items():
        c = counts(ws)
        a, b = c['tenth'], c['carrier']
        if not (a['opportunities'] or b['opportunities']):
            continue
        rooms.append([room, a['opportunities'], a['armed'], a['missed'], a['other'],
                      a['opportunities'] - a['not_full']['opportunities'], b['opportunities'], b['hit'], b['miss'],
                      b['left']])
    rooms.sort(key=lambda r: (-(r[1] + r[6]), r[0]))
    return {'runs': len(by_run), 'windows': len(windows), **counts(windows),
            'per_run': {str(seed): counts(ws) for seed, ws in sorted(by_run.items(), key=lambda kv: str(kv[0]))},
            'room_fields': ['room', 'tenth_opportunities', 'armed', 'missed', 'other', 'tenth_full_bag',
                            'carrier_opportunities', 'hit', 'miss', 'left'],
            'rooms': rooms}


def share(n: int, total: int) -> str:
    return f"{n} ({100 * n / total:.1f}%)" if total else str(n)


def markdown(s: dict, top_rooms: int = 12) -> str:
    """A check-in's tables: both metrics, all windows and without a full bag, and the rooms with most chances."""
    lines = [f"Bomb metrics over {s['runs']} runs (the owner, Nov 23; {s['windows']} windows):", '']
    for title, key, outcomes in (('Metric A: the 10th-kill bomb', 'tenth', TENTH),
                                 ('Metric B: the bomb carrier', 'carrier', CARRIER)):
        m, nf = s[key], s[key]['not_full']
        lines += [f'| {title} | all windows | without a full bag |', '|---|---|---|',
                  f"| opportunities | {m['opportunities']} | {nf['opportunities']} |"]
        lines += [f"| {o} | {share(m[o], m['opportunities'])} | {share(nf[o], nf['opportunities'])} |"
                  for o in outcomes]
        uncounted = ', '.join(f"{m[o]} {o}" for o in UNCOUNTED if m.get(o))
        lines += ['', f"Not counted: {uncounted or 'none'}.", '']
    lines += [f"Forced bomb drops after `armed`: {s['tenth']['forced']}. Bombs the carrier hits dropped: "
              f"{s['carrier']['bombs']}; unattributed: {s['carrier']['unattributed']}.", '',
              f"Rooms with the most opportunities (top {top_rooms}; the room each window closed in):", '',
              '| room | A opp | armed | missed | other | A full | B opp | hit | miss | left |',
              '|---|---|---|---|---|---|---|---|---|---|']
    lines += ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in s['rooms'][:top_rooms]]
    return '\n'.join(lines) + '\n'


def write_windows(path, windows: list):
    Path(path).write_text(''.join(json.dumps(w) + '\n' for w in windows))


def read_windows(path) -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('replays', nargs='*', help="replay files to score")
    parser.add_argument('--windows', default=None, help="summarize this bomb_windows.jsonl instead of scoring replays")
    parser.add_argument('--out', default=None, help="write the windows here (bomb_windows.jsonl)")
    parser.add_argument('--json', default=None, help="write the summary here")
    parser.add_argument('--rooms', type=int, default=12)
    args = parser.parse_args(argv)
    if args.windows:
        windows = read_windows(args.windows)
    else:
        windows = [w for path in args.replays for w in score_replay(path)]
    if args.out:
        write_windows(args.out, windows)
    summary = summarize(windows)
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1) + '\n')
    sys.stdout.write(markdown(summary, args.rooms))
    return 0


if __name__ == '__main__':
    sys.exit(main())
