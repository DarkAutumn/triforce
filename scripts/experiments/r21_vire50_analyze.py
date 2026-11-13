"""Round 21 (docs/experiments/round21-vire-50.md): what a trained level 4 room 0x50 arm did.

    python scripts/experiments/r21_vire50_analyze.py RUN_DIR            # -> RUN_DIR/analysis.json, summary on stdout
    python scripts/experiments/r21_vire50_analyze.py --table [MODELS_DIR] # markdown tables of every analysis.json

Reads ``RUN_DIR/metrics.jsonl`` (learning) and, per act mode, ``eval-<act>.json`` with the episode replays of
``replays-<act>/`` (``triforce evaluate --replays DIR --streams decisions,rewards``; file ``<job>-<ending>`` pairs
with the record whose ``seed`` is the job).  Every replay is played on the emulator to Link's position, the
monsters' positions and the room's kill count at each decision.

A completed block clip leaves Link exactly where no walk can: 8 px into one of the room's 20 clip squares
(``LANDINGS``, measured by putting Link on each of the 60 offered grid points of 4_50e_0 with the monsters
removed and running ``CLIP N``/``CLIP S``).  So a clip is a ``CLIP``/``MOVE`` N or S that ends on a landing from
that square's side (``landing``); the number placed is capped by the executor's own count (``clips.done``).
"""

import json
import sys
from collections import Counter
from pathlib import Path

from triforce._native import Replay, ReplayPlayer

# (heading, X) -> landing Ys: a clip N (S) on column X ends 8 px into the block above (below), on (X, landing).
# Offered from the three grid points 8, 16 and 24 px below (above) the landing.
LANDINGS = {
    ('N', 0x40): (0x5d, 0x9d), ('N', 0x50): (0x8d,), ('N', 0x60): (0x7d,), ('N', 0x70): (0x6d,),
    ('N', 0x90): (0x9d,), ('N', 0xa0): (0x8d,), ('N', 0xb0): (0x7d,), ('N', 0xc0): (0x6d,), ('N', 0xd0): (0x5d,),
    ('S', 0x20): (0xb5,), ('S', 0x30): (0xa5,), ('S', 0x40): (0x95,), ('S', 0x50): (0x85,), ('S', 0x60): (0x75,),
    ('S', 0x80): (0xa5,), ('S', 0x90): (0x95,), ('S', 0xa0): (0x85,), ('S', 0xb0): (0x75, 0xb5),
}
WAY_ON = {'N': 'W', 'S': 'E'}       # the sideways walk that goes on from a clip square (ClipMerge's scan)
OBJ_TYPE, OBJ_X, OBJ_Y = 0x34F, 0x70, 0x84
ROOM_KILLS = 0x34F                  # slot 0 of ObjType is RoomKillCount
NEAR = 0x18                         # a monster within 24 px (both axes) of Link counts as "near"
SOUTH_Y = 0xa5                      # the plain zigzag goes S along the east side to Y $AD before any W
ATTACKS = ('SWORD', 'BEAMS', 'BOMB', 'BOOMERANG', 'ARROW', 'CANDLE', 'WAND')


def landing(heading: str, x: int, y0: int, y1: int):
    """The landing Y if a decision from (x, y0) that ended on (x, y1) is a clip.  The skip is also offered between
    the grid points (the macro's run-up walks to one first), so the start only has to lie on the clip's side of
    the square, within its 0x20-pixel zone; the end has to be the landing itself (8 px into a block: no walk ends
    there)."""
    for land in LANDINGS.get((heading, x), ()):
        if y1 == land and (land < y0 <= land + 0x20 if heading == 'N' else land - 0x20 <= y0 < land):
            return land
    return None


def square(heading: str, x: int, land: int) -> str:
    return f"{heading}({x:02x},{land:02x})"


def action_name(meta: dict, indices) -> str:
    actions = meta['actions']
    kind = actions['kinds'][indices[0]]
    if kind == 'WAIT':
        return f"WAIT {actions['durations'][indices[2]]}"
    text = f"{kind} {actions['directions'][indices[1]]}"
    return text + (f" {actions['durations'][indices[2]]}" if kind == 'MOVE' else '')


def monsters(ram) -> list:
    """Live monsters (types $01-$48) as (slot, type, x, y)."""
    return [(s, int(ram[OBJ_TYPE + s]), int(ram[OBJ_X + s]), int(ram[OBJ_Y + s]))
            for s in range(1, 12) if 0x01 <= int(ram[OBJ_TYPE + s]) <= 0x48]


def episode(path: Path, record: dict) -> dict:
    data = path.read_bytes()
    replay = Replay(data)
    meta = replay.meta
    decisions = replay.decisions()
    rewards = replay.rewards()
    hit_frames = [frame for frame, term, _ in rewards if term == 'penalty-lost-health']
    player = ReplayPlayer(data, None)
    frames = [int(f) for f in decisions['frame']] + [replay.frames]
    positions, near, kills, types = [], [], [], Counter()
    for frame in frames:
        if frame > player.frame:
            player.step(frame - player.frame, False)
        ram = player.ram()
        x, y = int(ram[0x70]), int(ram[0x84])
        positions.append((x, y))
        live = monsters(ram)
        types.update(t for _, t, _, _ in live)
        near.append(any(abs(mx - x) <= NEAR and abs(my - y) <= NEAR for _, _, mx, my in live))
        kills.append(int(ram[ROOM_KILLS]))
    actions = [action_name(meta, a) for a in decisions['action']]
    clips = []
    for i, action in enumerate(actions):
        parts = action.split()
        if parts[0] not in ('CLIP', 'MOVE') or parts[1] not in ('N', 'S'):
            continue
        (x0, y0), (x1, y1) = positions[i], positions[i + 1]
        land = landing(parts[1], x1, y0, y1) if abs(x1 - x0) < 8 else None
        if land is None:
            continue
        # What came next: the first later decision that moved Link.
        following = next(((positions[j + 1], actions[j]) for j in range(i + 1, len(actions))
                          if positions[j + 1] != (x1, y1)), None)
        if following is None:
            went = 'stayed'
        else:
            (x2, y2), _ = following
            sideways = 'W' if x2 < x1 else 'E' if x2 > x1 else None
            went = ('on' if sideways == WAY_ON[parts[1]] else
                    'back' if sideways is None and (y2 > y1) == (parts[1] == 'N') else
                    'reverse' if sideways else 'further')
        clips.append({'step': i, 'square': square(parts[1], x1, land), 'from': (x0, y0), 'action': action,
                      'next': actions[i + 1] if i + 1 < len(actions) else None, 'went': went})
    unconfirmed = clips[record['clips']['done']:]
    clips = clips[:record['clips']['done']]
    # Time near a monster: frames of the decisions that began with one within NEAR px, over all but the last
    # decision (the exit's walk and scroll).
    durations = [b - a for a, b in zip(frames, frames[1:])][:-1]
    total = sum(durations) or 1
    near_frames = sum(d for d, n in zip(durations, near) if n)
    attacks = sum(a.split()[0] in ATTACKS for a in actions)
    south = any(y >= SOUTH_Y and x >= 0x80 for x, y in positions)
    # Where each hit landed: the decision it fell in, Link's position when that decision began, and whether it came
    # before, during (the clip's own decision) or after the episode's first clip.
    hits_at = []
    for f in hit_frames:
        i = max(0, next((k for k in range(len(frames) - 1) if frames[k] <= f < frames[k + 1]), len(frames) - 2))
        phase = ('no clip' if not clips else 'before clip' if i < clips[0]['step'] else
                 'clip' if i == clips[0]['step'] else 'after clip')
        x, y = positions[i]
        hits_at.append({'phase': phase, 'spot': f'({x:02x},{y:02x})', 'action': actions[i]})
    if not record['success']:
        way = 'failed'
    elif clips:
        way = 'clip'
    else:
        way = 'plain'
    return {'job': int(path.name[:8]), 'state': record['start_state'], 'success': record['success'],
            'ending': record['ending'], 'steps': record['steps'], 'frames': record['frames'],
            'start_health': record['start_health'], 'full': record['start_health'] >= record['start_max_health'],
            'health_change': record['health'] - record['start_health'],
            'hits': len(hit_frames), 'kills': max(kills), 'monster_types': sorted(types),
            'clips_done': record['clips']['done'], 'clips_aborted': record['clips']['aborted'],
            'clips': clips, 'unconfirmed': len(unconfirmed), 'way': way, 'south': south,
            'near_share': near_frames / total, 'near_frames': near_frames, 'attacks': attacks, 'hits_at': hits_at,
            'actions': actions, 'positions': positions}


WINDOW = 10        # iterations: one holds only 10-80 room-entry episodes, so learning reads a rolling window


def learning(run_dir: Path) -> dict:
    """Learning from metrics.jsonl.  Each value is the mean over the last WINDOW iterations, weighted by their
    room-entry episodes (``success_rate``, ``clips/done`` and ``health_change`` count those only; curriculum
    starts are left out)."""
    rows = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    raw, steps = [], 0
    for row in rows:
        steps = row.get('steps', steps + int(row.get('rollout/steps', 0)))
        raw.append((steps, row.get('episodes', 0), row.get('success_rate', 0.0), row.get('clips/done', 0.0),
                    row.get('health_change', 0.0)))
    curve = []
    for i, (s, *_) in enumerate(raw):
        window = raw[max(0, i - WINDOW + 1):i + 1]
        eps = sum(e for _, e, _, _, _ in window) or 1
        curve.append((s, *(sum(e * v[k] for _, e, *v in window) / eps for k in range(3))))
    full = curve[WINDOW - 1:]
    out = {'iterations': len(rows), 'steps': steps}
    for level in (0.5, 0.8, 0.9, 0.95):
        out[f'steps_to_{int(level * 100)}'] = next((s for s, rate, _, _ in full if rate >= level), None)
    out['at'] = {f'{k // 1000}k': round(next((r for s, r, _, _ in curve if s >= k), curve[-1][1]), 3)
                 for k in (250_000, 500_000, 1_000_000)}
    out['clips_first_iteration'] = raw[0][3] if raw else None
    out['first_clip_steps'] = next((s for s, _, _, clips, _ in raw if clips >= 0.05), None)
    peak = max(curve, key=lambda row: row[2])
    out['clip_peak'] = {'steps': peak[0], 'clips': peak[2]}
    # Where clip use settles: the first iteration from which the windowed clips per episode stay within 0.1 of
    # the last window's.
    final = curve[-1]
    out['settled_steps'] = next((s for i, (s, _, _, _) in enumerate(curve)
                                 if all(abs(c - final[2]) <= 0.1 for _, _, c, _ in curve[i:])), None)
    out['last10'] = {'success': final[1], 'clips': final[2], 'health_change': final[3]}
    out['curve'] = [(s, round(r, 3), round(c, 3), round(h, 3)) for s, r, c, h in curve]
    return out


def summary(episodes: list, evaluation: dict) -> dict:
    n = len(episodes)
    wins = [e for e in episodes if e['success']]
    clips = [c for e in episodes for c in e['clips']]
    # "Extra" clips: an episode's second and later clips (one clip is all the route needs).
    extra = sum(max(0, len(e['clips']) - 1) for e in episodes)
    return {
        'episodes': n, 'success': evaluation['success_rate'], 'health_change': evaluation['mean_health_change'],
        'steps': sum(e['steps'] for e in episodes) / n, 'frames': sum(e['frames'] for e in episodes) / n,
        'win_frames': sum(e['frames'] for e in wins) / len(wins) if wins else None,
        'clips_done': sum(e['clips_done'] for e in episodes) / n,
        'clip_share': sum(e['clips_done'] > 0 for e in episodes) / n,
        'clips_aborted': sum(e['clips_aborted'] for e in episodes) / n,
        'kills': sum(e['kills'] for e in episodes) / n,
        'attacks': sum(e['attacks'] for e in episodes) / n,
        'bombs': evaluation['mean_bombs_used'],
        'wins_by_clip': sum(e['way'] == 'clip' for e in wins) / len(wins) if wins else None,
        'south': sum(e['south'] for e in episodes) / n,
        'hit_episodes': sum(e['hits'] > 0 for e in episodes) / n,
        'hits': sum(e['hits'] for e in episodes) / n,
        'hit_phases': dict(Counter(h['phase'] for e in episodes for h in e['hits_at'])),
        'hit_spots': dict(Counter(h['spot'] for e in episodes for h in e['hits_at'])),
        'success_full': _rate([e for e in episodes if e['full']]),
        'success_partial': _rate([e for e in episodes if not e['full']]),
        'near_share': sum(e['near_share'] for e in episodes) / n,
        'near_frames': sum(e['near_frames'] for e in episodes) / n,
        'endings': dict(Counter(e['ending'] for e in episodes)),
        'clip_squares': dict(Counter(f"{c['square']} {c['action']}" for c in clips)),
        'clip_from': dict(Counter(f"({c['from'][0]:02x},{c['from'][1]:02x})" for c in clips)),
        'clip_went': dict(Counter(c['went'] for c in clips)),
        'clips_extra': extra,
        'unconfirmed_clips': sum(e['unconfirmed'] for e in episodes),
        'clips_placed': len(clips), 'clips_counted': sum(e['clips_done'] for e in episodes),
        'monster_types': sorted({t for e in episodes for t in e['monster_types']}),
        'by_state_health': {f"{s} {'full' if full else 'part'}": round(sum(e['success'] for e in group) / len(group), 3)
                            for (s, full), group in _groups(episodes).items()},
    }


def _rate(episodes):
    return sum(e['success'] for e in episodes) / len(episodes) if episodes else None


def _groups(episodes):
    groups = {}
    for e in episodes:
        groups.setdefault((e['state'], e['full']), []).append(e)
    return dict(sorted(groups.items()))


def _pct(value) -> str:
    return '-' if value is None else f'{100 * value:.0f}%'


def _k(value) -> str:
    return 'not reached' if value is None else f'{value / 1000:.0f}k'


def _runs(models: Path):
    for path in sorted(models.glob('vire50-*/r21-s*/analysis.json')):
        yield (path.parent.parent.name.removeprefix('vire50-'), path.parent.name.removeprefix('r21-s'),
               json.loads(path.read_text()))


def table(models: Path) -> int:
    """The report's tables, one row per arm and seed (and act mode)."""
    runs = list(_runs(models))
    top = lambda d, n=3: ', '.join(f'{k} {v}' for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:n]) or '-'
    print('| arm | seed | act | success | hearts | steps | frames | win frames | clips / ep | ep. with clip |'
          ' clips aborted | kills / ep | attacks / ep (bombs) | endings |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for arm, seed, result in runs:
        for act in ('sample', 'argmax'):
            if act not in result:
                continue
            m = result[act]['summary']
            win = '-' if m['win_frames'] is None else f"{m['win_frames']:.0f}"
            print(f"| {arm} | {seed} | {act} | {_pct(m['success'])} | {m['health_change']:+.2f} | {m['steps']:.0f} |"
                  f" {m['frames']:.0f} | {win} | {m['clips_done']:.2f} | {_pct(m['clip_share'])} |"
                  f" {m['clips_aborted']:.2f} | {m['kills']:.2f} | {m['attacks']:.1f} ({m['bombs']:.2f}) |"
                  f" {top(m['endings'], 4)} |")
    print()
    print('| arm | seed | 50% | 80% | 90% | 95% | success at 250k / 500k / 1M | clips / ep at it. 1 |'
          ' clip peak | clip use settles | last 10 it. success / clips / hearts |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for arm, seed, result in runs:
        l = result['learning']
        t = l['last10']
        clip_arm = not arm.startswith('none')
        peak = f"{l['clip_peak']['clips']:.1f} at {_k(l['clip_peak']['steps'])}" if clip_arm else '-'
        print(f"| {arm} | {seed} | {_k(l['steps_to_50'])} | {_k(l['steps_to_80'])} | {_k(l['steps_to_90'])} |"
              f" {_k(l['steps_to_95'])} | {' / '.join(_pct(v) for v in l['at'].values())} |"
              f" {l['clips_first_iteration']:.2f} | {peak} | {_k(l['settled_steps']) if clip_arm else '-'} |"
              f" {_pct(t['success'])} / {t['clips']:.2f} / {t['health_change']:+.2f} |")
    print()
    print('| arm | seed | act | clipped square (action) | from | after the clip | extra clips | wins by clip |'
          ' went S (zigzag) | hit episodes | frames near a monster (share) |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for arm, seed, result in runs:
        for act in ('sample', 'argmax'):
            if act not in result:
                continue
            m = result[act]['summary']
            print(f"| {arm} | {seed} | {act} | {top(m['clip_squares'])} | {top(m['clip_from'])} |"
                  f" {top(m['clip_went'])} | {m['clips_extra']} | {_pct(m['wins_by_clip'])} | {_pct(m['south'])} |"
                  f" {_pct(m['hit_episodes'])} | {m['near_frames']:.0f} ({_pct(m['near_share'])}) |")
    print()
    print('| arm | seed | act | hits / ep | hits before / during / after the clip (no clip) | where (Link at the'
          ' decision) | success full / partial health |')
    print('|---|---|---|---|---|---|---|')
    for arm, seed, result in runs:
        for act in ('sample', 'argmax'):
            if act not in result:
                continue
            m = result[act]['summary']
            p = m['hit_phases']
            phases = (f"{p.get('no clip', 0)}" if arm.startswith('none') else
                      f"{p.get('before clip', 0)} / {p.get('clip', 0)} / {p.get('after clip', 0)}"
                      f" ({p.get('no clip', 0)})")
            print(f"| {arm} | {seed} | {act} | {m['hits']:.2f} | {phases} | {top(m['hit_spots'], 4)} |"
                  f" {_pct(m['success_full'])} / {_pct(m['success_partial'])} |")
    return 0


def main(argv) -> int:
    if argv and argv[0] == '--table':
        return table(Path(argv[1] if len(argv) > 1 else 'models'))
    run_dir = Path(argv[0])
    result = {'run': str(run_dir), 'learning': learning(run_dir)}
    for act in ('sample', 'argmax'):
        eval_path = run_dir / f'eval-{act}.json'
        if not eval_path.exists():
            continue
        evaluation = json.loads(eval_path.read_text())
        records = {r['seed'] - evaluation['seed']: r for r in evaluation['records']}
        episodes = [episode(path, records[int(path.name[:8])])
                    for path in sorted((run_dir / f'replays-{act}').glob('*.tfreplay'))]
        result[act] = {'summary': summary(episodes, evaluation), 'episodes': episodes}
    (run_dir / 'analysis.json').write_text(json.dumps(result))
    print(json.dumps({'learning': {k: v for k, v in result['learning'].items() if k != 'curve'},
                      **{act: result[act]['summary'] for act in ('sample', 'argmax') if act in result}}, indent=1))
    # About 20 episodes per act mode (every 15th of 300) with their actions, for reading the behaviour.
    for act in ('sample', 'argmax'):
        for e in result.get(act, {}).get('episodes', [])[::15]:
            clips = '; '.join(f"{c['action']} into {c['square']} from ({c['from'][0]:02x},{c['from'][1]:02x}), "
                              f"{c['went']}" for c in e['clips'])
            print(f"{act} #{e['job']:03} {e['state']} hp {e['start_health']:.2f} {e['ending']:13} {e['way']:6} "
                  f"{e['steps']:3} steps {e['frames']:5} frames hits {e['hits']} kills {e['kills']} "
                  f"near {e['near_share']:.0%} south {int(e['south'])} | clips: {clips or '-'}")
            print(f"    {' | '.join(e['actions'][:40])}{' ...' if len(e['actions']) > 40 else ''}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
