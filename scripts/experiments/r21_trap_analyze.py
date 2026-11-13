"""Round 21 (docs/experiments/round21-trap-room.md): what a trained level 1 room 0x22 arm did.

    TRIFORCE_STATES_DIR=experiments/r21/states python scripts/experiments/r21_trap_analyze.py RUN_DIR
    python scripts/experiments/r21_trap_analyze.py --table [MODELS_DIR]     # markdown tables of every analysis.json

Reads ``RUN_DIR/metrics.jsonl`` (learning) and, per act mode, ``eval-<act>.json`` with the episode replays of
``replays-<act>/`` (``triforce evaluate --replays DIR --streams decisions,rewards``; file ``<job>-<ending>`` pairs
with the record whose ``seed`` is the job).  Every replay is played on the emulator to Link's position at each
decision.  A completed clip leaves Link 8 px inside one of the diamond's eight block squares (no walk can do that;
the west block's square only counts before it was pushed), so each clip is placed: the block, where Link started
it, with which action, and what came next.  A push is the ``reward-pushed-block`` term.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from triforce._native import Replay, ReplayPlayer

# The diamond's block squares (Link coordinates of the square's top-left; game.yaml room 1:0x22); the stairs
# are at (0x80, 0x8d), every block 0x20 (Manhattan) from them.
BLOCKS = {'north': (0x80, 0x6d), 'upper-west': (0x70, 0x7d), 'upper-east': (0x90, 0x7d), 'west': (0x60, 0x8d),
          'east': (0xa0, 0x8d), 'lower-west': (0x70, 0x9d), 'lower-east': (0x90, 0x9d), 'south': (0x80, 0xad)}
PUSH_SLIDE_FRAMES = 64
PUSH_SPOTS = {(0x60, 0x95): 'below', (0x60, 0x85): 'above'}


def inside(x: int, y: int, pushed: bool):
    """The block Link stands 8 px into at (x, y), exactly where a clip leaves him (on the square's column; from
    below at the square's own Y, from above 8 px higher), or None.  Walking W/E can overlap a square by a few
    pixels and pushing the west block walks through its square: neither stops on those exact points."""
    for name, (bx, by) in BLOCKS.items():
        if name == 'west' and pushed:
            continue
        if x == bx and y in (by, by - 8):
            return name
    return None


def region(x: int, y: int) -> str:
    """Around the diamond: ``in`` within 0x20 of the stairs, else N/S/W/E sides (r21_trap_tracks.py)."""
    dx, dy = x - 0x80, y - 0x8d
    if abs(dx) + abs(dy) <= 0x20:
        return 'in'
    vertical = 'S' if dy > 0x08 else 'N' if dy < -0x08 else ''
    horizontal = 'W' if dx < -0x18 else 'E' if dx > 0x18 else ''
    return (vertical + horizontal) or '?'


def action_name(meta: dict, indices) -> str:
    actions = meta['actions']
    kind = actions['kinds'][indices[0]]
    if kind == 'WAIT':
        return f"WAIT {actions['durations'][indices[2]]}"
    text = f"{kind} {actions['directions'][indices[1]]}"
    return text + (f" {actions['durations'][indices[2]]}" if kind == 'MOVE' else '')


def episode(path: Path, record: dict) -> dict:
    data = path.read_bytes()
    replay = Replay(data)
    meta = replay.meta
    decisions = replay.decisions()
    rewards = replay.rewards()
    push_frames = [frame for frame, term, _ in rewards if term == 'reward-pushed-block']
    hit_frames = [frame for frame, term, _ in rewards if term == 'penalty-lost-health']
    player = ReplayPlayer(data, None)
    frames = [int(f) for f in decisions['frame']] + [replay.frames]
    positions = []
    for frame in frames:
        if frame > player.frame:
            player.step(frame - player.frame, False)
        ram = player.ram()
        positions.append((int(ram[0x70]), int(ram[0x84])))
    actions = [action_name(meta, a) for a in decisions['action']]
    push_frame = push_frames[0] if push_frames else None

    def pushed(frame: int) -> bool:
        # The term comes when the block has slid its 16 px; Link walks into its square while it slides.
        return push_frame is not None and frame >= push_frame - PUSH_SLIDE_FRAMES
    # Pushed W/E (no reward; pushed E it walls the stairs in): a horizontal move that ends in the west block's
    # square from outside it, before any N/S push.
    sideways = None
    for i, action in enumerate(actions):
        (x0, _), (x1, y1) = positions[i], positions[i + 1]
        if (sideways is None and not pushed(frames[i + 1]) and action.startswith(('MOVE W', 'MOVE E'))
                and 0x86 <= y1 <= 0x94 and 0x50 < x1 < 0x70 and not 0x50 < x0 < 0x70):
            sideways = (i, 'E' if x0 <= 0x50 else 'W')
    clips, knocked_in = [], []
    for i, action in enumerate(actions):
        moved = sideways is not None and i >= sideways[0]
        before = inside(*positions[i], moved or pushed(frames[i]))
        after = inside(*positions[i + 1], moved or pushed(frames[i + 1]))
        if after is not None and after != before and not action.startswith(('CLIP', 'MOVE N', 'MOVE S')):
            knocked_in.append({'step': i, 'block': after, 'from': positions[i], 'action': action})
        elif after is not None and after != before:
            clips.append({'step': i, 'block': after, 'from': positions[i], 'to': positions[i + 1], 'action': action,
                          'push_spot': PUSH_SPOTS.get(positions[i]),
                          'next': actions[i + 1] if i + 1 < len(actions) else None})
    # Keep no more clips than the episode counted (the executor's own count is the ground truth).
    unconfirmed = clips[record['clips']['done']:]
    clips = clips[:record['clips']['done']]
    # Path before the diamond was first entered (by a clip or after the push) as regions.
    regions = []
    for x, y in positions:
        r = region(x, y)
        if r == 'in':
            break
        if not regions or regions[-1] != r:
            regions.append(r)
    south, north = any('S' in r for r in regions), any('N' in r for r in regions)
    side = 'both' if south and north else 'south' if south else 'north' if north else 'neither'
    approach = regions[-1] if regions else '-'     # the region Link entered the diamond (or clipped) from
    first_clip_frame = frames[clips[0]['step'] + 1] if clips else None
    if not record['success']:
        way = 'failed'
    elif clips and (push_frame is None or first_clip_frame < push_frame):
        way = 'clip'
    elif push_frame is not None or any(x == 0x60 and 0x7d < y < 0x95 for x, y in positions):
        way = 'push'        # a push whose term had not come yet when Link took the stairs counts too
    else:
        way = 'other'
    return {'job': int(path.name[:8]), 'state': record['start_state'], 'success': record['success'],
            'ending': record['ending'], 'steps': record['steps'], 'frames': record['frames'],
            'health_change': record['health'] - record['start_health'], 'hits': len(hit_frames),
            'clips_done': record['clips']['done'], 'clips_aborted': record['clips']['aborted'],
            'clips': clips, 'unconfirmed': unconfirmed, 'knocked_in': knocked_in,
            'pushed_sideways': sideways[1] if sideways else None, 'pushed': push_frame is not None, 'way': way,
            'side': side, 'approach': approach, 'path': '-'.join(regions),
            'actions': actions, 'positions': positions}


def learning(run_dir: Path) -> dict:
    rows = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    steps, out = 0, {'iterations': len(rows)}
    curve = []
    for row in rows:
        steps += int(row.get('rollout/steps', 0))
        push = row.get('reward/reward-pushed-block', 0.0) / 0.75     # once per room, 0.75 (halved when hurt)
        curve.append((steps, row.get('success_rate', 0.0), row.get('clips/done', 0.0), push))
    out['steps'] = steps
    for level in (0.5, 0.8, 0.9):
        out[f'steps_to_{int(level * 100)}'] = next((s for s, rate, _, _ in curve if rate >= level), None)
    out['first_clip_steps'] = next((s for s, _, clips, _ in curve if clips >= 0.05), None)
    out['first_push_steps'] = next((s for s, _, _, push in curve if push >= 0.05), None)
    # The clip "replaces" the push from the first iteration on whose clips per episode stay above the push share
    # and above 0.5 for the rest of the run.
    replaced = None
    for i, (s, _, clips, push) in enumerate(curve):
        if all(c > max(p, 0.5) for _, _, c, p in curve[i:]):
            replaced = s
            break
    out['clip_replaces_push_steps'] = replaced
    peak = max(curve, key=lambda row: row[3])
    out['push_peak'] = {'steps': peak[0], 'push': peak[3], 'clips': peak[2]}
    tail = curve[-10:]
    out['last10'] = {'success': sum(r for _, r, _, _ in tail) / len(tail),
                     'clips': sum(c for _, _, c, _ in tail) / len(tail),
                     'push': sum(p for _, _, _, p in tail) / len(tail)}
    out['curve'] = [(s, round(r, 3), round(c, 3), round(p, 3)) for s, r, c, p in curve[::5] + curve[-1:]]
    return out


def summary(episodes: list, evaluation: dict) -> dict:
    n = len(episodes)
    wins = [e for e in episodes if e['success']]
    clips = [c for e in episodes for c in e['clips']]
    # A clip is "extra" unless it is the last clip of an episode that succeeded by clipping.
    extra = sum(len(e['clips']) - (e['way'] == 'clip') for e in episodes)
    return {
        'episodes': n, 'success': evaluation['success_rate'], 'health_change': evaluation['mean_health_change'],
        'steps': sum(e['steps'] for e in episodes) / n, 'frames': sum(e['frames'] for e in episodes) / n,
        'win_frames': sum(e['frames'] for e in wins) / len(wins) if wins else None,
        'clips_done': sum(e['clips_done'] for e in episodes) / n,
        'clips_aborted': sum(e['clips_aborted'] for e in episodes) / n,
        'wins_by_clip': sum(e['way'] == 'clip' for e in wins) / len(wins) if wins else None,
        'wins_by_push': sum(e['way'] == 'push' for e in wins) / len(wins) if wins else None,
        'pushed': sum(e['pushed'] for e in episodes) / n,
        'hit_episodes': sum(e['hits'] > 0 for e in episodes) / n,
        'endings': dict(Counter(e['ending'] for e in episodes)),
        'sides': dict(Counter(e['side'] for e in episodes)),
        'clip_blocks': dict(Counter(f"{c['block']} {c['action']}" for c in clips)),
        'clip_from': dict(Counter(f"{c['block']}@({c['from'][0]:02x},{c['from'][1]:02x})" for c in clips)),
        'clips_at_push_spot': sum(c['push_spot'] is not None for c in clips),
        'clips_extra': extra,
        'knocked_into_block': sum(len(e['knocked_in']) for e in episodes),
        'unconfirmed_clips': sum(len(e['unconfirmed']) for e in episodes),
        'pushed_sideways': dict(Counter(e['pushed_sideways'] for e in episodes if e['pushed_sideways'])),
        'approach': dict(Counter(e['approach'] for e in episodes)),
        'episodes_with_2plus_clips': sum(len(e['clips']) >= 2 for e in episodes),
        'clips_placed': len(clips), 'clips_counted': sum(e['clips_done'] for e in episodes),
    }


def _pct(value) -> str:
    return '-' if value is None else f'{100 * value:.0f}%'


def _k(value) -> str:
    return 'not reached' if value is None else f'{value / 1000:.0f}k'


def table(models: Path) -> int:
    """The report's tables, one row per arm and seed (and act mode)."""
    runs = sorted(models.glob('trap22-*/r21-s*/analysis.json'))
    print('| arm | seed | act | success | hearts | steps | frames | win frames | clips done | clips aborted |'
          ' wins by clip / push | endings |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for path in runs:
        result = json.loads(path.read_text())
        arm, seed = path.parent.parent.name.removeprefix('trap22-'), path.parent.name.removeprefix('r21-s')
        for act in ('sample', 'argmax'):
            if act not in result:
                continue
            m = result[act]['summary']
            endings = ', '.join(f'{k} {v}' for k, v in sorted(m['endings'].items(), key=lambda kv: -kv[1]))
            win = '-' if m['win_frames'] is None else f"{m['win_frames']:.0f}"
            print(f"| {arm} | {seed} | {act} | {_pct(m['success'])} | {m['health_change']:+.2f} | {m['steps']:.0f} |"
                  f" {m['frames']:.0f} | {win} | {m['clips_done']:.2f} | {m['clips_aborted']:.2f} |"
                  f" {_pct(m['wins_by_clip'])} / {_pct(m['wins_by_push'])} | {endings} |")
    print()
    print('| arm | seed | 50% | 80% | 90% | first clip | clip replaces push | first push | push peak (clips then) |'
          ' last 10 it. success / clips / push |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for path in runs:
        result = json.loads(path.read_text())
        arm, seed = path.parent.parent.name.removeprefix('trap22-'), path.parent.name.removeprefix('r21-s')
        l = result['learning']
        t = l['last10']
        print(f"| {arm} | {seed} | {_k(l['steps_to_50'])} | {_k(l['steps_to_80'])} | {_k(l['steps_to_90'])} |"
              f" {_k(l['first_clip_steps'])} | {_k(l['clip_replaces_push_steps'])} | {_k(l['first_push_steps'])} |"
              f" {l['push_peak']['push']:.2f} at {_k(l['push_peak']['steps'])} ({l['push_peak']['clips']:.2f}) |"
              f" {_pct(t['success'])} / {t['clips']:.2f} / {t['push']:.2f} |")
    print()
    print('| arm | seed | act | clipped block (action) | from | extra clips | at push spot | knocked in |'
          ' approach | hit episodes | pushed N/S | pushed W/E |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for path in runs:
        result = json.loads(path.read_text())
        arm, seed = path.parent.parent.name.removeprefix('trap22-'), path.parent.name.removeprefix('r21-s')
        for act in ('sample', 'argmax'):
            if act not in result:
                continue
            m = result[act]['summary']
            top = lambda d, n=3: ', '.join(f'{k} {v}' for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:n]) or '-'
            print(f"| {arm} | {seed} | {act} | {top(m['clip_blocks'])} | {top(m['clip_from'])} | {m['clips_extra']} |"
                  f" {m['clips_at_push_spot']} | {m['knocked_into_block']} | {top(m['approach'])} |"
                  f" {_pct(m['hit_episodes'])} | {_pct(m['pushed'])} | {top(m['pushed_sideways'])} |")
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
    for act in ('sample', 'argmax'):
        for e in result.get(act, {}).get('episodes', [])[:: 15]:
            clips = '; '.join(f"{c['action']} into {c['block']} from ({c['from'][0]:02x},{c['from'][1]:02x})"
                              f"{' push spot' if c['push_spot'] else ''}, then {c['next']}" for c in e['clips'])
            print(f"{act} #{e['job']:03} {e['state']} {e['ending']:12} {e['way']:6} {e['steps']:3} steps "
                  f"{e['frames']:5} frames hits {e['hits']} pushed {int(e['pushed'])} path {e['path']} "
                  f"| clips: {clips or '-'}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
