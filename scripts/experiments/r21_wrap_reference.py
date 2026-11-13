"""Round 21 scripted reference (docs/experiments/round21.md): CLIP toward the entry edge, then MOVE out of the
opposite edge, EPISODES times per state with the gauntlet's hp randomisation (hp [0.15, max], full_hp 0.5; the
states hold 3 hearts).  Run from the repo root (the states dir is experiments/r21/states):

    python scripts/experiments/r21_wrap_reference.py STATE[,STATE...] [EPISODES] [OUT.json]"""
import json
import random
import sys

from triforce import _native as native

STATES = sys.argv[1].split(',')
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 100
OUT = sys.argv[3] if len(sys.argv) > 3 else 'scripted.json'
NEXT = {0x59: {'E': 0x5a, 'W': 0x58}, 0x5a: {'E': 0x5b, 'W': 0x59}, 0x5b: {'E': 0x5c, 'W': 0x5a}}
SPAN = {(0x5b, 'E'): (0, 9)}          # game.yaml: only 0x5b's upper E opening counts


def hp_values(rng, max_health):
    """start_states::ram_randomization_values for hp [0.15, max], full_hp 0.5."""
    total = 2 * max_health
    low = max(1, round(0.15 * total))
    half = total if rng.random() < 0.5 else rng.randint(low, total - 1)
    containers = (max_health - 1) & 0x0F
    if half >= total:
        return {'hearts_and_containers': (containers << 4) | containers, 'partial_hearts': 0xFF}
    return {'hearts_and_containers': (containers << 4) | (half // 2), 'partial_hearts': 0x7F if half % 2 else 0}


out = {}
for name in STATES:
    room = int(name[2:4], 16)
    clip_dir = name[4].upper()
    exit_dir = 'E' if clip_dir == 'W' else 'W'
    target = NEXT[room][exit_dir]
    rng = random.Random(name)
    game = native.Game(seed=0, states_dir='experiments/r21/states')
    game.set_clips(True)
    rows = []
    for ep in range(EPISODES):
        start = game.load(name, seed=ep, ram_overrides=hp_values(rng, 3))
        h0 = start.health
        change = game.act(f'CLIP {clip_dir}')
        clip = change.clip
        frames, steps = change.frames, 1
        result = None if clip['completed'] else f"clip-aborted-{clip['abort']}"
        if game.state.game_over or game.state.health <= 0:
            result = 'death'
        while result is None and steps < 40:
            before = game.state
            change = game.act(f'MOVE {exit_dir} 16')
            frames += change.frames
            steps += 1
            after = game.state
            if after.game_over or after.health <= 0:
                result = 'death'
            elif after.full_location != before.full_location:
                row = (before.link['y'] - 0x3D) // 8
                span = SPAN.get((room, exit_dir))
                ok = after.full_location == (0, target, False) and (span is None or span[0] <= row <= span[1])
                result = 'exit-correct' if ok else 'exit-wrong'
        rows.append({'result': result or 'steps', 'health_change': game.state.health - h0, 'steps': steps,
                     'frames': frames, 'clip': clip})
    n = len(rows)
    out[name] = {
        'episodes': n,
        'success': sum(r['result'] == 'exit-correct' for r in rows) / n,
        'mean_health_change': sum(r['health_change'] for r in rows) / n,
        'mean_frames': sum(r['frames'] for r in rows) / n,
        'mean_steps': sum(r['steps'] for r in rows) / n,
        'results': {k: sum(r['result'] == k for r in rows) for k in sorted({r['result'] for r in rows})},
    }
    print(name, json.dumps(out[name]), flush=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
