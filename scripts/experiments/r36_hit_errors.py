"""Round 36 (docs/experiments/round36.md "What the observation lacks", §1): where a hit classifier is wrong, on the
held-out episodes of ``r35_guard_data.py --probe`` data.

    python scripts/experiments/r36_hit_errors.py DATA.npz (--guard GUARD.pt | --scores SCORES.npz) OUT.md
        [--device cuda]

``--guard``: a round 35 guard (``triforce.guard``) scored here; ``--scores``: the held-out scores
``r36_hit_oracle.py --scores`` wrote. The decision's own draw, per mode, is vetoed at the matched rate (the top-k
by score, k the hits the look-ahead found: precision = recall), as round 35 measured it. Every draw is then a hit
caught, a hit missed (false negative), a false alarm or a quiet pass; the tables break the misses and the false
alarms down by: what hit Link (the monster slot nearest him at the frame his health dropped, its type, a shot or a
body: ``tried_hit_*``), the frame of the action the hit lands on, the action's kind and duration, Link's own state
(RAM: ObjState[0] ``$AC``, his invincibility timer ``$4F0``, his knockback ``$C0``), the geometry at the decision
(distance to the nearest live monster or shot, whether one is in his row or column), and the room. The last table
says how concentrated the misses are: the share of them in the largest cells of room × hitter × kind × hit frame.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r36_hit_common import KIND_NAMES, load  # noqa: E402

TYPE_NAMES = {0x01: 'blue Lynel', 0x02: 'red Lynel', 0x05: 'blue Goriya', 0x06: 'red Goriya', 0x0B: 'red Darknut',
              0x0C: 'blue Darknut', 0x1B: 'blue Keese', 0x1C: 'red Keese', 0x1D: 'black Keese', 0x2A: 'Stalfos',
              0x2B: 'Bubble', 0x30: 'Gibdo', 0x13: 'Zol', 0x14: 'Gel', 0x15: 'Gel', 0x17: 'Like-Like',
              0x27: 'Wallmaster', 0x42: 'Gleeok', 0x43: 'Gleeok (3)', 0x44: 'Gleeok (4)', 0x45: 'Gleeok (5)',
              0x46: 'Gleeok head', 0x3D: 'Aquamentus'}
DURATIONS = [4, 8, 16, 32]


def type_name(t: int) -> str:
    if t <= 0:
        return 'none'
    if 0x53 <= t <= 0x5C:
        return f'shot ${t:02X}'
    return TYPE_NAMES.get(t, f'${t:02X}')


def guard_scores(guard_path: Path, data, idx: np.ndarray, device: str) -> np.ndarray:
    from triforce.guard import hit_logits, load_guard  # pylint: disable=import-outside-toplevel
    guard = load_guard(guard_path).to(device)
    rows, joints = data.c_row[idx], data.c_joint[idx]
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 4096):
            r = rows[i:i + 4096]
            obs = {k: (v[r].astype(np.int64) if k == 'grid' else v[r]) for k, v in data.obs.items()}
            logits = hit_logits(guard, obs)
            j = torch.as_tensor(joints[i:i + 4096], device=logits.device)
            out.append(torch.sigmoid(logits[torch.arange(len(r), device=logits.device), j]).cpu().numpy())
    return np.concatenate(out)


def table(title: str, keys: np.ndarray, lost: np.ndarray, flagged: np.ndarray, top: int = 12) -> list[str]:
    fn, fp = lost & ~flagged, flagged & ~lost
    lines = [f'**{title}**', '', '| | draws | hits | missed (FN) | missed / hits | false alarms (FP) | FP / alarms |'
             ' share of all errors |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    errors = max(int(fn.sum() + fp.sum()), 1)
    uniq, inv = np.unique(keys, return_inverse=True)
    order = np.argsort(-(np.bincount(inv, fn) + np.bincount(inv, fp)))
    for u in order[:top]:
        m = inv == u
        h, f, a, p = int(lost[m].sum()), int(fn[m].sum()), int(flagged[m].sum()), int(fp[m].sum())
        lines.append(f'| {uniq[u]} | {int(m.sum())} | {h} | {f} | {f / max(h, 1):.2f} | {p} | {p / max(a, 1):.2f} '
                     f'| {(f + p) / errors:.2f} |')
    lines.append('')
    return lines


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--guard', type=Path)
    parser.add_argument('--scores', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    data = load(args.data)
    d = data.d
    held = data.held_rows[data.c_row]
    draw = np.flatnonzero(held & data.c_own)
    if args.scores:
        with np.load(args.scores) as s:
            by_index = dict(zip(s['index'].tolist(), s['scores'].tolist()))
        scores = np.array([by_index[i] for i in data.c_index[draw]])
    else:
        scores = guard_scores(args.guard, data, draw, args.device)
    rows = data.c_row[draw]
    lost = data.c_lost[draw]
    mode = data.mode_of_row[rows]
    flagged = np.zeros(len(draw), dtype=bool)
    for m in (0, 1):
        sel = np.flatnonzero(mode == m)
        k = int(lost[sel].sum())
        flagged[sel[np.argsort(-scores[sel])[:k]]] = True
    t = data.c_index[draw]
    hit_frame, hit_type, hit_shot = d['tried_hit_frame'][t], d['tried_hit_type'][t], d['tried_hit_shot'][t]
    frames = d['tried_frames'][t]
    joint = data.c_joint[draw]
    kind, duration = joint // 32, joint % 4
    ram = d['ram'][rows]
    lines = [f'# Errors of {args.guard or args.scores} on {args.data.name}', '',
             f'{len(draw)} held-out draws, {int(lost.sum())} hits; at the matched rate per mode: '
             f'{int((lost & flagged).sum())} caught, {int((lost & ~flagged).sum())} missed, '
             f'{int((flagged & ~lost).sum())} false alarms (precision = recall '
             f'{(lost & flagged).sum() / max(lost.sum(), 1):.3f}).', '']
    hitter = np.array([f'{type_name(int(ty))} ({"shot flag" if sh == 1 else "body"})' if ty >= 0 else '-'
                       for ty, sh in zip(hit_type, hit_shot)])
    lines += table('What hit Link (hits only; "-" for false alarms)', hitter, lost, flagged)
    fbin = np.select([hit_frame < 0, hit_frame == 0, hit_frame <= 3, hit_frame <= 7, hit_frame <= 15],
                     ['-', 'a: frame 0', 'b: 1-3', 'c: 4-7', 'd: 8-15'], 'e: 16+')
    lines += table('The frame of the action the hit lands on', fbin, lost, flagged)
    frac = np.where(hit_frame >= 0, hit_frame / np.maximum(frames, 1), -1)
    qbin = np.select([frac < 0, frac < 0.25, frac < 0.5, frac < 0.75], ['-', 'a: first quarter', 'b: second',
                                                                          'c: third'], 'd: last quarter')
    lines += table('Where in the action (hit frame / the action\'s frames)', qbin, lost, flagged)
    kd = np.array([f'{KIND_NAMES[k] if k < len(KIND_NAMES) else k} {DURATIONS[u] if k == 0 else ""}'.strip()
                   for k, u in zip(kind, duration)])
    lines += table('The action (kind, MOVE duration)', kd, lost, flagged)
    link_state = ram[:, 0xAC]
    inv = ram[:, 0x4F0]
    shove = ram[:, 0xC0]
    ls = np.array([f'state ${s:02X}' + (' invincible' if i else '') + (' knocked back' if sh else '')
                   for s, i, sh in zip(link_state, inv, shove)])
    lines += table("Link's own state at the decision (ObjState[0], $4F0, $C0)", ls, lost, flagged)
    lx, ly = ram[:, 0x70].astype(int), ram[:, 0x84].astype(int)
    types = ram[:, 0x350:0x35C]
    ox, oy = ram[:, 0x71:0x7D].astype(int), ram[:, 0x85:0x91].astype(int)
    live = types != 0
    dist = np.where(live, np.abs(ox - lx[:, None]) + np.abs(oy - ly[:, None]), 9999)
    near = dist.min(axis=1)
    dbin = np.select([near < 16, near < 32, near < 48, near < 80, near < 9999],
                     ['a: < 16 px', 'b: 16-31', 'c: 32-47', 'd: 48-79', 'e: 80+'], 'f: none')
    lines += table('Distance to the nearest live object (|dx| + |dy|, RAM)', dbin, lost, flagged)
    aligned = (live & ((np.abs(ox - lx[:, None]) < 8) | (np.abs(oy - ly[:, None]) < 8))).any(axis=1)
    lines += table('An object in his row or column (within 8 px)', np.where(aligned, 'aligned', 'not'), lost,
                   flagged)
    state_names = d['ep_state'][d['row_episode'][rows]]
    room = np.array([s.rsplit('_', 1)[0] for s in state_names])
    lines += table('Room (start state)', room, lost, flagged, top=20)
    lines += table('Mode', np.where(mode == 0, 'planner episodes', 'model episodes'), lost, flagged)
    # concentration of the misses
    cell = np.array([f'{r} | {h} | {k} | {f}' for r, h, k, f in zip(room, hitter, kd, fbin)])
    fn = lost & ~flagged
    counts = np.sort(np.unique(cell[fn], return_counts=True)[1])[::-1]
    total = max(int(fn.sum()), 1)
    lines += ['**Concentration of the misses** (cells room × hitter × action × hit frame)', '',
              f'{len(counts)} cells hold the {total} misses; the largest 5 hold {counts[:5].sum() / total:.2f}, '
              f'the largest 20 {counts[:20].sum() / total:.2f}, the largest 50 {counts[:50].sum() / total:.2f}.', '']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(lines))
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
