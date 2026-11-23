"""Round 38: the owner's threat lane, as inputs computed in Python for the supervised hit bench
(docs/experiments/round38.md "The threat lane").

    python scripts/experiments/r38_lane.py measure --family goriya            # shot speeds, the post-decision slice
    python scripts/experiments/r38_lane.py bench ARM --family goriya --seed S  # ARM: none | inline | lane | lane-noinline

The owner: "Perhaps it should be a 'column' but only maybe 3 tiles (ish, you measure) in front of the enemies with
projectiles. IE there's no reason to be scared of the tile in front of an octorok all the way across the screen,
but standing 1 tile away when their projectile is ready to fire is the right time [for it] to fire the
projectile. Perhaps based on the speed of the projectile."

Per entity token (observation slot i = RAM object slot i + 1), for a shooter that can fire (a Goriya not waiting for
its boomerang, `ObjState` bit 7 clear; an Octorok, Moblin or Lynel whose shoot timer `$451` is not in its last 16
frames, i.e. its shot not just released; neither stunned nor spawning), along its facing (`ObjDir`): whether Link's
middle is in the lane (ahead, within the 9-pixel touch test across it, within the boomerang's `$51`-pixel range),
the frames a shot fired as soon as it can (the Goriya's `ObjTimer`, the shooter's timer down to `$10`) takes to reach
him at the type's shot speed (boomerang 2.5, rock 3, arrow 2, sword shot 3 px/frame: zelda-asm/enemies.md), and the
same for Link standing or walking N, S, W, E for 8 frames (12 px): ``danger = in lane × max(0, 1 − frames / 32)``.
Eleven inputs: can fire, the delay / 64, in the lane now, the distance along / 64, across / 16, danger now, danger
after each of the five motions. Off (zero) for every other object and while the shooter's shot flies.

Arms (the bench's `hitnet` candidate: Link's token over type, features, rel, contact; 10 epochs, experiment 2's
splits): ``none`` (schema 6's `in_line` zeroed), ``inline`` (as is), ``lane`` (the lane inputs added),
``lane-noinline``. Scored as experiment 2 (P = R on the planner's / the model's draws) and on the slice the lane is
for: **hits by shots fired after the decision** (the hitter, `tried_hit_type`, is a shot and its slot held no such
shot at the decision): their AUC against every safe candidate, and their recall at the threshold where all held-out
judged candidates have precision = recall.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'experiments'))

import r36_hit_bench as hb  # noqa: E402
from r36_hit_common import auc, summary  # noqa: E402
from triforce import _native, nets  # noqa: E402
from triforce.nets import hitnet as hitnet_module  # noqa: E402

R38 = ROOT / 'target' / 'r38'
SHOOTERS = {0x05: (2.5, 81.0, 'goriya'), 0x06: (2.5, 81.0, 'goriya'),
            0x07: (3.0, 999.0, 'timer'), 0x08: (3.0, 999.0, 'timer'), 0x09: (3.0, 999.0, 'timer'),
            0x0A: (3.0, 999.0, 'timer'), 0x03: (2.0, 999.0, 'timer'), 0x04: (2.0, 999.0, 'timer'),
            0x01: (3.0, 999.0, 'timer'), 0x02: (3.0, 999.0, 'timer')}
MOTIONS = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]        # stand, N, S, W, E
STEP = 8                                                    # frames a motion is looked at
LINK_SPEED = 1.5
HORIZON = 32.0
LANE = 11
IN_LINE = list(_native.ENTITY_FEATURES).index('in_line')


def lane_columns(ram: np.ndarray) -> np.ndarray:
    """``[N, 12, 11]`` lane inputs (module docstring)."""
    r = ram.astype(np.int32)
    slots = np.arange(1, 13)
    typ, state = r[:, 0x34F + slots], r[:, 0xAC + slots]
    meta, stun = r[:, 0x405 + slots], r[:, 0x3D + slots]
    direction, timer, shoot = r[:, 0x98 + slots], r[:, 0x28 + slots], r[:, 0x451 + slots]
    attr = r[:, 0x4BF + slots]
    speed = np.zeros(typ.shape, np.float32)
    reach = np.zeros(typ.shape, np.float32)
    goriya = np.zeros(typ.shape, bool)
    timed = np.zeros(typ.shape, bool)
    for t, (s, rng, kind) in SHOOTERS.items():
        m = typ == t
        speed[m], reach[m] = s, rng
        (goriya if kind == 'goriya' else timed)[m] = True
    live = (meta == 0) & (stun == 0)
    can = live & ((goriya & ((state & 0x80) == 0)) | (timed & ((shoot == 0) | (shoot > 0x10))))
    delay = np.where(goriya, timer, np.where(shoot > 0x10, shoot - 0x10, 32)).astype(np.float32)
    mx = r[:, 0x70 + slots] + np.where(attr & 0x40, 4, 8)
    my = r[:, 0x84 + slots] + 8
    fx = (np.where(direction & 1, 1, 0) - np.where(direction & 2, 1, 0)).astype(np.float32)
    fy = (np.where(direction & 4, 1, 0) - np.where(direction & 8, 1, 0)).astype(np.float32)
    lx, ly = r[:, 0x70:0x71] + 8, r[:, 0x84:0x85] + 8
    out = np.zeros(typ.shape + (LANE,), np.float32)

    def danger(px, py, wait):
        rx = ((px - mx + 128) % 256 - 128).astype(np.float32)
        ry = ((py - my + 128) % 256 - 128).astype(np.float32)
        along = rx * fx + ry * fy
        across = np.abs(rx * fy - ry * fx)
        inside = can & (speed > 0) & (along > 0) & (across < 9) & (along - 8 <= reach) & ((fx != 0) | (fy != 0))
        frames = wait + 1 + np.maximum(along - 8, 0) / np.maximum(speed, 1e-3)
        return inside, along, across, np.where(inside, np.clip(1 - frames / HORIZON, 0, 1), 0)

    inside, along, across, now = danger(lx, ly, delay)
    out[..., 0] = can & (speed > 0)
    out[..., 1] = np.where(can, np.minimum(delay, 128) / 64, 0)
    out[..., 2] = inside
    out[..., 3] = np.where(can & (speed > 0), np.clip(along / 64, -4, 4), 0)
    out[..., 4] = np.where(can & (speed > 0), np.minimum(across, 64) / 16, 0)
    out[..., 5] = now
    for k, (ux, uy) in enumerate(MOTIONS):
        out[..., 6 + k] = danger(lx + ux * LINK_SPEED * STEP, ly + uy * LINK_SPEED * STEP,
                                 np.maximum(delay - STEP, 0))[3]
    return out


def post_decision_shots(data) -> np.ndarray:
    """Per judged future-0 candidate (``data.c_*``): lost to a shot that was not in the air at the decision."""
    d = data.d
    typ = d['tried_hit_type'][data.c_index].astype(np.int64)
    slot = d['tried_hit_slot'][data.c_index].astype(np.int64)
    rows = data.c_row
    at = np.where(slot > 0, d['ram'][rows, 0x34F + np.clip(slot, 0, 19)].astype(np.int64), -1)
    return data.c_lost.astype(bool) & (typ >= 0x53) & (at != typ)


def measure(family: str) -> dict:
    """Shot speeds by type (the RAM q-speed of flying shots), the post-decision slice, and how far ahead of a
    shooter Link stood at the decision for the post-decision hits (the lane's reach)."""
    data, _ = hb.load_family(family)
    d = data.d
    ram = d['ram']
    post = post_decision_shots(data)
    typ = d['tried_hit_type'][data.c_index].astype(np.int64)
    out = {'family': family, 'lost': int(data.c_lost.sum()), 'shot_hits': int((data.c_lost.astype(bool) & (typ >= 0x53)).sum()),
           'post_decision': int(post.sum()), 'post_by_type': {hex(t): int((post & (typ == t)).sum()) for t in np.unique(typ[post])},
           'hit_frame_post': np.percentile(d['tried_hit_frame'][data.c_index][post], [10, 50, 90]).tolist()}
    sub = ram[::25].astype(np.int32)
    slots = np.arange(1, 13)
    t, q, s = sub[:, 0x34F + slots], sub[:, 0x3BC + slots], sub[:, 0xAC + slots]
    speeds = {}
    for shot in sorted(set(t[t >= 0x53].tolist())):
        m = (t == shot) & ((s & 0xF0) == 0x10)
        if m.any():
            vals, counts = np.unique(q[m], return_counts=True)
            speeds[hex(shot)] = {'flying': int(m.sum()), 'qspeed_mode': int(vals[counts.argmax()]),
                                 'px_per_frame': float(vals[counts.argmax()] / 64)}
    out['shot_speed'] = speeds
    lanes = lane_columns(ram[data.c_row[post]])
    best = lanes[..., 5].argmax(-1)
    rows = np.arange(len(best))
    hit_lane = lanes[rows, best]
    out['post_in_lane_now'] = float((hit_lane[:, 2] > 0).mean()) if len(best) else 0.0
    out['post_along_px'] = np.percentile(hit_lane[:, 3][hit_lane[:, 2] > 0] * 64, [10, 50, 90]).tolist() \
        if (hit_lane[:, 2] > 0).any() else []
    any_lane = lanes[..., 6:].max(-1).max(-1) > 0
    out['post_in_any_motion_lane'] = float(any_lane.mean()) if len(best) else 0.0
    print(json.dumps(out, indent=1), flush=True)
    return out


def bench(arm: str, family: str, seed: int, epochs: int, lr: float = 4e-4, batch: int = 1024) -> dict:
    torch.manual_seed(seed)
    device = torch.device('cuda')
    t0 = time.time()
    data, checkpoint = hb.load_family(family)
    held_c = data.held_rows[data.c_row]
    train_c = np.flatnonzero(~held_c)
    held_idx = np.flatnonzero(held_c)
    obs_spec = hb.registry.load_model(checkpoint)[0].config.obs_spec
    keys = ['entity_features', 'entity_types', 'room_id', 'grid'] + hb.VECTOR_KEYS
    half = {'entity_features'}
    dev = {k: torch.as_tensor(data.obs[k].astype(np.float16) if k in half else data.obs[k]).to(device) for k in keys}
    if arm in ('none', 'lane-noinline'):
        dev['entity_features'][..., IN_LINE] = 0
    dev.update({k: torch.as_tensor(v).to(device) for k, v in hb.engine_columns(data).items()})
    tokens = ['type', 'features', 'rel', 'contact']
    full = {**obs_spec, **{k: {'shape': list(_native.OBS_SPEC[k]['shape']), 'vocab': None}
                           for k in ('entity_rel', 'entity_contact')}}
    if arm.startswith('lane'):
        hitnet_module.TOKEN_GROUPS['lane'] = 'entity_lane'
        if 'entity_lane' not in hitnet_module._SPECIAL_KEYS:  # pylint: disable=protected-access
            hitnet_module._SPECIAL_KEYS += ('entity_lane',)  # pylint: disable=protected-access
        tokens.append('lane')
        full['entity_lane'] = {'shape': [12, LANE], 'vocab': None}
        dev['entity_lane'] = torch.as_tensor(lane_columns(data.d['ram'])).to(device)
    rate = float(data.c_lost[train_c].mean())
    net = nets.create('hitnet', full, tokens=tokens, hit='joint', hit_prior=rate).to(device)
    rows_dev = torch.as_tensor(data.c_row).to(device)
    joint_dev = torch.as_tensor(data.c_joint).to(device)
    y_dev = torch.as_tensor(data.c_lost.astype(np.float32)).to(device)

    def forward(idx):
        batch_obs = {k: v[rows_dev[idx]] for k, v in dev.items()}
        return net.hit_logits(net.trunk(batch_obs)).gather(1, joint_dev[idx].unsqueeze(-1)).squeeze(-1)

    params = [p for n, p in net.named_parameters() if not n.startswith(
        ('policy_tower.', 'value_tower.', 'kind_', 'direction_head.', 'duration_head.', 'value_head.'))]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(train_c) + batch - 1) // batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=0.05)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    train_dev = torch.as_tensor(train_c).to(device)
    held_dev = torch.as_tensor(held_idx).to(device)
    for epoch in range(epochs):
        net.train()
        order = train_dev[torch.randperm(len(train_dev), generator=gen).to(device)]
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            loss = F.binary_cross_entropy_with_logits(forward(idx), y_dev[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            schedule.step()
        print(f'{arm} s{seed} epoch {epoch + 1} {time.time() - t0:.0f}s', flush=True)
    net.eval()
    with torch.no_grad():
        scores = torch.cat([torch.sigmoid(forward(held_dev[i:i + 8192])) for i in range(0, len(held_dev), 8192)]
                           ).float().cpu().numpy()
    result = summary(data, scores, held_c)
    post = post_decision_shots(data)[held_idx]
    lost = data.c_lost[held_idx].astype(bool)
    threshold = np.sort(scores)[::-1][int(lost.sum()) - 1]
    slice_mask = post | ~lost
    row = {'arm': arm, 'family': family, 'seed': seed, 'epochs': epochs, 'pr0': result['draw0']['pr'],
           'pr1': result['draw1']['pr'], 'judged_pr': float(lost[scores >= threshold].mean()),
           'post_n': int(post.sum()), 'post_auc': auc(scores[slice_mask], post[slice_mask]),
           'post_recall': float((scores[post] >= threshold).mean()) if post.any() else float('nan'),
           'minutes': round((time.time() - t0) / 60, 1)}
    with open(R38 / 'lane-bench.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    m = sub.add_parser('measure')
    m.add_argument('--family', default='goriya')
    b = sub.add_parser('bench')
    b.add_argument('arms', nargs='+', choices=['none', 'inline', 'lane', 'lane-noinline'])
    b.add_argument('--family', default='goriya')
    b.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    b.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args(argv)
    if args.command == 'measure':
        measure(args.family)
    else:
        for seed in args.seeds:
            for arm in args.arms:
                bench(arm, args.family, seed, args.epochs)
    return 0


if __name__ == '__main__':
    sys.exit(main())
