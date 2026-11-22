"""Round 36 (docs/experiments/round36.md "The bench"): hit prediction as the cheap test bench for the observation and
the network's shape. A named candidate (an input block and a network shape) is trained on a named family's oracle
data (``r35_guard_data.py --probe``: the look-ahead's verdict for every judged candidate, the RAM beside each
decision) with fixed splits, epochs and seeds, scored as round 35 / 36 score a guard, and appended as one row to a
results table:

    python scripts/experiments/r36_hit_bench.py CANDIDATE --family goriya [--epochs 10] [--seed 0]
        [--train-frac 1.0] [--results target/r36/bench.jsonl]
    python scripts/experiments/r36_hit_bench.py --table [--results target/r36/bench.jsonl]

**Splits** (fixed): every 7th episode held out (round 35); ``--train-frac``: that share of the training episodes,
drawn by the seed. **Score**: precision = recall at the look-ahead's veto rate on the held-out draws of the planner's
episodes and of the model's own (``r36_hit_common.summary``), the held-out log loss on every judged candidate, the
minutes. Held-out scores go to ``target/r36/bench/`` (``r36_hit_errors.py --scores`` reads them).

**Candidates** are ``+``-joined parts. A network starts from ``base``: today's policy shape (room embedding, a
1-layer entity transformer mean-pooled, the tile conv, the float blocks, a 256-wide 3-layer tower, one hit logit per
joint action: round 35's guard), from scratch; ``policy`` is the same initialised from the family's checkpoint (the
first agent's ``obs``). Inputs, computed from the observation and the RAM by small calculations (no look-ahead):

- per entity token (observation slot ``i`` is RAM object slot ``i + 1``; the treasure slot and Gleeok's heads from
  the observation alone): ``rel`` the middle-to-middle offset from Link in pixels / 8 (the game's 9-pixel touch test
  is one step), the gaps before it passes per axis, touching / in his row / in his column, the coarse offset / 64;
  ``ramvel`` the velocity from ObjDir and the q-speed (px / frame) and the game's "harms on touch" rule;
  ``contact`` the first agent's extrapolation per token (both in straight lines for 32 frames, Link standing or
  walking N S W E: how soon they touch, how close they come); ``wall`` the same stopped at walls (the tile grid:
  Link stops where his lower half meets a tile he cannot walk; walkers stop likewise, shots and fliers do not);
  ``prev1`` / ``prev2`` the previous one / two observations' offset and measured velocity of the same slot;
- ``patch``: a 9 x 9 tile patch centred on Link beside the whole grid; ``patchonly``: instead of it; ``nogrid``;
- ``linkx``: Link's own state bytes (ObjState[0] bits, ObjTimer[0], the invincibility timer, the knockback);
- ``cflat``: the first agent's flat contact block (RAM slots 1-12 geometry, ``r36_hit_common.contact_features``)
  through a zero-initialised projection added to the features (its ``obs+contact --contact geometry``).

Shapes: ``linktok2`` / ``linktok3`` Link as a token attending to the entity tokens, 2 / 3 layers, his final token the
entity block's output (no mean pool); ``relnet`` each entity paired with Link (an MLP), pooled by max and by sum;
``w2x`` / ``w4x`` the tower 512 / 1024 wide; ``deep`` 6 layers; ``critic`` the action as an input (one forward an
action, one logit) instead of one output an action; ``ehead`` a per-entity head: each entity token (with the tower's
summary) gives a hazard per (kind, direction, time bin: frames 0-3, 4-7, 8-15, 16+) plus one "other" hazard from the
tower; the action's hit probability is 1 - exp(-sum of the hazards of its kind and direction over the bins its
duration covers); trained as competing risks on the verdict, the hit frame's bin and the hitting slot the probe
recorded (``tried_hit_frame``, ``tried_hit_slot``); ``noslot`` leaves the slot out (the entities share the event as
their hazards say), ``noframe`` the frame (the hit counted in the last bin the action covers).

``trees``: XGBoost (600 rounds, depth 8) on the flat observation (``r36_hit_trees.features``), ``trees+rel+...``
with the token inputs flattened beside it. ``guard35``: round 35's guard scored as it is (goriya only).

GPU work: no job queue, one process at a time.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r36_hit_common import Data, _extrapolate, contact_features, summary  # noqa: E402

from triforce import registry  # noqa: E402
from triforce.model import MASK_KEY, EntityEncoder, GridEncoder, ModelConfig, ZeldaPolicy, layer_init  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R36 = ROOT / 'target' / 'r36'
FAMILIES = {
    'goriya': (['goriya.npz'], 'models/goriya/r32d1-goriya'),
    'goriya-all': (['goriya.npz', 'goriya-big.npz'], 'models/goriya/r32d1-goriya'),
    'boss-gleeok': (['gleeok.npz'], 'models/boss-gleeok/r34d4-gleeok4-s1-it2877'),
    'darknut': (['darknut.npz'], 'models/darknut/r34d5-dg-s362'),
}
GUARD35 = ROOT / 'target' / 'worktrees' / 'r35-guard' / 'r35' / 'guard-a' / 'guard.pt'   # round 35's guard (untracked)
TOKEN_GROUPS = ['rel', 'ramvel', 'contact', 'wall', 'prev1', 'prev2']
INPUT_PARTS = set(TOKEN_GROUPS) | {'patch', 'patchonly', 'nogrid', 'linkx', 'cflat'}
SHAPE_PARTS = {'base', 'policy', 'linktok2', 'linktok3', 'relnet', 'w2x', 'w4x', 'deep', 'critic', 'ehead'}
VECTOR_KEYS = ['history', 'link', 'objective', 'room_state', 'target']
KINDS, DIRS, DURS = 11, 8, 4
JOINT = KINDS * DIRS * DURS
WALKABLE = np.zeros(256, dtype=bool)
WALKABLE[[0, 3, 4, 5, 6, 11]] = True     # walkable, the four door corridors, stairs (zelda::tiles::TileClass)
FLIERS = {0x1B, 0x1C, 0x1D, 0x46, 0x2F, 0x42, 0x43, 0x44, 0x45}   # Keese, Gleeok's heads and body, Peahat: no walls
PATCH = 9
BINS = np.array([0] * 4 + [1] * 4 + [2] * 8 + [3] * 10000)


# ---------------------------------------------------------------------------------------------- data

def load_family(family: str) -> tuple[Data, Path]:
    files, checkpoint = FAMILIES[family]
    parts = []
    for name in files:
        with np.load(R36 / name) as z:
            parts.append({k: z[k] for k in z.files})
    d = parts[0]
    if len(parts) > 1:
        rows = eps = 0
        merged: dict[str, list] = {k: [] for k in d}
        for p in parts:
            for k, v in p.items():
                if k == 'row_episode':
                    v = v + eps
                elif k == 'tried_row':
                    v = v + rows
                merged[k].append(v)
            rows += len(p['mask'])
            eps += len(p['ep_seed'])
        d = {k: np.concatenate(v) for k, v in merged.items()}
    obs = {k[len('obs/'):]: v for k, v in d.items() if k.startswith('obs/')}
    obs[MASK_KEY] = d['mask']
    held_ep = (np.arange(len(d['ep_seed'])) % 7) == 0
    f0 = np.flatnonzero(d['tried_future'] == 0)
    row = d['tried_row'][f0].astype(np.int64)
    own = np.r_[True, row[1:] != row[:-1]]
    data = Data(d, obs, held_ep[d['row_episode']], d['ep_mode'][d['row_episode']], row,
                d['tried_joint'][f0].astype(np.int64), d['tried_lost'][f0], own, f0)
    return data, ROOT / checkpoint


def _entity_columns():
    from triforce import _native  # pylint: disable=import-outside-toplevel
    names = list(_native.ENTITY_FEATURES)
    return {n: names.index(n) for n in ('presence', 'x', 'y', 'vx', 'vy', 'touch_harms')}


def _travel(x, y, ux, uy, limit, grid, rows):
    """Pixels a 16 x 16 object at (x, y) can move along (ux, uy) (unit steps) before its lower half meets a tile it
    cannot walk (or the play area's edge), up to ``limit``; ``rows`` index each entry's grid."""
    out = np.full(len(x), float(limit), dtype=np.float32)
    free = np.ones(len(x), dtype=bool)
    for s in range(1, int(limit) + 1):
        if not free.any():
            break
        idx = np.flatnonzero(free)
        nx = x[idx] + ux[idx] * s
        ny = y[idx] + uy[idx] * s - 64
        ok = np.ones(len(idx), dtype=bool)
        for cx in (nx, nx + 15):
            for cy in (ny + 8, ny + 15):
                c, r = np.floor(cx / 8).astype(np.int64), np.floor(cy / 8).astype(np.int64)
                inside = (c >= 0) & (c < 32) & (r >= 0) & (r < 22)
                cls = grid[rows[idx], np.clip(r, 0, 21), np.clip(c, 0, 31)]
                ok &= inside & WALKABLE[cls]
        stop = idx[~ok]
        out[stop] = s - 1
        free[stop] = False
        out[idx[ok]] = np.minimum(out[idx[ok]], limit)
    return out


def _contact_walls(ddx, ddy, vx, vy, harmful, link_travel, mob_travel):
    """``contact_*`` / ``closest_*`` with both stopping where ``*_travel`` says (pixels along their motion)."""
    t = np.arange(33, dtype=np.float32)
    speed = np.hypot(vx, vy)
    unit_x = np.where(speed > 0, vx / np.maximum(speed, 1e-6), 0)
    unit_y = np.where(speed > 0, vy / np.maximum(speed, 1e-6), 0)
    mob = np.minimum(speed[..., None] * t, mob_travel[..., None])
    mx, my = unit_x[..., None] * mob, unit_y[..., None] * mob
    out = {}
    for d, (name, (lx, ly)) in enumerate(zip(('stand', 'N', 'S', 'W', 'E'),
                                             [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)])):
        if d == 0:
            link = np.zeros((len(ddx), 1, 33), dtype=np.float32)
        else:
            link = np.minimum(1.5 * t[None, None, :], link_travel[:, d - 1][:, None, None])
        rx = ddx[..., None] + mx - lx * link
        ry = ddy[..., None] + my - ly * link
        gap = np.maximum(np.abs(rx), np.abs(ry)) - 9
        touch = gap < 0
        first = np.where(touch.any(-1), touch.argmax(-1), 33)
        out[f'contact_{name}'] = np.where(harmful, (33 - first) / 33, 0).astype(np.float32)
        out[f'closest_{name}'] = np.where(harmful, np.clip(gap.min(-1), 0, 64) / 32, 2).astype(np.float32)
    return out


def token_features(data: Data, lo: int, hi: int) -> dict[str, np.ndarray]:
    """Rows ``lo:hi``: every token group (the module docstring), ``[n, 12, F]`` float32 each."""
    d, obs = data.d, data.obs
    col = _entity_columns()
    ef = obs['entity_features'][lo:hi].astype(np.float32)
    types = obs['entity_types'][lo:hi]
    ram = d['ram'][lo:hi].astype(np.int32)
    grid = obs['grid'][lo:hi]
    n = hi - lo
    present = ef[..., col['presence']] > 0
    ex = np.rint(ef[..., col['x']] * 255)
    ey = np.rint(ef[..., col['y']] * 176) + 64
    slots = np.arange(1, 12)
    backed = np.zeros((n, 12), dtype=bool)
    rtyp = ram[:, 0x34F + slots]
    backed[:, :11] = rtyp != 0
    attr = np.zeros((n, 12), dtype=np.int32)
    attr[:, :11] = ram[:, 0x4BF + slots]
    lx, ly = ram[:, 0x70].astype(np.float32), ram[:, 0x84].astype(np.float32)
    ddx = (ex + np.where(backed & (attr & 0x40 != 0), 4, 8) - (lx[:, None] + 8)).astype(np.float32)
    ddy = (ey + 8 - (ly[:, None] + 8)).astype(np.float32)
    gx, gy = np.maximum(np.abs(ddx) - 8, 0), np.maximum(np.abs(ddy) - 8, 0)
    pm = present[..., None].astype(np.float32)
    out = {}
    out['rel'] = np.stack([np.clip(ddx / 8, -12, 12), np.clip(ddy / 8, -12, 12), np.clip(gx / 8, 0, 12),
                           np.clip(gy / 8, 0, 12), np.clip(np.maximum(gx, gy) / 8, 0, 12),
                           (np.abs(ddx) < 9) & (np.abs(ddy) < 9), np.abs(ddy) < 9, np.abs(ddx) < 9,
                           ddx / 64, ddy / 64], axis=-1).astype(np.float32) * pm
    # RAM velocity and the touch rule (asm:Link_CheckMonsterTouch_NoReset), r36_hit_common.contact_features
    vx = np.zeros((n, 12), dtype=np.float32)
    vy = np.zeros((n, 12), dtype=np.float32)
    direction, speed = ram[:, 0x98 + slots], ram[:, 0x3BC + slots] / 64.0
    vx[:, :11] = np.where(direction & 1, speed, 0) - np.where(direction & 2, speed, 0)
    vy[:, :11] = np.where(direction & 4, speed, 0) - np.where(direction & 8, speed, 0)
    state, meta, stun = ram[:, 0xAC + slots], ram[:, 0x405 + slots], ram[:, 0x3D + slots]
    rule = (rtyp != 0) & (meta == 0) & (stun == 0) & ((rtyp < 0x53) | ((state & 0xF0) == 0x10))
    harmful = ef[..., col['touch_harms']] > 0
    harmful[:, :11] = np.where(backed[:, :11], rule, harmful[:, :11])
    harmful &= present
    out['ramvel'] = np.stack([vx, vy, harmful], axis=-1).astype(np.float32) * pm
    ext: dict = {}
    _extrapolate(ddx, ddy, vx, vy, harmful, ext)
    names = [f'{w}_{m}' for w in ('contact', 'closest') for m in ('stand', 'N', 'S', 'W', 'E')]
    out['contact'] = np.stack([harmful] + [ext[k] for k in names], axis=-1).astype(np.float32) * pm
    # stopped at walls
    rows = np.arange(n)
    link_travel = np.stack([_travel(lx, ly, np.full(n, ux), np.full(n, uy), 48, grid, rows)
                            for ux, uy in [(0, -1), (0, 1), (-1, 0), (1, 0)]], axis=1)
    mob_travel = np.full((n, 12), 1e3, dtype=np.float32)
    walker = present & harmful & ((vx != 0) | (vy != 0)) & backed
    walker[:, :11] &= (rtyp < 0x53) & ~np.isin(rtyp, list(FLIERS))
    r_idx, s_idx = np.nonzero(walker)
    if len(r_idx):
        sp = np.hypot(vx[r_idx, s_idx], vy[r_idx, s_idx])
        mob_travel[r_idx, s_idx] = _travel(ex[r_idx, s_idx], ey[r_idx, s_idx], vx[r_idx, s_idx] / sp,
                                           vy[r_idx, s_idx] / sp, 48, grid, r_idx)
    walls = _contact_walls(ddx, ddy, vx, vy, harmful, link_travel, mob_travel)
    out['wall'] = np.stack([walls[k] for k in names], axis=-1).astype(np.float32) * pm
    # the previous observations (same episode, same slot type, no room change between)
    ep = d['row_episode']
    hist_room = obs['history'][:, 33]   # room_changed
    for k, name in ((1, 'prev1'), (2, 'prev2')):
        prev = np.arange(lo, hi) - k
        ok = prev >= 0
        p = np.clip(prev, 0, None)
        ok &= ep[p] == ep[lo:hi]
        for j in range(k):
            ok &= hist_room[np.clip(np.arange(lo, hi) - j, 0, None)] == 0
        pef = obs['entity_features'][p].astype(np.float32)
        same = ok[:, None] & (obs['entity_types'][p] == types) & (pef[..., col['presence']] > 0) & present
        plx, ply = d['ram'][p, 0x70].astype(np.float32), d['ram'][p, 0x84].astype(np.float32)
        pdx = (np.rint(pef[..., col['x']] * 255) - plx[:, None]) / 8
        pdy = (np.rint(pef[..., col['y']] * 176) + 64 - ply[:, None]) / 8
        out[name] = np.stack([np.clip(pdx, -12, 12), np.clip(pdy, -12, 12), pef[..., col['vx']] * 4,
                              pef[..., col['vy']] * 4, same], axis=-1).astype(np.float32) * same[..., None]
    return out


def flat_features(data: Data, lo: int, hi: int) -> dict[str, np.ndarray]:
    """Rows ``lo:hi``: ``patch`` (9 x 9 tile classes centred on Link, uint8, the frame's wall outside) with his
    middle's place in its tile, ``linkx``, ``cflat``."""
    ram = data.d['ram'][lo:hi].astype(np.int32)
    grid = data.obs['grid'][lo:hi]
    n = hi - lo
    mx, my = ram[:, 0x70] + 8, ram[:, 0x84] + 8 - 64
    c, r = mx // 8, my // 8
    off = np.arange(PATCH) - PATCH // 2
    rr, cc = r[:, None, None] + off[None, :, None], c[:, None, None] + off[None, None, :]
    inside = (rr >= 0) & (rr < 22) & (cc >= 0) & (cc < 32)
    patch = np.where(inside, grid[np.arange(n)[:, None, None], np.clip(rr, 0, 21), np.clip(cc, 0, 31)], 1)
    place = np.stack([(mx % 8) / 8, (my % 8) / 8, (mx % 16) / 16, (my % 16) / 16], axis=1).astype(np.float32)
    state = ram[:, 0xAC]
    linkx = np.concatenate([((state[:, None] >> np.arange(8)) & 1).astype(np.float32),
                            np.stack([np.minimum(ram[:, 0x28] / 32, 2), np.minimum(ram[:, 0x4F0] / 32, 2),
                                      ram[:, 0xD3] / 16, ((ram[:, 0x394] + 128) % 256 - 128) / 16],
                                     axis=1).astype(np.float32),
                            ((ram[:, 0xC0:0xC1] >> np.arange(4)) & 1).astype(np.float32)], axis=1)
    return {'patch': patch.reshape(n, -1).astype(np.uint8), 'place': place, 'linkx': linkx,
            'cflat': contact_features(data.d['ram'][lo:hi], 'geometry')}


def features(data: Data, family: str) -> dict[str, np.ndarray]:
    """Every computed input of ``family``'s rows, cached in ``target/r36/bench/``."""
    path = R36 / 'bench' / f'features-{family}.npz'
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    t0 = time.time()
    n = len(data.d['mask'])
    parts: dict[str, list] = {}
    for lo in range(0, n, 20000):
        hi = min(n, lo + 20000)
        block = token_features(data, lo, hi)
        block.update(flat_features(data, lo, hi))
        for k, v in block.items():
            parts.setdefault(k, []).append(v.astype(np.float16) if v.dtype == np.float32 else v)
    out = {k: np.concatenate(v) for k, v in parts.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f'features of {family}: {n} rows in {time.time() - t0:.0f}s', flush=True)
    return out


# ---------------------------------------------------------------------------------------------- the network

def tower(inputs: int, hidden: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(depth):
        layers += [layer_init(nn.Linear(inputs if i == 0 else hidden, hidden)), nn.LayerNorm(hidden), nn.ReLU()]
    return nn.Sequential(*layers)


class LinkTokens(nn.Module):
    """Link's token (his float block) and the entity tokens through ``layers`` transformer layers; his final token
    is the output (no pooling); the entity tokens' final states are kept for a per-entity head."""

    def __init__(self, link_in: int, token_in: int, d_model: int, layers: int, out_dim: int):
        super().__init__()
        self.link_proj = layer_init(nn.Linear(link_in, d_model))
        self.token_proj = layer_init(nn.Linear(token_in, d_model))
        self.role = nn.Parameter(torch.zeros(2, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=2 * d_model, batch_first=True,
                                           dropout=0.0)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.out = layer_init(nn.Linear(d_model, out_dim))
        self.output_dim, self.state_dim = out_dim, d_model

    def forward(self, link, tokens, empty):
        x = torch.cat([(self.link_proj(link) + self.role[0]).unsqueeze(1), self.token_proj(tokens) + self.role[1]],
                      dim=1)
        mask = torch.cat([torch.zeros_like(empty[:, :1]), empty], dim=1)
        y = self.transformer(x, src_key_padding_mask=mask)
        return torch.relu(self.out(y[:, 0])), y[:, 1:]


class Relational(nn.Module):
    """Each entity token paired with Link's block through an MLP, pooled by max and by sum over present slots."""

    def __init__(self, link_in: int, token_in: int, width: int, out_dim: int):
        super().__init__()
        self.pair = nn.Sequential(layer_init(nn.Linear(link_in + token_in, width)), nn.ReLU(),
                                  layer_init(nn.Linear(width, width)), nn.ReLU())
        self.out = layer_init(nn.Linear(2 * width, out_dim))
        self.output_dim, self.state_dim = out_dim, width

    def forward(self, link, tokens, empty):
        h = self.pair(torch.cat([link.unsqueeze(1).expand(-1, tokens.shape[1], -1), tokens], dim=-1))
        present = (~empty).unsqueeze(-1).to(h.dtype)
        pooled_max = (h * present - 1e4 * (1 - present)).amax(dim=1).clamp(min=0)
        pooled_sum = (h * present).sum(dim=1) / 4
        return torch.relu(self.out(torch.cat([pooled_max, pooled_sum], dim=-1))), h


class MeanPool(nn.Module):
    """Today's entity encoder (``model.EntityEncoder``) with the per-slot states kept."""

    def __init__(self, token_in: int, types: int, layers: int):
        super().__init__()
        self.inner = EntityEncoder(types, token_in, 16, 64, 4, layers, 128, 64)
        self.output_dim, self.state_dim = 64, 64

    def forward(self, tokens, types, empty):
        enc = self.inner
        projected = enc.input_proj(torch.cat([tokens, enc.type_embedding(types.long())], dim=-1))
        safe = empty.clone()
        safe[:, 0] &= ~empty.all(dim=1)
        attended = enc.transformer(projected, src_key_padding_mask=safe)
        present = (~empty).unsqueeze(-1).to(attended.dtype)
        pooled = (attended * present).sum(dim=1) / present.sum(dim=1).clamp(min=1.0)
        return torch.relu(enc.output_proj(pooled)), attended


class BenchNet(nn.Module):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, spec: dict, obs_spec: dict, sizes: dict, base_rate: float):
        super().__init__()
        self.spec = spec
        config = ModelConfig.from_dict({'obs_spec': obs_spec})
        self.room_embedding = nn.Embedding(obs_spec['room_id']['vocab'], config.room_embed_dim)
        types = obs_spec['entity_types']['vocab']
        token_in = obs_spec['entity_features']['shape'][1] + sizes['tokens']
        link_in = obs_spec['link']['shape'][0] + sizes['linkx']
        self.type_embedding = nn.Embedding(types, 16)
        entity = spec['entity']
        if entity == 'mean':
            self.entities = MeanPool(token_in, types, 1)
        elif entity == 'linktok':
            self.entities = LinkTokens(link_in, token_in + 16, 128, spec['layers'], 128)
        else:
            self.entities = Relational(link_in, token_in + 16, 128, 128)
        features = config.room_embed_dim + self.entities.output_dim
        if spec['grid']:
            self.grid_encoder = GridEncoder(obs_spec['grid']['vocab'], 22, 32, config)
            features += self.grid_encoder.output_dim
        if spec['patch']:
            self.patch_embedding = nn.Embedding(obs_spec['grid']['vocab'], 4)
            self.patch = nn.Sequential(layer_init(nn.Linear(PATCH * PATCH * 4 + 4, 128)), nn.ReLU())
            features += 128
        features += sum(obs_spec[k]['shape'][0] for k in VECTOR_KEYS) + sizes['linkx']
        self.feature_dim = features
        if sizes['cflat']:
            self.cflat = nn.Sequential(nn.Linear(sizes['cflat'], 512), nn.LayerNorm(512), nn.ReLU(),
                                       nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU())
            self.cflat_proj = nn.Linear(256, features)
            nn.init.zeros_(self.cflat_proj.weight)
            nn.init.zeros_(self.cflat_proj.bias)
        hidden, depth = spec['hidden'], spec['depth']
        head = spec['head']
        self.tower = tower(features + (KINDS + DIRS + DURS if head == 'critic' else 0), hidden, depth)
        prior = math.log(base_rate / (1 - base_rate))
        if head == 'joint':
            self.head = nn.Linear(hidden, JOINT)
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias, prior)
        elif head == 'critic':
            self.head = nn.Linear(hidden, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias, prior)
        else:
            state = self.entities.state_dim
            self.summary = layer_init(nn.Linear(hidden, 64))
            self.entity_head = nn.Sequential(layer_init(nn.Linear(state + 64, 128)), nn.ReLU(),
                                             layer_init(nn.Linear(128, JOINT), std=0.01))
            self.other_head = layer_init(nn.Linear(hidden, JOINT), std=0.01)
            nn.init.constant_(self.entity_head[-1].bias, -5.0)
            nn.init.constant_(self.other_head.bias, -4.0)

    def forward(self, batch: dict, joint: torch.Tensor):
        """Per candidate: the hit logit (``joint`` / ``critic``) or the hazards ``[B, 13, 4]`` (``ehead``)."""
        spec = self.spec
        ef = batch['entity_features'].float()
        empty = ef[..., 0] <= 0
        tokens = torch.cat([ef, batch['tokens'].float()], dim=-1) if 'tokens' in batch else ef
        link = torch.cat([batch['link'].float(), batch['linkx'].float()], -1) if 'linkx' in batch else \
            batch['link'].float()
        if spec['entity'] == 'mean':
            pooled, states = self.entities(tokens, batch['entity_types'], empty)
        else:
            tokens = torch.cat([tokens, self.type_embedding(batch['entity_types'].long())], dim=-1)
            pooled, states = self.entities(link, tokens, empty)
        blocks = [self.room_embedding(batch['room_id'].long().reshape(-1)), pooled]
        if spec['grid']:
            blocks.append(self.grid_encoder(batch['grid']))
        if spec['patch']:
            p = self.patch_embedding(batch['patch'].long()).flatten(1)
            blocks.append(self.patch(torch.cat([p, batch['place'].float()], dim=-1)))
        blocks += [batch[k].float() for k in VECTOR_KEYS]
        if 'linkx' in batch:
            blocks.append(batch['linkx'].float())
        features = torch.cat(blocks, dim=-1)
        if 'cflat' in batch:
            features = features + self.cflat_proj(self.cflat(batch['cflat'].float()))
        kind, direction, duration = joint // 32, (joint // 4) % 8, joint % 4
        rows = torch.arange(len(joint), device=joint.device)
        if spec['head'] == 'critic':
            action = torch.cat([F.one_hot(kind, KINDS), F.one_hot(direction, DIRS), F.one_hot(duration, DURS)],
                               dim=-1).float()
            return self.head(self.tower(torch.cat([features, action], dim=-1))).squeeze(-1)
        h = self.tower(features)
        if spec['head'] == 'joint':
            return self.head(h)[rows, joint]
        # hazards for the candidate's kind and direction, per time bin: 12 entities and the "other" source
        g = self.summary(h).unsqueeze(1).expand(-1, states.shape[1], -1)
        per = self.entity_head(torch.cat([states, g], dim=-1))               # [B, 12, 352]
        base = (kind * 32 + direction * 4).unsqueeze(-1) + torch.arange(4, device=joint.device)  # [B, 4]
        ent = per.gather(2, base.unsqueeze(1).expand(-1, per.shape[1], -1))  # [B, 12, 4]
        ent = F.softplus(ent) * (~empty).unsqueeze(-1)
        other = F.softplus(self.other_head(h).gather(1, base)).unsqueeze(1)  # [B, 1, 4]
        return torch.cat([ent, other], dim=1)


def hazard_scores(hazards: torch.Tensor, joint: torch.Tensor) -> torch.Tensor:
    """P(hit) of the candidate: 1 - exp(-the hazards over the bins its duration covers)."""
    covered = (torch.arange(4, device=joint.device) <= (joint % 4).unsqueeze(-1)).float()
    total = (hazards.sum(1) * covered).sum(-1)
    return 1 - torch.exp(-total)


def hazard_loss(hazards, joint, lost, hit_bin, hit_slot):
    """Competing risks: survive every covered bin (safe); or survive the bins before the hit's, an event in its
    bin, and the event from the recorded slot."""
    dur = joint % 4
    b = torch.minimum(hit_bin, dur)
    total = hazards.sum(1)                                     # [B, 4]
    ar = torch.arange(4, device=joint.device)
    before = torch.where(lost.bool().unsqueeze(-1), (ar < b.unsqueeze(-1)).float(), (ar <= dur.unsqueeze(-1)).float())
    survive = (total * before).sum(-1)
    at = total.gather(1, b.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
    event = -torch.log(-torch.expm1(-at))
    rows = torch.arange(len(joint), device=joint.device)
    if hit_slot is None:    # ``noslot``: who hit is not told; the entities share the event as the hazards say
        return (survive + lost * event).mean()
    share = -torch.log(hazards[rows, hit_slot, b].clamp(min=1e-6) / at)
    return (survive + lost * (event + share)).mean()


# ---------------------------------------------------------------------------------------------- candidates

def parse(name: str) -> dict:
    parts = name.split('+')
    spec = {'name': name, 'trees': parts[0] in ('trees', 'guard35'), 'tokens': [], 'grid': True, 'patch': False,
            'linkx': False, 'cflat': False, 'entity': 'mean', 'layers': 1, 'hidden': 256, 'depth': 3,
            'head': 'joint', 'init': 'random', 'slot': True, 'frame': True}
    for p in parts:
        if p in TOKEN_GROUPS:
            spec['tokens'].append(p)
        elif p == 'patch':
            spec['patch'] = True
        elif p == 'patchonly':
            spec['patch'], spec['grid'] = True, False
        elif p == 'nogrid':
            spec['grid'] = False
        elif p in ('linkx', 'cflat'):
            spec[p] = True
        elif p == 'policy':
            spec['init'] = 'policy'
        elif p in ('linktok2', 'linktok3'):
            spec['entity'], spec['layers'] = 'linktok', int(p[-1])
        elif p == 'relnet':
            spec['entity'] = 'relnet'
        elif p == 'w2x':
            spec['hidden'] = 512
        elif p == 'w4x':
            spec['hidden'] = 1024
        elif p == 'deep':
            spec['depth'] = 6
        elif p in ('noslot', 'noframe'):
            spec[p[2:]] = False
        elif p in ('critic', 'ehead'):
            spec['head'] = p
        elif p not in ('base', 'trees', 'guard35'):
            raise SystemExit(f'unknown candidate part {p!r} (inputs {sorted(INPUT_PARTS)}, shapes {sorted(SHAPE_PARTS)})')
    return spec


def policy_init(net: BenchNet, checkpoint: Path) -> None:
    """The family's checkpoint's encoders and value tower into ``net`` where the shapes match (today's shape)."""
    policy: ZeldaPolicy = registry.load_model(checkpoint)[0]
    src = policy.state_dict()
    mapping = {'room_embedding.': 'room_embedding.', 'entities.inner.': 'entity_encoder.',
               'grid_encoder.': 'grid_encoder.', 'tower.': 'mlp_extractor.value_net.'}
    own = net.state_dict()
    copied = 0
    for key in own:
        for mine, theirs in mapping.items():
            if key.startswith(mine):
                other = theirs + key[len(mine):]
                if other in src and src[other].shape == own[key].shape:
                    own[key] = src[other]
                    copied += 1
    net.load_state_dict(own)
    print(f'policy init: {copied} of {len(own)} tensors from {checkpoint}', flush=True)


def run_trees(spec, data: Data, feats: dict, seed: int, train_c, held_c) -> tuple[np.ndarray, int]:
    import xgboost as xgb  # pylint: disable=import-outside-toplevel
    from r36_hit_trees import features as tree_features  # pylint: disable=import-outside-toplevel

    def x_of(c):
        rows = data.c_row[c]
        x, _, _ = tree_features(data, rows, data.c_joint[c], 'obs', np.zeros(0, dtype=np.int64))
        extra = [feats[g][rows].reshape(len(rows), -1).astype(np.float32) for g in spec['tokens']]
        if spec['patch']:
            extra.append(feats['patch'][rows].astype(np.float32))
        if spec['linkx']:
            extra.append(feats['linkx'][rows].astype(np.float32))
        return np.concatenate([x] + extra, axis=1)
    model = xgb.XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.08, subsample=0.8, colsample_bytree=0.6,
                              tree_method='hist', device='cuda', eval_metric='logloss', max_bin=256, random_state=seed)
    model.fit(x_of(train_c), data.c_lost[train_c])
    held = np.flatnonzero(held_c)
    return model.predict_proba(x_of(held))[:, 1], 0


def main(argv) -> int:
    # pylint: disable=too-many-locals, too-many-statements, too-many-branches
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('candidate', nargs='?')
    parser.add_argument('--family', choices=sorted(FAMILIES), default='goriya')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-frac', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--results', type=Path, default=R36 / 'bench.jsonl')
    parser.add_argument('--table', action='store_true', help='print the results table (markdown) and exit')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.table:
        print_table(args.results)
        return 0
    spec = parse(args.candidate)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    t0 = time.time()
    data, checkpoint = load_family(args.family)
    d = data.d
    held_c = data.held_rows[data.c_row]
    train_c = np.flatnonzero(~held_c)
    if args.train_frac < 1:
        episodes = np.unique(d['row_episode'][~data.held_rows])
        keep = np.random.default_rng(args.seed).choice(episodes, int(round(len(episodes) * args.train_frac)),
                                                         replace=False)
        train_c = train_c[np.isin(d['row_episode'][data.c_row[train_c]], keep)]
    held_idx = np.flatnonzero(held_c)
    needs_features = spec['tokens'] or spec['patch'] or spec['linkx'] or spec['cflat']
    feats = features(data, args.family) if needs_features else {}
    rate = float(data.c_lost[train_c].mean())
    params = 0
    epochs = args.epochs
    if spec['name'] == 'guard35':
        from r36_hit_errors import guard_scores  # pylint: disable=import-outside-toplevel
        scores = guard_scores(GUARD35, data, held_idx, args.device)
        epochs = 5
        history = []
    elif spec['trees']:
        scores, params = run_trees(spec, data, feats, args.seed, train_c, held_c)
        epochs = 600
        history = []
    else:
        network = registry.load_model(checkpoint)[0]
        obs_spec = network.config.obs_spec
        keys = ['entity_features', 'entity_types', 'room_id'] + VECTOR_KEYS + (['grid'] if spec['grid'] else [])
        # half precision on the device for the 2-million-row collection (the float vectors too: GPU memory is shared)
        half = {'entity_features'} | (set(VECTOR_KEYS) if len(data.d['mask']) > 1_000_000 else set())
        dev = {k: torch.as_tensor(data.obs[k].astype(np.float16) if k in half else data.obs[k]).to(device)
               for k in keys}
        if spec['tokens']:
            dev['tokens'] = torch.as_tensor(np.concatenate([feats[g] for g in spec['tokens']], axis=-1)).to(device)
        if spec['patch']:
            dev['patch'] = torch.as_tensor(feats['patch']).to(device)
            dev['place'] = torch.as_tensor(feats['place']).to(device)
        for k in ('linkx', 'cflat'):
            if spec[k]:
                dev[k] = torch.as_tensor(feats[k]).to(device)
        sizes = {'tokens': dev['tokens'].shape[-1] if 'tokens' in dev else 0,
                 'linkx': dev['linkx'].shape[-1] if 'linkx' in dev else 0,
                 'cflat': dev['cflat'].shape[-1] if 'cflat' in dev else 0}
        net = BenchNet(spec, obs_spec, sizes, rate).to(device)
        if spec['init'] == 'policy':
            policy_init(net, checkpoint)
        params = sum(p.numel() for p in net.parameters())
        rows_dev = torch.as_tensor(data.c_row).to(device)
        joint_dev = torch.as_tensor(data.c_joint).to(device)
        y_dev = torch.as_tensor(data.c_lost.astype(np.float32)).to(device)
        if spec['head'] == 'ehead':
            hf = d['tried_hit_frame'][data.c_index].astype(np.int64)
            hs = d['tried_hit_slot'][data.c_index].astype(np.int64) - 1
            present = data.obs['entity_features'][data.c_row, :, 0] > 0
            ok = (hs >= 0) & (hs < 12)
            ok[ok] &= present[np.flatnonzero(ok), hs[ok]]
            bins = np.where(hf >= 0, BINS[np.clip(hf, 0, len(BINS) - 1)], 3) if spec['frame'] else np.full(len(hf), 3)
            bin_dev = torch.as_tensor(bins).to(device)
            slot_dev = torch.as_tensor(np.where(ok, hs, 12)).to(device) if spec['slot'] else None
        print(f"{spec['name']} on {args.family}: {params} parameters, {len(train_c)} training candidates (rate "
              f"{rate:.4f}), {len(held_idx)} held out; loaded in {time.time() - t0:.0f}s", flush=True)

        def forward(idx):
            rows = rows_dev[idx]
            batch = {k: v[rows] for k, v in dev.items()}
            return net(batch, joint_dev[idx])

        def score(idx):
            out = forward(idx)
            return hazard_scores(out, joint_dev[idx]) if spec['head'] == 'ehead' else torch.sigmoid(out)

        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        steps = epochs * ((len(train_c) + args.batch - 1) // args.batch)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.05)
        gen = torch.Generator(device='cpu').manual_seed(args.seed)
        train_dev = torch.as_tensor(train_c).to(device)
        held_dev = torch.as_tensor(held_idx).to(device)
        history = []
        for epoch in range(epochs):
            net.train()
            order = train_dev[torch.randperm(len(train_dev), generator=gen).to(device)]
            total = 0.0
            for i in range(0, len(order), args.batch):
                idx = order[i:i + args.batch]
                out = forward(idx)
                if spec['head'] == 'ehead':
                    loss = hazard_loss(out, joint_dev[idx], y_dev[idx], bin_dev[idx],
                                       None if slot_dev is None else slot_dev[idx])
                else:
                    loss = F.binary_cross_entropy_with_logits(out, y_dev[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                schedule.step()
                total += loss.item() * len(idx)
            net.eval()
            with torch.no_grad():
                scores = torch.cat([score(held_dev[i:i + 8192]) for i in range(0, len(held_dev), 8192)]
                                   ).float().cpu().numpy()
            s = summary(data, scores, held_c)
            history.append({'epoch': epoch + 1, 'train_loss': total / len(order), 'log_loss': s['log_loss'],
                            'pr0': s['draw0']['pr'], 'pr1': s['draw1']['pr']})
            print(json.dumps(history[-1]), flush=True)
    result = summary(data, scores, held_c)
    minutes = (time.time() - t0) / 60
    tag = f"{args.family}-{spec['name']}-e{epochs}-f{args.train_frac:g}-s{args.seed}"
    (R36 / 'bench').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(R36 / 'bench' / f'scores-{tag}.npz', index=data.c_index[held_idx], scores=scores,
                        label=data.c_lost[held_idx])
    row = {'candidate': spec['name'], 'family': args.family, 'parameters': int(params), 'epochs': epochs,
           'seed': args.seed, 'train_frac': args.train_frac, 'train_candidates': int(len(train_c)),
           'pr0': result['draw0']['pr'], 'pr0_se': result['draw0']['pr_se'], 'pr1': result['draw1']['pr'],
           'pr1_se': result['draw1']['pr_se'], 'auc0': result['draw0']['auc'], 'auc1': result['draw1']['auc'],
           'n1': result['draw1']['n'], 'lost1': result['draw1']['lost'], 'log_loss': result['log_loss'],
           'minutes': round(minutes, 1), 'history': history}
    with open(args.results, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps({k: v for k, v in row.items() if k != 'history'}), flush=True)
    return 0


def print_table(path: Path) -> None:
    """One line per (family, candidate, epochs, data share): P = R per seed and their mean, ranked by the mean of
    the two draw sets within a family."""
    groups: dict[tuple, list] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        groups.setdefault((r['family'], r['candidate'], r['epochs'], r['train_frac']), []).append(r)
    print('| family | candidate | parameters | epochs | data | seeds | planner draws P = R | model draws P = R | '
          'held-out log loss | minutes |')
    print('|---|---|---:|---:|---:|---|---|---|---:|---:|')
    order = sorted(groups.items(), key=lambda kv: (kv[0][0], -np.mean([r['pr0'] + r['pr1'] for r in kv[1]])))
    for (family, cand, epochs, frac), rows in order:
        rows = sorted(rows, key=lambda r: r['seed'])
        seeds = ','.join(str(r['seed']) for r in rows)

        def cell(key):
            values = [r[key] for r in rows]
            mean = f'**{np.mean(values):.3f}**'
            return mean if len(values) == 1 else f"{mean} ({' / '.join(f'{v:.3f}' for v in values)})"
        print(f"| {family} | {cand} | {rows[0]['parameters']:,} | {epochs} | {frac:g} | {seeds} | {cell('pr0')} | "
              f"{cell('pr1')} | {np.mean([r['log_loss'] for r in rows]):.3f} | "
              f"{np.mean([r['minutes'] for r in rows]):.1f} |")


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
