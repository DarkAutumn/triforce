"""Round 36 (docs/experiments/round36.md "What the observation lacks"): what the oracle test's scripts share.

The data is ``r35_guard_data.py --probe``'s (``TRIFORCE_PLAN_PROBE``): per decision the policy's observation, the mask,
the action taken and the game's RAM (``ram`` [N, 2048]); per judged candidate the look-ahead's verdict and, with a
loss, the frame of the action it landed on and the monster slot nearest Link then (``tried_hit_*``).

- ``load``: the arrays, the observation dict, the held-out episodes (every ``holdout``-th, as round 35), the
  future-0 candidates (the labels) and which of them is the decision's own draw;
- ``RAM_GROUPS``: the RAM bytes by what they mean in the game (``zelda-asm/index/ram.md``), for ablation;
- ``ram_columns`` / ``ram_inputs``: the bytes kept (stack, sprite buffer and sound left out; constant ones dropped)
  and their encoding for a network (value / 255, signed value / 128, the 8 bits);
- ``auc``, ``matched``, ``bootstrap_pr``: the metrics of round 35 (``r35_guard_train.report``), with an error by
  resampling held-out episodes.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from triforce.model import MASK_KEY

ROOT = Path(__file__).resolve().parents[2]
RAM_MD = ROOT / 'zelda-asm' / 'index' / 'ram.md'
KIND_NAMES = ['MOVE', 'SWORD', 'BEAMS', 'BOMB', 'BOOMERANG', 'ARROW', 'WAND', 'CLIP', 'RECORDER', 'FOOD', 'CANDLE']


def _slots(base: int, first: int, last: int) -> set[int]:
    return {base + i for i in range(first, last + 1)}


# Object arrays: base address, slot count (ram.md).
OBJ20 = {'ObjX': 0x70, 'ObjY': 0x84, 'ObjDir': 0x98, 'ObjState': 0xAC, 'ObjTimer': 0x28, 'ObjGridOffset': 0x394,
         'ObjPosFrac': 0x3A8, 'ObjQSpeedFrac': 0x3BC, 'ObjAnimCounter': 0x3D0, 'ObjAnimFrame': 0x3E4,
         'ObjMovingLimit': 0x380, 'ObjInvincibilityTimer': 0x4F0}
OBJ13 = {'ObjStunTimer': 0x3D, 'ObjShoveDistance': 0xD3, 'ObjType': 0x34F, 'ObjInputDir': 0x3F8, 'ObjMetastate': 0x405,
         'ObjShootTimer': 0x451, 'ObjHP': 0x485, 'ObjInvincibilityMask': 0x4B2, 'ObjAttr': 0x4BF}
OBJ19 = {'ObjShoveDir': 0xC0}
# Per-type scratch arrays (aliases differ by object type: Wanderer_ObjTurnRate, ObjRefId, Flyer_*, Fireball_*, ...).
SCRATCH = [0x412, 0x41F, 0x42C, 0x437, 0x444, 0x451, 0x45E, 0x46B, 0x478]


def _link_bytes() -> set[int]:
    out = {base for base in list(OBJ20.values()) + list(OBJ13.values()) + list(OBJ19.values())}
    out -= {0x34F}  # ObjType[0] is RoomKillCount
    return out | {0x56, 0x57, 0x512, 0x34B}


LINK = _link_bytes()
RAM_GROUPS: dict[str, set[int]] = {
    'link_state': LINK,
    'position': _slots(0x70, 1, 19) | _slots(0x84, 1, 19),
    'subpixel': _slots(0x394, 1, 19) | _slots(0x3A8, 1, 19) | _slots(0x3BC, 1, 19),
    'direction': _slots(0x98, 1, 19) | _slots(0x3F8, 1, 12) | _slots(0xC0, 1, 18) | _slots(0xD3, 1, 12),
    'timers': (_slots(0x28, 1, 19) | _slots(0x3D, 1, 12) | _slots(0x4F0, 1, 19) | _slots(0x451, 1, 12)
               | _slots(0x3D0, 1, 19) | _slots(0x3E4, 1, 19) | {0x26, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E}),
    'obj_state': (_slots(0xAC, 1, 19) | _slots(0x405, 1, 12) | _slots(0x485, 1, 12) | _slots(0x34F, 1, 12)
                  | _slots(0x4B2, 1, 12) | _slots(0x4BF, 1, 12) | _slots(0x492, 1, 11)),
    'type_scratch': _slots(0x380, 1, 19) | {a + i for a in SCRATCH for i in range(1, 13)},
    'rng': set(range(0x18, 0x25)),
    'frame_counter': {0x15},
}
RAM_GROUPS['type_scratch'] -= RAM_GROUPS['timers'] | RAM_GROUPS['obj_state']  # $451 is ObjShootTimer
EXCLUDED = (set(range(0x100, 0x300)) | set(range(0x66, 0x70)) | set(range(0x5F0, 0x620)) | {0x58, 0x5C, 0x5E, 0x5F}
            | {0xE3, 0x14, 0x17})


def group_of(address: int) -> str:
    for name, members in RAM_GROUPS.items():
        if address in members:
            return name
    return 'other'


def ram_names() -> dict[int, str]:
    """Address → the disassembly's name (``ObjX[3]`` for slot 3 of an array), from ``ram.md``."""
    names: dict[int, str] = {}
    row = re.compile(r'^\| `\$([0-9A-F]{4})` \| `([^`]+)` \|[^|]*\|[^|]*\| (.*) \|$')
    for line in RAM_MD.read_text().splitlines():
        m = row.match(line)
        if not m:
            continue
        address, name, meaning = int(m.group(1), 16), m.group(2), m.group(3)
        size = re.match(r'\[(\d+)\]', meaning)
        count = int(size.group(1)) if size else 1
        for i in range(count):
            if address + i not in names or count > 1 and '[' not in names[address + i]:
                names[address + i] = f'{name}[{i}]' if count > 1 else name
    return names


@dataclass
class Data:
    d: dict
    obs: dict
    held_rows: np.ndarray       # per decision row: its episode is held out
    mode_of_row: np.ndarray     # per decision row: 0 planner, 1 model alone
    c_row: np.ndarray           # future-0 candidates: decision row, joint action, verdict, own draw
    c_joint: np.ndarray
    c_lost: np.ndarray
    c_own: np.ndarray
    c_index: np.ndarray         # their index into the tried_* arrays


def load(path, holdout: int = 7) -> Data:
    with np.load(path) as data:
        d = {k: data[k] for k in data.files}
    obs = {k[len('obs/'):]: v for k, v in d.items() if k.startswith('obs/')}
    obs[MASK_KEY] = d['mask']
    held_ep = (np.arange(len(d['ep_seed'])) % holdout) == 0
    held_rows = held_ep[d['row_episode']]
    mode_of_row = d['ep_mode'][d['row_episode']]
    f0 = np.flatnonzero(d['tried_future'] == 0)
    row = d['tried_row'][f0].astype(np.int64)
    own = np.r_[True, row[1:] != row[:-1]]
    return Data(d, obs, held_rows, mode_of_row, row, d['tried_joint'][f0].astype(np.int64), d['tried_lost'][f0], own,
                f0)


def ram_columns(ram: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The RAM addresses kept: not excluded, and not constant over ``rows``."""
    sample = ram[rows[:: max(1, len(rows) // 50000)]]
    varying = (sample != sample[:1]).any(axis=0)
    return np.array([a for a in range(ram.shape[1]) if varying[a] and a not in EXCLUDED], dtype=np.int64)


def ram_inputs(ram_rows: np.ndarray) -> np.ndarray:
    """``[B, C]`` uint8 bytes → ``[B, C * 10]`` float32: value / 255, signed value / 128, the 8 bits."""
    x = ram_rows.astype(np.float32)
    signed = np.where(ram_rows >= 128, x - 256, x) / 128
    bits = np.unpackbits(ram_rows[..., None], axis=-1).astype(np.float32)
    return np.concatenate([(x / 255)[..., None], signed[..., None], bits], axis=-1).reshape(len(ram_rows), -1)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def matched(scores: np.ndarray, labels: np.ndarray) -> float:
    """Precision (= recall) when as many are refused as the look-ahead refused: the top-k by score, k the hits."""
    k = int(labels.sum())
    if k == 0:
        return float('nan')
    top = np.argpartition(-scores, k - 1)[:k]
    return float(labels[top].mean())


def bootstrap_pr(scores, labels, groups, n: int = 200, seed: int = 0) -> float:
    """Standard error of ``matched`` by resampling ``groups`` (episodes)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = np.split(np.argsort(inv, kind='stable'), np.cumsum(np.bincount(inv))[:-1])
    out = []
    for _ in range(n):
        pick = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        out.append(matched(scores[pick], labels[pick]))
    return float(np.nanstd(out))


def summary(data: Data, scores: np.ndarray, mask: np.ndarray) -> dict:
    """Round 35's held-out numbers for candidate scores ``scores`` (aligned with ``data.c_*[mask]``)."""
    y = data.c_lost[mask]
    eps = 1e-6
    s = np.clip(scores, eps, 1 - eps)
    rate = float(y.mean())
    out = {'n': int(mask.sum()), 'rate': rate, 'auc': auc(scores, y),
           'log_loss': float(-(y * np.log(s) + (~y) * np.log(1 - s)).mean()),
           'log_loss_base': float(-(rate * np.log(rate) + (1 - rate) * np.log(1 - rate)))}
    own = data.c_own[mask]
    rows = data.c_row[mask]
    episodes = data.d['row_episode'][rows]
    for mode in (0, 1):
        m = own & (data.mode_of_row[rows] == mode)
        if m.any():
            out[f'draw{mode}'] = {'n': int(m.sum()), 'lost': int(y[m].sum()), 'auc': auc(scores[m], y[m]),
                                  'pr': matched(scores[m], y[m]), 'pr_se': bootstrap_pr(scores[m], y[m], episodes[m])}
    return out


LINK_SPEED = 1.5        # pixels per frame (measured: MOVE 4 frames moves him 6)
HORIZON = 32            # frames the contact features look ahead
SCENARIOS = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]  # Link stands, walks N, S, W, E (the action index order)
GEOMETRY = ['harmful', 'ddx', 'ddy', 'gap_x', 'gap_y', 'vx', 'vy', 'aligned'] + \
    [f'contact_{n}' for n in ('stand', 'N', 'S', 'W', 'E')] + [f'closest_{n}' for n in ('stand', 'N', 'S', 'W', 'E')]
TIMING = ['to_grid', 'obj_timer', 'wants_shoot', 'waiting', 'turn_timer', 'invincible']


def _extrapolate(ddx, ddy, vx, vy, harmful, out: dict) -> None:
    """The contact and closest-approach columns: both in straight lines for ``HORIZON`` frames, Link standing or
    walking N, S, W, E; the game's touch test is both middle distances under 9 pixels."""
    t = np.arange(HORIZON + 1, dtype=np.float32)
    for name, (lxv, lyv) in zip(('stand', 'N', 'S', 'W', 'E'), SCENARIOS):
        rx = ddx[..., None] + (vx - lxv * LINK_SPEED)[..., None] * t
        ry = ddy[..., None] + (vy - lyv * LINK_SPEED)[..., None] * t
        gap = np.maximum(np.abs(rx), np.abs(ry)) - 9
        touch = gap < 0
        first = np.where(touch.any(-1), touch.argmax(-1), HORIZON + 1)
        out[f'contact_{name}'] = np.where(harmful, (HORIZON + 1 - first) / (HORIZON + 1), 0).astype(np.float32)
        out[f'closest_{name}'] = np.where(harmful, np.clip(gap.min(-1), 0, 64) / 32, 2).astype(np.float32)


def contact_from_obs(entity_features: np.ndarray) -> np.ndarray:
    """The ``geometry`` columns of ``contact_features`` computed from the observation alone (per observation entity
    slot): ``harmful`` its ``touch_harms``, the offsets ``dx`` x 255 and ``dy`` x 176 pixels (corner to corner: the
    half widths taken as equal), the velocities ``vx``, ``vy`` x 4 pixels per frame (measured since the last
    observation). ``[N, 12 * len(GEOMETRY)]``: what the network could compute itself, put in the game's terms."""
    from triforce import _native  # pylint: disable=import-outside-toplevel
    names = list(_native.ENTITY_FEATURES)
    col = {n: names.index(n) for n in ('presence', 'dx', 'dy', 'vx', 'vy', 'touch_harms')}
    ef = entity_features.astype(np.float32)
    harmful = ef[..., col['touch_harms']] > 0
    ddx, ddy = ef[..., col['dx']] * 255, ef[..., col['dy']] * 176
    vx, vy = ef[..., col['vx']] * 4, ef[..., col['vy']] * 4
    out = {'harmful': harmful.astype(np.float32), 'ddx': ddx / 32, 'ddy': ddy / 32,
           'gap_x': np.maximum(np.abs(ddx) - 8, 0) / 32, 'gap_y': np.maximum(np.abs(ddy) - 8, 0) / 32,
           'vx': vx, 'vy': vy, 'aligned': ((np.abs(ddx) < 9) | (np.abs(ddy) < 9)).astype(np.float32)}
    _extrapolate(ddx, ddy, vx, vy, harmful, out)
    present = (ef[..., col['presence']] > 0)[..., None]
    per = np.stack([out[k].astype(np.float32) for k in GEOMETRY], axis=-1) * present
    return per.reshape(len(ef), -1)


def contact_features(ram: np.ndarray, which: str = 'all') -> np.ndarray:
    """Per object slot 1-12, what the game's collision rule makes of the RAM at a decision (round 36's representation
    test): ``[N, 12 * F + 1]`` float32. ``geometry``: whether the slot can harm Link on touch now (live, not dying or
    stunned; a shot only flying out, asm:Link_CheckMonsterTouch_NoReset), the offset of its middle from Link's
    (asm:Obj_GetMiddle; / 32), the gap per axis before the game's 9-pixel touch test passes, its velocity from ObjDir
    and its q-speed (/ 64 px per frame), whether it is in Link's row or column, and, extrapolating both in straight
    lines for 32 frames with Link standing or walking N, S, W, E at 1.5 px per frame, how soon they touch ((33 - t) /
    33, 0 never) and how close they come (gap / 32). ``timing``: frames to its next grid point at its speed (where a
    walker turns; / 32), its ObjTimer (/ 64), Shooter_ObjWantsToShoot, a Goriya waiting for its boomerang (state bit
    7), Wanderer_ObjTurnTimer (/ 256), its invincibility timer (/ 32). Last column: Link's own invincibility timer
    (/ 32)."""
    n = len(ram)
    slots = np.arange(1, 13)
    r = ram.astype(np.int32)
    typ, state = r[:, 0x34F + slots], r[:, 0xAC + slots]
    meta, stun = r[:, 0x405 + slots], r[:, 0x3D + slots]
    attr, direction, qspeed = r[:, 0x4BF + slots], r[:, 0x98 + slots], r[:, 0x3BC + slots]
    harmful = (typ != 0) & (meta == 0) & (stun == 0) & ((typ < 0x53) | ((state & 0xF0) == 0x10))
    mx = r[:, 0x70 + slots] + np.where(attr & 0x40, 4, 8)
    my = r[:, 0x84 + slots] + 8
    ddx = ((mx - (r[:, 0x70:0x71] + 8) + 128) % 256 - 128).astype(np.float32)
    ddy = ((my - (r[:, 0x84:0x85] + 8) + 128) % 256 - 128).astype(np.float32)
    speed = qspeed / 64.0
    vx = np.where(direction & 1, speed, 0) - np.where(direction & 2, speed, 0)
    vy = np.where(direction & 4, speed, 0) - np.where(direction & 8, speed, 0)
    out = {}
    out['harmful'] = harmful.astype(np.float32)
    out['ddx'], out['ddy'] = ddx / 32, ddy / 32
    out['gap_x'] = np.maximum(np.abs(ddx) - 8, 0) / 32
    out['gap_y'] = np.maximum(np.abs(ddy) - 8, 0) / 32
    out['vx'], out['vy'] = vx, vy
    out['aligned'] = ((np.abs(ddx) < 9) | (np.abs(ddy) < 9)).astype(np.float32)
    _extrapolate(ddx, ddy, vx, vy, harmful, out)
    grid = r[:, 0x394 + slots]
    grid = np.where(grid >= 128, grid - 256, grid)
    out['to_grid'] = np.where(speed > 0, np.clip((16 - np.abs(grid)) / np.maximum(speed, 1e-3), 0, 64) / 32, 2)
    out['obj_timer'] = np.minimum(r[:, 0x28 + slots] / 64, 4)
    out['wants_shoot'] = (r[:, 0x412 + slots] != 0) & np.isin(typ, [0x01, 0x02, 0x03, 0x04, 0x05, 0x06])
    out['waiting'] = ((state & 0x80) != 0) & np.isin(typ, [0x05, 0x06])
    out['turn_timer'] = r[:, 0x478 + slots] / 256
    out['invincible'] = np.minimum(r[:, 0x4F0 + slots] / 32, 2)
    names = GEOMETRY if which == 'geometry' else TIMING if which == 'timing' else GEOMETRY + TIMING
    live = (typ != 0)[..., None]
    per = np.stack([out[k].astype(np.float32) for k in names], axis=-1) * live
    return np.concatenate([per.reshape(n, -1), np.minimum(r[:, 0x4F0:0x4F1] / 32, 2).astype(np.float32)], axis=1)
