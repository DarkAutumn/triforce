"""Round 40 (docs/experiments/round40.md "Observation v7's danger block"): the supervised hit bench on the input
groups of the fresh generation's observation, each on and off.

    python scripts/experiments/r40_obs_bench.py ARM --data target/r40/data/goriya [--seed 0] [--epochs 10]
    python scripts/experiments/r40_obs_bench.py --table [--results target/r40/bench.jsonl]

Data: ``r40_hit_data.py``'s shards (the teacher's own decisions, the RAM beside each; every 7th episode held out).
Trained as experiment 2's bench (round36.md §7): 10 epochs, AdamW 4e-4 one-cycle, batch 1024, BCE on every judged
candidate (the action played and the look-ahead's others) of the training episodes.

**Arms** (``+``-joined): ``base`` today's policy shape (``r36_hit_bench``'s: the entity transformer mean-pooled,
one hit logit a joint action), from scratch; otherwise ``hitnet`` (``triforce/nets/hitnet.py``: Link's token over the
entity tokens, the hit head on its own tower) with the token groups named: ``type`` and ``features`` always, then
``rel`` (schema 7's ``entity_rel``), ``contact`` (schema 7's ``entity_contact``: 32 frames, unbounded), ``threat``
(schema 8's ``entity_threat`` from ``_native.threat_blocks``: ``soon``, ``near`` and ``lane`` columns), or parts of it
(``soon``, ``near``, ``lane``); ``hN`` its horizon (default the engine's), ``slackN`` the lane's slack (frames, ``inf``
none), ``noinv`` without the invincibility rule, ``noroom`` without the room embedding.

**Scores** (held-out episodes): precision = recall at the look-ahead's rate on the teacher's own draws (``draw``) and
on every judged candidate (``all``), their AUC, the held-out log loss; and two slices of the lost candidates, each
scored by its recall at the threshold where all held-out candidates have precision = recall, and by its AUC against
every safe candidate: ``shot`` (approximately the hits by a shot fired after the decision: a straight shot that was
not in its slot at the decision is in the RAM at the trial's end, or no object present at the decision touches Link
on the walled extrapolation within 32 frames while a shooter that can fire is in the room) and ``lane`` (Link stood in
a ready shooter's lane at the decision).
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'experiments'))

import r36_hit_bench as hb  # noqa: E402
from r36_hit_common import auc  # noqa: E402
from triforce import _native, jobs, nets, registry  # noqa: E402
from triforce.observation import ENTITY_THREAT_FEATURES  # noqa: E402

R40 = ROOT / 'target' / 'r40'
VECTOR_KEYS = ['history', 'link', 'objective', 'room_state', 'target']
BASE_KEYS = ['room_id', 'entity_types', 'entity_features', 'grid'] + VECTOR_KEYS
THREAT_PARTS = {'soon': list(range(0, 5)), 'near': list(range(5, 10)), 'lane': list(range(10, 14))}
SHOTS = np.array([0x53, 0x57, 0x5B, 0x5C])
HOLDOUT = 7


# ---------------------------------------------------------------------------------------------- data

class Data:
    """Decisions (``obs``, ``ram``, ``held``) and judged candidates (``c_row``, ``c_joint``, ``c_lost``,
    ``c_taken``, ``c_end``: their row into ``end_ram`` when lost)."""

    def __init__(self, directories):
        parts, rows = [], 0
        for directory in directories:
            for path in sorted(Path(directory).glob('shard-*.npz')):
                with np.load(path) as z:
                    d = {k: z[k] for k in z.files}
                d['tried_row'] = d['tried_row'] + rows
                rows += len(d['taken'])
                parts.append(d)
        if not parts:
            raise SystemExit(f'no shards in {directories}')
        ends, offset = [], 0
        for d in parts:
            d['tried_end'] = np.where(d['tried_end'] >= 0, d['tried_end'] + offset, -1)
            offset += len(d['tried_end_ram'])
            ends.append(d['tried_end_ram'])
        cat = lambda k: np.concatenate([d[k] for d in parts])  # noqa: E731
        self.obs = {k[4:]: cat(k) for k in parts[0] if k.startswith('obs/')}
        self.ram = cat('ram')
        episode = cat('episode')
        self.held = (np.unique(episode, return_inverse=True)[1] % HOLDOUT) == 0
        self.c_row, self.c_joint, self.c_lost = cat('tried_row'), cat('tried_joint'), cat('tried_lost')
        self.c_taken, self.c_end = cat('tried_taken').astype(bool), cat('tried_end')
        self.end_ram = np.concatenate(ends)
        self.rows = len(self.ram)

    def grid_u8(self):
        return np.ascontiguousarray(self.obs['grid'].astype(np.uint8))


def columns(data: Data, horizon=None, slack=None, invincibility=True) -> dict:
    """Schema 7's blocks and schema 8's ``entity_threat`` (with these choices) of every row, from the engine."""
    t0 = time.time()
    grid, feats = data.grid_u8(), np.ascontiguousarray(data.obs['entity_features'].astype(np.float32))
    rel, con, thr = [], [], []
    kwargs = {'invincibility': invincibility}
    if horizon is not None:
        kwargs['horizon'] = horizon
    if slack is not None:
        kwargs['lane_slack'] = slack
    for lo in range(0, data.rows, 50000):
        hi = min(data.rows, lo + 50000)
        r, c = _native.danger_blocks(data.ram[lo:hi], grid[lo:hi], feats[lo:hi])
        rel.append(r)
        con.append(c)
        thr.append(_native.threat_blocks(data.ram[lo:hi], grid[lo:hi], feats[lo:hi], **kwargs))
    print(f'engine columns ({kwargs}): {data.rows} rows in {time.time() - t0:.0f}s', flush=True)
    return {'entity_rel': np.concatenate(rel), 'entity_contact': np.concatenate(con),
            'entity_threat': np.concatenate(thr)}


def slices(data: Data) -> dict[str, np.ndarray]:
    """Per judged candidate: ``shot`` and ``lane`` (the module docstring)."""
    cols = columns(data, horizon=32, slack=math.inf, invincibility=False)
    names = list(ENTITY_THREAT_FEATURES)
    wall = cols['entity_contact'][..., 13:18].max(axis=(1, 2))          # schema 7's walled contact, 32 frames
    ready = cols['entity_threat'][..., names.index('lane_ready')].max(axis=1) > 0
    in_lane = cols['entity_threat'][..., names.index('lane_in')].max(axis=1) > 0
    r = data.c_row
    lost = data.c_lost.astype(bool)
    slots = np.arange(1, 13)
    at = data.ram[r][:, 0x34F + slots]
    new_shot = np.zeros(len(r), bool)
    idx = np.flatnonzero(lost & (data.c_end >= 0))
    end = data.end_ram[data.c_end[idx]][:, 0x34F + slots]
    new_shot[idx] = (np.isin(end, SHOTS) & (end != at[idx])).any(axis=1)
    unexplained = (wall[r] <= 0) & ready[r]
    return {'shot': lost & (new_shot | unexplained), 'lane': lost & in_lane[r]}


# ---------------------------------------------------------------------------------------------- arms

def parse(arm: str) -> dict:
    spec = {'arm': arm, 'base': False, 'tokens': ['type', 'features'], 'threat': [], 'horizon': None, 'slack': None,
            'invincibility': True, 'room': True}
    for p in arm.split('+'):
        if p == 'base':
            spec['base'] = True
        elif p in ('rel', 'contact'):
            spec['tokens'].append(p)
        elif p == 'threat':
            spec['threat'] = sorted(set(spec['threat']) | set(THREAT_PARTS))
        elif p in THREAT_PARTS:
            spec['threat'] = sorted(set(spec['threat']) | {p})
        elif p.startswith('h') and p[1:].isdigit():
            spec['horizon'] = int(p[1:])
        elif p.startswith('slack'):
            spec['slack'] = math.inf if p[5:] == 'inf' else float(p[5:])
        elif p == 'noinv':
            spec['invincibility'] = False
        elif p == 'noroom':
            spec['room'] = False
        elif p != 'linktok':
            raise SystemExit(f'unknown arm part {p!r}')
    if spec['threat']:
        spec['tokens'].append('threat')
    return spec


def full_spec(widths: dict) -> dict:
    out = {k: {'shape': list(_native.OBS_SPEC[k]['shape']), 'vocab': _native.OBS_SPEC[k]['vocab']} for k in BASE_KEYS}
    out.update({k: {'shape': [12, w], 'vocab': None} for k, w in widths.items()})
    return out


class HitNetArm(torch.nn.Module):
    def __init__(self, spec: dict, widths: dict, rate: float):
        super().__init__()
        self.net = nets.create('hitnet', full_spec(widths), tokens=spec['tokens'], room=spec['room'], hit='joint',
                               hit_prior=rate)

    def parameter_count(self) -> int:
        skip = ('policy_tower.', 'value_tower.', 'kind_', 'direction_head.', 'duration_head.', 'value_head.')
        return sum(p.numel() for name, p in self.net.named_parameters() if not name.startswith(skip))

    def forward(self, batch, joint):
        return self.net.hit_logits(self.net.trunk(batch)).gather(1, joint.unsqueeze(-1)).squeeze(-1)


def matched(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """(precision = recall at the label rate, the score threshold there)."""
    k = int(labels.sum())
    if k == 0:
        return float('nan'), float('inf')
    top = np.argsort(-scores, kind='mergesort')[:k]
    return float(labels[top].mean()), float(scores[top[-1]])


def run(args) -> dict:
    spec = parse(args.arm)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    t0 = time.time()
    data = Data(args.data)
    meta = json.loads((Path(args.data[0]) / 'dataset.json').read_text())
    held_c = data.held[data.c_row]
    train_c = np.flatnonzero(~held_c)
    held_idx = np.flatnonzero(held_c)
    rate = float(data.c_lost[train_c].mean())
    dev = {k: torch.as_tensor(data.obs[k]).to(device) for k in BASE_KEYS}
    widths = {}
    if not spec['base']:
        cols = columns(data, spec['horizon'], spec['slack'], spec['invincibility'])
        if 'rel' in spec['tokens']:
            dev['entity_rel'] = torch.as_tensor(cols['entity_rel']).to(device)
            widths['entity_rel'] = cols['entity_rel'].shape[-1]
        if 'contact' in spec['tokens']:
            dev['entity_contact'] = torch.as_tensor(cols['entity_contact']).to(device)
            widths['entity_contact'] = cols['entity_contact'].shape[-1]
        if spec['threat']:
            keep = sorted(c for part in spec['threat'] for c in THREAT_PARTS[part])
            dev['entity_threat'] = torch.as_tensor(np.ascontiguousarray(cols['entity_threat'][..., keep])).to(device)
            widths['entity_threat'] = len(keep)
        net = HitNetArm(spec, widths, rate).to(device)
        params = net.parameter_count()
    else:
        obs_spec = registry.load_model(meta['checkpoint'])[0].config.obs_spec
        net = hb.BenchNet(hb.parse('base'), obs_spec, {'tokens': 0, 'linkx': 0, 'cflat': 0}, rate).to(device)
        params = sum(p.numel() for p in net.parameters())
    rows_dev = torch.as_tensor(data.c_row).to(device)
    joint_dev = torch.as_tensor(data.c_joint).to(device)
    y_dev = torch.as_tensor(data.c_lost.astype(np.float32)).to(device)
    print(f"{args.arm}: {params} parameters, {len(train_c)} training candidates (rate {rate:.4f}), "
          f"{len(held_idx)} held out; loaded in {time.time() - t0:.0f}s", flush=True)

    def forward(idx):
        batch = {k: v[rows_dev[idx]] for k, v in dev.items()}
        batch['grid'] = batch['grid'].long()
        batch['room_id'] = batch['room_id'].long()
        batch['entity_types'] = batch['entity_types'].long()
        return net(batch, joint_dev[idx])

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(train_c) + args.batch - 1) // args.batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    gen = torch.Generator(device='cpu').manual_seed(args.seed)
    train_dev = torch.as_tensor(train_c).to(device)
    held_dev = torch.as_tensor(held_idx).to(device)
    for epoch in range(args.epochs):
        net.train()
        order = train_dev[torch.randperm(len(train_dev), generator=gen).to(device)]
        total = 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            loss = F.binary_cross_entropy_with_logits(forward(idx), y_dev[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            total += loss.item() * len(idx)
        print(json.dumps({'epoch': epoch + 1, 'train_loss': round(total / len(order), 5)}), flush=True)
    net.eval()
    with torch.no_grad():
        scores = torch.cat([torch.sigmoid(forward(held_dev[i:i + 8192])) for i in range(0, len(held_dev), 8192)]
                           ).float().cpu().numpy()
    labels = data.c_lost[held_idx].astype(bool)
    taken = data.c_taken[held_idx]
    pr_all, threshold = matched(scores, labels)
    pr_draw, _ = matched(scores[taken], labels[taken])
    eps = 1e-6
    log_loss = float(-np.mean(np.where(labels, np.log(scores + eps), np.log(1 - scores + eps))))
    out = {'arm': args.arm, 'data': [str(p) for p in args.data], 'seed': args.seed, 'epochs': args.epochs,
           'parameters': int(params), 'train_candidates': int(len(train_c)), 'held_candidates': int(len(held_idx)),
           'held_lost': int(labels.sum()), 'held_draws': int(taken.sum()), 'held_draws_lost': int(labels[taken].sum()),
           'pr_draw': pr_draw, 'pr_all': pr_all, 'auc_draw': auc(scores[taken], labels[taken]),
           'auc_all': auc(scores, labels), 'log_loss': log_loss}
    sl = slices(data)
    safe = ~labels
    for name, mask in sl.items():
        m = mask[held_idx]
        out[f'{name}_n'] = int(m.sum())
        out[f'{name}_recall'] = float((scores[m] >= threshold).mean()) if m.any() else float('nan')
        sub = m | safe
        out[f'{name}_auc'] = auc(scores[sub], m[sub]) if m.any() else float('nan')
    out['minutes'] = round((time.time() - t0) / 60, 1)
    with open(args.results, 'a', encoding='utf-8') as f:
        f.write(json.dumps(out) + '\n')
    print(json.dumps(out), flush=True)
    return out


def table(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups: dict = {}
    for r in rows:
        groups.setdefault((tuple(r['data']), r['arm']), []).append(r)
    for data in sorted({k[0] for k in groups}):
        print(f"\n**{', '.join(Path(d).name for d in data)}**\n")
        print('| arm | seeds | P = R draws | P = R all | AUC all | shot recall / AUC (n) | lane recall / AUC (n) | '
              'log loss |')
        print('|---|---:|---|---|---:|---|---|---:|')
        arms = [(a, g) for (d, a), g in groups.items() if d == data]
        arms.sort(key=lambda x: -np.mean([r['pr_draw'] + r['pr_all'] for r in x[1]]))
        for arm, g in arms:
            m = lambda k: np.nanmean([r[k] for r in g])  # noqa: E731
            each = lambda k: ' / '.join(f"{r[k]:.3f}" for r in g)  # noqa: E731
            print(f"| {arm} | {len(g)} | {m('pr_draw'):.3f} ({each('pr_draw')}) | {m('pr_all'):.3f} ({each('pr_all')}) "
                  f"| {m('auc_all'):.3f} | {m('shot_recall'):.3f} / {m('shot_auc'):.3f} ({g[0]['shot_n']}) "
                  f"| {m('lane_recall'):.3f} / {m('lane_auc'):.3f} ({g[0]['lane_n']}) | {m('log_loss'):.4f} |")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('arm', nargs='?')
    parser.add_argument('--data', nargs='+', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--results', type=Path, default=R40 / 'bench.jsonl')
    parser.add_argument('--table', action='store_true')
    args = parser.parse_args(argv)
    if args.table:
        table(args.results)
        return 0
    jobs.require_job()
    args.results.parent.mkdir(parents=True, exist_ok=True)
    run(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
