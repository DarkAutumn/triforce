"""Round 36 (docs/experiments/round36.md "What the observation lacks"): the oracle test. The same hit classifier as
round 35's guard (``triforce.guard``: the policy's network, one hit logit per joint action, initialised from the
family's checkpoint) trained on the look-ahead's verdicts with different inputs:

    python scripts/experiments/r36_hit_oracle.py DATA.npz CHECKPOINT OUT.json --inputs obs|obs+ram|ram
        [--drop GROUP ...] [--only GROUP ...] [--train-frac F] [--epochs 10] [--horizon K] [--init policy|random]
        [--scores OUT.npz] [--device cuda]

- ``obs``: the policy's observation only (round 35's guard);
- ``obs+ram``: the same network with the RAM at the decision beside it: an MLP of the kept bytes (``r36_hit_common.
  ram_inputs``) whose output is added to the network's features through a zero-initialised projection (it starts as
  the ``obs`` network);
- ``ram``: the RAM alone, an MLP to the 352 logits.
``--drop`` / ``--only``: RAM groups (``r36_hit_common.RAM_GROUPS``; ``other`` is the rest) left out / kept alone.
``--train-frac``: a share of the training episodes (the learning curve). ``--horizon K``: the label is "a loss that
lands within the action's first K frames" (later losses count as safe). ``--init random``: the network from scratch.
Held out as round 35 (every 7th episode); the numbers (``r36_hit_common.summary``) after every epoch into OUT.json;
``--scores``: the held-out candidates' final scores.  Training on the GPU is a job of the machine's queue
(``jobs.require_job``: ``triforce job run ... --timeout 1h -- .venv/bin/python scripts/experiments/r36_hit_oracle.py
...``).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r36_hit_common import contact_features, contact_from_obs, group_of, load, ram_columns, summary  # noqa: E402

from triforce import registry  # noqa: E402
from triforce.jobs import require_job  # noqa: E402
from triforce.guard import JOINT, make_guard  # noqa: E402
from triforce.model import MASK_KEY, ZeldaPolicy  # noqa: E402


def ram_encode(x: torch.Tensor) -> torch.Tensor:
    """uint8 ``[B, C]`` → ``[B, C * 10]`` (``r36_hit_common.ram_inputs`` on the device)."""
    f = x.float()
    signed = torch.where(x >= 128, f - 256, f) / 128
    bits = ((x.long().unsqueeze(-1) >> torch.arange(7, -1, -1, device=x.device)) & 1).float()
    return torch.cat([(f / 255).unsqueeze(-1), signed.unsqueeze(-1), bits], dim=-1).flatten(1)


def ram_smooth(x: torch.Tensor) -> torch.Tensor:
    """uint8 ``[B, C]`` → ``[B, C * 2]``: value / 255 and signed value / 128 (``--smooth``)."""
    f = x.float()
    return torch.cat([f / 255, torch.where(x >= 128, f - 256, f) / 128], dim=-1)


class Oracle(nn.Module):
    def __init__(self, guard: ZeldaPolicy | None, n_in: int, feature_dim: int, encode):
        super().__init__()
        self.guard = guard
        self.n_in = n_in
        self.encode = encode
        if n_in:
            self.ram = nn.Sequential(nn.Linear(n_in, 512), nn.LayerNorm(512), nn.ReLU(),
                                     nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU())
            if guard is not None:
                self.proj = nn.Linear(256, feature_dim)
                nn.init.zeros_(self.proj.weight)
                nn.init.zeros_(self.proj.bias)
            else:
                self.head = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, JOINT))

    def forward(self, obs: dict, extra: torch.Tensor | None) -> torch.Tensor:
        if self.guard is None:
            return self.head(self.ram(self.encode(extra)))
        features = self.guard._features(obs)  # pylint: disable=protected-access
        if self.n_in:
            features = features + self.proj(self.ram(self.encode(extra)))
        _, value = self.guard.mlp_extractor(features)
        return self.guard.value_head(value)


def to_device(obs: dict, keys, device) -> dict:
    out = {}
    for k in keys:
        v = torch.as_tensor(obs[k])
        out[k] = v.to(device)
    return out


def gather(obs_dev: dict, rows: torch.Tensor) -> dict:
    out = {}
    for k, v in obs_dev.items():
        a = v[rows]
        out[k] = a.long() if k in ('grid', 'entity_types', 'room_id') else a
    return out


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--inputs', choices=['obs', 'obs+ram', 'ram', 'obs+contact', 'contact'], default='obs')
    parser.add_argument('--contact', choices=['all', 'geometry', 'timing', 'obs'], default='all',
                        help="which contact features (r36_hit_common.contact_features) with --inputs *contact")
    parser.add_argument('--drop', nargs='*', default=[])
    parser.add_argument('--smooth', action='store_true',
                        help="RAM bytes as value / 255 and signed value / 128 only (no bits: less to memorise)")
    parser.add_argument('--only', nargs='*', default=[])
    parser.add_argument('--train-frac', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--horizon', type=int, default=0)
    parser.add_argument('--init', choices=['policy', 'random'], default='policy')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pos-weight', type=float, default=1.0, help="weight of a lost candidate in the loss")
    parser.add_argument('--train-on', choices=['all', 'own', 'taken'], default='all',
                        help="training labels: every judged candidate; only each decision's own draw; only the "
                             "draws of the model-alone episodes (plan:1:1: the action taken, its real outcome)")
    parser.add_argument('--scores', type=Path, default=None)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    require_job()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    t0 = time.time()
    data = load(args.data)
    d = data.d
    y_all = data.c_lost.copy()
    if args.horizon:
        hf = d['tried_hit_frame'][data.c_index]
        y_all = y_all & (hf >= 0) & (hf < args.horizon)
    held_c = data.held_rows[data.c_row]
    train_c = np.flatnonzero(~held_c)
    if args.train_frac < 1:
        episodes = np.unique(d['row_episode'][~data.held_rows])
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(episodes, int(round(len(episodes) * args.train_frac)), replace=False)
        train_c = train_c[np.isin(d['row_episode'][data.c_row[train_c]], keep)]
    if args.train_on != 'all':
        keep = data.c_own[train_c]
        if args.train_on == 'taken':
            keep &= data.mode_of_row[data.c_row[train_c]] == 1
        train_c = train_c[keep]
    held_idx = np.flatnonzero(held_c)

    use_obs = args.inputs.startswith('obs')
    use_ram = args.inputs.endswith('ram')
    use_contact = args.inputs.endswith('contact')
    columns = np.zeros(0, dtype=np.int64)
    if use_ram:
        columns = ram_columns(d['ram'], data.c_row[train_c])
        groups = np.array([group_of(int(a)) for a in columns])
        if args.only:
            columns = columns[np.isin(groups, args.only)]
        elif args.drop:
            columns = columns[~np.isin(groups, args.drop)]
    network = registry.load_model(args.checkpoint)[0]
    guard = None
    feature_dim = 0
    if use_obs:
        guard = make_guard(network, float(y_all[train_c].mean()))
        if args.init == 'random':
            fresh = ZeldaPolicy(config=network.config.to_dict())
            guard = make_guard(fresh, float(y_all[train_c].mean()))
        keys = [k for k in guard.config.obs_spec if k in data.obs]
        obs_dev = to_device(data.obs, keys, device)
        guard.to(device)
        with torch.no_grad():
            feature_dim = guard._features(gather(obs_dev, torch.zeros(2, dtype=torch.long, device=device))).shape[-1]  # pylint: disable=protected-access
    else:
        obs_dev = {}
    if use_contact:
        source = d['ram'] if args.contact != 'obs' else data.obs['entity_features']
        make = (lambda a: contact_from_obs(a)) if args.contact == 'obs' else (lambda a: contact_features(a, args.contact))
        extra = np.concatenate([make(source[i:i + 50000]) for i in range(0, len(source), 50000)]).astype(np.float16)
        extra_dev = torch.as_tensor(extra).to(device)
        n_in, encode = extra.shape[1], lambda x: x.float()
    elif use_ram:
        extra_dev = torch.as_tensor(d['ram'][:, columns]).to(device)
        n_in, encode = len(columns) * (2 if args.smooth else 10), (ram_smooth if args.smooth else ram_encode)
    else:
        extra_dev, n_in, encode = None, 0, None
    model = Oracle(guard, n_in, feature_dim, encode).to(device)
    if guard is None:
        rate = float(y_all[train_c].mean())
        nn.init.constant_(model.head[-1].bias, float(np.log(rate / (1 - rate))))
    rows_dev = torch.as_tensor(data.c_row).to(device)
    joint_dev = torch.as_tensor(data.c_joint).to(device)
    y_dev = torch.as_tensor(y_all.astype(np.float32)).to(device)
    print(f"{args.inputs} drop={args.drop} only={args.only}: {len(train_c)} training candidates "
          f"(rate {y_all[train_c].mean():.4f}), {len(held_idx)} held out, {len(columns)} RAM bytes, {n_in} inputs; "
          f"loaded in {time.time() - t0:.0f}s", flush=True)

    def logits_of(idx: torch.Tensor) -> torch.Tensor:
        rows = rows_dev[idx]
        obs = gather(obs_dev, rows) if use_obs else {}
        ram = extra_dev[rows] if extra_dev is not None else None
        out = model(obs, ram)
        return out[torch.arange(len(idx), device=device), joint_dev[idx]]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(train_c) + args.batch - 1) // args.batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    history = []
    gen = torch.Generator(device='cpu').manual_seed(args.seed)
    train_dev = torch.as_tensor(train_c).to(device)
    held_dev = torch.as_tensor(held_idx).to(device)
    labels_for_summary = data.c_lost
    data.c_lost = y_all
    for epoch in range(args.epochs):
        model.train()
        order = train_dev[torch.randperm(len(train_dev), generator=gen).to(device)]
        total = 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            loss = F.binary_cross_entropy_with_logits(logits_of(idx), y_dev[idx],
                                                      pos_weight=torch.tensor(args.pos_weight, device=device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            scores = torch.cat([torch.sigmoid(logits_of(held_dev[i:i + 8192]))
                                for i in range(0, len(held_dev), 8192)]).cpu().numpy()
        result = {'epoch': epoch + 1, 'train_loss': total / len(order), 'seconds': round(time.time() - t0),
                  **summary(data, scores, held_c)}
        history.append(result)
        print(json.dumps(result), flush=True)
    data.c_lost = labels_for_summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'args': {k: str(v) for k, v in vars(args).items()},
                                    'ram_columns': columns.tolist(), 'history': history}, indent=1))
    if args.scores:
        np.savez_compressed(args.scores, index=data.c_index[held_idx], scores=scores, label=y_all[held_idx])
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
