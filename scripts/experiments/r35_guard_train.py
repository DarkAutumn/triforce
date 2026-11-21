"""Round 35 (docs/experiments/round35.md "The guard"): trains a learned guard (``triforce.guard``) on the look-ahead's
verdicts (``r35_guard_data.py``) and measures it on held-out episodes.

    python scripts/experiments/r35_guard_train.py CHECKPOINT DATA.npz OUT_DIR [--epochs 6] [--holdout 7]

The guard starts as the family's policy (``CHECKPOINT``) with a hit head (one logit per joint action) and learns, by
binary cross-entropy on every judged candidate's verdict on future 0 (the future ``plan:16:1`` decides on), whether
the action loses health before the next decision. Episodes whose index is a multiple of ``--holdout`` are held out.

Measured on the held-out episodes (``OUT_DIR/report.json``, printed):
- every judged candidate: the base rate, ROC AUC, the log loss against the base rate's;
- the decision's own draw (the first candidate judged, the one a veto refuses or keeps), per mode (0 planner, 1
  model alone): at the threshold where the guard refuses as many of the held-out draws as the look-ahead did, and at
  fixed thresholds, the veto's precision and recall;
- the ceiling: a second redrawn future (``--audit`` in the data) as the predictor of the first, precision and recall
  (what a judge that knows the emulator's whole state but the dice reaches).

``OUT_DIR/guard.pt`` (``triforce.guard.save_guard``) goes beside the run's ``ckpt.pt`` to play ``veto:K:P``.
It trains, so it is a job: it waits its turn in the machine's line (``triforce.jobs``).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from triforce import registry
from triforce.events import logged_job
from triforce.guard import hit_logits, load_guard, make_guard, save_guard
from triforce.jobs import add_job_arguments, begin_job, work
from triforce.model import MASK_KEY


def load(path):
    with np.load(path) as data:
        d = {k: data[k] for k in data.files}
    obs = {k[len('obs/'):]: v for k, v in d.items() if k.startswith('obs/')}
    obs[MASK_KEY] = d['mask']
    return d, obs


def batch_obs(obs: dict, rows: np.ndarray) -> dict:
    out = {}
    for k, v in obs.items():
        a = v[rows]
        if k == 'grid':
            a = a.astype(np.int64)
        out[k] = torch.as_tensor(a)
    return out


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def pr(pred: np.ndarray, labels: np.ndarray) -> dict:
    tp = int((pred & labels).sum())
    return {'vetoes': int(pred.sum()), 'hits': int(labels.sum()), 'n': int(len(labels)),
            'precision': tp / max(int(pred.sum()), 1), 'recall': tp / max(int(labels.sum()), 1)}


def scores_of(guard, obs, rows, joints, batch=2048) -> np.ndarray:
    out = []
    guard.eval()
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            r, j = rows[i:i + batch], joints[i:i + batch]
            logits = hit_logits(guard, batch_obs(obs, r))
            out.append(torch.sigmoid(logits[torch.arange(len(r)), torch.as_tensor(j, dtype=torch.long)]).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def report(guard, d, obs, held_rows_mask, mode_of_row) -> dict:
    row, joint, future, lost = d['tried_row'], d['tried_joint'].astype(np.int64), d['tried_future'], d['tried_lost']
    f0 = (future == 0) & held_rows_mask[row]
    s = scores_of(guard, obs, row[f0], joint[f0])
    y = lost[f0]
    rate = float(y.mean())
    eps = 1e-6
    ll = float(-(y * np.log(s + eps) + (~y) * np.log(1 - s + eps)).mean())
    ll0 = float(-(rate * np.log(rate) + (1 - rate) * np.log(1 - rate)))
    out = {'candidates': {'n': int(f0.sum()), 'base_rate': rate, 'auc': auc(s, y), 'log_loss': ll, 'log_loss_base': ll0}}
    # the decision's own draw: the first future-0 entry of each decision row
    rows_f0 = row[f0]
    first = np.r_[True, rows_f0[1:] != rows_f0[:-1]]
    own_s, own_y, own_mode = s[first], y[first], mode_of_row[rows_f0[first]]
    for mode in (0, 1):
        m = own_mode == mode
        if not m.any():
            continue
        ys, ss = own_y[m], own_s[m]
        k = int(ys.sum())
        matched = float(np.sort(ss)[::-1][max(k - 1, 0)]) if k else 1.0
        entry = {'decisions': int(m.sum()), 'lost': k, 'rate': float(ys.mean()), 'auc': auc(ss, ys),
                 'matched_threshold': matched, 'matched': pr(ss >= matched, ys)}
        for t in (0.2, 0.3, 0.5, 0.7):
            entry[f'at_{t}'] = pr(ss > t, ys)
        out[f'own_mode{mode}'] = entry
    # the ceiling: a second redrawn future of the same (row, joint) as the predictor of the first
    key = row.astype(np.int64) * 512 + joint
    h = held_rows_mask[row]
    a0 = dict(zip(key[h & (future == 0)], lost[h & (future == 0)]))
    a1 = dict(zip(key[h & (future == 1)], lost[h & (future == 1)]))
    both = [k for k in a1 if k in a0]
    if both:
        y0 = np.array([a0[k] for k in both])
        y1 = np.array([a1[k] for k in both])
        out['ceiling'] = {'pairs': len(both), 'agree': float((y0 == y1).mean()), **pr(y1, y0)}
    return out


def train(checkpoint: Path, data: Path, out_dir: Path, epochs: int, holdout: int, lr: float, batch: int,
          threads: int, seed: int):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    out_dir.mkdir(parents=True, exist_ok=True)
    d, obs = load(data)
    episodes = len(d['ep_seed'])
    held_ep = (np.arange(episodes) % holdout) == 0
    held_rows = held_ep[d['row_episode']]
    mode_of_row = d['ep_mode'][d['row_episode']]
    row, joint, future, lost = d['tried_row'], d['tried_joint'].astype(np.int64), d['tried_future'], d['tried_lost']
    train_mask = (future == 0) & ~held_rows[row]
    t_row, t_joint, t_y = row[train_mask], joint[train_mask], lost[train_mask].astype(np.float32)
    base = float(t_y.mean())
    print(f"{len(d['mask'])} decisions, {episodes} episodes ({int(held_ep.sum())} held out); training on "
          f"{len(t_y)} judged candidates, {int(t_y.sum())} lost (base rate {base:.4f})", flush=True)
    network = registry.load_model(checkpoint)[0]
    guard = make_guard(network, base)
    optimizer = torch.optim.AdamW(guard.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(t_y) + batch - 1) // batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        guard.train()
        order = rng.permutation(len(t_y))
        total, n = 0.0, 0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            logits = hit_logits(guard, batch_obs(obs, t_row[idx]))
            picked = logits[torch.arange(len(idx)), torch.as_tensor(t_joint[idx])]
            loss = F.binary_cross_entropy_with_logits(picked, torch.as_tensor(t_y[idx]))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(guard.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            total += loss.item() * len(idx)
            n += len(idx)
        result = report(guard, d, obs, held_rows, mode_of_row)
        history.append({'epoch': epoch + 1, 'train_loss': total / n, **result})
        print(json.dumps(history[-1]), flush=True)
        save_guard(out_dir / f'guard-e{epoch + 1}.pt', guard, {'epoch': epoch + 1, 'data': str(data.name)})
    save_guard(out_dir / 'guard.pt', guard, {'epochs': epochs, 'data': str(data.name), 'base_rate': base,
                                             'report': history[-1]})
    (out_dir / 'report.json').write_text(json.dumps(history, indent=1))


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path, help="the family's run dir: the guard starts as its network")
    parser.add_argument('data', type=Path)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--holdout', type=int, default=7, help="every Nth episode is held out")
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=35)
    parser.add_argument('--evaluate', type=Path, default=None,
                        help="measure this guard.pt on every episode of DATA (no training; OUT_DIR gets report.json)")
    add_job_arguments(parser)
    args = parser.parse_args(argv)
    nice = begin_job(args)
    if args.evaluate is not None:
        torch.set_num_threads(args.threads)
        d, obs = load(args.data)
        everything = np.ones(len(d['mask']), dtype=bool)
        result = report(load_guard(args.evaluate), d, obs, everything, d['ep_mode'][d['row_episode']])
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / 'report.json').write_text(json.dumps(result, indent=1))
        print(json.dumps(result))
        return 0
    with np.load(args.data) as data:
        samples = int((data['tried_future'] == 0).sum())
    with logged_job('r35-guard-train', 'train', [{'specialist': 'goriya', 'run_id': args.out_dir.name,
                                                  'status': 'training'}], nice=nice,
                    work=work(train_steps=args.epochs * samples // 4)):
        train(args.checkpoint, args.data, args.out_dir, args.epochs, args.holdout, args.lr, args.batch,
              args.threads, args.seed)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
