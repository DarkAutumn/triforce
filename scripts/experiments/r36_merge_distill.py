"""Round 36, "One network by imitation": a student trained offline, on the GPU, to the teachers' distributions.

    python scripts/experiments/r36_merge_distill.py cache --data target/merge/data --cache target/merge/cache
    python scripts/experiments/r36_merge_distill.py train --cache target/merge/cache --name r36merge --run-id w4x \\
        --arch wide --arch-config '{"hidden_dim": 768, "depth": 3}' --epochs 8 [--fraction 0.25] [--init CKPT]

``cache``: every ``<data>/<key>/<source>/shard-*.npz`` (``r36_merge_record.py``) into one array a field
(``<cache>/<field>.npy``, compact dtypes, the teachers' joint log-probs as float16) with each row's key, source,
shard and env.  Several data dirs (``--data A B``: the teacher-acting data and DAgger rounds) concatenate.

``train``: KL(teacher ‖ student) over the 352 joint actions plus ``--value-coeff`` · ½(V − V_teacher)², Adam,
cosine decay.  Held out: env 9 of every rollout (each env's episodes are its own, so the held-out rows are whole
episodes, about 10%).  Reported per key on the held-out rows (``distill.json`` in the run): KL, top-1 agreement of
the joint action and of each head where it has a choice (kind; direction and duration given the teacher's kind),
the value's squared error.  The checkpoint is a normal run (``registry.save_checkpoint``), playable by the bench."""

import argparse
import json
import math
import os
import sys
import threading
import queue
import time
from pathlib import Path

import numpy as np
import torch

from triforce import nets, registry
from triforce.nets import roomheads  # noqa: F401  pylint: disable=unused-import  (registers 'roomheads')
from triforce.model import obs_spec_from_space
from triforce.observation import observation_space
from triforce.ppo import joint_kl

STEPS_PER_ENV = 512
HOLDOUT_ENV = 9
SOURCES = ('resets', 'entries', 'dagger', 'rooms')
DTYPES = {'obs/room_id': np.int16, 'obs/entity_types': np.uint8, 'obs/grid': np.uint8,
          'teacher_joint': np.float16}
SHAPE = (11, 8, 4)


def build_cache(data_dirs: list[str], cache: Path):
    shards = []
    for data in data_dirs:
        for meta in sorted(Path(data).glob('*/*/meta.json')):
            info = json.loads(meta.read_text())
            for path in sorted(meta.parent.glob('shard-*.npz')):
                shards.append((info['key'], meta.parent.name, Path(data).name, path))
    keys = sorted({s[0] for s in shards})
    tags = sorted({s[2] for s in shards})
    first = np.load(shards[0][3])
    fields = [f for f in first.files if f != 'action']
    rows = [np.load(s[3], mmap_mode=None)['teacher_value'].shape[0] for s in shards]
    total = sum(rows)
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for field in fields:
        dtype = DTYPES.get(field, first[field].dtype)
        out[field] = np.lib.format.open_memmap(cache / f'{field.replace("/", "_")}.npy', 'w+', dtype,
                                               (total, *first[field].shape[1:]))
    meta_arrays = {name: np.lib.format.open_memmap(cache / f'{name}.npy', 'w+', np.uint8, (total,))
                   for name in ('key', 'source', 'shard', 'env', 'data')}
    at = 0
    for (key, source, tag, path), n in zip(shards, rows):
        arrays = np.load(path)
        for field in fields:
            out[field][at:at + n] = arrays[field].astype(out[field].dtype)
        meta_arrays['key'][at:at + n] = keys.index(key)
        meta_arrays['source'][at:at + n] = SOURCES.index(source)
        meta_arrays['shard'][at:at + n] = int(path.stem.split('-')[1])
        meta_arrays['env'][at:at + n] = np.arange(n) // STEPS_PER_ENV
        meta_arrays['data'][at:at + n] = tags.index(tag)
        at += n
    for array in [*out.values(), *meta_arrays.values()]:
        array.flush()
    (cache / 'cache.json').write_text(json.dumps({'keys': keys, 'sources': list(SOURCES), 'data': tags,
                                                  'fields': fields, 'rows': total, 'from': data_dirs}, indent=1))
    print(f"cache: {total:,} rows, {len(keys)} keys, {len(shards)} shards -> {cache}", flush=True)


class Data:
    """The cache in memory (``mmap``: read on demand)."""

    def __init__(self, cache: Path, mmap: bool = False):
        self.meta = json.loads((cache / 'cache.json').read_text())
        self.fields = self.meta['fields']
        mode = 'r' if mmap else None
        self.arrays = {f: np.load(cache / f'{f.replace("/", "_")}.npy', mmap_mode=mode) for f in self.fields}
        for name in ('key', 'source', 'shard', 'env', 'data'):
            setattr(self, name, np.load(cache / f'{name}.npy'))
        self.keys = self.meta['keys']

    def batch(self, rows: np.ndarray, device) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = np.sort(rows)
        obs = {f[4:]: torch.from_numpy(np.ascontiguousarray(self.arrays[f][rows])).to(device, non_blocking=True)
               for f in self.fields if f.startswith('obs/')}
        obs['room_id'] = obs['room_id'].long()
        obs['entity_types'] = obs['entity_types'].long()
        obs['grid'] = obs['grid'].long()
        mask = torch.from_numpy(self.arrays['action_mask'][rows]).to(device)
        joint = torch.from_numpy(self.arrays['teacher_joint'][rows]).to(device).float()
        value = torch.from_numpy(self.arrays['teacher_value'][rows]).to(device)
        return obs, mask, joint, value


def head_agreement(teacher: torch.Tensor, student: torch.Tensor) -> dict:
    """Per row: top-1 agreement of the joint and of each head where the teacher has a choice (-1: no choice)."""
    t = teacher.view(-1, *SHAPE)
    s = student.view(-1, *SHAPE)
    rows = torch.arange(t.shape[0], device=t.device)
    out = {'joint': (teacher.argmax(-1) == student.argmax(-1)).float()}
    t_kind, s_kind = t.logsumexp((2, 3)), s.logsumexp((2, 3))
    k = t_kind.argmax(-1)
    t_dir, s_dir = t[rows, k].logsumexp(-1), s[rows, k].logsumexp(-1)
    t_dur, s_dur = t[rows, k].logsumexp(-2), s[rows, k].logsumexp(-2)
    for name, a, b in (('kind', t_kind, s_kind), ('direction', t_dir, s_dir), ('duration', t_dur, s_dur)):
        choice = torch.isfinite(a).sum(-1) > 1
        out[name] = torch.where(choice, (a.argmax(-1) == b.argmax(-1)).float(), -1.0)
    return out


@torch.no_grad()
def evaluate(network, data: Data, rows: np.ndarray, device, batch: int = 8192) -> dict:
    network.eval()
    keys = data.key[rows]
    sums: dict = {}
    for begin in range(0, len(rows), batch):
        part = rows[begin:begin + batch]
        part = part[np.argsort(part)]
        obs, mask, joint, value = data.batch(part, device)
        student = network.joint_log_probs(obs, mask)
        legal = torch.isfinite(joint)
        kl = (joint.exp() * torch.where(legal, joint - student, 0.0)).sum(-1)
        agree = head_agreement(joint, student)
        verr = (network.get_value(obs) - value).pow(2)
        k = torch.from_numpy(data.key[part].astype(np.int64)).to(device)
        for name, values in (('kl', kl), ('value_se', verr), *agree.items()):
            valid = values >= 0
            n = torch.zeros(len(data.keys), device=device).index_add_(0, k[valid], torch.ones_like(values[valid]))
            s = torch.zeros(len(data.keys), device=device).index_add_(0, k[valid], values[valid])
            acc = sums.setdefault(name, [torch.zeros_like(n), torch.zeros_like(s)])
            acc[0] += n
            acc[1] += s
    del keys
    result = {}
    for i, key in enumerate(data.keys):
        result[key] = {name: (float(s[i] / n[i]) if n[i] > 0 else None) for name, (n, s) in sums.items()}
        result[key]['rows'] = int(sums['kl'][0][i])
    total = {name: float(s.sum() / n.sum()) for name, (n, s) in sums.items()}
    total['rows'] = int(sums['kl'][0].sum())
    network.train()
    return {'total': total, 'keys': result}


def prefetch(data: Data, order: np.ndarray, batch: int, device, depth: int = 4):
    q: queue.Queue = queue.Queue(depth)

    def work():
        for begin in range(0, len(order) - batch + 1, batch):
            q.put(data.batch(order[begin:begin + batch], device))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def train(args):
    # pylint: disable=too-many-locals, too-many-statements
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    data = Data(Path(args.cache))
    held = data.env == HOLDOUT_ENV
    if args.sources:
        chosen = np.isin(data.source, [SOURCES.index(s) for s in args.sources])
    else:
        chosen = np.ones_like(held)
    shards = int(data.shard.max()) + 1
    keep = data.shard < max(1, math.ceil(shards * args.fraction))
    train_rows = np.flatnonzero(~held & chosen & keep)
    held = held & (data.shard < args.val_shards)
    val_rows = np.flatnonzero(held & (data.data == data.meta['data'].index(args.val_data))
                              if args.val_data else held)
    if args.init:
        network, _ = registry.load_model(args.init)
    else:
        spec = obs_spec_from_space(observation_space())
        network = nets.create(args.arch, spec, **json.loads(args.arch_config))
    network.to(device).train()
    for parameter in network.parameters():
        if parameter.numel() == 0:      # room_embed_dim 0: CUDA's embedding backward fails on an empty table
            parameter.requires_grad_(False)
    params = network.parameter_count()
    steps_per_epoch = len(train_rows) // args.batch
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, eps=1e-5)
    warmup = min(500, total_steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, warmup))
                                                 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps)))))
    root = registry.models_root(None)
    print(f"[distill] {args.name}/{args.run_id}: {params:,} parameters, {len(train_rows):,} train rows, "
          f"{len(val_rows):,} held out, {total_steps} steps", flush=True)
    history = []
    began = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(train_rows)
        kl_sum, v_sum, n = 0.0, 0.0, 0
        for obs, mask, joint, value in prefetch(data, order, args.batch, device):
            student = network.joint_log_probs(obs, mask)
            kl = joint_kl(joint, student)
            vloss = 0.5 * (network.get_value(obs) - value).pow(2).mean()
            loss = kl + args.value_coeff * vloss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            kl_sum += float(kl.detach())
            v_sum += float(vloss.detach())
            n += 1
        val = evaluate(network, data, val_rows, device)
        row = {'epoch': epoch, 'train_kl': kl_sum / max(1, n), 'train_value': v_sum / max(1, n),
               'val': val['total'], 'seconds': round(time.time() - began, 1)}
        history.append(row)
        print(f"[distill] epoch {epoch}: train kl {row['train_kl']:.4f} | held out kl {val['total']['kl']:.4f} "
              f"joint {val['total']['joint']:.3f} kind {val['total']['kind']:.3f} dir {val['total']['direction']:.3f} "
              f"dur {val['total']['duration']:.3f} value se {val['total']['value_se']:.3f} ({row['seconds']:.0f}s)",
              flush=True)
        path = registry.save_checkpoint(args.name, args.run_id, network, iterations=epoch, steps=step,
                                        root=root, extra={'distill': {**vars(args), 'parameters': params}})
    (path.parent / 'distill.json').write_text(json.dumps(
        {'args': vars(args), 'parameters': params, 'train_rows': int(len(train_rows)),
         'held_out_rows': int(len(val_rows)), 'history': history, 'held_out': val}, indent=1))
    print(f"[distill] done: {path}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('cache')
    c.add_argument('--data', nargs='+', required=True)
    c.add_argument('--cache', required=True)
    t = sub.add_parser('train')
    t.add_argument('--cache', required=True)
    t.add_argument('--name', default='r36merge')
    t.add_argument('--run-id', required=True)
    t.add_argument('--arch', default=nets.ZELDA)
    t.add_argument('--arch-config', default='{}')
    t.add_argument('--init', default=None, help="checkpoint to start from (DAgger rounds)")
    t.add_argument('--epochs', type=int, default=8)
    t.add_argument('--batch', type=int, default=4096)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--value-coeff', type=float, default=0.1)
    t.add_argument('--fraction', type=float, default=1.0, help="the first share of each part's rollouts")
    t.add_argument('--sources', nargs='*', default=None)
    t.add_argument('--val-data', default=None, help="hold out only this data dir's rows (its name)")
    t.add_argument('--val-shards', type=int, default=4,
                   help="held-out rows from the first N rollouts of a source only (the same set at every --fraction)")
    t.add_argument('--seed', type=int, default=36)
    t.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.command == 'cache':
        build_cache(args.data, Path(args.cache))
    else:
        train(args)
    return 0


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    sys.exit(main())
