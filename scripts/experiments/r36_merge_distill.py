"""Stage 2's imitation (rounds 36-38; round 40's generation, docs/lab.md "Stage 2: one network from its teachers"): a
student trained offline, on the GPU, to the teachers' distributions, its hit head on the recorded hit labels.

    python scripts/experiments/r36_merge_distill.py cache --data DIR [DIR...] --cache CACHE
    python scripts/experiments/r36_merge_distill.py train --cache CACHE --name NAME --run-id RUN \\
        --arch hitnet --arch-config JSON --aux hit=hit:0.5                         # from scratch: 20 epochs, lr 1e-3
    python scripts/experiments/r36_merge_distill.py train --cache CACHE --name NAME --run-id RUN2 --init NAME/RUN \\
        --aux hit=hit:0.5                                                          # a second pass: 5 epochs, lr 3e-4

``cache``: every ``<data>/<key>/<source>/shard-*.npz`` (``r36_merge_record.py``) into one array a field
(``<cache>/<field>.npy``, compact dtypes, the teachers' joint log-probs as float16) with each row's key, source,
shard, env and whether it is held out.  Several data dirs (``--data A B``: the teacher-acting data and correction
rounds) concatenate. A field is kept when every shard has it (data recorded without ``--try-others`` has no tried
candidates, data before round 40 no hit labels at all: the cache then has none). ``cache.json`` lists every source
with the observation switches it was recorded with (``obs`` of its ``meta.json``; none before round 40: off).

``train``: KL(teacher ‖ student) over the 352 joint actions plus ``--value-coeff`` · ½(V − V_teacher)² plus the
auxiliary losses ``--aux OUTPUT=TARGET:COEFF`` (``lab.auxiliary.aux_loss``, the lab trainer's own: ``hit=hit:C`` is
binary cross-entropy of the hit head on the action taken and the look-ahead's tried candidates), Adam, warm-up and
cosine decay; one forward a minibatch. From scratch (``--arch``, ``--arch-config``: ``lab.recipe.create_network``) the
defaults are round 38's first pass (20 epochs, lr 1e-3, value weight 0.02); from ``--init`` (a checkpoint, a run dir
or a store run ``NAME/RUN``: a second pass, or a correction round's imitation) its second (5 epochs, lr 3e-4); each
is an option. A student that reads a block whose observation switch was off in a source it trains on is refused
(``lab.recipe.network_reads``: it would learn from zeros). Held out: the last env of every rollout (each env's
episodes are its own, so the held-out rows are whole episodes, a tenth with 10 envs; round 38's env 9). Reported per
key on the held-out rows (``distill.json`` in the run): KL, top-1 agreement of the joint action and of each head where
it has a choice (kind; direction and duration given the teacher's kind), the value's squared error and, per
auxiliary output, its loss and (``hit`` targets) precision = recall on the actions taken and on every judged
candidate (``auxiliary.matched_precision``).  The checkpoint is a normal run (``registry.save_checkpoint``), playable
by the bench and loadable by ``lab.train --load``. In a checkout that plays the store (no models folder:
``$TRIFORCE_MODELS_DIR``, ``<repo>/models``) it is a store run ``NAME/RUN``, as ``lab.train --team`` makes one
(``store.allocate_run``, origin ``distill``, its parent the ``--init`` run, its data the cache and the recordings it
came from with their teachers; an existing run id is refused), written every epoch and sealed after the last; a
team slot then names it (``current+@GROUP=NAME/RUN`` for each group, ``scripts/experiments/r52_one_team.py``).
Both are jobs of the machine's queue (``jobs.require_job``: ``triforce job run ... --timeout 3h --
.venv/bin/python scripts/experiments/r36_merge_distill.py train ...``)."""

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

from triforce import nets, paths, registry, store
from triforce.jobs import require_job
from triforce.lab.auxiliary import JUDGED, AuxLoss, aux_loss, joint_index, matched_precision, target_values, \
    tried_labels
from triforce.lab.recipe import SWITCH_KEYS, create_network, network_reads
from triforce.nets import roomheads  # noqa: F401  pylint: disable=unused-import  (registers 'roomheads')
from triforce.ppo import joint_kl
from triforce.train import resolve_warm_start

STEPS_PER_ENV = 512
SOURCES = ('resets', 'entries', 'dagger', 'rooms')
DTYPES = {'obs/room_id': np.int16, 'obs/entity_types': np.uint8, 'obs/grid': np.uint8,
          'teacher_joint': np.float16, 'action': np.int8, 'tried_actions': np.int8, 'tried_lost': np.int8,
          'tried_blocked': np.int8, 'hits': np.int16, 'beam_lined': np.uint8}
# The fields an auxiliary loss reads (``lab.auxiliary``), as the learner's batch holds them.
LABELS = ('action', 'health_lost', 'hits', 'blocked', 'tried_actions', 'tried_lost', 'tried_blocked')
SKIPPED = ('family',)       # a recording's family index: the cache's ``key`` holds it
SHAPE = (11, 8, 4)
# Round 38's recipe: the first pass from scratch, then each later pass from the one before.
FIRST = {'epochs': 20, 'lr': 1e-3}
LATER = {'epochs': 5, 'lr': 3e-4}


def build_cache(data_dirs: list[str], cache: Path):
    # pylint: disable=too-many-locals
    shards, sources = [], []
    for data in data_dirs:
        for meta in sorted(Path(data).glob('*/*/meta.json')):
            info = json.loads(meta.read_text())
            sources.append({'data': Path(data).name, 'key': info['key'], 'source': meta.parent.name,
                            'obs': info.get('obs') or {}, 'teacher': info.get('teacher'),
                            'student': info.get('student'), 'try_others': info.get('try_others')})
            for path in sorted(meta.parent.glob('shard-*.npz')):
                shards.append((info['key'], meta.parent.name, Path(data).name, path))
    keys = sorted({s[0] for s in shards})
    tags = sorted({s[2] for s in shards})
    first = np.load(shards[0][3])
    every = set(first.files)
    for shard in shards[1:]:
        every &= set(np.load(shard[3]).files)
    fields = [f for f in first.files if f in every and f not in SKIPPED]
    dropped = sorted(set(first.files) - every)
    if dropped:
        print(f"cache: {dropped} left out (not in every shard)", flush=True)
    rows = [np.load(s[3], mmap_mode=None)['teacher_value'].shape[0] for s in shards]
    total = sum(rows)
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for field in fields:
        dtype = DTYPES.get(field, first[field].dtype)
        out[field] = np.lib.format.open_memmap(cache / f'{field.replace("/", "_")}.npy', 'w+', dtype,
                                               (total, *first[field].shape[1:]))
    meta_arrays = {name: np.lib.format.open_memmap(cache / f'{name}.npy', 'w+', np.uint8, (total,))
                   for name in ('key', 'source', 'shard', 'env', 'data', 'held')}
    at = 0
    for (key, source, tag, path), n in zip(shards, rows):
        arrays = np.load(path)
        for field in fields:
            if arrays[field].shape[1:] != out[field].shape[1:]:
                raise SystemExit(f"{path}: {field} is {arrays[field].shape[1:]}, the first shard's "
                                 f"{out[field].shape[1:]} (one --try-others K for every source)")
            out[field][at:at + n] = arrays[field].astype(out[field].dtype)
        env = np.arange(n) // STEPS_PER_ENV
        meta_arrays['key'][at:at + n] = keys.index(key)
        meta_arrays['source'][at:at + n] = SOURCES.index(source)
        meta_arrays['shard'][at:at + n] = int(path.stem.split('-')[1])
        meta_arrays['env'][at:at + n] = env
        meta_arrays['held'][at:at + n] = env == env[-1]
        meta_arrays['data'][at:at + n] = tags.index(tag)
        at += n
    for array in [*out.values(), *meta_arrays.values()]:
        array.flush()
    (cache / 'cache.json').write_text(json.dumps({'keys': keys, 'sources': list(SOURCES), 'data': tags,
                                                  'fields': fields, 'rows': total, 'from': data_dirs,
                                                  'recorded': sources}, indent=1))
    print(f"cache: {total:,} rows, {len(keys)} keys, {len(shards)} shards -> {cache}", flush=True)


class Data:
    """The cache in memory (``mmap``: read on demand)."""

    def __init__(self, cache: Path, mmap: bool = False):
        self.meta = json.loads((cache / 'cache.json').read_text())
        self.fields = self.meta['fields']
        if not (cache / 'held.npy').is_file():
            raise SystemExit(f"{cache}: a cache from before round 40 (no held.npy): build it again (cache --data ...)")
        mode = 'r' if mmap else None
        self.arrays = {f: np.load(cache / f'{f.replace("/", "_")}.npy', mmap_mode=mode) for f in self.fields}
        for name in ('key', 'source', 'shard', 'env', 'data'):
            setattr(self, name, np.load(cache / f'{name}.npy'))
        self.held = np.load(cache / 'held.npy').astype(bool)
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

    def labels(self, rows: np.ndarray, device) -> dict:
        """The auxiliary targets' fields of ``rows`` (sorted, as ``batch``) as the learner's batch holds them
        (``actions`` the taken one; ``lab.auxiliary``)."""
        rows = np.sort(rows)
        out = {}
        for field in LABELS:
            if field in self.arrays:
                value = torch.from_numpy(np.ascontiguousarray(self.arrays[field][rows])).to(device)
                out['actions' if field == 'action' else field] = (
                    value if field == 'health_lost' else value.bool() if field == 'blocked' else value.long())
        return out


def forward(network, obs: dict, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """(joint log-probs, value, auxiliary outputs) of a batch: a lab network's heads in one pass, today's
    ``ZeldaPolicy`` (no auxiliary output) by its own two."""
    if isinstance(network, nets.LabNet):
        heads = network._heads(obs)  # pylint: disable=protected-access
        return nets.factored_joint(heads, mask.bool()), heads.value, heads.aux
    return network.joint_log_probs(obs, mask), network.get_value(obs), {}


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
def evaluate(network, data: Data, rows: np.ndarray, device, aux_losses=(), batch: int = 8192) -> dict:
    # pylint: disable=too-many-locals
    network.eval()
    sums: dict = {}
    aux_sums: dict = {}
    judged: dict = {s.output: {'taken': ([], []), 'judged': ([], [])} for s in aux_losses if s.target in JUDGED}
    for begin in range(0, len(rows), batch):
        part = np.sort(rows[begin:begin + batch])
        obs, mask, joint, value = data.batch(part, device)
        student, student_value, aux = forward(network, obs, mask)
        legal = torch.isfinite(joint)
        kl = (joint.exp() * torch.where(legal, joint - student, 0.0)).sum(-1)
        agree = head_agreement(joint, student)
        verr = (student_value - value).pow(2)
        k = torch.from_numpy(data.key[part].astype(np.int64)).to(device)
        for name, values in (('kl', kl), ('value_se', verr), *agree.items()):
            valid = values >= 0
            n = torch.zeros(len(data.keys), device=device).index_add_(0, k[valid], torch.ones_like(values[valid]))
            s = torch.zeros(len(data.keys), device=device).index_add_(0, k[valid], values[valid])
            acc = sums.setdefault(name, [torch.zeros_like(n), torch.zeros_like(s)])
            acc[0] += n
            acc[1] += s
        if aux_losses:
            labels = data.labels(part, device)
            every = slice(None)
            _, stats = aux_loss(list(aux_losses), aux, labels, every)
            for spec in aux_losses:
                acc = aux_sums.setdefault(spec.output, [0.0, 0])
                acc[0] += stats[f'losses/aux_{spec.output}'] * len(part)
                acc[1] += len(part)
                if spec.output in judged:
                    logits = aux[spec.output]
                    taken = logits.gather(1, joint_index(labels['actions']).unsqueeze(-1)).squeeze(-1)
                    lost = target_values(spec.target, labels, every)
                    for name in ('taken', 'judged'):
                        judged[spec.output][name][0].append(taken.float().cpu())
                        judged[spec.output][name][1].append(lost.cpu())
                    if 'tried_lost' in labels:
                        verdicts = tried_labels(spec.target, labels, every)
                        tried = verdicts >= 0
                        index = joint_index(labels['tried_actions'].clamp(min=0))
                        judged[spec.output]['judged'][0].append(logits.gather(1, index)[tried].float().cpu())
                        judged[spec.output]['judged'][1].append(verdicts[tried].float().cpu())
    result = {}
    for i, key in enumerate(data.keys):
        result[key] = {name: (float(s[i] / n[i]) if n[i] > 0 else None) for name, (n, s) in sums.items()}
        result[key]['rows'] = int(sums['kl'][0][i])
    total = {name: float(s.sum() / n.sum()) for name, (n, s) in sums.items()}
    total['rows'] = int(sums['kl'][0].sum())
    for output, (loss, n) in aux_sums.items():
        total[f'aux_{output}'] = loss / max(1, n)
    for output, kinds in judged.items():
        for name, (scores, labels) in kinds.items():
            total[f'aux_{output}_pr_{name}'] = matched_precision(torch.cat(scores), torch.cat(labels))
            total[f'aux_{output}_lost_{name}'] = float(torch.cat(labels).sum())
    network.train()
    return {'total': total, 'keys': result}


def prefetch(data: Data, order: np.ndarray, batch: int, device, labels: bool, depth: int = 4):
    q: queue.Queue = queue.Queue(depth)

    def work():
        for begin in range(0, len(order) - batch + 1, batch):
            part = np.sort(order[begin:begin + batch])
            q.put((*data.batch(part, device), data.labels(part, device) if labels else None))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def refused_sources(network, data: Data, sources) -> list[str]:
    """The recorded sources (of ``sources``, every one with None) whose observation switch was off for a block the
    student reads (``lab.recipe.network_reads``), as ``<data>/<key>/<source> obs.<switch>``."""
    reads = network_reads(network)
    problems = []
    for entry in data.meta.get('recorded', []):
        if sources and entry['source'] not in sources:
            continue
        for switch, keys in SWITCH_KEYS.items():
            if any(k in reads for k in keys) and not entry['obs'].get(switch):
                problems.append(f"{entry['data']}/{entry['key']}/{entry['source']} obs.{switch}")
    return problems


def initial_network(args):
    """(network, init identity): ``--init`` (a checkpoint, run dir or store run) or a fresh ``--arch``."""
    if args.init:
        path = resolve_warm_start(args.init, None)
        network, checkpoint = registry.load_model(path)
        return network, {'source': args.init, 'file': str(path), 'sha1': registry.checkpoint_sha1(path),
                         'run': checkpoint.get('run_id'), 'family': checkpoint.get('specialist')}
    return create_network(args.arch, json.loads(args.arch_config)), None


def plays_store() -> bool:
    """Whether the run goes to the store: this checkout has no models folder (``$TRIFORCE_MODELS_DIR``,
    ``<repo>/models``, ``registry.models_root``'s first places) and the store is made."""
    return (not os.environ.get(registry.MODELS_DIR_ENV) and not paths.legacy_models_dir().exists()
            and paths.store_exists())


def file_in_store(args, data, init) -> 'store.Run':
    """The open store run ``NAME/RUN`` (``store.allocate_run``: an existing one is refused), origin ``distill``, its
    parent the ``--init`` network's run when that is a store run, its data the cache, the recordings it holds and
    their teachers and students; sealed by ``train`` after the last epoch (a run stopped before stays open)."""
    recorded = data.meta.get('recorded', [])
    provenance = {'kind': 'distill-cache', 'cache': str(Path(args.cache).resolve()), 'rows': data.meta['rows'],
                  'from': [str(Path(d).resolve()) for d in data.meta.get('from', [])],
                  'teachers': sorted({str(r['teacher']) for r in recorded if r.get('teacher')}),
                  'students': sorted({str(r['student']) for r in recorded if r.get('student')})}
    parent = store.run_of_path(init['file']) if init else None
    run = store.allocate_run(args.name, args.run_id, parent=parent, origin='distill', data=[provenance])
    print(f"[distill] store run {run.identity} (open until the last epoch)"
          + (f", from {parent.identity}" if parent else ''), flush=True)
    return run


def train(args):
    # pylint: disable=too-many-locals, too-many-statements
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    data = Data(Path(args.cache))
    recipe = LATER if args.init else FIRST
    args.epochs = recipe['epochs'] if args.epochs is None else args.epochs
    args.lr = recipe['lr'] if args.lr is None else args.lr
    aux_losses = [AuxLoss.parse(text) for text in args.aux]
    if aux_losses and 'action' not in data.fields:
        raise SystemExit(f"--aux: {args.cache} holds no hit labels (recorded before round 40: record again)")
    held = data.held.copy()
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
    network, init = initial_network(args)
    refused = refused_sources(network, data, args.sources)
    if refused:
        raise SystemExit("the student reads blocks these sources recorded with their switch off (it would learn from "
                         "zeros: record them with the switch on, or leave the token group out):\n  "
                         + '\n  '.join(refused))
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
    stored = file_in_store(args, data, init) if plays_store() else None
    root = paths.runs_dir() if stored is not None else registry.models_root(None)
    print(f"[distill] {args.name}/{args.run_id}: {nets.arch_of(network.config.to_dict())}, {params:,} parameters, "
          f"{len(train_rows):,} train rows, {len(val_rows):,} held out, {total_steps} steps ({args.epochs} epochs, "
          f"lr {args.lr}, value {args.value_coeff}, aux {args.aux or 'none'}, from {args.init or 'scratch'})",
          flush=True)
    history = []
    began = time.time()
    step = 0
    path = None
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(train_rows)
        kl_sum, v_sum, n = 0.0, 0.0, 0
        aux_sum: dict = {}
        for obs, mask, joint, value, labels in prefetch(data, order, args.batch, device, bool(aux_losses)):
            student, student_value, aux = forward(network, obs, mask)
            kl = joint_kl(joint, student)
            vloss = 0.5 * (student_value - value).pow(2).mean()
            loss = kl + args.value_coeff * vloss
            if aux_losses:
                term, stats = aux_loss(aux_losses, aux, labels, slice(None))
                loss = loss + term
                for spec in aux_losses:
                    aux_sum[spec.output] = aux_sum.get(spec.output, 0.0) + stats[f'losses/aux_{spec.output}']
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            kl_sum += float(kl.detach())
            v_sum += float(vloss.detach())
            n += 1
        val = evaluate(network, data, val_rows, device, aux_losses)
        row = {'epoch': epoch, 'train_kl': kl_sum / max(1, n), 'train_value': v_sum / max(1, n),
               **{f'train_aux_{k}': v / max(1, n) for k, v in aux_sum.items()},
               'val': val['total'], 'seconds': round(time.time() - began, 1)}
        history.append(row)
        hit = ''.join(f" | aux {k} train {row[f'train_aux_{k}']:.4f} held out {val['total'][f'aux_{k}']:.4f}"
                      + (f" P=R taken {val['total'][f'aux_{k}_pr_taken']:.3f} judged "
                         f"{val['total'][f'aux_{k}_pr_judged']:.3f}" if f'aux_{k}_pr_taken' in val['total'] else '')
                      for k in aux_sum)
        print(f"[distill] epoch {epoch}: train kl {row['train_kl']:.4f} | held out kl {val['total']['kl']:.4f} "
              f"joint {val['total']['joint']:.3f} kind {val['total']['kind']:.3f} dir {val['total']['direction']:.3f} "
              f"dur {val['total']['duration']:.3f} value se {val['total']['value_se']:.3f}{hit} "
              f"({row['seconds']:.0f}s)", flush=True)
        path = registry.save_checkpoint(args.name, args.run_id, network, iterations=epoch, steps=step,
                                        root=root, extra={'distill': {**vars(args), 'parameters': params,
                                                                      'init': init}})
    (path.parent / 'distill.json').write_text(json.dumps(
        {'args': vars(args), 'parameters': params, 'init': init, 'cache': data.meta,
         'train_rows': int(len(train_rows)), 'held_out_rows': int(len(val_rows)), 'history': history,
         'held_out': val}, indent=1, default=str))
    if stored is not None:
        store.seal_run(stored)
        print(f"[distill] store run {stored.identity} sealed", flush=True)
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
    t.add_argument('--arch', default=nets.ZELDA, choices=nets.names())
    t.add_argument('--arch-config', default='{}')
    t.add_argument('--init', default=None,
                   help="the network to start from (a checkpoint, run dir or store run NAME/RUN): a second pass or a "
                        "correction round; --arch is then its own")
    t.add_argument('--epochs', type=int, default=None,
                   help=f"default {FIRST['epochs']} from scratch, {LATER['epochs']} from --init (round 38)")
    t.add_argument('--batch', type=int, default=4096)
    t.add_argument('--lr', type=float, default=None,
                   help=f"default {FIRST['lr']} from scratch, {LATER['lr']} from --init (round 38)")
    t.add_argument('--value-coeff', type=float, default=0.02, help="round 38's 0.02")
    t.add_argument('--aux', nargs='+', action='extend', default=[], metavar='OUTPUT=TARGET:COEFF',
                   help="a loss on the network's auxiliary output (lab.auxiliary; the generation: hit=hit:C)")
    t.add_argument('--fraction', type=float, default=1.0, help="the first share of each part's rollouts")
    t.add_argument('--sources', nargs='*', default=None)
    t.add_argument('--val-data', default=None, help="hold out only this data dir's rows (its name)")
    t.add_argument('--val-shards', type=int, default=4,
                   help="held-out rows from the first N rollouts of a source only (the same set at every --fraction)")
    t.add_argument('--seed', type=int, default=36)
    t.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    require_job()
    if args.command == 'cache':
        build_cache(args.data, Path(args.cache))
    else:
        train(args)
    return 0


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    sys.exit(main())
