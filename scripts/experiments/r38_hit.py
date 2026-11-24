"""Round 38: what a good hit head is worth at play, and how it must be connected (docs/experiments/round38.md).

    python scripts/experiments/r38_hit.py combine --policy CKPT --hit CKPT --weights 0 1 2 --name r38shift --run base
    python scripts/experiments/r38_hit.py manifests                       # target/r38/manifest-{goriya,0x1b}.json
    python scripts/experiments/r38_hit.py calibrate --hit CKPT --policy CKPT --data SRC... --weights 0 1 2 4
    python scripts/experiments/r38_hit.py bench --manifest M... --arm NAME=CKPT[@ACT]... --out DIR
    python scripts/experiments/r38_hit.py pr --hit CKPT... --data SRC...
    python scripts/experiments/r38_hit.py retrain --init CKPT --data SRC... --save NAME/RUN [--epochs 10]
    python scripts/experiments/r38_hit.py table DIR [--base NAME]

A data source ``SRC`` is a dataset of ``triforce.lab.record`` (``--try-others``: the look-ahead's verdicts on the
recording actor's own states; every 7th shard held out) or ``r36:goriya-all`` (experiment 2's collection, the
planner's and the teacher's states on the first order, schema 7's fields from the engine; every 7th episode held
out). ``combine`` writes ``hitshift`` networks (``triforce/nets/hitshift.py``): today's policy and the supervised
hit network side by side, the policy's joint action shifted by ``-weight * softplus(hit logit)``; nothing is
trained. ``bench`` plays every arm on every manifest in one queued job (the yardstick's ``run_bench``; one JSON an
arm in ``DIR``, ``bench compare`` reads them); ``ACT`` an act mode (the planner beside:
``temperature:0.5+relax:16+plan:16:1`` on today's network).  ``calibrate``, ``bench``, ``pr`` and ``retrain`` play
or train networks: jobs of the machine's queue (``jobs.require_job``: ``triforce job run ... --timeout 1h --
.venv/bin/python scripts/experiments/r38_hit.py bench ...``).
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'experiments'))

from triforce import nets, registry  # noqa: E402
from triforce.actions import NUM_DURATIONS  # noqa: E402
from triforce.events import progress_job, write_json_atomic  # noqa: E402
from triforce.jobs import require_job, work  # noqa: E402
from triforce.lab import bench as yardstick  # noqa: E402
from triforce.lab.auxiliary import matched_precision  # noqa: E402
from triforce.model import MASK_KEY  # noqa: E402
from triforce.nets.hitshift import HitShift, shift_heads  # noqa: E402

R38 = ROOT / 'target' / 'r38'
ACT = 'temperature:0.5+relax:16'
TEMPERATURE = 0.5
SHARD_HOLDOUT = 7
YAML_0X1B = 'scripts/experiments/r38_goriya_0x1b.yaml'


# ---------------------------------------------------------------------------------------------- data

class Rows:
    """Decisions with their observation and hit labels: ``obs`` (key -> array, the action mask included),
    candidates ``c_row`` / ``c_joint`` / ``c_lost`` / ``c_taken`` (the action taken, its real outcome; or a
    candidate the look-ahead judged), ``held`` per decision row, ``mode`` per row (0 the planner's episodes, 1 the
    actor alone; a recording is all 1)."""

    def __init__(self, obs, c_row, c_joint, c_lost, c_taken, held, mode, name):
        self.obs, self.c_row, self.c_joint, self.c_lost, self.c_taken = obs, c_row, c_joint, c_lost, c_taken
        self.held, self.mode, self.name = held, mode, name

    @property
    def rows(self) -> int:
        return len(self.held)


def load_rows(source: str) -> Rows:
    """``source``: ``r36:FAMILY``, a recording ``DIR``, ``DIR+blocked`` (the label ``hit_or_blocked``: lost
    health, or a Bubble's touch landed, the sword blocked: main's round 38 targets), ``DIR+beam`` (round 39's beam
    head: ``beam_rows``), either with ``#KEY`` for one family's rows of a recording of several (``KEY`` as in its
    ``dataset.json``)."""
    if source.startswith('r36:'):
        return _load_r36(source[4:])
    source, _, family = source.partition('#')
    directory, _, label = source.partition('+')
    shards = sorted(Path(directory).glob('shard-*.npz'))
    parts = [dict(np.load(p)) for p in shards]
    held = np.concatenate([np.full(len(p['action']), i % SHARD_HOLDOUT == SHARD_HOLDOUT - 1)
                           for i, p in enumerate(parts)])
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if family:
        families = json.loads((Path(directory) / 'dataset.json').read_text())['families']
        keep = d['family'] == families.index(family)
        d = {k: v[keep] for k, v in d.items()}
        held = held[keep]
    obs = {k[len('obs/'):]: v for k, v in d.items() if k.startswith('obs/')}
    obs[MASK_KEY] = d['action_mask']
    if label == 'beam':
        return beam_rows(obs, d['beam_lined'], held, source + (f'#{family}' if family else ''))
    n = len(d['action'])
    a = d['action'].astype(np.int64)
    taken = (a[:, 0] * 8 + a[:, 1]) * NUM_DURATIONS + a[:, 2]
    taken_lost = d['health_lost'] > 0
    if label == 'blocked':
        taken_lost = taken_lost | d['blocked'].astype(bool)
    rows, joints, lost, own = [np.arange(n)], [taken], [taken_lost], [np.ones(n, bool)]
    if 'tried_lost' in d:
        tried = d['tried_lost'] >= 0
        r, k = np.nonzero(tried)
        t = d['tried_actions'][r, k].astype(np.int64)
        rows.append(r)
        joints.append((t[:, 0] * 8 + t[:, 1]) * NUM_DURATIONS + t[:, 2])
        tried_lost = d['tried_lost'][r, k] > 0
        if label == 'blocked':
            tried_lost = tried_lost | (d['tried_blocked'][r, k] > 0)
        lost.append(tried_lost)
        own.append(np.zeros(len(r), bool))
    return Rows(obs, np.concatenate(rows), np.concatenate(joints), np.concatenate(lost), np.concatenate(own), held,
                np.ones(n, np.int8), source + (f'#{family}' if family else ''))


def beam_rows(obs: dict, lined: np.ndarray, held: np.ndarray, name: str) -> Rows:
    """Round 39's beam labels as candidates: each decision whose mask allows BEAMS in a cardinal direction gets one
    candidate a legal direction, ``(BEAMS, d, duration 0)``, labelled 1 when the projection had a shot fired so meet
    a monster out of melee range (``beam_lined``'s ``Direction`` bit); the same columns as the hit labels', so
    ``retrain`` and ``pr`` read them unchanged."""
    from triforce.actions import CARDINALS, DIRECTION_INDEX, NUM_DIRECTIONS, NUM_KINDS  # noqa: E402
    from triforce.zelda_enums import ActionKind  # noqa: E402
    beams = int(ActionKind.BEAMS)
    mask = obs[MASK_KEY].astype(bool)
    rows, joints, labels = [], [], []
    for d in CARDINALS:
        i = DIRECTION_INDEX[d]
        legal = np.flatnonzero(mask[:, beams] & mask[:, NUM_KINDS + beams * NUM_DIRECTIONS + i])
        rows.append(legal)
        joints.append(np.full(len(legal), (beams * NUM_DIRECTIONS + i) * NUM_DURATIONS, np.int64))
        labels.append((lined[legal].astype(np.int64) & d.value) != 0)
    c_row = np.concatenate(rows)
    return Rows(obs, c_row, np.concatenate(joints), np.concatenate(labels), np.ones(len(c_row), bool), held,
                np.ones(len(held), np.int8), name)


def _load_r36(family: str) -> Rows:
    import r36_hit_bench as hb  # pylint: disable=import-outside-toplevel
    data, _ = hb.load_family(family)
    obs = dict(data.obs)
    obs.update(hb.engine_columns(data))
    return Rows(obs, data.c_row, data.c_joint, data.c_lost.astype(bool), data.c_own, data.held_rows,
                data.mode_of_row, f'r36:{family}')


def to_device(obs: dict, rows, device) -> dict:
    return {k: torch.as_tensor(v[rows]).to(device) for k, v in obs.items()}


def batched(n: int, size: int = 8192):
    for lo in range(0, n, size):
        yield np.arange(lo, min(n, lo + size))


def hit_logits(network, obs: dict, rows: np.ndarray, device) -> torch.Tensor:
    """``[len(rows), 352]`` hit logits of a ``hitnet`` (or a ``hitshift``'s hit body)."""
    hit = network.hit if isinstance(network, nets.ARCHS['hitshift']) else network
    out = []
    with torch.no_grad():
        for part in batched(len(rows)):
            batch = to_device(obs, rows[part], device)
            out.append(hit.hit_logits(hit.trunk(batch)).float())
    return torch.cat(out)


def matched(scores: np.ndarray, labels: np.ndarray) -> float:
    return matched_precision(torch.as_tensor(scores), torch.as_tensor(labels))


def pr_report(rows: Rows, logits: torch.Tensor, row_index: np.ndarray) -> dict:
    """P = R on the held-out decisions' candidates: the actions taken (``taken``; split by mode for the r36 data)
    and every judged candidate (``judged``)."""
    where = np.full(rows.rows, -1)
    where[row_index] = np.arange(len(row_index))
    keep = where[rows.c_row] >= 0
    c_rows, joints, lost, taken = rows.c_row[keep], rows.c_joint[keep], rows.c_lost[keep], rows.c_taken[keep]
    scores = logits[torch.as_tensor(where[c_rows]), torch.as_tensor(joints)].cpu().numpy()
    out = {'judged': matched(scores, lost), 'lost_judged': int(lost.sum()), 'candidates': int(len(lost))}
    for mode in np.unique(rows.mode[c_rows]):
        m = taken & (rows.mode[c_rows] == mode)
        out[f'taken{mode}'] = matched(scores[m], lost[m])
        out[f'lost_taken{mode}'] = int(lost[m].sum())
        out[f'decisions{mode}'] = int(m.sum())
    return out


# ---------------------------------------------------------------------------------------------- networks

def load(path):
    return registry.load_model(path, check_schema=False)[0]


def combine(policy_path: str, hit_path: str, weights: list[float], name: str, run: str, form: str,
            freeze_hit: bool = False, models_dir=None) -> list[Path]:
    """One ``hitshift`` run dir a weight, ``<models>/<name>/<run>-w<weight>``: ``ckpt.pt`` (its ``specialist`` is
    ``name``: a family's name makes it that family's run, promotable), ``version`` and ``act`` (the default act
    mode, bound to the weights), so every tool loads it as it loads any run."""
    policy, hitnet = load(policy_path), load(hit_path)
    out = []
    for weight in weights:
        network = HitShift.of(policy, hitnet, weight, form, freeze_hit)
        path = registry.save_checkpoint(name, f'{run}-w{weight:g}', network, root=models_dir,
                                        extra={'policy': str(policy_path), 'hit': str(hit_path)})
        registry.set_act_mode(path, registry.DEFAULT_ACT)
        out.append(path.parent)
        print(f'[combine] {path}', flush=True)
    return out


# ---------------------------------------------------------------------------------------------- calibration

def calibrate(rows: Rows, policy, hitnet, weights: list[float], device, form: str = 'softplus') -> list[dict]:
    """Per weight, on the source's actor-alone decisions: the share of decisions whose temperature-0.5 draw (the
    same Gumbel noise) differs from the unshifted policy's (the refusals), the expected hit probability of the
    draw (by the head) with the factorised shift and with the exact joint one, and the KL between them."""
    rows_alone = np.flatnonzero(rows.mode == 1)
    rng = torch.Generator(device='cpu').manual_seed(38)
    stats = {w: {'changed': 0.0, 'p_hit': 0.0, 'p_hit_exact': 0.0, 'kl': 0.0} for w in weights}
    base_hit = 0.0
    n = 0
    with torch.no_grad():
        for part in batched(len(rows_alone), 4096):
            idx = rows_alone[part]
            batch = to_device(rows.obs, idx, device)
            mask = batch[MASK_KEY].bool()
            heads = policy.heads(batch)[0]
            hit = hitnet.hit_logits(hitnet.trunk(batch)).float()
            p_hit = torch.sigmoid(hit)
            base = nets.factored_joint(heads, mask)
            uniform = torch.rand(base.shape, generator=rng).clamp(1e-20, 1)
            gumbel = -torch.log((-torch.log(uniform)).clamp_min(1e-20)).to(device)
            draw0 = (base / TEMPERATURE + gumbel).argmax(-1)
            q0 = torch.softmax(base / TEMPERATURE, -1)
            base_hit += float((q0 * p_hit).sum())
            for w in weights:
                shift = -w * (F.softplus(hit) if form == 'softplus' else hit)
                fact = nets.factored_joint(shift_heads(heads, shift, mask), mask)
                exact = torch.log_softmax(base + shift, -1)
                qf = torch.softmax(fact / TEMPERATURE, -1)
                qe = torch.softmax(exact / TEMPERATURE, -1)
                draw = (fact / TEMPERATURE + gumbel).argmax(-1)
                s = stats[w]
                s['changed'] += float((draw != draw0).sum())
                s['p_hit'] += float((qf * p_hit).sum())
                s['p_hit_exact'] += float((qe * p_hit).sum())
                legal = qe > 0
                s['kl'] += float(torch.where(legal, qe * (torch.log(qe.clamp_min(1e-30)) - torch.log(qf.clamp_min(1e-30))),
                                             0.0).sum())
            n += len(idx)
    taken_lost = rows.c_lost[rows.c_taken & (rows.mode[rows.c_row] == 1)]
    out = []
    for w in weights:
        s = stats[w]
        out.append({'weight': w, 'decisions': n, 'refused': s['changed'] / n, 'p_hit': s['p_hit'] / n,
                    'p_hit_exact': s['p_hit_exact'] / n, 'kl_exact_vs_factored': s['kl'] / n,
                    'p_hit_unshifted': base_hit / n, 'taken_lost_rate': float(taken_lost.mean())})
        print(json.dumps(out[-1]), flush=True)
    return out


# ---------------------------------------------------------------------------------------------- bench

def write_manifests(out: Path, goriya_states: int, goriya_episodes: int, episodes_0x1b: int) -> None:
    base = yardstick.load_manifest()
    states, source = yardstick.choose_states('goriya', goriya_states, ROOT / 'triforce' / 'states')
    goriya = {**base, 'episodes': goriya_episodes, 'states_per_family': goriya_states,
              'entries': [{'key': 'goriya@d3first', 'family': 'goriya', 'source': source, 'states': states}]}
    room = {**base, 'episodes': episodes_0x1b, 'states_per_family': 3,
            'entries': [{'key': '7:0x1b', 'family': YAML_0X1B, 'source': 'seed',
                         'states': [f'd3first_7_1bw_{i}' for i in range(3)]}]}
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / 'manifest-goriya.json', goriya)
    write_json_atomic(out / 'manifest-0x1b.json', room)
    print(f'[manifests] goriya {len(states)} states x {goriya_episodes}; 7:0x1b 3 states x {episodes_0x1b}')


def bench(manifests: list[str], arms: list[str], out: Path, parallel: int, logs_dir=None) -> None:
    loaded = [yardstick.load_manifest(m) for m in manifests]
    parsed = []
    for arm in arms:
        name, spec = arm.split('=', 1)
        checkpoint, _, act = spec.partition('@')
        parsed.append((name, checkpoint, act or None))
    total = sum(len(e['states']) * m['episodes'] for m in loaded for e in m['entries']) * len(parsed)
    out.mkdir(parents=True, exist_ok=True)
    with progress_job('lab-bench', 'eval', total, root=logs_dir, work=work(eval_episodes=total)) as progress:
        done = 0
        for name, checkpoint, act in parsed:
            path = out / f'bench-{name}.json'
            if path.exists():
                print(f'[bench] {name}: kept {path}', flush=True)
                continue
            merged = None
            for manifest in loaded:
                networks = {e['key']: checkpoint for e in manifest['entries']}
                result = yardstick.run_bench(networks, manifest, parallel, act=act)
                merged = result if merged is None else {**merged, 'keys': {**merged['keys'], **result['keys']}}
                done += sum(len(e['states']) * manifest['episodes'] for e in manifest['entries'])
                progress.update(done, result['total']['success'], key=name)
            every = [e for row in merged['keys'].values() for e in yardstick.records(row)]
            merged['total'] = yardstick.summarize(every)
            merged['arm'] = {'name': name, 'checkpoint': checkpoint, 'act': act}
            write_json_atomic(path, yardstick.compact(merged))
            print(f'[bench] {name}: ' + json.dumps({k: {m: round(v[m], 3) for m in ('success', 'hearts_lost',
                                                                                    'hits_per_visit', 'decisions')}
                                                    for k, v in merged['keys'].items()}), flush=True)


def table(directory: Path, base: str) -> str:
    results = {p.stem[len('bench-'):]: json.loads(p.read_text()) for p in sorted(directory.glob('bench-*.json'))}
    keys = sorted({k for r in results.values() for k in r['keys']})
    lines = []
    for key in keys:
        lines += [f'\n**{key}**\n', '| arm | episodes | hits a visit | Bubble touches a visit | hearts lost | success | '
                  'decisions | bombs | '
                  f'hits − {base} (per episode) | hearts − {base} | success − {base} | decisions − {base} |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        ref = results.get(base, {}).get('keys', {}).get(key)
        for name, result in results.items():
            row = result['keys'].get(key)
            if row is None:
                continue
            cells = [f"{row['hits_per_visit']:.3f}", f"{row.get('bubble_touches_per_visit', float('nan')):.3f}",
                     f"{row['hearts_lost']:.3f}", f"{row['success']:.3f}",
                     f"{row['decisions']:.1f}", f"{row['bombs']:.2f}"]
            if ref is not None and name != base:
                p = yardstick.paired(yardstick.records(ref), yardstick.records(row))
                cells += [f"{p[m][0]:+.3f} ± {p[m][1]:.3f}" for m in ('hits', 'hearts_lost', 'success', 'decisions')]
            else:
                cells += [''] * 4
            lines.append(f"| {name} | {row['episodes']} | " + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------------- retraining

def retrain(init: str, sources: list[str], save: str, epochs: int, lr: float, batch: int, device, seed: int,
             weights: list[float], fraction: float = 1.0, by_row: bool = False) -> None:
    """The hit network's trunk and hit tower trained on every source's training candidates (BCE, AdamW one-cycle,
    as experiment 2's bench), from ``init`` (a ``hitnet`` or ``hitshift`` checkpoint); P = R on each source's
    held-out decisions after every epoch. ``fraction``: the share of each source's training decisions kept (drawn by
    decision, the same draw for a given seed: the data curve). Round 40 retired ``fresh:POLICY`` (a new ``hitnet``
    with the default tokens alone, so never one reading ``threat`` or ``chase``): a generation's student is a
    ``hitnet`` whose hit head trains in the same pass as its imitation (``r36_merge_distill.py train --arch hitnet
    --arch-config ... --aux hit=hit:C``, docs/lab.md "Stage 2")."""
    if init.startswith('fresh:'):
        raise SystemExit("retrain --init fresh:... is retired (round 40): train a hitnet's head with its imitation, "
                         "r36_merge_distill.py train --arch hitnet --aux hit=hit:C")
    torch.manual_seed(seed)
    data = [load_rows(s) for s in sources]
    network = load(init).to(device)
    hitnet = network.hit if isinstance(network, nets.ARCHS['hitshift']) else network
    keys = None
    for rows in data:
        keys = set(rows.obs) if keys is None else keys & set(rows.obs)
    train = []
    for i, rows in enumerate(data):
        m = ~rows.held[rows.c_row]
        if fraction < 1.0:
            train_rows = np.flatnonzero(~rows.held)
            kept = np.zeros(rows.rows, bool)
            kept[np.random.default_rng(seed).choice(train_rows, int(round(fraction * len(train_rows))),
                                                    replace=False)] = True
            m &= kept[rows.c_row]
        reps = max(1, int(round(weights[i]))) if weights else 1
        units = np.unique(rows.c_row[m]) if by_row else np.flatnonzero(m)
        train += [np.stack([np.full(len(units), i), units], 1)] * reps
    train = np.concatenate(train)
    # by_row: a batch is ``batch`` decisions, the trunk run once a decision for all its candidates
    csr = []
    for rows in data:
        order = np.argsort(rows.c_row, kind='stable')
        csr.append((order, np.searchsorted(rows.c_row[order], np.arange(rows.rows + 1))))
    print(f"[retrain] {len(train)} training {'decisions' if by_row else 'candidates'} from {sources}", flush=True)
    params = [p for n, p in hitnet.named_parameters() if not n.startswith(
        ('policy_tower.', 'value_tower.', 'kind_', 'direction_head.', 'duration_head.', 'value_head.'))]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(train) / batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        hitnet.train()
        order = train[rng.permutation(len(train))]
        total, count = 0.0, 0
        for lo in range(0, len(order), batch):
            part = order[lo:lo + batch]
            losses, n = [], 0
            for i, rows in enumerate(data):
                sel = part[part[:, 0] == i, 1]
                if not len(sel):
                    continue
                if by_row:
                    c_order, starts = csr[i]
                    lengths = starts[sel + 1] - starts[sel]
                    offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
                    cand = c_order[np.repeat(starts[sel], lengths) + offsets]
                    obs = to_device({k: rows.obs[k] for k in keys}, sel, device)
                    logits = hitnet.hit_logits(hitnet.trunk(obs))
                    pos = torch.as_tensor(np.repeat(np.arange(len(sel)), lengths), device=device)
                    out = logits[pos, torch.as_tensor(rows.c_joint[cand], device=device)]
                else:
                    cand = sel
                    obs = to_device({k: rows.obs[k] for k in keys}, rows.c_row[sel], device)
                    logits = hitnet.hit_logits(hitnet.trunk(obs))
                    out = logits.gather(1, torch.as_tensor(rows.c_joint[sel], device=device).unsqueeze(-1)).squeeze(-1)
                y = torch.as_tensor(rows.c_lost[cand].astype(np.float32), device=device)
                losses.append(F.binary_cross_entropy_with_logits(out, y, reduction='sum'))
                n += len(cand)
            loss = sum(losses) / n
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            schedule.step()
            total += float(loss) * n
            count += n
        hitnet.eval()
        report = {'epoch': epoch + 1, 'train_loss': total / count}
        for rows in data:
            held = np.flatnonzero(rows.held)
            report[rows.name] = pr_report(rows, hit_logits(hitnet, rows.obs, held, device), held)
        history.append(report)
        print(json.dumps(report), flush=True)
    name, run = save.split('/', 1)
    path = registry.save_checkpoint(name, run, hitnet.cpu(), extra={'sources': sources, 'history': history,
                                                                     'init': init})
    print(f'[retrain] saved {path}', flush=True)


# ---------------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('combine')
    c.add_argument('--policy', required=True)
    c.add_argument('--hit', required=True)
    c.add_argument('--weights', type=float, nargs='+', required=True)
    c.add_argument('--form', default='softplus', choices=['softplus', 'logit'])
    c.add_argument('--name', default='r38shift')
    c.add_argument('--run', required=True)
    c.add_argument('--freeze-hit', action='store_true', help="the hit body out of training (a reinforcement arm)")
    m = sub.add_parser('manifests')
    m.add_argument('--goriya-states', type=int, default=48)
    m.add_argument('--goriya-episodes', type=int, default=10)
    m.add_argument('--episodes-0x1b', type=int, default=100)
    m.add_argument('--out', type=Path, default=R38)
    k = sub.add_parser('calibrate')
    k.add_argument('--hit', required=True)
    k.add_argument('--policy', required=True)
    k.add_argument('--data', nargs='+', required=True)
    k.add_argument('--weights', type=float, nargs='+', default=[0.5, 1, 2, 3, 4, 6, 8])
    k.add_argument('--form', default='softplus', choices=['softplus', 'logit'])
    k.add_argument('--out', type=Path, default=None)
    b = sub.add_parser('bench')
    b.add_argument('--manifest', nargs='+', required=True)
    b.add_argument('--arm', nargs='+', required=True, metavar='NAME=CKPT[@ACT]')
    b.add_argument('--out', type=Path, required=True)
    b.add_argument('--parallel', type=int, default=10)
    b.add_argument('--logs-dir', default=None)
    p = sub.add_parser('pr')
    p.add_argument('--hit', nargs='+', required=True)
    p.add_argument('--data', nargs='+', required=True)
    p.add_argument('--all', action='store_true', help='every decision, not only the held-out ones')
    p.add_argument('--out', type=Path, default=None)
    r = sub.add_parser('retrain')
    r.add_argument('--init', required=True)
    r.add_argument('--data', nargs='+', required=True)
    r.add_argument('--repeat', type=float, nargs='*', default=None, help='times each source is repeated an epoch')
    r.add_argument('--save', required=True, metavar='NAME/RUN')
    r.add_argument('--epochs', type=int, default=10)
    r.add_argument('--lr', type=float, default=4e-4)
    r.add_argument('--batch', type=int, default=1024)
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--fraction', type=float, default=1.0, help="share of each source's training decisions kept")
    r.add_argument('--by-row', action='store_true', help='batches of decisions (--batch of them), the trunk run once '
                   'a decision for all its candidates (fast on recordings: 9 candidates a decision)')
    t = sub.add_parser('table')
    t.add_argument('directory', type=Path)
    t.add_argument('--base', default='teacher')
    for q in (c, k, p, r):
        q.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.command in ('calibrate', 'bench', 'pr', 'retrain'):
        require_job()
    if args.command == 'combine':
        combine(args.policy, args.hit, args.weights, args.name, args.run, args.form, args.freeze_hit)
    elif args.command == 'manifests':
        write_manifests(args.out, args.goriya_states, args.goriya_episodes, args.episodes_0x1b)
    elif args.command == 'calibrate':
        device = torch.device(args.device)
        policy, hitnet = load(args.policy).to(device).eval(), load(args.hit).to(device).eval()
        results = {s: calibrate(load_rows(s), policy, hitnet, [0.0] + args.weights, device, args.form)
                   for s in args.data}
        if args.out:
            write_json_atomic(args.out, results)
    elif args.command == 'bench':
        bench(args.manifest, args.arm, args.out, args.parallel, args.logs_dir)
    elif args.command == 'pr':
        device = torch.device(args.device)
        results = {}
        for source in args.data:
            rows = load_rows(source)
            index = np.arange(rows.rows) if args.all else np.flatnonzero(rows.held)
            for hit in args.hit:
                network = load(hit).to(device).eval()
                results[f'{hit} on {source}'] = pr_report(rows, hit_logits(network, rows.obs, index, device), index)
                print(hit, source, json.dumps(results[f'{hit} on {source}']), flush=True)
        if args.out:
            write_json_atomic(args.out, results)
    elif args.command == 'retrain':
        retrain(args.init, args.data, args.save, args.epochs, args.lr, args.batch, torch.device(args.device),
                args.seed, args.repeat, args.fraction, args.by_row)
    elif args.command == 'table':
        print(table(args.directory, args.base))
    return 0


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    sys.exit(main())
