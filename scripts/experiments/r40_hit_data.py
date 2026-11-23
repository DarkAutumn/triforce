"""Round 40 (docs/experiments/round40.md "Observation v7's danger block"): hit labels with the RAM beside them, for
the supervised hit bench (``r40_obs_bench.py``).

    python scripts/experiments/r40_hit_data.py goriya@d3first --decisions 120000 --out target/r40/data/goriya
        [--route d3first] [--parallel 8] [--envs 8] [--try 4:0.5] [--seed 0] [--express]

Experiment 2's collections (round36.md §1, ``r35_guard_data.py --probe``) are gone with their worktree, and the lab's
``record`` keeps no RAM; this rebuilds what the bench needs from the single-env API alone (no engine change). The
family's teacher (``registry.policy_checkpoint(KEY, route=ROUTE)``) plays its task (``lab.collect.task_yaml``, on
``ROUTE``'s states: ``route_only``) at temperature 0.5; at every decision the RAM is kept beside the observation, and
on a share P of the decisions K other candidates (a sample without replacement from the teacher's plain policy, as
the lab's ``--try-others``) are each played from a snapshot in a second env (the game's own future, training-only
look-ahead; the played run never restores). A candidate is lost when Link's health is lower after it. Each lost
trial keeps the RAM at its end (``tried_end_ram``) so the bench can tell what hit (a shot that was not flying at the
decision, for the threat lane's slice).

One ``shard-<worker>-<chunk>.npz`` a chunk of decisions: every observation key as ``obs/<key>`` (``grid`` uint8,
``entity_types`` / ``room_id`` int16), ``mask``, ``ram`` [n, 2048] uint8, ``episode`` (worker * 10^6 + its
episode), ``taken`` (joint index), ``taken_lost``; the candidates ``tried_row`` (the decision's row in the shard),
``tried_joint``, ``tried_lost``, ``tried_taken`` (1: the action played, its real outcome), ``tried_end`` (row into
``tried_end_ram``, -1 when kept nothing). In the job queue (it plays a network)."""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEMPERATURE = 0.5
CHUNK = 2000
COMPACT = {'grid': np.uint8, 'entity_types': np.int16, 'room_id': np.int16}

_W: dict = {}


def _init(key: str, route: str, checkpoint: str, envs: int):
    import torch  # pylint: disable=import-outside-toplevel
    torch.set_num_threads(1)
    from triforce import _native, registry  # pylint: disable=import-outside-toplevel
    from triforce.lab.collect import route_only, task_yaml  # pylint: disable=import-outside-toplevel
    from triforce.specialist import load_specialist  # pylint: disable=import-outside-toplevel
    config = load_specialist(task_yaml(key))
    if route:
        config = route_only(config, route)
    _W.update(config=config, envs=envs, native=_native, torch=torch,
              model=registry.load_model(checkpoint)[0].eval())


def _joint_action(j: int) -> tuple[int, int, int]:
    return j // 32, (j // 4) % 8, j % 4


def _chunk(task):
    """``CHUNK`` decisions of worker ``wid`` (fresh envs seeded by the chunk), written to ``out``."""
    wid, chunk, seed, k, share, out = task
    torch, native, model = _W['torch'], _W['native'], _W['model']
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    n = _W['envs']
    envs = [native.Env(_W['config'], seed=seed * 64 + i, strict_actions=False) for i in range(n)]
    probe = native.Env(_W['config'], seed=seed * 64 + 63, strict_actions=False)
    obs = [e.reset(seed=seed * 64 + i)[0] for i, e in enumerate(envs)]
    episode = [chunk * 1000 + i for i in range(n)]
    next_episode = chunk * 1000 + n
    rows: dict[str, list] = {'ram': [], 'episode': [], 'taken': [], 'taken_lost': [], 'mask': []}
    obs_rows: dict[str, list] = {key: [] for key in obs[0] if key != 'action_mask'}
    tried = {'tried_row': [], 'tried_joint': [], 'tried_lost': [], 'tried_taken': [], 'tried_end': []}
    end_ram: list = []
    while len(rows['taken']) < CHUNK:
        batch = {key: torch.as_tensor(np.stack([o[key] for o in obs])) for key in obs[0]}
        with torch.no_grad():
            lp = model.joint_log_probs(batch, batch['action_mask'])
        plain = torch.softmax(lp, -1)
        tempered = torch.softmax(lp / TEMPERATURE, -1)
        taken = torch.multinomial(tempered, 1, generator=gen).squeeze(-1).numpy()
        for i, env in enumerate(envs):
            state = env.state
            ram = np.frombuffer(state.ram, dtype=np.uint8)[:2048].copy()
            hp = state.health
            row = len(rows['taken'])
            for key in obs_rows:
                obs_rows[key].append(obs[i][key].astype(COMPACT.get(key, obs[i][key].dtype), copy=False))
            rows['mask'].append(obs[i]['action_mask'])
            rows['ram'].append(ram)
            rows['episode'].append(wid * 1_000_000 + episode[i])
            rows['taken'].append(int(taken[i]))
            if rng.random() < share:
                p = plain[i].clone()
                p[taken[i]] = 0
                legal = int((p > 0).sum())
                if legal:
                    picks = torch.multinomial(p, min(k, legal), replacement=False, generator=gen).tolist()
                    snap = env.snapshot()
                    for j in picks:
                        probe.restore(snap, 'probe')
                        probe.step(_joint_action(j))
                        after = probe.state
                        lost = after.health < hp - 1e-6
                        tried['tried_row'].append(row)
                        tried['tried_joint'].append(j)
                        tried['tried_lost'].append(int(lost))
                        tried['tried_taken'].append(0)
                        tried['tried_end'].append(len(end_ram) if lost else -1)
                        if lost:
                            end_ram.append(np.frombuffer(after.ram, dtype=np.uint8)[:2048].copy())
            o, _, terminated, truncated, _ = env.step(_joint_action(int(taken[i])))
            after = env.state
            lost = after.health < hp - 1e-6
            rows['taken_lost'].append(int(lost))
            tried['tried_row'].append(row)
            tried['tried_joint'].append(int(taken[i]))
            tried['tried_lost'].append(int(lost))
            tried['tried_taken'].append(1)
            tried['tried_end'].append(len(end_ram) if lost else -1)
            if lost:
                end_ram.append(np.frombuffer(after.ram, dtype=np.uint8)[:2048].copy())
            if terminated or truncated:
                o = env.reset(seed=int(rng.integers(1 << 31)))[0]
                episode[i] = next_episode
                next_episode += 1
            obs[i] = o
    arrays = {f'obs/{key}': np.stack(v) for key, v in obs_rows.items()}
    arrays.update({key: np.asarray(v) for key, v in rows.items()})
    arrays['ram'] = np.stack(rows['ram'])
    arrays['mask'] = np.stack(rows['mask'])
    arrays.update({key: np.asarray(v, dtype=np.int64) for key, v in tried.items()})
    arrays['tried_end_ram'] = np.stack(end_ram) if end_ram else np.zeros((0, 2048), np.uint8)
    path = Path(out) / f'shard-{wid:02d}-{chunk:04d}.npz'
    np.savez_compressed(path.with_suffix('.tmp.npz'), **arrays)
    os.replace(path.with_suffix('.tmp.npz'), path)
    return len(rows['taken']), int(np.sum(tried['tried_lost'])), len(tried['tried_lost'])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('key', help='a policy key, e.g. goriya@d3first')
    parser.add_argument('--route', default='d3first')
    parser.add_argument('--decisions', type=int, default=120000)
    parser.add_argument('--parallel', type=int, default=8)
    parser.add_argument('--envs', type=int, default=8, help='envs a worker plays side by side')
    parser.add_argument('--try', dest='try_others', default='4:0.5', metavar='K:P')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--logs-dir', default=None)
    from triforce.jobs import add_job_arguments, begin_job, work  # pylint: disable=import-outside-toplevel
    from triforce.events import progress_job  # pylint: disable=import-outside-toplevel
    from triforce import registry  # pylint: disable=import-outside-toplevel
    add_job_arguments(parser)
    args = parser.parse_args(argv)
    begin_job(args)
    k, share = args.try_others.split(':')
    checkpoint = str(registry.policy_checkpoint(args.key, route=args.route))
    args.out.mkdir(parents=True, exist_ok=True)
    chunks = (args.decisions + CHUNK - 1) // CHUNK
    tasks = [(c % args.parallel, c, args.seed * 100_000 + c, int(k), float(share), str(args.out))
             for c in range(chunks) if not any(args.out.glob(f'shard-*-{c:04d}.npz'))]
    (args.out / 'dataset.json').write_text(json.dumps({
        'key': args.key, 'route': args.route, 'checkpoint': checkpoint, 'temperature': TEMPERATURE,
        'try': args.try_others, 'seed': args.seed, 'chunk': CHUNK, 'chunks': chunks}, indent=1))
    t0 = time.time()
    name = f"r40-hit-data-{args.key.replace('@', '-')}"
    with progress_job(name, 'eval', args.decisions, unit='decisions', root=args.logs_dir,
                      work=work(eval_episodes=max(1, args.decisions // 400))) as progress:
        done = (chunks - len(tasks)) * CHUNK
        lost = total = 0
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.parallel, initializer=_init, initargs=(args.key, args.route, checkpoint, args.envs)) as pool:
            for rows, l, t in pool.imap_unordered(_chunk, tasks):
                done, lost, total = done + rows, lost + l, total + t
                progress.update(done)
                print(f'{done} decisions, {total} judged candidates, {lost / max(total, 1):.3f} lost, '
                      f'{(time.time() - t0) / 60:.1f} min', flush=True)
    return 0


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    sys.exit(main())
