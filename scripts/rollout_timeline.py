"""Rollout timeline of one specialist's training pool under the promoted model (no PPO updates): per rollout,
how long it took, the share of it the main thread spent blocked in ``recv``, how many forwards it ran and their
mean batch, and when the first, median and last env finished its ``T`` steps (the tail).

    .venv/bin/python scripts/rollout_timeline.py SPECIALIST PARALLEL [ROLLOUTS] [--device cuda|cpu]

See docs/performance.md.
"""
import argparse
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from triforce import registry
from triforce._native import EnvPool, TrainTask
from triforce.ppo import Collector
from triforce.specialist import load_specialist
from triforce.train import curriculum_spec


class TimedPool:
    """Forwards to the pool, logging (start, end, env ids) of every ``recv``."""

    def __init__(self, pool):
        self.pool, self.log = pool, []

    def __getattr__(self, attr):
        return getattr(self.pool, attr)

    def recv(self, rows, min_ready=None, timeout=None):
        """``EnvPool.recv``, timed."""
        start = time.perf_counter()
        count = self.pool.recv(rows, min_ready=min_ready, timeout=timeout)
        self.log.append((start, time.perf_counter(), rows['env_ids'][:count].copy()))
        return count


def main():
    """Prints one line per rollout."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('specialist')
    parser.add_argument('parallel', type=int)
    parser.add_argument('rollouts', type=int, nargs='?', default=10)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps-per-update', type=int, default=4096)
    args = parser.parse_args()

    name, n = args.specialist, args.parallel
    network, _ = registry.load_model(f'models/{name}/' + Path(f'models/{name}/promoted').read_text().strip())
    config = load_specialist(name)
    out = Path(tempfile.mkdtemp())
    curriculum, _ = curriculum_spec(argparse.Namespace(no_curriculum=False, curriculum=None, curriculum_fraction=None,
                                                       curriculum_threshold=None), config, out)
    pool = TimedPool(EnvPool(TrainTask(config, str(out), curriculum=curriculum), num_envs=n, threads=n, seed=1))
    torch.manual_seed(1)
    length = args.steps_per_update // n
    collector = Collector(pool, length, 0.99, 0.95)
    for _ in range(args.rollouts):
        pool.log = []
        start = time.perf_counter()
        collector.collect(network, args.device)
        elapsed = time.perf_counter() - start
        blocked = sum(end - begin for begin, end, _ in pool.log)
        counts, finish = np.zeros(n, int), np.zeros(n)
        for _, end, ids in pool.log:
            counts[ids] += 1
            finish[ids[counts[ids] == length]] = end
        finish -= start
        batches = [len(ids) for _, _, ids in pool.log]
        print(f"rollout {elapsed:5.2f}s  main blocked in recv {blocked / elapsed:5.1%}  recvs {len(batches)} "
              f"mean batch {statistics.mean(batches):4.2f}  env finished: first {finish.min():5.2f}s "
              f"median {np.median(finish):5.2f}s last {finish.max():5.2f}s", flush=True)
    pool.close()


if __name__ == '__main__':
    main()
