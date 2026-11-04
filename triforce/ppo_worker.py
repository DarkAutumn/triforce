"""Subprocess rollout workers (port of v1 ``ml_ppo_worker``).

stable-retro allows one emulator per process, so each worker owns one env.  The main process writes the
current weights into shared tensors and releases the workers through a barrier; each worker fills its slice
of the shared ``RolloutBuffer`` and reports its ``RolloutMetrics`` (raw counts) over a pipe, then meets the
main process at a second barrier.
"""

import random
import threading
import traceback

import numpy as np
import torch
import torch.multiprocessing as mp

from .metrics import EpisodeTracker, RolloutMetrics
from .specialist import SpecialistConfig


class WorkerError(Exception):
    """Wraps a formatted traceback from a crashed worker subprocess."""


class ZeldaEnvFactory:
    """Picklable factory for ``env.ZeldaEnv`` in a worker process."""
    def __init__(self, specialist: str | SpecialistConfig, **kwargs):
        self.specialist = specialist
        self.kwargs = kwargs

    def __call__(self, session=None):
        from .env import ZeldaEnv  # pylint: disable=import-outside-toplevel
        return ZeldaEnv(self.specialist, session=session, **self.kwargs)


class ModelFactory:
    """Picklable factory that rebuilds the network in a worker (class + constructor args)."""
    def __init__(self, network_class, *args, **kwargs):
        self.network_class = network_class
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.network_class(*self.args, **self.kwargs)


def _worker_loop(env_index, shared_buffer, shared_weights, make_network, make_env, sync, seed):
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    torch.set_num_threads(1)
    weights_barrier, rollouts_barrier, close_flag, conn = sync
    env = None
    waiting_for_weights = True
    try:
        if seed is not None:
            seed = (seed + env_index) % (1 << 64)
            random.seed(seed)
            np.random.seed(seed % (1 << 32))
            torch.manual_seed(seed)
        env = make_env()
        network = make_network()
        tracker = EpisodeTracker()
        while True:
            weights_barrier.wait()
            waiting_for_weights = False
            if close_flag.value:
                break
            network.load_state_dict(shared_weights)
            shared_buffer.collect(env_index, network, env, tracker, seed=seed)
            conn.send(tracker.take())
            rollouts_barrier.wait()
            waiting_for_weights = True
    except Exception:  # pylint: disable=broad-except
        try:
            conn.send(WorkerError(f"Worker {env_index} crashed:\n{traceback.format_exc()}"))
        except (BrokenPipeError, OSError):
            pass
        # Meet the main process at the barriers it is (or will be) waiting on, so it sees the error.
        try:
            if waiting_for_weights and not close_flag.value:
                weights_barrier.wait(timeout=600)
            rollouts_barrier.wait(timeout=600)
        except Exception:  # pylint: disable=broad-except
            pass
    finally:
        if env is not None:
            env.close()
        conn.close()


class RolloutWorkerPool:
    """One subprocess per env, filling a shared-memory RolloutBuffer."""

    def __init__(self, n_workers, shared_buffer, network, make_network, make_env, *, seed=None):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.n_workers = n_workers
        self.workers = []
        self.conns = []
        ctx = mp.get_context('spawn')
        self._weights_barrier = ctx.Barrier(n_workers + 1)
        self._rollouts_barrier = ctx.Barrier(n_workers + 1)
        self._close_flag = ctx.Value('b', False)
        self._shared_weights = {k: v.detach().cpu().clone().share_memory_() for k, v in network.state_dict().items()}

        for i in range(n_workers):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=_worker_loop,
                                  args=(i, shared_buffer, self._shared_weights, make_network, make_env,
                                        (self._weights_barrier, self._rollouts_barrier, self._close_flag, child), seed),
                                  daemon=True)
            process.start()
            child.close()
            self.conns.append(parent)
            self.workers.append(process)

    def collect(self, network) -> RolloutMetrics:
        """Publishes weights, waits for every worker to fill its slice, returns merged metrics."""
        with torch.no_grad():
            for key, value in network.state_dict().items():
                self._shared_weights[key].copy_(value.detach().cpu())
        self._weights_barrier.wait()
        self._rollouts_barrier.wait()

        merged = RolloutMetrics()
        errors = []
        for conn in self.conns:
            result = conn.recv()
            if isinstance(result, WorkerError):
                errors.append(str(result))
            else:
                merged.merge(result)
        if errors:
            raise WorkerError('\n'.join(errors))
        return merged

    def close(self):
        """Shuts down all workers."""
        self._close_flag.value = True
        try:
            self._weights_barrier.wait(timeout=5)
        except (threading.BrokenBarrierError, BrokenPipeError, OSError):
            pass
        for conn in self.conns:
            try:
                conn.close()
            except (BrokenPipeError, OSError):
                pass
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
        self.conns.clear()
        self.workers.clear()
