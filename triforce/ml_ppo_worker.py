"""Subprocess worker for parallel PPO rollout collection using shared memory."""

import torch.multiprocessing as mp

from .metrics import MetricTracker


class EnvFactory:
    """Picklable factory for creating Zelda environments in worker subprocesses."""
    def __init__(self, scenario_def, action_space, **kwargs):
        self.scenario_def = scenario_def
        self.action_space = action_space
        self.kwargs = kwargs

    def __call__(self):
        from .zelda_env import make_zelda_env  # pylint: disable=import-outside-toplevel
        return make_zelda_env(self.scenario_def, self.action_space, **self.kwargs)


class SimpleEnvFactory:
    """Picklable factory that creates an env from a class and args."""
    def __init__(self, env_class, *args, **kwargs):
        self.env_class = env_class
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.env_class(*self.args, **self.kwargs)


def _worker_loop(env_index, shared_buffer, shared_weights, network_class,
                 create_env_fn, sync):
    """Main loop for a rollout worker subprocess using shared memory.

    Args:
        sync: tuple of (weights_barrier, rollouts_barrier, close_flag, metrics_conn)
    """
    import torch  # pylint: disable=import-outside-toplevel
    torch.set_num_threads(2)

    weights_barrier, rollouts_barrier, close_flag, metrics_conn = sync

    env = create_env_fn()
    try:
        network = network_class(env.observation_space, env.action_space)

        while True:
            # Wait for main to update shared weights
            weights_barrier.wait()

            if close_flag.value:
                break

            # Load weights from shared memory (memcpy, not pickle)
            network.load_state_dict(shared_weights)
            network.eval()

            # Collect rollout directly into shared buffer slice
            shared_buffer.ppo_main_loop(env_index, network, env, None)

            # Send metrics via pipe (small data, pickle is fine)
            metrics = MetricTracker.get_metrics_and_clear()
            metrics_conn.send(metrics)

            # Signal rollout complete
            rollouts_barrier.wait()
    finally:
        env.close()
        metrics_conn.close()


def _aggregate_metrics(metrics_list):
    """Averages metric dicts from multiple workers."""
    if not metrics_list:
        return {}

    combined = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            if key not in combined:
                combined[key] = []
            combined[key].append(value)

    return {key: sum(values) / len(values) for key, values in combined.items()}


class RolloutWorkerPool:
    """Manages subprocess workers for parallel rollout collection via shared memory."""
    def __init__(self, n_workers, shared_buffer, network, create_env_fn, network_class):
        self.n_workers = n_workers
        self.workers = []
        self.metric_conns = []

        ctx = mp.get_context('spawn')
        self._weights_barrier = ctx.Barrier(n_workers + 1)
        self._rollouts_barrier = ctx.Barrier(n_workers + 1)
        self._close_flag = ctx.Value('b', False)

        # Create shared weight tensors (memcpy instead of pickle for updates)
        self._shared_weights = {k: v.cpu().clone().share_memory_()
                                for k, v in network.state_dict().items()}

        for i in range(n_workers):
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=_worker_loop,
                args=(i, shared_buffer, self._shared_weights, network_class,
                      create_env_fn,
                      (self._weights_barrier, self._rollouts_barrier,
                       self._close_flag, child_conn)),
                daemon=True
            )
            process.start()
            child_conn.close()
            self.metric_conns.append(parent_conn)
            self.workers.append(process)

    def update_weights(self, network):
        """Copies network weights to shared memory and signals workers to start."""
        for k, v in network.state_dict().items():
            self._shared_weights[k].copy_(v.cpu())

        # Release all workers past the weights barrier
        self._weights_barrier.wait()

    def collect_rollouts(self, target_buffer, progress=None):
        """Waits for all workers to finish writing into the shared buffer."""
        # Wait for all workers to complete their rollouts
        self._rollouts_barrier.wait()

        # Collect metrics from pipes (small data)
        all_metrics = []
        for conn in self.metric_conns:
            metrics = conn.recv()
            if metrics:
                all_metrics.append(metrics)

        if progress:
            progress.update(target_buffer.memory_length * target_buffer.n_envs)

        aggregated = _aggregate_metrics(all_metrics)
        return target_buffer.memory_length, aggregated

    def close(self):
        """Shuts down all workers."""
        self._close_flag.value = True
        try:
            self._weights_barrier.wait(timeout=5)
        except mp.context.TimeoutError:
            pass

        for conn in self.metric_conns:
            try:
                conn.close()
            except (BrokenPipeError, OSError):
                pass

        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        self.metric_conns.clear()
        self.workers.clear()
