"""Subprocess worker for parallel PPO rollout collection using shared memory."""

import threading
import traceback

import torch.multiprocessing as mp

from .metrics import MetricTracker


class WorkerError(Exception):
    """Wraps a formatted traceback from a crashed worker subprocess."""


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


class WeightedEnvFactory:
    """Picklable factory for creating weighted Zelda environments in worker subprocesses.

    The selector_proxy is a multiprocessing.managers proxy to a WeightedScenarioSelector
    in the main process. Manager proxies are picklable (they serialize over sockets).
    """
    def __init__(self, scenario_defs, action_space, selector_proxy, **kwargs):
        self.scenario_defs = scenario_defs
        self.action_space = action_space
        self.selector_proxy = selector_proxy
        self.kwargs = kwargs

    def __call__(self):
        from .zelda_env import make_weighted_zelda_env  # pylint: disable=import-outside-toplevel
        return make_weighted_zelda_env(self.scenario_defs, self.action_space,
                                       self.selector_proxy, **self.kwargs)


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
    except Exception:  # pylint: disable=broad-except
        # Send formatted traceback to main process instead of printing to stderr
        tb = traceback.format_exc()
        try:
            metrics_conn.send({"__worker_error__": f"Worker {env_index} crashed:\n{tb}"})
        except (BrokenPipeError, OSError):
            pass
        # Still reach the rollouts barrier so main doesn't deadlock
        try:
            rollouts_barrier.wait()
        except Exception:  # pylint: disable=broad-except
            pass  # best-effort barrier release
    finally:
        env.close()
        metrics_conn.close()


def _reduce_metric(key, values):
    """Reduce a list of metric values: max() for '/max' keys, mean for everything else."""
    if '/max' in key:
        return max(values)
    return sum(values) / len(values)


def _aggregate_metrics(metrics_list):
    """Averages metric dicts from multiple workers.

    Handles both flat dicts {metric: value} and per-scenario nested dicts
    {scenario: {metric: value}} from weighted mode.
    """
    if not metrics_list:
        return {}

    # Detect weighted mode: values are dicts instead of numbers
    first_value = next(iter(metrics_list[0].values()), None)
    if isinstance(first_value, dict):
        return _aggregate_weighted_metrics(metrics_list)

    combined = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            if key not in combined:
                combined[key] = []
            combined[key].append(value)

    return {key: _reduce_metric(key, values) for key, values in combined.items()}
def _aggregate_weighted_metrics(metrics_list):
    """Averages per-scenario metric dicts: {scenario: {metric: [values]}}."""
    combined = {}
    for metrics in metrics_list:
        for scenario, scenario_metrics in metrics.items():
            if scenario not in combined:
                combined[scenario] = {}
            for key, value in scenario_metrics.items():
                if key not in combined[scenario]:
                    combined[scenario][key] = []
                combined[scenario][key].append(value)

    return {scenario: {key: _reduce_metric(key, vals) for key, vals in metrics.items()}
            for scenario, metrics in combined.items()}


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

    def collect_rollouts(self, target_buffer):
        """Waits for all workers to finish writing into the shared buffer."""
        # Wait for all workers to complete their rollouts
        self._rollouts_barrier.wait()

        # Collect metrics from pipes (small data)
        all_metrics = []
        for conn in self.metric_conns:
            metrics = conn.recv()
            if metrics:
                # Check for worker crash
                if "__worker_error__" in metrics:
                    raise WorkerError(metrics["__worker_error__"])
                all_metrics.append(metrics)

        aggregated = _aggregate_metrics(all_metrics)
        return target_buffer.memory_length, aggregated

    def close(self):
        """Shuts down all workers."""
        self._close_flag.value = True
        try:
            self._weights_barrier.wait(timeout=5)
        except (threading.BrokenBarrierError, BrokenPipeError, OSError):
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
