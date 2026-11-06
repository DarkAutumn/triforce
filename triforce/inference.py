"""Batched policy inference over native pools: training rollouts (``ppo.Collector``), evaluation
(``evaluate``) and chain runs (``runner.ChainPool``).

Every forward has a fixed shape of one row per env (``RolloutPolicy``), so a row's outputs depend only on its own
observation and noise, never on which other envs happen to be waiting; on CUDA it is a replayed CUDA graph.
Sampled actions use Gumbel noise: training draws it per rollout (``ppo.Collector``); evaluation and chains key it
to the job the env is playing (``keyed_noise``: the eval job's or chain run's seed and the decision index), so a
seed plays the same game whichever env or thread runs it.  ``--deterministic`` is zero noise (argmax).
"""

import copy
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .actions import NUM_KINDS
from .model import MASK_KEY, NOISE_SIZE, ZeldaPolicy

_NOISE_TAG = 0x7F4A_7C15_9E37_79B9       # separates this stream from other uses of the same seeds


def _mix(x: np.ndarray) -> np.ndarray:
    """SplitMix64's finalizer on uint64 arrays (wrapping arithmetic)."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def keyed_noise(jobs, steps) -> np.ndarray:
    """Standard Gumbel noise ``[k, NOISE_SIZE]`` (float32) for decision ``steps[i]`` of job ``jobs[i]``: a pure
    function of (job, step, column), counter-based, so it doesn't matter which env draws it or when."""
    with np.errstate(over='ignore'):
        jobs = np.asarray(jobs, dtype=np.int64).astype(np.uint64)
        steps = np.asarray(steps, dtype=np.int64).astype(np.uint64)
        key = _mix(_mix(jobs ^ np.uint64(_NOISE_TAG)) + steps * np.uint64(0x9E3779B97F4A7C15))
        columns = np.arange(NOISE_SIZE, dtype=np.uint64) * np.uint64(0xD1B54A32D192ED03)
        bits = _mix(key[:, None] ^ columns[None, :])
    uniform = ((bits >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0 ** -53       # (0, 1)
    return (-np.log(-np.log(uniform))).astype(np.float32)


def check_kind_masks(masks: np.ndarray, envs: np.ndarray):
    """Raises ValueError naming the first env whose flat mask (``masks[i]`` for ``envs[i]``) allows no kind."""
    kinds = np.asarray(masks)[:, :NUM_KINDS].any(axis=1)
    if not kinds.all():
        bad = int(np.flatnonzero(~kinds)[0])
        raise ValueError(f"Empty kind mask for env {int(envs[bad])}: {np.asarray(masks)[bad].tolist()}")


def _flat_layout(spec: dict) -> tuple[dict, int]:
    """Byte offsets of ``spec`` (name → (shape, numpy dtype)) packed into one buffer, 64-byte aligned."""
    layout, offset = {}, 0
    for name, (shape, dtype) in spec.items():
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        layout[name] = (offset, nbytes, tuple(shape), torch.from_numpy(np.empty(0, dtype)).dtype)
        offset += -(-nbytes // 64) * 64
    return layout, offset


def _flat_views(flat: torch.Tensor, layout: dict) -> dict:
    """Typed tensor views into the byte buffer ``flat``."""
    return {name: flat[offset:offset + nbytes].view(dtype).view(shape)
            for name, (offset, nbytes, shape, dtype) in layout.items()}


class RolloutPolicy:
    """The rollout forward pass over a fixed batch of one row per env.

    Holds its own eval-mode copy of the network on ``device`` (``load`` refreshes the weights in place), host
    staging arrays (``obs`` per key including ``action_mask``, ``noise``; numpy views of one pinned buffer) that hold
    each env's latest observation, and the outputs (``actions``, ``logp_ent_val``).  Every forward evaluates all
    ``n`` rows, so a row's outputs depend only on its own observation and noise, not on which other envs are
    waiting.  On CUDA the forward is captured once as a CUDA graph whose first and last nodes copy the staging in
    and the outputs out (one transfer each way); ``launch`` queues it and ``wait`` blocks until the outputs are on
    the host, so the caller can do bookkeeping in between.
    """
    # pylint: disable=too-many-instance-attributes

    _NOISE = '_noise'

    def __init__(self, network: nn.Module, observation: dict, device):
        """``observation``: per-key ``[n, *shape]`` arrays (the pool's ``obs`` rows): shapes, dtypes and the
        initial staged rows."""
        self.device = torch.device(device)
        self.network = copy.deepcopy(network).to(self.device).eval()
        n = len(observation[MASK_KEY])
        inputs = {key: (np.shape(value), np.asarray(value).dtype) for key, value in observation.items()}
        inputs[self._NOISE] = ((n, NOISE_SIZE), np.float32)
        in_layout, in_bytes = _flat_layout(inputs)
        out_layout, out_bytes = _flat_layout({'actions': ((n, 3), np.int64), 'logp_ent_val': ((n, 3), np.float32)})
        cuda = self.device.type == 'cuda'
        self._host_in = torch.zeros(in_bytes, dtype=torch.uint8, pin_memory=cuda)
        self._host_out = torch.zeros(out_bytes, dtype=torch.uint8, pin_memory=cuda)
        host_in, host_out = _flat_views(self._host_in, in_layout), _flat_views(self._host_out, out_layout)
        self.obs = {key: host_in[key].numpy() for key in observation}
        for key, value in observation.items():
            self.obs[key][...] = value
        self.noise = host_in[self._NOISE].numpy()
        self.actions, self.logp_ent_val = host_out['actions'].numpy(), host_out['logp_ent_val'].numpy()
        self._graph = None
        if cuda:
            self._device_in, self._device_out = self._host_in.to(self.device), self._host_out.to(self.device)
            self._done = torch.cuda.Event()
            self._capture(_flat_views(self._device_in, in_layout), _flat_views(self._device_out, out_layout))
        else:
            self._views = (host_in, host_out)

    @torch.no_grad()
    def _forward(self, inputs: dict, outputs: dict):
        obs = {key: value for key, value in inputs.items() if key not in (MASK_KEY, self._NOISE)}
        out = self.network.evaluate(obs, inputs[MASK_KEY], noise=inputs[self._NOISE])
        outputs['actions'].copy_(out.actions)
        outputs['logp_ent_val'].copy_(torch.stack([out.log_prob, out.entropy, out.value], dim=-1))

    def _capture(self, inputs: dict, outputs: dict):
        def step():
            self._device_in.copy_(self._host_in, non_blocking=True)
            self._forward(inputs, outputs)
            self._host_out.copy_(self._device_out, non_blocking=True)

        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):              # warm up kernels/allocations off the capture stream
            for _ in range(3):
                step()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            step()

    def load(self, network: nn.Module):
        """Copies ``network``'s weights into the rollout copy (in place, so the captured graph sees them)."""
        self.network.load_state_dict(network.state_dict())

    def launch(self):
        """Starts evaluating every staged row (on CPU: evaluates them)."""
        if self._graph is None:
            self._forward(*self._views)
        else:
            self._graph.replay()
            self._done.record()

    def wait(self) -> tuple[np.ndarray, np.ndarray]:
        """(actions ``[n, 3]`` int64, ``[n, 3]`` log-prob/entropy/value) of the last ``launch``."""
        if self._graph is not None:
            self._done.synchronize()
        return self.actions, self.logp_ent_val

    @torch.no_grad()
    def value(self, obs: dict) -> np.ndarray:
        """Value estimates ``[k]`` of ``obs`` (per-key ``[k, *shape]``; eager, any batch size)."""
        return self.network.get_value({k: v for k, v in obs.items() if k != MASK_KEY}).cpu().numpy()


class PoolInference:
    """Actions for rows of an eval or chain pool, one fixed-shape ``RolloutPolicy`` per policy index.

    ``networks[p]`` acts on rows whose ``policy`` column is ``p``.  A ``ZeldaPolicy`` runs through a lazily built
    ``RolloutPolicy`` over ``num_envs`` rows (staging row = env id); anything else with ``get_action(obs,
    deterministic=...)`` (scripted test policies) is called on its rows directly.  Sampled actions use
    ``keyed_noise(job, step)``; ``deterministic`` uses zero noise, i.e. each head's argmax.
    """

    def __init__(self, networks: Sequence, num_envs: int, device='cpu', deterministic: bool = False):
        self.networks = list(networks)
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.deterministic = deterministic
        self._forwards: dict[int, RolloutPolicy] = {}

    def _forward(self, policy: int, obs: dict) -> RolloutPolicy:
        forward = self._forwards.get(policy)
        if forward is None:
            template = {key: np.zeros((self.num_envs, *np.shape(value)[1:]), dtype=np.asarray(value).dtype)
                        for key, value in obs.items()}
            forward = self._forwards[policy] = RolloutPolicy(self.networks[policy], template, self.device)
        return forward

    def actions(self, envs, obs: dict, policy, jobs, steps) -> np.ndarray:
        """``[k, 3]`` int64 actions for env ``envs[i]`` with observation row ``i`` of ``obs`` (per key
        ``[k, ...]``), acted on by ``policy[i]``, at decision ``steps[i]`` of job ``jobs[i]``."""
        envs, policy = np.asarray(envs, dtype=np.int64), np.asarray(policy, dtype=np.int64)
        actions = np.zeros((len(envs), 3), dtype=np.int64)
        if not len(envs):
            return actions
        noise = np.zeros((len(envs), NOISE_SIZE), np.float32) if self.deterministic else keyed_noise(jobs, steps)
        launched = []
        for index in np.unique(policy):
            rows = np.flatnonzero(policy == index)
            network = self.networks[int(index)]
            subset = {key: value[rows] for key, value in obs.items()}
            if not isinstance(network, ZeldaPolicy):
                with torch.no_grad():
                    chosen = network.get_action(subset, deterministic=self.deterministic)
                actions[rows] = np.asarray(torch.as_tensor(chosen).cpu(), dtype=np.int64).reshape(-1, 3)
                continue
            check_kind_masks(subset[MASK_KEY], envs[rows])
            forward, ids = self._forward(int(index), obs), envs[rows]
            for key, value in subset.items():
                forward.obs[key][ids] = value
            forward.noise[ids] = noise[rows]
            forward.launch()
            launched.append((forward, rows, ids))
        for forward, rows, ids in launched:
            actions[rows] = forward.wait()[0][ids]
        return actions
