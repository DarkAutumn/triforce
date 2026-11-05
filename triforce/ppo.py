"""PPO trainer (port of v1 ``ml_ppo``) for one specialist.

Kept from v1: KL early stop, KL rollback (weights *and* optimizer state restored when KL exceeds
``kl_rollback × target_kl``), checkpointable optimizer state, per-minibatch advantage normalization, legacy
time-limit termination, optional demo-regularization (behavior-cloning loss on a fixed demo batch).

Changed for v2: dict observations, factored (kind, direction, duration) actions, flat masks, frame-aware GAE
(see ``rollout_buffer``), and ``train`` counts PPO *iterations* (one rollout + one optimization each).

Rollouts come from a native ``EnvPool`` (``TrainTask``), stepped asynchronously (``Collector``): every env that is
ready gets its next action at once, without waiting for the others, and fills its own rows of the env-major buffer;
a rollout ends when every env has its ``T`` steps.  Determinism does not depend on which envs happen to be ready
together: actions are sampled with per-(env, step) Gumbel noise drawn up front, and the policy always evaluates a
fixed batch of one row per env (``RolloutPolicy``; on CUDA a replayed CUDA graph), so a row's numbers depend only
on its own inputs.  Envs auto-reset: a terminal row's ``obs`` already starts the next episode and ``final_obs``
holds the terminal observation.  Opted-in no-progress truncations (the ``bootstrap`` column) add
``gamma ** (frames / 8) * V(final_obs)`` to that step's reward without crossing the episode reset.
"""

import copy
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn

from .actions import NUM_KINDS
from .demo import compute_demo_accuracy
from .metrics import RolloutMetrics
from .model import MASK_KEY, NOISE_SIZE, gumbel_noise
from .rollout_buffer import FRAMES_PER_DISCOUNT, RolloutBuffer


@dataclass
class PPOConfig:
    """PPO hyperparameters (v1 defaults)."""
    learning_rate: float = 1e-4
    gamma: float = 0.99
    lam: float = 0.95
    target_steps: int = 4096
    num_epochs: int = 10
    minibatches: int = 4
    clip_coeff: float = 0.2
    ent_coeff: float = 0.01
    vf_coeff: float = 0.5
    max_grad_norm: float = 0.5
    epsilon: float = 1e-5
    norm_advantages: bool = True
    clip_val_loss: bool = False
    target_kl: Optional[float] = 0.02
    kl_rollback: Optional[float] = 4.0
    demo_bc_coeff: float = 0.0

    def to_dict(self) -> dict:
        """Plain-data form for checkpoints/logs."""
        return asdict(self)


@dataclass
class IterationResult:
    """What ``PPO.train`` reports to its callback after each iteration."""
    iteration: int            # 1-based count of iterations completed in this call
    total_steps: int          # env steps trained including any loaded checkpoint's steps
    rollout: RolloutMetrics
    stats: dict               # optimization stats + entropy per head + steps/sec


class RolloutPolicy:
    """The rollout forward pass over a fixed batch of one row per env.

    Holds its own eval-mode copy of the network on ``device`` (``load`` refreshes the weights in place) and host
    staging tensors (``obs`` per key including ``action_mask``, ``noise``) that the collector writes ready envs'
    rows into.  Every call evaluates all ``n`` rows, so a row's outputs depend only on its own observation and
    noise, not on which other envs are waiting.  On CUDA the forward is captured once as a CUDA graph and replayed:
    one pinned host→device copy of the staging, one device→host copy of the outputs, one sync per call.
    """

    def __init__(self, network: nn.Module, observation: dict, device):
        """``observation``: per-key ``[n, *shape]`` arrays (the pool's ``obs`` rows) giving shapes and dtypes."""
        self.device = torch.device(device)
        self.network = copy.deepcopy(network).to(self.device).eval()
        pin = self.device.type == 'cuda'
        self.obs = {key: torch.from_numpy(np.array(value)).pin_memory() if pin else torch.from_numpy(np.array(value))
                    for key, value in observation.items()}
        n = len(self.obs[MASK_KEY])
        self.noise = torch.zeros((n, NOISE_SIZE), pin_memory=pin)
        self.actions = torch.zeros((n, 3), dtype=torch.int64, pin_memory=pin)
        self.logp_ent_val = torch.zeros((n, 3), pin_memory=pin)
        self._graph = None
        if pin:
            self._capture()

    def _forward(self, obs, noise):
        out = self.network.evaluate({k: v for k, v in obs.items() if k != MASK_KEY}, obs[MASK_KEY], noise=noise)
        return out.actions, torch.stack([out.log_prob, out.entropy, out.value], dim=-1)

    def _capture(self):
        self._device_obs = {key: value.to(self.device) for key, value in self.obs.items()}
        self._device_noise = self.noise.to(self.device)
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.no_grad():
            with torch.cuda.stream(stream):          # warm up kernels/allocations off the capture stream
                for _ in range(3):
                    self._forward(self._device_obs, self._device_noise)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._device_out = self._forward(self._device_obs, self._device_noise)

    def load(self, network: nn.Module):
        """Copies ``network``'s weights into the rollout copy (in place, so the captured graph sees them)."""
        self.network.load_state_dict(network.state_dict())

    @torch.no_grad()
    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        """(actions ``[n, 3]`` int64, ``[n, 3]`` log-prob/entropy/value) for every staged row."""
        if self._graph is None:
            actions, logp_ent_val = self._forward(self.obs, self.noise)
            self.actions.copy_(actions)
            self.logp_ent_val.copy_(logp_ent_val)
        else:
            for key, value in self.obs.items():
                self._device_obs[key].copy_(value, non_blocking=True)
            self._device_noise.copy_(self.noise, non_blocking=True)
            self._graph.replay()
            self.actions.copy_(self._device_out[0], non_blocking=True)
            self.logp_ent_val.copy_(self._device_out[1], non_blocking=True)
            torch.cuda.current_stream(self.device).synchronize()
        return self.actions.numpy(), self.logp_ent_val.numpy()

    @torch.no_grad()
    def value(self, obs: dict) -> np.ndarray:
        """Value estimates ``[k]`` of ``obs`` (per-key ``[k, *shape]``; eager, any batch size)."""
        return self.network.get_value({k: v for k, v in obs.items() if k != MASK_KEY}).cpu().numpy()


class Collector:
    """Asynchronous rollouts from a pool into an env-major ``RolloutBuffer``.

    ``pool`` is a native ``EnvPool`` (or anything with its ``num_envs`` / ``make_buffers`` / ``recv`` / ``send`` /
    ``poll_events`` interface).  The first ``recv`` returns every env's reset observation; between ``collect``
    calls every env waits for its next action, so episodes continue across rollouts.  ``seed`` seeds the action
    sampling noise (default: drawn from the global torch RNG).
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, pool, memory_length: int, gamma: float, lam: float, seed: Optional[int] = None):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.pool = pool
        self.n_envs = pool.num_envs
        self.gamma = gamma
        self.rows = pool.make_buffers()
        self.buffer = RolloutBuffer(memory_length, self.n_envs, self.rows['obs'], gamma, lam)
        count = self.pool.recv(self.rows, min_ready=self.n_envs)
        if count != self.n_envs or not np.array_equal(self.rows['env_ids'], np.arange(self.n_envs)):
            raise RuntimeError(f"first recv returned {count} of {self.n_envs} envs: {self.rows['env_ids']}")
        self.dones = np.zeros(self.n_envs, dtype=np.float32)
        if seed is None:
            seed = int(torch.randint(0, 2 ** 62, ()))
        self.generator = torch.Generator().manual_seed(seed)
        self.policy: Optional[RolloutPolicy] = None

    def _policy(self, network: nn.Module, device) -> RolloutPolicy:
        if self.policy is None or self.policy.device != torch.device(device):
            current = None if self.policy is None else self.policy.obs
            self.policy = RolloutPolicy(network, current or self.rows['obs'], device)
        else:
            self.policy.load(network)
        return self.policy

    def collect(self, network: nn.Module, device) -> RolloutMetrics:
        """Fills the buffer with ``memory_length`` steps per env and computes returns and advantages."""
        # pylint: disable=too-many-locals, too-many-statements
        buffer, rows, n, length = self.buffer, self.rows, self.n_envs, self.buffer.memory_length
        policy = self._policy(network, device)
        staged = policy.obs
        noise = gumbel_noise((n, length, NOISE_SIZE), self.generator)
        rewards = np.zeros((n, length), dtype=np.float32)
        frames = np.zeros((n, length), dtype=np.int64)
        interrupts = np.zeros((n, length), dtype=np.int64)
        counts = np.zeros(n, dtype=np.int64)
        bootstraps, events = [], []

        everyone = np.arange(n)
        buffer.store_obs(everyone, counts, staged, self.dones)
        ready, outstanding = everyone, 0
        while True:
            if len(ready):
                steps = counts[ready]
                masks = staged[MASK_KEY][ready, :NUM_KINDS].numpy()
                if not masks.any(axis=1).all():
                    bad = int(ready[np.flatnonzero(~masks.any(axis=1))[0]])
                    raise ValueError(f"Empty kind mask for env {bad}: {staged[MASK_KEY][bad].tolist()}")
                policy.noise[ready] = noise[ready, steps]
                actions, logp_ent_val = policy()
                buffer.store_policy(ready, steps, actions[ready], logp_ent_val[ready])
                self.pool.send(ready, actions[ready])
                outstanding += len(ready)
            if not outstanding:
                break

            count = self.pool.recv(rows, min_ready=1)
            outstanding -= count
            ids = rows['env_ids'][:count].copy()
            steps = counts[ids]
            rewards[ids, steps] = rows['reward'][:count]
            frames[ids, steps] = rows['frames'][:count]
            interrupts[ids, steps] = rows['interrupt'][:count]
            bootstrap = np.flatnonzero(rows['bootstrap'][:count])
            if len(bootstrap):
                bootstraps.append((ids[bootstrap], steps[bootstrap],
                                   {key: value[bootstrap].copy() for key, value in rows['final_obs'].items()}))
            dones = (rows['terminated'][:count] | rows['truncated'][:count]).astype(np.float32)
            self.dones[ids] = dones
            counts[ids] += 1
            ids_tensor = torch.from_numpy(ids)
            for key, value in staged.items():
                value[ids_tensor] = torch.from_numpy(rows['obs'][key][:count])
            buffer.store_obs(ids, counts[ids], {key: value[ids_tensor] for key, value in staged.items()}, dones)
            events += [e for e in self.pool.poll_events() if e['type'] == 'episode']
            ready = ids[counts[ids] < length]

        if bootstraps:
            envs = np.concatenate([b[0] for b in bootstraps])
            steps = np.concatenate([b[1] for b in bootstraps])
            order = np.lexsort((steps, envs))                       # env-major: independent of arrival order
            envs, steps = envs[order], steps[order]
            final = {key: np.concatenate([b[2][key] for b in bootstraps])[order] for key in bootstraps[0][2]}
            values = policy.value(final)
            rewards[envs, steps] += self.gamma ** (frames[envs, steps].astype(np.float32) / FRAMES_PER_DISCOUNT) \
                * values
        buffer.store_results(rewards, frames)
        buffer.compute_returns(policy.value(staged))

        metrics = RolloutMetrics()
        metrics.add_steps(buffer.actions.reshape(-1, 3).numpy(), frames.reshape(-1), interrupts.reshape(-1))
        for event in sorted(events, key=lambda e: e['env']):     # stable: per env in step order
            metrics.add_episode(event['info'])
        return metrics


class PPO:
    """PPO with factored heads and frame-aware GAE."""

    def __init__(self, config: Optional[PPOConfig] = None, device='cpu', optimizer_state=None, demo_batch=None):
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._pending_optimizer_state = optimizer_state
        self.demo_batch = demo_batch
        self.total_steps = 0

    # ------------------------------------------------------------------ optimizer

    def setup_optimizer(self, network: nn.Module) -> torch.optim.Optimizer:
        """Creates the Adam optimizer for network (restoring checkpointed state if given)."""
        network.to(self.device)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(network.parameters(), lr=self.config.learning_rate,
                                              eps=self.config.epsilon)
            if self._pending_optimizer_state is not None:
                self.optimizer.load_state_dict(self._pending_optimizer_state)
                self._pending_optimizer_state = None
                for group in self.optimizer.param_groups:
                    group['lr'] = self.config.learning_rate
                for state in self.optimizer.state.values():
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor) and key != 'step':
                            state[key] = value.to(self.device)
        return self.optimizer

    # ------------------------------------------------------------------ training loop

    def train(self, network: nn.Module, pool, iterations: int, callback: Optional[Callable] = None,
              start_steps: int = 0, seed: Optional[int] = None) -> nn.Module:
        """Runs ``iterations`` rounds of (collect ``target_steps`` env steps across ``pool``'s envs, optimize).

        Args:
            pool: a native ``EnvPool`` over a ``TrainTask`` (the caller owns and closes it).
            callback: called with an ``IterationResult`` after each iteration; return False to stop.
            seed: seeds the training RNGs (action sampling, minibatch order); the pool has its own seed.
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
        cfg = self.config
        steps_per_env = max(1, cfg.target_steps // pool.num_envs)
        self.setup_optimizer(network)
        self.total_steps = start_steps
        if seed is not None:
            torch.manual_seed(seed)
            seed = torch.initial_seed()
            random.seed(seed)
            np.random.seed(seed % (1 << 32))

        collector = Collector(pool, steps_per_env, cfg.gamma, cfg.lam)
        buffer = collector.buffer
        start = time.time()
        steps_this_run = 0
        for iteration in range(1, iterations + 1):
            collect_start = time.time()
            rollout = collector.collect(network, self.device)
            collect_time = time.time() - collect_start

            batch_steps = buffer.memory_length * buffer.n_envs
            steps_this_run += batch_steps
            self.total_steps += batch_steps

            optimize_start = time.time()
            stats = self.optimize(network, buffer)
            elapsed = time.time() - start
            stats['charts/SPS'] = steps_this_run / elapsed if elapsed > 0 else 0.0
            stats['charts/rollout_SPS'] = batch_steps / collect_time if collect_time > 0 else 0.0
            stats['charts/rollout_seconds'] = collect_time
            stats['charts/optimize_seconds'] = time.time() - optimize_start

            if callback is not None:
                if callback(IterationResult(iteration, self.total_steps, rollout, stats)) is False:
                    break
        return network

    # ------------------------------------------------------------------ optimization

    def optimize(self, network: nn.Module, buffer: RolloutBuffer) -> dict:
        """One PPO update over the buffer.  Returns loss/KL/entropy stats."""
        # pylint: disable=too-many-locals, too-many-statements, too-many-branches
        cfg = self.config
        optimizer = self.setup_optimizer(network)
        network.to(self.device)
        network.train()

        batch = buffer.flatten(self.device)
        b_obs, b_masks, b_actions = batch['obs'], batch['masks'], batch['actions']
        b_logprobs, b_values = batch['logprobs'], batch['values']
        b_advantages, b_returns = batch['advantages'], batch['returns']
        batch_size = b_actions.shape[0]
        minibatch_size = max(1, batch_size // cfg.minibatches)

        demo = None
        if self.demo_batch is not None and cfg.demo_bc_coeff > 0:
            demo_obs, demo_masks, demo_targets = self.demo_batch
            demo = ({k: v.to(self.device) for k, v in demo_obs.items()}, demo_masks.to(self.device),
                    demo_targets.to(self.device))
        demo_bc_loss = torch.tensor(0.0)

        rollback_threshold = None
        saved_state = saved_optimizer = None
        if cfg.kl_rollback and cfg.target_kl:
            rollback_threshold = cfg.target_kl * cfg.kl_rollback
            saved_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
            saved_optimizer = copy.deepcopy(optimizer.state_dict())

        clipfracs = []
        approx_kl = old_approx_kl = torch.tensor(0.0)
        pg_loss = v_loss = entropy_loss = torch.tensor(0.0)
        kl_exceeded = False
        epochs_run = 0
        updates = 0
        for _ in range(cfg.num_epochs):
            if kl_exceeded:
                break
            epochs_run += 1
            indices = torch.randperm(batch_size, device=self.device)
            for begin in range(0, batch_size, minibatch_size):
                mb = indices[begin:begin + minibatch_size]
                mb_obs = {k: v[mb] for k, v in b_obs.items()}
                out = network.evaluate(mb_obs, b_masks[mb], b_actions[mb])

                logratio = out.log_prob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coeff).float().mean().item())

                mb_adv = b_advantages[mb]
                if cfg.norm_advantages and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(-mb_adv * ratio,
                                    -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coeff, 1 + cfg.clip_coeff)).mean()

                new_value = out.value.view(-1)
                if cfg.clip_val_loss:
                    clipped = b_values[mb] + torch.clamp(new_value - b_values[mb], -cfg.clip_coeff, cfg.clip_coeff)
                    v_loss = 0.5 * torch.max((new_value - b_returns[mb]) ** 2, (clipped - b_returns[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * (new_value - b_returns[mb]).pow(2).mean()

                entropy_loss = out.entropy.mean()
                loss = pg_loss - cfg.ent_coeff * entropy_loss + cfg.vf_coeff * v_loss
                if demo is not None:
                    demo_out = network.evaluate(demo[0], demo[1], demo[2])
                    demo_bc_loss = -demo_out.log_prob.mean()
                    loss = loss + cfg.demo_bc_coeff * demo_bc_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), cfg.max_grad_norm)
                optimizer.step()
                updates += 1

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    kl_exceeded = True
                    break

        rolled_back = False
        if rollback_threshold is not None and approx_kl > rollback_threshold:
            network.load_state_dict(saved_state)
            optimizer.load_state_dict(saved_optimizer)
            rolled_back = True

        network.eval()
        with torch.no_grad():
            head_stats = network.head_entropies(b_obs, b_masks, b_actions)
            demo_accuracy = None
            if demo is not None:
                demo_accuracy = compute_demo_accuracy(network.get_action(demo[0], demo[1], deterministic=True),
                                                      demo[2])
        network.train()
        network.to('cpu')

        var_y = torch.var(b_returns)
        explained_var = float('nan') if var_y == 0 else float(1 - torch.var(b_returns - b_values) / var_y)
        stats = {
            'charts/learning_rate': optimizer.param_groups[0]['lr'],
            'losses/value_loss': v_loss.item(),
            'losses/policy_loss': pg_loss.item(),
            'losses/entropy': entropy_loss.item(),
            'losses/old_approx_kl': old_approx_kl.item(),
            'losses/approx_kl': approx_kl.item(),
            'losses/clipfrac': float(sum(clipfracs) / len(clipfracs)) if clipfracs else 0.0,
            'losses/explained_variance': explained_var,
            'losses/kl_rollback': 1.0 if rolled_back else 0.0,
            'losses/kl_early_stop': 1.0 if kl_exceeded else 0.0,
            'losses/epochs': float(epochs_run),
            'losses/updates': float(updates),
            'charts/advantage_mean': float(b_advantages.mean()),
            'charts/return_mean': float(b_returns.mean()),
            **head_stats,
        }
        if demo is not None:
            stats['losses/demo_bc_loss'] = demo_bc_loss.item()
            stats['charts/demo_bc_accuracy'] = demo_accuracy
            stats['charts/demo_bc_coeff'] = cfg.demo_bc_coeff
        return stats
