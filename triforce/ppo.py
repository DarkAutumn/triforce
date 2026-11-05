"""PPO trainer (port of v1 ``ml_ppo``) for one specialist.

Kept from v1: KL early stop, KL rollback (weights *and* optimizer state restored when KL exceeds
``kl_rollback × target_kl``), checkpointable optimizer state, per-minibatch advantage normalization, legacy
time-limit termination, optional demo-regularization (behavior-cloning loss on a fixed demo batch).

Changed for v2: dict observations, factored (kind, direction, duration) actions, flat masks, frame-aware GAE
(see ``rollout_buffer``), and ``train`` counts PPO *iterations* (one rollout + one optimization each).

Rollouts come from a native ``EnvPool`` (``TrainTask``) in lockstep (``Collector``): ``recv`` every env's row,
store it, run one batched ``network.evaluate`` on the PPO device, ``send`` the actions.  Envs auto-reset: a
terminal row's ``obs`` already starts the next episode and ``final_obs`` holds the terminal observation.
Opted-in no-progress truncations (the ``bootstrap`` column) add ``gamma ** (frames / 8) * V(final_obs)`` to that
step's reward without crossing the episode reset.
"""

import copy
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn

from .demo import compute_demo_accuracy
from .metrics import RolloutMetrics
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


class Collector:
    """Lockstep rollouts from a pool into an env-major ``RolloutBuffer``.

    ``pool`` is a native ``EnvPool`` (or anything with its ``num_envs`` / ``make_buffers`` / ``recv`` / ``send`` /
    ``poll_events`` interface).  The first ``recv`` returns every env's reset observation; between ``collect``
    calls every env waits for its next action, so episodes continue across rollouts.
    """

    def __init__(self, pool, memory_length: int, gamma: float, lam: float):
        self.pool = pool
        self.n_envs = pool.num_envs
        self.gamma = gamma
        self.rows = pool.make_buffers()
        self.buffer = RolloutBuffer(memory_length, self.n_envs, self.rows['obs'], gamma, lam)
        self._recv()
        self.dones = np.zeros(self.n_envs, dtype=np.float32)

    def _recv(self):
        count = self.pool.recv(self.rows, min_ready=self.n_envs)
        if count != self.n_envs or not np.array_equal(self.rows['env_ids'], np.arange(self.n_envs)):
            raise RuntimeError(f"lockstep recv returned {count} of {self.n_envs} envs: {self.rows['env_ids']}")

    def collect(self, network: nn.Module, device) -> RolloutMetrics:
        """Fills the buffer with ``memory_length`` steps per env and computes returns and advantages."""
        buffer, rows = self.buffer, self.rows
        metrics = RolloutMetrics()
        was_training = network.training
        network.to(device)
        network.eval()
        try:
            with torch.no_grad():
                for t in range(buffer.memory_length):
                    buffer.store_obs(t, rows['obs'], self.dones)
                    out = network.evaluate(buffer.obs_at(t), buffer.masks[:, t])
                    actions = out.actions.cpu()
                    buffer.store_policy(t, actions, out.log_prob, out.entropy, out.value)
                    self.pool.send(rows['env_ids'], actions.numpy())
                    self._recv()

                    rewards = rows['reward'].astype(np.float32)
                    frames = rows['frames'].astype(np.float32)
                    bootstrap = rows['bootstrap'].astype(bool)
                    if bootstrap.any():
                        final = {key: value[bootstrap] for key, value in rows['final_obs'].items()}
                        values = network.get_value(final).cpu().numpy()
                        rewards[bootstrap] += self.gamma ** (frames[bootstrap] / FRAMES_PER_DISCOUNT) * values
                    buffer.store_result(t, rewards, frames)
                    self.dones = (rows['terminated'] | rows['truncated']).astype(np.float32)

                    metrics.add_steps(actions.numpy(), rows['frames'], rows['interrupt'])
                    events = sorted((e for e in self.pool.poll_events() if e['type'] == 'episode'),
                                    key=lambda e: e['env'])
                    for event in events:
                        metrics.add_episode(event['info'])

                buffer.store_obs(buffer.memory_length, rows['obs'], self.dones)
                last_values = network.get_value(buffer.obs_at(buffer.memory_length))
                buffer.compute_returns(last_values)
        finally:
            network.train(was_training)
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
