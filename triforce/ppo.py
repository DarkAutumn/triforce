"""PPO trainer (port of v1 ``ml_ppo``) for one specialist.

Kept from v1: KL early stop, KL rollback (weights *and* optimizer state restored when KL exceeds
``kl_rollback × target_kl``), checkpointable optimizer state, per-minibatch advantage normalization, legacy
time-limit termination, optional demo-regularization (behavior-cloning loss on a fixed demo batch), optional
self-imitation learning (``SilReplay``; Oh et al., 2018), optional Gumbel expert iteration (``PPOConfig.search``).

Changed for v2: dict observations, factored (kind, direction, duration) actions, flat masks, frame-aware GAE
(see ``rollout_buffer``), and ``train`` counts PPO *iterations* (one rollout + one optimization each).

Rollouts come from a native ``EnvPool`` (``TrainTask``) in one call per rollout (``Collector``): the pool's worker
threads step their envs ``T`` decisions each, acting with the Rust copy of the network (``ZeldaPolicy.to_native``)
and keyed Gumbel noise (per episode job and decision), and write the env-major buffer (docs/rust-actor.md). Envs
auto-reset and episodes continue across rollouts. ``train`` overlaps each PPO update with the next rollout (Lee,
Nov 9): that rollout acts with the weights from before the update, one update behind. The learner recomputes every
step's log-prob, entropy and value with its current torch network (``RolloutBuffer.recompute``), so the update never
mixes the Rust actor's numerics in and its KL budget isn't spent on the lag. Opted-in no-progress truncations add
``gamma ** (frames / 8) * V(final_obs)`` (the actor's value) to that step's reward without crossing the reset.

Expert iteration (better-play §2, docs/rust-actor.md "Search"): with ``search`` the pool runs a Gumbel search on a
share of the decisions, executes its choice and reports its improved policy π' over the joint actions.  Those rows
train the policy by distillation, ``search_coeff · CE(π', π_θ)``, instead of the PPO ratio term (their action is no
sample of π); value loss, entropy bonus and GAE cover every row as before.
"""

import copy
import random
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn

from .demo import compute_demo_accuracy
from .metrics import RolloutMetrics
from .rollout_buffer import RolloutBuffer


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
    # Share of a train() call's iterations, at its end, over which ent_coeff falls linearly to 0 (better-play §6b:
    # less sampling noise in the final policy).  0 = constant.
    ent_anneal: float = 0.0
    # Iterations over which the pool's shaping scale (``EnvPool.shaping``, every non-terminal reward/penalty) falls
    # linearly from its starting value to 0, then stays 0 (better-play §1a outcome fine-tuning).  0 = untouched.
    shaping_anneal: int = 0
    vf_coeff: float = 0.5
    max_grad_norm: float = 0.5
    epsilon: float = 1e-5
    norm_advantages: bool = True
    clip_val_loss: bool = False
    target_kl: Optional[float] = 0.02
    kl_rollback: Optional[float] = 4.0
    demo_bc_coeff: float = 0.0
    # Self-imitation learning (better-play §3): weight of the SIL loss on replayed rows whose λ=1 return beat the
    # critic (0 = off), its value-term weight β, and how many such rows the replay keeps (FIFO).
    sil_coeff: float = 0.0
    sil_value_coeff: float = 0.01
    sil_capacity: int = 32768
    # Gumbel expert iteration (better-play §2): ``EnvPool.rollout``'s ``search`` settings (None = off) and the weight
    # of the distillation loss CE(π', π_θ) on searched rows.
    search: Optional[dict] = None
    search_coeff: float = 1.0

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
    """Rollouts from a native pool into env-major ``RolloutBuffer``s, one ``EnvPool.rollout`` call each.

    ``pool`` is a native ``EnvPool`` over a ``TrainTask`` (or anything with its ``num_envs`` / ``obs_spec`` /
    ``rollout`` / ``poll_events`` interface).  Episodes continue across rollouts; the noise that samples each
    action is keyed by the episode's job and decision step (the pool's seed decides the job keys).

    ``start(network)`` exports the network's weights and rolls out on a background thread into the spare of two
    buffers (the GIL is released while the pool steps); ``finish(network, device)`` waits for it, then computes
    values, bootstrap rewards, returns and advantages and makes it ``buffer``. The other buffer stays untouched,
    so an update can read it meanwhile.  ``collect`` is both in a row.  ``search``: the pool's search settings
    (``EnvPool.rollout(search=...)``); the buffers then keep ``search_mask`` and ``search_policy``.
    """

    def __init__(self, pool, memory_length: int, gamma: float, lam: float, search: Optional[dict] = None):
        self.pool = pool
        self.n_envs = pool.num_envs
        self.search = search
        self._buffers = [RolloutBuffer(memory_length, self.n_envs, pool.obs_spec, gamma, lam,
                                       search=search is not None) for _ in range(2)]
        self._views = [buffer.rollout_buffers() for buffer in self._buffers]
        self._next = 0
        self.buffer = self._buffers[1]
        self._thread: Optional[threading.Thread] = None
        self._result: dict = {}
        self.rollout_seconds = 0.0

    def start(self, network: nn.Module):
        """Starts the next rollout with ``network``'s current weights."""
        if self._thread is not None:
            raise RuntimeError("a rollout is already running")
        policy, index = network.to_native(), self._next
        self._result = {}

        def run():
            began = time.time()
            search = {} if self.search is None else {'search': self.search}
            try:
                self.pool.rollout([policy], self._buffers[index].memory_length, self._views[index], **search)
            except BaseException as err:  # pylint: disable=broad-exception-caught
                self._result['error'] = err
            self._result['seconds'] = time.time() - began

        self._thread = threading.Thread(target=run, name='rollout', daemon=True)
        self._thread.start()

    def wait(self):
        """Waits for the running rollout (if any); re-raises its error."""
        if self._thread is None:
            return
        self._thread.join()
        self._thread = None
        if 'error' in self._result:
            raise self._result['error']

    def finish(self, network: nn.Module, device) -> RolloutMetrics:
        """Waits for the rollout ``start`` began and readies its buffer (now ``buffer``) for an update."""
        self.wait()
        self.rollout_seconds = self._result['seconds']
        buffer = self.buffer = self._buffers[self._next]
        self._next = 1 - self._next
        events = [e for e in self.pool.poll_events() if e['type'] == 'episode']
        last_values = buffer.recompute(network, device)
        buffer.add_bootstrap()
        buffer.compute_returns(last_values)

        metrics = RolloutMetrics()
        metrics.add_steps(buffer.actions.reshape(-1, 3).numpy(), buffer.frames.reshape(-1).numpy(),
                          buffer.interrupts.reshape(-1).numpy())
        for event in sorted(events, key=lambda e: e['env']):     # stable: per env in step order
            metrics.add_episode(event['info'])
        return metrics

    def collect(self, network: nn.Module, device) -> RolloutMetrics:
        """One rollout, waited for: ``start`` then ``finish``."""
        self.start(network)
        return self.finish(network, device)


class SilReplay:
    """Self-imitation replay (Oh et al., 2018): the rollout rows whose Monte-Carlo return beat the critic.

    ``add`` takes a readied ``RolloutBuffer`` (values recomputed by the learner, rewards bootstrapped) and keeps
    the rows with λ = 1 return R > V(s) (``RolloutBuffer.monte_carlo_returns``): observation, mask, action and R,
    on CPU in a ring of ``capacity`` rows (the oldest are overwritten first).  ``sample`` draws rows uniformly
    without replacement.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self._next = 0
        self.obs: dict = {}
        self.masks = self.actions = self.returns = None

    def __len__(self) -> int:
        return self.size

    def add(self, buffer: RolloutBuffer) -> float:
        """Stores ``buffer``'s rows with R > V; returns their share of the rollout's rows."""
        length = buffer.memory_length
        returns, gaps = buffer.monte_carlo_returns()
        keep = (gaps > 0).reshape(-1)
        count = int(keep.sum())
        if count:
            rows = {key: value[:, :length].reshape(-1, *value.shape[2:])[keep]
                    for key, value in buffer.observation.items()}
            self._write(rows, buffer.masks[:, :length].reshape(-1, buffer.masks.shape[-1])[keep],
                        buffer.actions.reshape(-1, 3)[keep], returns.reshape(-1)[keep])
        return count / keep.numel()

    def _write(self, obs: dict, masks, actions, returns):
        if self.masks is None:
            self.obs = {key: value.new_empty(self.capacity, *value.shape[1:]) for key, value in obs.items()}
            self.masks = masks.new_empty(self.capacity, *masks.shape[1:])
            self.actions = actions.new_empty(self.capacity, 3)
            self.returns = returns.new_empty(self.capacity)
        count = returns.shape[0]
        skip = max(0, count - self.capacity)            # more new rows than capacity: only the newest fit
        positions = (self._next + torch.arange(skip, count)) % self.capacity
        for key, value in obs.items():
            self.obs[key][positions] = value[skip:]
        self.masks[positions] = masks[skip:]
        self.actions[positions] = actions[skip:]
        self.returns[positions] = returns[skip:]
        self._next = (self._next + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, count: int, device):
        """``(obs, masks, actions, returns)`` of ``min(count, len(self))`` uniformly drawn rows on ``device``."""
        pick = torch.randperm(self.size)[:count]
        return ({key: value[pick].to(device) for key, value in self.obs.items()}, self.masks[pick].to(device),
                self.actions[pick].to(device), self.returns[pick].to(device))


class PPO:
    """PPO with factored heads and frame-aware GAE."""

    def __init__(self, config: Optional[PPOConfig] = None, device='cpu', optimizer_state=None, demo_batch=None):
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._pending_optimizer_state = optimizer_state
        self.demo_batch = demo_batch
        self.total_steps = 0
        self.ent_coeff = self.config.ent_coeff     # this update's entropy bonus (ent_anneal lowers it)
        self.sil: Optional[SilReplay] = None       # created by the first optimize with sil_coeff > 0

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
        anneal = max(0, min(iterations, round(cfg.ent_anneal * iterations)))
        steps_per_env = max(1, cfg.target_steps // pool.num_envs)
        self.setup_optimizer(network)
        self.total_steps = start_steps
        if seed is not None:
            torch.manual_seed(seed)
            seed = torch.initial_seed()
            random.seed(seed)
            np.random.seed(seed % (1 << 32))

        # Iteration k (1-based) rolls out with shaping start * max(0, 1 - (k-1)/N); set before its collector.start.
        shaping_start = pool.shaping if cfg.shaping_anneal > 0 and hasattr(pool, 'set_shaping') else None

        def shaping_for(k: int) -> Optional[float]:
            if shaping_start is None:
                return None
            scale = shaping_start * max(0.0, 1.0 - (k - 1) / cfg.shaping_anneal)
            pool.set_shaping(scale)
            return scale

        collector = Collector(pool, steps_per_env, cfg.gamma, cfg.lam, cfg.search)
        start = time.time()
        steps_this_run = 0
        shaping = shaping_for(1)
        collector.start(network)
        try:
            for iteration in range(1, iterations + 1):
                wait_start = time.time()
                rollout = collector.finish(network, self.device)
                wait_time = time.time() - wait_start
                buffer = collector.buffer
                rollout_shaping = shaping
                # The next rollout acts with these weights while the update below changes them.
                if iteration < iterations:
                    shaping = shaping_for(iteration + 1)
                    collector.start(network)

                batch_steps = buffer.memory_length * buffer.n_envs
                steps_this_run += batch_steps
                self.total_steps += batch_steps

                optimize_start = time.time()
                # entropy bonus: constant, then down to 0 at the last iteration over the final `anneal` ones
                left = iterations - iteration
                self.ent_coeff = cfg.ent_coeff * (min(1.0, left / anneal) if anneal else 1.0)
                stats = self.optimize(network, buffer)
                elapsed = time.time() - start
                rollout_time = collector.rollout_seconds
                stats['charts/SPS'] = steps_this_run / elapsed if elapsed > 0 else 0.0
                stats['charts/rollout_SPS'] = batch_steps / rollout_time if rollout_time > 0 else 0.0
                stats['charts/rollout_seconds'] = rollout_time
                stats['charts/rollout_wait_seconds'] = wait_time
                stats['charts/optimize_seconds'] = time.time() - optimize_start
                if rollout_shaping is not None:
                    stats['charts/shaping_scale'] = rollout_shaping

                if callback is not None:
                    if callback(IterationResult(iteration, self.total_steps, rollout, stats)) is False:
                        break
        finally:
            collector.wait()    # a rollout started for an iteration that won't run is discarded
        return network

    # ------------------------------------------------------------------ optimization

    def optimize(self, network: nn.Module, buffer: RolloutBuffer) -> dict:
        """One PPO update over the buffer.  Returns loss/KL/entropy stats.

        Rows the search chose (``buffer.search_mask``) train the policy by distillation toward the search's π'
        instead of the clipped ratio term (and stay out of its KL/clipfrac stats)."""
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
        b_searched, b_search_policy = batch.get('search_mask'), batch.get('search_policy')    # None without search

        demo = None
        if self.demo_batch is not None and cfg.demo_bc_coeff > 0:
            demo_obs, demo_masks, demo_targets = self.demo_batch
            demo = ({k: v.to(self.device) for k, v in demo_obs.items()}, demo_masks.to(self.device),
                    demo_targets.to(self.device))
        demo_bc_loss = torch.tensor(0.0)

        sil = None
        if cfg.sil_coeff > 0:
            if self.sil is None:
                self.sil = SilReplay(cfg.sil_capacity)
            sil = self.sil
        sil_positive_share = sil.add(buffer) if sil is not None else 0.0
        sil_policy_loss = sil_value_loss = torch.tensor(0.0)

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

                # The PPO term covers the rows the policy sampled; searched rows executed the search's choice.
                pg_rows, searched = slice(None), None
                if b_searched is not None:
                    searched = mb[b_searched[mb]]
                    pg_rows = ~b_searched[mb]
                logratio = out.log_prob[pg_rows] - b_logprobs[mb][pg_rows]
                ratio = logratio.exp()
                if logratio.numel():
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coeff).float().mean().item())

                    mb_adv = b_advantages[mb][pg_rows]
                    if cfg.norm_advantages and mb_adv.numel() > 1:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    pg_loss = torch.max(-mb_adv * ratio,
                                        -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coeff, 1 + cfg.clip_coeff)).mean()
                else:
                    pg_loss = torch.zeros((), device=self.device)

                new_value = out.value.view(-1)
                if cfg.clip_val_loss:
                    clipped = b_values[mb] + torch.clamp(new_value - b_values[mb], -cfg.clip_coeff, cfg.clip_coeff)
                    v_loss = 0.5 * torch.max((new_value - b_returns[mb]) ** 2, (clipped - b_returns[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * (new_value - b_returns[mb]).pow(2).mean()

                entropy_loss = out.entropy.mean()
                loss = pg_loss - self.ent_coeff * entropy_loss + cfg.vf_coeff * v_loss
                if searched is not None and searched.numel():
                    joint = network.joint_log_probs({k: v[searched] for k, v in b_obs.items()}, b_masks[searched])
                    search_ce = search_cross_entropy(b_search_policy[searched], joint)
                    loss = loss + cfg.search_coeff * search_ce
                if demo is not None:
                    rows = demo[2].shape[0]
                    if rows > minibatch_size:   # a large batch (best-of-N runs): a minibatch-sized sample each time
                        pick = torch.randperm(rows, device=self.device)[:minibatch_size]
                        demo_out = network.evaluate({k: v[pick] for k, v in demo[0].items()}, demo[1][pick],
                                                    demo[2][pick])
                    else:
                        demo_out = network.evaluate(demo[0], demo[1], demo[2])
                    demo_bc_loss = -demo_out.log_prob.mean()
                    loss = loss + cfg.demo_bc_coeff * demo_bc_loss
                if sil is not None and len(sil) > 0:
                    sil_obs, sil_masks, sil_actions, sil_returns = sil.sample(minibatch_size, self.device)
                    sil_out = network.evaluate(sil_obs, sil_masks, sil_actions)
                    sil_gap = torch.clamp(sil_returns - sil_out.value.view(-1), min=0.0)
                    sil_policy_loss = -(sil_out.log_prob * sil_gap.detach()).mean()
                    sil_value_loss = 0.5 * sil_gap.pow(2).mean()
                    loss = loss + cfg.sil_coeff * (sil_policy_loss + cfg.sil_value_coeff * sil_value_loss)

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
            search_stats = {}
            if b_searched is not None:
                # After the update, over every searched row: CE(π', π_θ) and how often their argmaxes agree.
                searched = b_searched.nonzero().squeeze(-1)
                search_stats['charts/search_rows'] = searched.numel() / batch_size
                if searched.numel():
                    joint = network.joint_log_probs({k: v[searched] for k, v in b_obs.items()}, b_masks[searched])
                    target = b_search_policy[searched]
                    search_stats['losses/search_ce'] = search_cross_entropy(target, joint).item()
                    search_stats['charts/search_agreement'] = float(
                        (target.argmax(dim=-1) == joint.argmax(dim=-1)).float().mean())
        network.train()
        network.to('cpu')

        var_y = torch.var(b_returns)
        explained_var = float('nan') if var_y == 0 else float(1 - torch.var(b_returns - b_values) / var_y)
        stats = {
            'charts/learning_rate': optimizer.param_groups[0]['lr'],
            'charts/ent_coeff': self.ent_coeff,
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
        if sil is not None:
            stats['losses/sil_policy_loss'] = sil_policy_loss.item()
            stats['losses/sil_value_loss'] = sil_value_loss.item()
            stats['charts/sil_rows'] = float(len(sil))
            stats['charts/sil_positive_share'] = sil_positive_share
        stats.update(search_stats)
        return stats


def search_cross_entropy(target: torch.Tensor, joint_log_probs: torch.Tensor) -> torch.Tensor:
    """Mean over rows of CE(π', π_θ) = −Σ_j π'_j log π_θ(j): ``target`` π' ``[B, J]`` (0 off the legal actions),
    ``joint_log_probs`` ``[B, J]`` (``ZeldaPolicy.joint_log_probs``, −inf where illegal; those j are skipped)."""
    return -(target * torch.where(target > 0, joint_log_probs, 0.0)).sum(dim=-1).mean()
