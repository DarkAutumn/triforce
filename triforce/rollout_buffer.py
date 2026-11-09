"""Rollout storage for PPO (port of v1 ``ml_ppo_rollout_buffer``), with dict observations, factored actions,
flat action masks, and frame-aware GAE.  ``ppo.Collector`` fills it from a native ``EnvPool``'s ``rollout``: the
pool's workers act with the Rust copy of the network and write every env's rows in one call.

Layout (``n`` envs, ``T`` steps per env; index ``T`` holds the observation to resume from):

- ``observation[key]``: ``[n, T+1, *shape]`` for every observation key except ``action_mask``
  (integer keys stay int64, the rest float32).
- ``masks``: bool ``[n, T+1, MASK_SIZE]`` (the observation's ``action_mask``).
- ``dones``: ``[n, T+1]``; ``dones[:, t]`` is 1 when the observation at ``t`` starts a new episode.
- ``actions``: int64 ``[n, T, 3]`` (kind, direction, duration).
- ``logp_ent_val``: ``[n, T, 3]``.
- ``rewards``, ``frames``, ``returns``, ``advantages``: ``[n, T]``.

Frame-aware GAE (design §5.4).  Steps take different numbers of frames, so step ``t`` that lasted ``f_t``
frames uses ``gamma_t = gamma ** (f_t / FRAMES_PER_DISCOUNT)`` and ``lambda_t = lambda ** (f_t /
FRAMES_PER_DISCOUNT)`` with ``FRAMES_PER_DISCOUNT = 8``.  Lambda is scaled the same way so the bias/variance
trade-off spans a fixed amount of game time rather than a fixed number of decisions (a 32-frame MOVE
should not keep 4× more of the trace than four 8-frame ones). Legacy time limits still mask bootstrap.
Opted-in ``bootstrap`` transitions add the discounted final-observation value to the learning reward before
reset (the collector does that); done flags still stop the trace from crossing into the next episode.
"""

import numpy as np
import torch
from gymnasium import spaces

from .actions import MASK_SIZE
from .model import MASK_KEY

FRAMES_PER_DISCOUNT = 8.0


def _storage(shape, integer: bool, n_envs, length):
    return torch.zeros(n_envs, length, *shape, dtype=torch.int64 if integer else torch.float32)


def _space_layout(space) -> tuple[tuple, bool]:
    if isinstance(space, spaces.Discrete):
        return (1,), True
    integer = np.issubdtype(space.dtype, np.integer) or isinstance(space, spaces.MultiDiscrete)
    return tuple(space.shape), bool(integer)


def frame_aware_gae(rewards, values, frames, dones_next, last_value, gamma, lam,
                    frames_per_discount=FRAMES_PER_DISCOUNT):
    """Returns (returns, advantages) for trajectory segments ``[..., T]`` (one per leading index, e.g. per env).

    Args:
        rewards, values, frames: ``[..., T]`` tensors for steps 0..T-1.
        dones_next: ``[..., T]`` — 1 when step t ended the episode (terminated or truncated).
        last_value: ``[...]`` value of the observation after step T-1 (ignored if that step ended an episode).
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    rewards = torch.as_tensor(rewards, dtype=torch.float64)
    values64 = torch.as_tensor(values, dtype=torch.float64)
    exponent = torch.as_tensor(frames, dtype=torch.float64) / frames_per_discount
    step_gamma = torch.pow(torch.tensor(gamma, dtype=torch.float64), exponent)
    step_lambda = torch.pow(torch.tensor(lam, dtype=torch.float64), exponent)
    nonterminal = 1.0 - torch.as_tensor(dones_next, dtype=torch.float64)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(rewards.shape[:-1], dtype=torch.float64)
    next_value = torch.as_tensor(last_value, dtype=torch.float64).reshape(rewards.shape[:-1])
    for t in reversed(range(rewards.shape[-1])):
        delta = rewards[..., t] + step_gamma[..., t] * next_value * nonterminal[..., t] - values64[..., t]
        last_gae = delta + step_gamma[..., t] * step_lambda[..., t] * nonterminal[..., t] * last_gae
        advantages[..., t] = last_gae
        next_value = values64[..., t]
    advantages = advantages.to(torch.float32)
    return advantages + torch.as_tensor(values).to(torch.float32), advantages


class RolloutBuffer:
    """Env-major rollout storage for ``n_envs`` envs × ``memory_length`` steps."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, memory_length: int, n_envs: int, observation_space, gamma: float, lam: float):
        """``observation_space``: a ``spaces.Dict``, or a dict of per-key ``[rows, *shape]`` arrays (the pool's
        ``obs`` buffers) whose shapes and dtypes define the layout."""
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.memory_length = memory_length
        self.n_envs = n_envs
        self.gamma = gamma
        self.lam = lam
        if isinstance(observation_space, spaces.Dict):
            layout = {key: _space_layout(space) for key, space in observation_space.spaces.items()}
        else:
            layout = {key: (tuple(array.shape[1:]), bool(np.issubdtype(array.dtype, np.integer)))
                      for key, array in observation_space.items()}
        self.observation = {key: _storage(shape, integer, n_envs, memory_length + 1)
                            for key, (shape, integer) in layout.items() if key != MASK_KEY}
        self.masks = torch.zeros(n_envs, memory_length + 1, MASK_SIZE, dtype=torch.bool)
        self.dones = torch.zeros(n_envs, memory_length + 1, dtype=torch.float32)
        self.actions = torch.zeros(n_envs, memory_length, 3, dtype=torch.int64)
        self.logp_ent_val = torch.zeros(n_envs, memory_length, 3, dtype=torch.float32)
        self.rewards = torch.zeros(n_envs, memory_length, dtype=torch.float32)
        self.frames = torch.zeros(n_envs, memory_length, dtype=torch.float32)
        self.returns = torch.zeros(n_envs, memory_length, dtype=torch.float32)
        self.advantages = torch.zeros(n_envs, memory_length, dtype=torch.float32)
        self.bootstrap_value = torch.zeros(n_envs, memory_length, dtype=torch.float32)
        self.interrupts = torch.zeros(n_envs, memory_length, dtype=torch.int64)

    def rollout_buffers(self) -> dict:
        """numpy views of this buffer's memory in the layout ``EnvPool.rollout`` writes: ``obs`` (every key, the
        mask included), ``starts`` (``dones``), ``actions``, ``logp_ent_val``, ``rewards`` (raw step rewards),
        ``frames``, ``bootstrap_value``, ``interrupts``."""
        obs = {key: value.numpy() for key, value in self.observation.items()}
        obs[MASK_KEY] = self.masks.numpy()
        return {'obs': obs, 'starts': self.dones.numpy(), 'actions': self.actions.numpy(),
                'logp_ent_val': self.logp_ent_val.numpy(), 'rewards': self.rewards.numpy(),
                'frames': self.frames.numpy(), 'bootstrap_value': self.bootstrap_value.numpy(),
                'interrupts': self.interrupts.numpy()}

    @torch.no_grad()
    def recompute(self, network, device, chunk: int = 4096):
        """Replaces the actors' log-prob, entropy and value with ``network``'s (torch) over every stored step, and
        returns the values ``[n]`` of the observations at index ``T``: the update then compares torch with torch,
        never with the Rust actor's numerics."""
        network.eval()
        length = self.memory_length
        obs = {key: value[:, :length].reshape(-1, *value.shape[2:]) for key, value in self.observation.items()}
        masks = self.masks[:, :length].reshape(-1, MASK_SIZE)
        actions = self.actions.reshape(-1, 3)
        out = torch.empty(actions.shape[0], 3)
        for begin in range(0, actions.shape[0], chunk):
            rows = slice(begin, begin + chunk)
            result = network.evaluate({k: v[rows].to(device) for k, v in obs.items()}, masks[rows].to(device),
                                      actions[rows].to(device))
            out[rows] = torch.stack([result.log_prob, result.entropy, result.value], dim=-1).cpu()
        self.logp_ent_val.copy_(out.view(self.n_envs, length, 3))
        last = {key: value[:, length].to(device) for key, value in self.observation.items()}
        return network.get_value(last).cpu()

    def add_bootstrap(self):
        """Adds each bootstrapped truncation's discounted final-observation value to its step's reward (design
        §5.4: the trace stops at the done flag, the value of where the episode was cut off doesn't)."""
        discount = torch.pow(torch.tensor(self.gamma), self.frames / FRAMES_PER_DISCOUNT)
        self.rewards.add_(discount * self.bootstrap_value)

    def compute_returns(self, last_values):
        """Frame-aware GAE per env; ``last_values[i]`` is the value of env ``i``'s observation at index ``T``."""
        last_values = torch.as_tensor(last_values).cpu()
        returns, advantages = frame_aware_gae(self.rewards, self.logp_ent_val[:, :, 2], self.frames,
                                              self.dones[:, 1:], last_values, self.gamma, self.lam)
        self.returns.copy_(returns)
        self.advantages.copy_(advantages)

    def flatten(self, device) -> dict:
        """Batch tensors [n·T, ...] for optimization."""
        length = self.memory_length
        batch = {'obs': {key: value[:, :length].reshape(-1, *value.shape[2:]).to(device)
                         for key, value in self.observation.items()}}
        batch['masks'] = self.masks[:, :length].reshape(-1, MASK_SIZE).to(device)
        batch['actions'] = self.actions.reshape(-1, 3).to(device)
        batch['logprobs'] = self.logp_ent_val[:, :, 0].reshape(-1).to(device)
        batch['values'] = self.logp_ent_val[:, :, 2].reshape(-1).to(device)
        batch['advantages'] = self.advantages.reshape(-1).to(device)
        batch['returns'] = self.returns.reshape(-1).to(device)
        return batch
