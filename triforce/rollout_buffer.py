"""Rollout storage for PPO (port of v1 ``ml_ppo_rollout_buffer``), with dict observations, factored actions,
flat action masks, and frame-aware GAE.

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
Opted-in ``bootstrap_truncation`` transitions add the discounted final-observation value to the learning
reward before reset; done flags still stop the trace from crossing into the next episode.
"""

import numpy as np
import torch
from gymnasium import spaces

from .actions import MASK_SIZE
from .model import MASK_KEY

FRAMES_PER_DISCOUNT = 8.0


def _storage(space, n_envs, length):
    if isinstance(space, spaces.Discrete):
        return torch.zeros(n_envs, length, 1, dtype=torch.int64)
    integer = np.issubdtype(space.dtype, np.integer) or isinstance(space, spaces.MultiDiscrete)
    return torch.zeros(n_envs, length, *space.shape, dtype=torch.int64 if integer else torch.float32)


def frame_aware_gae(rewards, values, frames, dones_next, last_value, gamma, lam,
                    frames_per_discount=FRAMES_PER_DISCOUNT):
    """Returns (returns, advantages) for one env's trajectory segment.

    Args:
        rewards, values, frames: [T] tensors for steps 0..T-1.
        dones_next: [T] — 1 when step t ended the episode (terminated or truncated).
        last_value: value of the observation after step T-1 (ignored if that step ended an episode).
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    length = rewards.shape[0]
    exponent = frames.to(torch.float64) / frames_per_discount
    step_gamma = torch.pow(torch.tensor(gamma, dtype=torch.float64), exponent)
    step_lambda = torch.pow(torch.tensor(lam, dtype=torch.float64), exponent)
    advantages = torch.zeros(length, dtype=torch.float64)
    last_gae = 0.0
    next_value = float(last_value)
    for t in reversed(range(length)):
        nonterminal = 1.0 - float(dones_next[t])
        delta = float(rewards[t]) + float(step_gamma[t]) * next_value * nonterminal - float(values[t])
        last_gae = delta + float(step_gamma[t]) * float(step_lambda[t]) * nonterminal * last_gae
        advantages[t] = last_gae
        next_value = float(values[t])
    advantages = advantages.to(torch.float32)
    return advantages + values.to(torch.float32), advantages


class RolloutBuffer:
    """Per-env rollout storage that can live in shared memory and be filled by subprocess workers."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, memory_length: int, n_envs: int, observation_space: spaces.Dict, gamma: float, lam: float):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.memory_length = memory_length
        self.n_envs = n_envs
        self.gamma = gamma
        self.lam = lam
        self.observation = {key: _storage(space, n_envs, memory_length + 1)
                            for key, space in observation_space.spaces.items() if key != MASK_KEY}
        self.masks = torch.zeros(n_envs, memory_length + 1, MASK_SIZE, dtype=torch.bool)
        self.dones = torch.zeros(n_envs, memory_length + 1, dtype=torch.float32)
        self.actions = torch.zeros(n_envs, memory_length, 3, dtype=torch.int64)
        self.logp_ent_val = torch.zeros(n_envs, memory_length, 3, dtype=torch.float32)
        self.rewards = torch.zeros(n_envs, memory_length, dtype=torch.float32)
        self.frames = torch.zeros(n_envs, memory_length, dtype=torch.float32)
        self.returns = torch.zeros(n_envs, memory_length, dtype=torch.float32)
        self.advantages = torch.zeros(n_envs, memory_length, dtype=torch.float32)
        # Shared so a worker's "I have a resumable observation" survives across rollouts in shared memory.
        self.has_data = torch.zeros(n_envs, dtype=torch.bool)

    def share_memory_(self) -> 'RolloutBuffer':
        """Moves every tensor to shared memory for subprocess workers."""
        for tensor in self.observation.values():
            tensor.share_memory_()
        for tensor in (self.masks, self.dones, self.actions, self.logp_ent_val, self.rewards, self.frames,
                       self.returns, self.advantages, self.has_data):
            tensor.share_memory_()
        return self

    def _store_obs(self, env_index, t, obs):
        for key, storage in self.observation.items():
            storage[env_index, t] = torch.as_tensor(np.asarray(obs[key])).reshape(storage.shape[2:])
        self.masks[env_index, t] = torch.as_tensor(np.asarray(obs[MASK_KEY], dtype=bool))

    def _load_obs(self, env_index, t):
        obs = {key: storage[env_index, t] for key, storage in self.observation.items()}
        obs[MASK_KEY] = self.masks[env_index, t]
        return obs

    def collect(self, env_index: int, network, env, episode_tracker=None, *, seed=None):
        """Fills this env's slice with ``memory_length`` steps, then computes returns and advantages.

        ``episode_tracker`` (metrics.EpisodeTracker) sees every step for training metrics.
        ``seed`` applies only to the initial reset, not later episodes or rollout batches.
        """
        # pylint: disable=too-many-locals
        if not bool(self.has_data[env_index]):
            obs, _ = env.reset(seed=seed)
            if episode_tracker is not None:
                episode_tracker.begin()
            done = 0.0
        else:
            obs = self._load_obs(env_index, self.memory_length)
            done = float(self.dones[env_index, self.memory_length])

        was_training = network.training
        network.eval()
        try:
            with torch.no_grad():
                for t in range(self.memory_length):
                    self.dones[env_index, t] = done
                    self._store_obs(env_index, t, obs)
                    out = network.evaluate(obs, self.masks[env_index, t].unsqueeze(0))
                    action = out.actions[0]
                    self.actions[env_index, t] = action
                    self.logp_ent_val[env_index, t, 0] = out.log_prob[0]
                    self.logp_ent_val[env_index, t, 1] = out.entropy[0]
                    self.logp_ent_val[env_index, t, 2] = out.value[0]

                    obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                    self.rewards[env_index, t] = float(reward)
                    self.frames[env_index, t] = float(info['frames'])
                    if episode_tracker is not None:
                        episode_tracker.step(action, float(reward), info, terminated or truncated)
                    if truncated and not terminated and info.get('bootstrap_truncation', False):
                        final_value = float(network.get_value(obs)[0])
                        self.rewards[env_index, t] += (
                            self.gamma ** (float(info['frames']) / FRAMES_PER_DISCOUNT) * final_value)

                    done = 1.0 if (terminated or truncated) else 0.0
                    if done:
                        obs, _ = env.reset()
                        if episode_tracker is not None:
                            episode_tracker.begin()

                self._store_obs(env_index, self.memory_length, obs)
                self.dones[env_index, self.memory_length] = done
                last_value = float(network.get_value(self._load_obs(env_index, self.memory_length))[0])
                returns, advantages = frame_aware_gae(
                    self.rewards[env_index], self.logp_ent_val[env_index, :, 2], self.frames[env_index],
                    self.dones[env_index, 1:], last_value, self.gamma, self.lam)
                self.returns[env_index] = returns
                self.advantages[env_index] = advantages
                self.has_data[env_index] = True
        finally:
            network.train(was_training)

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
