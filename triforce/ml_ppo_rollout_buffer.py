import torch
from gymnasium.spaces import Box, MultiBinary, Dict, Discrete, MultiDiscrete


class PPORolloutBuffer:
    """Training variables for PPO.

    Supports both Discrete and MultiDiscrete action spaces. Actions are stored
    separately from log-probs/entropy/values to handle variable action dimensions.
    """
    def __init__(self, memory_length, n_envs, observation_space, action_space, gamma, lam):
        self.observation_space = observation_space
        self.action_space = action_space
        self.memory_length = memory_length
        self.n_envs = n_envs
        self._gamma = gamma
        self._lambda = lam

        # Determine action dimensionality and mask size for Discrete vs MultiDiscrete
        if isinstance(action_space, MultiDiscrete):
            self.action_dim = len(action_space.nvec)
            mask_size = int(sum(action_space.nvec))
        else:
            self.action_dim = 1
            mask_size = action_space.n

        if isinstance(observation_space, Dict):
            obs = {}
            for key, value in observation_space.items():
                obs[key] = self._get_observation_part(value)
            self.observation = obs
        else:
            self.observation = self._get_observation_part(observation_space)

        self.dones = torch.empty(n_envs, memory_length + 1, dtype=torch.float32, device="cpu")

        # Actions stored separately: [n_envs, memory_length, action_dim]
        # Discrete: action_dim=1, MultiDiscrete: action_dim=len(nvec) (typically 2)
        self.actions = torch.empty(n_envs, memory_length, self.action_dim, device="cpu")
        # Log-prob, entropy, value: [n_envs, memory_length, 3]
        self.logp_ent_val = torch.empty(n_envs, memory_length, 3, device="cpu")

        self.rewards = torch.empty(n_envs, memory_length, dtype=torch.float32, device="cpu")
        self.masks = torch.empty(n_envs, memory_length + 1, mask_size, dtype=torch.bool, device="cpu")
        self.ones_mask = torch.ones(mask_size, dtype=torch.bool, device="cpu")
        self.returns = torch.empty(n_envs, memory_length, dtype=torch.float32, device="cpu")
        self.advantages = torch.empty(n_envs, memory_length, dtype=torch.float32, device="cpu")

        self.has_data = [False] * n_envs

    def __setitem__(self, idx, other):
        """Assigns the result of a single environment to idx."""
        assert other.n_envs == 1
        if isinstance(self.observation, dict):
            for key in self.observation:
                self.observation[key][idx] = other.observation[key][0]
        else:
            self.observation[idx] = other.observation[0]

        self.dones[idx] = other.dones[0]
        self.actions[idx] = other.actions[0]
        self.logp_ent_val[idx] = other.logp_ent_val[0]
        self.rewards[idx] = other.rewards[0]
        self.masks[idx] = other.masks[0]
        self.returns[idx] = other.returns[0]
        self.advantages[idx] = other.advantages[0]
        self.has_data[idx] = True

    def share_memory_(self):
        """Move all tensors to shared memory for cross-process access."""
        if isinstance(self.observation, dict):
            for key in self.observation:
                self.observation[key].share_memory_()
        else:
            self.observation.share_memory_()

        self.dones.share_memory_()
        self.actions.share_memory_()
        self.logp_ent_val.share_memory_()
        self.rewards.share_memory_()
        self.masks.share_memory_()
        self.ones_mask.share_memory_()
        self.returns.share_memory_()
        self.advantages.share_memory_()
        return self

    def _get_observation_part(self, space):
        if isinstance(space, MultiBinary):
            return torch.empty(self.n_envs, self.memory_length + 1, space.n, dtype=torch.float32, device="cpu")

        if isinstance(space, Box):
            return torch.empty(self.n_envs, self.memory_length + 1, *space.shape, dtype=torch.float32, device="cpu")

        if isinstance(space, Discrete):
            return torch.empty(self.n_envs, self.memory_length + 1, space.n, dtype=torch.float32, device="cpu")

        if isinstance(space, MultiDiscrete):
            return torch.empty(self.n_envs, self.memory_length + 1, len(space.nvec), dtype=torch.float32, device="cpu")

        raise ValueError(f"Unsupported observation space: {space}")

    def ppo_main_loop(self, batch_index, network, env, callback, total_steps=0):
        """Processes a single loop of training, filling one batch of variables."""

        # pylint: disable=too-many-locals, too-many-branches, too-many-statements

        if not self.has_data[batch_index]:
            obs, info = env.reset()
            action_mask = info.get('action_mask', None)
            done = 0.0
        else:
            if isinstance(self.observation, dict):
                obs = {key: o[batch_index, self.memory_length] for key, o in self.observation.items()}
            else:
                obs = self.observation[batch_index, self.memory_length]

            done = self.dones[batch_index, self.memory_length]
            action_mask = self.masks[batch_index, self.memory_length]

        infos = []

        with torch.no_grad():
            for t in range(self.memory_length):
                # Store current obs/done
                self.dones[batch_index, t] = done

                # Unsqueeze obs and the action_mask, since get_action_and_value expects a batch
                if isinstance(obs, dict):
                    for key, ob_tensor in obs.items():
                        self.observation[key][batch_index, t] = ob_tensor

                else:
                    self.observation[batch_index, t] = obs

                if action_mask is not None:
                    action_mask = action_mask.unsqueeze(0)

                # Record the action, logp, entropy, and value
                action, logp, ent, val = network.get_action_and_value(obs, action_mask)

                # Store action: scalar for Discrete, [action_dim] for MultiDiscrete
                if self.action_dim == 1:
                    self.actions[batch_index, t, 0] = action.item()
                else:
                    self.actions[batch_index, t] = action.squeeze(0)

                self.logp_ent_val[batch_index, t, 0] = logp.squeeze()
                self.logp_ent_val[batch_index, t, 1] = ent.squeeze()
                self.logp_ent_val[batch_index, t, 2] = val.squeeze()

                self.masks[batch_index, t] = action_mask if action_mask is not None else self.ones_mask

                # step environment
                if self.action_dim == 1:
                    step_action = action.item()
                else:
                    step_action = action.squeeze(0)
                next_obs, reward, terminated, truncated, info = env.step(step_action)

                action_mask = info.get('action_mask', None)
                infos.append(info)
                self.rewards[batch_index, t] = float(reward)

                done = 1.0 if (terminated or truncated) else 0.0
                if terminated or truncated:
                    next_obs, info = env.reset()
                    action_mask = info.get('action_mask', None)

                obs = next_obs

                if callback:
                    callback.on_progress(1, total_steps)

            # Store final obs/done/mask
            if isinstance(obs, dict):
                for key, ob_tensor in obs.items():
                    self.observation[key][batch_index, self.memory_length] = ob_tensor
            else:
                self.observation[batch_index, self.memory_length] = obs

            self.dones[batch_index, self.memory_length] = done
            if action_mask is not None:
                self.masks[batch_index, self.memory_length] = action_mask
            else:
                self.masks[batch_index, self.memory_length] = self.ones_mask

            # Get the value of the final observation and calculate returns/advantages
            last_value = network.get_value(obs).item()
            returns, advantages = self._compute_returns_advantages(batch_index, last_value)
            self.returns[batch_index] = returns
            self.advantages[batch_index] = advantages
            self.has_data[batch_index] = True

        return infos

    def _compute_returns_advantages(self, batch_idx, last_value):
        with torch.no_grad():
            advantages = torch.zeros(self.memory_length, device="cpu")
            last_gae = 0
            for t in reversed(range(self.memory_length)):
                mask = 1.0 - self.dones[batch_idx, t + 1]

                if t + 1 < self.memory_length:
                    next_value = self.logp_ent_val[batch_idx, t + 1, 2]
                else:
                    next_value = last_value

                reward = self.rewards[batch_idx, t]
                current_val = self.logp_ent_val[batch_idx, t, 2]

                delta = reward + self._gamma * next_value * mask - current_val
                advantages[t] = last_gae = delta + self._gamma * self._lambda * mask * last_gae

            returns = advantages + self.logp_ent_val[batch_idx, :, 2]
            return returns, advantages
