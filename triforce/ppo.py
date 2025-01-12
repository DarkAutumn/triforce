import math
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

NORM_ADVANTAGES = True
CLIP_VAL_LOSS = True
LEARNING_RATE = 0.00025
MIN_LR = LEARNING_RATE
ANNEALING_FACTOR = 1    # none
GAMMA = 0.99
LAMBDA = 0.95
CLIP_COEFF = 0.2
ENT_COEFF = 0.001 # lowered, original = 0.01
VF_COEFF = 0.5
MAX_GRAD_NORM = 0.5

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPO:
    def __init__(self, network, device, log_dir):
        self.network = network
        self.log_dir = log_dir
        self.device = device
        self.tensorboard = SummaryWriter(log_dir) if log_dir else None

        self.memory_length = 4096
        self.batch_size = 128
        self.minibatches = 4
        self.minibatch_size = self.batch_size // self.minibatches
        self.num_epochs = 4
        self.total_steps = 0
        self.n_envs = 1

        obs_image = torch.empty(self.n_envs, self.memory_length + 1, 1, self.network.viewport_size,
                                self.network.viewport_size, dtype=torch.float32, device=device)

        obs_vectors = torch.empty(self.n_envs, self.memory_length + 1, self.network.vectors_size[0],
                                  self.network.vectors_size[1], self.network.vectors_size[2], dtype=torch.float32,
                                  device=device)

        obs_features = torch.empty(self.n_envs, self.memory_length + 1, self.network.info_size, dtype=torch.float32,
                                   device=device)

        self.obs = obs_image, obs_vectors, obs_features

        self.dones = torch.empty(self.n_envs, self.memory_length + 1, dtype=torch.float32, device=device)
        self.act_logp_ent_val_mask = torch.empty(self.n_envs, self.memory_length, 5, device=device)
        self.rewards = torch.empty(self.n_envs, self.memory_length, dtype=torch.float32,
                                   device=device)



    def build_one_batch(self, batch_index, env, progress):
        next_obs, info, action_mask = env.reset()
        ep_steps = 0
        next_done = 0.0
        infos = [info]

        with torch.no_grad():
            for t in range(self.memory_length):
                self.total_steps += 1
                ep_steps += 1

                act_logp_ent_val_mask = self.network.get_act_logp_ent_val_mask(obs, action_mask)
                self.act_logp_ent_val_mask[batch_index, t] = act_logp_ent_val_mask[0]
                actions = act_logp_ent_val_mask[0, :, 0]
                obs, reward, terminated, truncated, info, action_mask = env.step(actions)
                infos.append(info)

                done = 1.0 if terminated or truncated else 0.0

                for obs_idx, ob_tensor in enumerate(obs):
                    self.obs[obs_idx][batch_index, t] = ob_tensor

                self.dones[batch_index, t] = next_done
                self.rewards[batch_index, t] = reward

                if terminated or truncated:
                    next_obs, info = env.reset()
                    next_done = 0.0
                    action_mask = None
                    infos.append(info)

                    ep_steps = 0

                else:
                    next_obs = obs
                    next_done = done

                if progress:
                    progress.update(1)

            self.dones[batch_index, self.memory_length] = next_done
            next_value = self.network.get_value(next_obs, action_mask)
            return infos, next_value

    def train(self, env, iterations, progress=None):
        batch_returns = torch.zeros(self.n_envs, self.memory_length, device=self.device)
        batch_advantages = torch.zeros(self.n_envs, self.memory_length, device=self.device)

        iteration = 0
        while iteration < iterations:
            for i in range(self.n_envs):
                infos, next_value = self.build_one_batch(i, env, progress)
                returns, advantages = self._compute_returns(i, next_value)

                batch_returns[i] = returns
                batch_advantages[i] = advantages

            self.optimize(batch_returns, batch_advantages, infos)

            iteration += self.n_envs * self.memory_length


    def _compute_returns(self, idx, last_value):
        with torch.no_grad():
            advantages = torch.zeros(self.memory_length, device=self.device)
            last_gae = 0
            for t in reversed(range(self.memory_length)):
                mask = 1.0 - self.dones[idx, t]

                if t + 1 < self.memory_length:
                    next_value = self.act_logp_ent_val_mask[idx, t + 1, 3]
                else:
                    next_value = last_value

                reward = self.rewards[idx, t]
                current_val = self.act_logp_ent_val_mask[idx, t, 3]  # index 3 is "value"

                delta = reward + GAMMA * next_value * mask - current_val
                advantages[t] = last_gae = delta + GAMMA * LAMBDA * mask * last_gae

            returns = advantages + self.act_logp_ent_val_mask[idx, :, :, 3]
            return returns, advantages
