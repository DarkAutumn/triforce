from collections import Counter
import time
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


from .rewards import StepRewards

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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
EPSILON = 1e-5

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
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE, eps=EPSILON)

        self.reward_values = {}
        self.endings = {}
        self.start_time = None

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

    def build_one_batch(self, batch_index, env, progress, state=None):
        if state is None or state[1] == 1.0:
            # If we have no state or the previous state was done
            obs, _, action_mask = env.reset()
            done = 0.0
        else:
            # Otherwise, we continue from the previous environment state
            obs, done, action_mask = state

        infos = []

        with torch.no_grad():
            for t in range(self.memory_length):
                # (a) Store the *current* obs in our buffers
                for obs_idx, ob_tensor in enumerate(obs):
                    self.obs[obs_idx][batch_index, t] = ob_tensor

                self.dones[batch_index, t] = done

                # (b) Get action logits/logp/entropy/value from policy
                act_logp_ent_val_mask = self.network.get_act_logp_ent_val_mask(obs, action_mask)
                self.act_logp_ent_val_mask[batch_index, t] = act_logp_ent_val_mask[0]

                # (c) Extract the actual actions (assuming single env)
                actions = act_logp_ent_val_mask[0, :, 0]

                # (d) Step environment
                next_obs, reward, terminated, truncated, info, action_mask = env.step(actions)
                infos.append(info)
                self.rewards[batch_index, t] = reward

                # (e) Check if environment finished
                next_done = 1.0 if (terminated or truncated) else 0.0
                if terminated or truncated:
                    next_obs, _ = env.reset()
                    next_done = 0.0
                    action_mask = None

                # (f) Prepare for next iteration
                obs = next_obs
                done = next_done

                if progress:
                    progress.update(1)

            # ------------------------------------------
            # 2) Store final obs/done for bootstrapping
            # ------------------------------------------
            for obs_idx, ob_tensor in enumerate(obs):
                self.obs[obs_idx][batch_index, self.memory_length] = ob_tensor
            self.dones[batch_index, self.memory_length] = done

            # (g) Get value for the final state
            next_value = self.network.get_value(obs, action_mask)

        # Return carry-over state: current obs, done, and action_mask
        state = (obs, done, action_mask)
        return infos, next_value, state

    def train(self, env, iterations, progress=None):
        self.start_time = time.time()

        batch_returns = torch.zeros(self.n_envs, self.memory_length, device=self.device)
        batch_advantages = torch.zeros(self.n_envs, self.memory_length, device=self.device)

        # Initialize carry-over states for each environment
        # None means "no previous state," so we'll reset in build_one_batch
        states = [None] * self.n_envs

        iteration = 0
        while iteration < iterations:
            infos = []
            for i in range(self.n_envs):
                info, next_value, new_state = self.build_one_batch(i, env, progress, states[i])

                states[i] = new_state
                infos.extend(info)

                returns, advantages = self._compute_returns(i, next_value)
                batch_returns[i] = returns
                batch_advantages[i] = advantages

            self._batch_update(infos)
            self.optimize(batch_returns, batch_advantages)
            iteration += self.n_envs * self.memory_length

    def _batch_update(self, infos):
        success_rate = []
        evaluation = []
        endings = []

        for info in infos:
            rewards : StepRewards = info.get('rewards', None)
            if rewards is not None:
                for outcome in rewards:
                    self.reward_values[outcome.name] = outcome.value + self.reward_values.get(outcome.name, 0)

                if rewards.ending is not None:
                    endings.append(rewards.ending)
                    if rewards.ending.startswith('success'):
                        success_rate.append(1)
                    else:
                        success_rate.append(0)

                    evaluation.append(rewards.score)

        if success_rate:
            self.tensorboard.add_scalar('evaluation/success-rate', np.mean(success_rate))
        if evaluation:
            self.tensorboard.add_scalar('evaluation/score', np.mean(evaluation))

        endings = Counter(endings)
        for ending, count in endings.items():
            self.endings[ending] = count + self.endings.get(ending, 0)
            self.tensorboard.add_scalar('end/' + ending, count)

        for name, rew in self.reward_values.items():
            parts = name.split('-', 1)
            self.tensorboard.add_scalar(f"{parts[0]}/{parts[1]}", rew)

        for key in self.reward_values:
            self.reward_values[key] = 0

        for key in self.endings:
            self.endings[key] = 0

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

    def optimize(self, returns, advantages, iterations):
        # b_variable block
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.num_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.network.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > CLIP_COEFF).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if NORM_ADVANTAGES:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEFF, 1 + CLIP_COEFF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if CLIP_VAL_LOSS:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -CLIP_COEFF,
                                                                CLIP_COEFF)

                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEFF * entropy_loss + v_loss * VF_COEFF

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()


        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        self.tensorboard.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], iterations)
        self.tensorboard.add_scalar("losses/value_loss", v_loss.item(), iterations)
        self.tensorboard.add_scalar("losses/policy_loss", pg_loss.item(), iterations)
        self.tensorboard.add_scalar("losses/entropy", entropy_loss.item(), iterations)
        self.tensorboard.add_scalar("losses/old_approx_kl", old_approx_kl.item(), iterations)
        self.tensorboard.add_scalar("losses/approx_kl", approx_kl.item(), iterations)
        self.tensorboard.add_scalar("losses/clipfrac", np.mean(clipfracs), iterations)
        self.tensorboard.add_scalar("losses/explained_variance", explained_var, iterations)
        self.tensorboard.add_scalar("charts/SPS", int(iterations / (time.time() - self.start_time)), iterations)
