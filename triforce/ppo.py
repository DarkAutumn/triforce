from collections import Counter
import time
import numpy as np
import torch
from torch import nn
import torch.distributions as dist
from torch.utils.tensorboard import SummaryWriter

from .ppo_subprocess import PPOSubprocess
from .rewards import StepRewards

NORM_ADVANTAGES = True
CLIP_VAL_LOSS = True
LEARNING_RATE = 0.00025
GAMMA = 0.99
LAMBDA = 0.95
CLIP_COEFF = 0.2
ENT_COEFF = 0.001 # lowered, original = 0.01
VF_COEFF = 0.5
MAX_GRAD_NORM = 0.5
EPSILON = 1e-5

class Network(nn.Module):
    """The base class of neural networks used for PPO training."""
    base : nn.Module
    action_net : nn.Module
    value_net : nn.Module

    def __init__(self, base_network : nn.Module, obs_shape, action_size):
        super().__init__()
        self.observation_shape = obs_shape
        self.action_size = action_size
        self.tuple_obs = isinstance(obs_shape[0], tuple)

        self.base = base_network
        self.action_net = self._layer_init(nn.Linear(64, action_size), std=0.01)
        self.value_net = self._layer_init(nn.Linear(64, 1), std=1.0)

    def forward(self, *inputs):
        """Forward pass."""
        x = self.base(*inputs)
        action = self.action_net(x)
        value = self.value_net(x)
        return action, value

    def get_action_and_value(self, obs, mask, actions=None, deterministic=False):
        """Gets the action, logprob, entropy, and value."""
        logits, value = self.forward(*obs if self.tuple_obs else obs)

        # mask out invalid actions
        if mask is not None:
            logits = logits.clone()
            invalid_mask = ~mask
            logits[invalid_mask] = -1e9

        # distribution for entropy calculation
        distribution = dist.Categorical(logits=logits)
        entropy = distribution.entropy()

        # sample an action if not provided
        if actions is None:
            if deterministic:
                actions = logits.argmax(dim=-1)
            else:
                actions = distribution.sample()

        log_prob = distribution.log_prob(actions)

        # value has shape [batch_size, 1], flatten if needed
        return actions, log_prob, entropy, value.view(-1)

    def get_value(self, obs):
        """Get value estimate."""
        _, value = self.forward(*obs if self.tuple_obs else obs)
        return value.view(-1)

    @staticmethod
    def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

class PPO:
    """PPO Implementation.  Cribbed from https://www.youtube.com/watch?v=MEt6rrxH8W4."""
    def __init__(self, network : Network, device, log_dir):
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

        # do we have a multi-part observation?
        observation = []
        if isinstance(network.observation_shape[0], tuple):
            for shape in network.observation_shape:
                assert isinstance(shape, tuple)
                obs_part = torch.empty(self.n_envs, self.memory_length + 1, *shape, dtype=torch.float32, device=device)
                observation.append(obs_part)
        else:
            obs_part = torch.empty(self.n_envs, self.memory_length + 1, *network.observation_shape,
                                   dtype=torch.float32, device=device)
            observation.append(obs_part)

        self.observation = tuple(observation)

        self.dones = torch.empty(self.n_envs, self.memory_length + 1, dtype=torch.float32, device=device)
        self.act_logp_ent_val = torch.empty(self.n_envs, self.memory_length, 4, device=device)
        self.masks = torch.empty(self.n_envs, self.memory_length, self.network.action_size, dtype=torch.bool,
                                 device=device)
        self.rewards = torch.empty(self.n_envs, self.memory_length, dtype=torch.float32,
                                   device=device)

        self.non_mask = torch.ones(self.network.action_size, dtype=torch.bool, device=device)


    def get_batch(self, batch_index):
        returns, advantages = self._compute_returns(batch_index, self.act_logp_ent_val[batch_index, -1, 3])
        batch = []
        for i in range(self.memory_length - 1):
            obs = []
            for obs_part in self.observation:
                obs.append(obs_part[batch_index, i])

            obs = tuple(obs)
            action = self.act_logp_ent_val[batch_index, i, 0]
            #logp = self.act_logp_ent_val[batch_index, i, 1]
            #ent = self.act_logp_ent_val[batch_index, i, 2]
            value = self.act_logp_ent_val[batch_index, i, 3]
            mask = self.masks[batch_index, i]
            reward = self.rewards[batch_index, i]
            done = self.dones[batch_index, i]

            batch.append([i, obs, action, mask, reward, done, value, returns[i], advantages[i]])

        headers = ("Step", "Observation", "Action", "Mask", "Reward", "Done", "Value", "Return",
                   "Advantage")
        return headers, batch

    def print_batch(self, headers, data):
        # Determine column widths
        col_widths = [
            max(len(str(item)) for item in col) if col else 0
            for col in data
        ]

        # Adjust widths to accommodate headers
        if headers:
            col_widths = [
                max(len(header), width)
                for header, width in zip(headers, col_widths)
            ]

        # Create format string for each row
        row_format = " | ".join(f"{{:<{width}}}" for width in col_widths)

        # Print headers (if provided)
        if headers:
            print(row_format.format(*headers))
            print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))

        # Print rows
        for row in data:
            print(row_format.format(*row))

    def train(self, create_env, iterations, progress=None):
        """
        create_env: a callable that returns a fresh environment
        """
        # pylint: disable=too-many-locals

        if self.n_envs == 1:
            self._train_single(create_env, iterations, progress)
        else:
            self._train_multiproc(create_env, iterations, progress)

    def _train_single(self, create_env, iterations, progress):
        """
        create_env: a callable that returns a fresh environment
        """
        self.start_time = time.time()

        batch_returns = torch.zeros(self.n_envs, self.memory_length, device=self.device)
        batch_advantages = torch.zeros(self.n_envs, self.memory_length, device=self.device)

        state = None
        env = create_env()

        iteration = 0
        while iteration < iterations:
            infos, next_value, state = self.build_one_batch(0, env, progress, state)
            returns, advantages = self._compute_returns(0, next_value)
            batch_returns[0] = returns
            batch_advantages[0] = advantages

            iteration += self.n_envs * self.memory_length
            self._batch_update(infos)
            self._optimize(batch_returns, batch_advantages, iteration)

        env.close()

    def _train_multiproc(self, create_env, iterations, progress):
        # pylint: disable=too-many-locals
        # multi-process mode
        # 1) Spawn PPOSubprocess for each environment

        batch_returns = torch.zeros(self.n_envs, self.memory_length, device=self.device)
        batch_advantages = torch.zeros(self.n_envs, self.memory_length, device=self.device)

        iteration = 0

        workers = []
        for i in range(self.n_envs):
            # We'll create ppo_kwargs with the same network arch, device='cpu', etc.
            # or if you want them on GPU for rollout, do device='cuda' if you have memory
            ppo_kwargs = {
                "network": self.network,  # or a constructor
                "device": "cpu",          # typically CPU for env stepping
                "log_dir": None,
            }
            worker = PPOSubprocess(
                idx=i,
                create_env=create_env,
                ppo_class=PPO,
                ppo_kwargs=ppo_kwargs
            )
            workers.append(worker)

        while iteration < iterations:
            # 2) Send 'build_batch' command to each worker
            # optionally sync worker weights => self.state_dict()
            # if you want them to use the main policy weights for rollout
            for w in workers:
                w.build_batch_async(iterations=iteration, progress=progress, weights=self.state_dict())

            # 3) Collect results
            infos = []
            for i, w in enumerate(workers):
                # block until worker i returns
                infos, next_value, obs, dones, act_logp_ent_val, masks, rewards = w.get_result()
                for observation in obs:
                    self.observation[0][i] = observation

                self.dones[i] = dones
                self.rewards[i] = rewards
                self.act_logp_ent_val[i] = act_logp_ent_val
                self.masks[i] = masks
                returns, advantages = self._compute_returns(i, next_value)
                batch_returns[i] = returns
                batch_advantages[i] = advantages
                infos.extend(infos)

            iteration += self.n_envs * self.memory_length

            # 4) update the aggregator / logs
            self._batch_update(infos)

            # 5) do the global optimize
            self._optimize(batch_returns, batch_advantages, iteration)

            # 6) Optionally push the updated weights to each worker if you want new policy rollout
            # for w in workers:
            #     w.update_weights_async(self.state_dict())
            #     ack = w.get_result()

        # 7) Cleanup
        for w in workers:
            w.close()

    def build_one_batch(self, batch_idx, env, progress, state=None):
        """Build a single batch of data from the environment."""
        # pylint: disable=too-many-locals
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
                    self.observation[obs_idx][batch_idx, t] = ob_tensor

                self.dones[batch_idx, t] = done

                # Unsqueeze obs and the action_mask, since get_action_and_value expects a batch
                obs_batched = tuple(o.unsqueeze(0).to(self.device) for o in obs)
                if action_mask is not None:
                    action_mask = action_mask.unsqueeze(0).to(self.device)

                act_logp_ent_val = self.network.get_action_and_value(obs_batched, action_mask)
                self.act_logp_ent_val[batch_idx, t] = torch.stack(act_logp_ent_val, dim=-1)
                self.masks[batch_idx, t] = action_mask if action_mask is not None else self.non_mask

                # (c) Extract the actual actions (assuming single env)
                action = act_logp_ent_val[0].item()

                # (d) Step environment
                next_obs, reward, terminated, truncated, info, action_mask = env.step(action)
                infos.append(info)
                self.rewards[batch_idx, t] = reward

                # (e) Check if environment finished
                next_done = 1.0 if (terminated or truncated) else 0.0
                if terminated or truncated:
                    next_obs, _, action_mask = env.reset()
                    next_done = 0.0

                # (f) Prepare for next iteration
                obs = next_obs
                done = next_done

                if progress:
                    progress.update(1)

            # ------------------------------------------
            # 2) Store final obs/done for bootstrapping
            # ------------------------------------------
            for obs_idx, ob_tensor in enumerate(obs):
                self.observation[obs_idx][batch_idx, self.memory_length] = ob_tensor
            self.dones[batch_idx, self.memory_length] = done

            # (g) Get value for the final state
            obs_batched = tuple(o.unsqueeze(0).to(self.device) for o in obs)
            next_value = self.network.get_value(obs_batched).item()

        # Return carry-over state: current obs, done, and action_mask
        state = (obs, done, action_mask)
        return infos, next_value, state

    def _batch_update(self, infos):
        # pylint: disable=too-many-branches, too-many-locals
        success_rate = []
        total_seconds = []
        steps = []
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
                    if 'total_frames' in info:
                        total_seconds.append(info['total_frames'] / 60.1)
                    if 'steps' in info:
                        steps.append(info['steps'])

        if success_rate:
            self.tensorboard.add_scalar('evaluation/success-rate', np.mean(success_rate))
        if evaluation:
            self.tensorboard.add_scalar('evaluation/score', np.mean(evaluation))
        if total_seconds:
            self.tensorboard.add_scalar('rollout/seconds_per_episode', np.mean(total_seconds))
        if steps:
            self.tensorboard.add_scalar('rollout/steps_per_episode', np.mean(steps))

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

    def _compute_returns(self, batch_idx, last_value):
        with torch.no_grad():
            advantages = torch.zeros(self.memory_length, device=self.device)
            last_gae = 0
            for t in reversed(range(self.memory_length)):
                mask = 1.0 - self.dones[batch_idx, t]

                if t + 1 < self.memory_length:
                    next_value = self.act_logp_ent_val[batch_idx, t + 1, 3]
                else:
                    next_value = last_value

                reward = self.rewards[batch_idx, t]
                current_val = self.act_logp_ent_val[batch_idx, t, 3]

                delta = reward + GAMMA * next_value * mask - current_val
                advantages[t] = last_gae = delta + GAMMA * LAMBDA * mask * last_gae

            returns = advantages + self.act_logp_ent_val[batch_idx, :, 3]
            return returns, advantages

    def _optimize(self, returns, advantages, iterations):
        # pylint: disable=too-many-locals, too-many-statements

        # flatten observations
        b_obs = []
        for obs_part in self.observation:
            b_obs.append(obs_part[:, :self.memory_length])


        for i, obs_part in enumerate(b_obs):
            b_obs[i] = obs_part.reshape(-1, *obs_part.shape[2:])

        b_obs = tuple(b_obs)

        # flatten actions, logprobs, values, masks
        actions   = self.act_logp_ent_val[:, :, 0]
        logprobs  = self.act_logp_ent_val[:, :, 1]
        values    = self.act_logp_ent_val[:, :, 3]

        b_actions  = actions.reshape(-1)   # [n_envs*memory_length]
        b_logprobs = logprobs.reshape(-1)  # [n_envs*memory_length]
        b_values   = values.reshape(-1)    # [n_envs*memory_length]

        masks     = self.masks
        b_masks    = masks.reshape(-1)     # [n_envs*memory_length]

        # flatten returns, advantages
        b_advantages = advantages.reshape(-1)
        b_returns    = returns.reshape(-1)

        # standard PPO update
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for _ in range(self.num_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                # Slice each item for this mini-batch
                mb_obs = []
                for obs_part in b_obs:
                    mb_obs.append(obs_part[mb_inds])

                mb_obs = tuple(mb_obs)

                mb_actions   = b_actions[mb_inds].long()
                mb_logprobs  = b_logprobs[mb_inds]
                mb_masks     = b_masks[mb_inds]
                mb_values    = b_values[mb_inds]
                mb_returns   = b_returns[mb_inds]
                mb_adv       = b_advantages[mb_inds]

                _, newlogprob, entropy, newvalue = self.network.get_action_and_value(mb_obs, mb_actions, mb_masks)

                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > CLIP_COEFF).float().mean().item())

                # Normalize advantages
                if NORM_ADVANTAGES:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_COEFF, 1 + CLIP_COEFF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)  # shape [minibatch_size]
                if CLIP_VAL_LOSS:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(newvalue - mb_values,
                                                        -CLIP_COEFF, CLIP_COEFF)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * (newvalue - mb_returns).pow(2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEFF * entropy_loss + VF_COEFF * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

        # After training, compute stats like explained variance
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if self.tensorboard:
            self.tensorboard.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], iterations)
            self.tensorboard.add_scalar("losses/value_loss", v_loss.item(), iterations)
            self.tensorboard.add_scalar("losses/policy_loss", pg_loss.item(), iterations)
            self.tensorboard.add_scalar("losses/entropy", entropy_loss.item(), iterations)
            self.tensorboard.add_scalar("losses/old_approx_kl", old_approx_kl.item(), iterations)
            self.tensorboard.add_scalar("losses/approx_kl", approx_kl.item(), iterations)
            self.tensorboard.add_scalar("losses/clipfrac", np.mean(clipfracs), iterations)
            self.tensorboard.add_scalar("losses/explained_variance", explained_var, iterations)
            if self.start_time is not None:
                self.tensorboard.add_scalar("charts/SPS", int(iterations / (time.time() - self.start_time)), iterations)
