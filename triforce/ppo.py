from collections import Counter
import math
from multiprocessing import Queue
import time
import numpy as np
import torch
from torch import nn
import torch.distributions as dist
from torch.utils.tensorboard import SummaryWriter

from .ppo_subprocess import PPOSubprocess
from .rewards import StepRewards

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

NORM_ADVANTAGES = True
CLIP_VAL_LOSS = True
LEARNING_RATE = 0.00025
GAMMA = 0.99
LAMBDA = 0.95
CLIP_COEFF = 0.2
ENT_COEFF = 0.001 # lowered, original = 0.01
VS_COEFF = 0.5
MAX_GRAD_NORM = 0.5
EPSILON = 1e-5

class PPO:
    """PPO Implementation.  Cribbed from https://www.youtube.com/watch?v=MEt6rrxH8W4."""
    def __init__(self, network : Network, device, log_dir, n_envs, **kwargs):
        self._norm_advantages = kwargs.get('norm_advantages', NORM_ADVANTAGES)
        self._clip_val_loss = kwargs.get('clip_val_loss', CLIP_VAL_LOSS)
        self._learning_rate = kwargs.get('learning_rate', LEARNING_RATE)
        self._gamma = kwargs.get('gamma', GAMMA)
        self._lambda = kwargs.get('lambda', LAMBDA)
        self._clip_coeff = kwargs.get('clip_coeff', CLIP_COEFF)
        self._ent_coeff = kwargs.get('ent_coeff', ENT_COEFF)
        self._vf_coeff = kwargs.get('vf_coeff', VS_COEFF)
        self._max_grad_norm = kwargs.get('max_grad_norm', MAX_GRAD_NORM)
        self._epsilon = kwargs.get('epsilon', EPSILON)
        self.memory_length = kwargs.get('memory_length', 128)
        self.minibatches = kwargs.get('minibatches', 4)
        self.num_epochs = kwargs.get('num_epochs', 4)

        if kwargs:
            raise ValueError(f"Unknown arguments: {kwargs}")

        self.kwargs = kwargs

        self.optimizer_state = None

        self.network = network
        self.log_dir = log_dir
        self.device = device
        self.tensorboard = SummaryWriter(log_dir) if log_dir else None

        self.total_steps = 0
        self.n_envs = n_envs

        self.reward_values = {}
        self.endings = {}
        self.start_time = None

        # do we have a multi-part observation?
        observation = []
        if isinstance(network.observation_shape[0], tuple):
            for shape in network.observation_shape:
                assert isinstance(shape, tuple)
                obs_part = torch.empty(self.n_envs, self.memory_length + 1, *shape, dtype=torch.float32, device="cpu")
                observation.append(obs_part)
        else:
            obs_part = torch.empty(self.n_envs, self.memory_length + 1, *network.observation_shape,
                                   dtype=torch.float32, device="cpu")
            observation.append(obs_part)

        self.observation = tuple(observation)

        self.dones = torch.empty(self.n_envs, self.memory_length + 1, dtype=torch.float32, device="cpu")
        self.act_logp_ent_val = torch.empty(self.n_envs, self.memory_length, 4, device="cpu")
        self.masks = torch.empty(self.n_envs, self.memory_length + 1, self.network.action_size, dtype=torch.bool,
                                 device="cpu")
        self.rewards = torch.empty(self.n_envs, self.memory_length, dtype=torch.float32,
                                   device="cpu")

        self.ones_mask = torch.ones(self.network.action_size, dtype=torch.bool, device="cpu")

    def _get_and_remove(self, dictionary, key, default):
        if key in dictionary:
            value = dictionary[key]
            del dictionary[key]
            return value

        return default

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

        batch_returns = torch.zeros(self.n_envs, self.memory_length, device="cpu")
        batch_advantages = torch.zeros(self.n_envs, self.memory_length, device="cpu")

        env = create_env()

        iteration = 0
        while iteration < iterations:
            infos, next_value = self.build_one_batch(0, env, progress, iteration)
            returns, advantages = self._compute_returns(0, next_value)
            batch_returns[0] = returns
            batch_advantages[0] = advantages

            iteration += self.n_envs * self.memory_length
            self._batch_update(infos)
            self._optimize(batch_returns, batch_advantages, iteration)

        env.close()

    def _train_multiproc(self, create_env, iterations, progress):
        # pylint: disable=too-many-locals
        kwargs = self.kwargs.copy()
        kwargs['network'] = self.network
        kwargs['device'] = "cpu"
        kwargs['log_dir'] = None
        kwargs['n_envs'] = 1

        workers = []
        result_queue = Queue()
        for idx in range(self.n_envs):
            worker = PPOSubprocess(idx, create_env, PPO, kwargs, result_queue)
            workers.append(worker)

        steps = math.ceil(iterations / (self.n_envs * self.memory_length))
        try:
            for _ in range(steps):
                weights = self.network.state_dict()
                for worker in workers:
                    worker.build_batch_async(weights, None)

                infos = []
                batch_returns = torch.zeros(self.n_envs, self.memory_length, device="cpu")
                batch_advantages = torch.zeros(self.n_envs, self.memory_length, device="cpu")

                recieved = []
                for _ in range(self.n_envs):
                    message = result_queue.get()
                    match message['command']:
                        case 'exit':
                            print(f"Unexpected exit: {message['idx']}")
                            raise EOFError()

                        case 'error':
                            print(f"Error in worker {message['idx']}:")
                            error = message['error']
                            print(error)
                            print(message['traceback'])
                            raise error

                        case 'build_batch':
                            idx = message['idx']
                            if idx in recieved:
                                raise ValueError(f"Duplicate message for {idx}")

                            recieved.append(idx)

                            returns, advantages, info = self._process_batch(message)
                            batch_returns[idx] = returns
                            batch_advantages[idx] = advantages
                            infos.extend(info)

                            if progress is not None:
                                progress.update(self.memory_length)

                self._batch_update(infos)
                self._optimize(batch_returns, batch_advantages, self.total_steps)

        except EOFError:
            pass

        finally:
            # Tell workers to shut down regardless
            for worker in workers:
                worker.close_async()

        # only hit without exception
        for worker in workers:
            worker.join()

    def _process_batch(self, message):
        idx = message['idx']
        infos = message['infos']
        next_value = message['next_value']

        for i, observation in enumerate(message['observation']):
            self.observation[i][idx] = observation

        self.dones[idx] = message['dones']
        self.rewards[idx] = message['rewards']
        self.act_logp_ent_val[idx] = message['act_logp_ent_val']
        self.masks[idx] = message['masks']

        returns, advantages = self._compute_returns(idx, next_value)
        return returns, advantages, infos

    def build_one_batch(self, batch_idx, env, progress, iteration):
        """Build a single batch of data from the environment."""
        # pylint: disable=too-many-locals
        if iteration == 0:
            obs, info = env.reset()
            action_mask = info.get('action_mask', None)
            done = 0.0
        else:
            obs = [o[batch_idx, self.memory_length] for o in self.observation]
            done = self.dones[batch_idx, self.memory_length]
            action_mask = self.masks[batch_idx, self.memory_length]

        infos = []

        with torch.no_grad():
            for t in range(self.memory_length):
                # Store current obs/done
                self.dones[batch_idx, t] = done
                for obs_idx, ob_tensor in enumerate(obs):
                    self.observation[obs_idx][batch_idx, t] = ob_tensor

                # Unsqueeze obs and the action_mask, since get_action_and_value expects a batch
                obs_batched = tuple(o.unsqueeze(0) for o in obs)
                if action_mask is not None:
                    action_mask = action_mask.unsqueeze(0)

                # Record the action, logp, entropy, and value
                act_logp_ent_val = self.network.get_action_and_value(obs_batched, action_mask)
                self.act_logp_ent_val[batch_idx, t] = torch.stack(act_logp_ent_val, dim=-1)
                self.masks[batch_idx, t] = action_mask if action_mask is not None else self.ones_mask

                # step environment
                action = act_logp_ent_val[0].item()
                next_obs, reward, terminated, truncated, info = env.step(action)

                action_mask = info.get('action_mask', None)
                infos.append(info)
                self.rewards[batch_idx, t] = reward

                next_done = 1.0 if (terminated or truncated) else 0.0
                if terminated or truncated:
                    next_obs, info = env.reset()
                    next_done = 0.0

                obs = next_obs
                done = next_done

                if progress:
                    progress.update(1)

            # Store final obs/done/mask
            for obs_idx, ob_tensor in enumerate(obs):
                self.observation[obs_idx][batch_idx, self.memory_length] = ob_tensor

            self.dones[batch_idx, self.memory_length] = done
            self.masks[batch_idx, self.memory_length] = action_mask if action_mask is not None else self.ones_mask

            # Get the value of the final observation
            obs_batched = tuple(o.unsqueeze(0) for o in obs)
            next_value = self.network.get_value(obs_batched).item()

        # Return carry-over state: current obs, done, and action_mask
        return infos, next_value

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
            advantages = torch.zeros(self.memory_length, device="cpu")
            last_gae = 0
            for t in reversed(range(self.memory_length)):
                mask = 1.0 - self.dones[batch_idx, t]

                if t + 1 < self.memory_length:
                    next_value = self.act_logp_ent_val[batch_idx, t + 1, 3]
                else:
                    next_value = last_value

                reward = self.rewards[batch_idx, t]
                current_val = self.act_logp_ent_val[batch_idx, t, 3]

                delta = reward + self._gamma * next_value * mask - current_val
                advantages[t] = last_gae = delta + self._gamma * self._lambda * mask * last_gae

            returns = advantages + self.act_logp_ent_val[batch_idx, :, 3]
            return returns, advantages

    def _optimize(self, returns, advantages, iterations):
        # pylint: disable=too-many-locals, too-many-statements

        # flatten observations
        b_obs = []
        for obs_part in self.observation:
            b_obs.append(obs_part[:, :self.memory_length])


        for i, obs_part in enumerate(b_obs):
            b_obs[i] = obs_part.reshape(-1, *obs_part.shape[2:]).to(self.device)

        b_obs = tuple(b_obs)

        # flatten actions, logprobs, values, masks
        actions   = self.act_logp_ent_val[:, :, 0].to(self.device)
        logprobs  = self.act_logp_ent_val[:, :, 1].to(self.device)
        values    = self.act_logp_ent_val[:, :, 3].to(self.device)

        b_actions  = actions.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_values   = values.reshape(-1)

        masks     = self.masks
        b_masks    = masks.reshape(-1, masks.shape[-1]).to(self.device)

        # flatten returns, advantages
        b_advantages = advantages.reshape(-1).to(self.device)
        b_returns    = returns.reshape(-1).to(self.device)

        # standard PPO update
        batch_size = self.memory_length * self.n_envs
        minibatch_size = batch_size // self.minibatches

        network = self.network.to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self._learning_rate, eps=self._epsilon)
        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)

        b_inds = np.arange(batch_size)
        clipfracs = []
        for _ in range(self.num_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # Slice each item for this mini-batch
                mb_obs = []
                for obs_part in b_obs:
                    mb_obs.append(obs_part[mb_inds])

                mb_obs = tuple(mb_obs)

                mb_actions   = b_actions[mb_inds].long()
                mb_logprobs  = b_logprobs[mb_inds]
                mb_values    = b_values[mb_inds]
                mb_returns   = b_returns[mb_inds]
                mb_adv       = b_advantages[mb_inds]

                mb_masks = b_masks[mb_inds, :]

                _, newlogprob, entropy, newvalue = network.get_action_and_value(mb_obs, mb_masks, mb_actions)

                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > self._clip_coeff).float().mean().item())

                # Normalize advantages
                if self._norm_advantages:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self._clip_coeff, 1 + self._clip_coeff)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self._clip_val_loss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(newvalue - mb_values,
                                                        -self._clip_coeff, self._clip_coeff)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * (newvalue - mb_returns).pow(2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self._ent_coeff * entropy_loss + self._vf_coeff * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), self._max_grad_norm)
                optimizer.step()

        self.network = network.to("cpu")
        self.optimizer_state = optimizer.state_dict()

        # After training, compute stats like explained variance
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if self.tensorboard:
            self.tensorboard.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iterations)
            self.tensorboard.add_scalar("losses/value_loss", v_loss.item(), iterations)
            self.tensorboard.add_scalar("losses/policy_loss", pg_loss.item(), iterations)
            self.tensorboard.add_scalar("losses/entropy", entropy_loss.item(), iterations)
            self.tensorboard.add_scalar("losses/old_approx_kl", old_approx_kl.item(), iterations)
            self.tensorboard.add_scalar("losses/approx_kl", approx_kl.item(), iterations)
            self.tensorboard.add_scalar("losses/clipfrac", np.mean(clipfracs), iterations)
            self.tensorboard.add_scalar("losses/explained_variance", explained_var, iterations)
            if self.start_time is not None:
                self.tensorboard.add_scalar("charts/SPS", int(iterations / (time.time() - self.start_time)), iterations)
