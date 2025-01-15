from collections import Counter
import math
from multiprocessing import Queue
import time
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .ml_subprocess import SubprocessWorker
from .ml_ppo_rollout_buffer import PPORolloutBuffer
from .models import Network, create_network
from .rewards import StepRewards

# default hyperparameters
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
TARGET_STEPS = 2048
EPOCHS = 10
MINIBATCHES = 4
LOG_RATE = 20_000
SAVE_INTERVAL = 50_000

class PPO:
    """PPO Implementation.  Adapted from from https://www.youtube.com/watch?v=MEt6rrxH8W4."""
    def __init__(self, device, log_dir, **kwargs):
        self._target_steps = kwargs.get('target_steps', TARGET_STEPS)
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
        self.minibatches = kwargs.get('minibatches', MINIBATCHES)
        self.num_epochs = kwargs.get('num_epochs', EPOCHS)

        self.kwargs = kwargs

        self.optimizer_state = None

        self.log_dir = log_dir
        self.device = device
        self.tensorboard = SummaryWriter(log_dir) if log_dir else None

        self.total_steps = 0

        self._logging = {}
        self.start_time = None

    def _get_and_remove(self, dictionary, key, default):
        if key in dictionary:
            value = dictionary[key]
            del dictionary[key]
            return value

        return default

    def train(self, network, create_env, iterations, progress=None, n_envs=1):
        """Train the network."""
        self.start_time = time.time()

        env = create_env()
        network = create_network(network, env.observation_space, env.action_space)

        if n_envs == 1:
            return self._train_single(network, env, iterations, progress)

        return self._train_multiproc(network, env, create_env, iterations, progress, n_envs)

    def _train_single(self, network, env, iterations, progress):
        memory_length = self._target_steps
        buffer = PPORolloutBuffer(memory_length, 1, env.observation_space, env.action_space,
                                  self._gamma, self._lambda)

        iteration = 0
        while iteration < iterations:
            infos = buffer.ppo_main_loop(0, network, env, progress)

            iteration += buffer.memory_length
            self._update_infos(infos, iteration)
            network = self._optimize(network, buffer, iteration)

        env.close()
        return network

    def _train_multiproc(self, network, env, create_env, iterations, progress, n_envs):
        memory_length = self._target_steps // n_envs
        variables = PPORolloutBuffer(memory_length, n_envs, env.observation_space, env.action_space,
                                     self._gamma, self._lambda)
        env.close()

        result_queue = Queue()
        kwargs = {
                'gamma': self._gamma,
                'lambda': self._lambda,
                'steps': variables.memory_length,
                }

        workers = [SubprocessWorker(idx, create_env, network, result_queue, kwargs) for idx in range(n_envs)]
        try:
            steps = math.ceil(iterations / (n_envs * memory_length))
            for step in range(steps):
                infos = self._subprocess_ppo(network, progress, variables, result_queue, workers)
                self._update_infos(infos, step * n_envs * memory_length)
                self._optimize(network, variables, step * n_envs * memory_length)

        except EOFError:
            pass

        finally:
            # Tell workers to shut down regardless
            for worker in workers:
                worker.close_async()

        # only hit without exception
        for worker in workers:
            worker.join()

        return network

    def _subprocess_ppo(self, network, progress, variables, result_queue, workers):
        weights = network.state_dict()
        for worker in workers:
            worker.run_main_loop_async(weights)

        infos = []
        recieved = []
        for _ in range(len(workers)):
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
                    result = message['result']
                    variables[idx] = result
                    infos.extend(message['infos'])

                    if progress is not None:
                        progress.update(result.memory_length)
        return infos

    def _update_infos(self, infos, iterations):
        curr = time.time()
        # pylint: disable=too-many-branches, too-many-locals
        if self.tensorboard is None:
            return

        reward_list = self._logging.setdefault('rewards', [])
        total_seconds = self._logging.setdefault('total_seconds', [])
        steps = self._logging.setdefault('steps', [])
        scores = self._logging.setdefault('scores', [])
        endings = self._logging.setdefault('endings', {})
        reward_values = self._logging.setdefault('reward_values', {})
        reward_counts = self._logging.setdefault('reward_counts', {})

        for info in infos:
            rewards : StepRewards = info.get('rewards', None)
            if rewards is not None and rewards.ending is not None:
                self._logging['total'] = self._logging.get('total', 0) + 1

                if rewards.ending.startswith('success'):
                    self._logging['success'] = self._logging.get('success', 0) + 1

                scores.append(rewards.score)
                if 'total_frames' in info:
                    total_seconds.append(info['total_frames'] / 60.1)
                if 'steps' in info:
                    steps.append(info['steps'])

                reward_list.append(info['total_reward'])

                endings[rewards.ending] = endings.get(rewards.ending, 0) + 1

                for name, value in info['reward_values'].items():
                    reward_values[name] = reward_values.get(name, 0) + value

                for name, value in info['reward_counts'].items():
                    reward_counts[name] = reward_counts.get(name, 0) + value

        next_log = self._logging.get('next_log', LOG_RATE)
        if iterations < next_log or not self._logging['total']:
            return

        self._logging['next_log'] = next_log + LOG_RATE
        total = self._logging['total']
        self.tensorboard.add_scalar('rollout/total-completed', total, iterations, curr)

        success_total = self._logging.get('success', 0)
        self.tensorboard.add_scalar('evaluation/success-rate', success_total / total, iterations, curr)

        if reward_list:
            self.tensorboard.add_scalar('evaluation/ep-reward-avg', np.mean(reward_list), iterations, curr)

        if scores:
            self.tensorboard.add_scalar('evaluation/score', np.mean(scores), iterations, curr)

        if total_seconds:
            self.tensorboard.add_scalar('rollout/seconds-per-episode', np.mean(total_seconds), iterations, curr)

        if steps:
            self.tensorboard.add_scalar('rollout/steps-per-episode', np.mean(steps), iterations, curr)

        endings_count = sum(endings.values())
        for key, value in endings.items():
            self.tensorboard.add_scalar(f'endings/{key}', value / endings_count, iterations, curr)
            endings[key] = 0

        for key, value in reward_values.items():
            self.tensorboard.add_scalar(f'rewards/{key}', value / total, iterations, curr)
            reward_values[key] = 0

        for key, value in reward_counts.items():
            self.tensorboard.add_scalar(f'reward-counts/{key}', value / total, iterations, curr)
            reward_counts[key] = 0

        self._logging.clear()
        self._logging['endings'] = endings
        self._logging['reward_values'] = reward_values
        self._logging['reward_counts'] = reward_counts

    def _optimize(self, network : Network, variables : PPORolloutBuffer, iterations : int):
        # pylint: disable=too-many-locals, too-many-statements

        # flatten observations
        if isinstance(variables.observation, dict):
            b_obs = {}
            for key, obs_part in variables.observation.items():
                part = obs_part[:, :variables.memory_length]
                b_obs[key] = part.reshape(-1, *obs_part.shape[2:]).to(self.device)
        else:
            part = variables.observation[:, :variables.memory_length]
            b_obs = part.reshape(-1, *part.shape[2:]).to(self.device)

        # flatten actions, logprobs, values, masks
        actions   = variables.act_logp_ent_val[:, :, 0].to(self.device)
        logprobs  = variables.act_logp_ent_val[:, :, 1].to(self.device)
        values    = variables.act_logp_ent_val[:, :, 3].to(self.device)

        b_actions  = actions.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_values   = values.reshape(-1)

        masks     = variables.masks
        b_masks    = masks.reshape(-1, masks.shape[-1]).to(self.device)

        # flatten returns, advantages
        b_advantages = variables.advantages.reshape(-1).to(self.device)
        b_returns    = variables.returns.reshape(-1).to(self.device)

        # standard PPO update
        batch_size = variables.memory_length * variables.n_envs
        minibatch_size = batch_size // self.minibatches

        network = network.to(self.device)
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
                if isinstance(b_obs, dict):
                    mb_obs = {}
                    for key, obs in b_obs.items():
                        mb_obs[key] = obs[mb_inds]
                else:
                    mb_obs = b_obs[mb_inds]

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

        network = network.to("cpu")
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

        return network
