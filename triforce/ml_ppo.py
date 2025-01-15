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
from .rewards import TotalRewards

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
SAVE_INTERVAL = 40_000

class Threshold:
    """A counter class to see if we've reached our intervals."""
    def __init__(self, limit):
        self.limit = limit
        self.current = 0

    def add(self, value):
        """Add a value to the counter, returning True if we've reached the limit."""
        self.current += value
        if self.current > self.limit:
            self.current = self.current - self.limit
            return True

        return False

class PPO:
    """PPO Implementation.  Adapted from from https://www.youtube.com/watch?v=MEt6rrxH8W4."""
    def __init__(self, device, log_dir, **kwargs):
        self.target_steps = kwargs.get('target_steps', TARGET_STEPS)
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

    def train(self, network, create_env, iterations, progress=None, **kwargs):
        """Train the network."""
        self.start_time = time.time()

        env = create_env()
        network = create_network(network, env.observation_space, env.action_space)

        envs = kwargs.get('envs', 1)

        if envs == 1:
            return self._train_single(network, env, iterations, progress, **kwargs)

        return self._train_multiproc(network, env, create_env, iterations, progress, **kwargs)

    def _train_single(self, network, env, iterations, progress, **kwargs):
        # tracking
        save_path = kwargs.get('save_path', None)
        next_update = Threshold(LOG_RATE)
        next_save = Threshold(SAVE_INTERVAL)
        rewards = TotalRewards()
        reward_stats = None

        memory_length = self.target_steps
        buffer = PPORolloutBuffer(memory_length, 1, env.observation_space, env.action_space,
                                  self._gamma, self._lambda)

        total_iterations = 0
        while total_iterations < iterations:
            infos = buffer.ppo_main_loop(0, network, env, progress)

            total_iterations += buffer.memory_length
            self._process_infos(rewards, infos)

            if next_update.add(buffer.memory_length):
                reward_stats = rewards.get_stats_and_clear()
                reward_stats.to_tensorboard(self.tensorboard, total_iterations)
                network.stats = reward_stats

            if save_path and next_save.add(buffer.memory_length):
                network.save(f"{save_path}/network_{total_iterations}.pt")

            network = self._optimize(network, buffer, total_iterations)
            network.steps_trained += buffer.memory_length

        env.close()
        return network

    def _train_multiproc(self, network, env, create_env, iterations, progress, **kwargs):
        # pylint: disable=too-many-locals
        envs = kwargs['envs']

        # tracking
        save_path = kwargs.get('save_path', None)
        next_update = Threshold(LOG_RATE)
        next_save = Threshold(SAVE_INTERVAL)
        rewards = TotalRewards()
        reward_stats = None

        # ppo variables
        memory_length = self.target_steps // envs
        variables = PPORolloutBuffer(memory_length, envs, env.observation_space, env.action_space,
                                     self._gamma, self._lambda)
        env.close()

        result_queue = Queue()
        kwargs = {
                'gamma': self._gamma,
                'lambda': self._lambda,
                'steps': variables.memory_length,
                }

        workers = [SubprocessWorker(idx, create_env, network, result_queue, kwargs) for idx in range(envs)]
        try:
            steps = math.ceil(iterations / (envs * memory_length))
            for step in range(steps):
                total_iterations = step * envs * memory_length
                iterations_processed = envs * memory_length

                # Process the results from the workers
                infos = self._subprocess_ppo(network, progress, variables, result_queue, workers)
                self._process_infos(rewards, infos)

                # Update reward stats and save the network if requested, this should be done before optimization
                if next_update.add(iterations_processed):
                    reward_stats = rewards.get_stats_and_clear()
                    reward_stats.to_tensorboard(self.tensorboard, total_iterations)
                    network.stats = reward_stats

                if save_path and next_save.add(iterations_processed) and reward_stats:
                    model_name = kwargs.get('model_name', 'network')
                    network.save(f"{save_path}/{model_name}-{total_iterations:_}.pt")

                # Update the network
                self._optimize(network, variables, step * envs * memory_length)
                network.steps_trained += iterations_processed

        except EOFError:
            pass

        finally:
            # Tell workers to shut down regardless
            for worker in workers:
                worker.close_async()

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

    def _process_infos(self, total_rewards : TotalRewards, infos):
        for info in infos:
            if (ep_rewards := info.get('episode_rewards', None)) is not None:
                total_rewards.add(ep_rewards)

    def _optimize(self, network : Network, variables : PPORolloutBuffer, iterations : int):
        # pylint: disable=too-many-locals, too-many-statements, too-many-branches

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
