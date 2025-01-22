import math
import time
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .metrics import MetricTracker
from .ml_ppo_rollout_buffer import PPORolloutBuffer
from .models import Network, create_network

# default hyperparameters
LEARNING_RATE_MAX = 0.00025
LEARNING_RATE_HIGH = 0.00015
LEARNING_RATE_MEDIUM = 0.0001
LEARNING_RATE_MINIMUM = 0.000025
NORM_ADVANTAGES = True
CLIP_VAL_LOSS = True
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
LOG_RATE = 25_000
SAVE_INTERVAL = 25_000

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
    def __init__(self, log_dir, **kwargs):
        self.device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_steps = kwargs.get('target_steps', TARGET_STEPS)
        self._norm_advantages = kwargs.get('norm_advantages', NORM_ADVANTAGES)
        self._clip_val_loss = kwargs.get('clip_val_loss', CLIP_VAL_LOSS)
        self._gamma = kwargs.get('gamma', GAMMA)
        self._lambda = kwargs.get('lambda', LAMBDA)
        self._clip_coeff = kwargs.get('clip_coeff', CLIP_COEFF)
        self._ent_coeff = kwargs.get('ent_coeff', ENT_COEFF)
        self._vf_coeff = kwargs.get('vf_coeff', VS_COEFF)
        self._max_grad_norm = kwargs.get('max_grad_norm', MAX_GRAD_NORM)
        self._epsilon = kwargs.get('epsilon', EPSILON)
        self.minibatches = kwargs.get('minibatches', MINIBATCHES)
        self.num_epochs = kwargs.get('num_epochs', EPOCHS)
        self.optimizer = None

        self.kwargs = kwargs

        self.log_dir = log_dir
        self.tensorboard = SummaryWriter(log_dir) if log_dir else None

        self.total_steps = 0

        self._logging = {}
        self.start_time = None

    def train(self, network, create_env, iterations, progress=None, **kwargs):
        """Train the network."""
        self.start_time = time.time()

        env = create_env()
        try:
            network = kwargs.get('model', None) or create_network(network, env.observation_space, env.action_space)

            if kwargs.get('dynamic_lr', False):
                self.optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE_MEDIUM, eps=self._epsilon)
            else:
                self.optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE_MAX, eps=self._epsilon)

            envs = kwargs.get('envs', 1)
            if envs > 1:
                raise NotImplementedError("Multiprocessing not yet implemented.")

            return self._train_single(network, env, iterations, progress, **kwargs)
        finally:
            env.close()

    def _train_single(self, network, env, iterations, progress, **kwargs):
        buffer = PPORolloutBuffer(self.target_steps, 1, env.observation_space, env.action_space,
                                  self._gamma, self._lambda)

        save_path = kwargs.get('save_path', None)
        exit_criteria = kwargs.get('exit_criteria', None)
        exit_threshold = kwargs.get('exit_threshold', None)
        next_tensorboard = Threshold(LOG_RATE)
        next_model_save = Threshold(SAVE_INTERVAL)
        progress.total = math.ceil(iterations / buffer.memory_length) * buffer.memory_length
        total_iterations = 0

        while total_iterations < iterations:
            # Collect training data
            buffer.ppo_main_loop(0, network, env, progress)
            total_iterations += buffer.memory_length

            # Save metrics
            if next_tensorboard.add(buffer.memory_length):
                network.metrics = MetricTracker.get_metrics_and_clear()
                if network.metrics:
                    self._write_metrics(network.metrics, network.steps_trained)

                    if exit_criteria and self._hit_exit_criteria(network.metrics, exit_criteria, exit_threshold):
                        break

                    if kwargs.get('dynamic_lr', False):
                        self._adjust_learning_rate(network.metrics)

            # Save model, hopefully log rate and save interval are multiples of each other
            if save_path and next_model_save.add(buffer.memory_length):
                network.save(f"{save_path}/network_{network.steps_trained}.pt")

            # Optimize the network
            network.steps_trained += buffer.memory_length
            network = self._optimize(network, buffer, network.steps_trained)

        return network

    def _hit_exit_criteria(self, metrics, exit_criteria, exit_threshold):
        return metrics.get(exit_criteria, 0) >= exit_threshold

    def _write_metrics(self, metrics, total_iterations):
        timestamp = time.time()
        for name, value in metrics.items():
            if '/' not in name:
                name = f"metrics/{name}"

            self.tensorboard.add_scalar(name, value, total_iterations, timestamp)

        if metrics:
            self.tensorboard.flush()

    def _adjust_learning_rate(self, metrics):
        success_rate = metrics.get("success-rate", None)
        assert success_rate is not None, "Attempted to adjust learning rate without success rate."

        # Example logic:
        #  - if success_rate == 0 => high LR
        #  - If success_rate < 0.2 => elevated LR
        #  - If success_rate < 0.95 => moderate LR
        #  - Else => very low LR
        if success_rate < 0.01:
            new_lr = LEARNING_RATE_MAX
        elif success_rate < 0.2:
            new_lr = LEARNING_RATE_HIGH
        elif success_rate < 0.95:
            new_lr = LEARNING_RATE_MEDIUM
        else:
            new_lr = LEARNING_RATE_MINIMUM

        assert self.optimizer, "Attempted to adjust learning rate before optimizer was created."
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

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
        optimizer = self.optimizer

        b_inds = torch.arange(batch_size)
        clipfracs = []
        for _ in range(self.num_epochs):
            b_inds = b_inds[torch.randperm(batch_size)]
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

        # After training, compute stats like explained variance
        y_pred = b_values.cpu()
        y_true = b_returns.cpu()
        var_y = torch.var(y_true)
        explained_var = float('nan') if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y

        if self.tensorboard:
            self.tensorboard.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iterations)
            self.tensorboard.add_scalar("losses/value_loss", v_loss.item(), iterations)
            self.tensorboard.add_scalar("losses/policy_loss", pg_loss.item(), iterations)
            self.tensorboard.add_scalar("losses/entropy", entropy_loss.item(), iterations)
            self.tensorboard.add_scalar("losses/old_approx_kl", old_approx_kl.item(), iterations)
            self.tensorboard.add_scalar("losses/approx_kl", approx_kl.item(), iterations)
            self.tensorboard.add_scalar("losses/clipfrac", torch.mean(torch.tensor(clipfracs)).item(), iterations)
            self.tensorboard.add_scalar("losses/explained_variance", explained_var, iterations)
            if self.start_time is not None:
                self.tensorboard.add_scalar("charts/SPS", int(iterations / (time.time() - self.start_time)), iterations)

        return network
