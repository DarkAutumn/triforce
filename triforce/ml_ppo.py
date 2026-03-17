import math
import time
import torch
from torch import nn

from .metrics import MetricTracker
from .ml_ppo_callback import NullCallback
from .ml_ppo_rollout_buffer import PPORolloutBuffer
from .models import Network, create_network

# default hyperparameters
LEARNING_RATE = 0.0001
NORM_ADVANTAGES = True
CLIP_VAL_LOSS = False
GAMMA = 0.99
LAMBDA = 0.95
CLIP_COEFF = 0.2
ENT_COEFF = 0.01
VS_COEFF = 0.5
MAX_GRAD_NORM = 0.5
EPSILON = 1e-5
TARGET_STEPS = 4096
EPOCHS = 10
MINIBATCHES = 4
LOG_RATE = 25_000
SAVE_INTERVAL = 50_000
TARGET_KL = 0.02

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
    def __init__(self, **kwargs):
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
        self._minibatches_override = 'minibatches' in kwargs
        self.num_epochs = kwargs.get('num_epochs', EPOCHS)
        self._target_kl = kwargs.get('target_kl', TARGET_KL)
        self.optimizer = None

        self.kwargs = kwargs

        self.total_steps = 0

        self._logging = {}
        self.start_time = None
        self._steps_at_start = 0

    def train(self, network, create_env, iterations, callback=None, **kwargs):
        """Train the network."""
        self.start_time = time.time()
        if callback is None:
            callback = NullCallback()

        n_envs = kwargs.get('envs', 1)

        env = create_env()
        network = kwargs.get('model', None) or create_network(
            network, env.observation_space, env.action_space,
            model_kind=kwargs.get('model_kind'),
            action_space_name=kwargs.get('action_space_name_str'))
        self._steps_at_start = network.steps_trained
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, eps=self._epsilon)
        else:
            # Reattach optimizer to new parameter references while preserving momentum/variance
            for param_group, new_params in zip(self.optimizer.param_groups,
                                                [list(network.parameters())]):
                param_group['params'] = new_params

        if n_envs > 1:
            # Multi-env mode: close the initial env (workers create their own) and train
            env.close()
            return self._train_multi(network, n_envs, iterations, callback, **kwargs)

        try:
            return self._train_single(network, env, iterations, callback, **kwargs)
        finally:
            env.close()

    def _train_single(self, network, env, iterations, callback, **kwargs):
        buffer = PPORolloutBuffer(self.target_steps, 1, env.observation_space, env.action_space,
                                  self._gamma, self._lambda)

        save_path = kwargs.get('save_path', None)
        exit_criteria = kwargs.get('exit_criteria', None)
        exit_threshold = kwargs.get('exit_threshold', None)
        next_metrics = Threshold(LOG_RATE)
        next_model_save = Threshold(SAVE_INTERVAL)
        total_steps = math.ceil(iterations / buffer.memory_length) * buffer.memory_length
        total_iterations = 0

        while total_iterations < iterations:
            # Collect training data
            buffer.ppo_main_loop(0, network, env, callback, total_steps)
            total_iterations += buffer.memory_length

            # Save metrics
            if next_metrics.add(buffer.memory_length):
                network.metrics = MetricTracker.get_metrics_and_clear()
                if network.metrics:
                    callback.on_metrics(network.metrics, network.steps_trained, total_steps)

                    if exit_criteria and self._hit_exit_criteria(network.metrics, exit_criteria, exit_threshold):
                        break

            # Save model, hopefully log rate and save interval are multiples of each other
            if save_path and next_model_save.add(buffer.memory_length):
                model_name = kwargs.get('model_name', "network").replace(' ', '_')
                network.save(f"{save_path}/{model_name}_{network.steps_trained}.pt")

            # Optimize the network
            network.steps_trained += buffer.memory_length
            network = self._optimize(network, buffer, network.steps_trained, callback, total_steps)
            callback.check_pause()

        return network

    def _train_multi(self, network, n_envs, iterations, callback, **kwargs):
        """Train with multiple environments in separate subprocesses."""
        # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
        from .ml_ppo_worker import RolloutWorkerPool, EnvFactory  # pylint: disable=import-outside-toplevel

        obs_space = network.observation_space
        act_space = network.action_space
        steps_per_env = self.target_steps // n_envs
        buffer = PPORolloutBuffer(steps_per_env, n_envs, obs_space, act_space,
                                  self._gamma, self._lambda)
        buffer.share_memory_()

        # Use pre-built env_factory if provided, otherwise build from kwargs
        env_factory = kwargs.get('env_factory')
        if env_factory is None:
            scenario_def = kwargs.get('scenario_def')
            action_space_name = kwargs.get('action_space_name')
            env_kwargs = {k: v for k, v in kwargs.items()
                          if k in ('render_mode', 'translation', 'frame_stack', 'obs_kind', 'multihead')}
            env_factory = EnvFactory(scenario_def, action_space_name, **env_kwargs)

        network_class = kwargs.get('network_class', type(network))
        pool = RolloutWorkerPool(n_envs, buffer, network, env_factory, network_class)
        try:
            save_path = kwargs.get('save_path', None)
            exit_criteria = kwargs.get('exit_criteria', None)
            exit_threshold = kwargs.get('exit_threshold', None)
            next_metrics = Threshold(LOG_RATE)
            next_model_save = Threshold(SAVE_INTERVAL)

            # Each collection gathers target_steps total env steps (split across workers).
            # Count total env steps for iteration tracking, matching single-env behavior.
            env_steps_per_iteration = buffer.memory_length * n_envs
            total_steps = math.ceil(iterations / env_steps_per_iteration) * env_steps_per_iteration
            total_iterations = 0
            accumulated_metrics = []

            while total_iterations < iterations:
                # Send current weights to workers and collect rollouts
                pool.update_weights(network)
                _, worker_metrics = pool.collect_rollouts(buffer)
                total_iterations += env_steps_per_iteration
                callback.on_progress(env_steps_per_iteration, total_steps)
                if worker_metrics:
                    accumulated_metrics.append(worker_metrics)

                # Log aggregated metrics from worker subprocesses
                if next_metrics.add(env_steps_per_iteration):
                    network.metrics = self._average_metric_dicts(accumulated_metrics)
                    accumulated_metrics.clear()
                    if network.metrics:
                        callback.on_metrics(network.metrics, network.steps_trained, total_steps)

                        if exit_criteria and self._hit_exit_criteria(network.metrics, exit_criteria,
                                                                     exit_threshold):
                            break

                # Save model
                if save_path and next_model_save.add(env_steps_per_iteration):
                    model_name = kwargs.get('model_name', "network").replace(' ', '_')
                    network.save(f"{save_path}/{model_name}_{network.steps_trained}.pt")

                # Optimize the network
                network.steps_trained += env_steps_per_iteration
                network = self._optimize(network, buffer, network.steps_trained, callback, total_steps)
                callback.check_pause()

            return network
        finally:
            pool.close()

    def _hit_exit_criteria(self, metrics, exit_criteria, exit_threshold):
        return metrics.get(exit_criteria, 0) >= exit_threshold

    @staticmethod
    def _average_metric_dicts(dicts):
        """Average a list of metric dicts into one."""
        if not dicts:
            return {}
        combined = {}
        for d in dicts:
            for key, value in d.items():
                if key not in combined:
                    combined[key] = []
                combined[key].append(value)
        return {key: sum(values) / len(values) for key, values in combined.items()}

    def _optimize(self, network : Network, variables : PPORolloutBuffer, iterations : int,
                  callback=None, total_steps=0):
        # pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-arguments
        # pylint: disable=too-many-positional-arguments

        # flatten observations
        if isinstance(variables.observation, dict):
            b_obs = {}
            for key, obs_part in variables.observation.items():
                part = obs_part[:, :variables.memory_length]
                b_obs[key] = part.reshape(-1, *obs_part.shape[2:]).to(self.device)
        else:
            part = variables.observation[:, :variables.memory_length]
            b_obs = part.reshape(-1, *part.shape[2:]).to(self.device)

        # flatten actions, logprobs, values, masks — supports both Discrete and MultiDiscrete
        actions   = variables.actions.to(self.device)
        logprobs  = variables.logp_ent_val[:, :, 0].to(self.device)
        values    = variables.logp_ent_val[:, :, 2].to(self.device)

        if variables.action_dim == 1:
            b_actions = actions.reshape(-1)
        else:
            b_actions = actions.reshape(-1, variables.action_dim)
        b_logprobs = logprobs.reshape(-1)
        b_values   = values.reshape(-1)

        masks     = variables.masks[:, :variables.memory_length]
        b_masks    = masks.reshape(-1, masks.shape[-1]).to(self.device)

        # flatten returns, advantages
        b_advantages = variables.advantages.reshape(-1).to(self.device)
        b_returns    = variables.returns.reshape(-1).to(self.device)

        # standard PPO update
        batch_size = variables.memory_length * variables.n_envs
        minibatches = self.minibatches
        if not self._minibatches_override:
            minibatches = max(minibatches,
                              getattr(network, 'recommended_minibatches', minibatches))
        minibatch_size = batch_size // minibatches

        network = network.to(self.device)
        optimizer = self.optimizer

        b_inds = torch.arange(batch_size)
        clipfracs = []
        kl_exceeded = False
        for _ in range(self.num_epochs):
            if kl_exceeded:
                break

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

                if self._target_kl is not None and approx_kl > self._target_kl:
                    kl_exceeded = True
                    break

        network = network.to("cpu")

        # After training, compute stats like explained variance
        y_pred = b_values.cpu()
        y_true = b_returns.cpu()
        var_y = torch.var(y_true)
        explained_var = float('nan') if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y

        # Compute per-head entropy for MultiHeadAgent tensorboard logging
        per_head_entropy = {}
        if hasattr(network, 'get_entropy_details'):
            with torch.no_grad():
                # Use last minibatch obs/masks (still on device) — move to CPU for network
                detail_obs = {k: v.cpu() for k, v in mb_obs.items()} if isinstance(mb_obs, dict) else mb_obs.cpu()
                detail_masks = mb_masks.cpu()
                per_head_entropy = network.get_entropy_details(detail_obs, detail_masks)

        # Compute attention entropy for IMPALA models
        attention_stats = {}
        if hasattr(network, 'get_attention_entropy'):
            with torch.no_grad():
                if not per_head_entropy:
                    detail_obs = {k: v.cpu() for k, v in mb_obs.items()} \
                        if isinstance(mb_obs, dict) else mb_obs.cpu()
                attention_stats = network.get_attention_entropy(detail_obs)

        stats = {
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": torch.mean(torch.tensor(clipfracs)).item(),
            "losses/explained_variance": explained_var,
        }

        if self.start_time is not None:
            steps_this_run = iterations - self._steps_at_start
            sps = int(steps_this_run / (time.time() - self.start_time))
            stats["charts/SPS"] = sps

        for name, value in per_head_entropy.items():
            stats[f"losses/{name}"] = value

        for name, value in attention_stats.items():
            stats[f"losses/{name}"] = value

        if callback is not None:
            callback.on_optimize(stats, iterations, total_steps)

        return network
