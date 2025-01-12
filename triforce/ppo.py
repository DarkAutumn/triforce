from collections import Counter
import time
import multiprocessing
from multiprocessing import Queue, Process
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


from .rewards import StepRewards

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


class PPOSubprocess:
    """
    A persistent worker process that owns:
      - A single environment (from create_env())
      - A local PPO(n_envs=1)
      - Its own carry-over (obs, done, action_mask) state
    It listens on a queue for commands like:
      - 'build_batch': run build_one_batch, compute_returns, etc.
      - 'update_weights': update local PPO state_dict
      - 'close': terminate
    """

    def __init__(self, idx, create_env, ppo_class, ppo_kwargs):
        """
        idx: worker index
        create_env: callable that returns an environment
        ppo_class: the PPO class (or a factory function) for local instantiation
        ppo_kwargs: dictionary of arguments to init local PPO
        """
        self.idx = idx
        self.command_queue = Queue()
        self.result_queue = Queue()

        # We'll create a separate Process that runs self._run()
        self.process = Process(target=self._run, args=(create_env, ppo_class, ppo_kwargs))
        self.process.start()

    def _run(self, create_env, ppo_class, ppo_kwargs):
        """
        The target method running inside the worker process.
        """
        # 1) Create environment and local PPO(n_envs=1)
        env = create_env()
        local_ppo = ppo_class(**ppo_kwargs)  # e.g. PPO(network=..., device=..., n_envs=1, ...)
        local_ppo.n_envs = 1  # ensure single env in the worker

        # 2) Maintain carry-over state for the environment
        #    None means "we haven't started yet" => we reset in build_one_batch
        worker_state = None

        # 3) Loop, waiting for commands
        while True:
            cmd, data = self.command_queue.get()
            if cmd == 'build_batch':
                # data might be (iterations, progress, state_dict) or similar
                # if you want to sync weights:
                if 'weights' in data and data['weights'] is not None:
                    local_ppo.load_state_dict(data['weights'])

                progress = data.get('progress', None)
                iterations = data.get('iterations', 0)

                # Run build_one_batch + compute_returns
                # TODO: push returns/advs to the main process
                # TODO: implement loading/saving weights
                i = self.idx  # worker index
                infos, next_value, worker_state = local_ppo.build_one_batch(i, env, progress, worker_state)


                # Send results back
                self.result_queue.put((infos, next_value, local_ppo.obs, local_ppo.dones,
                                       local_ppo.act_logp_ent_val_mask, local_ppo.rewards))

            elif cmd == 'update_weights':
                # data == new_state_dict
                local_ppo.load_state_dict(data)
                self.result_queue.put("weights_updated")

            elif cmd == 'close':
                # Clean up
                env.close()
                break  # exit the loop => process ends

            else:
                print(f"[Worker {self.idx}] Unknown command: {cmd}")
                self.result_queue.put(None)

    def build_batch_async(self, iterations=None, progress=None, weights=None):
        """
        Asynchronously request that the worker build a batch of data.
        weights: optionally pass in main PPO's state_dict if you want to sync.
        """
        self.command_queue.put((
            'build_batch',
            {
                'iterations': iterations,
                'progress': progress,
                'weights': weights,  # optional
            }
        ))

    def get_result(self):
        """Blocking call to retrieve the last result from the worker."""
        return self.result_queue.get()

    def update_weights_async(self, new_weights):
        """Send a message to update the local PPO's weights."""
        self.command_queue.put(('update_weights', new_weights))

    def close(self):
        """Close the worker process."""
        self.command_queue.put(('close', None))
        self.process.join()


class PPO:
    """PPO Implementation.  Cribbed from https://www.youtube.com/watch?v=MEt6rrxH8W4."""
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
            self.optimize(batch_returns, batch_advantages, iteration)

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
                infos, next_value, obs, dones, act_logp_ent_val_mask, rewards = w.get_result()
                self.obs[0][i] = obs[0]
                self.obs[1][i] = obs[1]
                self.obs[2][i] = obs[2]
                self.dones[i] = dones
                self.rewards[i] = rewards
                self.act_logp_ent_val_mask[i] = act_logp_ent_val_mask
                returns, advantages = self._compute_returns(i, next_value)
                batch_returns[i] = returns
                batch_advantages[i] = advantages
                infos.extend(infos)

            iteration += self.n_envs * self.memory_length

            # 4) update the aggregator / logs
            self._batch_update(infos)

            # 5) do the global optimize
            self.optimize(batch_returns, batch_advantages, iteration)

            # 6) Optionally push the updated weights to each worker if you want new policy rollout
            # for w in workers:
            #     w.update_weights_async(self.state_dict())
            #     ack = w.get_result()

        # 7) Cleanup
        for w in workers:
            w.close()

    def build_one_batch(self, batch_index, env, progress, state=None):
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

    def _batch_update(self, infos):
        # pylint: disable=too-many-branches
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
        """
        returns, advantages: shape [n_envs, memory_length]
        self.obs: a tuple of 3 tensors, each shape [n_envs, memory_length+1, ...],
                but we only want the first memory_length steps for each rollout.
        self.act_logp_ent_val_mask: shape [n_envs, memory_length, 5]
            typically index 0=action, 1=logprob, 2=entropy, 3=value, 4=any other mask
        """

        # pylint: disable=too-many-locals, too-many-statements

        # -----------------------------
        # 1) Flatten Observations
        # -----------------------------
        # Each obs component: shape [n_envs, memory_length, ...].
        # We'll flatten along (n_envs * memory_length).

        # obs_image: [n_envs, memory_length, 1, viewport_size, viewport_size]
        b_obs_image = self.obs[0][:, :self.memory_length]  # discard the +1
        b_obs_image = b_obs_image.reshape(-1, 1, self.network.viewport_size, self.network.viewport_size)

        # obs_vectors: [n_envs, memory_length, vectors_size[0], vectors_size[1], vectors_size[2]]
        b_obs_vectors = self.obs[1][:, :self.memory_length]
        b_obs_vectors = b_obs_vectors.reshape(-1, self.network.vectors_size[0], self.network.vectors_size[1],
                                            self.network.vectors_size[2])

        # obs_features: [n_envs, memory_length, info_size]
        b_obs_features = self.obs[2][:, :self.memory_length]
        b_obs_features = b_obs_features.reshape(-1, self.network.info_size)

        # -----------------------------
        # 2) Flatten Actions, Logprobs, Values
        # -----------------------------
        # self.act_logp_ent_val_mask has shape [n_envs, memory_length, 5].
        # Let’s define each index carefully:
        #   0 -> action
        #   1 -> logp
        #   2 -> entropy
        #   3 -> value
        #   4 -> mask
        actions   = self.act_logp_ent_val_mask[:, :, 0]
        logprobs  = self.act_logp_ent_val_mask[:, :, 1]
        values    = self.act_logp_ent_val_mask[:, :, 3]
        masks     = self.act_logp_ent_val_mask[:, :, 4]

        b_actions  = actions.reshape(-1)   # [n_envs*memory_length]
        b_logprobs = logprobs.reshape(-1)  # [n_envs*memory_length]
        b_values   = values.reshape(-1)    # [n_envs*memory_length]
        b_masks    = masks.reshape(-1)     # [n_envs*memory_length]

        # -----------------------------
        # 3) Flatten advantages, returns
        # -----------------------------
        # returns, advantages: shape [n_envs, memory_length]
        b_advantages = advantages.reshape(-1)
        b_returns    = returns.reshape(-1)

        # (At this point, b_obs_xxx have shape [n_envs*memory_length, ...],
        #  while b_actions, b_logprobs, b_values, b_returns, b_advantages
        #  each have shape [n_envs*memory_length].)

        # -----------------------------
        # 4) Standard PPO update loop
        # -----------------------------
        b_inds = np.arange(self.batch_size)  # Typically batch_size = n_envs*memory_length
        clipfracs = []
        for _ in range(self.num_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                # Slice each item for this mini-batch
                mb_obs = b_obs_image[mb_inds], b_obs_vectors[mb_inds], b_obs_features[mb_inds]
                mb_actions   = b_actions[mb_inds].long()
                mb_logprobs  = b_logprobs[mb_inds]
                mb_masks     = b_masks[mb_inds]
                mb_values    = b_values[mb_inds]
                mb_returns   = b_returns[mb_inds]
                mb_adv       = b_advantages[mb_inds]

                _, newlogprob, entropy, newvalue = self.network.get_action_and_value(mb_obs, mb_actions, mb_masks)

                # ratio etc.
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > CLIP_COEFF).float().mean().item())

                # Normalize advantages if desired
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
