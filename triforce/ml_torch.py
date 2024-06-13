#!/usr/bin/python

import math
import torch
import numpy as np
from torch import nn
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

VIEWPORT_SIZE = 128
FEATURE_HIDDEN_SIZE = 64

CONV_HIDDEN_SIZE = 512
COMBINED_SIZE = FEATURE_HIDDEN_SIZE + CONV_HIDDEN_SIZE
SHARED_LAYER_SIZE = 128
DANGER_SENSE_HIDDEN_SIZE = 0
PATHFINDER_HIDDEN_SIZE = 0
ACTION_HIDDEN_SIZE = 0

#pylint: disable=missing-class-docstring
#pylint: disable=missing-function-docstring

class ZeldaMultiHeadNetwork(nn.Module):
    def __init__(self, viewport_size, num_features, device, **kwargs):
        super().__init__()

        self.conv_hidden_size = kwargs.get("conv-hidden-size", CONV_HIDDEN_SIZE)
        self.feature_hidden_size = kwargs.get("feature-hidden-size", FEATURE_HIDDEN_SIZE)
        self.combined_size = self.feature_hidden_size + self.conv_hidden_size
        self.shared_layer_size = kwargs.get("shared-layer-size", SHARED_LAYER_SIZE)
        self.pathfinder_hidden_size = kwargs.get("pathfinder-hidden-size", PATHFINDER_HIDDEN_SIZE)
        self.danger_sense_hidden_size = kwargs.get("danger-sense-hidden-size", DANGER_SENSE_HIDDEN_SIZE)
        self.action_hidden_size = kwargs.get("action-selector-hidden-size", ACTION_HIDDEN_SIZE)

        self.trained_steps = 0

        self.viewport_size = viewport_size
        self.num_features = num_features
        self.device = device

        self.heads = 3
        self.obs_shape = [(1, viewport_size, viewport_size), (num_features,)]

        # Convolutional layers for image data
        conv_size = self.__conv_output_size(viewport_size, 8, 4)
        conv_size = self.__conv_output_size(conv_size, 4, 2)
        conv_size = self.__conv_output_size(conv_size, 3, 1)
        conv_input = 64 * conv_size * conv_size

        self.conv_layer = nn.Sequential(
            self.__init_layer(nn.Conv2d(1, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            self.__init_layer(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            self.__init_layer(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.__init_layer(nn.Linear(conv_input, self.conv_hidden_size)),
            nn.ReLU()
        )

        # Fully connected layers for feature data
        self.feature_layer = nn.Sequential(
            self.__init_layer(nn.Linear(num_features, self.feature_hidden_size)),
            nn.ReLU()
        )

        # combined layers
        self.shared_layer = nn.Sequential(
            self.__init_layer(nn.Linear(self.combined_size, self.shared_layer_size)),
            nn.ReLU()
        )

        # Head for pathfinding
        if self.pathfinder_hidden_size:
            self.pathfinder = nn.Sequential(
                self.__init_layer(nn.Linear(self.shared_layer_size, self.pathfinder_hidden_size)),
                nn.ReLU(),
                self.__init_layer(nn.Linear(self.pathfinder_hidden_size, 4), std=0.01)
            )
            self.pathfinder_critic = nn.Sequential(
                self.__init_layer(nn.Linear(self.shared_layer_size, self.pathfinder_hidden_size)),
                nn.ReLU(),
                self.__init_layer(nn.Linear(self.pathfinder_hidden_size, 1), std=1.0)
            )

        else:
            self.pathfinder = self.__init_layer(nn.Linear(self.shared_layer_size, 4), std=0.01)
            self.pathfinder_critic = self.__init_layer(nn.Linear(self.shared_layer_size, 1), std=1.0)

        # Head for danger sense

        if self.danger_sense_hidden_size:
            self.danger_sense = nn.Sequential(
                self.__init_layer(nn.Linear(self.shared_layer_size, self.danger_sense_hidden_size)),
                nn.ReLU(),
                self.__init_layer(nn.Linear(self.danger_sense_hidden_size, 5), std=0.01)
            )
            self.danger_sense_critic = nn.Sequential(
                self.__init_layer(nn.Linear(self.shared_layer_size, self.danger_sense_hidden_size)),
                nn.ReLU(),
                self.__init_layer(nn.Linear(self.danger_sense_hidden_size, 1), std=1.0)
            )
        else:
            self.danger_sense = self.__init_layer(nn.Linear(self.shared_layer_size, 5), std=0.01)
            self.danger_sense_critic = self.__init_layer(nn.Linear(self.shared_layer_size, 1), std=1.0)

        # Head for action selector
        if self.action_hidden_size:
            self.action_selector = nn.Sequential(
                self.__init_layer(nn.Linear(self.shared_layer_size, self.action_hidden_size)),
                nn.ReLU(),
                self.__init_layer(nn.Linear(self.action_hidden_size, 3), std=0.01)
            )
            self.action_selector_critic = nn.Sequential(
                self.__init_layer(nn.Linear(self.shared_layer_size, self.action_hidden_size)),
                nn.ReLU(),
                self.__init_layer(nn.Linear(self.action_hidden_size, 1), std=1.0)
            )
        else:
            self.action_selector = self.__init_layer(nn.Linear(self.shared_layer_size, 3), std=0.01)
            self.action_selector_critic = self.__init_layer(nn.Linear(self.shared_layer_size, 1), std=1.0)

        self.to(device)

    def __conv_output_size(self, input_size, kernel_size, stride, padding=0):
        return (input_size - kernel_size + 2 * padding) // stride + 1

    def __init_layer(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)

        return layer

    def forward(self, image, features):
        shared, combat = self._forward_steps(image, features)

        logits = [
            self.pathfinder(shared),
            self.danger_sense(combat),
            self.action_selector(combat)
        ]

        critic = [
            self.pathfinder_critic(shared),
            self.danger_sense_critic(combat),
            self.action_selector_critic(combat)
        ]

        return logits, critic

    def _forward_steps(self, image, features):
        image = image.to(self.device)
        features = features.to(self.device)

        image_data = self.conv_layer(image)
        feature_data = self.feature_layer(features)

        combined = torch.cat((image_data, feature_data), dim=1)
        shared = self.shared_layer(combined)

        return shared, shared

    def get_value(self, image, features):
        shared, combat = self._forward_steps(image, features)

        return torch.stack([
            self.pathfinder_critic(shared),
            self.danger_sense_critic(combat),
            self.action_selector_critic(combat)
        ], dim=2).to(self.device)

    def get_action(self, image, features):
        shared, combat = self._forward_steps(image, features)

        logits = [
            self.pathfinder(shared),
            self.danger_sense(combat),
            self.action_selector(combat)
        ]

        dists = [torch.distributions.Categorical(logits=l) for l in logits]
        actions = [dist.sample() for dist in dists]

        return torch.stack(actions, dim=1).to(self.device)

    def get_act_logp_ent_val(self, image, features, actions = None):
        shared, combat = self._forward_steps(image, features)

        logits = [
            self.pathfinder(shared),
            self.danger_sense(combat),
            self.action_selector(combat)
        ]

        values = [
            self.pathfinder_critic(shared).squeeze(1),
            self.danger_sense_critic(combat).squeeze(1),
            self.action_selector_critic(combat).squeeze(1)
        ]

        dists = [torch.distributions.Categorical(logits=l) for l in logits]
        if actions is None:
            actions = [dist.sample() for dist in dists]
            actions_tensor = torch.stack(actions, dim=1).to(self.device)
            logprob_tensor = torch.stack([dist.log_prob(a) for dist, a in zip(dists, actions)], dim=1)
            logprob_tensor = logprob_tensor.to(self.device)
        else:
            actions_tensor = actions

            logprobs = []
            for i in range(self.heads):
                dist = dists[i]
                action = actions[:, i]
                logprob = dist.log_prob(action)
                logprobs.append(logprob)

            logprob_tensor = torch.stack(logprobs).to(self.device).T

        entropy_tensor = torch.stack([dist.entropy() for dist in dists], dim=1).to(self.device)
        value_tensor = torch.stack(values, dim=1).to(self.device)

        act_logp_ent_val = torch.stack([actions_tensor, logprob_tensor, entropy_tensor, value_tensor], dim=2)
        return act_logp_ent_val.to(self.device)

class MultiHeadPPO:
    def __init__(self, network : ZeldaMultiHeadNetwork, device, train_callback = None):
        self.network = network
        self.device = device
        self.train_callback = train_callback

        self.optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, eps=1e-5)

        self.memory_length = 4096
        self.batch_size = 128
        self.minibatches = 4
        self.minibatch_size = self.batch_size // self.minibatches
        self.num_epochs = 4

        self.total_steps = 0

        self.obs_image = torch.empty(self.memory_length + 1, 1, network.viewport_size,
                                     network.viewport_size, device=device)
        self.obs_features = torch.empty(self.memory_length + 1, network.num_features, device=device)

        self.dones = torch.empty(self.memory_length + 1, dtype=torch.float32, device=device)

        self.act_logp_ent_val = torch.empty(self.memory_length, self.network.heads, 4, device=device)
        self.rewards = torch.empty(self.memory_length, self.network.heads, dtype=torch.float32, device=device)

        self.advantages = torch.empty(self.memory_length, self.network.heads, dtype=torch.float32, device=device)
        self.returns = torch.empty(self.memory_length, self.network.heads, dtype=torch.float32, device=device)

    def _obs_to_batched(self, obs):
        image, features = obs
        image = image.unsqueeze(0).to(self.device)
        features = features.flatten().unsqueeze(0).to(self.device)
        return image, features

    def train(self, env, iterations, epochs=4):
        next_obs, _ = env.reset()
        next_obs = self._obs_to_batched(next_obs)
        next_done = 0.0

        max_steps = math.ceil(iterations / self.memory_length) * self.memory_length
        progress = tqdm(total=max_steps, desc='Training', unit='steps')

        while self.total_steps < iterations:
            with torch.no_grad():
                for i in range(self.memory_length):
                    self.total_steps += 1

                    act_logp_ent_val = self.network.get_act_logp_ent_val(next_obs[0], next_obs[1])
                    self.act_logp_ent_val[i] = act_logp_ent_val[0]
                    actions = act_logp_ent_val[0, :, 0]
                    obs, reward, terminated, truncated, _ = env.step(actions)

                    obs = self._obs_to_batched(obs)
                    done = 1.0 if terminated or truncated else 0.0

                    self.obs_image[i] = next_obs[0]
                    self.obs_features[i] = next_obs[1]
                    self.dones[i] = next_done
                    self.rewards[i] = torch.tensor(reward, dtype=torch.float32, device=self.device)

                    next_obs = obs
                    next_done = done
                    progress.update(1)

                self.dones[self.memory_length] = next_done
                next_value = self.network.get_value(next_obs[0], next_obs[1])
                self._update_tensors(next_value[0])


            self._optimize(epochs)
            if self.train_callback and self.train_callback(self.total_steps, self):
                break

        progress.close()

    def _update_tensors(self, last_value):
        self.advantages = torch.zeros_like(self.advantages).to(self.device)
        self.returns = torch.zeros_like(self.returns).to(self.device)

        with torch.no_grad():
            last_gae = 0
            for t in reversed(range(self.memory_length)):
                mask = 1.0 - self.dones[t]
                next_value = self.act_logp_ent_val[t + 1, :, 3] if t + 1 < self.memory_length else last_value

                delta = self.rewards[t, :] + GAMMA * next_value * mask - self.act_logp_ent_val[t, :, 3]
                self.advantages[t, :] = last_gae = delta + GAMMA * LAMBDA * mask * last_gae

            self.returns = self.advantages + self.act_logp_ent_val[:, :, 3]

    def _optimize(self, epochs):
        b_obs_image = self.obs_image
        b_obs_features = self.obs_features
        b_actions = self.act_logp_ent_val[:, :, 0]
        b_logprobs = self.act_logp_ent_val[:, :, 1]
        b_advantages = self.advantages
        b_returns = self.returns
        b_values = self.act_logp_ent_val[:, :, 3]

        b_inds = np.arange(self.memory_length)
        for _ in range(epochs):
            np.random.shuffle(b_inds)

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_inds = b_inds[start:end]


                actions = b_actions.long()[minibatch_inds]
                act_logp_ent_val = self.network.get_act_logp_ent_val(b_obs_image[minibatch_inds],
                                                                     b_obs_features[minibatch_inds],
                                                                     actions)

                new_logprobs = act_logp_ent_val[:, :, 1]
                entropy = act_logp_ent_val[:, :, 2]
                new_values = act_logp_ent_val[:, :, 3]

                logratio = new_logprobs - b_logprobs[minibatch_inds]
                ratio = logratio.exp()

                mb_adv = b_advantages[minibatch_inds]
                if NORM_ADVANTAGES:
                    mean = mb_adv.mean(dim=0, keepdim=True)
                    std = mb_adv.std(dim=0, keepdim=True)
                    mb_adv = (mb_adv - mean) / (std + 1e-8)

                # policy loss
                loss1 = -mb_adv * ratio
                loss2 = -mb_adv * torch.clamp(ratio, 1.0 - CLIP_COEFF, 1.0 + CLIP_COEFF)
                policy_loss = torch.max(loss1, loss2)
                policy_loss = policy_loss.mean(dim=0, keepdim=True)

                # value loss
                mb_value = new_values

                if CLIP_VAL_LOSS:
                    v_loss_unclipped = (mb_value - b_returns[minibatch_inds]) ** 2
                    v_loss_clipped = b_values[minibatch_inds] + torch.clamp(
                        mb_value - b_values[minibatch_inds],
                        -CLIP_COEFF, CLIP_COEFF)

                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean(dim=0, keepdim=True)

                else:
                    v_loss = 0.5 * ((mb_value - b_returns[minibatch_inds]) ** 2).mean(dim=0, keepdim=True)

                # entropy loss
                entropy_loss = entropy.mean(dim=0, keepdim=True)

                # total loss
                loss = policy_loss.mean() - ENT_COEFF * entropy_loss.mean() + v_loss.mean() * VF_COEFF

                # backwards
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

        self.network.trained_steps += self.memory_length

__all__ = [MultiHeadPPO.__name__, ZeldaMultiHeadNetwork.__name__]
