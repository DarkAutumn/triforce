#!/usr/bin/python

from collections import Counter
from enum import Enum
import math
import torch
import numpy as np
from torch import nn
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

class ActionLayout(Enum):
    """The layout of the action space."""
    PATHFINDING = 0
    DANGER_SENSE = 1
    SELECTED_ACTION = 2

    @staticmethod
    def decode(action : torch.tensor):
        """Decodes the action tensor into a list of actions."""
        actions = action.tolist()
        return SelectedDirection(actions[0]), SelectedDirection(actions[1]), SelectedAction(actions[2])

class SelectedAction(Enum):
    """Enumeration of possible actions."""
    MOVEMENT = 0
    ATTACK = 1
    BEAMS = 2

class SelectedDirection(Enum):
    """Enumeration of possible directions."""
    E = 0
    W = 1
    S = 2
    N = 3
    NONE = 4

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

    def get_action(self, image, features, masks = None):
        shared, combat = self._forward_steps(image, features)

        # Choose the pathfinding (movement) direction
        pathfinding_logits = self._get_pathfinding_logits(masks, shared)
        pathfinding_direction = torch.distributions.Categorical(logits=pathfinding_logits).sample()

        # Choose the action the agent will take
        selected_action_logits = self._get_selected_action_logits(masks, combat)
        selected_action = torch.distributions.Categorical(logits=selected_action_logits).sample()

        # Choose danger sense direction. If we choose to attack, we cannot chose no danger direction
        danger_sense_logits = self._get_danger_sense_logits(masks, combat, selected_action)
        danger_sense_direction = torch.distributions.Categorical(logits=danger_sense_logits).sample()

        return torch.stack([pathfinding_direction, danger_sense_direction, selected_action], dim=1).to(self.device)

    def get_act_logp_ent_val(self, image, features, actions = None, masks = None):
        shared, combat = self._forward_steps(image, features)

        # Choose the pathfinding (movement) direction
        pathfinding_logits = self._get_pathfinding_logits(masks, shared)
        pathfinding_dists = torch.distributions.Categorical(logits=pathfinding_logits)
        pathfinding_direction = pathfinding_dists.sample() if actions is None else actions[:, 0]

        # Choose the action the agent will take
        selected_action_logits = self._get_selected_action_logits(masks, combat)
        selected_action_dists = torch.distributions.Categorical(logits=selected_action_logits)
        selected_action = selected_action_dists.sample() if actions is None else actions[:, 2]

        # Choose danger sense direction. If we choose to attack, we cannot chose no danger direction
        danger_sense_logits = self._get_danger_sense_logits(masks, combat, selected_action)

        danger_sense_dists = torch.distributions.Categorical(logits=danger_sense_logits)
        danger_sense_direction = danger_sense_dists.sample() if actions is None else actions[:, 1]

        value_tensor = torch.stack([self.pathfinder_critic(shared).squeeze(1),
                                    self.danger_sense_critic(combat).squeeze(1),
                                    self.action_selector_critic(combat).squeeze(1)
                                    ], dim=1).to(self.device)

        dists = [pathfinding_dists, danger_sense_dists, selected_action_dists]
        if actions is None:
            actions = [pathfinding_direction, danger_sense_direction, selected_action]
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

            logprob_tensor = torch.stack(logprobs, dim=1).to(self.device)

        entropy_tensor = torch.stack([dist.entropy() for dist in dists], dim=1).to(self.device)
        act_logp_ent_val = torch.stack([actions_tensor, logprob_tensor, entropy_tensor, value_tensor], dim=2)
        assert actions_tensor[0, ActionLayout.SELECTED_ACTION.value] == SelectedAction.MOVEMENT.value \
             or actions_tensor[0, ActionLayout.DANGER_SENSE.value] != SelectedDirection.NONE.value
        return act_logp_ent_val.to(self.device)

    def _get_pathfinding_logits(self, masks, shared):
        logits = self.pathfinder(shared)
        mask = None if masks is None else masks[ActionLayout.PATHFINDING.value]
        if mask is not None:
            logits += (1 - mask.to(self.device)) * -1e9
        return logits

    def _get_selected_action_logits(self, masks, combat):
        logits = self.action_selector(combat)
        mask = None if masks is None else masks[ActionLayout.SELECTED_ACTION.value]
        if mask is not None:
            logits += (1 - mask.to(self.device)) * -1e9
        return logits

    def _get_danger_sense_logits(self, masks, combat, selected_action):
        batch_size = selected_action.shape[0]

        if masks is None:
            mask = torch.ones((batch_size, 5), dtype=torch.float32)

        else:
            # Use the provided mask for the DANGER_SENSE action
            mask = masks[ActionLayout.DANGER_SENSE.value]
            if mask.ndimension() == 1:
                mask = mask.unsqueeze(0).expand(batch_size, -1)

        # Iterate through the batch and set the NONE direction to 0 where selected_action matches MOVEMENT
        for i in range(batch_size):
            if selected_action[i] != SelectedAction.MOVEMENT.value:
                mask[i, SelectedDirection.NONE.value] = 0.0

        logits = self.danger_sense(combat)
        logits += (1 - mask.to(self.device)) * -1e9
        return logits

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        data = torch.load(path)
        self.load_state_dict(data)

class MultiHeadPPO:
    def __init__(self, network : ZeldaMultiHeadNetwork, device, train_callback = None,
                 tensorboard_dir = None):
        self.network = network
        self.device = device
        self.train_callback = train_callback
        self.tensorboard = SummaryWriter(tensorboard_dir) if tensorboard_dir else None

        self.optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, eps=1e-5)

        self.memory_length = 4096
        self.batch_size = 128
        self.minibatches = 4
        self.minibatch_size = self.batch_size // self.minibatches
        self.num_epochs = 4
        self.total_steps = 0

        self.stats = {}

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
        ep_steps = 0
        next_obs = self._obs_to_batched(next_obs)
        next_done = 0.0
        next_mask = None

        max_steps = math.ceil(iterations / self.memory_length) * self.memory_length
        progress = tqdm(total=max_steps, desc='Training', unit='steps')

        stats = {}

        while self.total_steps < iterations:
            with torch.no_grad():
                for i in range(self.memory_length):
                    self.total_steps += 1
                    ep_steps += 1

                    act_logp_ent_val = self.network.get_act_logp_ent_val(next_obs[0], next_obs[1],
                                                                         masks=next_mask)
                    self.act_logp_ent_val[i] = act_logp_ent_val[0]
                    actions = act_logp_ent_val[0, :, 0]
                    obs, reward, terminated, truncated, info = env.step(actions)
                    next_mask = info.get('masks', None)

                    obs = self._obs_to_batched(obs)
                    done = 1.0 if terminated or truncated else 0.0

                    self.obs_image[i] = next_obs[0]
                    self.obs_features[i] = next_obs[1]
                    self.dones[i] = next_done
                    self.rewards[i] = torch.tensor(reward, dtype=torch.float32, device=self.device)

                    self._update_step(stats, info)
                    if terminated or truncated:
                        next_obs, _ = env.reset()
                        next_obs = self._obs_to_batched(next_obs)
                        next_done = 0.0
                        next_mask = None

                        self._update_episode_complete(stats, ep_steps)
                        ep_steps = 0

                    else:
                        next_obs = obs
                        next_done = done

                    progress.update(1)

                self.dones[self.memory_length] = next_done
                next_value = self.network.get_value(next_obs[0], next_obs[1])

                self._update_tensors(next_value[0])

            self._optimize(epochs)
            self._update_stats(progress)
            if self.train_callback and self.train_callback(self.total_steps, self):
                break

        progress.close()

    def _update_step(self, stats, info):
        if (reward_values := info.get('rewards', None)):
            for key, value in reward_values.items():
                if key.startswith('pf-'):
                    kind = 'pathfinding'
                    new_name = key[3:]
                elif key.startswith('ds-'):
                    kind = 'danger_sense'
                    new_name = key[3:]
                elif key.startswith('sa-'):
                    kind = 'action'
                    new_name = key[3:]

                counts = stats.setdefault(f"{kind}-counts", {})
                counts[new_name] = counts.get(key, 0) + 1

                values = stats.setdefault(kind, {})
                values[new_name] = values.get(key, 0) + value

        if (end_reason := info.get('end_reason', None)):
            endings = self.stats.setdefault('endings', {})
            endings[end_reason] = endings.get(end_reason, 0) + 1

    def _update_episode_complete(self, stats, steps):
        for key, value in stats.items():
            if isinstance(value, dict):
                self._update_dictionary(key, value)

        self.stats.setdefault('ep_len', []).append(steps)

    def _update_dictionary(self, name, dictionary):
        global_stats = self.stats.setdefault(name, {})
        for key, value in dictionary.items():
            global_stats.setdefault(key, []).append(value)

    def _clear_stats(self, stats):
        if isinstance(stats, dict):
            for key, value in stats.items():
                if isinstance(value, int):
                    stats[key] = 0
                elif isinstance(value, list):
                    value.clear()
                elif isinstance(value, dict):
                    self._clear_stats(value)
                else:
                    raise ValueError(f"Unknown value type {type(value)}")
        elif isinstance(stats, list):
            stats.clear()

    def _update_stats(self, progress):
        rewards = self.rewards.mean(dim=0, keepdim=True).squeeze(0).tolist()
        if self.tensorboard:
            self.tensorboard.add_scalar('rewards/pathfinding', rewards[0], self.total_steps)
            self.tensorboard.add_scalar('rewards/danger_sense', rewards[1], self.total_steps)
            self.tensorboard.add_scalar('rewards/selected_action', rewards[2], self.total_steps)
            self.tensorboard.add_scalar('rewards/total', sum(rewards), self.total_steps)

            if (ep_len := self.stats.get('ep_len', None)) and len(ep_len) > 0:
                self.tensorboard.add_scalar('rollout/ep_len', np.average(ep_len), self.total_steps)

            for key, value in self.stats.items():
                if isinstance(value, dict):
                    self._write_one_dictionary(key, value)

            self.tensorboard.flush()

        self._clear_stats(self.stats)

        rewards = " ".join(f"{r:.2f}" for r in rewards)
        progress.set_postfix(rewards=rewards)

    def _write_one_dictionary(self, kind, dictionary):
        if len(dictionary) == 0:
            return

        if isinstance(next(iter(dictionary.values()), None), list):
            length = max(len(x) for x in dictionary.values())
            if length == 0:
                length = 1

            for key, value in dictionary.items():
                avg = np.sum(value) / length if value else 0
                self.tensorboard.add_scalar(f'{kind}/{key}', avg, self.total_steps)
                value.clear()
        else:
            for key, value in dictionary.items():
                self.tensorboard.add_scalar(f'{kind}/{key}', value, self.total_steps)
                dictionary[key] = 0


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

def direction_to_action(direction):
    """Converts a direction into a model action."""
    match direction:
        case SelectedDirection.N: return 0
        case SelectedDirection.S: return 1
        case SelectedDirection.W: return 2
        case SelectedDirection.E: return 3
        case None: return 4
        case _: raise ValueError(f"Invalid direction: {direction}")

def mask_actions(disabled_actions):
    """Returns a mask for the given actions."""
    mask = torch.ones(3, dtype=torch.float32)
    for action in disabled_actions:
        if isinstance(action, SelectedAction):
            mask[action.value] = 0.0
        else:
            mask[action] = 0.0

    return mask

def mask_movement(disabled_directions):
    """Returns a mask for the given directions."""
    mask = torch.ones(5, dtype=torch.float32)
    for direction in disabled_directions:
        mask[direction] = 0.0

    return mask

def action_to_selection(action):
    """Converts a model action into a selection."""
    match action:
        case 0: return 0
        case 1: return 1
        case 2: return 2
        case 3: return None
        case _: raise ValueError(f"Invalid selection action: {action}")

__all__ = [
    MultiHeadPPO.__name__,
    ZeldaMultiHeadNetwork.__name__,
    direction_to_action.__name__,
    SelectedAction.__name__,
    mask_actions.__name__,
    mask_movement.__name__,
    action_to_selection.__name__
    ]
