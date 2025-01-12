import torch
import torch.distributions as dist
from torch import nn
import numpy as np


def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class NatureCNN(nn.Module):
    """Simple CNN."""
    def __init__(self, input_channels=1, linear_output_size=256):
        super().__init__()
        self.cnn = nn.Sequential(
            _layer_init(nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten()
        )
        # Calculate the output size of the CNN for the linear layer
        cnn_output_size = 64 * 9 * 16  # depends on your input size
        self.linear = nn.Sequential(
            _layer_init(nn.Linear(cnn_output_size, linear_output_size)),
            nn.ReLU()
        )

    def forward(self, x):
        """Forward pass."""
        x = self.cnn(x)
        x = self.linear(x)
        return x


class CombinedExtractor(nn.Module):
    """Combine image, vectors, and information."""
    def __init__(self, image_channels=1, image_linear_size=256, vectors_size=18, info_size=12):
        super().__init__()
        self.image_extractor = NatureCNN(input_channels=image_channels, linear_output_size=image_linear_size)
        self.flatten_info = nn.Flatten()
        self.flatten_vectors = nn.Flatten()
        self.vectors_size = vectors_size
        self.info_size = info_size

    def forward(self, image, vectors, information):
        """Forward pass."""
        image_features = self.image_extractor(image)
        vectors_features = self.flatten_vectors(vectors)
        info_features = self.flatten_info(information)
        combined_features = torch.cat([image_features, vectors_features, info_features], dim=1)
        return combined_features


class MlpExtractor(nn.Module):
    """MLP for policy and value."""
    def __init__(self, input_size, policy_hidden_size=64, value_hidden_size=64):
        super().__init__()
        self.policy_net = nn.Sequential(
            _layer_init(nn.Linear(input_size, policy_hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(policy_hidden_size, policy_hidden_size)),
            nn.Tanh()
        )
        self.value_net = nn.Sequential(
            _layer_init(nn.Linear(input_size, value_hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(value_hidden_size, value_hidden_size)),
            nn.Tanh()
        )

    def forward(self, combined_features):
        """Forward pass."""
        policy_features = self.policy_net(combined_features)
        value_features = self.value_net(combined_features)
        return policy_features, value_features


class MultiInputActorCriticPolicy(nn.Module):
    """Actor-critic policy with multiple inputs + action masking."""
    def __init__(self, image_channels=1, vectors_size=(3, 2, 3), info_size=12, action_size=8):
        super().__init__()
        self.action_size = action_size
        self.image_linear_size = 256

        combined_input_size = (
            self.image_linear_size +
            vectors_size[0] * vectors_size[1] * vectors_size[2] +
            info_size
        )

        # 1) Feature extractor
        self.feature_extractor = CombinedExtractor(
            image_channels=image_channels,
            image_linear_size=self.image_linear_size,
            vectors_size=vectors_size[0]*vectors_size[1]*vectors_size[2],
            info_size=info_size
        )

        # 2) MLP for policy/value features
        self.mlp_extractor = MlpExtractor(input_size=combined_input_size)

        # 3) The final policy & value heads
        self.action_net = _layer_init(nn.Linear(64, action_size), std=0.01)
        self.value_net = _layer_init(nn.Linear(64, 1), std=1.0)

    def forward(self, image, vectors, information):
        """
        Returns unmasked action logits and the value estimate.
        This is typically called inside get_action_and_value(...) or for debugging.
        """
        # Get combined features from CNN + Flatten + Cat
        combined_features = self.feature_extractor(image, vectors, information)
        # Extract separate features for policy and value
        policy_features, value_features = self.mlp_extractor(combined_features)

        # Logits for each possible action
        action_logits = self.action_net(policy_features)
        # Value estimate
        value = self.value_net(value_features)
        return action_logits, value

    def get_action_and_value(self, obs_tuple, actions=None, mask=None):
        """
        obs_tuple: (image, vectors, info)
        actions:   optional tensor of discrete actions (for log_prob, value_loss, etc.)
        mask:      optional boolean mask with shape [batch_size, action_size].
                   True = valid, False = invalid (we will zero out or -inf invalid).

        Returns:
            distribution, log_prob, entropy, value
        """
        # Unpack the 3-part observation
        image, vectors, information = obs_tuple

        # 1) Forward pass for logits, value
        logits, value = self.forward(image, vectors, information)  # shape of logits: [batch_size, action_size]

        # 2) Apply mask if provided
        if mask is not None:
            # mask.shape: [batch_size, action_size], dtype=bool
            # We can forcibly set invalid actions' logits to a large negative number
            # so that distribution probabilities go to zero.
            logits = logits.clone()  # avoid modifying in-place
            invalid_mask = ~mask      # True where the action is invalid
            logits[invalid_mask] = -1e9

        # 3) Create a categorical distribution over the valid actions
        dist_cat = dist.Categorical(logits=logits)

        # 4) If actions is provided, compute log_prob and distribution entropy
        if actions is not None:
            # actions should have shape [batch_size], each entry in [0 .. action_size-1]
            log_prob = dist_cat.log_prob(actions)
            entropy = dist_cat.entropy()
        else:
            # For inference, you may not pass actions. Return None placeholders.
            log_prob = None
            entropy = dist_cat.entropy()  # you can still measure distribution entropy

        # value has shape [batch_size, 1], flatten if needed
        return dist_cat, log_prob, entropy, value.view(-1)

    def get_act_logp_ent_val_mask(self, obs_tuple, actions, mask):
        """Same as get_action_and_value, but returns the mask as well."""
        result = self.get_action_and_value(obs_tuple, actions, mask)
        return result[0], result[1], result[2], result[3], mask
