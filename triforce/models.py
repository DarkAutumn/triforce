# pylint: disable=too-many-lines
import inspect
import pickle
import subprocess
import sys
import os
from typing import List
import torch
from torch import nn
import torch.distributions as dist
import numpy as np
import yaml
from gymnasium.spaces import Dict


def _get_git_commit():
    """Get the current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

class Network(nn.Module):
    """The base class of neural networks used for PPO training."""
    base : nn.Module
    action_net : nn.Module
    value_net : nn.Module
    is_multihead = False

    def __init__(self, base_network : nn.Module, obs_space, action_space,
                 model_kind=None, action_space_name=None, base_output_size=64):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = action_space
        self.model_kind = model_kind
        self.action_space_name = action_space_name
        self.steps_trained = 0
        self.episodes_evaluated = 0
        self.metrics : dict[str, float] = {}
        self.git_commit = _get_git_commit()

        self.base = base_network
        self.action_net = self.layer_init(nn.Linear(base_output_size, action_space.n), std=0.01)
        self.value_net = self.layer_init(nn.Linear(base_output_size, 1), std=1.0)

    def forward(self, obs):
        """Forward pass."""
        obs = self._unsqueeze(obs)

        if isinstance(obs, dict):
            inputs = []
            for key in obs:
                inputs.append(obs[key])
            x = self.base(*inputs)
        else:
            x = self.base(obs)

        action = self.action_net(x)
        value = self.value_net(x)
        return action, value

    def _unsqueeze(self, obs):
        """Unsqueeze the observation."""
        if isinstance(obs, dict):
            obs = obs.copy()
            for key in obs:
                if obs[key].shape == self.observation_space[key].shape:
                    obs[key] = obs[key].unsqueeze(0)
        else:
            if obs.shape == self.observation_space.shape:
                obs = obs.unsqueeze(0)

        return obs

    def get_action_and_value(self, obs, mask, actions=None, deterministic=False):
        """Gets the action, logprob, entropy, and value."""
        logits, value = self.forward(obs)

        # mask out invalid actions
        if mask is not None:
            assert mask.any(dim=-1).all(), "Mask must contain at least one valid action"
            logits = logits.clone()
            invalid_mask = ~mask
            logits[invalid_mask] = -1e9

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
        _, value = self.forward(obs)
        return value.view(-1)

    def get_action(self, obs, mask = None, deterministic = False):
        """Get the action from the observation."""
        action, _, _, _ = self.get_action_and_value(obs, mask, deterministic=deterministic)
        return action

    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        """Initialize a linear layer."""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def save(self, path):
        """Save the network to a file."""
        save_data = {
            "model_state_dict": self.state_dict(),
            "steps_trained": self.steps_trained,
            "episodes_evaluated" : self.episodes_evaluated,
            "metrics": pickle.dumps(self.metrics) if self.metrics else None,
            "obs_space": self.observation_space,
            "action_space": self.action_space,
            "model_kind": self.model_kind,
            "action_space_name": self.action_space_name,
            "git_commit": _get_git_commit(),
        }

        torch.save(save_data, path)

    def load(self, path) -> 'Network':
        """Load the network from a file."""
        save_data = torch.load(path, weights_only=False)

        self.load_state_dict(save_data["model_state_dict"])
        self.steps_trained = save_data["steps_trained"]
        self.episodes_evaluated = save_data.get("episodes_evaluated", 0)
        self.model_kind = save_data.get("model_kind")
        self.action_space_name = save_data.get("action_space_name")
        self.git_commit = save_data.get("git_commit")
        metrics_pickled = save_data.get("metrics")
        self.metrics = pickle.loads(metrics_pickled) if metrics_pickled else {}

        if self.observation_space != save_data["obs_space"]:
            raise ValueError("Mismatch in observation space!")

        if self.action_space != save_data["action_space"]:
            raise ValueError("Mismatch in action space!")

        return self

    @staticmethod
    def load_metrics(path):
        """Load the metrics from a file."""
        save_data = torch.load(path, weights_only=False)
        metrics_pickled = save_data.get("metrics")
        metrics = pickle.loads(metrics_pickled) if metrics_pickled else {}
        episodes_evaluated = save_data.get("episodes_evaluated", 0)
        return metrics, episodes_evaluated

    @staticmethod
    def load_spaces(path):
        """Load the observation and action spaces from a file."""
        save_data = torch.load(path, weights_only=False)
        return save_data["obs_space"], save_data["action_space"]

    @staticmethod
    def load_metadata(path):
        """Load model_kind and action_space_name from a saved .pt file."""
        save_data = torch.load(path, weights_only=False)
        metrics_pickled = save_data.get("metrics")
        metrics = pickle.loads(metrics_pickled) if metrics_pickled else {}
        return {
            "model_kind": save_data.get("model_kind"),
            "action_space_name": save_data.get("action_space_name"),
            "steps_trained": save_data.get("steps_trained", 0),
            "obs_space": save_data.get("obs_space"),
            "action_space": save_data.get("action_space"),
            "git_commit": save_data.get("git_commit"),
            "metrics": metrics,
        }

class NatureCNN(nn.Module):
    """Simple CNN that adjusts to input size dynamically."""
    def __init__(self, input_channels=1, input_height=128, input_width=128, linear_output_size=256):
        super().__init__()
        self.cnn = nn.Sequential(
            Network.layer_init(nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            Network.layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            Network.layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten()
        )

        # Use dynamic input size for figuring out output shape
        dummy_input = torch.zeros(1, input_channels, input_height, input_width)
        with torch.no_grad():
            cnn_out = self.cnn(dummy_input)
        cnn_output_size = cnn_out.shape[1]

        self.linear = nn.Sequential(
            Network.layer_init(nn.Linear(cnn_output_size, linear_output_size)),
            nn.ReLU()
        )

    def forward(self, tensor):
        """Forward pass."""
        cnn_out = self.cnn(tensor)
        linear_out = self.linear(cnn_out)
        return linear_out

class MlpExtractor(nn.Module):
    """MLP for policy and value."""
    def __init__(self, input_size, policy_hidden_size=256, value_hidden_size=256):
        super().__init__()
        self.policy_output_dim = policy_hidden_size
        self.value_output_dim = value_hidden_size

        self.policy_net = nn.Sequential(
            Network.layer_init(nn.Linear(input_size, policy_hidden_size)),
            nn.LayerNorm(policy_hidden_size),
            nn.ReLU(),
            Network.layer_init(nn.Linear(policy_hidden_size, policy_hidden_size)),
            nn.LayerNorm(policy_hidden_size),
            nn.ReLU(),
            Network.layer_init(nn.Linear(policy_hidden_size, policy_hidden_size)),
            nn.LayerNorm(policy_hidden_size),
            nn.Tanh(),
        )
        self.value_net = nn.Sequential(
            Network.layer_init(nn.Linear(input_size, value_hidden_size)),
            nn.LayerNorm(value_hidden_size),
            nn.ReLU(),
            Network.layer_init(nn.Linear(value_hidden_size, value_hidden_size)),
            nn.LayerNorm(value_hidden_size),
            nn.ReLU(),
            Network.layer_init(nn.Linear(value_hidden_size, value_hidden_size)),
            nn.LayerNorm(value_hidden_size),
            nn.ReLU(),
        )

    def forward(self, combined_features):
        """Forward pass."""
        policy_features = self.policy_net(combined_features)
        value_features = self.value_net(combined_features)
        return policy_features, value_features

class EntityAttentionEncoder(nn.Module):
    """Self-attention encoder for a unified entity list.

    Processes a flat list of entity slots (enemies, items, projectiles, treasure)
    through a transformer encoder, masking empty slots, then mean-pools present
    entities into a fixed-size output vector.
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(self, num_entity_types, continuous_features=7, embedding_dim=8,
                 d_model=64, num_heads=4, num_layers=1, ff_dim=128, output_dim=64):
        super().__init__()
        self.type_embedding = nn.Embedding(num_entity_types, embedding_dim)
        input_dim = continuous_features + embedding_dim
        self.input_proj = Network.layer_init(nn.Linear(input_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim,
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                enable_nested_tensor=False)
        self.output_proj = Network.layer_init(nn.Linear(d_model, output_dim))
        self.output_dim = output_dim

    def _encode(self, entity_features, entity_types):
        """Shared encoder logic: embed, project, mask, transform.

        Returns:
            attended: (batch, slots, d_model) transformer output
            empty_mask: (batch, slots) bool — True for empty slots
        """
        type_embeds = self.type_embedding(entity_types.long())
        combined = torch.cat([entity_features, type_embeds], dim=-1)
        projected = self.input_proj(combined)

        # Mask empty slots (presence == 0) so they don't attend or get attended to
        empty_mask = entity_features[:, :, 0] == 0

        # Prevent all-masked rows: unmask slot 0 so attention softmax never sees all -inf
        all_empty = empty_mask.all(dim=1)
        safe_mask = empty_mask.clone()
        safe_mask[all_empty, 0] = False

        attended = self.transformer(projected, src_key_padding_mask=safe_mask)
        return attended, empty_mask

    def forward(self, entity_features, entity_types):
        """Forward pass.

        Args:
            entity_features: (batch, slots, continuous_features)
            entity_types: (batch, slots) long tensor of type IDs
        Returns:
            (batch, output_dim) pooled entity representation
        """
        attended, empty_mask = self._encode(entity_features, entity_types)

        # Mean pool over genuinely present entities
        present = (~empty_mask).unsqueeze(-1).float()
        num_present = present.sum(dim=1).clamp(min=1)
        pooled = (attended * present).sum(dim=1) / num_present

        return self.output_proj(pooled)

    def forward_with_tokens(self, entity_features, entity_types):
        """Forward pass returning both pooled output and per-entity tokens.

        Used by cross-attention: entity tokens query the spatial feature map.

        Returns:
            pooled: (batch, output_dim) mean-pooled entity vector
            tokens: (batch, slots, d_model) per-entity transformer output
            empty_mask: (batch, slots) bool — True for empty slots
        """
        attended, empty_mask = self._encode(entity_features, entity_types)

        present = (~empty_mask).unsqueeze(-1).float()
        num_present = present.sum(dim=1).clamp(min=1)
        pooled = (attended * present).sum(dim=1) / num_present

        return self.output_proj(pooled), attended, empty_mask


class EntitySpatialCrossAttention(nn.Module):
    """Multi-head cross-attention: entity tokens query spatial feature map positions.

    Each entity asks "where am I on screen?" via scaled dot-product attention over
    the spatial feature map. The result is a spatially-grounded context vector.
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(self, entity_dim=64, spatial_channels=32, hidden_dim=64,
                 output_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = Network.layer_init(nn.Linear(entity_dim, hidden_dim))
        self.k_proj = nn.Conv2d(spatial_channels, hidden_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(spatial_channels, hidden_dim, kernel_size=1)
        self.attn_dropout = nn.Dropout(p=dropout)

        self.output_proj = nn.Sequential(
            Network.layer_init(nn.Linear(hidden_dim, output_dim)),
            nn.ReLU()
        )
        self.output_dim = output_dim

    def forward(self, entity_tokens, feature_map, empty_mask):
        """Cross-attention: entities attend to spatial positions.

        Args:
            entity_tokens: (batch, slots, entity_dim) per-entity transformer output
            feature_map: (batch, C, H, W) spatial feature map from ResNet
            empty_mask: (batch, slots) bool — True for empty entity slots

        Returns:
            context: (batch, output_dim) pooled cross-attention context vector
            cross_attn_weights: (batch, num_heads, slots, H, W) or None during training
        """
        batch, slots, _ = entity_tokens.shape
        _, _, h, w = feature_map.shape
        n_spatial = h * w

        # Project entity tokens to queries: (batch, slots, hidden_dim)
        q = self.q_proj(entity_tokens)
        # Reshape to multi-head: (batch, num_heads, slots, head_dim)
        q = q.view(batch, slots, self.num_heads, self.head_dim).transpose(1, 2)

        # Project spatial feature map to keys and values
        k = self.k_proj(feature_map)  # (batch, hidden_dim, H, W)
        v = self.v_proj(feature_map)  # (batch, hidden_dim, H, W)
        # Reshape to multi-head: (batch, num_heads, head_dim, N) → (batch, num_heads, N, head_dim)
        k = k.view(batch, self.num_heads, self.head_dim, n_spatial).transpose(2, 3)
        v = v.view(batch, self.num_heads, self.head_dim, n_spatial).transpose(2, 3)

        # Scaled dot-product attention: (batch, num_heads, slots, N)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_logits, dim=-1)
        if self.training:
            attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of spatial values: (batch, num_heads, slots, head_dim)
        attended = torch.matmul(attn_weights, v)
        # Concatenate heads: (batch, slots, hidden_dim)
        attended = attended.transpose(1, 2).reshape(batch, slots, self.hidden_dim)

        # Zero out empty entity slots
        present = (~empty_mask).unsqueeze(-1).float()  # (batch, slots, 1)
        attended = attended * present

        # Mean pool over present entities
        num_present = present.sum(dim=1).clamp(min=1)  # (batch, 1)
        pooled = attended.sum(dim=1) / num_present  # (batch, hidden_dim)

        context = self.output_proj(pooled)

        # Only materialize attention map during eval (for visualization)
        cross_attn_map = None if self.training else attn_weights.view(batch, self.num_heads, slots, h, w)

        return context, cross_attn_map


class CombinedExtractor(nn.Module):
    """Combined extractor for CNN image + entity attention + boolean info."""
    # pylint: disable=too-many-positional-arguments, too-many-arguments
    def __init__(
        self,
        image_channels: int,
        input_height: int,
        input_width: int,
        num_entity_types: int,
        entity_features: int = 9,
        embedding_dim: int = 8,
        attention_heads: int = 4,
        attention_layers: int = 1,
        attention_output_dim: int = 64,
        info_size: int = 14,
        image_linear_size: int = 256,
    ):
        super().__init__()

        self.image_extractor = NatureCNN(
            input_channels=image_channels,
            input_height=input_height,
            input_width=input_width,
            linear_output_size=image_linear_size
        )

        self.entity_encoder = EntityAttentionEncoder(
            num_entity_types=num_entity_types,
            continuous_features=entity_features,
            embedding_dim=embedding_dim,
            num_heads=attention_heads,
            num_layers=attention_layers,
            output_dim=attention_output_dim
        )

        self.info_size = info_size
        self.output_dim = image_linear_size + attention_output_dim + info_size

    def forward(self, image: torch.Tensor, entities: torch.Tensor,
                entity_types: torch.Tensor, information: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        img_out = self.image_extractor(image)
        entity_out = self.entity_encoder(entities, entity_types)
        info_float = information.float()
        return torch.cat([img_out, entity_out, info_float], dim=1)

class ResBlock(nn.Module):
    """Pre-activation residual block: ReLU → Conv3×3 → ReLU → Conv3×3 + skip."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Network.layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.conv2 = Network.layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1))

    def forward(self, x):
        """Forward pass."""
        residual = x
        x = torch.relu(x)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x + residual


class SpatialAttentionPool(nn.Module):
    """Multi-head self-attention pooling over spatial positions in a feature map.

    Projects the feature map to queries, keys, and values via 1×1 convolutions,
    splits into multiple heads, computes per-head scaled dot-product attention,
    concatenates, and projects to a single feature vector per batch element.
    """
    def __init__(self, in_channels, hidden_dim=64, output_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.key_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.output_proj = nn.Sequential(
            Network.layer_init(nn.Linear(hidden_dim, output_dim)),
            nn.ReLU()
        )
        self.attn_dropout = nn.Dropout(p=dropout)
        self.scale = self.head_dim ** -0.5
        self.output_dim = output_dim

    def forward(self, feature_map):
        """Forward pass.

        Args:
            feature_map: (batch, C, H, W) feature map from the ResNet backbone.
        Returns:
            features: (batch, output_dim) pooled feature vector.
            attn_weights: (batch, num_heads, H, W) per-head attention weights.
        """
        batch, _, h, w = feature_map.shape
        n = h * w

        # Project to Q, K, V: (batch, hidden_dim, H, W) → (batch, num_heads, N, head_dim)
        q = self.query_proj(feature_map).reshape(batch, self.num_heads, self.head_dim, n).transpose(2, 3)
        k = self.key_proj(feature_map).reshape(batch, self.num_heads, self.head_dim, n).transpose(2, 3)
        v = self.value_proj(feature_map).reshape(batch, self.num_heads, self.head_dim, n).transpose(2, 3)

        # Compute attention (manual path needed for attention-weighted pooling)
        q_flat = q.reshape(batch * self.num_heads, n, self.head_dim)
        k_flat = k.reshape(batch * self.num_heads, n, self.head_dim)
        v_flat = v.reshape(batch * self.num_heads, n, self.head_dim)

        attn_logits = torch.bmm(q_flat, k_flat.transpose(1, 2)) * self.scale
        attn = torch.softmax(attn_logits, dim=-1)
        if self.training:
            attn = self.attn_dropout(attn)

        attended = torch.bmm(attn, v_flat)  # (batch*num_heads, N, head_dim)
        attn_weights = attn.mean(dim=1)  # (batch*num_heads, N)

        pooled = (attended * attn_weights.unsqueeze(-1)).sum(dim=1)
        pooled = pooled.view(batch, self.num_heads, self.head_dim)

        # Only materialize attention map during eval (for visualization)
        attn_map = None if self.training else attn_weights.view(batch, self.num_heads, h, w)

        features = self.output_proj(pooled.reshape(batch, -1))
        return features, attn_map


class ImpalaResNet(nn.Module):
    """IMPALA-style ResNet with CoordConv and spatial attention pooling.

    Processes full-screen RGB frames (with HUD trimmed) through three residual stacks
    and produces a fixed-size feature vector via spatial self-attention pooling.
    All spatial dimensions are computed dynamically from (input_height, input_width).
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(self, input_channels=3, input_height=168, input_width=240,
                 channels=(16, 32, 32), attention_hidden=64, output_dim=256):
        super().__init__()

        # CoordConv: 2 static coordinate channels
        h_coords = torch.linspace(0, 1, input_height).view(1, 1, input_height, 1)
        coord_h = h_coords.expand(1, 1, input_height, input_width).clone()
        w_coords = torch.linspace(0, 1, input_width).view(1, 1, 1, input_width)
        coord_w = w_coords.expand(1, 1, input_height, input_width).clone()
        self.register_buffer('coord_h', coord_h)
        self.register_buffer('coord_w', coord_w)

        in_ch = input_channels + 2  # RGB + 2 coord channels

        # Three residual stacks: Conv3×3 → MaxPool(stride=2) → 2× ResBlock
        stacks = []
        for out_ch in channels:
            stacks.append(nn.Sequential(
                Network.layer_init(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ResBlock(out_ch),
                ResBlock(out_ch),
            ))
            in_ch = out_ch
        self.stacks = nn.Sequential(*stacks)

        # Spatial attention pooling
        self.attention_pool = SpatialAttentionPool(
            in_channels=channels[-1],
            hidden_dim=attention_hidden,
            output_dim=output_dim
        )
        self.output_dim = output_dim

    def forward(self, image):
        """Forward pass.

        Args:
            image: (batch, C, H, W) RGB image tensor.
        Returns:
            features: (batch, output_dim) feature vector.
            attn_weights: (batch, num_heads, H', W') spatial attention weights (None during training).
            feature_map: (batch, C_out, H', W') spatial feature map before attention pooling.
        """
        batch = image.shape[0]
        coord_h = self.coord_h.expand(batch, -1, -1, -1)
        coord_w = self.coord_w.expand(batch, -1, -1, -1)
        x = torch.cat([image, coord_h, coord_w], dim=1)

        x = self.stacks(x)
        x = torch.relu(x)

        features, attn_weights = self.attention_pool(x)
        return features, attn_weights, x


class ImpalaCombinedExtractor(nn.Module):
    """Combined extractor using ImpalaResNet image + entity attention + cross-attention + boolean info.

    Architecture:
        Image  → ImpalaResNet → feature_map → SpatialAttentionPool → 256-d
        Entities → EntityAttentionEncoder → pooled (64-d) + per-entity tokens
        Entity tokens + feature_map → EntitySpatialCrossAttention → 64-d context
        Concat [img(256), entity_pooled(64), cross_attn(64), info(15)] → 399-d
    """
    # pylint: disable=too-many-positional-arguments, too-many-arguments
    def __init__(
        self,
        image_channels: int,
        input_height: int,
        input_width: int,
        num_entity_types: int,
        entity_features: int = 9,
        embedding_dim: int = 8,
        attention_heads: int = 4,
        attention_layers: int = 1,
        attention_output_dim: int = 64,
        cross_attn_output_dim: int = 64,
        info_size: int = 14,
        image_output_size: int = 256,
    ):
        super().__init__()

        self.image_extractor = ImpalaResNet(
            input_channels=image_channels,
            input_height=input_height,
            input_width=input_width,
            output_dim=image_output_size
        )

        self.entity_encoder = EntityAttentionEncoder(
            num_entity_types=num_entity_types,
            continuous_features=entity_features,
            embedding_dim=embedding_dim,
            num_heads=attention_heads,
            num_layers=attention_layers,
            output_dim=attention_output_dim
        )

        # Cross-attention: entity tokens query spatial feature map
        # entity_dim = d_model from EntityAttentionEncoder (default 64)
        # spatial_channels = last ResNet stack channel count (default 32)
        self.cross_attention = EntitySpatialCrossAttention(
            entity_dim=64,
            spatial_channels=32,
            hidden_dim=64,
            output_dim=cross_attn_output_dim,
            num_heads=attention_heads
        )

        self.info_size = info_size
        self.output_dim = image_output_size + attention_output_dim + cross_attn_output_dim + info_size

    def forward(self, image, entities, entity_types, information):
        """Forward pass.

        Returns:
            combined: (batch, output_dim) concatenated features
            spatial_attn: (batch, num_heads, H', W') or None during training
            cross_attn: (batch, num_heads, slots, H', W') or None during training
        """
        img_out, spatial_attn, feature_map = self.image_extractor(image)

        entity_pooled, entity_tokens, empty_mask = self.entity_encoder.forward_with_tokens(
            entities, entity_types)

        cross_context, cross_attn = self.cross_attention(entity_tokens, feature_map, empty_mask)

        info_float = information.float()
        combined = torch.cat([img_out, entity_pooled, cross_context, info_float], dim=1)
        return combined, spatial_attn, cross_attn


class MultiHeadAgent(Network):
    """Two-head action decomposition: action_type (K) + direction (4).

    Instead of a flat Discrete(N) action space, this agent uses MultiDiscrete([K, 4])
    where K = number of action types. The two heads share the same CNN + MLP backbone
    but output independent logits. Joint log-prob: log π(a|s) = log π_type + log π_dir.
    """
    is_multihead = True

    # pylint: disable=super-init-not-called
    def __init__(self, obs_space: Dict, action_space, model_kind=None, action_space_name=None):
        channels, height, width = obs_space["image"].shape
        image_linear_size = 256

        base_network = CombinedExtractor(
            image_channels=channels,
            input_height=height,
            input_width=width,
            num_entity_types=int(obs_space["entity_types"].nvec[0]),
            entity_features=obs_space["entities"].shape[1],
            info_size=obs_space["information"].n,
            image_linear_size=image_linear_size
        )

        # Bypass Network.__init__ because MultiDiscrete has no .n attribute.
        # Manually set all attributes that Network.__init__ would create.
        # pylint: disable=non-parent-init-called
        nn.Module.__init__(self)
        self.observation_space = obs_space
        self.action_space = action_space
        self.model_kind = model_kind
        self.action_space_name = action_space_name
        self.steps_trained = 0
        self.episodes_evaluated = 0
        self.metrics = {}
        self.git_commit = _get_git_commit()

        self.base = base_network
        self.mlp_extractor = MlpExtractor(input_size=self.base.output_dim)

        # Two action heads: action_type (K) and direction (4)
        num_action_types = int(action_space.nvec[0])
        self.action_type_net = self.layer_init(
            nn.Linear(self.mlp_extractor.policy_output_dim, num_action_types), std=0.01)
        self.direction_net = self.layer_init(
            nn.Linear(self.mlp_extractor.policy_output_dim, 4), std=0.01)

        # Single shared value head
        self.value_net = self.layer_init(
            nn.Linear(self.mlp_extractor.value_output_dim, 1), std=1.0)

        # Placeholder so Network.base class annotation is satisfied
        self.action_net = nn.Identity()

    def forward(self, obs):
        """Returns (action_type_logits, direction_logits, value)."""
        obs = self._unsqueeze(obs)

        combined_features = self.base(
            image=obs["image"],
            entities=obs["entities"],
            entity_types=obs["entity_types"],
            information=obs["information"]
        )

        policy_features, value_features = self.mlp_extractor(combined_features)

        action_type_logits = self.action_type_net(policy_features)
        direction_logits = self.direction_net(policy_features)
        value = self.value_net(value_features)
        return action_type_logits, direction_logits, value

    def get_action_and_value(self, obs, mask, actions=None, deterministic=False):
        """Gets action, joint log-prob, summed entropy, and value.

        Args:
            obs: Dict observation.
            mask: Joint [batch, K*4] mask where mask[..., i*4+j] indicates whether
                  action type i with direction j is valid.
            actions: Optional [batch, 2] tensor of (action_type, direction) indices.
            deterministic: If True, use argmax instead of sampling.

        Returns:
            (actions [batch, 2], log_prob [batch], entropy [batch], value [batch])
        """
        action_type_logits, direction_logits, value = self.forward(obs)
        num_action_types = int(self.action_space.nvec[0])

        if mask is not None:
            # Reshape [batch, K*4] → [batch, K, 4] for per-type direction masking
            mask_2d = mask.view(*mask.shape[:-1], num_action_types, 4)

            # Type mask: a type is valid if ANY of its directions are valid
            action_type_mask = mask_2d.any(dim=-1)

            bad_type = ~action_type_mask.any(dim=-1)
            if bad_type.any():
                idx = bad_type.nonzero(as_tuple=True)[0][0].item()
                info_vec = obs['info'][idx].tolist() if 'info' in obs else 'N/A'
                raise ValueError(
                    f"Empty action mask in batch element {idx}.\n"
                    f"  action_type_mask={action_type_mask[idx].tolist()}\n"
                    f"  full_mask={mask[idx].tolist()}\n"
                    f"  num_action_types={num_action_types}\n"
                    f"  nvec={self.action_space.nvec.tolist()}\n"
                    f"  info={info_vec}"
                )

            action_type_logits = action_type_logits.clone()
            action_type_logits[~action_type_mask] = -1e9

        type_dist = dist.Categorical(logits=action_type_logits)

        if actions is None:
            if deterministic:
                type_action = action_type_logits.argmax(dim=-1)
            else:
                type_action = type_dist.sample()
        else:
            type_action = actions[..., 0].long()

        # Condition direction mask on the selected/given action type
        if mask is not None:
            batch_range = torch.arange(mask_2d.shape[0], device=mask.device)
            dir_mask = mask_2d[batch_range, type_action]  # [batch, 4]

            direction_logits = direction_logits.clone()
            direction_logits[~dir_mask] = -1e9

        dir_dist = dist.Categorical(logits=direction_logits)

        if actions is None:
            if deterministic:
                dir_action = direction_logits.argmax(dim=-1)
            else:
                dir_action = dir_dist.sample()
            actions = torch.stack([type_action, dir_action], dim=-1)

        # Joint log-prob: log π(a|s) = log π_type(a_type|s) + log π_dir(a_dir|s,a_type)
        log_prob = type_dist.log_prob(actions[..., 0]) + dir_dist.log_prob(actions[..., 1])

        # Entropy: sum of per-head entropies
        entropy = type_dist.entropy() + dir_dist.entropy()

        return actions, log_prob, entropy, value.view(-1)

    def get_value(self, obs):
        """Get value estimate."""
        _, _, value = self.forward(obs)
        return value.view(-1)

    def get_action(self, obs, mask=None, deterministic=False):
        """Get the action from the observation."""
        action, _, _, _ = self.get_action_and_value(obs, mask, deterministic=deterministic)
        return action

    def get_entropy_details(self, obs, mask):
        """Returns per-head entropy means for Tensorboard logging.

        Used by PPO._optimize to log action_type and direction entropy separately,
        enabling diagnosis of per-head entropy collapse.

        Direction entropy is averaged over all samples using the marginal direction
        mask (union across types) since we don't have a specific type per sample here.
        """
        action_type_logits, direction_logits, _ = self.forward(obs)
        num_action_types = int(self.action_space.nvec[0])

        if mask is not None:
            mask_2d = mask.view(*mask.shape[:-1], num_action_types, 4)
            action_type_mask = mask_2d.any(dim=-1)
            # Marginal direction mask (union across types) for entropy logging
            direction_mask = mask_2d.any(dim=-2)
            action_type_logits = action_type_logits.clone()
            action_type_logits[~action_type_mask] = -1e9
            direction_logits = direction_logits.clone()
            direction_logits[~direction_mask] = -1e9

        type_dist = dist.Categorical(logits=action_type_logits)
        dir_dist = dist.Categorical(logits=direction_logits)

        return {
            "entropy/action_type": type_dist.entropy().mean().item(),
            "entropy/direction": dir_dist.entropy().mean().item(),
        }


class SharedNatureAgent(Network):
    """Actor-critic policy with multiple inputs, action masking, and shared CNN."""
    def __init__(self, obs_space: Dict, action_space, model_kind=None, action_space_name=None):
        channels, height, width = obs_space["image"].shape

        # We'll do a straightforward approach: let the user define a 'linear_output_size'.
        # We'll keep 'image_linear_size' for the final MLP size.
        image_linear_size = 256  # or some fixed size, ignoring (height+width) approach

        super().__init__(
            base_network=CombinedExtractor(
                image_channels=channels,
                input_height=height,
                input_width=width,
                num_entity_types=int(obs_space["entity_types"].nvec[0]),
                entity_features=obs_space["entities"].shape[1],
                info_size=obs_space["information"].n,
                image_linear_size=image_linear_size
            ),
            obs_space=obs_space,
            action_space=action_space,
            model_kind=model_kind,
            action_space_name=action_space_name
        )

        # Now we create an MLP for policy/value.
        self.mlp_extractor = MlpExtractor(input_size=self.base.output_dim)

        # Overwrite action_net and value_net to match MLP output dimensions.
        self.action_net = self.layer_init(
            nn.Linear(self.mlp_extractor.policy_output_dim, action_space.n), std=0.01)
        self.value_net = self.layer_init(
            nn.Linear(self.mlp_extractor.value_output_dim, 1), std=1.0)

    def forward(self, obs):
        obs = self._unsqueeze(obs)

        combined_features = self.base(
            image=obs["image"],
            entities=obs["entities"],
            entity_types=obs["entity_types"],
            information=obs["information"]
        )

        policy_features, value_features = self.mlp_extractor(combined_features)

        action_logits = self.action_net(policy_features)
        value = self.value_net(value_features)
        return action_logits, value


class ImpalaSharedAgent(Network):
    """Actor-critic with IMPALA ResNet, CoordConv, and spatial attention pooling."""
    recommended_minibatches = 16  # spatial attention needs smaller minibatches

    def __init__(self, obs_space: Dict, action_space, model_kind=None, action_space_name=None):
        channels, height, width = obs_space["image"].shape
        image_output_size = 256

        super().__init__(
            base_network=ImpalaCombinedExtractor(
                image_channels=channels,
                input_height=height,
                input_width=width,
                num_entity_types=int(obs_space["entity_types"].nvec[0]),
                entity_features=obs_space["entities"].shape[1],
                info_size=obs_space["information"].n,
                image_output_size=image_output_size
            ),
            obs_space=obs_space,
            action_space=action_space,
            model_kind=model_kind,
            action_space_name=action_space_name
        )

        self.mlp_extractor = MlpExtractor(input_size=self.base.output_dim)
        self.action_net = self.layer_init(
            nn.Linear(self.mlp_extractor.policy_output_dim, action_space.n), std=0.01)
        self.value_net = self.layer_init(
            nn.Linear(self.mlp_extractor.value_output_dim, 1), std=1.0)

    def forward(self, obs):
        obs = self._unsqueeze(obs)
        combined_features, _, _ = self.base(
            image=obs["image"],
            entities=obs["entities"],
            entity_types=obs["entity_types"],
            information=obs["information"]
        )
        policy_features, value_features = self.mlp_extractor(combined_features)
        action_logits = self.action_net(policy_features)
        value = self.value_net(value_features)
        return action_logits, value

    def forward_with_attention(self, obs):
        """Forward pass returning spatial and cross-attention weights for visualization."""
        obs = self._unsqueeze(obs)
        combined_features, spatial_attn, cross_attn = self.base(
            image=obs["image"],
            entities=obs["entities"],
            entity_types=obs["entity_types"],
            information=obs["information"]
        )
        policy_features, value_features = self.mlp_extractor(combined_features)
        action_logits = self.action_net(policy_features)
        value = self.value_net(value_features)
        return action_logits, value, spatial_attn, cross_attn

    def get_attention_entropy(self, obs):
        """Returns attention entropy and top-1 concentration for tensorboard."""
        was_training = self.training
        self.eval()
        _, _, spatial_attn, cross_attn = self.forward_with_attention(obs)
        if was_training:
            self.train()
        stats = _attention_entropy_stats(spatial_attn)
        if cross_attn is not None:
            stats.update(_cross_attention_entropy_stats(cross_attn))
        return stats


class ImpalaMultiHeadAgent(Network):
    """Two-head action decomposition with IMPALA ResNet backbone."""
    is_multihead = True
    recommended_minibatches = 16  # spatial attention needs smaller minibatches

    # pylint: disable=super-init-not-called
    def __init__(self, obs_space: Dict, action_space, model_kind=None, action_space_name=None):
        channels, height, width = obs_space["image"].shape
        image_output_size = 256

        base_network = ImpalaCombinedExtractor(
            image_channels=channels,
            input_height=height,
            input_width=width,
            num_entity_types=int(obs_space["entity_types"].nvec[0]),
            entity_features=obs_space["entities"].shape[1],
            info_size=obs_space["information"].n,
            image_output_size=image_output_size
        )

        # Bypass Network.__init__ because MultiDiscrete has no .n attribute.
        # pylint: disable=non-parent-init-called
        nn.Module.__init__(self)
        self.observation_space = obs_space
        self.action_space = action_space
        self.model_kind = model_kind
        self.action_space_name = action_space_name
        self.steps_trained = 0
        self.episodes_evaluated = 0
        self.metrics = {}
        self.git_commit = _get_git_commit()

        self.base = base_network
        self.mlp_extractor = MlpExtractor(input_size=self.base.output_dim)

        num_action_types = int(action_space.nvec[0])
        self.action_type_net = self.layer_init(
            nn.Linear(self.mlp_extractor.policy_output_dim, num_action_types), std=0.01)
        self.direction_net = self.layer_init(
            nn.Linear(self.mlp_extractor.policy_output_dim, 4), std=0.01)
        self.value_net = self.layer_init(
            nn.Linear(self.mlp_extractor.value_output_dim, 1), std=1.0)
        self.action_net = nn.Identity()

    def forward(self, obs):
        """Returns (action_type_logits, direction_logits, value)."""
        obs = self._unsqueeze(obs)
        combined_features, _, _ = self.base(
            image=obs["image"],
            entities=obs["entities"],
            entity_types=obs["entity_types"],
            information=obs["information"]
        )
        policy_features, value_features = self.mlp_extractor(combined_features)
        action_type_logits = self.action_type_net(policy_features)
        direction_logits = self.direction_net(policy_features)
        value = self.value_net(value_features)
        return action_type_logits, direction_logits, value

    def forward_with_attention(self, obs):
        """Forward pass returning spatial and cross-attention weights for visualization."""
        obs = self._unsqueeze(obs)
        combined_features, spatial_attn, cross_attn = self.base(
            image=obs["image"],
            entities=obs["entities"],
            entity_types=obs["entity_types"],
            information=obs["information"]
        )
        policy_features, value_features = self.mlp_extractor(combined_features)
        action_type_logits = self.action_type_net(policy_features)
        direction_logits = self.direction_net(policy_features)
        value = self.value_net(value_features)
        return action_type_logits, direction_logits, value, spatial_attn, cross_attn

    def get_action_and_value(self, obs, mask, actions=None, deterministic=False):
        """Gets action, joint log-prob, summed entropy, and value."""
        action_type_logits, direction_logits, value = self.forward(obs)
        num_action_types = int(self.action_space.nvec[0])

        if mask is not None:
            mask_2d = mask.view(*mask.shape[:-1], num_action_types, 4)
            action_type_mask = mask_2d.any(dim=-1)

            bad_type = ~action_type_mask.any(dim=-1)
            if bad_type.any():
                idx = bad_type.nonzero(as_tuple=True)[0][0].item()
                info_vec = obs['info'][idx].tolist() if 'info' in obs else 'N/A'
                raise ValueError(
                    f"Empty action mask in batch element {idx}.\n"
                    f"  action_type_mask={action_type_mask[idx].tolist()}\n"
                    f"  full_mask={mask[idx].tolist()}\n"
                    f"  num_action_types={num_action_types}\n"
                    f"  nvec={self.action_space.nvec.tolist()}\n"
                    f"  info={info_vec}"
                )

            action_type_logits = action_type_logits.clone()
            action_type_logits[~action_type_mask] = -1e9

        type_dist = dist.Categorical(logits=action_type_logits)

        if actions is None:
            if deterministic:
                type_action = action_type_logits.argmax(dim=-1)
            else:
                type_action = type_dist.sample()
        else:
            type_action = actions[..., 0].long()

        if mask is not None:
            batch_range = torch.arange(mask_2d.shape[0], device=mask.device)
            dir_mask = mask_2d[batch_range, type_action]
            direction_logits = direction_logits.clone()
            direction_logits[~dir_mask] = -1e9

        dir_dist = dist.Categorical(logits=direction_logits)

        if actions is None:
            if deterministic:
                dir_action = direction_logits.argmax(dim=-1)
            else:
                dir_action = dir_dist.sample()
            actions = torch.stack([type_action, dir_action], dim=-1)

        log_prob = type_dist.log_prob(actions[..., 0]) + dir_dist.log_prob(actions[..., 1])
        entropy = type_dist.entropy() + dir_dist.entropy()
        return actions, log_prob, entropy, value.view(-1)

    def get_value(self, obs):
        """Get value estimate."""
        _, _, value = self.forward(obs)
        return value.view(-1)

    def get_action(self, obs, mask=None, deterministic=False):
        """Get the action from the observation."""
        action, _, _, _ = self.get_action_and_value(obs, mask, deterministic=deterministic)
        return action

    def get_entropy_details(self, obs, mask):
        """Returns per-head entropy means for Tensorboard logging."""
        action_type_logits, direction_logits, _ = self.forward(obs)
        num_action_types = int(self.action_space.nvec[0])

        if mask is not None:
            mask_2d = mask.view(*mask.shape[:-1], num_action_types, 4)
            action_type_mask = mask_2d.any(dim=-1)
            direction_mask = mask_2d.any(dim=-2)
            action_type_logits = action_type_logits.clone()
            action_type_logits[~action_type_mask] = -1e9
            direction_logits = direction_logits.clone()
            direction_logits[~direction_mask] = -1e9

        type_dist = dist.Categorical(logits=action_type_logits)
        dir_dist = dist.Categorical(logits=direction_logits)

        return {
            "entropy/action_type": type_dist.entropy().mean().item(),
            "entropy/direction": dir_dist.entropy().mean().item(),
        }

    def get_attention_entropy(self, obs):
        """Returns attention entropy and top-1 concentration for tensorboard."""
        was_training = self.training
        self.eval()
        _, _, _, spatial_attn, cross_attn = self.forward_with_attention(obs)
        if was_training:
            self.train()
        stats = _attention_entropy_stats(spatial_attn)
        if cross_attn is not None:
            stats.update(_cross_attention_entropy_stats(cross_attn))
        return stats


def _attention_entropy_stats(attn_map):
    """Compute attention entropy and top-1 weight from a multi-head attention map.

    Args:
        attn_map: (batch, num_heads, H, W) attention weights that sum to 1 over H×W per head.
    Returns:
        dict with combined and per-head 'attention/entropy' and 'attention/top1_weight'.
    """
    batch, num_heads, h, w = attn_map.shape
    flat = attn_map.reshape(batch, num_heads, h * w)  # (batch, num_heads, N)
    log_flat = torch.log(flat + 1e-10)
    per_head_entropy = -(flat * log_flat).sum(dim=2).mean(dim=0)  # (num_heads,)
    per_head_top1 = flat.max(dim=2).values.mean(dim=0)  # (num_heads,)

    stats = {
        "attention/entropy": per_head_entropy.mean().item(),
        "attention/top1_weight": per_head_top1.mean().item(),
    }
    for i in range(num_heads):
        stats[f"attention/head_{i}/entropy"] = per_head_entropy[i].item()
        stats[f"attention/head_{i}/top1_weight"] = per_head_top1[i].item()
    return stats


def _cross_attention_entropy_stats(cross_attn_map):
    """Compute entropy stats for entity-spatial cross-attention.

    Args:
        cross_attn_map: (batch, num_heads, slots, H, W) cross-attention weights.
    Returns:
        dict with 'cross_attention/entropy' and 'cross_attention/top1_weight'.
    """
    batch, num_heads, slots, h, w = cross_attn_map.shape
    # Flatten spatial dims: (batch, num_heads, slots, N)
    flat = cross_attn_map.reshape(batch, num_heads, slots, h * w)
    log_flat = torch.log(flat + 1e-10)
    # Per-entity entropy: (batch, num_heads, slots)
    entity_entropy = -(flat * log_flat).sum(dim=3)
    entity_top1 = flat.max(dim=3).values

    # Average over batch, heads, and entities
    return {
        "cross_attention/entropy": entity_entropy.mean().item(),
        "cross_attention/top1_weight": entity_top1.mean().item(),
    }


def create_network(network, obs_space, action_space, model_kind=None, action_space_name=None):
    """Create a network from a class or instance."""
    if isinstance(network, type) and issubclass(network, Network):
        network = network(obs_space, action_space, model_kind=model_kind, action_space_name=action_space_name)
    elif not isinstance(network, Network):
        raise ValueError("network must be a Network or a Network subclass")

    return network

def _init_models():
    # Get all classes defined in this module
    result = {}
    current_module = sys.modules[__name__]
    for cls_name, cls_obj in inspect.getmembers(current_module, inspect.isclass):
        if issubclass(cls_obj, Network) and cls_obj is not Network:
            result[cls_name] = cls_obj

    return result

NEURAL_NETWORK_DEFINITIONS = _init_models()

def register_neural_network(name, model_class):
    """Register a neural network definition."""
    if not issubclass(model_class, Network):
        raise ValueError("model_class must be a subclass of Network")

    if name in NEURAL_NETWORK_DEFINITIONS:
        raise ValueError(f"Model {name} already exists")

    if model_class == Network:
        raise ValueError("Cannot register the base Network")

    NEURAL_NETWORK_DEFINITIONS[name] = model_class

def get_neural_network(name):
    """Get a model by name."""
    return NEURAL_NETWORK_DEFINITIONS[name]

def _load_triforce_yaml():
    """Load and return the parsed triforce.yaml."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(script_dir, 'triforce.yaml'), encoding='utf-8') as f:
        return yaml.safe_load(f)


class ActionSpaceDefinition:
    """A named action space from triforce.yaml."""
    def __init__(self, name: str, actions: List[str], default: bool = False):
        self.name = name
        self.actions = actions
        self.default = default

    @staticmethod
    def get_all():
        """Load all action space definitions from triforce.yaml."""
        result = {}
        data = _load_triforce_yaml()
        for name, entry in data["action-spaces"].items():
            result[name] = ActionSpaceDefinition(
                name=name,
                actions=entry["actions"],
                default=entry.get("default", False),
            )
        return result

    @staticmethod
    def get(name):
        """Get an action space by name."""
        return ActionSpaceDefinition.get_all()[name]

    @staticmethod
    def get_default():
        """Get the default action space."""
        for asd in ActionSpaceDefinition.get_all().values():
            if asd.default:
                return asd
        raise ValueError("No default action space defined in triforce.yaml")


class ModelKindDefinition:
    """A named model kind from triforce.yaml mapping to a Network subclass."""
    def __init__(self, name: str, network_class: type, default: bool = False):
        self.name = name
        self.network_class = network_class
        self.default = default

    @staticmethod
    def get_all():
        """Load all model kind definitions from triforce.yaml."""
        result = {}
        data = _load_triforce_yaml()
        for name, entry in data["model-kinds"].items():
            result[name] = ModelKindDefinition(
                name=name,
                network_class=get_neural_network(entry["class"]),
                default=entry.get("default", False),
            )
        return result

    @staticmethod
    def get(name):
        """Get a model kind by name."""
        return ModelKindDefinition.get_all()[name]

    @staticmethod
    def get_default():
        """Get the default model kind."""
        for mkd in ModelKindDefinition.get_all().values():
            if mkd.default:
                return mkd
        raise ValueError("No default model kind defined in triforce.yaml")


__all__ = [
    Network.__name__,
    MultiHeadAgent.__name__,
    SharedNatureAgent.__name__,
    ImpalaSharedAgent.__name__,
    ImpalaMultiHeadAgent.__name__,
    register_neural_network.__name__,
    get_neural_network.__name__,
    ActionSpaceDefinition.__name__,
    ModelKindDefinition.__name__,
    ]
