"""The v2 policy network (design §7.3): structured observations in, factored timed-option actions out.

No vision.  Every observation block from ``observation.ObservationBuilder`` gets its own small encoder:

- ``room_id`` → ``nn.Embedding`` (dim ~16).
- ``entity_types`` + ``entity_features`` → type embedding, projection, 1–2 layer transformer encoder over the
  slots with a padding mask built from the presence feature (``entity_features[..., 0]``), mean-pooled over
  present slots.  An all-empty row unmasks slot 0 so attention never sees an all ``-inf`` row.
- ``grid`` → tile-class embedding, then either a small conv stack or flatten + linear
  (``ModelConfig.grid_encoder``: ``conv`` | ``linear``).
- Every other float block (``room_state``, ``link``, ``history``, ``objective``) is concatenated as is.

The concatenated features feed separate policy and value MLP towers (v1 ``MlpExtractor``).  Action heads are
factored: kind (K) → direction (8) and duration (4), where the direction/duration heads are conditioned on the
kind (a per-kind embedding is added to a projection of the policy features).

Masking follows ``actions.ActionMask.to_flat()``: the kind mask comes from ``mask[:K]``; the direction and
duration masks are the rows of the sampled (or given) kind.  A head whose row is all False for that kind
does not apply: its log-prob and entropy are 0 and its sampled index is 0 (matching ``Action.to_indices``).
When only one kind is legal, the kind head is excluded from log-prob and entropy (v1's forced-type rule).
"""

from dataclasses import asdict, dataclass, field, fields
from typing import Optional

import numpy as np
import torch
from torch import nn
from gymnasium import spaces

from .actions import MASK_SIZE, NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS

MASKED_LOGIT = -1e9

# Observation keys with special encoders; every other non-mask key is a float vector concatenated directly.
ROOM_KEY = 'room_id'
ENTITY_TYPES_KEY = 'entity_types'
ENTITY_FEATURES_KEY = 'entity_features'
GRID_KEY = 'grid'
MASK_KEY = 'action_mask'
_SPECIAL_KEYS = (ROOM_KEY, ENTITY_TYPES_KEY, ENTITY_FEATURES_KEY, GRID_KEY, MASK_KEY)


@dataclass
class ModelConfig:
    """Network hyperparameters plus the observation layout the network was built for.

    ``obs_spec`` maps each observation key to ``{'shape': [...], 'vocab': int | None}`` and is filled from the
    observation space, so a checkpoint's ``model_config`` is enough to rebuild the network.
    """
    room_embed_dim: int = 16
    entity_embed_dim: int = 16
    entity_d_model: int = 64
    entity_heads: int = 4
    entity_layers: int = 1
    entity_ff_dim: int = 128
    entity_out_dim: int = 64
    tile_embed_dim: int = 4
    grid_encoder: str = 'conv'
    grid_channels: tuple = (16, 32, 32)
    grid_out_dim: int = 128
    hidden_dim: int = 256
    obs_spec: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Plain-data form for checkpoints."""
        result = asdict(self)
        result['grid_channels'] = list(self.grid_channels)
        return result

    @staticmethod
    def from_dict(data: Optional[dict]) -> 'ModelConfig':
        """Inverse of to_dict.  Unknown keys raise so typos in configs are caught."""
        data = dict(data or {})
        known = {f.name for f in fields(ModelConfig)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown model config keys: {sorted(unknown)}")
        if 'grid_channels' in data:
            data['grid_channels'] = tuple(data['grid_channels'])
        return ModelConfig(**data)


def _space_vocab(space) -> Optional[int]:
    """Embedding vocabulary size for an integer space, or None for float spaces."""
    if isinstance(space, spaces.MultiDiscrete):
        return int(np.max(space.nvec))
    if isinstance(space, spaces.Discrete):
        return int(space.n)
    if isinstance(space, spaces.Box) and np.issubdtype(space.dtype, np.integer):
        return int(np.max(space.high)) + 1
    return None


def obs_spec_from_space(observation_space: spaces.Dict) -> dict:
    """Shapes and vocab sizes of every observation key (see ModelConfig.obs_spec)."""
    spec = {}
    for key, space in observation_space.spaces.items():
        shape = list(space.shape) if space.shape is not None else []
        if isinstance(space, spaces.Discrete):
            shape = [1]
        spec[key] = {'shape': shape, 'vocab': _space_vocab(space)}
    for key in (ROOM_KEY, ENTITY_TYPES_KEY, GRID_KEY):
        if key not in spec or spec[key]['vocab'] is None:
            raise ValueError(f"Observation space needs an integer '{key}' block")
    if ENTITY_FEATURES_KEY not in spec:
        raise ValueError(f"Observation space needs '{ENTITY_FEATURES_KEY}'")
    return spec


def layer_init(layer: nn.Module, std: float = float(np.sqrt(2)), bias_const: float = 0.0) -> nn.Module:
    """Orthogonal init (v1 Network.layer_init)."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def _masked_head(logits, mask, given, deterministic):
    """(log_prob [B], entropy [B], choice [B]) of a categorical head with illegal logits set to MASKED_LOGIT.

    Hand-rolled instead of ``torch.distributions.Categorical``: it is twice as fast for single-step rollouts.
    """
    log_probs = torch.log_softmax(logits.masked_fill(~mask, MASKED_LOGIT), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    if given is not None:
        choice = given
    elif deterministic:
        choice = log_probs.argmax(dim=-1)
    else:
        choice = torch.multinomial(probs, 1).squeeze(-1)
    return log_probs.gather(-1, choice.unsqueeze(-1)).squeeze(-1), entropy, choice


class MlpExtractor(nn.Module):
    """Separate policy and value towers (ported from v1)."""
    def __init__(self, input_size: int, hidden: int):
        super().__init__()
        self.output_dim = hidden
        self.policy_net = nn.Sequential(
            layer_init(nn.Linear(input_size, hidden)), nn.LayerNorm(hidden), nn.ReLU(),
            layer_init(nn.Linear(hidden, hidden)), nn.LayerNorm(hidden), nn.ReLU(),
            layer_init(nn.Linear(hidden, hidden)), nn.LayerNorm(hidden), nn.Tanh(),
        )
        self.value_net = nn.Sequential(
            layer_init(nn.Linear(input_size, hidden)), nn.LayerNorm(hidden), nn.ReLU(),
            layer_init(nn.Linear(hidden, hidden)), nn.LayerNorm(hidden), nn.ReLU(),
            layer_init(nn.Linear(hidden, hidden)), nn.LayerNorm(hidden), nn.ReLU(),
        )

    def forward(self, features):
        """(policy_features, value_features)."""
        return self.policy_net(features), self.value_net(features)


class EntityEncoder(nn.Module):
    """Transformer over entity slots with a presence padding mask, mean-pooled over present slots (v1 port)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(self, num_types, num_features, embed_dim, d_model, heads, layers, ff_dim, out_dim):
        super().__init__()
        self.type_embedding = nn.Embedding(num_types, embed_dim)
        self.input_proj = layer_init(nn.Linear(num_features + embed_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=ff_dim,
                                                   batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers, enable_nested_tensor=False)
        self.output_proj = layer_init(nn.Linear(d_model, out_dim))
        self.output_dim = out_dim

    def forward(self, entity_features, entity_types):
        """(B, S, F), (B, S) → (B, out_dim)."""
        embedded = self.type_embedding(entity_types.long())
        projected = self.input_proj(torch.cat([entity_features, embedded], dim=-1))

        empty = entity_features[..., 0] <= 0
        safe = empty.clone()
        safe[empty.all(dim=1), 0] = False
        attended = self.transformer(projected, src_key_padding_mask=safe)

        present = (~empty).unsqueeze(-1).to(attended.dtype)
        pooled = (attended * present).sum(dim=1) / present.sum(dim=1).clamp(min=1.0)
        return torch.relu(self.output_proj(pooled))


class GridEncoder(nn.Module):
    """Tile-class grid → vector.  ``conv``: 3 small conv layers; ``linear``: flatten + linear."""
    def __init__(self, num_classes, height, width, config: ModelConfig):
        super().__init__()
        self.tile_embedding = nn.Embedding(num_classes, config.tile_embed_dim)
        self.kind = config.grid_encoder
        if self.kind == 'conv':
            layers = []
            channels = config.tile_embed_dim
            for i, out_channels in enumerate(config.grid_channels):
                stride = 1 if i == 0 else 2
                layers += [layer_init(nn.Conv2d(channels, out_channels, 3, stride=stride, padding=1)), nn.ReLU()]
                channels = out_channels
            self.body = nn.Sequential(*layers, nn.Flatten())
            with torch.no_grad():
                flat = self.body(torch.zeros(1, config.tile_embed_dim, height, width)).shape[1]
        elif self.kind == 'linear':
            self.body = nn.Flatten()
            flat = config.tile_embed_dim * height * width
        else:
            raise ValueError(f"grid_encoder must be 'conv' or 'linear', not {self.kind!r}")
        self.head = nn.Sequential(layer_init(nn.Linear(flat, config.grid_out_dim)), nn.ReLU())
        self.output_dim = config.grid_out_dim

    def forward(self, grid):
        """(B, H, W) int → (B, out)."""
        embedded = self.tile_embedding(grid.long()).permute(0, 3, 1, 2)
        return self.head(self.body(embedded))


@dataclass
class PolicyOutput:
    """Everything one policy evaluation produces (log-probs/entropies per head for diagnostics)."""
    actions: torch.Tensor          # [B, 3] int64
    log_prob: torch.Tensor         # [B] joint log-prob over the heads that apply
    entropy: torch.Tensor          # [B] summed entropy over the heads that apply
    value: torch.Tensor            # [B]
    head_entropy: dict             # name → [B] entropy (0 where the head doesn't apply)
    head_applies: dict             # name → [B] bool


class ZeldaPolicy(nn.Module):
    """Actor-critic for one specialist.  Same architecture for every specialist; only weights differ."""

    def __init__(self, observation_space: Optional[spaces.Dict] = None, config: ModelConfig | dict | None = None):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        config = config or ModelConfig()
        if observation_space is not None:
            config.obs_spec = obs_spec_from_space(observation_space)
        if not config.obs_spec:
            raise ValueError("ZeldaPolicy needs an observation space or a config with obs_spec")
        self.config = config
        spec = config.obs_spec

        self.room_embedding = nn.Embedding(spec[ROOM_KEY]['vocab'], config.room_embed_dim)
        num_slots, num_features = spec[ENTITY_FEATURES_KEY]['shape']
        assert spec[ENTITY_TYPES_KEY]['shape'] == [num_slots], "entity_types and entity_features slot mismatch"
        self.entity_encoder = EntityEncoder(spec[ENTITY_TYPES_KEY]['vocab'], num_features, config.entity_embed_dim,
                                            config.entity_d_model, config.entity_heads, config.entity_layers,
                                            config.entity_ff_dim, config.entity_out_dim)
        height, width = spec[GRID_KEY]['shape']
        self.grid_encoder = GridEncoder(spec[GRID_KEY]['vocab'], height, width, config)

        self.vector_keys = sorted(k for k, v in spec.items() if k not in _SPECIAL_KEYS)
        for key in self.vector_keys:
            if spec[key]['vocab'] is not None or len(spec[key]['shape']) != 1:
                raise ValueError(f"Observation key '{key}' must be a 1-D float vector")
        vector_size = sum(spec[k]['shape'][0] for k in self.vector_keys)

        features = config.room_embed_dim + self.entity_encoder.output_dim + self.grid_encoder.output_dim + vector_size
        self.feature_dim = features
        hidden = config.hidden_dim
        self.mlp_extractor = MlpExtractor(features, hidden)

        self.kind_head = layer_init(nn.Linear(hidden, NUM_KINDS), std=0.01)
        self.kind_condition = layer_init(nn.Linear(hidden, hidden))
        self.kind_embedding = nn.Embedding(NUM_KINDS, hidden)
        self.direction_head = layer_init(nn.Linear(hidden, NUM_DIRECTIONS), std=0.01)
        self.duration_head = layer_init(nn.Linear(hidden, NUM_DURATIONS), std=0.01)
        self.value_head = layer_init(nn.Linear(hidden, 1), std=1.0)

    # ------------------------------------------------------------------ inputs

    def _batch(self, obs) -> dict:
        """Converts numpy/tensor observations to batched tensors on this module's device."""
        device = self.kind_head.weight.device
        result = {}
        for key, value in obs.items():
            tensor = torch.as_tensor(value, device=device)
            spec = self.config.obs_spec.get(key)
            if spec is not None and tensor.dim() == len(spec['shape']):
                tensor = tensor.unsqueeze(0)
            result[key] = tensor
        return result

    def _features(self, obs: dict) -> torch.Tensor:
        room = self.room_embedding(obs[ROOM_KEY].long().reshape(-1))
        entities = self.entity_encoder(obs[ENTITY_FEATURES_KEY].float(), obs[ENTITY_TYPES_KEY])
        grid = self.grid_encoder(obs[GRID_KEY])
        blocks = [room, entities, grid] + [obs[k].float() for k in self.vector_keys]
        return torch.cat(blocks, dim=-1)

    def forward(self, obs):
        """(policy_features [B, H], value [B])."""
        obs = self._batch(obs)
        policy_features, value_features = self.mlp_extractor(self._features(obs))
        return policy_features, self.value_head(value_features).view(-1)

    def get_value(self, obs) -> torch.Tensor:
        """Value estimate [B]."""
        obs = self._batch(obs)
        _, value_features = self.mlp_extractor(self._features(obs))
        return self.value_head(value_features).view(-1)

    # ------------------------------------------------------------------ policy

    def evaluate(self, obs, mask=None, actions=None, deterministic=False) -> PolicyOutput:
        """Samples (or scores the given) factored actions under the action mask."""
        # pylint: disable=too-many-locals
        obs = self._batch(obs)
        if mask is None:
            mask = obs[MASK_KEY]
        mask = torch.as_tensor(mask, device=self.kind_head.weight.device).bool()
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        assert mask.shape[-1] == MASK_SIZE, f"mask has {mask.shape[-1]} entries, expected {MASK_SIZE}"

        policy_features, value_features = self.mlp_extractor(self._features(obs))
        value = self.value_head(value_features).view(-1)
        batch = policy_features.shape[0]
        rows = torch.arange(batch, device=mask.device)

        kind_mask = mask[:, :NUM_KINDS]
        dir_masks = mask[:, NUM_KINDS:NUM_KINDS + NUM_KINDS * NUM_DIRECTIONS].view(batch, NUM_KINDS, NUM_DIRECTIONS)
        dur_masks = mask[:, NUM_KINDS + NUM_KINDS * NUM_DIRECTIONS:].view(batch, NUM_KINDS, NUM_DURATIONS)
        if not kind_mask.any(dim=-1).all():
            bad = int((~kind_mask.any(dim=-1)).nonzero()[0, 0])
            raise ValueError(f"Empty kind mask in batch element {bad}: {mask[bad].tolist()}")

        if actions is not None:
            actions = torch.as_tensor(actions, device=mask.device).long().view(batch, 3)
        kind_lp, kind_ent, kind = _masked_head(self.kind_head(policy_features), kind_mask,
                                               None if actions is None else actions[:, 0], deterministic)

        conditioned = torch.relu(self.kind_condition(policy_features) + self.kind_embedding(kind))
        dir_lp, dir_ent, direction, dir_applies = self._sub_head(
            self.direction_head(conditioned), dir_masks[rows, kind],
            None if actions is None else actions[:, 1], deterministic)
        dur_lp, dur_ent, duration, dur_applies = self._sub_head(
            self.duration_head(conditioned), dur_masks[rows, kind],
            None if actions is None else actions[:, 2], deterministic)

        # v1's forced-type rule: with one legal kind, the kind head contributes nothing.
        kind_applies = kind_mask.sum(dim=-1) > 1
        kind_lp = torch.where(kind_applies, kind_lp, 0.0)
        kind_ent = torch.where(kind_applies, kind_ent, 0.0)

        if actions is None:
            actions = torch.stack([kind, direction, duration], dim=-1)

        return PolicyOutput(actions=actions,
                            log_prob=kind_lp + dir_lp + dur_lp,
                            entropy=kind_ent + dir_ent + dur_ent,
                            value=value,
                            head_entropy={'kind': kind_ent, 'direction': dir_ent, 'duration': dur_ent},
                            head_applies={'kind': kind_applies, 'direction': dir_applies, 'duration': dur_applies})

    @staticmethod
    def _sub_head(logits, row_mask, given, deterministic):
        """Log-prob, entropy, chosen index, applies for a kind-conditioned head; zeros where the row is empty."""
        applies = row_mask.any(dim=-1)
        safe_mask = row_mask | ~applies.unsqueeze(-1)
        log_prob, entropy, choice = _masked_head(logits, safe_mask, given, deterministic)
        return (torch.where(applies, log_prob, 0.0), torch.where(applies, entropy, 0.0),
                torch.where(applies, choice, 0), applies)

    def get_action_and_value(self, obs, mask=None, actions=None, deterministic=False):
        """(actions [B,3], log_prob [B], entropy [B], value [B]) — the v1 PPO interface."""
        out = self.evaluate(obs, mask, actions, deterministic)
        return out.actions, out.log_prob, out.entropy, out.value

    def get_action(self, obs, mask=None, deterministic=False) -> torch.Tensor:
        """Actions [B, 3]."""
        return self.evaluate(obs, mask, deterministic=deterministic).actions

    def head_entropies(self, obs, mask=None, actions=None) -> dict[str, float]:
        """Mean entropy per head over the samples where that head applies (for logging)."""
        out = self.evaluate(obs, mask, actions)
        result = {}
        for name, entropy in out.head_entropy.items():
            applies = out.head_applies[name]
            result[f"entropy/{name}"] = float(entropy[applies].mean()) if applies.any() else 0.0
            result[f"applies/{name}"] = float(applies.float().mean())
        return result

    def parameter_count(self) -> int:
        """Trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: ModelConfig | dict, observation_space: Optional[spaces.Dict] = None) -> ZeldaPolicy:
    """Builds a policy from a checkpoint's model_config (obs_spec inside) or a config plus space."""
    return ZeldaPolicy(observation_space, config)
