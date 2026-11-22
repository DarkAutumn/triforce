"""``wide``: today's encoders (room embedding, entity transformer, tile convs) under wider and deeper towers, with
today's kind-conditioned heads. The first lab architecture and the template for the next (docs/lab.md)."""

import torch
from torch import nn

from ..model import (ENTITY_FEATURES_KEY, ENTITY_TYPES_KEY, GRID_KEY, ROOM_KEY, _SPECIAL_KEYS, EntityEncoder,
                     GridEncoder, ModelConfig, layer_init)
from ..actions import NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS
from . import register
from .base import Heads, LabNet


class Encoders(nn.Module):
    """``ZeldaPolicy``'s observation encoders, configured by a ``ModelConfig``'s encoder settings: one feature
    vector per observation (room embedding, pooled entities, grid, the float vectors)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        spec = config.obs_spec
        self.room_embedding = nn.Embedding(spec[ROOM_KEY]['vocab'], config.room_embed_dim)
        slots, features = spec[ENTITY_FEATURES_KEY]['shape']
        self.entity_encoder = EntityEncoder(spec[ENTITY_TYPES_KEY]['vocab'], features, config.entity_embed_dim,
                                            config.entity_d_model, config.entity_heads, config.entity_layers,
                                            config.entity_ff_dim, config.entity_out_dim)
        height, width = spec[GRID_KEY]['shape']
        self.grid_encoder = GridEncoder(spec[GRID_KEY]['vocab'], height, width, config)
        self.vector_keys = sorted(k for k in spec if k not in _SPECIAL_KEYS)
        self.output_dim = (config.room_embed_dim + self.entity_encoder.output_dim + self.grid_encoder.output_dim
                           + sum(spec[k]['shape'][0] for k in self.vector_keys))
        del slots

    def forward(self, obs: dict) -> torch.Tensor:
        room = self.room_embedding(obs[ROOM_KEY].long().reshape(-1))
        entities = self.entity_encoder(obs[ENTITY_FEATURES_KEY].float(), obs[ENTITY_TYPES_KEY])
        grid = self.grid_encoder(obs[GRID_KEY])
        return torch.cat([room, entities, grid] + [obs[k].float() for k in self.vector_keys], dim=-1)


def tower(inputs: int, hidden: int, depth: int, last=nn.ReLU) -> nn.Sequential:
    """``depth`` × (linear, LayerNorm, activation), the last activation ``last``."""
    layers: list[nn.Module] = []
    for i in range(depth):
        layers += [layer_init(nn.Linear(inputs if i == 0 else hidden, hidden)), nn.LayerNorm(hidden),
                   last() if i == depth - 1 else nn.ReLU()]
    return nn.Sequential(*layers)


@register
class WidePolicy(LabNet):
    """Separate policy and value towers of ``depth`` layers ``hidden_dim`` wide over ``Encoders``; ``encoders``:
    ``ModelConfig`` encoder settings over its defaults."""
    arch = 'wide'
    defaults = {'encoders': {}, 'hidden_dim': 512, 'depth': 3}

    def __init__(self, config: dict):
        super().__init__(config)
        c = self.config
        self.encoders = Encoders(ModelConfig.from_dict({**c['encoders'], 'obs_spec': c.obs_spec}))
        hidden, depth = c['hidden_dim'], c['depth']
        self.policy_tower = tower(self.encoders.output_dim, hidden, depth, nn.Tanh)
        self.value_tower = tower(self.encoders.output_dim, hidden, depth)
        self.kind_head = layer_init(nn.Linear(hidden, NUM_KINDS), std=0.01)
        self.kind_condition = layer_init(nn.Linear(hidden, hidden))
        self.kind_embedding = nn.Embedding(NUM_KINDS, hidden)
        self.direction_head = layer_init(nn.Linear(hidden, NUM_DIRECTIONS), std=0.01)
        self.duration_head = layer_init(nn.Linear(hidden, NUM_DURATIONS), std=0.01)
        self.value_head = layer_init(nn.Linear(hidden, 1), std=1.0)

    def heads(self, obs: dict, state=None):
        features = self.encoders(obs)
        policy = self.policy_tower(features)
        conditioned = torch.relu(self.kind_condition(policy).unsqueeze(1) + self.kind_embedding.weight)
        return Heads(self.kind_head(policy), self.direction_head(conditioned), self.duration_head(conditioned),
                     self.value_head(self.value_tower(features)).view(-1)), None
