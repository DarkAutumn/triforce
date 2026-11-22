"""``hitnet``: Link as a token attending to the entity tokens, and a hit head on its own tower (round 36,
docs/experiments/round36.md "Observation v7 and the hit network"; experiment 2's recommendation, §8).

The body keeps today's blocks but the entity encoder: the room embedding, the tile grid's convs, the float vectors
(``history``, ``link``, ``objective``, ``room_state``, ``target``). The entities become tokens (any of the type
embedding, ``entity_features``, schema 7's ``entity_rel`` and ``entity_contact``: ``tokens``), Link's ``link`` block
a token of its own, and ``layers`` transformer layers of width ``d_model`` let Link's token attend to them (empty
slots masked); Link's final token, not a mean over the entities, joins the trunk. Policy and value towers and the
kind-conditioned heads are today's.

The hit head (``hit``): its own tower over the trunk, "taking this action now gets Link hit", returned as
``aux['hit']`` ``[B, 352]`` (one logit a joint action, ``kind * 32 + direction * 4 + duration``) for the lab's
``--aux hit=hit:COEFF`` (labels: the action taken, and with ``--try-others`` the look-ahead's verdicts on candidates
not taken, in training only). ``hit: 'kind_direction'`` has one logit per kind and direction slot (11 × 8 = 88),
shared by the four durations: hits land in an action's first frames (half by frame 2, three quarters by frame 4), so
the duration rarely decides the verdict and the 88 outputs see four times the labels each (measured worse on the
supervised screen: round36.md). ``feed``: its probabilities, with the gradient
stopped (the action heads cannot bend the predictor), are an extra input of the policy tower. At play nothing looks
ahead: the head reads the observation alone and the network is served like any other (``serve``).
"""

import math

import torch
from torch import nn

from ..actions import NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS
from ..model import (ENTITY_CONTACT_KEY, ENTITY_FEATURES_KEY, ENTITY_REL_KEY, ENTITY_TYPES_KEY, GRID_KEY, ROOM_KEY,
                     _SPECIAL_KEYS, GridEncoder, ModelConfig, layer_init)
from . import register
from .base import Heads, LabNet
from .wide import tower

JOINT = NUM_KINDS * NUM_DIRECTIONS * NUM_DURATIONS
LINK_KEY = 'link'
TOKEN_GROUPS = {'features': ENTITY_FEATURES_KEY, 'rel': ENTITY_REL_KEY, 'contact': ENTITY_CONTACT_KEY}
HIT_SHAPES = {'joint': JOINT, 'kind_direction': NUM_KINDS * NUM_DIRECTIONS, 'none': 0}


@register
class HitNet(LabNet):
    """Room embedding, grid convs, float vectors and Link's token over the entity tokens -> trunk; trunk (and the hit
    probabilities, stopped, with ``feed``) -> policy tower -> today's heads; trunk -> value tower; trunk -> hit
    tower -> ``aux['hit']``. ``detach_body``: the hit loss trains its own tower only."""
    arch = 'hitnet'
    defaults = {'tokens': ['type', 'features', 'rel', 'contact'], 'room': True, 'd_model': 128, 'layers': 2,
                'attention_heads': 4, 'type_dim': 16, 'link_out': 128, 'hidden_dim': 256, 'depth': 3,
                'hit': 'joint', 'hit_dim': 256, 'hit_depth': 3, 'hit_prior': 0.05, 'feed': False,
                'detach_body': False}

    def __init__(self, config: dict):
        super().__init__(config)
        c, spec = self.config, self.config.obs_spec
        unknown = set(c['tokens']) - set(TOKEN_GROUPS) - {'type'}
        if unknown or not c['tokens']:
            raise ValueError(f"hitnet tokens {c['tokens']}: a non-empty subset of {['type', *TOKEN_GROUPS]}")
        if c['hit'] not in HIT_SHAPES:
            raise ValueError(f"hitnet hit {c['hit']!r}: one of {sorted(HIT_SHAPES)}")
        if c['feed'] and c['hit'] == 'none':
            raise ValueError("hitnet feed needs a hit head")
        self.token_keys = [TOKEN_GROUPS[g] for g in TOKEN_GROUPS if g in c['tokens']]
        missing = [k for k in self.token_keys if k not in spec]
        if missing:
            raise ValueError(f"hitnet tokens need observation keys {missing} (schema 7)")
        d = c['d_model']
        grid = ModelConfig()                                  # today's grid convs and room embedding sizes
        token_in = sum(spec[k]['shape'][1] for k in self.token_keys)
        if 'type' in c['tokens']:
            self.type_embedding = nn.Embedding(spec[ENTITY_TYPES_KEY]['vocab'], c['type_dim'])
            token_in += c['type_dim']
        if c['room']:
            self.room_embedding = nn.Embedding(spec[ROOM_KEY]['vocab'], grid.room_embed_dim)
        self.link_proj = layer_init(nn.Linear(spec[LINK_KEY]['shape'][0], d))
        self.token_proj = layer_init(nn.Linear(token_in, d))
        self.role = nn.Parameter(torch.zeros(2, d))          # Link's token, an entity's token
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=c['attention_heads'], dim_feedforward=2 * d,
                                           batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(layer, num_layers=c['layers'], enable_nested_tensor=False)
        self.link_out = layer_init(nn.Linear(d, c['link_out']))
        height, width = spec[GRID_KEY]['shape']
        self.grid_encoder = GridEncoder(spec[GRID_KEY]['vocab'], height, width, grid)
        self.vector_keys = sorted(k for k in spec if k not in _SPECIAL_KEYS)
        self.feature_dim = ((grid.room_embed_dim if c['room'] else 0) + c['link_out'] + self.grid_encoder.output_dim
                            + sum(spec[k]['shape'][0] for k in self.vector_keys))
        self.hit_outputs = HIT_SHAPES[c['hit']]
        hidden, depth, features = c['hidden_dim'], c['depth'], self.feature_dim
        if self.hit_outputs:
            self.hit_tower = tower(features, c['hit_dim'], c['hit_depth'])
            self.hit_head = nn.Linear(c['hit_dim'], self.hit_outputs)
            nn.init.zeros_(self.hit_head.weight)
            nn.init.constant_(self.hit_head.bias, math.log(c['hit_prior'] / (1 - c['hit_prior'])))
        self.policy_tower = tower(features + (self.hit_outputs if c['feed'] else 0), hidden, depth, nn.Tanh)
        self.value_tower = tower(features, hidden, depth)
        self.kind_head = layer_init(nn.Linear(hidden, NUM_KINDS), std=0.01)
        self.kind_condition = layer_init(nn.Linear(hidden, hidden))
        self.kind_embedding = nn.Embedding(NUM_KINDS, hidden)
        self.direction_head = layer_init(nn.Linear(hidden, NUM_DIRECTIONS), std=0.01)
        self.duration_head = layer_init(nn.Linear(hidden, NUM_DURATIONS), std=0.01)
        self.value_head = layer_init(nn.Linear(hidden, 1), std=1.0)

    def trunk(self, obs: dict) -> torch.Tensor:
        """``[B, feature_dim]``: room, Link's final token, grid, float vectors."""
        ef = obs[ENTITY_FEATURES_KEY].float()
        empty = ef[..., 0] <= 0
        parts = [obs[k].float() for k in self.token_keys]
        if 'type' in self.config['tokens']:
            parts.append(self.type_embedding(obs[ENTITY_TYPES_KEY].long()))
        tokens = self.token_proj(torch.cat(parts, dim=-1)) + self.role[1]
        link = (self.link_proj(obs[LINK_KEY].float()) + self.role[0]).unsqueeze(1)
        mask = torch.cat([torch.zeros_like(empty[:, :1]), empty], dim=1)
        y = self.transformer(torch.cat([link, tokens], dim=1), src_key_padding_mask=mask)
        blocks = [torch.relu(self.link_out(y[:, 0])), self.grid_encoder(obs[GRID_KEY])]
        if self.config['room']:
            blocks.insert(0, self.room_embedding(obs[ROOM_KEY].long().reshape(-1)))
        return torch.cat(blocks + [obs[k].float() for k in self.vector_keys], dim=-1)

    def hit_logits(self, features: torch.Tensor) -> torch.Tensor:
        """The hit head's own outputs (``[B, 352]`` or ``[B, 88]``)."""
        return self.hit_head(self.hit_tower(features.detach() if self.config['detach_body'] else features))

    def heads(self, obs: dict, state=None):
        features = self.trunk(obs)
        aux, policy_input = {}, features
        if self.hit_outputs:
            own = self.hit_logits(features)
            aux['hit'] = own if self.hit_outputs == JOINT else own.repeat_interleave(NUM_DURATIONS, dim=-1)
            if self.config['feed']:
                policy_input = torch.cat([features, torch.sigmoid(own).detach()], dim=-1)
        policy = self.policy_tower(policy_input)
        conditioned = torch.relu(self.kind_condition(policy).unsqueeze(1) + self.kind_embedding.weight)
        return Heads(self.kind_head(policy), self.direction_head(conditioned), self.duration_head(conditioned),
                     self.value_head(self.value_tower(features)).view(-1), aux), None
