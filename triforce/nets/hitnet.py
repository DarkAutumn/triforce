"""``hitnet``: Link as a token attending to the entity tokens, and a hit head on its own tower (round 36,
docs/experiments/round36.md "Observation v7 and the hit network"; experiment 2's recommendation, §8).

The body keeps today's blocks but the entity encoder: the room embedding, the tile grid's convs, the float vectors
(``history``, ``link``, ``objective``, ``room_state``, ``target``). The entities become tokens (any of the type
embedding, ``entity_features``, schema 7's ``entity_rel`` and ``entity_contact``, schema 8's ``entity_threat``,
schema 9's ``entity_chase``, schema 10's ``entity_shooter``: ``tokens``; ``threat`` needs a task with ``obs.threat``
on, ``chase`` one with ``obs.chase`` on and ``shooter`` one with ``obs.shooter`` on, the blocks read zeros otherwise),
Link's ``link`` block a token of its own (with the group ``chase``, schema 9's ``chase`` block beside it: where the
monsters are sent), and ``layers`` transformer layers of width ``d_model`` let Link's token attend to them (empty
slots masked); Link's final token, not a mean over the entities, joins the trunk. Policy and value towers and the
kind-conditioned heads are today's. ``room: False`` leaves the room embedding out.

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
from ..model import (CHASE_KEY, ENTITY_CHASE_KEY, ENTITY_CONTACT_KEY, ENTITY_FEATURES_KEY, ENTITY_REL_KEY,
                     ENTITY_SHOOTER_KEY, ENTITY_THREAT_KEY, ENTITY_TYPES_KEY, GRID_KEY, ROOM_KEY, _SPECIAL_KEYS,
                     GridEncoder, ModelConfig, layer_init)
from . import register
from .base import Heads, LabNet
from .wide import tower

JOINT = NUM_KINDS * NUM_DIRECTIONS * NUM_DURATIONS
LINK_KEY = 'link'
# In this order an entity token's columns are concatenated: a group added later goes last, so a network without it
# reads its blocks where it always did.
TOKEN_GROUPS = {'features': ENTITY_FEATURES_KEY, 'rel': ENTITY_REL_KEY, 'contact': ENTITY_CONTACT_KEY,
                'threat': ENTITY_THREAT_KEY, 'chase': ENTITY_CHASE_KEY, 'shooter': ENTITY_SHOOTER_KEY}
# Blocks a token group adds to Link's own token.
LINK_GROUPS = {'chase': CHASE_KEY}
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
        self.link_keys = [LINK_KEY] + [LINK_GROUPS[g] for g in LINK_GROUPS if g in c['tokens']]
        missing = [k for k in self.token_keys + self.link_keys if k not in spec]
        if missing:
            raise ValueError(f"hitnet tokens need observation keys {missing} (schema 7 and later)")
        d = c['d_model']
        grid = ModelConfig()                                  # today's grid convs and room embedding sizes
        token_in = sum(spec[k]['shape'][1] for k in self.token_keys)
        if 'type' in c['tokens']:
            self.type_embedding = nn.Embedding(spec[ENTITY_TYPES_KEY]['vocab'], c['type_dim'])
            token_in += c['type_dim']
        if c['room']:
            self.room_embedding = nn.Embedding(spec[ROOM_KEY]['vocab'], grid.room_embed_dim)
        self.link_proj = layer_init(nn.Linear(sum(spec[k]['shape'][0] for k in self.link_keys), d))
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
        link_in = torch.cat([obs[k].float() for k in self.link_keys], dim=-1)
        link = (self.link_proj(link_in) + self.role[0]).unsqueeze(1)
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


def _blocks(tokens, spec: dict, type_dim: int):
    """The column blocks of the token projection's and Link's projection's inputs, in order: ``(group, width)``."""
    token = [(g, spec[TOKEN_GROUPS[g]]['shape'][1]) for g in TOKEN_GROUPS if g in tokens]
    if 'type' in tokens:
        token.append(('type', type_dim))
    link = [(LINK_KEY, spec[LINK_KEY]['shape'][0])] + [(g, spec[LINK_GROUPS[g]]['shape'][0]) for g in LINK_GROUPS
                                                        if g in tokens]
    return {'token_proj.weight': token, 'link_proj.weight': link}


def widen_tokens(network: HitNet, tokens, obs_spec: dict) -> HitNet:
    """``network`` rebuilt to read the token groups ``tokens`` (its own and more) on the layout ``obs_spec``: every
    weight copied, the projections' input columns of the added groups zero, so the widened network computes exactly
    what ``network`` does whatever the new blocks hold (docs/lab.md "Continuing with a new input"), and training grows
    them from there.  A dropped group, a block or layer whose shape the new layout changes, or a loaded parameter the
    rebuilt network has no place for is refused (``ValueError``)."""
    old = network.config
    dropped = sorted(set(old['tokens']) - set(tokens))
    unknown = sorted(set(tokens) - set(TOKEN_GROUPS) - {'type'})
    if dropped or unknown:
        raise ValueError(f"hitnet tokens {old['tokens']} can't become {list(tokens)}: "
                         + (f"dropped {dropped}" if dropped else f"unknown {unknown}"))
    ordered = [g for g in ('type', *TOKEN_GROUPS) if g in set(tokens)]
    wide = HitNet({**old.to_dict(), 'tokens': ordered, 'obs_spec': obs_spec})
    before, after = _blocks(old['tokens'], old.obs_spec, old['type_dim']), _blocks(ordered, obs_spec, old['type_dim'])
    state = network.state_dict()
    lost = sorted(set(state) - set(wide.state_dict()))
    if lost:
        raise ValueError(f"widening would drop the loaded parameters {lost}: the rebuilt network has no place for them")
    with torch.no_grad():
        for name, value in wide.state_dict().items():
            if name in after:
                starts, at = {}, 0
                for group, width in after[name]:
                    starts[group], at = (at, width), at + width
                value.zero_()
                at = 0
                for group, width in before[name]:
                    start, new_width = starts[group]
                    if new_width != width:
                        raise ValueError(f"hitnet group {group!r} is {width} columns wide in the network, {new_width} "
                                         "in the new layout")
                    value[:, start:start + width] = state[name][:, at:at + width]
                    at += width
            elif name not in state:
                if not name.startswith('type_embedding.'):
                    raise ValueError(f"widening adds {name}, which only the type embedding may")
            elif value.shape != state[name].shape:
                raise ValueError(f"widening changes {name} from {tuple(state[name].shape)} to {tuple(value.shape)}: the "
                                 "new layout is not the network's plus token blocks")
            else:
                value.copy_(state[name])
    return wide.to(next(network.parameters()).device)
