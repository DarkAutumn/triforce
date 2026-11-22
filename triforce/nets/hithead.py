"""``hithead``: a hit-prediction head inside the network whose output feeds the action choice (backlog Track M,
experiment 6; docs/lab.md "Auxiliary targets").

Today's body (``wide.Encoders``, policy and value towers, the kind-conditioned heads) plus a hit tower and head: one
logit per joint action (352), "taking this action now gets Link hit". The head is trained only by its own loss (a
run names it: ``--aux hit=hit:COEFF``, labels from the action taken and, with ``--try-others``, the look-ahead's
verdicts on candidates not taken, in training only). Its probabilities, with the gradient stopped (so the action
heads cannot bend the predictor), are an extra input to the policy tower (``hit_input``; off: the head is an
auxiliary target only). At play the network is served like any other: nothing looks ahead, the head predicts from
the observation alone.
"""

import torch
from torch import nn

from ..actions import NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS
from ..model import ModelConfig, layer_init
from . import register
from .base import Heads, LabNet
from .wide import Encoders, tower

JOINT = NUM_KINDS * NUM_DIRECTIONS * NUM_DURATIONS


@register
class HitHeadPolicy(LabNet):
    """Encoders -> (hit tower -> 352 hit logits, ``aux['hit']``); (encoders, stopped hit probabilities) -> policy
    tower -> today's heads; encoders -> value tower -> value. ``detach_body``: the hit loss doesn't train the
    encoders either."""
    arch = 'hithead'
    defaults = {'encoders': {}, 'hidden_dim': 256, 'depth': 2, 'hit_dim': 256, 'hit_input': True,
                'detach_body': False}

    def __init__(self, config: dict):
        super().__init__(config)
        c = self.config
        self.encoders = Encoders(ModelConfig.from_dict({**c['encoders'], 'obs_spec': c.obs_spec}))
        hidden, depth, features = c['hidden_dim'], c['depth'], self.encoders.output_dim
        self.hit_tower = tower(features, c['hit_dim'], 1)
        self.hit_head = layer_init(nn.Linear(c['hit_dim'], JOINT), std=0.01)
        self.policy_tower = tower(features + (JOINT if c['hit_input'] else 0), hidden, depth, nn.Tanh)
        self.value_tower = tower(features, hidden, depth)
        self.kind_head = layer_init(nn.Linear(hidden, NUM_KINDS), std=0.01)
        self.kind_condition = layer_init(nn.Linear(hidden, hidden))
        self.kind_embedding = nn.Embedding(NUM_KINDS, hidden)
        self.direction_head = layer_init(nn.Linear(hidden, NUM_DIRECTIONS), std=0.01)
        self.duration_head = layer_init(nn.Linear(hidden, NUM_DURATIONS), std=0.01)
        self.value_head = layer_init(nn.Linear(hidden, 1), std=1.0)

    def heads(self, obs: dict, state=None):
        features = self.encoders(obs)
        hit = self.hit_head(self.hit_tower(features.detach() if self.config['detach_body'] else features))
        policy_input = features
        if self.config['hit_input']:
            policy_input = torch.cat([features, torch.sigmoid(hit).detach()], dim=-1)
        policy = self.policy_tower(policy_input)
        conditioned = torch.relu(self.kind_condition(policy).unsqueeze(1) + self.kind_embedding.weight)
        return Heads(self.kind_head(policy), self.direction_head(conditioned), self.duration_head(conditioned),
                     self.value_head(self.value_tower(features)).view(-1), {'hit': hit}), None
