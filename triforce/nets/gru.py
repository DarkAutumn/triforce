"""``gru``: a network with memory, the reference for recurrent training in the lab (docs/lab.md "Memory").

Today's encoders (``wide.Encoders``) feed a projection and a GRU cell; the cell's state is the network's memory
from decision to decision within an episode (reset at the episode's start). The policy and value towers and today's
kind-conditioned heads read the cell's output. Acting keeps the state per env (``serve.Server``); the learner
trains on sequences of the rollout with truncated back-propagation through time from stored states
(``LabNet.unroll``, ``RolloutBuffer.recompute_sequence``).
"""

import torch
from torch import nn

from ..actions import NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS
from ..model import ModelConfig, layer_init
from . import register
from .base import Heads, LabNet
from .wide import Encoders, tower


@register
class GruPolicy(LabNet):
    """Encoders -> ``hidden_dim`` projection -> GRU cell of ``core_dim`` -> policy and value towers of ``depth``
    layers -> today's heads. State: ``[B, core_dim]``, zeros at an episode's start."""
    arch = 'gru'
    defaults = {'encoders': {}, 'hidden_dim': 256, 'core_dim': 256, 'depth': 1}

    def __init__(self, config: dict):
        super().__init__(config)
        c = self.config
        self.encoders = Encoders(ModelConfig.from_dict({**c['encoders'], 'obs_spec': c.obs_spec}))
        hidden, core = c['hidden_dim'], c['core_dim']
        self.project = tower(self.encoders.output_dim, hidden, 1)
        self.core = nn.GRUCell(hidden, core)
        self.policy_tower = tower(core + hidden, hidden, c['depth'], nn.Tanh)
        self.value_tower = tower(core + hidden, hidden, c['depth'])
        self.kind_head = layer_init(nn.Linear(hidden, NUM_KINDS), std=0.01)
        self.kind_condition = layer_init(nn.Linear(hidden, hidden))
        self.kind_embedding = nn.Embedding(NUM_KINDS, hidden)
        self.direction_head = layer_init(nn.Linear(hidden, NUM_DIRECTIONS), std=0.01)
        self.duration_head = layer_init(nn.Linear(hidden, NUM_DURATIONS), std=0.01)
        self.value_head = layer_init(nn.Linear(hidden, 1), std=1.0)

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, self.config['core_dim'], device=self.device)

    def heads(self, obs: dict, state=None):
        x = self.project(self.encoders(obs))
        if state is None:
            state = self.initial_state(x.shape[0])
        state = self.core(x, state)
        features = torch.cat([state, x], dim=-1)
        policy = self.policy_tower(features)
        conditioned = torch.relu(self.kind_condition(policy).unsqueeze(1) + self.kind_embedding.weight)
        return Heads(self.kind_head(policy), self.direction_head(conditioned), self.duration_head(conditioned),
                     self.value_head(self.value_tower(features)).view(-1)), state
