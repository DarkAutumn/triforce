"""``roomheads``: one shared body, a set of action heads a boss, the head chosen by a fixed room-to-head table inside
the network (backlog Track M, "a head a boss"; round36.md "The fights"). No learned selector: the table is the
network's configuration (``heads``: head name -> the room ids, ``obs['room_id']``, it plays), every other room is
played by the shared head.

Body as ``wide``: ``Encoders`` -> policy tower (Tanh last) and value tower, ``depth`` layers ``hidden_dim`` wide;
each head is today's factored set (kind, kind-conditioned direction and duration); one value head for all.
"""

import torch
from torch import nn

from ..actions import NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS
from ..model import ROOM_KEY, ModelConfig, layer_init
from . import register
from .base import Heads, LabNet
from .wide import Encoders, tower

SHARED = 'shared'


@register
class RoomHeadsPolicy(LabNet):
    """``wide``'s body; head 0 (``shared``) plays every room not in ``heads``, head i the rooms of the i-th name of
    ``heads`` in sorted order."""
    arch = 'roomheads'
    defaults = {'encoders': {}, 'hidden_dim': 256, 'depth': 3, 'heads': {}}

    def __init__(self, config: dict):
        super().__init__(config)
        c = self.config
        self.encoders = Encoders(ModelConfig.from_dict({**c['encoders'], 'obs_spec': c.obs_spec}))
        hidden, depth = c['hidden_dim'], c['depth']
        self.policy_tower = tower(self.encoders.output_dim, hidden, depth, nn.Tanh)
        self.value_tower = tower(self.encoders.output_dim, hidden, depth)
        self.head_names = [SHARED, *sorted(c['heads'])]
        table = torch.zeros(c.obs_spec[ROOM_KEY]['vocab'], dtype=torch.long)
        for i, name in enumerate(self.head_names[1:], 1):
            for room in c['heads'][name]:
                if table[room]:
                    raise ValueError(f"room {room} is in two heads ({self.head_names[int(table[room])]}, {name})")
                table[room] = i
        # Derived from the config (rebuilt on load), so not part of the weights.
        self.register_buffer('head_of_room', table, persistent=False)
        count = len(self.head_names)
        self.kind_head = nn.ModuleList(layer_init(nn.Linear(hidden, NUM_KINDS), std=0.01) for _ in range(count))
        self.kind_condition = nn.ModuleList(layer_init(nn.Linear(hidden, hidden)) for _ in range(count))
        self.kind_embedding = nn.ModuleList(nn.Embedding(NUM_KINDS, hidden) for _ in range(count))
        self.direction_head = nn.ModuleList(layer_init(nn.Linear(hidden, NUM_DIRECTIONS), std=0.01)
                                            for _ in range(count))
        self.duration_head = nn.ModuleList(layer_init(nn.Linear(hidden, NUM_DURATIONS), std=0.01)
                                           for _ in range(count))
        self.value_head = layer_init(nn.Linear(hidden, 1), std=1.0)

    def head_index(self, obs: dict) -> torch.Tensor:
        """``[B]``: the head each row's room is played by."""
        return self.head_of_room[obs[ROOM_KEY].long().reshape(-1)]

    def heads(self, obs: dict, state=None):
        features = self.encoders(obs)
        policy = self.policy_tower(features)
        batch = policy.shape[0]
        index = self.head_index(obs)
        kind = policy.new_zeros(batch, NUM_KINDS)
        direction = policy.new_zeros(batch, NUM_KINDS, NUM_DIRECTIONS)
        duration = policy.new_zeros(batch, NUM_KINDS, NUM_DURATIONS)
        for i in range(len(self.head_names)):
            rows = (index == i).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            p = policy[rows]
            conditioned = torch.relu(self.kind_condition[i](p).unsqueeze(1) + self.kind_embedding[i].weight)
            kind = kind.index_copy(0, rows, self.kind_head[i](p))
            direction = direction.index_copy(0, rows, self.direction_head[i](conditioned))
            duration = duration.index_copy(0, rows, self.duration_head[i](conditioned))
        return Heads(kind, direction, duration, self.value_head(self.value_tower(features)).view(-1)), None
