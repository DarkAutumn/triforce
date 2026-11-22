"""The interface every lab network implements (docs/lab.md "Adding a network").

A network computes, for a batch of observations, the raw logits of the factored action heads and a value:

- ``kind`` ``[B, K]``: the kind head;
- ``direction`` ``[B, K, D]``: row ``k`` is the direction head given kind ``k``;
- ``duration`` ``[B, K, U]``: row ``k`` is the duration head given kind ``k``;
- ``value`` ``[B]``; ``aux``: any auxiliary outputs (a hit predictor, ...), for the network's own losses.

Masks, the forced-kind rule (one legal kind: the kind head contributes nothing), heads that don't apply (a kind's
row all masked), sampling, act modes and log-probs are not the network's business: ``factored_evaluate`` and
``factored_joint`` (torch, for the learner) and ``policy::logits::Heads::act`` (Rust, for acting in the pool)
compute them from the logits exactly as ``ZeldaPolicy`` does from its own.

A network with memory returns a new state beside its heads; the server keeps it per env (``serve.Server``).
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

import torch
from torch import nn

from ..actions import NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS
from ..model import MASK_KEY, PolicyOutput, ZeldaPolicy, _masked_head, split_noise


@dataclass
class Heads:
    """One batch's head logits and value (the module docstring)."""
    kind: torch.Tensor
    direction: torch.Tensor
    duration: torch.Tensor
    value: torch.Tensor
    aux: dict = field(default_factory=dict)

    def flat(self) -> torch.Tensor:
        """``[B, K * (1 + D + U)]``: the action mask's layout (``ExternalQueue.answer``, ``policy::logits``)."""
        batch = self.kind.shape[0]
        return torch.cat([self.kind, self.direction.reshape(batch, -1), self.duration.reshape(batch, -1)], dim=-1)


def split_mask(mask: torch.Tensor):
    """(kind ``[B, K]``, direction ``[B, K, D]``, duration ``[B, K, U]``) parts of a flat action mask."""
    batch = mask.shape[0]
    kind = mask[:, :NUM_KINDS]
    directions = mask[:, NUM_KINDS:NUM_KINDS + NUM_KINDS * NUM_DIRECTIONS].view(batch, NUM_KINDS, NUM_DIRECTIONS)
    durations = mask[:, NUM_KINDS + NUM_KINDS * NUM_DIRECTIONS:].view(batch, NUM_KINDS, NUM_DURATIONS)
    return kind, directions, durations


def factored_evaluate(heads: Heads, mask: torch.Tensor, actions=None, deterministic: bool = False,
                      noise=None) -> PolicyOutput:
    """``ZeldaPolicy.evaluate`` on given logits: samples (or scores ``actions``) under ``mask``."""
    # pylint: disable=too-many-locals
    batch = heads.kind.shape[0]
    rows = torch.arange(batch, device=mask.device)
    kind_mask, dir_masks, dur_masks = split_mask(mask)
    if noise is None and not kind_mask.any(dim=-1).all():
        bad = int((~kind_mask.any(dim=-1)).nonzero()[0, 0])
        raise ValueError(f"Empty kind mask in batch element {bad}: {mask[bad].tolist()}")
    kind_noise = dir_noise = dur_noise = None
    if noise is not None:
        kind_noise, dir_noise, dur_noise = split_noise(torch.as_tensor(noise, device=mask.device))
    if actions is not None:
        actions = torch.as_tensor(actions, device=mask.device).long().view(batch, 3)
    kind_lp, kind_ent, kind = _masked_head(heads.kind, kind_mask, None if actions is None else actions[:, 0],
                                           deterministic, kind_noise)
    dir_lp, dir_ent, direction, dir_applies = ZeldaPolicy._sub_head(  # pylint: disable=protected-access
        heads.direction[rows, kind], dir_masks[rows, kind], None if actions is None else actions[:, 1],
        deterministic, dir_noise)
    dur_lp, dur_ent, duration, dur_applies = ZeldaPolicy._sub_head(  # pylint: disable=protected-access
        heads.duration[rows, kind], dur_masks[rows, kind], None if actions is None else actions[:, 2],
        deterministic, dur_noise)
    kind_applies = kind_mask.sum(dim=-1) > 1
    kind_lp = torch.where(kind_applies, kind_lp, 0.0)
    kind_ent = torch.where(kind_applies, kind_ent, 0.0)
    if actions is None:
        actions = torch.stack([kind, direction, duration], dim=-1)
    return PolicyOutput(actions=actions, log_prob=kind_lp + dir_lp + dur_lp, entropy=kind_ent + dir_ent + dur_ent,
                        value=heads.value,
                        head_entropy={'kind': kind_ent, 'direction': dir_ent, 'duration': dur_ent},
                        head_applies={'kind': kind_applies, 'direction': dir_applies, 'duration': dur_applies},
                        head_log_prob={'kind': kind_lp, 'direction': dir_lp, 'duration': dur_lp})


def factored_joint(heads: Heads, mask: torch.Tensor) -> torch.Tensor:
    """``ZeldaPolicy.joint_log_probs`` on given logits: ``[B, K * D * U]``, ``-inf`` where illegal."""
    batch = heads.kind.shape[0]
    kind_mask, dir_masks, dur_masks = split_mask(mask)
    kind_lp = torch.log_softmax(heads.kind.masked_fill(~kind_mask, -1e9), dim=-1)
    kind_lp = torch.where((kind_mask.sum(dim=-1) > 1).unsqueeze(-1), kind_lp, 0.0)
    dir_lp, dir_legal = ZeldaPolicy._joint_sub_head(heads.direction, dir_masks)  # pylint: disable=protected-access
    dur_lp, dur_legal = ZeldaPolicy._joint_sub_head(heads.duration, dur_masks)  # pylint: disable=protected-access
    joint = (kind_lp[:, :, None, None] + dir_lp[:, :, :, None]) + dur_lp[:, :, None, :]
    legal = kind_mask[:, :, None, None] & dir_legal[:, :, :, None] & dur_legal[:, :, None, :]
    return joint.masked_fill(~legal, float('-inf')).reshape(batch, -1)


class NetConfig(dict):
    """A lab network's ``model_config``: its ``arch``, its ``obs_spec`` and its own settings (plain data)."""

    @property
    def obs_spec(self) -> dict:
        return self['obs_spec']

    def to_dict(self) -> dict:
        return dict(self)


class LabNet(nn.Module):
    """Base of every lab network: subclasses set ``arch`` and ``defaults``, build their layers in ``__init__``
    (call ``super().__init__(config)`` first: ``self.config`` then holds the defaults with the given settings over
    them) and implement ``heads``.  The rest (what the learner, the evaluations and the chains call) is here."""
    arch: ClassVar[str] = ''
    defaults: ClassVar[dict] = {}

    def __init__(self, config: dict):
        super().__init__()
        config = dict(config)
        if config.get('arch', self.arch) != self.arch:
            raise ValueError(f"model_config arch {config['arch']!r} given to {type(self).__name__} ({self.arch!r})")
        unknown = set(config) - set(self.defaults) - {'arch', 'obs_spec'}
        if unknown:
            raise ValueError(f"Unknown {self.arch} config keys: {sorted(unknown)}")
        if not config.get('obs_spec'):
            raise ValueError(f"{self.arch} needs a config with obs_spec")
        self.config = NetConfig({**self.defaults, **config, 'arch': self.arch})

    # ------------------------------------------------------------------ what a network implements

    def heads(self, obs: dict, state: Any = None) -> tuple[Heads, Any]:
        """(``Heads`` of the batch ``obs`` (key -> tensor ``[B, ...]`` on this module's device), the new state).
        ``state``: None for a network without memory; else a tensor ``[B, ...]`` from ``initial_state`` or the last
        call, row for row."""
        raise NotImplementedError

    def initial_state(self, batch: int) -> Optional[torch.Tensor]:  # pylint: disable=unused-argument
        """A fresh episode's state ``[batch, ...]`` (None: no memory)."""
        return None

    # ------------------------------------------------------------------ derived (the learner's interface)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _batch(self, obs) -> dict:
        """Numpy/tensor observations as batched tensors on this module's device (keys outside ``obs_spec`` are
        dropped, except the action mask); ``ZeldaPolicy._batch``."""
        device, result = self.device, {}
        for key, value in obs.items():
            spec = self.config.obs_spec.get(key)
            if spec is None and key != MASK_KEY:
                continue
            tensor = torch.as_tensor(value, device=device)
            if spec is not None and tensor.dim() == len(spec['shape']):
                tensor = tensor.unsqueeze(0)
            result[key] = tensor
        return result

    def _mask(self, obs: dict, mask) -> torch.Tensor:
        mask = torch.as_tensor(obs[MASK_KEY] if mask is None else mask, device=self.device).bool()
        return mask.unsqueeze(0) if mask.dim() == 1 else mask

    def _heads(self, obs: dict) -> Heads:
        """The heads of a batch of independent rows (a recurrent network from its initial state)."""
        batch = next(iter(obs.values())).shape[0]
        state = self.initial_state(batch)
        return self.heads(obs, None if state is None else state.to(self.device))[0]

    def evaluate(self, obs, mask=None, actions=None, deterministic=False, noise=None) -> PolicyOutput:
        """``ZeldaPolicy.evaluate`` (a recurrent network's rows from its initial state: ``_heads``)."""
        obs = self._batch(obs)
        return factored_evaluate(self._heads(obs), self._mask(obs, mask), actions, deterministic, noise)

    def joint_log_probs(self, obs, mask=None) -> torch.Tensor:
        obs = self._batch(obs)
        return factored_joint(self._heads(obs), self._mask(obs, mask))

    def get_value(self, obs) -> torch.Tensor:
        return self._heads(self._batch(obs)).value

    def get_action(self, obs, mask=None, deterministic=False) -> torch.Tensor:
        return self.evaluate(obs, mask, deterministic=deterministic).actions

    def head_entropies(self, obs, mask=None, actions=None) -> dict[str, float]:
        return ZeldaPolicy.head_entropies(self, obs, mask, actions)  # type: ignore[arg-type]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to_native(self):
        """The network as a pool policy: an ``_native.ExternalPolicy`` served from this process (``serve``), on a
        frozen copy of the current weights (the learner may change these meanwhile)."""
        from .serve import default_server  # pylint: disable=import-outside-toplevel
        return default_server().serve_copy(self)
