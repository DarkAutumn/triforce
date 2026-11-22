"""``hitshift``: a policy and a hit predictor in one network, the predictor shifting the action choice (round 38,
docs/experiments/round38.md).

Two bodies side by side, nothing shared: ``policy`` (any network, today's ``ZeldaPolicy`` by its ``model_config``)
gives the factored action logits and the value; ``hitnet`` (``nets.hitnet.HitNet``, its trunk and per-action hit
head only) gives one hit logit ``h`` a joint action, "taking this action now gets Link hit", from the observation
alone (no emulator trial: it is served like any network). The joint distribution played is

    q(k, d, u)  ∝  p(k, d, u) · exp(shift(k, d, u)),    shift = −weight · softplus(h) = weight · log P(not hit)

(``form: 'logit'``: shift = −weight · h, the raw logit: it also reranks the safe actions by how safe they read).
With ``weight`` 0 the network plays its policy exactly; growing weights refuse what the head reads as a hit. The
Rust actor draws kind, then direction and duration given the kind, so the network returns the factored heads whose
product is closest to q (the marginals ``q(k)``, ``q(d | k)``, ``q(u | k)``: exact on (kind, direction) and (kind,
duration); only the coupling of direction and duration within a kind is lost, measured in round38.md). The act
modes (temperature, relax) then apply to it as to any network.

The shift is computed with the hit logits' gradient stopped: the policy's losses never reach the hit body, and
``freeze_hit`` takes the hit body out of training altogether (the lab's ``--aux hit=hit:C`` trains it by its own
loss otherwise: ``aux['hit']`` is the head's output).
"""

import torch
import torch.nn.functional as F

from ..model import MASK_KEY
from . import build, register
from .base import Heads, LabNet, factored_joint

FORMS = ('softplus', 'logit')


def shift_heads(heads: Heads, shift: torch.Tensor, mask: torch.Tensor) -> Heads:
    """The factored heads whose product is the marginal-matched factorisation of ``joint(heads) + shift``
    (``shift`` ``[B, 352]``, joint order ``kind * 32 + direction * 4 + duration``); entries the mask makes illegal
    keep their logits (no head reads them)."""
    batch, kinds = heads.kind.shape
    dirs, durs = heads.direction.shape[-1], heads.duration.shape[-1]
    q = (factored_joint(heads, mask) + shift).view(batch, kinds, dirs, durs)
    log_k = torch.logsumexp(q.flatten(2), dim=-1)                       # [B, K], -inf for an illegal kind
    finite_k = torch.isfinite(log_k)
    safe_k = torch.where(finite_k, log_k, 0.0)
    log_d = torch.logsumexp(q, dim=-1) - safe_k.unsqueeze(-1)            # [B, K, D]
    log_u = torch.logsumexp(q, dim=-2) - safe_k.unsqueeze(-1)            # [B, K, U]

    def keep(new, old):
        return torch.where(torch.isfinite(new), new, old)

    return Heads(keep(log_k, heads.kind), keep(log_d, heads.direction), keep(log_u, heads.duration), heads.value,
                 dict(heads.aux))


@register
class HitShift(LabNet):
    """``policy`` -> factored heads and value; ``hitnet``'s trunk -> hit tower -> ``aux['hit']`` [B, 352]; the
    heads shifted by ``-weight * softplus(hit)`` (gradient stopped) over the joint action (the module docstring)."""
    arch = 'hitshift'
    defaults = {'policy': {}, 'hitnet': {}, 'weight': 0.0, 'form': 'softplus', 'freeze_hit': False}

    def __init__(self, config: dict):
        super().__init__(config)
        c = self.config
        if c['form'] not in FORMS:
            raise ValueError(f"hitshift form {c['form']!r}: one of {FORMS}")
        if not c['policy'] or not c['hitnet']:
            raise ValueError("hitshift needs the policy's and the hit network's model_config")
        self.policy = build(c['policy'])
        self.hit = build(c['hitnet'])
        if getattr(self.hit, 'hit_outputs', 0) != 352:
            raise ValueError("hitshift needs a hitnet with the joint hit head (hit: 'joint')")
        missing = sorted(set(self.policy.config.obs_spec) - set(c.obs_spec))
        if missing:
            raise ValueError(f"hitshift: the policy reads observation keys the network's obs_spec lacks: {missing}")
        if c['freeze_hit']:
            for p in self.hit.parameters():
                p.requires_grad_(False)

    def hit_logits(self, obs: dict) -> torch.Tensor:
        return self.hit.hit_logits(self.hit.trunk(obs))

    def shift(self, hit: torch.Tensor) -> torch.Tensor:
        h = hit.detach().float()
        return -self.config['weight'] * (F.softplus(h) if self.config['form'] == 'softplus' else h)

    def heads(self, obs: dict, state=None):
        if MASK_KEY not in obs:
            raise ValueError("hitshift needs the action mask among the observation's keys")
        policy, _ = self.policy.heads(obs)
        hit = self.hit_logits(obs)
        policy.aux = {**policy.aux, 'hit': hit}
        if self.config['weight'] == 0:
            return policy, None
        mask = obs[MASK_KEY].bool()
        mask = mask.unsqueeze(0) if mask.dim() == 1 else mask
        return shift_heads(policy, self.shift(hit), mask), None

    def _with_mask(self, obs, mask) -> dict:
        obs = self._batch(obs)
        if mask is not None or MASK_KEY not in obs:
            obs[MASK_KEY] = self._mask(obs, mask)
        return obs

    def evaluate(self, obs, mask=None, actions=None, deterministic=False, noise=None):
        return super().evaluate(self._with_mask(obs, mask), mask, actions, deterministic, noise)

    def joint_log_probs(self, obs, mask=None):
        return super().joint_log_probs(self._with_mask(obs, mask), mask)

    def get_value(self, obs):
        return self.policy.heads(self._batch(obs))[0].value
