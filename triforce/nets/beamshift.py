"""``beamshift``: a policy and a sword-shot predictor in one network, the predictor raising BEAMS where a shot would
hit (round 39, docs/experiments/round39.md "A beam head").

The owner's note of Nov 22 ("Beams at range, not melee"): lined up for a shot, the networks move or wait instead
83-99% of the time, and continuations with the beam rewards did not change it (round38.md). ``hitshift``'s pattern
turned the other way: a supervised head reads from the observation alone "a sword shot fired now in direction d
meets a monster out of melee range" (labels: ``zelda::beams::lined_up``, the projection the beam rewards compute, in
training only: ``lab.record``'s ``beam_lined``), and at play the joint action is

    q(k, d, u)  ∝  p(k, d, u) · exp(shift(k, d, u)),    shift(BEAMS, d, ·) = weight · σ(b(d)),  0 for every other kind

so a shot the head reads as meeting a monster gains up to ``weight`` in log-odds, and nothing else moves. The beam
network is a ``hitnet`` (its trunk and joint head; the BEAMS entries of duration 0 are the trained ones, shared by the
four durations here); ``policy`` is any network, a ``hitshift`` included (a family with a hit head keeps it under the
beam shift). The shift's gradient is stopped and ``freeze_beam`` takes the beam body out of training. The factored
heads are matched to ``q`` as ``hitshift.shift_heads`` does.
"""

import torch

from ..actions import NUM_DIRECTIONS, NUM_DURATIONS
from ..model import MASK_KEY
from ..zelda_enums import ActionKind
from . import build, register
from .base import LabNet
from .hitshift import shift_heads

BEAMS = int(ActionKind.BEAMS)
SLOTS = NUM_DIRECTIONS * NUM_DURATIONS


@register
class BeamShift(LabNet):
    """``policy`` -> factored heads and value; ``beamnet``'s trunk -> hit tower -> ``aux['beam']`` ``[B, 352]``; the
    BEAMS entries raised by ``weight * sigmoid(beam)`` (gradient stopped; the module docstring)."""
    arch = 'beamshift'
    defaults = {'policy': {}, 'beamnet': {}, 'weight': 0.0, 'freeze_beam': True}

    def __init__(self, config: dict):
        super().__init__(config)
        c = self.config
        if not c['policy'] or not c['beamnet']:
            raise ValueError("beamshift needs the policy's and the beam network's model_config")
        self.policy = build(c['policy'])
        self.beam = build(c['beamnet'])
        if getattr(self.beam, 'hit_outputs', 0) != 352:
            raise ValueError("beamshift needs a hitnet with the joint head (hit: 'joint')")
        for part in (self.policy, self.beam):
            missing = sorted(set(part.config.obs_spec) - set(c.obs_spec))
            if missing:
                raise ValueError(f"beamshift: a part reads observation keys the network's obs_spec lacks: {missing}")
        if c['freeze_beam']:
            for p in self.beam.parameters():
                p.requires_grad_(False)

    @classmethod
    def of(cls, policy, beamnet, weight: float) -> 'BeamShift':
        """The combination of a trained ``policy`` and a trained beam ``hitnet`` (their weights copied, nothing
        trained), on the union of their observation layouts."""
        network = cls({'obs_spec': {**policy.config.obs_spec, **beamnet.config.obs_spec},
                       'policy': policy.config.to_dict(), 'beamnet': beamnet.config.to_dict(),
                       'weight': float(weight)})
        network.policy.load_state_dict(policy.state_dict())
        network.beam.load_state_dict(beamnet.state_dict())
        return network

    def beam_logits(self, obs: dict) -> torch.Tensor:
        """``[B, 352]``: the beam head's joint outputs (the BEAMS entries of duration 0 are the trained ones)."""
        return self.beam.hit_logits(self.beam.trunk(obs))

    def shift(self, beam: torch.Tensor) -> torch.Tensor:
        """``[B, 352]``: ``weight * sigmoid`` of each direction's duration-0 BEAMS logit on the four BEAMS entries of
        that direction, 0 elsewhere."""
        b = beam.detach().float()[:, BEAMS * SLOTS:(BEAMS + 1) * SLOTS].view(-1, NUM_DIRECTIONS, NUM_DURATIONS)
        shift = torch.zeros(beam.shape, dtype=torch.float32, device=beam.device)
        raised = self.config['weight'] * torch.sigmoid(b[..., :1]).expand(-1, -1, NUM_DURATIONS)
        shift[:, BEAMS * SLOTS:(BEAMS + 1) * SLOTS] = raised.reshape(-1, SLOTS)
        return shift

    def heads(self, obs: dict, state=None):
        if MASK_KEY not in obs:
            raise ValueError("beamshift needs the action mask among the observation's keys")
        policy, _ = self.policy.heads(obs)
        beam = self.beam_logits(obs)
        policy.aux = {**policy.aux, 'beam': beam}
        if self.config['weight'] == 0:
            return policy, None
        mask = obs[MASK_KEY].bool()
        mask = mask.unsqueeze(0) if mask.dim() == 1 else mask
        return shift_heads(policy, self.shift(beam), mask), None

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
