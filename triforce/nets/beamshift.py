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

``objectives`` gates the shift by the room's objective kind, the observation's one-hot ``objective_<KIND>`` (round 39,
"Only where the route fights"): ``['FIGHT']`` raises BEAMS only while the objective is a fight (clear the room for
its treasure or a shutter, a boss) and plays the policy alone in a room the route crosses (6:0x78's Red Wizzrobes,
an objective MOVE). Empty: every decision.
"""

import torch

from ..actions import NUM_DIRECTIONS, NUM_DURATIONS
from ..model import MASK_KEY
from ..observation import OBJECTIVE_FEATURES
from ..zelda_enums import ActionKind
from . import build, register
from .base import LabNet
from .hitshift import shift_heads

BEAMS = int(ActionKind.BEAMS)
SLOTS = NUM_DIRECTIONS * NUM_DURATIONS
OBJECTIVE_KEY = 'objective'


@register
class BeamShift(LabNet):
    """``policy`` -> factored heads and value; ``beamnet``'s trunk -> hit tower -> ``aux['beam']`` ``[B, 352]``; the
    BEAMS entries raised by ``weight * sigmoid(beam)`` (gradient stopped), only under ``objectives`` when given (the
    module docstring)."""
    arch = 'beamshift'
    defaults = {'policy': {}, 'beamnet': {}, 'weight': 0.0, 'freeze_beam': True, 'objectives': []}

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
        unknown = sorted(set(c['objectives']) - {f[len('objective_'):] for f in OBJECTIVE_FEATURES
                                                 if f.startswith('objective_')})
        if unknown:
            raise ValueError(f"beamshift: objective kinds {unknown} are not among the observation's")
        if c['objectives'] and OBJECTIVE_KEY not in c.obs_spec:
            raise ValueError("beamshift: gating by objective kind needs the 'objective' observation block")
        self.gate = [OBJECTIVE_FEATURES.index(f'objective_{k}') for k in c['objectives']]
        if c['freeze_beam']:
            for p in self.beam.parameters():
                p.requires_grad_(False)

    @classmethod
    def of(cls, policy, beamnet, weight: float, objectives=()) -> 'BeamShift':
        """The combination of a trained ``policy`` and a trained beam ``hitnet`` (their weights copied, nothing
        trained), on the union of their observation layouts, the shift gated to ``objectives`` (kind names) if any."""
        network = cls({'obs_spec': {**policy.config.obs_spec, **beamnet.config.obs_spec},
                       'policy': policy.config.to_dict(), 'beamnet': beamnet.config.to_dict(),
                       'weight': float(weight), 'objectives': list(objectives)})
        network.policy.load_state_dict(policy.state_dict())
        network.beam.load_state_dict(beamnet.state_dict())
        return network

    def beam_logits(self, obs: dict) -> torch.Tensor:
        """``[B, 352]``: the beam head's joint outputs (the BEAMS entries of duration 0 are the trained ones)."""
        return self.beam.hit_logits(self.beam.trunk(obs))

    def shift(self, beam: torch.Tensor, obs: dict | None = None) -> torch.Tensor:
        """``[B, 352]``: ``weight * sigmoid`` of each direction's duration-0 BEAMS logit on the four BEAMS entries of
        that direction, 0 elsewhere; with ``objectives``, 0 in the rows whose objective kind is not among them."""
        b = beam.detach().float()[:, BEAMS * SLOTS:(BEAMS + 1) * SLOTS].view(-1, NUM_DIRECTIONS, NUM_DURATIONS)
        shift = torch.zeros(beam.shape, dtype=torch.float32, device=beam.device)
        raised = self.config['weight'] * torch.sigmoid(b[..., :1]).expand(-1, -1, NUM_DURATIONS)
        if self.gate:
            kind = obs[OBJECTIVE_KEY].float()
            kind = kind.unsqueeze(0) if kind.dim() == 1 else kind
            raised = raised * (kind[:, self.gate].sum(-1) > 0.5).float().view(-1, 1, 1)
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
        return shift_heads(policy, self.shift(beam, obs), mask), None

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
