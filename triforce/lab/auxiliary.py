"""Auxiliary targets: losses on a network's named auxiliary outputs (docs/lab.md "Auxiliary targets").

The rollout carries, per decision, what the action taken did (``RolloutBuffer``: ``health_lost``, ``hits``,
``frames``) and, with ``tried`` (``EnvPool.rollout``; ``--try-others K:P``), the look-ahead's verdict on K candidates
NOT taken on a share P of the decisions (``tried_actions``, ``tried_lost``; training only). A ``LabNet`` returns
named auxiliary outputs in ``Heads.aux``; a run names a loss per output: ``--aux OUTPUT=TARGET:COEFF``.

Targets (per decision):

- ``hit``: the action lost health (``health_lost > 0``); binary, BCE on logits. On a per-action output
  ``[B, 352]`` the tried candidates' verdicts are labels too (their joint actions' logits).
- ``damaged``: the action dealt a monster a hit (``hits > 0``); binary.
- ``hearts_lost``: hearts the action lost (net, clamped at 0); regression, ½ squared error.
- ``frames``: the frames the action took, / 16; regression.

An output ``[B]`` is the taken action's prediction; ``[B, 352]`` is one per joint action (``kind * 32 + direction *
4 + duration``), read at the taken action (and the tried ones for ``hit``).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..actions import NUM_DIRECTIONS, NUM_DURATIONS

TARGETS = ('hit', 'damaged', 'hearts_lost', 'frames')
BINARY = ('hit', 'damaged')
FRAME_SCALE = 16.0


@dataclass(frozen=True)
class AuxLoss:
    """``output`` (a key of ``Heads.aux``) against ``target`` (``TARGETS``) with weight ``coeff``."""
    output: str
    target: str
    coeff: float

    @staticmethod
    def parse(text: str) -> 'AuxLoss':
        """``OUTPUT=TARGET:COEFF``."""
        output, _, rest = text.partition('=')
        target, _, coeff = rest.partition(':')
        if not output or target not in TARGETS or not coeff:
            raise ValueError(f"--aux {text!r}: OUTPUT=TARGET:COEFF with TARGET one of {TARGETS}")
        return AuxLoss(output, target, float(coeff))


def joint_index(actions: torch.Tensor) -> torch.Tensor:
    """``[..., 3]`` (kind, direction, duration) -> the joint index ``[...]``."""
    return (actions[..., 0] * NUM_DIRECTIONS + actions[..., 1]) * NUM_DURATIONS + actions[..., 2]


def target_values(target: str, batch: dict, rows) -> torch.Tensor:
    """The taken actions' targets ``[n]`` (float) of ``rows`` of a flattened batch."""
    if target == 'hit':
        return (batch['health_lost'][rows] > 0).float()
    if target == 'damaged':
        return (batch['hits'][rows] > 0).float()
    if target == 'hearts_lost':
        return batch['health_lost'][rows].clamp(min=0.0)
    return batch['frames'][rows] / FRAME_SCALE


def _loss(target: str, prediction: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if target in BINARY:
        return F.binary_cross_entropy_with_logits(prediction, value, reduction='sum')
    return 0.5 * (prediction - value).pow(2).sum()


def aux_loss(losses: list[AuxLoss], aux: dict, batch: dict, rows) -> tuple[torch.Tensor, dict]:
    """The weighted sum of ``losses`` on a minibatch (``aux``: its outputs, row for row with ``rows`` of the
    flattened ``batch``) and stats ``losses/aux_<output>`` (each loss's mean over its labels) and
    ``charts/aux_<output>_labels`` (labels a row: above 1 where tried candidates add theirs)."""
    total = torch.zeros((), device=batch['actions'].device)
    stats = {}
    for spec in losses:
        if spec.output not in (aux or {}):
            raise ValueError(f"--aux {spec.output}: the network has no auxiliary output {spec.output!r} "
                             f"(it has {sorted(aux or {})})")
        prediction = aux[spec.output]
        value = target_values(spec.target, batch, rows)
        if prediction.dim() == 2:
            taken = joint_index(batch['actions'][rows]).unsqueeze(-1)
            loss = _loss(spec.target, prediction.gather(1, taken).squeeze(-1), value)
            labels = value.numel()
            if spec.target == 'hit' and 'tried_lost' in batch:
                lost = batch['tried_lost'][rows]                     # [n, K]: -1 untried
                tried = lost >= 0
                if tried.any():
                    index = joint_index(batch['tried_actions'][rows].clamp(min=0))
                    loss = loss + _loss('hit', prediction.gather(1, index)[tried], lost[tried].float())
                    labels += int(tried.sum())
        else:
            loss = _loss(spec.target, prediction.view(-1), value)
            labels = value.numel()
        loss = loss / max(1, labels)
        total = total + spec.coeff * loss
        stats[f'losses/aux_{spec.output}'] = float(loss.detach())
        stats[f'charts/aux_{spec.output}_labels'] = labels / max(1, value.numel())
    return total, stats
