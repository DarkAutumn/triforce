"""Auxiliary targets: losses on a network's named auxiliary outputs (docs/lab.md "Auxiliary targets").

The rollout carries, per decision, what the action taken did (``RolloutBuffer``: ``health_lost``, ``hits``,
``blocked``, ``frames``, the aimless score's ``calm``, ``progress``, ``waste``) and, with ``tried``
(``EnvPool.rollout``; ``--try-others K:P``), the look-ahead's verdict on
K candidates NOT taken on a share P of the decisions (``tried_actions``, ``tried_lost``, ``tried_blocked``; training
only). A ``LabNet`` returns named auxiliary outputs in ``Heads.aux``; a run names a loss per output: ``--aux
OUTPUT=TARGET:COEFF``.

Targets (per decision):

- ``hit``: the action lost health (``health_lost > 0``); binary, BCE on logits. On a per-action output
  ``[B, 352]`` the tried candidates' verdicts are labels too (their joint actions' logits).
- ``hit_or_blocked``: ``hit``, or a Bubble's touch landed during the action (``blocked``: the sword blocked for
  151-160 frames, no health lost; round 38), so a head learns to avoid Bubbles as it avoids damage; binary, the tried
  candidates' verdicts likewise (``tried_lost`` or ``tried_blocked``).
- ``damaged``: the action dealt a monster a hit (``hits > 0``); binary.
- ``hearts_lost``: hearts the action lost (net, clamped at 0); regression, ½ squared error.
- ``frames``: the frames the action took, / 16; regression.
- ``calm``: the decision was calm (round 38, ``zelda::aimless``: nothing in the room can hurt Link, the goal does not
  go away, no dropped item pulls him); binary.
- ``progress``: the frames of progress the action made toward the goal at walking speed (negative: ground lost; 0
  when not calm), / 16; regression.
- ``waste``: the frames the action added to its calm stretch's waste (``penalty-aimless``'s measure), / 16;
  regression.

An output ``[B]`` is the taken action's prediction; ``[B, 352]`` is one per joint action (``kind * 32 + direction *
4 + duration``), read at the taken action (and the tried ones for ``hit`` and ``hit_or_blocked``).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..actions import NUM_DIRECTIONS, NUM_DURATIONS

TARGETS = ('hit', 'hit_or_blocked', 'damaged', 'hearts_lost', 'frames', 'calm', 'progress', 'waste')
BINARY = ('hit', 'hit_or_blocked', 'damaged', 'calm')
# The targets the look-ahead's trials label too (a tried candidate's verdict), and held_out reads.
JUDGED = ('hit', 'hit_or_blocked')
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
    if target == 'hit_or_blocked':
        return ((batch['health_lost'][rows] > 0) | batch['blocked'][rows].bool()).float()
    if target == 'damaged':
        return (batch['hits'][rows] > 0).float()
    if target == 'hearts_lost':
        return batch['health_lost'][rows].clamp(min=0.0)
    if target == 'calm':
        return batch['calm'][rows].float()
    if target in ('progress', 'waste'):
        return batch[target][rows] / FRAME_SCALE
    return batch['frames'][rows] / FRAME_SCALE


def tried_labels(target: str, batch: dict, rows) -> torch.Tensor:
    """The tried candidates' verdicts ``[n, K]`` (int64: 1, 0, -1 untried) of ``rows`` for a ``JUDGED`` target."""
    lost = batch['tried_lost'][rows]
    if target == 'hit_or_blocked':
        return torch.where(lost >= 0, lost | batch['tried_blocked'][rows], lost)
    return lost


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
            if spec.target in JUDGED and 'tried_lost' in batch:
                lost = tried_labels(spec.target, batch, rows)        # [n, K]: -1 untried
                tried = lost >= 0
                if tried.any():
                    index = joint_index(batch['tried_actions'][rows].clamp(min=0))
                    loss = loss + _loss(spec.target, prediction.gather(1, index)[tried], lost[tried].float())
                    labels += int(tried.sum())
        else:
            loss = _loss(spec.target, prediction.view(-1), value)
            labels = value.numel()
        loss = loss / max(1, labels)
        total = total + spec.coeff * loss
        stats[f'losses/aux_{spec.output}'] = float(loss.detach())
        stats[f'charts/aux_{spec.output}_labels'] = labels / max(1, value.numel())
    return total, stats


def matched_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Precision (= recall) when as many are refused as are lost: the top-k by score, k the lost (round 35/36's
    "P = R at the look-ahead's veto rate"); NaN without a lost one."""
    k = int(labels.sum())
    if k == 0:
        return float('nan')
    return float(labels[torch.topk(scores, k).indices].float().mean())


def held_out(losses: list[AuxLoss], network, batch: dict, chunk: int = 4096) -> dict:
    """Before an update trains on ``batch`` (``RolloutBuffer.flatten``), how each per-action ``hit`` or
    ``hit_or_blocked`` output reads it (against its own target's labels): held out from every update so far, on the
    policy's own states. ``charts/aux_<output>_pr_taken``: precision = recall on the actions taken (their real
    outcome; the model's own draws of experiment 2's bench); ``..._pr_judged``: on every judged candidate (the taken
    ones and, with ``--try-others``, the tried ones); ``..._lost_taken`` / ``..._lost_judged`` the positive labels
    behind each (pool iterations by them)."""
    judged = [s for s in losses if s.target in JUDGED]
    if not judged:
        return {}
    rows = batch['actions'].shape[0]
    outputs: dict = {s.output: [] for s in judged}
    with torch.no_grad():
        for start in range(0, rows, chunk):
            part = slice(start, min(rows, start + chunk))
            out = network.evaluate({k: v[part] for k, v in batch['obs'].items()}, batch['masks'][part],
                                   batch['actions'][part])
            for name in outputs:
                if out.aux[name].dim() != 2:
                    raise ValueError(f"held_out: {name} is not a per-action output")
                outputs[name].append(out.aux[name])
    stats = {}
    taken = joint_index(batch['actions']).unsqueeze(-1)
    every = slice(None)
    for spec in judged:
        name = spec.output
        logits = torch.cat(outputs[name])
        lost = target_values(spec.target, batch, every)
        scores, labels = [logits.gather(1, taken).squeeze(-1)], [lost]
        stats[f'charts/aux_{name}_pr_taken'] = matched_precision(scores[0], labels[0])
        if 'tried_lost' in batch:
            verdicts = tried_labels(spec.target, batch, every)
            tried = verdicts >= 0
            index = joint_index(batch['tried_actions'].clamp(min=0))
            scores.append(logits.gather(1, index)[tried])
            labels.append(verdicts[tried].float())
        scores_all, labels_all = torch.cat(scores), torch.cat(labels)
        stats[f'charts/aux_{name}_pr_judged'] = matched_precision(scores_all, labels_all)
        stats[f'charts/aux_{name}_lost_taken'] = float(lost.sum())
        stats[f'charts/aux_{name}_lost_judged'] = float(labels_all.sum())
    return stats
