"""A learned guard (round 35, docs/experiments/round35.md; an experiment): a network that sees only what the policy
sees and says, per joint action, how likely that action gets Link hit before the next decision.

It is the policy's own network (``ZeldaPolicy``, the same observation and trunk) with the value head widened to one
logit per joint action (11 × 8 × 4 = 352; entry ``kind * 32 + direction * 4 + duration``); the policy heads ride
along unused, so the Rust ``Policy`` loads it as it is (``Policy::values``). Trained by
``scripts/experiments/r35_guard_train.py`` on the honest planner's look-ahead verdicts (``r35_guard_data.py``).

At play the act mode ``veto:K:P`` (``crates/policy/src/act.rs``) asks the guard instead of the emulator: the
model's draw is refused when its hit probability is above P, and the next of the mode's ranking is judged, at most K
in all, as ``plan:K:S`` does with trials. A run dir holding ``guard.pt`` beside its ``ckpt.pt`` carries its guard
(``attach``); a run without one plays a veto mode as the plain mode, and no mode but ``veto`` reads a guard.
"""

import copy
import math
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from .actions import NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS
from .model import ZeldaPolicy

GUARD_NAME = 'guard.pt'
JOINT = NUM_KINDS * NUM_DIRECTIONS * NUM_DURATIONS


def make_guard(policy: ZeldaPolicy, base_rate: float = 0.05) -> ZeldaPolicy:
    """A guard initialised from ``policy`` (every weight copied), its value head one logit per joint action, zero
    weights and the bias at ``base_rate``'s logit."""
    guard = copy.deepcopy(policy)
    guard.value_head = nn.Linear(policy.config.hidden_dim, JOINT)
    nn.init.zeros_(guard.value_head.weight)
    nn.init.constant_(guard.value_head.bias, math.log(base_rate / (1 - base_rate)))
    return guard


def hit_logits(guard: ZeldaPolicy, obs: dict) -> torch.Tensor:
    """``[B, 352]`` hit logits of a batch of observations (the value tower and head, as ``Policy::values``)."""
    batch = guard._batch(obs)  # pylint: disable=protected-access
    _, value_features = guard.mlp_extractor(guard._features(batch))  # pylint: disable=protected-access
    return guard.value_head(value_features)


def save_guard(path, guard: ZeldaPolicy, meta: Optional[dict] = None):
    torch.save({'model_config': guard.config.to_dict(), 'state_dict': guard.state_dict(), 'meta': meta or {}}, path)


def load_guard(path) -> ZeldaPolicy:
    data = torch.load(path, map_location='cpu', weights_only=False)
    guard = ZeldaPolicy(config=data['model_config'])
    guard.value_head = nn.Linear(guard.config.hidden_dim, JOINT)
    guard.load_state_dict(data['state_dict'])
    return guard.eval()


def guard_file(checkpoint) -> Optional[Path]:
    """``guard.pt`` beside the checkpoint (a run dir or its ``ckpt.pt``), None without one."""
    path = Path(checkpoint)
    path = (path if path.is_dir() else path.parent) / GUARD_NAME
    return path if path.is_file() else None


def attach(native, checkpoint):
    """``native`` (a ``_native.Policy``) with the guard beside ``checkpoint``, or as it is without one."""
    path = guard_file(checkpoint)
    return native if path is None else native.with_guard(load_guard(path).to_native())
