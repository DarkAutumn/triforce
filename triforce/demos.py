"""Demonstrations: (observation, action mask, action) rows recorded from a scripted rule, cloned during PPO.

``train --demos FILE --demo-coeff C`` adds ``C × −log π(action | observation)`` over the file's rows to every PPO
minibatch (a minibatch-sized sample when the file holds more rows; ``PPO.demo_batch``, the term best-of-N fills with
the policy's own best runs).  A file is a NumPy ``.npz`` written by :func:`save_demos`: ``obs/<key>`` for every
observation key, ``mask`` and ``actions`` (``[rows, 3]``: kind, direction, duration indices, as ``EnvPool`` records
them).  A duration of :data:`ANY_DURATION` (-1) clones the row's kind and direction only: the rule's pace there is
not taught (round 25 §5: cloned in full, a rule's short steps spread to the family's other rooms).
``scripts/experiments/r25_d5_37_tracks.py --demos`` writes one for level 5 room 0x37 (round 25).

A task may declare its own (round 40): ``demos: {files: [NAME, ...], coeff: C}``, the names under the ``demos`` data
folder (``triforce.inputs.task_demos``); ``train`` and ``lab.train`` clone them with what ``--demos`` adds, and
``--no-task-demos`` leaves them out (an arm "that run, except the demonstrations"); :func:`recipe` makes the list.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import _native
from .model import MASK_KEY

ANY_DURATION = -1
DEFAULT_COEFF = 0.2         # the cloning weight when neither the command line nor a task names one

_DIRECTIONS = list(_native.DIRECTIONS)
_DURATIONS = [int(d) for d in _native.DURATIONS]
_KINDS = list(_native.ACTION_KINDS)


def action_indices(text: str) -> tuple[int, int, int]:
    """``'MOVE W 4'`` / ``'SWORD N'`` / ``'WAIT 8'`` as (kind, direction, duration) indices; a part the kind
    doesn't take is 0 (what the policy's masked heads score)."""
    kind, *rest = text.split()
    direction = duration = 0
    for part in rest:
        if part in _DIRECTIONS:
            direction = _DIRECTIONS.index(part)
        else:
            duration = _DURATIONS.index(int(part))
    return _KINDS.index(kind), direction, duration


def legal(mask, indices: tuple[int, int, int]) -> bool:
    """Whether the action mask (``[kinds | kinds × directions | kinds × durations]``) allows the action: its kind,
    and its direction and duration where the kind's row offers any."""
    mask = np.asarray(mask).astype(bool)
    kind, direction, duration = indices
    kinds, directions, durations = len(_KINDS), len(_DIRECTIONS), len(_DURATIONS)
    dir_row = mask[kinds + kind * directions:kinds + (kind + 1) * directions]
    dur_start = kinds + kinds * directions
    dur_row = mask[dur_start + kind * durations:dur_start + (kind + 1) * durations]
    return bool(mask[kind]) and (not dir_row.any() or bool(dir_row[direction])) and \
        (not dur_row.any() or bool(dur_row[duration]))


def recipe(declared, extra, coeff: Optional[float]) -> tuple[Optional[list[str]], float]:
    """The demonstration files a training clones and their weight: every task's declared files (``declared``:
    [(files, coeff)] as ``inputs.task_demos`` returns them, one a task), then the command line's ``extra``; the
    weight is ``coeff`` (``--demo-coeff``) when given, else the tasks' own (they must agree: SystemExit), else
    ``DEFAULT_COEFF``.  No file: (None, the weight)."""
    files = [str(path) for found, _ in declared for path in found] + list(extra or ())
    weights = sorted({weight for found, weight in declared if found and weight is not None})
    if coeff is None:
        if len(weights) > 1:
            raise SystemExit(f"the tasks' demos weigh {weights}: one cloning weight trains them all; name it with "
                             f"--demo-coeff")
        coeff = weights[0] if weights else DEFAULT_COEFF
    return (files or None), coeff


class DemoRecorder:
    """Collects rows: ``add(obs, action_text)`` for every decision of a kept episode (see ``keep``/``drop``)."""

    def __init__(self):
        self.rows: list[tuple[dict, tuple[int, int, int]]] = []
        self.pending: list[tuple[dict, tuple[int, int, int]]] = []

    def add(self, obs: dict, action: str, repeat: int = 1, any_duration: bool = False):
        """One decision of the episode being played; ``obs`` must carry the action mask, and the action must be
        legal under it (ValueError).  ``repeat`` writes the row that many times: a rare decision (a swing among
        hundreds of steps) weighs that much more in the cloning loss.  ``any_duration`` clones the row's kind and
        direction only (:data:`ANY_DURATION`)."""
        indices = action_indices(action)
        if not legal(obs[MASK_KEY], indices):
            raise ValueError(f"{action} is masked")
        if any_duration:
            indices = (indices[0], indices[1], ANY_DURATION)
        row = ({k: np.array(v) for k, v in obs.items()}, indices)
        self.pending += [row] * repeat

    def keep(self):
        """The episode's decisions join the rows."""
        self.rows += self.pending
        self.pending = []

    def drop(self):
        """The episode's decisions are discarded."""
        self.pending = []

    def save(self, path) -> int:
        """Writes the rows (``save_demos``); returns their number."""
        save_demos(path, [obs for obs, _ in self.rows], [action for _, action in self.rows])
        return len(self.rows)


def save_demos(path, observations: list[dict], actions: list[tuple[int, int, int]]):
    """Writes demonstration rows to ``path`` (``.npz``)."""
    if not observations:
        raise ValueError("no demonstration rows to save")
    keys = sorted(observations[0])
    if MASK_KEY not in keys:
        raise ValueError(f"demonstration observations need '{MASK_KEY}'")
    arrays = {f'obs/{k}': np.stack([np.asarray(o[k]) for o in observations]) for k in keys if k != MASK_KEY}
    arrays['mask'] = np.stack([np.asarray(o[MASK_KEY]) for o in observations]).astype(bool)
    arrays['actions'] = np.asarray(actions, dtype=np.int64).reshape(-1, 3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as file:
        np.savez_compressed(file, **arrays)


def load_demos(paths) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Reads one or more demonstration files as ``PPO.demo_batch``: (obs, masks, actions) tensors."""
    parts = []
    for path in paths:
        with np.load(path) as data:
            parts.append({k: data[k] for k in data.files})
    keys = sorted(k for k in parts[0] if k.startswith('obs/'))
    for part, path in zip(parts, paths):
        if sorted(k for k in part if k.startswith('obs/')) != keys:
            raise ValueError(f"{path}: observation keys differ from {paths[0]}")
    obs = {k[len('obs/'):]: torch.as_tensor(np.concatenate([p[k] for p in parts])) for k in keys}
    masks = torch.as_tensor(np.concatenate([p['mask'] for p in parts])).bool()
    actions = torch.as_tensor(np.concatenate([p['actions'] for p in parts])).long()
    for value in obs.values():
        if value.shape[0] != actions.shape[0]:
            raise ValueError("demonstration files hold observation and action rows of different counts")
    if masks.shape[0] != actions.shape[0]:
        raise ValueError("demonstration files hold mask and action rows of different counts")
    return obs, masks, actions
