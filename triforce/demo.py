"""Expert demonstrations for behavior-cloning warm starts and demo-regularized PPO (ported from v1).

A demo trace is a text file with one action per line in ``Action.parse`` syntax (``MOVE N 16``, ``SWORD E``,
``WAIT 8``); blank lines and ``#`` comments are ignored.  ``collect_demo_batch`` replays a trace through a
native ``Env`` (refusing actions the mask forbids) and returns ``(observations, masks, targets)`` tensors that
``behavior_clone`` trains on and ``ppo.PPO(demo_batch=...)`` uses as its demo-regularization term.
"""

from typing import Optional

import numpy as np
import torch

from .actions import Action, ActionMask
from .model import MASK_KEY


def parse_demo_actions(path: str) -> list[Action]:
    """Reads a demo trace file."""
    actions = []
    with open(path, encoding='utf-8') as file:
        for line in file:
            text = line.split('#', 1)[0].strip()
            if text:
                actions.append(Action.parse(text))
    if not actions:
        raise ValueError(f"Demo trace {path} has no actions")
    return actions


def stack_observations(observations: list[dict], device='cpu') -> dict[str, torch.Tensor]:
    """Stacks observation dicts into one batched tensor dict (action_mask excluded)."""
    keys = [k for k in observations[0] if k != MASK_KEY]
    return {k: torch.as_tensor(np.stack([np.asarray(o[k]) for o in observations])).to(device) for k in keys}


def collect_demo_batch(env, actions: list[Action], device='cpu', require_success: bool = True):
    """Replays ``actions`` from ``env.reset()`` (a native ``Env``); returns (obs dict [N,...], masks bool
    [N, MASK_SIZE], targets int64 [N, 3]) for every step taken.

    Raises if an action is illegal under the current mask, or (``require_success``) if the episode does not
    end with ``info['outcome'] == 'success'``.
    """
    observations, masks, targets = [], [], []
    obs, _ = env.reset()
    outcome = None
    for step, action in enumerate(actions, start=1):
        mask = np.asarray(obs[MASK_KEY], dtype=bool)
        if not ActionMask.from_flat(mask).is_allowed(action):
            raise RuntimeError(f"Demo step {step}: {action} is not allowed by mask {ActionMask.from_flat(mask)}")
        indices = action.to_indices()
        observations.append(obs)
        masks.append(mask)
        targets.append(indices)
        obs, _, terminated, truncated, info = env.step(indices)
        if terminated or truncated:
            outcome = info.get('outcome')
            break
    if require_success and outcome != 'success':
        raise RuntimeError(f"Demo did not end in success (outcome={outcome}, steps={len(targets)})")
    return (stack_observations(observations, device), torch.as_tensor(np.stack(masks)).to(device),
            torch.as_tensor(np.stack(targets), dtype=torch.int64).to(device))


def compute_demo_accuracy(pred_actions: torch.Tensor, target_actions: torch.Tensor) -> float:
    """Exact (kind, direction, duration) accuracy; heads that don't apply are 0 in both."""
    return (pred_actions.long() == target_actions.long()).all(dim=-1).float().mean().item()


def behavior_clone(network, demo_batch, epochs: int = 400, lr: float = 1e-4,
                   log_every: Optional[int] = None) -> dict:
    """Maximizes the demo actions' log-probability.  Returns first/final loss and greedy accuracy."""
    obs, masks, targets = demo_batch
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    network.train()
    first_loss = final_loss = None
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = -network.evaluate(obs, masks, targets).log_prob.mean()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        first_loss = final_loss if first_loss is None else first_loss
        if log_every and (epoch + 1) % log_every == 0:
            print(f"epoch={epoch + 1} loss={final_loss:.6f}")
    network.eval()
    with torch.no_grad():
        accuracy = compute_demo_accuracy(network.get_action(obs, masks, deterministic=True), targets)
    return {'first_loss': first_loss, 'final_loss': final_loss, 'accuracy': accuracy,
            'examples': int(targets.shape[0])}
