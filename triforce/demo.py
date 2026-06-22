"""Reusable expert demonstration helpers for Triforce training tools."""

import re

import numpy as np
import torch

from triforce import ActionSpaceDefinition, Network, TrainingScenarioDefinition, make_zelda_env
from triforce.action_space import ActionKind
from triforce.observation_wrapper import infer_obs_kind
from triforce.zelda_enums import Direction


DEMO_ACTION_TYPE_INDEX = {ActionKind.MOVE: 0}
DEMO_DIRECTION_INDEX = {Direction.N: 0, Direction.S: 1, Direction.W: 2, Direction.E: 3}


def parse_demo_trace(path: str) -> list[tuple[ActionKind, Direction]]:
    """Parse a movement trace from normalized or original reverse-trace format."""
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    normalized = []
    in_normalized = False
    for line in lines:
        stripped = line.strip()
        if stripped == "## Normalized forward trace":
            in_normalized = True
            continue
        if stripped == "## Original reverse trace":
            break
        if in_normalized and stripped and not stripped.startswith('#'):
            parts = stripped.split()
            if len(parts) == 2:
                normalized.append((ActionKind[parts[0]], Direction[parts[1]]))

    if normalized:
        actions = normalized
    else:
        reverse_entries = []
        pattern = re.compile(r"^#(\d+)\s+(\S+)(?:\s+(\S+))?")
        for line in lines:
            match = pattern.match(line.strip())
            if not match:
                continue
            step = int(match.group(1))
            action_name = match.group(2)
            direction_name = match.group(3)
            if action_name == "None":
                continue
            reverse_entries.append((step, ActionKind[action_name], Direction[direction_name]))
        actions = [(action, direction) for _, action, direction in sorted(reverse_entries)]

    if not actions:
        raise ValueError("Demo trace did not contain any actions")
    if any(action != ActionKind.MOVE for action, _ in actions):
        raise ValueError("Wallmaster demo traces must contain only MOVE actions")
    return actions


def demo_action_to_indices(action: ActionKind, direction: Direction) -> tuple[int, int]:
    """Convert a demo action to all-items MultiDiscrete indices."""
    return DEMO_ACTION_TYPE_INDEX[action], DEMO_DIRECTION_INDEX[direction]


def is_demo_action_allowed(action_mask, action: ActionKind, direction: Direction) -> bool:
    """Return whether a demo action is allowed by the joint action mask."""
    action_type, direction_index = demo_action_to_indices(action, direction)
    return bool(action_mask[action_type * 4 + direction_index])


def stack_demo_observations(observations: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    """Stack observation dictionaries into one batched tensor dictionary."""
    keys = observations[0].keys()
    return {key: torch.stack([torch.as_tensor(obs[key]) for obs in observations]).to(device) for key in keys}


def compute_demo_accuracy(pred_actions: torch.Tensor, target_actions: torch.Tensor) -> float:
    """Return exact two-head action accuracy."""
    return (pred_actions.long() == target_actions.long()).all(dim=-1).float().mean().item()


def collect_demo_batch(model_path: str, scenario_name: str, trace_path: str, prefix_east: int,
                       device: torch.device, *, translation: bool = True):
    """Replay an expert trace and return observations, masks, and MultiDiscrete target actions."""
    metadata = Network.load_metadata(model_path)
    obs_kind, frame_stack = infer_obs_kind(metadata["obs_space"])
    scenario_def = TrainingScenarioDefinition.get(scenario_name)
    action_space_def = ActionSpaceDefinition.get("all-items")
    env = make_zelda_env(scenario_def, action_space_def.actions, multihead=True,
                         translation=translation, obs_kind=obs_kind, frame_stack=frame_stack)

    observations = []
    masks = []
    targets = []
    actions = [(ActionKind.MOVE, Direction.E)] * prefix_east + parse_demo_trace(trace_path)
    try:
        obs, info = env.reset()
        ending = ""
        total_reward = 0.0
        for step, (action, direction) in enumerate(actions, start=1):
            action_mask = info.get('action_mask')
            if action_mask is None:
                raise RuntimeError(f"Missing action mask at demo step {step}")
            if not is_demo_action_allowed(action_mask, action, direction):
                raise RuntimeError(f"Invalid demo action at step {step}: {action.name} {direction.name}")
            action_indices = demo_action_to_indices(action, direction)
            observations.append(obs)
            masks.append(torch.as_tensor(action_mask, dtype=torch.bool))
            targets.append(torch.as_tensor(action_indices, dtype=torch.long))
            obs, reward, terminated, truncated, info = env.step(np.asarray(action_indices, dtype=np.int64))
            total_reward += float(reward)
            if terminated or truncated:
                ending = info.get('rewards', {}).get('ending') or info.get('ending') or "unknown"
                break
        success = ending.startswith("success-") or (translation and total_reward > 1.0)
        if not success:
            raise RuntimeError(f"Demo did not appear to reach success; ending={ending}, total_reward={total_reward:.3f}")
    finally:
        env.close()

    return stack_demo_observations(observations, device), torch.stack(masks).to(device), torch.stack(targets).to(device)
