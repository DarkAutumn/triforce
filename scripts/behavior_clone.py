#!/usr/bin/env python3
"""Behavior-clone a Triforce policy from a short expert movement trace."""

import argparse
import os
import sys
from pathlib import Path


import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnose import demo_action_to_indices, is_demo_action_allowed, parse_demo_trace
from triforce import ActionSpaceDefinition, ModelKindDefinition, Network, TrainingScenarioDefinition, make_zelda_env
from triforce.observation_wrapper import infer_obs_kind
from triforce.zelda_enums import ActionKind, Direction


def compute_demo_accuracy(pred_actions, target_actions):
    """Return exact two-head action accuracy."""
    return (pred_actions.long() == target_actions.long()).all(dim=-1).float().mean().item()


def _stack_observations(observations, device):
    keys = observations[0].keys()
    return {key: torch.stack([torch.as_tensor(obs[key]) for obs in observations]).to(device) for key in keys}


def _collect_demo(model_path, scenario_name, trace_path, prefix_east, device):
    metadata = Network.load_metadata(model_path)
    obs_kind, frame_stack = infer_obs_kind(metadata["obs_space"])
    scenario_def = TrainingScenarioDefinition.get(scenario_name)
    action_space_def = ActionSpaceDefinition.get("all-items")
    env = make_zelda_env(scenario_def, action_space_def.actions, multihead=True,
                         translation=True, obs_kind=obs_kind, frame_stack=frame_stack)

    observations = []
    masks = []
    targets = []
    actions = [(ActionKind.MOVE, Direction.E)] * prefix_east + parse_demo_trace(trace_path)
    try:
        obs, info = env.reset()
        ending = ""
        final_location = None
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
        # If the translated info did not carry ending, infer success from the final reset-free trace state where possible.
        success = ending.startswith("success-") or total_reward > 1.0
        if not success:
            raise RuntimeError(f"Demo did not appear to reach success; ending={ending}, total_reward={total_reward:.3f}")
    finally:
        env.close()

    return _stack_observations(observations, device), torch.stack(masks).to(device), torch.stack(targets).to(device)


def behavior_clone(args):
    """Run behavior cloning."""
    device = torch.device(args.device)
    metadata = Network.load_metadata(args.model_path)
    model_kind = ModelKindDefinition.get(metadata["model_kind"] or "impala-multihead")
    obs_space, act_space = Network.load_spaces(args.model_path)
    network = model_kind.network_class(obs_space, act_space).to(device)
    network.load(args.model_path)
    network.train()

    obs, masks, targets = _collect_demo(args.model_path, args.scenario, args.demo_trace, args.prefix_east, device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    first_loss = None
    final_loss = None
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        _, logprob, _, _ = network.get_action_and_value(obs, masks, targets)
        loss = -logprob.mean()
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = loss.item()
        final_loss = loss.item()
        if (epoch + 1) % max(args.epochs // 10, 1) == 0:
            print(f"epoch={epoch + 1} loss={final_loss:.6f}")

    network.eval()
    with torch.no_grad():
        pred = network.get_action(obs, masks, deterministic=True)
        accuracy = compute_demo_accuracy(pred, targets)
    print(f"examples={targets.shape[0]} first_loss={first_loss:.6f} final_loss={final_loss:.6f} accuracy={accuracy:.6f}")
    if accuracy < args.min_accuracy:
        raise SystemExit(f"Accuracy {accuracy:.6f} below required {args.min_accuracy:.6f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    network.save(args.output)
    print(f"Saved behavior-cloned checkpoint: {args.output}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Behavior-clone a Triforce policy from a movement demo")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--demo-trace", required=True)
    parser.add_argument("--prefix-east", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    return parser.parse_args()


if __name__ == "__main__":
    behavior_clone(parse_args())
