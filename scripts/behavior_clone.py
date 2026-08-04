#!/usr/bin/env python3
"""Behavior-clone a Triforce policy from a short expert movement trace."""

import argparse
import os
import sys
from pathlib import Path


import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triforce import ModelKindDefinition, Network
from triforce.demo import collect_demo_batch, compute_demo_accuracy, parse_demo_trace




def behavior_clone(args):
    """Run behavior cloning."""
    device = torch.device(args.device)
    metadata = Network.load_metadata(args.model_path)
    model_kind = ModelKindDefinition.get(metadata["model_kind"] or "impala-multihead")
    obs_space, act_space = Network.load_spaces(args.model_path)
    network = model_kind.network_class(obs_space, act_space).to(device)
    network.load(args.model_path)
    network.train()

    obs, masks, targets = collect_demo_batch(args.model_path, args.scenario, args.demo_trace, args.prefix_east, device)
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
