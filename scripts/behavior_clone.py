#!/usr/bin/env python3
"""Behavior-clone a specialist's policy from expert demo traces (BC warm start, design §8).

    python scripts/behavior_clone.py --specialist stalfos-keese-gel --start-state 1_73w_0 --demo demos/d1_73.txt \\
        [--demo more.txt ...] [--load CKPT] [--epochs 400] [--run-id ID] [--models-dir DIR]

Each trace (one ``Action.parse`` line per step, ``#`` comments allowed) is replayed from ``--start-state``
through the specialist's native ``Env``; every action must be legal and the episode must end in success.  The
cloned model is saved as ``<models>/<specialist>/<run-id>/ckpt.pt`` with no optimizer state, ready for
``python -m triforce.train --specialist <name> --load <that run dir>`` (add ``--demo`` there for
demo-regularized PPO).
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pylint: disable=wrong-import-position
from triforce import registry
from triforce._native import Env
from triforce.demo import behavior_clone, collect_demo_batch, parse_demo_actions
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.observation import observation_space
from triforce.specialist import load_specialist


def parse_args(argv=None):
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('--specialist', required=True)
    parser.add_argument('--start-state', required=True, help="savestate every demo starts from (name or path)")
    parser.add_argument('--states-dir', default=None, help="where bare state names resolve")
    parser.add_argument('--demo', action='append', required=True, help="demo trace file (repeatable)")
    parser.add_argument('--load', default=None, help="checkpoint to fine-tune instead of a fresh model")
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min-accuracy', type=float, default=0.95)
    parser.add_argument('--allow-incomplete', action='store_true',
                        help="accept demos that stop before the episode ends in success")
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--models-dir', default=None)
    return parser.parse_args(argv)


def main(argv=None):
    """Collect demos, clone, save."""
    args = parse_args(argv)
    if args.load:
        network, _ = registry.load_model(args.load)
    else:
        network = ZeldaPolicy(observation_space(), ModelConfig())

    config = load_specialist(args.specialist)
    env = Env(config, start_state=args.start_state, capture_dir=None, states_dir=args.states_dir)
    batches = [collect_demo_batch(env, parse_demo_actions(path), require_success=not args.allow_incomplete)
               for path in args.demo]
    obs = {key: torch.cat([b[0][key] for b in batches]) for key in batches[0][0]}
    masks = torch.cat([b[1] for b in batches])
    targets = torch.cat([b[2] for b in batches])

    result = behavior_clone(network, (obs, masks, targets), epochs=args.epochs, lr=args.lr,
                            log_every=max(args.epochs // 10, 1))
    print(f"examples={result['examples']} first_loss={result['first_loss']:.6f} "
          f"final_loss={result['final_loss']:.6f} accuracy={result['accuracy']:.4f}")
    if result['accuracy'] < args.min_accuracy:
        raise SystemExit(f"Accuracy {result['accuracy']:.4f} below required {args.min_accuracy:.4f}")

    run_id = args.run_id or registry.new_run_id() + '-bc'
    path = registry.save_checkpoint(config.name, run_id, network, None,
                                    metrics={'bc/accuracy': result['accuracy'], 'bc/loss': result['final_loss']},
                                    root=args.models_dir, extra={'demos': list(args.demo)})
    print(f"Saved behavior-cloned checkpoint: {path}")


if __name__ == '__main__':
    main()
