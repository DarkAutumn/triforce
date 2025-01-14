#! /usr/bin/python3
"""Train models to play The Legend of Zelda (NES)."""

# pylint: disable=duplicate-code

import argparse
import sys
from triforce import ZELDA_MODELS, ZeldaAI

def main():
    """Main entry point."""
    args = parse_args()
    iterations = None if args.iterations <= 0 else args.iterations
    output_path = args.output if args.output else 'training/'

    models = args.models if args.models else ZELDA_MODELS.keys()
    for model_name in models:
        zelda_ml = ZeldaAI(ZELDA_MODELS[model_name], verbose=args.verbose, ent_coef=args.ent_coef)
        zelda_ml.train(output_path, iterations, args.parallel, obs_kind=args.obs_kind)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="train - Train Zelda ML models")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--obs-kind", choices=['gameplay', 'viewport', 'full'], default='viewport',
                        help="The kind of observation to use.")

    parser.add_argument('models', nargs='*', help='List of models to train')
    parser.add_argument("--output", type=str, help="Location to write to.")
    parser.add_argument("--iterations", type=int, default=-1, help="Override iteration count.")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel environments to run.")


    try:
        args = parser.parse_args()
        return args

    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(e)
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
