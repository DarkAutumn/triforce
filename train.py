#! /usr/bin/python3
"""Train models to play The Legend of Zelda (NES)."""

import argparse
import sys
from triforce_lib import ZeldaML

def main():
    """Main entry point."""
    args = parse_args()
    iterations = None if args.iterations <= 0 else args.iterations
    models = args.models if args.models else None

    zelda_ml = ZeldaML(args.color, args.frame_stack, render_mode=None, verbose=args.verbose, ent_coef=args.ent_coef,
                       device="cuda", obs_kind=args.obs_kind)
    zelda_ml.train(args.output, models, iterations, args.parallel)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ZeldaML - An ML agent to play The Legned of Zelda (NES).")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--color", action='store_true',
                        help="Give the model a color version of the game (instead of grayscale).")
    parser.add_argument("--obs-kind", choices=['gameplay', 'viewport', 'full'], default='viewport',
                        help="The kind of observation to use.")
    parser.add_argument("--frame-stack", type=int, default=1, help="Number of frames to stack together.")

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
