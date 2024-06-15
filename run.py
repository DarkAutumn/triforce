#! /usr/bin/python
"""Run the ZeldaML agent to play The Legend of Zelda (NES)."""

# While we want to keep this file relatively clean, it's fine to have a bit of a large render function.

# pylint: disable=too-few-public-methods,too-many-locals,too-many-branches,too-many-statements
# pylint: disable=too-many-nested-blocks,duplicate-code

import argparse
import os
import sys
from triforce import TRAINING_SCENARIOS
from zui import DisplayWindowSB, DisplayWindowTorch

def main():
    """Main function."""
    args = parse_args()
    if args.scenario is None:
        args.scenario = 'zelda'

    scenario = TRAINING_SCENARIOS.get(args.scenario, None)
    if not scenario:
        print(f'Unknown scenario {args.scenario}')
        return

    if args.torch:
        display = DisplayWindowTorch(scenario, args.model_path)
        display.show(args.headless_recording)
    else:
        model_path = args.model_path[0] if args.model_path else \
                        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')
        display = DisplayWindowSB(scenario, model_path)
        display.show(args.headless_recording)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Triforce - An ML agent to play The Legned of Zelda (NES).")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--color", action='store_true',
                        help="Give the model a color version of the game (instead of grayscale).")
    parser.add_argument("--obs-kind", choices=['gameplay', 'viewport', 'full'], default='viewport',
                        help="The kind of observation to use.")
    parser.add_argument("--model-path", nargs=1, help="Location to read models from.")
    parser.add_argument("--frame-stack", type=int, default=1, help="Number of frames the model was trained with.")
    parser.add_argument("--headless-recording", action='store_true', help="Record the game without displaying it.")
    parser.add_argument("--torch", action='store_true', help="Use a PyTorch model.")

    parser.add_argument('scenario', nargs='?', help='Scenario name')

    try:
        args = parser.parse_args()
        return args

    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(e)
        parser.print_help()
        sys.exit(0)

if __name__ == '__main__':
    main()
