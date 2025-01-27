#! /usr/bin/python
"""Run the ZeldaML agent to play The Legend of Zelda (NES)."""

# While we want to keep this file relatively clean, it's fine to have a bit of a large render function.

# pylint: disable=too-few-public-methods,too-many-locals,too-many-branches,too-many-statements
# pylint: disable=too-many-nested-blocks,duplicate-code

import argparse
import os
import sys

from triforce import TrainingScenarioDefinition
from zui import RewardDebugger

def main():
    """Main function."""
    args = parse_args()
    model_path = args.model_path if args.model_path else os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                      'models')
    scenario = TrainingScenarioDefinition.get(args.scenario, None)
    if not scenario:
        print(f'Unknown scenario {args.scenario}')
        return

    display = RewardDebugger(scenario, model_path, args.model, args.frame_stack)
    display.show()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Triforce - An ML agent to play The Legned of Zelda (NES).")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--obs-kind", choices=['gameplay', 'viewport', 'full'], default='viewport',
                        help="The kind of observation to use.")
    parser.add_argument("--model-path", nargs=str, help="Location to read models from.")
    parser.add_argument("--frame-stack", type=int, default=3, help="Number of frames to stack.")

    parser.add_argument('model', type=str, help='Model name')
    parser.add_argument('scenario', type=str, help='Scenario name')

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
