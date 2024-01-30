#! /usr/bin/python

import argparse
import os

from triforce_lib import ZeldaML, pygame_render

def main(args):
    render_mode = 'rgb_array'
    model_path = args.model_path[0] if args.model_path else os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

    zelda_ml = ZeldaML(args.color, render_mode=render_mode, verbose=args.verbose, ent_coef=args.ent_coef, device="cuda", obs_kind=args.obs_kind)
    zelda_ml.load_models(model_path)

    if args.scenario is None:
        args.scenario = 'zelda'

    pygame_render(zelda_ml, args.scenario)

def parse_args():
    parser = argparse.ArgumentParser(description="ZeldaML - An ML agent to play The Legned of Zelda (NES).")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--color", action='store_true', help="Record the environment.")
    parser.add_argument("--obs-kind", choices=['gameplay', 'viewport', 'full'], default='viewport', help="The kind of observation to use.")
    parser.add_argument("--model-path", nargs=1, help="Location to read models from.")

    parser.add_argument('scenario', nargs='?', help='Scenario name')

    try:
        args = parser.parse_args()
        return args
    except Exception as e:
        print(e)
        parser.print_help()
        exit(0)

if __name__ == '__main__':
    args = parse_args()
    main(args)
