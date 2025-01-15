#! /usr/bin/python3
"""Train models to play The Legend of Zelda (NES)."""

# pylint: disable=duplicate-code

import argparse
import sys
import os

from tqdm import tqdm
from triforce import ZELDA_MODELS
from triforce.ml_ppo import PPO
from triforce.zelda_env import make_zelda_env

def _train_one(model_name, args):
    model_def = ZELDA_MODELS[model_name]
    def create_env():
        return make_zelda_env(model_def.training_scenario, model_def.action_space, obs_kind=args.obs_kind)

    iterations = None if args.iterations <= 0 else args.iterations
    output_path = args.output if args.output else 'training/'
    model_directory = f"{output_path}/{model_name}"
    log_dir = f"{model_directory}/logs"
    save_name = model_name.replace(' ', '-')
    os.makedirs(model_directory, exist_ok=True)

    os.makedirs(log_dir, exist_ok=True)

    ppo = PPO(args.device, log_dir, ent_coef=args.ent_coef)
    model = ppo.train(model_def.neural_net, create_env, iterations, tqdm(total=args.iterations),
                      envs = args.parallel, save_path=model_directory, model_name=save_name)

    model.save(f"{model_directory}/model.pt")

def main():
    """Main entry point."""
    args = parse_args()
    models = args.models if args.models else ZELDA_MODELS.keys()
    for model_name in models:
        _train_one(model_name, args)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="train - Train Zelda ML models")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--obs-kind", choices=['gameplay', 'viewport', 'full'], default='viewport',
                        help="The kind of observation to use.")
    parser.add_argument("--device", choices=['cpu', 'cuda'], default='cpu', help="The device to use.")

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
