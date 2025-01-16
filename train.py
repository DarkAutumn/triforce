#! /usr/bin/python3
"""Train models to play The Legend of Zelda (NES)."""

# pylint: disable=duplicate-code

import argparse
import sys
import os
import faulthandler
import traceback

import torch
from tqdm import tqdm
from triforce import ModelDefinition, TrainingScenarioDefinition
from triforce.ml_ppo import PPO
from triforce.zelda_env import make_zelda_env

def _dump_trace_with_locals(exc_type, exc_value, exc_traceback):
    with open("crash_log.txt", "w", encoding="utf8") as f:
        f.write(f"Unhandled exception: {exc_type.__name__}: {exc_value}\n\n")

        for frame, lineno in traceback.walk_tb(exc_traceback):
            f.write(f"File: {frame.f_code.co_filename}, Line: {lineno}, Function: {frame.f_code.co_name}\n")
            f.write("Locals:\n")
            for var_name, var_value in frame.f_locals.items():
                typename = type(var_value).__name__
                f.write(f"  {typename} {var_name}: {var_value}\n")
            f.write("\n")

def main():
    """Main entry point."""
    args = parse_args()

    if args.hook_exceptions:
        faulthandler.enable()
        sys.excepthook = _dump_trace_with_locals

    model_name = args.model
    model_def = ModelDefinition.get(model_name)
    if model_def is None:
        raise ValueError(f"Unknown model: {model_name}")

    scenario_def = TrainingScenarioDefinition.get(args.scenario)

    def create_env():
        return make_zelda_env(scenario_def, model_def.action_space, obs_kind=args.obs_kind)

    device = args.device if args.device else  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    iterations = None if args.iterations <= 0 else args.iterations
    output_path = args.output if args.output else 'training/'
    model_directory = f"{output_path}/{model_name}"
    log_dir = f"{model_directory}/logs"
    save_name = model_name.replace(' ', '-')
    os.makedirs(model_directory, exist_ok=True)

    os.makedirs(log_dir, exist_ok=True)

    ppo = PPO(device, log_dir, ent_coef=args.ent_coef)
    model = ppo.train(model_def.neural_net, create_env, iterations, tqdm(total=args.iterations),
                      envs = args.parallel, save_path=model_directory, model_name=save_name, load_from=args.load)

    model.save(f"{model_directory}/model.pt")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="train - Train Zelda ML models")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--obs-kind", choices=['gameplay', 'viewport', 'full'], default='viewport',
                        help="The kind of observation to use.")
    parser.add_argument("--device", choices=['cpu', 'cuda'], default=None, help="The device to use.")

    parser.add_argument('model', type=str, help='The model to train.')
    parser.add_argument('scenario', type=str, help='The scenario to train on.')
    parser.add_argument("--output", type=str, help="Location to write to.")
    parser.add_argument("--iterations", type=int, default=-1, help="Override iteration count.")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel environments to run.")
    parser.add_argument("--load", type=str, help="Load a model to continue training.")
    parser.add_argument("--hook-exceptions", action='store_true', help="Dump tracebacks on unhandled exceptions.")

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
