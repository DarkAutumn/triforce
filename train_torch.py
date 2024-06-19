#!/usr/bin/python

import math
import argparse
import os
import sys
import torch
from triforce import MultiHeadPPO, ZeldaMultiHeadNetwork, make_multihead_zelda_env

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Triforce - An ML agent to play The Legend of Zelda (NES).")
    parser.add_argument("output", nargs='?', type=str, help="Location to write to.")
    parser.add_argument("--iterations", type=int, default=-1, help="Override iteration count.")
    parser.add_argument("--log-dir", type=str, help="Location to write logs to.")
    parser.add_argument("--intermediate", type=str, help="Location to save intermediate models to.")
    parser.add_argument("--render", action='store_true', help="Render the game.")

    try:
        args = parser.parse_args()
        return args

    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(e)
        parser.print_help()
        sys.exit(0)

def make_dirs(args):
    """Create directories for the output."""
    if args.output:
        parentdir = os.path.dirname(args.output)
        os.makedirs(parentdir, exist_ok=True)

    if args.log_dir:
        for i in range(4096):
            subdir = f"run{i:04d}"
            full_path = os.path.join(args.log_dir, subdir)
            if not os.path.exists(full_path):
                break

        args.log_dir = full_path
        os.makedirs(args.log_dir, exist_ok=True)

    if args.intermediate:
        os.makedirs(args.intermediate, exist_ok=True)

def main(args):
    """Entry point to train the multi-headed model."""
    make_dirs(args)

    def save_intermediate(iteration, agent):
        filename = f"{args.intermediate}/model_{iteration}.pt"
        agent.network.save(filename)

    iterations = 10_000 if args.iterations <= 0 else args.iterations

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    render_mode = 'human' if args.render else None
    env = make_multihead_zelda_env('0_67s.state', device=device, render_mode=render_mode)
    try:
        obs_space = env.observation_space
        image_dim = obs_space[0].shape[-1]
        features = math.prod(obs_space[1].shape)
        network = ZeldaMultiHeadNetwork(image_dim, features, device)

        callback = save_intermediate if args.intermediate else None
        ppo = MultiHeadPPO(network, device, train_callback=callback, tensorboard_dir=args.log_dir)

        ppo.train(env, iterations)
        if args.output:
            network.save(args.output)

    finally:
        env.close()

if __name__ == '__main__':
    main(parse_args())
