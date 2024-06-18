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

    try:
        args = parser.parse_args()
        return args

    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(e)
        parser.print_help()
        sys.exit(0)

def main(args):
    """Entry point to train the multi-headed model."""
    output = args.output
    if output:
        parentdir = os.path.dirname(output)
        os.makedirs(parentdir, exist_ok=True)

    iterations = 10_000 if args.iterations <= 0 else args.iterations

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = make_multihead_zelda_env('0_67s.state', device=device)
    try:
        obs_space = env.observation_space
        image_dim = obs_space[0].shape[-1]
        features = math.prod(obs_space[1].shape)
        network = ZeldaMultiHeadNetwork(image_dim, features, device)
        ppo = MultiHeadPPO(network, device)

        ppo.train(env, iterations)
        if output:
            network.save(output)

    finally:
        env.close()

if __name__ == '__main__':
    main(parse_args())
