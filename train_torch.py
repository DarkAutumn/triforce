#!/usr/bin/python

import math
import torch
from triforce import MultiHeadPPO, ZeldaMultiHeadNetwork, make_multihead_zelda_env

def main():
    """Entry point to train the multi-headed model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = make_multihead_zelda_env('0_67s.state', device=device)
    try:
        obs_space = env.observation_space
        image_dim = obs_space[0].shape[-1]
        features = math.prod(obs_space[1].shape)
        network = ZeldaMultiHeadNetwork(image_dim, features, device)
        ppo = MultiHeadPPO(network, device)

        ppo.train(env, 10_000)

    finally:
        env.close()

if __name__ == '__main__':
    main()
