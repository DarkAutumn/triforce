# pylint: disable

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import retro
from triforce.observation_wrapper import ObservationWrapper
import pickle

def main():
    with open('reshape_error.pkl', 'rb') as f:
        data = pickle.load(f)

    frames = [data['frame']]

    env = retro.make(game='Zelda-NES', state="0_77.state", inttype=retro.data.Integrations.CUSTOM_ONLY)
    env = ObservationWrapper(env, frames, grayscale=True, kind='viewport', normalize=False)

    env.trim_normalize_grayscale(data['info'], frames[0])


if __name__ == '__main__':
    main()
