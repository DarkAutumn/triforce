# pylint: disable

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from tests.utilities import ZeldaActionReplay

def main(savestate):
    replay = ZeldaActionReplay(savestate, render_mode="human")
    replay.reset()

    while True:
        action = input('Enter action: ')
        if 'q' in action:
            break

        replay.run_steps(action)
        print(replay.actions_taken)


if __name__ == '__main__':
    savestate = sys.argv[1]
    main(savestate)
