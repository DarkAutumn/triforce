import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from triforce_lib import action_replay

def main(savestate):
    replay = action_replay.ZeldaActionReplay(savestate)
    replay.reset()

    while True:
        action = input('Enter action: ')

        i = 0
        while i < len(action):
            a = action[i]
            count = 0
            idx = i + 1
            while idx < len(action) and '0' <= action[idx] <= '9':
                count = count * 10 + int(action[idx])
                idx += 1

            if a == 'q':
                return

            if a == 'c':
                replay.reset()
                break

            for i in range(max(count, 1)):
                replay.step(a)

            i = idx
        print(replay.actions_taken)


if __name__ == '__main__':
    savestate = sys.argv[1]
    main(savestate)