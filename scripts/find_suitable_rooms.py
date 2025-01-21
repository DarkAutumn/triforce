# pylint: disable=all
import os
import sys
import retro

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from triforce.zelda_enums import Direction
from triforce.frame_skip_wrapper import FrameSkipWrapper
from triforce.state_change_wrapper import StateChangeWrapper

def main(directory):
    """Find and print all states with 4 exits."""

    env = None
    for file in os.listdir(directory):
        if file.endswith(".state"):
            if env is None:
                env = retro.make(game='Zelda-NES', state=file, inttype=retro.data.Integrations.CUSTOM_ONLY,
                                 render_mode='rgb_array')
                env = FrameSkipWrapper(env)
                env = StateChangeWrapper(env, None)

            env.load_state(file, retro.data.Integrations.CUSTOM_ONLY)
            _, state = env.reset()

            exits = 0
            for direction in [Direction.N, Direction.E, Direction.S, Direction.W]:
                if direction in state.room.exits and state.room.exits[direction]:
                    exits += 1

            if exits == 4:
                print(f'"{file}",')

    if env:
        env.close()

if __name__ == "__main__":
    integrations = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'triforce', 'custom_integrations')
    retro.data.Integrations.add_custom_path(os.path.join(integrations, 'custom_integrations'))
    main(sys.argv[1])
