"""Play Zelda by hand (or as an LLM) through the exact code path the agent uses.

The session persists between invocations in a snapshot file, so each call can take a few actions and inspect the
result::

    python -m triforce.play --state 1_44w_0           # load a savestate, print the room
    python -m triforce.play "MOVE N 16" "SWORD E"      # continue: run actions, print what happened
    python -m triforce.play "press UP+A 3" --screenshot /tmp/z.png
    python -m triforce.play --ram 0x34f:12             # dump RAM bytes
    python -m triforce.play --save-state /tmp/here.state
    python -m triforce.play --capture /tmp/states "MOVE N 32" "MOVE N 32"   # save a state on each new room entry

Actions use the v2 schema: ``MOVE <dir> <frames>``, ``WAIT <frames>``, ``SWORD|BEAMS|BOMB|ARROW <dir>``,
``BOOMERANG <dir incl. diagonals>``.  ``press <BUTTON[+BUTTON]> <frames>`` drives the raw controller instead
(bypasses the option executor; useful for menus and debugging).
"""

import argparse
import fnmatch
import os
import pickle
import sys

from ._native import Game
from .actions import Action
from .paths import STATES_DIR

DEFAULT_SESSION = os.path.join('.play', 'session.pkl')


def _press(game: Game, command: str) -> str:
    parts = command.split()
    names = parts[1].upper().split('+') if len(parts) > 1 and parts[1].upper() != 'NONE' else []
    frames = int(parts[2]) if len(parts) > 2 else 1
    game.press(names, frames)
    return f"press {'+'.join(names) or 'NONE'} for {frames} frames"


def _parse_ram(spec: str):
    addr, _, length = spec.partition(':')
    return int(addr, 0), int(length, 0) if length else 1


def _capture(game: Game, change, directory: str):
    """Saves a room-entry savestate named like v1's: <level>_<room hex><entry heading n/s/e/w, or c for caves>."""
    state = change.state
    heading = 'c' if state.in_cave else Action.parse(change.action).direction.name.lower()[:1]
    path = os.path.join(directory, f"{state.level}_{state.location:02x}{heading}.state")
    if os.path.exists(path):
        print(f"  (capture skipped, exists: {path})")
        return
    game.save_state_file(path)
    print(f"  captured {path}")


def main(argv=None):
    """CLI entry point."""
    # pylint: disable=too-many-branches, too-many-statements
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('actions', nargs='*', help="actions to run in order")
    parser.add_argument('--state', help="start fresh from this savestate (name like 1_44w_0, or a path)")
    parser.add_argument('--session', default=DEFAULT_SESSION, help="session snapshot file")
    parser.add_argument('--seed-rng', action='store_true', help="keep the savestate's RNG (default: randomize)")
    parser.add_argument('--no-grid', action='store_true', help="don't print the ASCII room map")
    parser.add_argument('--quiet', action='store_true', help="only print per-action results")
    parser.add_argument('--screenshot', help="write a PNG of the current screen")
    parser.add_argument('--save-state', help="write the current emulator state as a .state file")
    parser.add_argument('--ram', action='append', default=[], help="dump RAM: ADDR[:LEN] (hex ok)")
    parser.add_argument('--list-states', nargs='?', const='*', help="list savestates matching a glob")
    parser.add_argument('--set', action='append', default=[], metavar='NAME=VALUE',
                        help="write a named RAM variable before running actions (e.g. hearts_and_containers=0x22)")
    parser.add_argument('--capture', metavar='DIR',
                        help="save a savestate in DIR whenever an action enters a new room (v1 naming)")
    args = parser.parse_args(argv)

    if args.list_states:
        names = sorted(f[:-6] for f in os.listdir(STATES_DIR) if f.endswith('.state'))
        print(' '.join(n for n in names if fnmatch.fnmatch(n, args.list_states)))
        return 0

    game = Game(render=True, states_dir=str(STATES_DIR))
    screen = None       # the emulator only renders on emulated frames; keep the last one across invocations
    if args.state:
        game.load(args.state, randomize_rng=not args.seed_rng)
        screen = game.screen()
    elif os.path.exists(args.session):
        with open(args.session, 'rb') as f:
            saved = pickle.load(f)
        screen = saved['screen']
        game.restore(saved['snapshot'])
    else:
        print(f"No session at {args.session}; start one with --state <name>", file=sys.stderr)
        return 1

    for assignment in args.set:
        name, _, value = assignment.partition('=')
        game.set_value(name.strip(), int(value, 0))

    for command in args.actions:
        if command.lower().startswith('press'):
            print(_press(game, command))
            screen = game.screen()
            continue
        action = Action.parse(command)
        legal = game.is_allowed(str(action))
        change = game.act(str(action))
        screen = game.screen()
        print(change.describe() + ('' if legal else '  (WARNING: action was masked out)'))
        if args.capture and change.changed_location and not change.state.game_over:
            _capture(game, change, args.capture)
        if change.state.game_over:
            print("GAME OVER")
            break

    os.makedirs(os.path.dirname(os.path.abspath(args.session)), exist_ok=True)
    with open(args.session, 'wb') as f:
        pickle.dump({'snapshot': game.snapshot(), 'screen': screen}, f)

    if not args.quiet:
        print(game.describe_state())
        if not args.no_grid:
            print(game.render_grid())

    for spec in args.ram:
        addr, length = _parse_ram(spec)
        values = game.ram()[addr:addr + length]
        print(f"RAM 0x{addr:03x}: " + ' '.join(f"{v:02x}" for v in values))

    if args.screenshot and screen is not None:
        from PIL import Image  # pylint: disable=import-outside-toplevel
        Image.fromarray(screen).resize((512, 480), Image.NEAREST).save(args.screenshot)
        print(f"screenshot: {args.screenshot}")

    if args.save_state:
        game.save_state_file(args.save_state)
        print(f"saved state: {args.save_state}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
