#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pyside6-essentials>=6.7", "pygame>=2.1"]
# ///
"""Play The Legend of Zelda yourself on this project's emulator, one room or the whole game, and keep it as a replay.

    uv run emu-zelda.py                         # the whole game from console power on, as LINK
    uv run emu-zelda.py --name ZELDA            # the second quest
    uv run emu-zelda.py --reset 120 --invincible --bombs infinite
    uv run emu-zelda.py 3 59                    # Dungeon 3's room 0x59, as d3first's start state has it
    uv run emu-zelda.py 5 05 --bombs 0          # the Olive Garden with no bombs
    uv run emu-zelda.py o 1f                    # the overworld's 0x1f (level 0 or o)
    uv run emu-zelda.py 3 69 --entry cave --hearts 2.5 --seed 7
    uv run emu-zelda.py 5 05 --bombs 0 --item bombs --enemies 2   # bombs on B, two of the six Blue Darknuts
    .venv/bin/python emu-zelda.py 3 59          # the same from the project's environment

The whole game (no LEVEL and ROOM): the console powers on, the name (--name, LINK by default) is registered in slot 1
and the game starts, all by controller inputs and all in the recording; you take over on the first screen, 0x77,
paused.  --reset [DELAY] then presses the console's RESET, as the real button does (RAM and the battery-saved file
kept; the game's own reset code seeds its random bytes), presses START on the title screen as soon as it takes it,
waits DELAY frames (0 if none is given) on the file-select screen once it takes input, and starts slot 1: the game
restarts its random bytes from the same seed at every reset, so the delay alone decides them (the same delay, the
same game, whatever the name).  The frame and the thirteen random bytes $18-$24 at the hand-over are printed and
shown under the picture.  A name is 1 to 8 of A-Z, 0-9 and space; one that starts with ZELDA plays the second quest,
as on the console.  The take is DONE only at the ending, on the frame after Link touches Zelda; dying is the game's
own (CONTINUE, SAVE, RETRY: Select and Start).  Its replay: start_<name>[_reset<delay>][_<cheats>]_<time>.tfreplay.

One room (LEVEL ROOM): LEVEL is 0 (or o) for the overworld, 1-9 for a dungeon; ROOM the room id in hex without 0x.
Link walks in with the route's own start state for the room (--route, default d3first: its first visit, from the
side the route first enters by, chain 0; --entry, --visit, --chain pick another); --bombs, --hearts, --keys,
--rupees and --item (the B item selected, one Link holds there) are written over it and kept in the replay as memory
edits.  --enemies N leaves at most N monsters in a dungeon room: the others count as killed before Link came in (the
game's own kill count for the room, as training's staged rooms bank them), so the game makes N, where and which as
its spawn draw from the seed says, and the room still opens when they die.  A room the route never enters is
refused, with the rooms it does enter.  The take is DONE when Link dies or stands in another room (a cave or cellar
under the room is still the same take); it is named for the room, entry, overrides, seed and time, e.g.
3x59_N_bombs4_seed123_261124-101502.tfreplay.

Both: --invincible puts Link back to full hearts on every frame he lost some (a hit still knocks him back and
flashes him; he cannot die); --bombs, --keys and --rupees take a count or `infinite`: 8 bombs, 8 keys or 255 rupees
to start with, back to that the frame after one is spent.  Each of their writes is a memory edit in the replay.

The game loads PAUSED (two bars over the picture).  Any game button starts it, and that press counts on the first
frame; the recording runs on.  The replay is saved by its key only when DONE or paused, into ~/triforce-review/played/
(--out); it plays in play-zelda.py and passes `tf replay verify`.  Saving while paused keeps the recording going.  The
state key saves the machine at that frame as a TFZSTATE file into the same folder, named for the room Link stands in,
the frame and the time (5x47_frame81234_261125-101502.state), while play goes on.  Starting over (DONE or paused; the
whole game: DONE only) loads the room again, or powers the console on again, paused, and starts a new recording;
the whole game's take is saved first, and on closing the window, if it holds frames no saved file holds.

The rate starts at the NES's 60.1 frames a second; - and + move it by 5 (5.1 to 240.1) in any state, the title
shows it.  The sound (on unless --no-sound) plays only between 50.1 and 70.1.  The window opens on the centermost
monitor.

Keys:
{keys}
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))           # `uv run` runs this in its own environment: the checkout's triforce package

from PySide6.QtWidgets import QApplication  # noqa: E402  pylint: disable=wrong-import-position

try:
    from triforce import emu, emu_view  # noqa: E402  pylint: disable=wrong-import-position
except ImportError as error:              # the compiled module is not built in this checkout
    sys.exit(f"emu-zelda: cannot load the project's compiled module ({error}); build it with "
             f"`uv pip install --python .venv/bin/python -e '.[dev]'` or run `.venv/bin/python emu-zelda.py`")

__doc__ = __doc__.replace('{keys}', emu_view.KEYS_HELP)  # pylint: disable=redefined-builtin

ONE_ROOM = ('route', 'entry', 'visit', 'chain', 'hearts', 'item', 'enemies', 'seed', 'done')
WHOLE_GAME = ('name', 'reset')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='emu-zelda.py', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('level', nargs='?', type=emu.parse_level,
                        help="0 or o: the overworld; 1-9: a dungeon (none: the whole game from power on)")
    parser.add_argument('room', nargs='?', type=emu.parse_room, help="the room id in hex without 0x (59, 0f)")
    whole = parser.add_argument_group('the whole game')
    whole.add_argument('--name', nargs='?', const=emu.DEFAULT_NAME,
                       help=f"the name registered (default {emu.DEFAULT_NAME}); ZELDA: the second quest")
    whole.add_argument('--reset', nargs='?', type=int, const=0, metavar='DELAY',
                       help="after the game starts, press RESET and start it again after DELAY frames on the "
                            "file-select screen (default 0)")
    room = parser.add_argument_group('one room')
    room.add_argument('--route', help=f"whose start state (default {emu.DEFAULT_ROUTE})")
    room.add_argument('--entry', choices=emu.ENTRIES, type=lambda t: 'cave' if t.lower() == 'cave' else t.upper(),
                      help="the side Link comes in by (cave: out of a cave or a dungeon's stairs); default: the "
                           "route's first")
    room.add_argument('--visit', type=int, help="the route's visit from that side (default 1)")
    room.add_argument('--chain', type=int, choices=(0, 1, 2), help="the start state's chain (default 0)")
    room.add_argument('--hearts', type=float, help="hearts, halves allowed (the containers grow to hold them)")
    room.add_argument('--item', choices=list(emu.ITEMS),
                      help="the B item selected at the load (one Link holds there; bombs always)")
    room.add_argument('--enemies', type=int, help="at most N monsters in the (dungeon) room")
    room.add_argument('--seed', type=int, help="the first start's seed (the RNG phase; default: drawn)")
    room.add_argument('--done', choices=sorted(emu.DONE_RULES),
                      help="when a take is DONE (room, the default: Link dies or leaves the room)")
    both = parser.add_argument_group('both')
    both.add_argument('--invincible', action='store_true', help="Link back to full hearts whenever he loses some")
    both.add_argument('--bombs', type=emu.parse_count, metavar='N|infinite', help="bombs held")
    both.add_argument('--keys', type=emu.parse_count, metavar='N|infinite', help="keys held")
    both.add_argument('--rupees', type=emu.parse_count, metavar='N|infinite', help="rupees held")
    both.add_argument('--no-sound', action='store_true', help="no sound")
    both.add_argument('--scale', type=int, default=3, help="the picture's whole-number scale (default 3)")
    both.add_argument('--out', type=Path, default=emu.PLAYED_DIR,
                      help=f"where replays and states go (default {emu.PLAYED_DIR})")
    return parser


def session_of(parser: argparse.ArgumentParser, args: argparse.Namespace):
    """The session the arguments ask for: ``GameSession`` without LEVEL and ROOM, else ``Session``."""
    given = {name for name in ONE_ROOM + WHOLE_GAME if getattr(args, name) is not None}
    counts = {'bombs': args.bombs, 'keys': args.keys, 'rupees': args.rupees, 'invincible': args.invincible}
    if args.level is None:
        if given & set(ONE_ROOM):
            parser.error(f"--{', --'.join(sorted(given & set(ONE_ROOM)))}: one room only (give LEVEL and ROOM)")
        spec = emu.GameSpec(name=args.name or emu.DEFAULT_NAME, reset=args.reset, **counts)
        if spec.second_quest:
            print(f"the name {spec.name.upper()} plays the second quest", flush=True)
        return emu.GameSession(spec)
    if args.room is None:
        parser.error("give ROOM after LEVEL (or neither, for the whole game)")
    if given & set(WHOLE_GAME):
        parser.error(f"--{', --'.join(sorted(given & set(WHOLE_GAME)))}: the whole game only (no LEVEL and ROOM)")
    spec = emu.Spec(args.level, args.room, route=args.route or emu.DEFAULT_ROUTE, entry=args.entry,
                    visit=1 if args.visit is None else args.visit, chain=args.chain or 0, hearts=args.hearts,
                    item=args.item, enemies=args.enemies, **counts)
    return emu.Session(spec, seed=args.seed, done_rule=args.done or 'room')


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.scale < 1:
        parser.error("--scale is a whole number from 1")
    try:
        session = session_of(parser, args)
    except (emu.StartError, ValueError, RuntimeError) as error:
        print(f"emu-zelda: {error}", file=sys.stderr)
        return 2
    app = QApplication.instance() or QApplication(sys.argv[:1])
    pad = emu_view.Pad()
    print(session.start_text(), flush=True)
    print(f"controller: {pad.name}" if pad.name else f"{pad.reason}", flush=True)
    print(emu_view.KEYS_HELP, flush=True)
    window = emu_view.Window(session, emu_view.Options(scale=args.scale, sound=not args.no_sound,
                                                       played_dir=args.out), pad)
    window.place()
    window.show()
    window.activateWindow()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
