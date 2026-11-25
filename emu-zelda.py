#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pyside6-essentials>=6.7", "pygame>=2.1"]
# ///
"""Play one room of The Legend of Zelda yourself, on this project's emulator, and keep it as a replay.

    uv run emu-zelda.py 3 59                    # Dungeon 3's room 0x59, as d3first's start state has it
    uv run emu-zelda.py 5 05 --bombs 0          # the Olive Garden with no bombs
    uv run emu-zelda.py o 1f                    # the overworld's 0x1f (level 0 or o)
    uv run emu-zelda.py 3 69 --entry cave --hearts 2.5 --seed 7
    uv run emu-zelda.py 5 05 --bombs 0 --item bombs --enemies 2   # bombs on B, two of the six Blue Darknuts
    .venv/bin/python emu-zelda.py 3 59          # the same from the project's environment

LEVEL is 0 (or o) for the overworld, 1-9 for a dungeon; ROOM the room id in hex without 0x.  Link walks in with the
route's own start state for the room (--route, default d3first: its first visit, from the side the route first
enters by, chain 0; --entry, --visit, --chain pick another); --bombs, --hearts, --keys and --item (the B item
selected, one Link holds there) are written over it and kept in the replay as memory edits.  --enemies N leaves at
most N monsters in a dungeon room: the others count as killed before Link came in (the game's own kill count for
the room, as training's staged rooms bank them), so the game makes N, where and which as its spawn draw from the
seed says, and the room still opens when they die.  A room the route never enters is refused, with the rooms it
does enter.

The room loads PAUSED (two bars over the picture).  Any game button starts it, and that press counts on the first
frame; the recording runs from the load.  The take is DONE when Link dies or stands in another room (a cave or
cellar under the room is still the same take).  The replay is saved by its key only when DONE or paused, into
~/triforce-review/played/ (--out), named for the room, entry, overrides, seed and time, e.g.
3x59_N_bombs4_seed123_261124-101502.tfreplay; it plays in play-zelda.py and passes `tf replay verify`.  Saving while
paused keeps the recording going.  A reset (DONE or paused) loads the room again, paused, and starts a new recording.

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='emu-zelda.py', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('level', type=emu.parse_level, help="0 or o: the overworld; 1-9: a dungeon")
    parser.add_argument('room', type=emu.parse_room, help="the room id in hex without 0x (59, 0f)")
    parser.add_argument('--route', default=emu.DEFAULT_ROUTE, help=f"whose start state (default {emu.DEFAULT_ROUTE})")
    parser.add_argument('--entry', choices=emu.ENTRIES, type=lambda t: 'cave' if t.lower() == 'cave' else t.upper(),
                        help="the side Link comes in by (cave: out of a cave or a dungeon's stairs); default: the "
                             "route's first")
    parser.add_argument('--visit', type=int, default=1, help="the route's visit from that side (default 1)")
    parser.add_argument('--chain', type=int, default=0, choices=(0, 1, 2),
                        help="the start state's chain (default 0; d2first has 0-2, d3first only 0: one state per door "
                             "since round 44)")
    parser.add_argument('--bombs', type=int, help="bombs held (written over the start state)")
    parser.add_argument('--hearts', type=float, help="hearts, halves allowed (the containers grow to hold them)")
    parser.add_argument('--keys', type=int, help="keys held")
    parser.add_argument('--item', choices=list(emu.ITEMS),
                        help="the B item selected at the load (one Link holds there; bombs always)")
    parser.add_argument('--enemies', type=int, help="at most N monsters in the (dungeon) room")
    parser.add_argument('--seed', type=int, help="the first start's seed (the RNG phase; default: drawn)")
    parser.add_argument('--no-sound', action='store_true', help="no sound")
    parser.add_argument('--scale', type=int, default=3, help="the picture's whole-number scale (default 3)")
    parser.add_argument('--done', default='room', choices=sorted(emu.DONE_RULES),
                        help="when a take is DONE (room: Link dies or leaves the room)")
    parser.add_argument('--out', type=Path, default=emu.PLAYED_DIR, help=f"where replays go (default {emu.PLAYED_DIR})")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.scale < 1:
        build_parser().error("--scale is a whole number from 1")
    spec = emu.Spec(args.level, args.room, route=args.route, entry=args.entry, visit=args.visit, chain=args.chain,
                    bombs=args.bombs, hearts=args.hearts, keys=args.keys, item=args.item, enemies=args.enemies)
    try:
        session = emu.Session(spec, seed=args.seed, done_rule=args.done)
    except (emu.StartError, ValueError) as error:
        print(f"emu-zelda: {error}", file=sys.stderr)
        return 2
    app = QApplication.instance() or QApplication(sys.argv[:1])
    pad = emu_view.Pad()
    print(f"{spec.name} from {session.start.entry} ({session.start.state}), seed {session.seed}", flush=True)
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
