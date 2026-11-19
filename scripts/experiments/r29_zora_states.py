"""Round 29 (ow-zora, docs/experiments/round29.md "One ow-zora for every stretch"): the pre-entry states of route
steps inside ow-zora's stretch as chain runs enter them, as ow-zora's harvested starts.

    python scripts/experiments/r29_zora_states.py REPLAY_DIR OUT_DIR --step 44:0_38n [--step 69:0_38w ...]
        [--count N]

``eval-chain --harvest`` saves handoffs only (a room where the acting family changes); 0:0x38 from N follows
ow-zora's own 0x28, so its entry has no handoff. This cuts the frame before the room entry (as ``entry_states`` and
``eval-chain --harvest`` save it; ``r26_arrow_states.py`` did the same for 0:0x44 at X 128) out of chain replays
(``eval-chain --replays DIR``) for every replay that enters the route step, and writes it as a harvested state
``OUT_DIR/<stem>_h<seed>_0.state`` (``STEP:STEM``: the step's seed-state stem, e.g. ``44:0_38n``). A harvested start
keeps the hearts, bombs and position the chain brought (no ``ram_randomization``). ``--count`` keeps the first N
per step (by seed). Prints Link's X and Y on entry (12 frames in) of the states written.
"""

import argparse
import collections
import glob
import os

from triforce import _native

# zelda::room_entry: the game modes of a room transition and of a room entry, $12 and $11 (IsUpdatingMode).
TRANSITION_MODES = {0x02, 0x03, 0x06, 0x07, 0x0A, 0x10}
ENTER_MODES = {0x04, 0x09, 0x0B, 0x0C}
MODE, IS_UPDATING_MODE, LEVEL, ROOM, LINK_X, LINK_Y = 0x12, 0x11, 0x10, 0xEB, 0x70, 0x84


def is_pre_entry(ram) -> bool:
    mode = int(ram[MODE])
    return mode in TRANSITION_MODES or (mode in ENTER_MODES and ram[IS_UPDATING_MODE] == 0)


def pre_entry_frame(player, entry_frame: int, level: int, room: int):
    """The last frame up to the entry event where the game is about to enter (level, room)."""
    found = None
    for frame in range(max(0, entry_frame - 120), entry_frame + 1):
        player.seek(frame)
        ram = player.ram()
        if int(ram[MODE]) in (0x03, 0x04) and is_pre_entry(ram) and ram[LEVEL] == level and ram[ROOM] == room:
            found = frame
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('replays')
    parser.add_argument('out')
    parser.add_argument('--step', action='append', required=True, metavar='STEP:STEM',
                        help="a route step and its seed-state stem (repeatable)")
    parser.add_argument('--count', type=int, default=0, help="at most N states a step (0: every one)")
    args = parser.parse_args()
    steps = {int(step): stem for step, stem in (item.split(':', 1) for item in args.step)}
    os.makedirs(args.out, exist_ok=True)
    saved = collections.Counter()
    positions = collections.defaultdict(collections.Counter)
    paths = sorted(glob.glob(os.path.join(args.replays, '*.tfreplay')),
                   key=lambda p: int(os.path.basename(p).split('-')[0]))
    for path in paths:
        data = open(path, 'rb').read()
        replay = _native.Replay(data)
        player = None
        seed = int(os.path.basename(path).split('-')[0])
        for step, stem in steps.items():
            if args.count and saved[stem] >= args.count:
                continue
            entries = [e for e in replay.events() if e.get('type') == 'room' and e.get('step') == step]
            if not entries:
                continue
            entry = entries[0]
            player = player or _native.ReplayPlayer(data)
            frame = pre_entry_frame(player, entry['frame'], entry['level'], entry['room'])
            if frame is None:
                continue
            player.seek(entry['frame'] + 12)
            ram = player.ram()
            position = (int(ram[LINK_X]), int(ram[LINK_Y]))
            player.seek(frame)
            target = os.path.join(args.out, f"{stem}_h{seed}_0.state")
            with open(target, 'wb') as f:
                f.write(player.save_state())
            game = _native.Game()
            game.load(target)
            assert game.ram()[ROOM] == entry['room'], target
            saved[stem] += 1
            positions[stem][position] += 1
    for stem, count in sorted(saved.items()):
        common = ', '.join(f"X {x} Y {y}: {n}" for (x, y), n in positions[stem].most_common(6))
        print(f"{stem}: {count} states in {args.out} (on entry {common})")


if __name__ == '__main__':
    main()
