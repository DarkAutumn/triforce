"""Start states for 0:0x44 entered from S as the clips way enters it (round 26, the arrow shop).

The seed states 0_44s_0..2 were built on the plain way and put Link at X 112, 112 and 120; a chain run that comes up
0x54 by the clips way (E -> N) enters at X 128 about half the time, and from there ow-zora walks on N past the fork
to the shop's cave into 0x34 (the run ends `missed`: left 0x44 without the arrows). This cuts the pre-entry state of
that entry out of chain replays (`eval-chain --replays DIR`), the frame before the room entry as `entry_states` and
`eval-chain --harvest` save it, for the replays that enter 0x44 at route step 217 at X 128, and writes them as
harvested states (`<out>/0_44s_h<seed>_0.state`; the family's `harvested_dir`, read at `harvested_fraction`).

    python scripts/experiments/r26_arrow_states.py REPLAY_DIR OUT_DIR [--step 217] [--x 128]
"""

import argparse
import glob
import os

from triforce import _native

# zelda::room_entry: the game modes of a room transition and of a room entry, $12 and $11 (IsUpdatingMode).
TRANSITION_MODES = {0x02, 0x03, 0x06, 0x07, 0x0A, 0x10}
ENTER_MODES = {0x04, 0x09, 0x0B, 0x0C}
MODE, IS_UPDATING_MODE, LEVEL, ROOM, LINK_X = 0x12, 0x11, 0x10, 0xEB, 0x70


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
    parser.add_argument('--step', type=int, default=217, help="the route step entering 0:0x44 from S")
    parser.add_argument('--x', type=int, default=128, help="Link's X on entry to keep")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    saved = 0
    for path in sorted(glob.glob(os.path.join(args.replays, '*.tfreplay'))):
        data = open(path, 'rb').read()
        replay = _native.Replay(data)
        entries = [e for e in replay.events() if e.get('type') == 'room' and e.get('step') == args.step]
        if not entries:
            continue
        entry = entries[0]
        player = _native.ReplayPlayer(data)
        frame = pre_entry_frame(player, entry['frame'], entry['level'], entry['room'])
        if frame is None:
            continue
        player.seek(entry['frame'] + 12)
        if int(player.ram()[LINK_X]) != args.x:
            continue
        player.seek(frame)
        seed = int(os.path.basename(path).split('-')[0])
        target = os.path.join(args.out, f"0_44s_h{seed}_0.state")
        with open(target, 'wb') as f:
            f.write(player.save_state())
        game = _native.Game()
        game.load(target)
        ram = game.ram()
        assert ram[ROOM] == entry['room'] and int(ram[LINK_X]) == args.x, target
        saved += 1
    print(f"{saved} states in {args.out}")


if __name__ == '__main__':
    main()
