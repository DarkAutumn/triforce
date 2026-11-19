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


def matches(ram, level: int, room: int) -> bool:
    return int(ram[MODE]) in (0x03, 0x04) and is_pre_entry(ram) and ram[LEVEL] == level and ram[ROOM] == room


def cut(player, entry: dict):
    """(the state of the last frame up to the entry event where the game is about to enter the entry's room, Link's
    (X, Y) 12 frames after the entry), or None; the player only moves forward (a backward seek replays from the
    start)."""
    start = max(0, entry['frame'] - 120)
    if player.frame < start:
        player.seek(start)
    found = None
    while True:
        if matches(player.ram(), entry['level'], entry['room']):
            found = player.save_state()
        if player.frame >= entry['frame'] or not player.step(1):
            break
    if found is None:
        return None
    player.step(12)
    ram = player.ram()
    return found, (int(ram[LINK_X]), int(ram[LINK_Y]))


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
        seed = int(os.path.basename(path).split('-')[0])
        entries = []
        for step, stem in steps.items():
            found = [e for e in replay.events() if e.get('type') == 'room' and e.get('step') == step]
            if found and not (args.count and saved[stem] >= args.count):
                entries.append((found[0], stem))
        player = _native.ReplayPlayer(data) if entries else None
        for entry, stem in sorted(entries, key=lambda item: item[0]['frame']):
            if player.frame > entry['frame']:
                continue        # past it already (never seek back)
            result = cut(player, entry)
            if result is None:
                continue
            state, position = result
            target = os.path.join(args.out, f"{stem}_h{seed}_0.state")
            with open(target, 'wb') as f:
                f.write(state)
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
