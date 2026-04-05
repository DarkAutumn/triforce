#!/usr/bin/env python
"""Walk every reachable tile in a room and validate can_link_move against actual NES behavior.

Usage:
    python scripts/test_walkability.py <savestate> [savestate2 ...] [--start-x 10] [--start-y 10]

Example:
    python scripts/test_walkability.py debug_0_68 debug_0_77 debug_1_73
"""

import argparse
import sys

import numpy as np
import stable_retro as retro

from triforce.zelda_enums import Direction, Position, GAMEPLAY_START_Y, MapLocation
from triforce.zelda_game import ZeldaGame

# Directions to test at each tile
CARDINAL_DIRS = [Direction.N, Direction.S, Direction.W, Direction.E]

# Maximum raw NES frames to press a direction button when testing movement
MAX_MOVE_FRAMES = 16

# Colors for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def get_location(info):
    """Build a MapLocation from the raw retro info dict."""
    return MapLocation(info['level'], info['location'], info['mode'] == 11)


def pixel_moved_in_direction(start, end, direction):
    """Check if Link's pixel position moved in the given direction (ignoring perpendicular axis)."""
    match direction:
        case Direction.N:
            return end.y < start.y
        case Direction.S:
            return end.y > start.y
        case Direction.E:
            return end.x > start.x
        case Direction.W:
            return end.x < start.x
    return False


def make_button_action(buttons, direction):
    """Create a MultiBinary NES action pressing the given direction."""
    action = np.zeros(len(buttons), dtype=np.int8)
    name_map = {
        Direction.N: 'UP',
        Direction.S: 'DOWN',
        Direction.W: 'LEFT',
        Direction.E: 'RIGHT',
    }
    btn_name = name_map[direction]
    action[buttons.index(btn_name)] = 1
    return action


def sync_step(env, no_action):
    """Step once with no action to sync the info dict after set_state."""
    obs, _, terminated, truncated, info = env.step(no_action)
    return info


def make_game(env, info):
    """Construct a ZeldaGame from the raw retro env + info dict."""
    return ZeldaGame(env, dict(info), 0)


def settle_grid_offset(env, buttons, direction):
    """Continue pressing the movement direction until link_grid_offset reaches 0.

    After walking to a new tile, Link may still be mid-grid (grid_offset != 0).
    The NES constrains movement direction until grid_offset is 0, so we must
    keep pressing the same direction to reach the next grid line.

    Returns (info, settled): info dict and whether grid_offset reached 0.
    """
    action = make_button_action(buttons, direction)
    for _ in range(MAX_MOVE_FRAMES):
        obs, _, terminated, truncated, info = env.step(action)
        if info.get('link_grid_offset', 0) == 0:
            return info, True
    # Fallback: step with no action to avoid runaway movement
    no_action = np.zeros(len(buttons), dtype=np.int8)
    obs, _, terminated, truncated, info = env.step(no_action)
    return info, False


def test_direction(env, buttons, direction, start_info):
    """Press a direction for up to MAX_MOVE_FRAMES and report whether Link actually moved.

    Returns:
        (moved: bool, new_tile: tuple|None, new_room: bool, end_pos: Position)
    """
    no_action = np.zeros(len(buttons), dtype=np.int8)
    action = make_button_action(buttons, direction)
    start_pos = Position(start_info['link_x'], start_info['link_y'])
    start_tile = start_pos.tile_index
    start_loc = get_location(start_info)

    moved = False
    new_tile = None
    new_room = False
    end_pos = start_pos

    for _ in range(MAX_MOVE_FRAMES):
        obs, _, terminated, truncated, info = env.step(action)

        end_pos = Position(info['link_x'], info['link_y'])
        cur_loc = get_location(info)

        # Room transition?
        if cur_loc != start_loc:
            new_room = True
            moved = True
            break

        # Pixel movement in target direction?
        if pixel_moved_in_direction(start_pos, end_pos, direction):
            moved = True

        # Tile boundary crossed?
        cur_tile = end_pos.tile_index
        if (cur_tile.x, cur_tile.y) != (start_tile.x, start_tile.y):
            new_tile = (cur_tile.x, cur_tile.y)
            break

    # If we moved but haven't reached a new tile yet, keep trying
    if moved and new_tile is None and not new_room:
        for _ in range(MAX_MOVE_FRAMES):
            obs, _, terminated, truncated, info = env.step(action)

            end_pos = Position(info['link_x'], info['link_y'])
            cur_loc = get_location(info)

            if cur_loc != start_loc:
                new_room = True
                break

            cur_tile = end_pos.tile_index
            if (cur_tile.x, cur_tile.y) != (start_tile.x, start_tile.y):
                new_tile = (cur_tile.x, cur_tile.y)
                break

    # If we reached a new tile, settle grid_offset to 0 before saving state.
    if new_tile and not new_room:
        settle_info, settled = settle_grid_offset(env, buttons, direction)
        if not settled:
            # Grid offset didn't reach 0 — tile state is unreliable, skip it
            new_tile = None
        else:
            end_pos = Position(settle_info['link_x'], settle_info['link_y'])
            settled_tile = end_pos.tile_index
            new_tile = (settled_tile.x, settled_tile.y)

    return moved, new_tile, new_room, end_pos


def run_walkability_test(savestate, start_x, start_y):
    """Main BFS walkability test. Returns list of bug strings."""
    env = retro.make(
        game='Zelda-NES',
        state=savestate,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )

    buttons = env.buttons
    no_action = np.zeros(len(buttons), dtype=np.int8)

    obs, info = env.reset()

    # Max out health to reduce enemy interference
    env.data.set_value('hearts_and_containers', 0xFF)
    env.data.set_value('partial_hearts', 0xFF)

    # Position Link at the start tile
    start_px = start_x * 8
    start_py = start_y * 8 + GAMEPLAY_START_Y
    env.data.set_value('link_x', start_px)
    env.data.set_value('link_y', start_py)
    env.data.set_value('link_grid_offset', 0)

    # Save emulator state BEFORE stepping so the NES can't override our position.
    initial_state = env.em.get_state()

    # Step once to populate the info dict
    info = sync_step(env, no_action)

    # Restore our clean state (the step may have moved Link via game logic)
    env.em.set_state(initial_state)
    info = sync_step(env, no_action)

    # Verify starting position and grid alignment
    actual_pos = Position(info['link_x'], info['link_y'])
    actual_tile = actual_pos.tile_index
    grid_off = info.get('link_grid_offset', 0)
    print(f"{BOLD}=== {savestate} ==={RESET}")
    print(f"  Level: {info['level']}, Location: 0x{info['location']:02X}, "
          f"Start tile: ({actual_tile.x}, {actual_tile.y}), grid_offset: {grid_off}")

    if grid_off != 0:
        print(f"  {YELLOW}Warning: grid_offset={grid_off} at start{RESET}")

    # Re-save the state after the sync step
    initial_state = env.em.get_state()

    # BFS data structures
    start_tile = (actual_tile.x, actual_tile.y)
    visited = {start_tile}
    stack = [(start_tile, initial_state)]

    bugs = []
    passes = 0
    tiles_tested = 0
    directions_tested = 0

    while stack:
        tile, em_state = stack.pop()
        tiles_tested += 1
        tx, ty = tile

        for direction in CARDINAL_DIRS:
            # Restore emulator state for this tile
            env.em.set_state(em_state)
            info = sync_step(env, no_action)

            # Construct ZeldaGame and get the prediction
            game = make_game(env, info)
            predicted = game.can_link_move(direction)
            game.deactivate()

            # Actually try to move in the NES
            moved, new_tile, new_room, end_pos = test_direction(
                env, buttons, direction, info
            )

            directions_tested += 1

            # Compare prediction vs reality
            start_pos = Position(info['link_x'], info['link_y'])
            grid_off = info.get('link_grid_offset', 0)

            if moved and not predicted:
                bug = (f"FALSE NEGATIVE at tile ({tx},{ty}) dir={direction.name}: "
                       f"Link moved to ({end_pos.x},{end_pos.y}) but can_link_move=False "
                       f"[start=({start_pos.x},{start_pos.y}), grid_off={grid_off}]")
                bugs.append(bug)
                print(f"  {RED}FAIL{RESET} ({tx:2},{ty:2}) {direction.name}: "
                      f"moved but MASKED  px=({start_pos.x},{start_pos.y})")
            elif not moved and predicted:
                bug = (f"FALSE POSITIVE at tile ({tx},{ty}) dir={direction.name}: "
                       f"Link stuck at ({end_pos.x},{end_pos.y}) but can_link_move=True "
                       f"[start=({start_pos.x},{start_pos.y}), grid_off={grid_off}]")
                bugs.append(bug)
                print(f"  {RED}FAIL{RESET} ({tx:2},{ty:2}) {direction.name}: "
                      f"stuck but UNMASKED  px=({start_pos.x},{start_pos.y})")
            else:
                passes += 1

            # If we reached a new tile in the same room, add it to the BFS
            if new_tile and not new_room and new_tile not in visited:
                new_tile_state = env.em.get_state()
                visited.add(new_tile)
                stack.append((new_tile, new_tile_state))

    # Summary
    print(f"  Tiles: {tiles_tested}, Dirs: {directions_tested}, "
          f"{GREEN}Pass: {passes}{RESET}, {RED}Fail: {len(bugs)}{RESET}")

    if bugs:
        for i, bug in enumerate(bugs, 1):
            print(f"    {i}. {RED}{bug}{RESET}")

    env.close()
    return bugs



def main():
    parser = argparse.ArgumentParser(
        description="Walk every reachable tile in a room and validate can_link_move masking."
    )
    parser.add_argument('savestates', nargs='+', help='Name(s) of savestates to test')
    parser.add_argument('--start-x', type=int, default=10,
                        help='Starting tile X coordinate (default: 10)')
    parser.add_argument('--start-y', type=int, default=10,
                        help='Starting tile Y coordinate (default: 10)')
    args = parser.parse_args()

    total_bugs = 0
    all_bugs = {}
    for savestate in args.savestates:
        bugs = run_walkability_test(savestate, args.start_x, args.start_y)
        if bugs:
            all_bugs[savestate] = bugs
        total_bugs += len(bugs)
        print()

    # Grand summary
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}Grand Summary: {len(args.savestates)} states tested{RESET}")
    if total_bugs == 0:
        print(f"  {GREEN}{BOLD}ALL CLEAN — no bugs found!{RESET}")
    else:
        print(f"  {RED}{BOLD}{total_bugs} total bugs across {len(all_bugs)} states{RESET}")
        for state, bugs in all_bugs.items():
            print(f"  {state}: {len(bugs)} bugs")

    sys.exit(1 if total_bugs > 0 else 0)


if __name__ == '__main__':
    main()
