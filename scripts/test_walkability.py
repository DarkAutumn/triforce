#!/usr/bin/env python
"""Walk every reachable tile in a room and validate movement + weapon masking against NES behavior.

Usage:
    python scripts/test_walkability.py <savestate> [savestate2 ...] [--start-x 10] [--start-y 10]
    python scripts/test_walkability.py <savestate> --weapons-only

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


def test_weapon_at_tile(env, buttons, no_action, direction, weapon):
    """Test if a weapon can actually be used at Link's current position facing the given direction.

    Args:
        weapon: 'sword', 'bomb', or 'boomerang'
    Returns:
        True if the weapon activates within 4 frames.
    """
    data = env.data
    data.set_value('link_direction', direction.value)
    data.set_value('sword_animation', 0)
    data.set_value('bomb_or_flame_animation', 0)
    data.set_value('bait_or_boomerang_animation', 0)

    if weapon == 'bomb':
        data.set_value('selected_item', 1)  # Bombs
        data.set_value('bombs', 8)
    elif weapon == 'boomerang':
        data.set_value('selected_item', 0)  # Boomerang

    # Step once to apply RAM changes
    env.step(no_action)

    # Press the appropriate button
    action = np.zeros(len(buttons), dtype=np.int8)
    if weapon == 'sword':
        action[buttons.index('A')] = 1
        anim_key = 'sword_animation'
    elif weapon == 'bomb':
        action[buttons.index('B')] = 1
        anim_key = 'bomb_or_flame_animation'
    elif weapon == 'boomerang':
        action[buttons.index('B')] = 1
        anim_key = 'bait_or_boomerang_animation'

    for _ in range(4):
        _, _, _, _, info = env.step(action)
        if info[anim_key] != 0:
            return True
    return False


def run_weapon_test(savestate, start_x, start_y):
    """BFS to collect all reachable tiles, then test weapons at each tile."""
    env = retro.make(
        game='Zelda-NES',
        state=savestate,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )

    buttons = env.buttons
    no_action = np.zeros(len(buttons), dtype=np.int8)

    obs, info = env.reset()

    # Max health, give equipment
    env.data.set_value('hearts_and_containers', 0xFF)
    env.data.set_value('partial_hearts', 0xFF)
    env.data.set_value('sword', 1)       # Wood sword
    env.data.set_value('bombs', 8)
    env.data.set_value('regular_boomerang', 1)

    # Position Link at the start tile
    start_px = start_x * 8
    start_py = start_y * 8 + GAMEPLAY_START_Y
    env.data.set_value('link_x', start_px)
    env.data.set_value('link_y', start_py)
    env.data.set_value('link_grid_offset', 0)

    initial_state = env.em.get_state()
    info = sync_step(env, no_action)
    env.em.set_state(initial_state)
    info = sync_step(env, no_action)

    actual_pos = Position(info['link_x'], info['link_y'])
    actual_tile = actual_pos.tile_index
    print(f"{BOLD}=== {savestate} (weapons) ==={RESET}")
    print(f"  Level: {info['level']}, Location: 0x{info['location']:02X}")

    initial_state = env.em.get_state()

    # Phase 1: BFS to collect all reachable tiles with saved emulator states
    start_tile = (actual_tile.x, actual_tile.y)
    visited = {start_tile}
    stack = [(start_tile, initial_state)]
    tile_states = {start_tile: initial_state}

    while stack:
        tile, em_state = stack.pop()
        for direction in CARDINAL_DIRS:
            env.em.set_state(em_state)
            info = sync_step(env, no_action)
            moved, new_tile, new_room, _ = test_direction(env, buttons, direction, info)
            if new_tile and not new_room and new_tile not in visited:
                new_state = env.em.get_state()
                visited.add(new_tile)
                stack.append((new_tile, new_state))
                tile_states[new_tile] = new_state

    print(f"  Reachable tiles: {len(tile_states)}")

    # Phase 2: Test weapons at each tile
    bugs = []
    passes = 0
    skipped = 0
    weapons = ['sword', 'bomb', 'boomerang']

    for tile, em_state in sorted(tile_states.items()):
        tx, ty = tile

        # Quick check: if ALL weapons in ALL directions are blocked, the NES is in a
        # non-gameplay state (e.g., cave transition with scroll_type=0xFF in debug states).
        # Skip this tile rather than reporting false positives.
        env.em.set_state(em_state)
        info = sync_step(env, no_action)
        env.data.set_value('sword', 1)
        env.data.set_value('bombs', 8)
        env.data.set_value('regular_boomerang', 1)
        any_weapon_works = False
        for d in CARDINAL_DIRS:
            env.em.set_state(em_state)
            sync_step(env, no_action)
            env.data.set_value('sword', 1)
            if test_weapon_at_tile(env, buttons, no_action, d, 'sword'):
                any_weapon_works = True
                break
        if not any_weapon_works:
            skipped += 1
            print(f"  {YELLOW}SKIP{RESET} ({tx:2},{ty:2}) — NES blocks all weapons (transition state)")
            continue

        for direction in CARDINAL_DIRS:
            # Get predictions
            env.em.set_state(em_state)
            info = sync_step(env, no_action)
            game = make_game(env, info)
            sword_dirs = set(game.link.get_sword_directions_allowed())
            item_dirs = set(game.link.get_item_directions_allowed())
            game.deactivate()

            for weapon in weapons:
                if weapon == 'sword':
                    predicted = direction in sword_dirs
                else:
                    predicted = direction in item_dirs

                # Test actual NES behavior
                env.em.set_state(em_state)
                info = sync_step(env, no_action)
                # Give equipment (state may not have it)
                env.data.set_value('sword', 1)
                env.data.set_value('bombs', 8)
                env.data.set_value('regular_boomerang', 1)

                actual = test_weapon_at_tile(env, buttons, no_action, direction, weapon)

                if actual and not predicted:
                    pos = Position(info['link_x'], info['link_y'])
                    bug = (f"FALSE NEG {weapon.upper()} at tile ({tx},{ty}) dir={direction.name}: "
                           f"weapon WORKS but predicted=False [px=({pos.x},{pos.y})]")
                    bugs.append(bug)
                    print(f"  {RED}FAIL{RESET} ({tx:2},{ty:2}) {direction.name} "
                          f"{weapon:10s}: works but MASKED  px=({pos.x},{pos.y})")
                elif not actual and predicted:
                    pos = Position(info['link_x'], info['link_y'])
                    bug = (f"FALSE POS {weapon.upper()} at tile ({tx},{ty}) dir={direction.name}: "
                           f"weapon BLOCKED but predicted=True [px=({pos.x},{pos.y})]")
                    bugs.append(bug)
                    print(f"  {RED}FAIL{RESET} ({tx:2},{ty:2}) {direction.name} "
                          f"{weapon:10s}: blocked but UNMASKED  px=({pos.x},{pos.y})")
                else:
                    passes += 1

    skipped_note = f", {YELLOW}{skipped} skipped{RESET}" if skipped else ""
    print(f"  Weapon tests: {passes} pass, {RED}{len(bugs)} fail{RESET}{skipped_note}")
    if bugs:
        for i, bug in enumerate(bugs, 1):
            print(f"    {i}. {RED}{bug}{RESET}")

    env.close()
    return bugs


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
        description="Walk every reachable tile in a room and validate movement + weapon masking."
    )
    parser.add_argument('savestates', nargs='+', help='Name(s) of savestates to test')
    parser.add_argument('--start-x', type=int, default=10,
                        help='Starting tile X coordinate (default: 10)')
    parser.add_argument('--start-y', type=int, default=10,
                        help='Starting tile Y coordinate (default: 10)')
    parser.add_argument('--weapons-only', action='store_true',
                        help='Only test weapon masking, skip movement')
    parser.add_argument('--movement-only', action='store_true',
                        help='Only test movement masking, skip weapons')
    args = parser.parse_args()

    total_bugs = 0
    all_bugs = {}

    for savestate in args.savestates:
        state_bugs = []

        if not args.weapons_only:
            bugs = run_walkability_test(savestate, args.start_x, args.start_y)
            state_bugs.extend(bugs)
            print()

        if not args.movement_only:
            bugs = run_weapon_test(savestate, args.start_x, args.start_y)
            state_bugs.extend(bugs)
            print()

        if state_bugs:
            all_bugs[savestate] = state_bugs
        total_bugs += len(state_bugs)

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
