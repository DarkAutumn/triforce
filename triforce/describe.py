"""Human/LLM-readable text dumps of the game state.  Used by the ``play`` tool and for debugging."""

from enum import Enum

from .zelda_enums import Direction

CARDINALS = (Direction.N, Direction.S, Direction.W, Direction.E)


def _name(value):
    return value.name if isinstance(value, Enum) else str(value)


def describe_state(state, mask=None) -> str:
    """Multi-line summary: location, Link, inventory, room, entities, legal actions."""
    link = state.link
    lines = []
    lines.append(f"Location: level {state.level} room 0x{state.location:02x}{' (cave)' if state.in_cave else ''}"
                 f"  mode={state.mode}")
    lines.append(f"Link: pos=({link.position.x},{link.position.y}) tile=({link.tile.x},{link.tile.y}) "
                 f"facing={_name(link.direction)} hp={link.health:g}/{link.max_health} status=0x{link.status:02x}")
    lines.append(f"Inventory: sword={_name(link.sword)} bombs={link.bombs} keys={link.keys} rupees={link.rupees} "
                 f"boomerang={_name(link.boomerang)} bow={int(bool(link.bow))} arrows={_name(link.arrows)} "
                 f"beams={'yes' if link.has_beams else 'no'} triforce={link.triforce_pieces}")

    if state.level != 0:
        doors = []
        for d in CARDINALS:
            if state.is_door_locked(d):
                doors.append(f"{d.name}=locked")
            elif state.is_door_barred(d):
                doors.append(f"{d.name}=barred")
            elif state.room.is_door_tile_walkable(d):
                doors.append(f"{d.name}=open")
            else:
                doors.append(f"{d.name}=wall")
        lines.append("Doors: " + ' '.join(doors) + f"  push_block_pending={state.push_block_pending}"
                     f"  bomb_walls={[w.wall_direction.name for w in state.bomb_walls]}")

    treasure = state.treasure
    if treasure is not None:
        lines.append(f"Treasure: at ({treasure.position.x},{treasure.position.y}) tile=({treasure.tile.x},"
                     f"{treasure.tile.y})")

    lines.append(f"Entities ({len(state.enemies)} enemies, {len(state.items)} items, "
                 f"{len(state.projectiles)} projectiles):")
    for enemy in sorted(state.enemies, key=lambda e: e.index):
        flags = []
        if not enemy.is_active:
            flags.append('inactive')
        if enemy.is_dying:
            flags.append('dying')
        if enemy.is_stunned:
            flags.append(f'stun={enemy.stun_timer}')
        lines.append(f"  [{enemy.index:2}] enemy {_name(enemy.id)} pos=({enemy.position.x},{enemy.position.y}) "
                     f"tile=({enemy.tile.x},{enemy.tile.y}) hp={enemy.health} dir={_name(enemy.direction)} "
                     f"dist={enemy.distance:.0f} {' '.join(flags)}")
    for item in sorted(state.items, key=lambda e: e.index):
        lines.append(f"  [{item.index:2}] item {_name(item.id)} pos=({item.position.x},{item.position.y}) "
                     f"timer={item.timer} dist={item.distance:.0f}")
    for proj in sorted(state.projectiles, key=lambda e: e.index):
        lines.append(f"  [{proj.index:2}] projectile {_name(proj.id)} pos=({proj.position.x},{proj.position.y}) "
                     f"dir={_name(proj.direction)} dist={proj.distance:.0f}")

    if mask is not None:
        lines.append(f"Legal: {mask}")
    return '\n'.join(lines)


def render_grid(state) -> str:
    """ASCII map of the 32x22 tile grid.

    Legend: '.' walkable, '#' blocked, 'L' Link, 'E' enemy, 'i' item, '*' projectile, 'T' treasure,
    'K' locked door, '=' barred door, 'P' push block, 'C' cave entrance."""
    # pylint: disable=too-many-branches
    room = state.room
    cols, rows = room.tiles.shape
    grid = [['.' if room.is_tile_walkable(c, r) else '#' for c in range(cols)] for r in range(rows)]

    for c in range(cols):
        for r in range(rows):
            if state.level == 0 and int(room.tiles[c, r]) == 0xF3:
                grid[r][c] = 'C'
    for c, r in state.push_block_tiles:
        if 0 <= c < cols and 0 <= r < rows:
            grid[r][c] = 'P'

    if state.level != 0:
        from .room import NORTH_DOOR_TILE, SOUTH_DOOR_TILE, EAST_DOOR_TILE, WEST_DOOR_TILE  # pylint: disable=import-outside-toplevel
        for direction, (c, r) in ((Direction.N, NORTH_DOOR_TILE), (Direction.S, SOUTH_DOOR_TILE),
                                  (Direction.E, EAST_DOOR_TILE), (Direction.W, WEST_DOOR_TILE)):
            if state.is_door_locked(direction):
                grid[r][c] = 'K'
            elif state.is_door_barred(direction):
                grid[r][c] = '='

    def mark(obj, char, w=2, h=2):
        tile = obj.tile
        for dx in range(w):
            for dy in range(h):
                c, r = tile.x + dx, tile.y + dy
                if 0 <= c < cols and 0 <= r < rows:
                    grid[r][c] = char

    if state.treasure is not None:
        mark(state.treasure, 'T', 1, 1)
    for item in state.items:
        mark(item, 'i', 1, 1)
    for proj in state.projectiles:
        mark(proj, '*', 1, 1)
    for enemy in state.enemies:
        mark(enemy, 'E' if enemy.is_active else 'e')
    mark(state.link, 'L')

    header = '    ' + ''.join(f"{c % 16:x}" for c in range(cols))
    return '\n'.join([header] + [f"{r:3x} " + ''.join(row) for r, row in enumerate(grid)])


def describe_change(change) -> str:
    """One step's outcome: frames, interrupt, hits, items, health."""
    parts = [f"{change.action} -> {change.frames} frames"]
    if change.interrupt:
        parts.append(f"interrupt={change.interrupt}")
    if change.changed_location:
        parts.append(f"moved to level {change.state.level} room 0x{change.state.location:02x}")
    if change.enemies_hit:
        parts.append(f"hits={dict(change.enemies_hit)}")
    if change.enemies_stunned:
        parts.append(f"stunned={change.enemies_stunned}")
    if change.items_gained:
        parts.append(f"items={[_name(i) for i in change.items_gained]}")
    if change.health_lost:
        parts.append(f"health_lost={change.health_lost:g}")
    if change.health_gained:
        parts.append(f"health_gained={change.health_gained:g}")
    return '  '.join(parts)
