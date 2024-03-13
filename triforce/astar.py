"""A* algorithm implementation for pathfinding.  Used to train the model how to move around the map."""

import heapq
from .zelda_game import TileState, Direction

def get_tile_weight(ny, nx, tile_weight_map):
    """Returns the weight of the tile at the given position."""
    if ny < 0 or nx < 0:
        return None

    if ny >= tile_weight_map.shape[0] or nx >= tile_weight_map.shape[1]:
        return None

    return tile_weight_map[ny, nx]

def heuristic(current, direction : Direction, tile_state_map):
    """Calculate the value of a direction or step."""
    # Use the bottom left corner of the tile for the heuristic, this avoids miscounting impassible tiles since
    # the link can stand with his top half inside of an impassible tile.
    y, x = current
    tile_weight = get_tile_weight(y, x, tile_state_map)
    tile_weight = tile_weight if tile_weight is not None else 1

    map_height, map_width = tile_state_map.shape

    match direction:
        case Direction.N:
            return y + tile_weight
        case Direction.S:
            return map_height - y - 1 + tile_weight
        case Direction.W:
            return x + tile_weight
        case Direction.E:
            return map_width - x - 1 + tile_weight
        case _:
            ny, nx = direction
            dist = mahattan_distance(y, x, ny, nx)
            if dist:
                return dist + tile_weight

            return 0

def mahattan_distance(y, x, ny, nx):
    """Returns the manhattan distance between two points"""
    return abs(nx - x) + abs(ny - y)


WALKABLE_TILES = [TileState.WALKABLE.value,
                           TileState.DANGER.value,
                           TileState.WARNING.value,
                           TileState.BRICK.value
                           ]

def is_valid_tile(coords, tile_weight_map):
    """Returns True if the move is valid, False otherwise."""

    # coords is link's top left tile (link is a 2x2 sprite).  The game only collides with the bottom two tiles.
    tile0 = get_tile_weight(coords[0], coords[1], tile_weight_map)
    tile1 = get_tile_weight(coords[0], coords[1] + 1, tile_weight_map)

    return all([tile0 in WALKABLE_TILES, tile1 in WALKABLE_TILES])

def get_neighbors(position, tile_weight_map):
    """Returns neighbors of position that are both valid and walkable."""
    y, x = position
    potential_neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
    return [(ny, nx) for ny, nx in potential_neighbors if is_valid_tile((ny, nx), tile_weight_map)]

def reconstruct_path(start, came_from, current, target):
    """Reconstructs the path from the start node to the current node using the came_from dictionary."""
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]

    path.reverse()
    append_final_direction(path, target, start)
    return path

def get_tile_from_direction(tile, direction):
    """Returns the next tile in the given direction from the current tile."""
    y, x = tile
    match direction:
        case Direction.N:
            return y - 1, x
        case Direction.S:
            return y + 1, x
        case Direction.W:
            return y, x - 1
        case Direction.E:
            return y, x + 1

    return tile

def append_final_direction(path, target, start):
    """Add an additional location off of the map if asked to walk towards a cardinal direction."""
    if not isinstance(target, Direction):
        return

    last = path[-1] if path else get_tile_from_direction(start, target)
    path.append(get_tile_from_direction(last, target))

def a_star(link_bottom_left_tile, tile_weight_map, direction):
    """
    A* algorithm implementation for pathfinding.

    Args:
        link_position (tuple): The starting position of the link.
        tiles (numpy.ndarray): The map tile weights, see TileState.
        direction (Direction): The direction in which the link is moving.

    Returns:
        list: The path from the starting position to the goal position.
    """

    # quick out for touching the map's edge
    if direction == Direction.W and link_bottom_left_tile[1] <= 0:
        return [get_tile_from_direction(link_bottom_left_tile, direction)]

    if direction == Direction.E and link_bottom_left_tile[1] >= tile_weight_map.shape[1] - 1:
        return [get_tile_from_direction(link_bottom_left_tile, direction)]

    if direction == Direction.N and link_bottom_left_tile[0] <= 0:
        return [get_tile_from_direction(link_bottom_left_tile, direction)]

    if direction == Direction.S and link_bottom_left_tile[0] >= tile_weight_map.shape[0] - 1:
        return [get_tile_from_direction(link_bottom_left_tile, direction)]

    came_from = {}
    open_set = []
    g_score = {}
    f_score = {}
    closest_node = None
    closest_distance = float('inf')

    start = link_bottom_left_tile
    if 0 <= start[0] < tile_weight_map.shape[0] and 0 <= start[1] < tile_weight_map.shape[1]:
        heapq.heappush(open_set, (0, start))

        current_distance = heuristic(start, direction, tile_weight_map)
        g_score[start] = 0
        f_score[start] = current_distance

        if current_distance < closest_distance:
            closest_node = start

    while open_set:
        _, current = heapq.heappop(open_set)
        current_distance = heuristic(current, direction, tile_weight_map)
        if current_distance < closest_distance:
            closest_node = current
            closest_distance = current_distance

        if current_distance == 0:
            return reconstruct_path(start, came_from, current, direction)

        for neighbor in get_neighbors(current, tile_weight_map):
            tentative_g_score = g_score[current] + tile_weight_map[current]
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                h_value = heuristic(neighbor, direction, tile_weight_map)
                f_score[neighbor] = tentative_g_score + h_value
                if neighbor not in [item[1] for item in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return reconstruct_path(start, came_from, closest_node, direction)
