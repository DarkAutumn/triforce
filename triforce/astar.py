"""A* algorithm implementation for pathfinding.  Used to train the model how to move around the map."""

import heapq
from .zelda_game import TileState, Direction

def heuristic(current, direction : Direction, dimensions, tile_weight : int):
    """
    Calculate the heuristic value for the A* algorithm.

    Parameters:
    - current: Tuple representing the current position (y, x).
    - direction: A Direction representing the direction to move or a tuple representing the next position (ny, nx).
    - dimensions: Tuple representing the dimensions of the map (height, width).
    - tile_weight: The weight of the current tile.

    Returns:
    - The heuristic value for the given parameters.
    """
    y, x = current
    map_height, map_width = dimensions

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

def get_neighbors(position, tile_weight_map):
    """Returns neighbors of position that are both valid and walkable."""
    dimensions = tile_weight_map.shape
    y, x = position
    potential_neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
    return [(ny, nx) for ny, nx in potential_neighbors
            if 0 <= nx < dimensions[1] and 0 <= ny < dimensions[0] and tile_weight_map[ny, nx] in WALKABLE_TILES]

def reconstruct_path(start_tiles, came_from, current, target, tile_state_map):
    """Reconstructs the path from the start node to the current node using the came_from dictionary."""
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]

    path.reverse()
    append_final_direction(path, target, start_tiles, tile_state_map)
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

def append_final_direction(path, target, start_tiles, tile_state_map):
    """Add an additional location off of the map if asked to walk towards a cardinal direction."""
    if not isinstance(target, Direction):
        return

    if path:
        last = path[-1]
    else:
        last = None
        for location in start_tiles:
            next_tile = get_tile_from_direction(location, target)
            if next_tile in path or next_tile in start_tiles:
                continue

            if last is None:
                last = next_tile
            else:
                row, col = next_tile
                max_row, max_col = tile_state_map.shape
                if row < 0 or row >= max_row or col < 0 or col >= max_col \
                                                        or tile_state_map[next_tile] in WALKABLE_TILES:
                    last = next_tile
                    break

    if last:
        path.append(get_tile_from_direction(last, target))

def a_star(start_tiles, tile_weight_map, map_dimensions, direction):
    """
    A* algorithm implementation for pathfinding.

    Args:
        link_position (tuple): The starting position of the link.
        tiles (numpy.ndarray): The map tiles.
        direction (Direction): The direction in which the link is moving.

    Returns:
        list: The path from the starting position to the goal position.
    """
    # pylint: disable=too-many-locals
    came_from = {}
    open_set = []
    g_score = {}
    f_score = {}
    closest_node = None
    closest_distance = float('inf')

    for start in start_tiles:
        heapq.heappush(open_set, (0, start))

        current_distance = heuristic(start, direction, map_dimensions, tile_weight_map[start])
        g_score[start] = 0
        f_score[start] = current_distance

        if current_distance < closest_distance:
            closest_node = start

    while open_set:
        _, current = heapq.heappop(open_set)
        current_distance = heuristic(current, direction, map_dimensions, tile_weight_map[current])
        if current_distance < closest_distance:
            closest_node = current
            closest_distance = current_distance

        if current_distance == 0:
            return reconstruct_path(start_tiles, came_from, current, direction, tile_weight_map)

        for neighbor in get_neighbors(current, tile_weight_map):
            tentative_g_score = g_score[current] + tile_weight_map[current]
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                h_value = heuristic(neighbor, direction, map_dimensions, tile_weight_map[neighbor])
                f_score[neighbor] = tentative_g_score + h_value
                if neighbor not in [item[1] for item in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return reconstruct_path(start_tiles, came_from, closest_node, direction, tile_weight_map)
