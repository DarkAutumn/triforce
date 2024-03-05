"""A* algorithm implementation for pathfinding.  Used to train the model how to move around the map."""

import heapq

from .zelda_game import TileState, Direction

def heuristic(current, direction : Direction, dimensions, tile : TileState):
    """
    Calculate the heuristic value for the A* algorithm.

    Parameters:
    - current: Tuple representing the current position (y, x).
    - direction: A Direction representing the direction to move or a tuple representing the next position (ny, nx).
    - map_width: Integer representing the width of the map.
    - map_height: Integer representing the height of the map.
    - is_exterior_dangerous: Boolean indicating if the exterior is dangerous.

    Returns:
    - The heuristic value for the given parameters.
    """
    y, x = current
    map_height, map_width = dimensions

    weight = tile.astar_weight
    if direction == Direction.N:
        return y + weight
    if direction == Direction.S:
        return map_height - y - 1 + weight
    if direction == Direction.W:
        return x + weight
    if direction == Direction.E:
        return map_width - x - 1 + weight

    ny, nx = direction
    return abs(nx - x) + abs(ny - y) + weight

# Special case dungeon bricks.  Link actually walks through them so they are walkable, but only if
# coming from a non-brick tile.  Otherwise the A* algorithm will try to route link around the bricks
# outside the play area.
def get_neighbors(position, tile_states, dimensions):
    """
    Returns a list of neighboring positions that are valid for movement.

    Args:
        position (tuple): The position of the tile to get the neighbors of.
        tiles (list): The grid of tiles.

    Returns:
        list: A list of neighboring positions that are valid for movement.
    """
    y, x = position
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        next_position = y + dy, x + dx
        if 0 <= x + dx < dimensions[1] and 0 <= y + dy < dimensions[0]:
            next_position = y + dy, x + dx
            tile = tile_states.get(next_position, TileState.IMPASSABLE)
            if tile.is_walkable:
                neighbors.append((next_position, tile))

    return neighbors

def reconstruct_path(start, came_from, current):
    """
    Reconstructs the path from the start node to the current node using the came_from dictionary.

    Args:
        start: The start node of the path.
        came_from: A dictionary that maps each node to its previous node in the path.
        current: The current node of the path.

    Returns:
        A list representing the reconstructed path from the start node to the current node.
    """
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]

    path.append(start)
    return path[::-1]

def a_star(link_position, tile_state_map, map_dimensions, direction):
    """
    A* algorithm implementation for pathfinding.

    Args:
        link_position (tuple): The starting position of the link.
        tiles (numpy.ndarray): The map tiles.
        direction (Direction): The direction in which the link is moving.

    Returns:
        list: The path from the starting position to the goal position.
    """
    # pylint: disable-msg=too-many-locals
    start = link_position
    start_tile_state = tile_state_map.get(start, TileState.IMPASSABLE)

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, direction, map_dimensions, start_tile_state)}

    closest_node = start
    closest_distance = heuristic(start, direction, map_dimensions, start_tile_state)

    while open_set:
        _, current = heapq.heappop(open_set)
        current_tile_state = tile_state_map.get(current, TileState.IMPASSABLE)

        current_distance = heuristic(current, direction, map_dimensions, current_tile_state)
        if current_distance < closest_distance:
            closest_node = current
            closest_distance = current_distance

        if current_distance == 0:
            return reconstruct_path(start, came_from, current)

        for neighbor, tile in get_neighbors(current, tile_state_map, map_dimensions):
            tentative_g_score = g_score[current] + tile.astar_weight
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                h_value = heuristic(neighbor, direction, map_dimensions, tile)
                f_score[neighbor] = tentative_g_score + h_value
                if neighbor not in [item[1] for item in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return reconstruct_path(start, came_from, closest_node)

def add_direction(target, current, path):
    """
    Adds a new coordinate to the given path based on the target direction.

    Args:
        target (Direction): The target direction.
        current (tuple): The current coordinate (x, y).
        path (list): The list of coordinates representing the path.

    Returns:
        None
    """
    if target == Direction.N:
        path.append((current[0] - 1, current[1]))
    elif target == Direction.S:
        path.append((current[0] + 1, current[1]))
    elif target == Direction.W:
        path.append((current[0], current[1] - 1))
    elif target == Direction.E:
        path.append((current[0], current[1] + 1))
    else:
        raise ValueError(f"Unknown direction: {target}")

__all__ = ['a_star']
