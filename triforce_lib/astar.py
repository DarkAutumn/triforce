import heapq

from .zelda_game import walkable_tiles

def heuristic(current, direction, map_width, map_height, is_exterior_dangerous):
    """
    Calculate the heuristic value for the A* algorithm.

    Parameters:
    - current: Tuple representing the current position (y, x).
    - direction: String representing the direction to move ('N', 'S', 'W', 'E') or a tuple
                 representing the next position (ny, nx).
    - map_width: Integer representing the width of the map.
    - map_height: Integer representing the height of the map.
    - is_exterior_dangerous: Boolean indicating if the exterior is dangerous.

    Returns:
    - The heuristic value for the given parameters.
    """
    y, x = current

    weight = 0
    if is_exterior_dangerous:
        if x in [0x04, 0x05, 0x1a, 0x1b] or y in [0x04, 0x05, 0x10, 0x11]:
            weight = 10

    if direction == 'N':
        return y + weight
    elif direction == 'S':
        return map_height - y - 1 + weight
    elif direction == 'W':
        return x + weight
    elif direction == 'E':
        return map_width - x - 1 + weight
    else:
        ny, nx = direction
        return abs(nx - x) + abs(ny - y) + weight

def get_tile(position, tiles):
    """
    Retrieves the value of a tile at the given position from the tiles array.

    Args:
        position (tuple): The position of the tile in the form (y, x).
        tiles (numpy.ndarray): The array of tiles.

    Returns:
        int: The value of the tile at the given position, or 0 if the position is out of bounds.
    """
    y, x = position
    if 0 <= x < tiles.shape[1] and 0 <= y < tiles.shape[0]:
        return tiles[y, x]

    return 0

# Special case dungeon bricks.  Link actually walks through them so they are walkable, but only if
# coming from a non-brick tile.  Otherwise the A* algorithm will try to route link around the bricks
# outside the play area.
BRICK_TILE = 0xf6
def get_neighbors(position, tiles):
    """
    Returns a list of neighboring positions that are valid for movement.

    Args:
        position (tuple): The position of the tile to get the neighbors of.
        tiles (list): The grid of tiles.

    Returns:
        list: A list of neighboring positions that are valid for movement.
    """
    prev_tile_is_brick = get_tile(position, tiles) == BRICK_TILE

    y, x = position
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        next_position = y + dy, x + dx
        tile = get_tile(next_position, tiles)

        # If the previous tile is a brick, we can't move to another brick tile
        if prev_tile_is_brick and tile == BRICK_TILE:
            continue

        if walkable_tiles[tile]:
            neighbors.append(next_position)

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

def a_star(link_position, tiles, direction, is_exterior_dangerous):
    """
    A* algorithm implementation for pathfinding.

    Args:
        link_position (tuple): The starting position of the link.
        tiles (numpy.ndarray): The map tiles.
        direction (str): The direction in which the link is moving.
        is_exterior_dangerous (bool): Flag indicating if the exterior is dangerous.

    Returns:
        list: The path from the starting position to the goal position.
    """
    # pylint: disable-msg=too-many-locals
    map_height, map_width = tiles.shape
    start = link_position

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, direction, map_width, map_height, is_exterior_dangerous)}

    closest_node = start
    closest_distance = heuristic(start, direction, map_width, map_height, is_exterior_dangerous)

    while open_set:
        _, current = heapq.heappop(open_set)

        current_distance = heuristic(current, direction, map_width, map_height, is_exterior_dangerous)
        if current_distance < closest_distance:
            closest_node = current
            closest_distance = current_distance

        if current_distance == 0:
            return reconstruct_path(start, came_from, current)

        for neighbor in get_neighbors(current, tiles):
            tentative_g_score = g_score[current] + 1  # Assuming uniform cost
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                h_value = heuristic(neighbor, direction, map_width, map_height, is_exterior_dangerous)
                f_score[neighbor] = tentative_g_score + h_value
                if neighbor not in [item[1] for item in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return reconstruct_path(start, came_from, closest_node)

def add_direction(target, current, path):
    """
    Adds a new coordinate to the given path based on the target direction.

    Args:
        target (str): The target direction ('N', 'S', 'W', or 'E').
        current (tuple): The current coordinate (x, y).
        path (list): The list of coordinates representing the path.

    Returns:
        None
    """
    if target == 'N':
        path.append((current[0] - 1, current[1]))
    elif target == 'S':
        path.append((current[0] + 1, current[1]))
    elif target == 'W':
        path.append((current[0], current[1] - 1))
    elif target == 'E':
        path.append((current[0], current[1] + 1))

__all__ = ['a_star']
