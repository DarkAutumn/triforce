import heapq

from .zelda_game import walkable_tiles

def heuristic(current, direction, map_width, map_height, is_exterior_dangerous):
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
    y, x = position
    if 0 <= x < tiles.shape[1] and 0 <= y < tiles.shape[0]:
        return tiles[y, x]

    return 0

# Special case dungeon bricks.  Link actually walks through them so they are walkable, but only if
# coming from a non-brick tile.  Otherwise the A* algorithm will try to route link around the bricks
# outside the play area.
brick_tile = 0xf6
def get_neighbors(position, tiles):
    prev_tile_is_brick = get_tile(position, tiles) == brick_tile

    y, x = position
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:


        next = y + dy, x + dx
        tile = get_tile(next, tiles)

        # If the previous tile is a brick, we can't move to another brick tile
        if prev_tile_is_brick and tile == brick_tile:
            continue

        if walkable_tiles[tile]:
            neighbors.append(next)

    return neighbors

def reconstruct_path(start, came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]

    path.append(start)
    return path[::-1]

def a_star(link_position, tiles, direction, is_exterior_dangerous):
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
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, direction, map_width, map_height, is_exterior_dangerous)
                if neighbor not in [item[1] for item in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return reconstruct_path(start, came_from, closest_node)

def add_direction(target, current, path):
    if target == 'N':
        path.append((current[0] - 1, current[1]))
    elif target == 'S':
        path.append((current[0] + 1, current[1]))
    elif target == 'W':
        path.append((current[0], current[1] - 1))
    elif target == 'E':
        path.append((current[0], current[1] + 1))
