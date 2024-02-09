import heapq

from .zelda_game import is_tile_walkable

def heuristic(current, direction, map_width, map_height):
    y, x = current
    if direction == 'N':
        return y
    elif direction == 'S':
        return map_height - y - 1
    elif direction == 'W':
        return x
    elif direction == 'E':
        return map_width - x - 1
    else:
        ny, nx = direction
        return abs(nx - x) + abs(ny - y)

def get_neighbors(position, tiles):
    y, x = position
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

        prev_tile = 0
        if 0 <= x < tiles.shape[1] and 0 <= y < tiles.shape[0]:
            prev_tile = tiles[y, x]

        nx, ny = x + dx, y + dy
        if 0 <= nx < tiles.shape[1] and 0 <= ny < tiles.shape[0] and is_tile_walkable(prev_tile, tiles[ny, nx]):
            neighbors.append((ny, nx))
    return neighbors

def a_star(start, tiles, target):
    map_height, map_width = tiles.shape

    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, target, map_width, map_height)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if heuristic(current, target, map_width, map_height) == 0:
            # Reconstruct path
            path = []
            while current in came_from:
                if len(path) == 0 and isinstance(target, str):
                    add_direction(target, current, path)

                path.append(current)
                current = came_from[current]

            if not path:
                add_direction(target, start, path)
            path.append(start)
            return path[::-1]

        for neighbor in get_neighbors(current, tiles):
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, target, map_width, map_height)
                if neighbor not in [item[1] for item in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

def add_direction(target, current, path):
    if target == 'N':
        path.append((current[0] - 1, current[1]))
    elif target == 'S':
        path.append((current[0] + 1, current[1]))
    elif target == 'W':
        path.append((current[0], current[1] - 1))
    elif target == 'E':
        path.append((current[0], current[1] + 1))