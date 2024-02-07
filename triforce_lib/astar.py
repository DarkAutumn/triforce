import heapq

from .zelda_game import is_tile_walkable

def heuristic(current, direction, map_width, map_height):
    x, y = current
    if direction == 'N':
        return y
    elif direction == 'S':
        return map_height - y - 1
    elif direction == 'W':
        return map_width - x - 1
    elif direction == 'E':
        return x
    else:
        nx, ny = direction
        return abs(nx - x) + abs(ny - y)

def get_neighbors(position, tiles):
    x, y = position
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= ny < tiles.shape[1] and 0 <= nx < tiles.shape[0] and is_tile_walkable(tiles[nx, ny]):
            neighbors.append((nx, ny))
    return neighbors

def a_star(link_position, tiles, target):
    map_height, map_width = tiles.shape
    start = link_position

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
                path.append(current)
                current = came_from[current]
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