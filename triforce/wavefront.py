
from typing import Sequence, Tuple
import heapq

from .zelda_enums import Direction
from .zelda_objects import ZeldaObject

class Wavefront:
    """A wavefront for a room."""
    _room_cache = {}
    def __init__(self, room : 'Room', # type: ignore
                 targets : Sequence[ZeldaObject | Direction | Tuple[int, int]],
                impassible : Sequence[Tuple[int, int] | ZeldaObject] = None):
        """Calculates the wavefront for the room for Link."""
        wavefront = {}
        todo = []

        for tile in targets:
            wavefront[tile] = 0
            heapq.heappush(todo, (0, tile))

        while todo:
            dist, tile = heapq.heappop(todo)
            for neighbor in self._get_neighbors(room, tile):
                if neighbor in wavefront:
                    continue

                if neighbor in impassible:
                    continue

                if not room.walkable[neighbor]:
                    continue

                wavefront[neighbor] = dist + 1
                heapq.heappush(todo, (dist + 1, neighbor))

        # fill in the remaineder of the room
        for tile, dist in wavefront.items():
            heapq.heappush(todo, (-dist, tile))

        while todo:
            dist, tile = heapq.heappop(todo)
            dist = -dist
            for neighbor in self._get_neighbors(room, tile):
                if neighbor in wavefront:
                    continue

                wavefront[neighbor] = dist + 1
                heapq.heappush(todo, (-dist - 1, neighbor))

        self._wavefront = wavefront
        self._targets = targets

    def __getitem__(self, tile):
        if tile not in self._wavefront:
            return 1000

        return self._wavefront[tile]

    def get(self, tile, default=None):
        """Get the distance to a tile, or a default value if the tile is not reachable."""
        return self._wavefront.get(tile, default)

    def _get_neighbors(self, room, tile):
        x, y = tile
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if -1 <= nx < room.tiles.shape[0] and -1 <= ny < room.tiles.shape[1]:
                yield nx, ny
