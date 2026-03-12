
from typing import Sequence, Tuple
import heapq

from .zelda_enums import Direction
from .zelda_objects import ZeldaObject

# Maps (dx, dy) deltas to the Direction needed to move FROM the neighbor back
# toward the source tile. Used by wavefront: if expanding from A to neighbor B,
# the question is "can Link at B move toward A?" — the reverse direction.
_DELTA_TO_REVERSE_DIR = {
    (-1, 0): Direction.E,   # B is left of A → B moves right (E) to reach A
    (1, 0): Direction.W,    # B is right of A → B moves left (W) to reach A
    (0, -1): Direction.S,   # B is above A → B moves down (S) to reach A
    (0, 1): Direction.N,    # B is below A → B moves up (N) to reach A
}

class Wavefront:
    """A wavefront for a room."""
    _room_cache = {}
    def __init__(self, room : 'Room', # type: ignore
                 targets : Sequence[ZeldaObject | Direction | Tuple[int, int]],
                impassible : Sequence[Tuple[int, int] | ZeldaObject] = None):
        """Calculates the wavefront for the room for Link."""
        cols, rows = room.tiles.shape
        wavefront = {}
        todo = []

        for tile in targets:
            wavefront[tile] = 0
            heapq.heappush(todo, (0, tile))

        while todo:
            dist, tile = heapq.heappop(todo)

            # Off-screen seeds (placed by _get_wf_start for exit gradient) are
            # not real tiles, so skip the can_move check when expanding from them.
            tx, ty = tile
            source_offscreen = tx < 0 or ty < 0 or tx >= cols or ty >= rows

            for neighbor, direction in self._get_neighbors(room, tile):
                if neighbor in wavefront:
                    continue

                if neighbor in impassible:
                    continue

                # Check if Link can move FROM the neighbor TO the current tile.
                # The wavefront expands outward from targets, so the actual movement
                # direction is from the neighbor toward the target (the reverse direction).
                if not source_offscreen:
                    nx, ny = neighbor
                    if not room.can_move(nx, ny, direction):
                        continue

                wavefront[neighbor] = dist + 1
                heapq.heappush(todo, (dist + 1, neighbor))

        self._wavefront = wavefront
        self._targets = targets

    def __getitem__(self, tile):
        if tile not in self._wavefront:
            return 1000

        return self._wavefront[tile]

    def get(self, tile, default=None):
        """Get the distance to a tile, or a default value if the tile is not reachable."""
        return self._wavefront.get(tile, default)

    @staticmethod
    def _get_neighbors(room, tile):
        x, y = tile
        for (dx, dy), move_dir in _DELTA_TO_REVERSE_DIR.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < room.tiles.shape[0] and 0 <= ny < room.tiles.shape[1]:
                yield (nx, ny), move_dir
