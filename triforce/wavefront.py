"""Wavefront expansion of objectives for rewards."""

from typing import OrderedDict
import numpy as np

from .zelda_game import WALKABLE_TILES, BRICK_TILE, Direction, ZeldaItem, is_health_full, position_to_tile_index
from .objective_selector import ObjectiveKind, Dungeon1Orchestrator, OverworldOrchestrator

def is_walkable(tile):
    """Returns True if the tile is walkable, False otherwise."""
    return tile == BRICK_TILE or tile in WALKABLE_TILES

ORCHESTRATORS = {
    0 : OverworldOrchestrator(),
    1 : Dungeon1Orchestrator()
}

class Wave:
    """A set of targets and values for this wavefront."""
    def __init__(self, targets, values):
        self.targets = targets
        self.values = values

class RoomWavefront:
    """The wavefront for a single room."""
    def __init__(self, room_id, tiles):
        self.cache_len = 256
        level, _ = room_id
        self.room_id = room_id
        self.orchestrator = ORCHESTRATORS[level]
        self.tiles = tiles
        self._wavefront_lru = OrderedDict()

    def get_wavefront(self, info):
        """Get the wavefront for the room."""
        link_pos =  np.array(info['link_pos'], dtype=np.float32)
        location_objective, objective_vector, pos_dir, kind = self.orchestrator.get_objectives(info, link_pos)
        info['location_objective'] = location_objective
        info['objective_vector'] = objective_vector
        info['position_or_direction'] = pos_dir

        targets = []
        if kind == ObjectiveKind.FIGHT:
            targets.extend(x.tile_coordinates for x in info['active_enemies'])

        if isinstance(pos_dir, Direction):
            if pos_dir in (Direction.S, Direction.N):
                y = self.tiles.shape[0] - 1 if pos_dir == Direction.S else 0
                for x in range(self.tiles.shape[1]):
                    if is_walkable(self.tiles[y, x]):
                        targets.append((y, x))

            else:
                x = self.tiles.shape[1] - 1 if pos_dir == Direction.E else 0
                for y in range(self.tiles.shape[0]):
                    if is_walkable(self.tiles[y, x]):
                        targets.append((y, x))
        else:
            targets.append(position_to_tile_index(*pos_dir))

        ignore_hearts = is_health_full(info)
        ignore_bombs = info['bombs'] == info['bomb_max']
        for item in info['items']:
            if ignore_hearts and item.id in (ZeldaItem.Heart, ZeldaItem.Fairy):
                continue

            if ignore_bombs and item.id == ZeldaItem.Bombs:
                continue

            targets.append(item.tile_coordinates)

        targets = tuple(sorted(set(targets)))
        assert targets
        if (result := self._get_lru(targets)) is not None:
            return result

        result = self._calculate_wavefront(targets)
        self._put_lru(targets, result)
        return result

    def _calculate_wavefront(self, targets):
        """Calculate the wavefront for the room."""
        maxint = np.iinfo(np.int32).max
        wavefront = np.full(self.tiles.shape, maxint, dtype=np.int32)

        curr_value = 0
        curr_tiles = targets
        while curr_tiles:
            next_tiles = []
            for curr in curr_tiles:
                tile = self.tiles[curr[0], curr[1]]
                value = wavefront[curr[0], curr[1]]
                if value == maxint and is_walkable(tile):
                    wavefront[curr[0], curr[1]] = curr_value

                for x in self._get_directions(curr, wavefront):
                    if x not in next_tiles:
                        next_tiles.append(x)

            curr_tiles = next_tiles
            curr_value += 1

        return Wave(targets, wavefront)

    def _get_directions(self, coords, wavefront):
        possible_directions = [(coords[0] - 1, coords[1]), (coords[0] + 1, coords[1]),
                               (coords[0], coords[1] - 1), (coords[0], coords[1] + 1)]

        intmax = np.iinfo(np.int32).max
        for y, x in possible_directions:
            if 0 <= y < self.tiles.shape[0] and 0 <= x < self.tiles.shape[1] and wavefront[y, x] == intmax:
                tile = self.tiles[y, x]
                if is_walkable(tile):
                    yield (y, x)

    def _get_lru(self, targets):
        result = self._wavefront_lru.get(targets, None)
        if result is not None:
            self._wavefront_lru.move_to_end(targets)

        return result

    def _put_lru(self, targets, wavefront):
        if len(self._wavefront_lru) > self.cache_len:
            self._wavefront_lru.popitem(last=False)

        self._wavefront_lru[targets] = wavefront
