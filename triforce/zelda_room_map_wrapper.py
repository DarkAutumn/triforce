import gymnasium as gym

from triforce.link import Link
from triforce.zelda_enums import ZeldaEnemyId

from .zelda_game_data import zelda_game_data
from .tile_states import TileState, is_room_loaded, tiles_to_weights
from .zelda_game_state import ZeldaGameState

class ZeldaRoomMapWrapper(gym.Wrapper):
    """Generates a tile map for the current room."""
    def __init__(self, env):
        super().__init__(env)
        self._room_maps = {}
        self._rooms_with_locks = set()
        self._rooms_with_locks.add((1, 0x35, False))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        for room in self._rooms_with_locks:
            self._room_maps.pop(room, None)

        self._create_tile_maps(self.env.unwrapped.get_ram(), info)
        return obs, info

    def step(self, action):
        obs, rewards, terminated, truncated, info = self.env.step(action)
        self._create_tile_maps(self.env.unwrapped.get_ram(), info)

        return obs, rewards, terminated, truncated, info

    def _create_tile_maps(self, ram, info):
        if 'state_change' in info:
            change = info['state_change']
            prev = change.previous
            curr = change.current
        else:
            prev = curr = info['state']
        result = self._get_tile_maps(ram, prev, curr)
        info['tiles'], info['tile_states'], info['link_warning_tiles'], info['link_danger_tiles'] = result

    def _get_tile_maps(self, ram, prev : ZeldaGameState, curr : ZeldaGameState):
        tiles = self._get_tiles(ram, prev, curr)
        tile_states = ZeldaRoomMapWrapper._get_tile_states(tiles, curr.enemies, curr.projectiles)
        # calculate how many squares link overlaps with dangerous tiles
        warning_tiles, danger_tiles = self._count_danger_tile_overlaps(curr.link, tile_states)

        north_locked = tiles[2, 16] == 0x9a
        if north_locked:
            self._rooms_with_locks.add(curr.full_location)

        return tiles, tile_states, warning_tiles, danger_tiles

    def _get_tiles(self, ram, prev : ZeldaGameState, curr : ZeldaGameState):
        index = curr.full_location

        # check if we spent a key, if so the tile layout of the room changed
        if curr.link.keys < prev.link.keys:
            self._room_maps.pop(index, None)

        if len(prev.enemies) != len(curr.enemies):
            self._room_maps.pop(index, None)

        if index not in self._room_maps:
            map_offset, map_len = zelda_game_data.tables['tile_layout']
            tiles = ram[map_offset:map_offset+map_len]
            tiles = tiles.reshape((32, 22)).T

            if is_room_loaded(tiles):
                self._room_maps[index] = tiles
        else:
            tiles = self._room_maps[index]

        return tiles

    def _count_danger_tile_overlaps(self, link : Link, tile_states):
        warning_tiles = 0
        danger_tiles = 0
        for pos in link.tile_coordinates:
            y, x = pos
            if 0 <= y < tile_states.shape[0] and 0 <= x < tile_states.shape[1]:
                state = tile_states[y, x]
                if state == TileState.WARNING.value:
                    warning_tiles += 1
                elif state == TileState.DANGER.value:
                    danger_tiles += 1

        return warning_tiles, danger_tiles

    @staticmethod
    def _get_tile_states(tiles, enemies, projectiles):
        tiles = tiles.copy()
        tiles_to_weights(tiles)
        saw_wallmaster = False
        for obj in enemies:
            if obj.is_active:
                ZeldaRoomMapWrapper._add_enemy_or_projectile(tiles, obj.tile_coordinates)

            if obj.id == ZeldaEnemyId.WallMaster and not saw_wallmaster:
                saw_wallmaster = True
                ZeldaRoomMapWrapper._add_wallmaster_tiles(tiles)

        for proj in projectiles:
            ZeldaRoomMapWrapper._add_enemy_or_projectile(tiles, proj.tile_coordinates)

        return tiles

    @staticmethod
    def _add_wallmaster_tiles(result):
        x = 4
        while x < 28:
            result[4, x] = TileState.WARNING.value
            result[17, x] = TileState.WARNING.value
            x += 1

        y = 4
        while y < 18:
            result[(y, 4)] = TileState.WARNING.value
            result[(y, 27)] = TileState.WARNING.value
            y += 1

    @staticmethod
    def _add_enemy_or_projectile(tiles, coords):
        min_y = min(coord[0] for coord in coords)
        max_y = max(coord[0] for coord in coords)
        min_x = min(coord[1] for coord in coords)
        max_x = max(coord[1] for coord in coords)

        for coord in coords:
            if 0 <= coord[0] < tiles.shape[0] and 0 <= coord[1] < tiles.shape[1]:
                tiles[coord] = TileState.DANGER.value

        for ny in range(min_y - 1, max_y + 2):
            for nx in range(min_x - 1, max_x + 2):
                if 0 <= ny < tiles.shape[0] and 0 <= nx < tiles.shape[1]:
                    if tiles[ny, nx] == TileState.WALKABLE.value:
                        tiles[ny, nx] = TileState.WARNING.value
