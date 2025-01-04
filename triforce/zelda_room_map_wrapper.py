import gymnasium as gym

from .zelda_game_data import zelda_game_data
from .zelda_game import TileState, ZeldaEnemy, is_in_cave, is_room_loaded, tiles_to_weights

class ZeldaRoomMapWrapper(gym.Wrapper):
    """Generates a tile map for the current room."""
    def __init__(self, env):
        super().__init__(env)
        self._room_maps = {}
        self._rooms_with_locks = set()
        self._rooms_with_locks.add((1, 0x35, False))
        self._last_info = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        for room in self._rooms_with_locks:
            self._room_maps.pop(room, None)

        self._create_tile_maps(info, self.env.unwrapped.get_ram(), info['link'])
        self._last_info = info
        return obs, info

    def step(self, action):
        obs, rewards, terminated, truncated, info = self.env.step(action)
        self._create_tile_maps(info, self.env.unwrapped.get_ram(), info['link'])

        self._last_info = info
        return obs, rewards, terminated, truncated, info

    def _get_full_location(self, info):
        return (info['level'], info['location'], is_in_cave(info))

    def _create_tile_maps(self, info, ram, link):
        tiles = self._get_tiles(info, ram)
        tile_states = ZeldaRoomMapWrapper._get_tile_states(tiles, info['enemies'], info['projectiles'])
        info['tiles'] = tiles
        info['tile_states'] = tile_states

        # calculate how many squares link overlaps with dangerous tiles
        warning_tiles, danger_tiles = self._count_danger_tile_overlaps(link, tile_states)
        info['link_warning_tiles'] = warning_tiles
        info['link_danger_tiles'] = danger_tiles

        north_locked = tiles[2, 16] == 0x9a
        if north_locked:
            self._rooms_with_locks.add(self._get_full_location(info))

    def _get_tiles(self, info, ram):
        index = self._get_full_location(info)

        # check if we spent a key, if so the tile layout of the room changed
        if self._last_info:
            curr_keys = info['keys']
            last_keys = self._last_info.get('keys', curr_keys)
            if curr_keys < last_keys:
                self._room_maps.pop(index, None)

            if len(self._last_info['enemies']) != len(info['enemies']):
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

    def _count_danger_tile_overlaps(self, link, tile_states):
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

            if obj.id == ZeldaEnemy.WallMaster and not saw_wallmaster:
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
