import os

class RoomInformation:
    """
    Stores information about a room in the game Zelda-NES. This includes the level, location, exits, enemies, reward,
    and bomb secrets.
    """
    all_save_files = []
    def __init__(self, level, location, exits, enemies, reward = None, bomb_secrets = None):
        self.level = int(level)
        self.location = location
        self.exits = exits
        self.enemies = int(enemies) if enemies else 0
        self.reward = reward
        self.bomb_secrets = bomb_secrets

        self.save_states = []

        if not RoomInformation.all_save_files:
            data_dir = os.path.dirname(os.path.realpath(__file__))
            save_state_dir = os.path.join(data_dir, 'custom_integrations', 'Zelda-NES')
            RoomInformation.all_save_files = os.listdir(save_state_dir)

        self.save_states = [x for x in RoomInformation.all_save_files if x.startswith(f'{level}_{location}')]

class ZeldaGameData:
    """Information about the game Zelda-NES. This includes room data, memory addresses, and tables.
    Parsed from the file zelda_game_data.txt."""
    def __init__(self):
        data_dir = os.path.dirname(os.path.realpath(__file__))
        data_filename = os.path.join(data_dir, 'zelda_game_data.txt')

        self.rooms = {}
        self.memory = {}
        self.tables = {}

        with open(data_filename, 'r', encoding="utf-8") as f:
            is_room = False
            is_memory = False
            is_table = False

            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue

                if line == '[rooms]':
                    is_room = True
                    is_memory = False
                    is_table = False
                    continue

                if line == '[memory]':
                    is_room = False
                    is_memory = True
                    is_table = False
                    continue

                if line == '[tables]':
                    is_room = False
                    is_memory = False
                    is_table = True
                    continue

                if line.startswith('[') or line.endswith(']'):
                    is_room = False
                    is_memory = False
                    continue

                if line == '':
                    continue

                parts = [x for x in line.split(' ') if x]

                if is_room:
                    room = RoomInformation(*parts)
                    self.rooms[f'{room.level}_{room.location}'] = room

                elif is_memory:
                    name = parts[1]
                    address = int(parts[0], 16)
                    self.memory[name] = address

                elif is_table:
                    offset = int(parts[0], 16)
                    size = int(parts[1], 16)
                    name = parts[2]
                    self.tables[name] = (offset, size)

zelda_game_data = ZeldaGameData()
__all__ = ['zelda_game_data']
