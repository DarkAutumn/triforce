import os

class ZeldaRoom:
    def __init__(self, save_state_dir, level, location, exits, enemies, reward = None, bomb_secrets = None):
        self.level = level
        self.location = location
        self.exits = exits
        self.enemies = enemies
        self.reward = reward
        self.bomb_secrets = bomb_secrets
        self.save_states = [""]

class ZeldaGameData:
    def __init__(self):
        data_dir = os.path.dirname(os.path.realpath(__file__))
        data_filename = os.path.join(data_dir, 'zelda_game_data.txt')
        save_state_dir = os.path.join(data_dir, 'custom_integrations', 'Zelda-NES')

        self.rooms = []
        self.memory = {}

        with open(data_filename, 'r') as f:
            is_room = False
            is_memory = False

            for line in f:
                line = line.strip()
                if line == '[rooms]':
                    is_room = True
                    is_memory = False
                    continue

                elif line == '[memory]':
                    is_room = False
                    is_memory = True
                    continue

                elif line.startswith('[') or line.endswith(']'):
                    is_room = False
                    is_memory = False
                    continue

                elif line == '':
                    continue

                parts = line.split(' ')

                if is_room:
                    room = ZeldaRoom(save_state_dir, *parts)
                    self.rooms.append(room)

                elif is_memory:
                    name = parts[1]
                    address = int(parts[0], 16)
                    self.memory[name] = address
    

