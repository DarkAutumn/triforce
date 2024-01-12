import os

class ZeldaRoom:
    all_save_files = []
    def __init__(self, level, location, exits, enemies, reward = None, bomb_secrets = None):
        self.level = int(level)
        self.location = location
        self.exits = exits
        self.enemies = enemies
        self.reward = reward
        self.bomb_secrets = bomb_secrets
        
        self.save_states = []

        if not ZeldaRoom.all_save_files:
            data_dir = os.path.dirname(os.path.realpath(__file__))
            save_state_dir = os.path.join(data_dir, 'custom_integrations', 'Zelda-NES')
            ZeldaRoom.all_save_files = os.listdir(save_state_dir)

        self.save_states = [x for x in ZeldaRoom.all_save_files if x.startswith(f'{level}_{location}')]

class ZeldaGameData:
    def __init__(self):
        data_dir = os.path.dirname(os.path.realpath(__file__))
        data_filename = os.path.join(data_dir, 'zelda_game_data.txt')

        self.rooms = {}
        self.memory = {}

        with open(data_filename, 'r') as f:
            is_room = False
            is_memory = False

            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue

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

                parts = [x for x in line.split(' ') if x]

                if is_room:
                    room = ZeldaRoom(*parts)
                    self.rooms[f'{room.level}_{room.location}'] = room

                elif is_memory:
                    name = parts[1]
                    address = int(parts[0], 16)
                    self.memory[name] = address

    def get_room_by_savestate(self, state):
        for room in self.rooms:
            for savestate in room.save_states:
                if savestate.startswith(state):
                    return room
                
        return None
    
    def get_savestates_by_name(self, room_name):
        full_location = room_name
        room_name = self.strip_direction(room_name)
        
        room = self.rooms.get(room_name, None)
        if room is None:
            level, loc = room_name.split('_')
            room = ZeldaRoom(level, loc, None, None, None, None)
        
        return [x for x in room.save_states if x.startswith(full_location)]

    def strip_direction(self, room):
        if room[-1] not in '0123456789':
            room = room[:-1]
        return room

zelda_game_data = ZeldaGameData()
__all__ = ['zelda_game_data']