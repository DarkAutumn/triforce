
class ZeldaGameModes:
    """An enum of game states"""
    def __init__(self):
        self.titleTransition = 0
        self.selectionScreen = 1
        self.completed_scrolling = 4
        self.gameplay = 5
        self.prepare_scrolling = 6
        self.scrolling = 7
        self.registration = 0xe
        self.game_over = 0xf

zelda_game_modes = ZeldaGameModes()

class zelda_memory_layout:
    """Raw memory addresses in the game and what they map to"""
    def __init__(self):
        self.locations = [
            ( 0x10, "level"),
            ( 0xeb, "location"),
            ( 0x12, "mode"),

            ( 0xb9, "sword_animation"),
            ( 0xba, "beam_animation"),
            ( 0xbc, "bomb_or_flame_animation"),
            ( 0xbd, "bomb_or_flame_animation2"),
            ( 0xbe, "arrow_magic_animation"),
            
            (0x671, "triforce"),
            (0x66f, "hearts_and_containers"),
            (0x670, "partial_hearts"),
            
            (0x66d, "rupees"),
            (0x67d, "rupees_to_add"),
            
            (0x658, "bombs"),
            (0x67C, "bombMax"),
            (0x657, "sword"),
            (0x659, "arrows"),
            (0x65A, "bow"),
            (0x65B, "candle"),
            (0x65C, "whistle"),
            (0x65D, "food"),
            (0x65E, "potion"),
            (0x65F, "magic_rod"),
            (0x660, "raft"),
            (0x661, "magic_book"),
            (0x662, "ring"),
            (0x663, "step_ladder"),
            (0x664, "magic_Key"),
            (0x665, "power_braclet"),
            (0x666, "letter"),
            (0x674, "regular_boomerang"),
            (0x675, "magic_boomerang"),
            
            (0x667, "compass"),
            (0x668, "map"),
            (0x669, "compass9"),
            (0x66A, "map9"),
            
            (0x66C, "clock"),
            
            (0x66C, "keys"),
            
            (0x656, "selected_item"),
            
            (0x627, "room_kill_count"),
            ]
        
        self.index_map = {name: index for index, (_, name) in enumerate(self.locations)}

    def get_index(self, name):
        if name in self.index_map:
            return self.index_map[name]
        raise Exception(f"Unknown memory location {name}") 
        
    def get_address_list(self):
        return [x for x, _ in self.locations]
    
zelda_memory_layout = zelda_memory_layout()