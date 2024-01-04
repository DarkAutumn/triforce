# constants

zelda_mode_completed_scrolling = 4
zelda_mode_gameplay = 5
zelda_mode_prepare_scrolling = 6
zelda_mode_scrolling = 7
zelda_mode_gameover = 0x11

# offsets
zelda_memory_offsets = {
    # game state
    "level" : 0x10,
    "location" : 0xeb,
    "mode" : 0x12,

    # animation
    "sword_animation" : 0xb9,
    "beam_animation" : 0xba,
    "bomb_or_flame_animation" : 0xbc,
    "bomb_or_flame_animation2" : 0xbd,
    "arrow_magic_animation" : 0xbe,
    
    # triforce
    "triforce" : 0x671,
    "triforce_of_power" : 0x672,

    # hearts
    "hearts_and_containers" : 0x66f,
    "partial_hearts" : 0x670,

    #rupees
    "rupees" : 0x66d,
    "rupees_to_add" : 0x67d,

    # items
    "selected_item" : 0x656,

    "bombs" : 0x658,
    "bombMax" : 0x67c,
    "sword" : 0x657,
    "arrows" : 0x659,
    "bow" : 0x65a,
    "candle" : 0x65b,
    "whistle" : 0x65c,
    "food" : 0x65d,
    "potion" : 0x65e,
    "wand" : 0x65f,
    "raft" : 0x660,
    "magic_book" : 0x661,
    "ring" : 0x662,
    "step_ladder" : 0x663,
    "magic_Key" : 0x664,
    "power_braclet" : 0x665,
    "letter" : 0x666,
    "regular_boomerang" : 0x674,
    "magic_boomerang" : 0x675,

    # dungeon items
    "compass" : 0x667,
    "map" : 0x668,
    "compass_level9" : 0x669,
    "map9" : 0x66a,
    "keys" : 0x66c,

    # temporary items
    "clock" : 0x66c,

    # kill counts
    "room_kill_count" : 0x627,

    # save data
    "save_enabled_0" : 0x0633,
    "save_enabled_1" : 0x0634,
    "save_enabled_2" : 0x0635,
    "save_name_0" : 0x638,
    "save_name_1" : 0x640,
    "save_name_2" : 0x648,
}

## Uncomment to re-generate the properties below
#def generate_properties(property_dict):
#    properties_code = "    # begin generated code\n"
#    for name in property_dict.keys():
#            properties_code += f"""
#    @property
#    def {name}(self):
#        return self["{name}"]
#
#    @{name}.setter
#    def {name}(self, value):
#        self["{name}"] = value
#    """
#    properties_code += "\n    # end generated code\n"
#    return properties_code

#print(generate_properties(zelda_memory_offsets))


class ZeldaMemory():
    """Properties with memory offsets for """
    def __init__(self, nes_memory):
        self.nes_memory = nes_memory

    def __getitem__(self, key):
        if isinstance(key, str):
            offset = zelda_memory_offsets.get(key, None)
            return self.nes_memory[offset]
            
        elif isinstance(key, int):
            return self.nes_memory[key]
        
        raise KeyError(f"Key must be str or int, not {type(key)}")
    
    def __setitem__(self, key, value):
        if isinstance(key, str):
            offset = zelda_memory_offsets.get(key, None)
            
        elif isinstance(key, int):
            offset = key

        else:
            raise KeyError(f"Key must be str or int, not {type(key)}")

        # check if value is a sequence, if so, set each byte
        if hasattr(value, "__iter__"):
            for i in range(0, len(value)):
                self.nes_memory[offset + i] = self._normalizeValue(value[i])
        else:
            self.nes_memory[offset] = self._normalizeValue(value)

    def _normalizeValue(self, value):
        if isinstance(value, bool):
            value = 1 if value else 0
        elif isinstance(value, str):
            value = ord(value[0]) & 0xff
        else:
            value = value & 0xff

        return value
    
    def snapshot(self):
        return ZeldaMemory(self.nes_memory.copy())
    
    @property
    def hearts(self) -> float:
        full_hearts = (self.hearts_and_containers & 0x0F) + 1
        partialHealth = self.partial_hearts
        if partialHealth > 0xf0:
            return full_hearts
        
        return full_hearts - 1 + float(partialHealth) / 255
    
    @property
    def heart_containers(self):
        return (self.hearts_and_containers >> 4) + 1
    
    def has_triforce(self, level : int):
        level = level - 1
        return self.triforce & (1 << level)
    
    @property
    def triforce_pieces(self):
        # count the bits set in self.triforce
        count = 0
        for i in range(0, 8):
            if self.triforce & (1 << i):
                count += 1

        return count
    
    # begin generated code     

    @property
    def level(self):
        return self["level"]   

    @level.setter
    def level(self, value):    
        self["level"] = value  

    @property
    def location(self):        
        return self["location"]

    @location.setter
    def location(self, value):
        self["location"] = value

    @property
    def mode(self):
        return self["mode"]

    @mode.setter
    def mode(self, value):
        self["mode"] = value

    @property
    def sword_animation(self):
        return self["sword_animation"]

    @sword_animation.setter
    def sword_animation(self, value):
        self["sword_animation"] = value

    @property
    def beam_animation(self):
        return self["beam_animation"]

    @beam_animation.setter
    def beam_animation(self, value):
        self["beam_animation"] = value

    @property
    def bomb_or_flame_animation(self):
        return self["bomb_or_flame_animation"]

    @bomb_or_flame_animation.setter
    def bomb_or_flame_animation(self, value):
        self["bomb_or_flame_animation"] = value

    @property
    def bomb_or_flame_animation2(self):
        return self["bomb_or_flame_animation2"]

    @bomb_or_flame_animation2.setter
    def bomb_or_flame_animation2(self, value):
        self["bomb_or_flame_animation2"] = value

    @property
    def arrow_magic_animation(self):
        return self["arrow_magic_animation"]

    @arrow_magic_animation.setter
    def arrow_magic_animation(self, value):
        self["arrow_magic_animation"] = value

    @property
    def triforce(self):
        return self["triforce"]

    @triforce.setter
    def triforce(self, value):
        self["triforce"] = value

    @property
    def triforce_of_power(self):
        return self["triforce_of_power"]

    @triforce_of_power.setter
    def triforce_of_power(self, value):
        self["triforce_of_power"] = value

    @property
    def hearts_and_containers(self):
        return self["hearts_and_containers"]

    @hearts_and_containers.setter
    def hearts_and_containers(self, value):
        self["hearts_and_containers"] = value

    @property
    def partial_hearts(self):
        return self["partial_hearts"]

    @partial_hearts.setter
    def partial_hearts(self, value):
        self["partial_hearts"] = value

    @property
    def rupees(self):
        return self["rupees"]

    @rupees.setter
    def rupees(self, value):
        self["rupees"] = value

    @property
    def rupees_to_add(self):
        return self["rupees_to_add"]

    @rupees_to_add.setter
    def rupees_to_add(self, value):
        self["rupees_to_add"] = value

    @property
    def selected_item(self):
        return self["selected_item"]

    @selected_item.setter
    def selected_item(self, value):
        self["selected_item"] = value

    @property
    def bombs(self):
        return self["bombs"]

    @bombs.setter
    def bombs(self, value):
        self["bombs"] = value

    @property
    def bombMax(self):
        return self["bombMax"]

    @bombMax.setter
    def bombMax(self, value):
        self["bombMax"] = value

    @property
    def sword(self):
        return self["sword"]

    @sword.setter
    def sword(self, value):
        self["sword"] = value

    @property
    def arrows(self):
        return self["arrows"]

    @arrows.setter
    def arrows(self, value):
        self["arrows"] = value

    @property
    def bow(self):
        return self["bow"]

    @bow.setter
    def bow(self, value):
        self["bow"] = value

    @property
    def candle(self):
        return self["candle"]

    @candle.setter
    def candle(self, value):
        self["candle"] = value

    @property
    def whistle(self):
        return self["whistle"]

    @whistle.setter
    def whistle(self, value):
        self["whistle"] = value

    @property
    def food(self):
        return self["food"]

    @food.setter
    def food(self, value):
        self["food"] = value

    @property
    def potion(self):
        return self["potion"]

    @potion.setter
    def potion(self, value):
        self["potion"] = value

    @property
    def wand(self):
        return self["wand"]

    @wand.setter
    def wand(self, value):
        self["wand"] = value

    @property
    def raft(self):
        return self["raft"]

    @raft.setter
    def raft(self, value):
        self["raft"] = value

    @property
    def magic_book(self):
        return self["magic_book"]

    @magic_book.setter
    def magic_book(self, value):
        self["magic_book"] = value

    @property
    def ring(self):
        return self["ring"]

    @ring.setter
    def ring(self, value):
        self["ring"] = value

    @property
    def step_ladder(self):
        return self["step_ladder"]

    @step_ladder.setter
    def step_ladder(self, value):
        self["step_ladder"] = value

    @property
    def magic_Key(self):
        return self["magic_Key"]

    @magic_Key.setter
    def magic_Key(self, value):
        self["magic_Key"] = value

    @property
    def power_braclet(self):
        return self["power_braclet"]

    @power_braclet.setter
    def power_braclet(self, value):
        self["power_braclet"] = value

    @property
    def letter(self):
        return self["letter"]

    @letter.setter
    def letter(self, value):
        self["letter"] = value

    @property
    def regular_boomerang(self):
        return self["regular_boomerang"]

    @regular_boomerang.setter
    def regular_boomerang(self, value):
        self["regular_boomerang"] = value

    @property
    def magic_boomerang(self):
        return self["magic_boomerang"]

    @magic_boomerang.setter
    def magic_boomerang(self, value):
        self["magic_boomerang"] = value

    @property
    def compass(self):
        return self["compass"]

    @compass.setter
    def compass(self, value):
        self["compass"] = value

    @property
    def map(self):
        return self["map"]

    @map.setter
    def map(self, value):
        self["map"] = value

    @property
    def compass_level9(self):
        return self["compass_level9"]

    @compass_level9.setter
    def compass_level9(self, value):
        self["compass_level9"] = value

    @property
    def map9(self):
        return self["map9"]

    @map9.setter
    def map9(self, value):
        self["map9"] = value

    @property
    def keys(self):
        return self["keys"]

    @keys.setter
    def keys(self, value):
        self["keys"] = value

    @property
    def clock(self):
        return self["clock"]

    @clock.setter
    def clock(self, value):
        self["clock"] = value

    @property
    def room_kill_count(self):
        return self["room_kill_count"]

    @room_kill_count.setter
    def room_kill_count(self, value):
        self["room_kill_count"] = value

    @property
    def save_enabled_0(self):
        return self["save_enabled_0"]

    @save_enabled_0.setter
    def save_enabled_0(self, value):
        self["save_enabled_0"] = value

    @property
    def save_enabled_1(self):
        return self["save_enabled_1"]

    @save_enabled_1.setter
    def save_enabled_1(self, value):
        self["save_enabled_1"] = value

    @property
    def save_enabled_2(self):
        return self["save_enabled_2"]

    @save_enabled_2.setter
    def save_enabled_2(self, value):
        self["save_enabled_2"] = value

    @property
    def save_name_0(self):
        return self["save_name_0"]

    @save_name_0.setter
    def save_name_0(self, value):
        self["save_name_0"] = value

    @property
    def save_name_1(self):
        return self["save_name_1"]

    @save_name_1.setter
    def save_name_1(self, value):
        self["save_name_1"] = value

    @property
    def save_name_2(self):
        return self["save_name_2"]

    @save_name_2.setter
    def save_name_2(self, value):
        self["save_name_2"] = value
    
    # end generated code
