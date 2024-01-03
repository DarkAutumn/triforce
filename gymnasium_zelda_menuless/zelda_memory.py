
zelda_memory_offsets = {
    "level" : 0x10,
    "location" : 0xeb,
    "mode" : 0x12,
    "sword_animation" : 0xb9,
    "beam_animation" : 0xba,
    "bomb_or_flame_animation" : 0xbc,
    "bomb_or_flame_animation2" : 0xbd,
    "arrow_magic_animation" : 0xbe,
    "triforce" : 0x671,
    "hearts_and_containers" : 0x66f,
    "partial_hearts" : 0x670,
    "rupees" : 0x66d,
    "rupees_to_add" : 0x67d,
    "bombs" : 0x658,
    "bombMax" : 0x67c,
    "sword" : 0x657,
    "arrows" : 0x659,
    "bow" : 0x65a,
    "candle" : 0x65b,
    "whistle" : 0x65c,
    "food" : 0x65d,
    "potion" : 0x65e,
    "magic_rod" : 0x65f,
    "raft" : 0x660,
    "magic_book" : 0x661,
    "ring" : 0x662,
    "step_ladder" : 0x663,
    "magic_Key" : 0x664,
    "power_braclet" : 0x665,
    "letter" : 0x666,
    "regular_boomerang" : 0x674,
    "magic_boomerang" : 0x675,
    "compass" : 0x667,
    "map" : 0x668,
    "compass_level9" : 0x669,
    "map9" : 0x66a,
    "clock" : 0x66c,
    "keys" : 0x66c,

    "selected_item" : 0x656,

    "room_kill_count" : 0x627,

    "save_enabled_0" : 0x0633,
    "save_enabled_1" : 0x0634,
    "save_enabled_2" : 0x0635,
    "save_name_0" : 0x638,
    "save_name_1" : 0x640,
    "save_name_2" : 0x648,
}

class DictionaryBasedMetaClass(type):
    def __new__(meta, name, bases, dct):        
        def make_property(key):
            def getter(self):
                return self._get_property(key)
            return property(getter)

        for key in zelda_memory_offsets.keys():
            dct[key] = make_property(key)

        return super(DictionaryBasedMetaClass, meta).__new__(meta, name, bases, dct)

class ZeldaMemory(metaclass=DictionaryBasedMetaClass):
    """Properties with memory offsets for """
    def __init__(self, nes_memory):
        self.nes_memory = nes_memory

    def get_property(self, name):
        offset = zelda_memory_offsets(name, None)
        return self.nes_memory[offset]

    def set_save_name(self, slot : int, name = "link"):
        if slot not in range(0, 2):
            raise IndexError()
        
        offset = self.get_property(f"save_enabled_{slot}")
        self.nes_memory[offset] = 1

        offset = self.get_property(f"save_name_{slot}")
        for i in range(0, max(8, len(name))):
            self.nes_memory[offset + i] = ord(name[i]) & 0xff
    