# pylint: disable=all
"""Layer 0: Verify RAM address mapping between zelda_game_data.txt and assembly.

These tests verify that every named address in our data files correctly maps
to the NES RAM layout defined in zelda-asm/src/Variables.inc.
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from conftest import ZeldaFixture
from asm_addresses import *

from triforce.zelda_game_data import zelda_game_data


# --- T0.1: Named variable addresses match between data sources ---

class TestNamedVariableAddresses:
    """Verify zelda_game_data.txt [memory] addresses match data.json and raw RAM."""

    def test_link_position_matches_ram(self, emu):
        """link_x/link_y in data.json match ObjX[0]/ObjY[0] in RAM."""
        ram = emu.ram
        assert emu.get('link_x') == ram[OBJ_X], "link_x doesn't match ObjX[0]"
        assert emu.get('link_y') == ram[OBJ_Y], "link_y doesn't match ObjY[0]"

    def test_link_status_matches_ram(self, emu):
        """link_status in data.json matches ObjState[0] in RAM."""
        assert emu.get('link_status') == emu.ram[OBJ_STATE], "link_status doesn't match ObjState[0]"

    def test_link_direction_matches_ram(self, emu):
        """link_direction in data.json matches ObjDir[0] in RAM."""
        assert emu.get('link_direction') == emu.ram[OBJ_DIR], "link_direction doesn't match ObjDir[0]"

    def test_game_state_addresses(self, emu):
        """level, location, mode match their RAM addresses."""
        ram = emu.ram
        assert emu.get('level') == ram[CUR_LEVEL]
        assert emu.get('location') == ram[ROOM_ID]
        assert emu.get('mode') == ram[GAME_MODE]

    def test_hearts_addresses(self, emu):
        """hearts_and_containers and partial_hearts match HeartValues/HeartPartial."""
        ram = emu.ram
        assert emu.get('hearts_and_containers') == ram[HEART_VALUES]
        assert emu.get('partial_hearts') == ram[HEART_PARTIAL]

    def test_inventory_addresses(self, emu):
        """Spot-check inventory addresses against RAM."""
        ram = emu.ram
        assert emu.get('sword') == ram[INV_SWORD]
        assert emu.get('bombs') == ram[INV_BOMBS]
        assert emu.get('arrows') == ram[INV_ARROWS]
        assert emu.get('bow') == ram[INV_BOW]
        assert emu.get('rupees') == ram[INV_RUPEES]
        assert emu.get('keys') == ram[INV_KEYS]
        assert emu.get('triforce') == ram[INV_TRIFORCE]
        assert emu.get('selected_item') == ram[SELECTED_ITEM]
        assert emu.get('ring') == ram[INV_RING]
        assert emu.get('bomb_max') == ram[INV_BOMB_MAX]

    def test_sound_address(self, emu):
        """sound_pulse_1 matches Tune0 at $605."""
        assert emu.get('sound_pulse_1') == emu.ram[TUNE0]

    def test_triforce_of_power_address(self, emu):
        """triforce_of_power at $672 matches LastBossDefeated in assembly.
        Semantics: boolean flag, 0=not defeated, non-zero=defeated."""
        assert emu.get('triforce_of_power') == emu.ram[INV_TRIFORCE_P]

    def test_rng_addresses(self, emu):
        """RNG bytes at $018-$023 are contiguous."""
        ram = emu.ram
        for i in range(12):
            assert emu.get(f'rng_{i}') == ram[RNG_BASE + i], f"rng_{i} mismatch"

    def test_treasure_addresses(self, emu):
        """Treasure x/y/flag match their RAM addresses."""
        ram = emu.ram
        assert emu.get('treasure_x') == ram[TREASURE_X]
        assert emu.get('treasure_y') == ram[TREASURE_Y]
        assert emu.get('treasure_flag') == ram[TREASURE_FLAG]

    def test_kill_tracking_addresses(self, emu):
        """kill_streak matches WorldKillCount at $627."""
        assert emu.get('kill_streak') == emu.ram[WORLD_KILL_CNT]


# --- T0.2: Object table addresses and lengths ---

class TestObjectTables:
    """Verify zelda_game_data.txt [tables] offsets and lengths."""

    TABLE_SPECS = {
        # name: (expected_offset, expected_length, assembly_symbol)
        'obj_pos_x':     (0x070, 0x0C, 'ObjX'),
        'obj_pos_y':     (0x084, 0x0C, 'ObjY'),
        'obj_direction':  (0x098, 0x0C, 'ObjDir'),
        'obj_status':     (0x0AC, 0x0C, 'ObjState'),
        'obj_stun_timer': (0x03D, 0x0C, 'ObjStunTimer'),
        'obj_id':         (0x34F, 0x0C, 'ObjType'),
        'obj_spawn_state':(0x405, 0x0C, 'ObjMetastate'),
        'item_timer':     (0x3A8, 0x0C, 'ObjPosFrac/ItemLifetime'),
        'obj_health':     (0x485, 0x0C, 'ObjHP'),
        'tile_layout':    (0xD30, 0x2C0, 'PlayAreaTiles'),
    }

    @pytest.mark.parametrize("table_name", TABLE_SPECS.keys())
    def test_table_offset_and_length(self, table_name):
        """Each table's offset and length in zelda_game_data.txt is correct."""
        expected_offset, expected_length, asm_name = self.TABLE_SPECS[table_name]
        offset, length = zelda_game_data.tables[table_name]
        assert offset == expected_offset, \
            f"{table_name} offset: got 0x{offset:03X}, expected 0x{expected_offset:03X} ({asm_name})"
        assert length == expected_length, \
            f"{table_name} length: got 0x{length:02X}, expected 0x{expected_length:02X}"

    def test_table_slot0_matches_link_vars(self, emu):
        """Slot 0 of each object table matches the corresponding Link variable."""
        ram = emu.ram
        tables = emu.object_tables()

        assert tables.read('obj_pos_x')[0] == emu.get('link_x')
        assert tables.read('obj_pos_y')[0] == emu.get('link_y')
        assert tables.read('obj_direction')[0] == emu.get('link_direction')
        assert tables.read('obj_status')[0] == emu.get('link_status')


# --- T0.3: Link is always object slot 0 ---

class TestLinkSlotZero:
    """Verify Link occupies slot 0 in all object tables."""

    def test_link_is_slot_zero(self, dungeon_emu):
        """Across multiple savestates, Link's data is at slot 0."""
        ram = dungeon_emu.ram
        assert ram[OBJ_X] == dungeon_emu.get('link_x')
        assert ram[OBJ_Y] == dungeon_emu.get('link_y')
        assert ram[OBJ_DIR] == dungeon_emu.get('link_direction')
        assert ram[OBJ_STATE] == dungeon_emu.get('link_status')


# --- T0.4: room_kills / ObjType overlap ---

class TestRoomKillsOverlap:
    """Verify room_kills at $34F is ObjType[0] (intentional overlap)."""

    def test_room_kills_is_obj_type_slot0(self, emu):
        """room_kills and ObjType[0] are the same byte at $34F."""
        ram = emu.ram
        assert ram[OBJ_TYPE] == emu.get('room_kills'), \
            "room_kills should be the same byte as ObjType[0]"

    def test_room_kills_writable(self, emu):
        """room_kills and ObjType[0] are the same writable byte at $34F.
        Writing to one is visible via the other."""
        emu.set('room_kills', 5)
        assert emu.ram[OBJ_TYPE] == 5, "Writing room_kills should update ObjType[0]"


# --- T0.5: item_timer / ObjPosFrac union ---

class TestItemTimerUnion:
    """Verify item_timer table at $3A8 is ObjPosFrac (context-dependent union)."""

    def test_item_timer_table_address(self):
        """item_timer table starts at $3A8 with 12 entries."""
        offset, length = zelda_game_data.tables['item_timer']
        assert offset == OBJ_POS_FRAC, f"Expected 0x{OBJ_POS_FRAC:03X}, got 0x{offset:03X}"
        assert length == OBJ_SLOT_COUNT


# --- T0.6: Weapon slot ObjState addresses ---

class TestWeaponSlotAddresses:
    """Verify that animation addresses are ObjState for the correct weapon slots."""

    WEAPON_SLOTS = {
        'sword_animation':              (SLOT_SWORD, SWORD_STATE),
        'beam_animation':               (SLOT_BEAM, BEAM_STATE),
        'bait_or_boomerang_animation':  (SLOT_BOOMERANG, BOOMERANG_STATE),
        'bomb_or_flame_animation':      (SLOT_BOMB1, BOMB1_STATE),
        'bomb_or_flame_animation2':     (SLOT_BOMB2, BOMB2_STATE),
        'arrow_magic_animation':        (SLOT_ARROW, ARROW_STATE),
    }

    @pytest.mark.parametrize("name,slot_info", WEAPON_SLOTS.items(),
                             ids=WEAPON_SLOTS.keys())
    def test_weapon_animation_is_obj_state(self, emu, name, slot_info):
        """Each animation address equals OBJ_STATE + slot number."""
        slot, expected_addr = slot_info
        ram = emu.ram
        # Verify the address arithmetic
        assert OBJ_STATE + slot == expected_addr, \
            f"{name}: OBJ_STATE(0x{OBJ_STATE:02X}) + slot(0x{slot:02X}) != 0x{expected_addr:02X}"
        # Verify the data API reads from the same address
        assert emu.get(name) == ram[expected_addr], \
            f"{name}: data API value != ram[0x{expected_addr:02X}]"


# --- T0.7: Individual enemy health addresses ---

class TestEnemyHealthAddresses:
    """Verify obj_health_N addresses in data.json match the obj_health table."""

    def test_health_table_contiguous(self, dungeon_emu):
        """The obj_health table reads 12 contiguous bytes from $485.
        Individual data.json entries for slots 1-A should match table[slot]."""
        tables = dungeon_emu.object_tables()
        health_table = tables.read('obj_health')

        # Slots 1-A (0x0A) should match their data.json entries
        for slot in range(1, 0x0B):
            name = f'obj_health_{slot:x}'
            table_val = int(health_table[slot])
            data_val = dungeon_emu.get(name)
            assert table_val == data_val, \
                f"{name}: table[{slot}]={table_val} != data.json={data_val}"

    def test_health_b_c_data_json_offset(self):
        """Document: data.json obj_health_b is at $491 (not $490).
        This is a known 1-byte offset in data.json for slots B and C.
        The table-based read (used by game code) is correct."""
        import json
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 'triforce', 'custom_integrations', 'Zelda-NES', 'data.json')
        with open(data_path) as f:
            data = json.load(f)

        # obj_health_b/c should match ObjHP + slot offset
        addr_b = data['info']['obj_health_b']['address']
        addr_c = data['info']['obj_health_c']['address']
        expected_b = OBJ_HP + 0x0B  # $490
        expected_c = OBJ_HP + 0x0C  # $491

        assert addr_b == expected_b, f"obj_health_b expected at ${expected_b:03X}, got ${addr_b:03X}"
        assert addr_c == expected_c, f"obj_health_c expected at ${expected_c:03X}, got ${addr_c:03X}"


# --- Cross-check: all memory entries have a matching asm_addresses constant ---

class TestAddressCompleteness:
    """Verify zelda_game_data.txt [memory] addresses are consistent with data.json."""

    def test_memory_entries_match_data_json(self, emu):
        """Every address in zelda_game_data.txt [memory] should be readable via data API."""
        for name, address in zelda_game_data.memory.items():
            ram_val = emu.ram[address]
            data_val = emu.get(name)
            assert ram_val == data_val, \
                f"{name} at 0x{address:03X}: ram={ram_val} != data={data_val}"
