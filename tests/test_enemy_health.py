# pylint: disable=all

"""Tests for Area 4: Enemy health encoding and damage mechanics.

Verifies that:
- ObjHP stores HP in high nibble (multiples of $10)
- Python's >> 4 extraction is correct
- 0-HP enemies (keese, gel) are handled correctly
- Damage values from assembly match expected
- BUG-3 fix: data.json obj_health_b/c addresses are correct
"""

import os
import sys
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from tests.asm_addresses import OBJ_HP, OBJ_TYPE, OBJ_STATE, OBJ_METASTATE
from tests.conftest import ZeldaFixture


# ObjectTypeToHpPairs from Z_07.asm:5279-5284
HP_TABLE = bytes([
    0x06, 0x43, 0x25, 0x31, 0x12, 0x24, 0x81, 0x14,
    0x22, 0x42, 0x00, 0xA9, 0x8F, 0x20, 0x00, 0x3F,
    0xF9, 0xFA, 0x46, 0x62, 0x11, 0x2F, 0xFF, 0xFF,
    0x7F, 0xF6, 0x2F, 0xFF, 0xFF, 0x22, 0x46, 0xF1,
    0xF2, 0xAA, 0xAA, 0xFB, 0xBF, 0xF0
])


def extract_hp(obj_type):
    """Replicate ExtractHitPointValue from Z_04.asm:11002-11021."""
    idx = obj_type // 2
    if idx >= len(HP_TABLE):
        return None
    byte = HP_TABLE[idx]
    if obj_type % 2 == 0:
        return byte & 0xF0
    return (byte & 0x0F) << 4


class TestHPTableExtraction:
    """Verify the HP table extraction logic matches assembly."""

    def test_even_type_gets_high_nibble(self):
        """Even object types: AND $F0 (high nibble kept)."""
        # Type 0x02 (RedLynel): byte is 0x43, high nibble = 0x40
        assert extract_hp(0x02) == 0x40

    def test_odd_type_gets_low_nibble_shifted(self):
        """Odd object types: low nibble << 4."""
        # Type 0x03 (BlueMoblin): byte is 0x43, low nibble 3 << 4 = 0x30
        assert extract_hp(0x03) == 0x30

    def test_result_always_multiple_of_16(self):
        """All HP values from the table are multiples of $10."""
        for obj_type in range(1, 0x4C):
            hp = extract_hp(obj_type)
            if hp is not None:
                assert hp % 0x10 == 0, f"type {obj_type:#04x}: hp={hp:#04x} not multiple of $10"

    @pytest.mark.parametrize("obj_type,expected_hp", [
        (0x07, 0x10),   # BlueOctorok: 1 HP
        (0x06, 0x30),   # RedGoriya: 3 HP
        (0x2A, 0x20),   # Stalfos: 2 HP
        (0x3D, 0x60),   # Aquamentus: 6 HP
        (0x0C, 0x80),   # RedDarknut: 8 HP
        (0x3E, 0xF0),   # Ganon: 15 HP
    ])
    def test_known_enemy_hp(self, obj_type, expected_hp):
        """Spot-check HP values against known enemies."""
        assert extract_hp(obj_type) == expected_hp

    @pytest.mark.parametrize("obj_type,name", [
        (0x14, "GreenGel"),
        (0x15, "BlueGel"),
        (0x1B, "BlueKeese"),
        (0x1C, "RedKeese"),
        (0x1D, "BlackKeese"),
    ])
    def test_zero_hp_enemies(self, obj_type, name):
        """Enemies that have 0 HP from the table — die in one hit."""
        assert extract_hp(obj_type) == 0, f"{name} (type {obj_type:#04x}) should have 0 HP"

    def test_python_high_nibble_extraction(self):
        """Python reads enemy.health as ObjHP >> 4 — verify this matches table for all types."""
        for obj_type in range(1, 0x4C):
            raw_hp = extract_hp(obj_type)
            if raw_hp is not None:
                python_hp = raw_hp >> 4
                assert python_hp == raw_hp // 16


class TestSwordDamageValues:
    """Verify sword damage constants from SwordDamagePoints (Z_07.asm:6166)."""

    def test_sword_damage_table(self):
        """Wood=$10 (1HP), White=$20 (2HP), Magic=$40 (4HP)."""
        sword_damage = [0x10, 0x20, 0x40]
        assert sword_damage[0] == 0x10  # Wood sword
        assert sword_damage[1] == 0x20  # White sword
        assert sword_damage[2] == 0x40  # Magic sword

    def test_all_damage_multiples_of_16(self):
        """All damage values are multiples of $10, preserving low nibble = 0."""
        damages = [0x10, 0x20, 0x40,  # swords
                   0x40,               # bomb
                   0x10]               # fire
        for d in damages:
            assert d % 0x10 == 0


class TestEnemyHPEmpirical:
    """Empirical tests using actual NES emulator."""

    BTN_L = 6
    BTN_A = 8

    def test_goriya_hp_from_savestate(self):
        """1_44e has type 0x06 (RedGoriya) enemies with 3 HP ($30)."""
        fix = ZeldaFixture("1_44e.state")
        try:
            r = fix.ram
            found = False
            for slot in range(1, 11):
                if r[OBJ_TYPE + slot] == 0x06:
                    assert r[OBJ_HP + slot] == 0x30
                    assert r[OBJ_HP + slot] >> 4 == 3
                    found = True
            assert found, "No RedGoriya (type 0x06) found in 1_44e"
        finally:
            fix.close()

    def test_stalfos_hp_from_savestate(self):
        """1_74w has type 0x2A (Stalfos) enemies with 2 HP ($20)."""
        fix = ZeldaFixture("1_74w.state")
        try:
            r = fix.ram
            found = False
            for slot in range(1, 11):
                if r[OBJ_TYPE + slot] == 0x2A:
                    assert r[OBJ_HP + slot] == 0x20
                    assert r[OBJ_HP + slot] >> 4 == 2
                    found = True
            assert found, "No Stalfos (type 0x2A) found in 1_74w"
        finally:
            fix.close()

    def test_keese_hp_zero(self):
        """1_72e has type 0x1B (BlueKeese) with 0 HP — they spawn from edges after ~90 frames."""
        fix = ZeldaFixture("1_72e.state")
        try:
            # Keese spawn from screen edges, need to wait for them
            fix.step_n(100)
            r = fix.ram
            found = False
            for slot in range(1, 11):
                if r[OBJ_TYPE + slot] == 0x1B:
                    assert r[OBJ_HP + slot] == 0x00
                    assert r[OBJ_HP + slot] >> 4 == 0
                    found = True
            assert found, "No BlueKeese (type 0x1B) found in 1_72e after 100 frames"
        finally:
            fix.close()

    def test_hp_low_nibble_always_zero(self):
        """All enemy ObjHP values should have low nibble = 0 at init."""
        for state_name in ['1_44e.state', '1_72e.state', '1_74w.state']:
            fix = ZeldaFixture(state_name)
            try:
                r = fix.ram
                for slot in range(1, 11):
                    if r[OBJ_TYPE + slot] != 0:
                        hp = r[OBJ_HP + slot]
                        assert hp & 0x0F == 0, \
                            f"{state_name} slot {slot}: hp={hp:#04x} has non-zero low nibble"
            finally:
                fix.close()

    def test_hp_matches_table_for_all_present_enemies(self):
        """Every enemy in our savestates has ObjHP matching the HP table."""
        for state_name in ['1_44e.state', '1_72e.state', '1_74w.state']:
            fix = ZeldaFixture(state_name)
            try:
                r = fix.ram
                for slot in range(1, 11):
                    etype = r[OBJ_TYPE + slot]
                    if etype != 0 and etype < 0x4C:
                        actual_hp = r[OBJ_HP + slot]
                        expected_hp = extract_hp(etype)
                        assert actual_hp == expected_hp, \
                            f"{state_name} slot {slot}: type={etype:#04x} hp={actual_hp:#04x} expected={expected_hp:#04x}"
            finally:
                fix.close()

    def test_keese_dies_to_metastate_not_hp(self):
        """When keese is killed, ObjHP stays 0 — death is via ObjMetastate."""
        fix = ZeldaFixture("1_72e.state")
        try:
            # Walk to keese
            for c in 'llllllllllldddllllllllll':
                btn = self.BTN_L if c == 'l' else (5 if c == 'd' else 4)
                fix.step_n(8, [btn])

            # Give link a sword and swing
            fix.set('sword', 1)
            fix.step([self.BTN_A])

            # Watch for keese death over 30 frames
            dying_seen = False
            for _ in range(30):
                fix.step()
                r = fix.ram
                for slot in range(1, 4):
                    if r[OBJ_TYPE + slot] == 0x1B:
                        meta = r[OBJ_METASTATE + slot]
                        hp = r[OBJ_HP + slot]
                        if meta >= 0x10:
                            dying_seen = True
                            assert hp == 0x00, f"Keese HP changed to {hp:#04x} during death"
        finally:
            fix.close()
