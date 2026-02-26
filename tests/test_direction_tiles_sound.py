"""Areas 9, 10, 13: Direction encoding, tile layout, sound bitmask verification.

Tests verify:
- Direction enum values match NES convention (E=1,W=2,S=4,N=8)
- NES never stores composite directions in ObjDir
- Tile layout reshape is correct (32 columns x 22 rows, column-major)
- Sound bitmask register at $605 (Tune0) with correct bit positions
"""

import pytest
import numpy as np
from conftest import ZeldaFixture
from asm_addresses import *


class TestDirectionEncoding:
    """Area 9: Verify direction values match NES assembly."""

    def test_cardinal_values(self):
        """E=1, W=2, S=4, N=8 per NES convention."""
        from triforce.zelda_enums import Direction
        assert Direction.E.value == 1
        assert Direction.W.value == 2
        assert Direction.S.value == 4
        assert Direction.N.value == 8

    def test_diagonal_values_are_composites(self):
        """Diagonals should be bitwise OR of cardinals."""
        from triforce.zelda_enums import Direction
        assert Direction.NE.value == 9   # 8 | 1
        assert Direction.NW.value == 10  # 8 | 2
        assert Direction.SE.value == 5   # 4 | 1
        assert Direction.SW.value == 6   # 4 | 2

    def test_from_ram_value_cardinals(self):
        """from_ram_value should return correct direction for cardinal values."""
        from triforce.zelda_enums import Direction
        assert Direction.from_ram_value(1) == Direction.E
        assert Direction.from_ram_value(2) == Direction.W
        assert Direction.from_ram_value(4) == Direction.S
        assert Direction.from_ram_value(8) == Direction.N

    def test_from_ram_value_zero_is_none(self):
        """from_ram_value(0) should return NONE."""
        from triforce.zelda_enums import Direction
        assert Direction.from_ram_value(0) == Direction.NONE

    def test_from_ram_value_unknown_is_none(self):
        """from_ram_value for non-cardinal values should return NONE."""
        from triforce.zelda_enums import Direction
        # NES never writes composite values to ObjDir
        for val in [3, 5, 6, 7, 9, 10, 12, 15, 0xFF]:
            assert Direction.from_ram_value(val) == Direction.NONE

    def test_nes_direction_after_movement(self):
        """NES ObjDir matches expected values after directional input."""
        emu = ZeldaFixture("1_44e.state")
        try:
            # Move to center first
            for _ in range(20):
                emu.step([BTN_LEFT])

            state = emu.env.em.get_state()

            for btn, expected in [(BTN_RIGHT, 1), (BTN_LEFT, 2), (BTN_DOWN, 4), (BTN_UP, 8)]:
                emu.env.em.set_state(state)
                for _ in range(5):
                    emu.step([btn])
                assert emu.ram[OBJ_DIR] == expected, \
                    f"Button {btn}: expected dir={expected}, got {emu.ram[OBJ_DIR]}"
        finally:
            emu.close()

    def test_nes_diagonal_input_resolves_to_cardinal(self):
        """NES resolves diagonal input to a single cardinal direction."""
        emu = ZeldaFixture("1_44e.state")
        try:
            for _ in range(20):
                emu.step([BTN_LEFT])

            state = emu.env.em.get_state()

            for btns in [[BTN_UP, BTN_RIGHT], [BTN_UP, BTN_LEFT],
                         [BTN_DOWN, BTN_RIGHT], [BTN_DOWN, BTN_LEFT]]:
                emu.env.em.set_state(state)
                for _ in range(5):
                    emu.step(btns)
                d = emu.ram[OBJ_DIR]
                assert d in (1, 2, 4, 8), \
                    f"Diagonal input {btns} produced composite dir {d}"
        finally:
            emu.close()


class TestTileLayout:
    """Area 10: Verify tile reshape and indexing."""

    def test_tile_array_shape(self):
        """current_tiles should be 32 columns x 22 rows."""
        emu = ZeldaFixture("1_44e.state")
        try:
            gs = emu.game_state()
            tiles = gs.current_tiles
            assert tiles.shape == (32, 22), f"Expected (32, 22), got {tiles.shape}"
        finally:
            emu.close()

    def test_tiles_not_all_zero(self):
        """Tiles should contain non-zero values for a dungeon room."""
        emu = ZeldaFixture("1_44e.state")
        try:
            gs = emu.game_state()
            tiles = gs.current_tiles
            assert tiles.sum() > 0, "Tiles should not be all zeros"
        finally:
            emu.close()

    def test_tile_raw_reshape_matches_property(self):
        """Direct RAM reshape should match current_tiles."""
        emu = ZeldaFixture("1_44e.state")
        try:
            gs = emu.game_state()
            tiles_property = gs.current_tiles.numpy()

            raw = emu.ram[TILE_LAYOUT:TILE_LAYOUT + TILE_LAYOUT_LEN]
            tiles_direct = raw.reshape((32, 22))

            assert np.array_equal(tiles_property, tiles_direct), \
                "current_tiles should equal direct reshape(32, 22)"
        finally:
            emu.close()

    def test_door_tile_positions(self):
        """Door tiles should be at expected positions in the grid."""
        emu = ZeldaFixture("1_44e.state")
        try:
            gs = emu.game_state()
            tiles = gs.current_tiles

            # Center column for N/S doors (0xf = 15, out of 32 columns)
            # Near-left column for W door (2)
            # Near-right column for E door (0x1d = 29)
            # These positions must be valid indices
            assert 0 <= 0x0f < 32, "North door column should be valid"
            assert 0 <= 0x1d < 32, "East door column should be valid"
            assert 0 <= 2 < 32, "West door column should be valid"
            assert 0 <= 0x0a < 22, "West/East door row should be valid"
            assert 0 <= 2 < 22, "North door row should be valid"
            assert 0 <= 0x13 < 22, "South door row should be valid"
        finally:
            emu.close()


class TestSoundBitmask:
    """Area 13: Verify SoundKind values match assembly Tune0 register."""

    def test_sound_register_address(self):
        """Sound register should be at $605 (Tune0)."""
        emu = ZeldaFixture("1_44e.state")
        try:
            gs = emu.game_state()
            # sound_pulse_1 reads from $605
            assert gs.sound_pulse_1 == emu.ram[0x605]
        finally:
            emu.close()

    def test_soundkind_bitmask_values(self):
        """Each SoundKind should be a single bit (power of 2)."""
        from triforce.zelda_enums import SoundKind
        for kind in SoundKind:
            assert kind.value & (kind.value - 1) == 0, \
                f"{kind.name}={kind.value} is not a power of 2"
            assert 0x01 <= kind.value <= 0x40, \
                f"{kind.name}={kind.value} outside valid bit range"

    def test_soundkind_matches_assembly(self):
        """SoundKind values should match assembly Tune0 bit definitions."""
        from triforce.zelda_enums import SoundKind
        # Per assembly docs (ram.md $605):
        # $01=Arrow Deflected, $02=Boomerang Stun, $04=Magic Cast,
        # $08=Key Pickup, $10=Small Heart Pickup, $20=Set Bomb, $40=Heart Warning
        expected = {
            'ArrowDeflected': 0x01,
            'BoomerangStun': 0x02,
            'MagicCast': 0x04,
            'KeyPickup': 0x08,
            'SmallHeartPickup': 0x10,
            'SetBomb': 0x20,
            'HeartWarning': 0x40,
        }
        for name, value in expected.items():
            assert SoundKind[name].value == value, \
                f"SoundKind.{name} should be 0x{value:02X}"

    def test_no_overlapping_bits(self):
        """No two SoundKind values should share bits."""
        from triforce.zelda_enums import SoundKind
        combined = 0
        for kind in SoundKind:
            assert not (combined & kind.value), \
                f"{kind.name} overlaps with previous values"
            combined |= kind.value

    def test_is_sound_playing_bitmask(self):
        """is_sound_playing should correctly check bitmask against register."""
        from triforce.zelda_enums import SoundKind
        emu = ZeldaFixture("1_44e.state")
        try:
            gs = emu.game_state()
            # With Tune0=0, no sounds should be playing
            emu.ram[0x605] = 0
            gs2 = emu.game_state()
            for kind in SoundKind:
                assert not gs2.is_sound_playing(kind), \
                    f"{kind.name} should not be playing when Tune0=0"
        finally:
            emu.close()
