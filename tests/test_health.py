"""Area 7: Link health system tests.

Verifies health nibble encoding, beam health check, and health setter/getter
round-trip against the NES assembly (Z_07.asm MakeSwordShot lines 4632-4648).

Assembly health encoding:
  HeartValues ($066F): high nibble = containers_minus_one, low nibble = hearts_filled
  HeartPartial ($0670): $00=empty, $01-$7F=partial, $80-$FF=full partial

Assembly beam health check:
  containers_minus_one == hearts_filled AND partial >= $80
"""

import pytest
from tests.conftest import ZeldaFixture
from tests.asm_addresses import BTN_A, BEAM_STATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_gameplay(emu, max_frames=100):
    for _ in range(max_frames):
        emu.step()
        if emu.get('mode') == 5:
            return
    raise RuntimeError("Never reached gameplay mode")


def _set_health_and_step(emu, hc_val, partial_val):
    """Set hearts_and_containers and partial_hearts, then step to apply."""
    emu.set('hearts_and_containers', hc_val)
    emu.set('partial_hearts', partial_val)
    emu.step()


def _nes_fires_beam(emu):
    """Press A and check if the NES actually fires a beam (ObjState[0x0E] != 0)."""
    save = emu.save()
    emu.step([BTN_A])
    for _ in range(20):
        emu.step()
        if emu.ram[BEAM_STATE] != 0:
            emu.restore(save)
            return True
    emu.restore(save)
    return False


# ---------------------------------------------------------------------------
# Health nibble encoding
# ---------------------------------------------------------------------------

class TestHealthEncoding:
    """Verify hearts_and_containers nibble structure matches assembly."""

    def test_high_nibble_is_containers_minus_one(self):
        """High nibble of HeartValues = number of containers - 1."""
        emu = ZeldaFixture("debug_0_67_1772056964.state")
        _wait_for_gameplay(emu)

        game = emu.game_state()
        hc = emu.ram[0x66F]
        containers_minus_one = (hc >> 4) & 0x0F
        assert game.link.max_health == containers_minus_one + 1
        emu.close()

    def test_low_nibble_is_hearts_filled(self):
        """Low nibble of HeartValues = number of fully filled hearts."""
        emu = ZeldaFixture("debug_0_67_1772056964.state")
        _wait_for_gameplay(emu)

        hc = emu.ram[0x66F]
        hearts_filled = hc & 0x0F
        partial = emu.ram[0x670]
        # With full health: filled should equal containers_minus_one, partial >= $80
        containers_minus_one = (hc >> 4) & 0x0F
        assert hearts_filled == containers_minus_one
        assert partial >= 0x80
        emu.close()

    @pytest.mark.parametrize("hc_val,expected_max,expected_filled", [
        (0x00, 1, 0),   # 1 container, 0 filled
        (0x10, 2, 0),   # 2 containers, 0 filled
        (0x11, 2, 1),   # 2 containers, 1 filled
        (0x22, 3, 2),   # 3 containers, 2 filled
        (0xF0, 16, 0),  # 16 containers, 0 filled
        (0xFF, 16, 15), # 16 containers, 15 filled (max)
    ])
    def test_nibble_extraction(self, hc_val, expected_max, expected_filled):
        """Verify Python correctly extracts max_health and filled hearts."""
        emu = ZeldaFixture("debug_0_67_1772056964.state")
        _wait_for_gameplay(emu)

        emu.set('hearts_and_containers', hc_val)
        emu.step()

        game = emu.game_state()
        assert game.link.max_health == expected_max
        emu.close()


# ---------------------------------------------------------------------------
# Beam health check (BUG-1 fix verification)
# ---------------------------------------------------------------------------

class TestBeamHealthCheck:
    """Verify _is_health_full_for_beams matches NES assembly exactly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.emu = ZeldaFixture("debug_0_67_1772056964.state")
        _wait_for_gameplay(self.emu)
        self.cm1 = 2  # containers_minus_one for this state
        yield
        self.emu.close()

    def _check(self, hc_val, partial_val):
        """Set health, return (python_result, nes_result)."""
        save = self.emu.save()
        _set_health_and_step(self.emu, hc_val, partial_val)
        game = self.emu.game_state()
        py_result = game.link._is_health_full_for_beams
        nes_result = _nes_fires_beam(self.emu)
        self.emu.restore(save)
        return py_result, nes_result

    def test_full_health_fires_beam(self):
        """c-1 == filled, partial=$FF → beams fire."""
        py, nes = self._check((self.cm1 << 4) | self.cm1, 0xFF)
        assert py is True
        assert nes is True

    def test_partial_0x80_fires_beam(self):
        """c-1 == filled, partial=$80 (threshold) → beams fire."""
        py, nes = self._check((self.cm1 << 4) | self.cm1, 0x80)
        assert py is True
        assert nes is True

    def test_partial_0x7F_no_beam(self):
        """c-1 == filled, partial=$7F (below threshold) → no beams."""
        py, nes = self._check((self.cm1 << 4) | self.cm1, 0x7F)
        assert py is False
        assert nes is False

    def test_filled_equals_containers_no_beam(self):
        """filled == containers (not c-1), partial=$00 → no beams.

        This is the BUG-1 case: Python's old is_health_full returned True here."""
        py, nes = self._check((self.cm1 << 4) | (self.cm1 + 1), 0x00)
        assert py is False
        assert nes is False

    def test_filled_equals_containers_with_partial_no_beam(self):
        """filled == containers, partial=$80 → still no beams.

        Even with a 'full' partial heart, if filled == containers (not c-1),
        the NES assembly check fails because containers_minus_one != hearts_filled."""
        py, nes = self._check((self.cm1 << 4) | (self.cm1 + 1), 0x80)
        assert py is False
        assert nes is False

    def test_one_heart_short_no_beam(self):
        """filled == c-2, partial=$FF → no beams (one whole heart missing)."""
        py, nes = self._check((self.cm1 << 4) | (self.cm1 - 1), 0xFF)
        assert py is False
        assert nes is False

    def test_zero_filled_no_beam(self):
        """filled == 0, partial=$FF → no beams."""
        py, nes = self._check((self.cm1 << 4) | 0, 0xFF)
        assert py is False
        assert nes is False


# ---------------------------------------------------------------------------
# is_health_full (general-purpose, used by observation wrapper)
# ---------------------------------------------------------------------------

class TestIsHealthFull:
    """Verify is_health_full as a general 'at max health' check."""

    def test_full_health_is_full(self):
        emu = ZeldaFixture("debug_0_67_1772056964.state")
        _wait_for_gameplay(emu)
        game = emu.game_state()
        assert game.link.is_health_full is True
        emu.close()

    def test_missing_partial_not_full(self):
        """c-1 == filled but partial=$7F → not full."""
        emu = ZeldaFixture("debug_0_67_1772056964.state")
        _wait_for_gameplay(emu)
        _set_health_and_step(emu, 0x22, 0x7F)
        game = emu.game_state()
        assert game.link.is_health_full is False
        emu.close()

    def test_missing_heart_not_full(self):
        """filled < c-1 → not full regardless of partial."""
        emu = ZeldaFixture("debug_0_67_1772056964.state")
        _wait_for_gameplay(emu)
        _set_health_and_step(emu, 0x21, 0xFF)
        game = emu.game_state()
        assert game.link.is_health_full is False
        emu.close()


# ---------------------------------------------------------------------------
# Health getter/setter round-trip
# ---------------------------------------------------------------------------

class TestHealthRoundTrip:
    """Verify health setter produces correct RAM values."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.emu = ZeldaFixture("debug_0_67_1772056964.state")
        _wait_for_gameplay(self.emu)
        yield
        self.emu.close()

    @pytest.mark.parametrize("set_val,expected_min,expected_max", [
        (3.0, 2.99, 3.01),    # Full health (3 containers)
        (2.5, 2.49, 2.51),    # Half heart
        (2.0, 1.99, 2.01),    # Exact integer
        (1.0, 0.99, 1.01),    # One heart
        (0.5, 0.49, 0.51),    # Half heart
        (0.0, -0.01, 0.01),   # Zero health
    ])
    def test_setter_getter_roundtrip(self, set_val, expected_min, expected_max):
        """Setting health to X, then reading it back, returns ~X."""
        game = self.emu.game_state()
        game.link.health = set_val
        # Step to ensure values are applied
        self.emu.step()
        game2 = self.emu.game_state()
        result = game2.link.health
        assert expected_min < result < expected_max, \
            f"Set {set_val}, got {result}"

    def test_full_health_sets_beams_compatible(self):
        """Setting health to max_health produces the exact NES beam-firing state."""
        game = self.emu.game_state()
        max_h = game.link.max_health
        game.link.health = float(max_h)
        self.emu.step()

        # Check raw RAM matches assembly beam check
        hc = self.emu.ram[0x66F]
        partial = self.emu.ram[0x670]
        cm1 = (hc >> 4) & 0x0F
        hf = hc & 0x0F
        assert cm1 == hf, f"Expected c-1==filled, got c-1={cm1}, filled={hf}"
        assert partial >= 0x80, f"Expected partial >= $80, got {partial:#04x}"

    def test_16_hearts_special_case(self):
        """16 hearts (maximum) encodes as 0xFF with partial=$FF."""
        game = self.emu.game_state()
        game.link.max_health = 16
        game.link.health = 16.0
        self.emu.step()

        hc = self.emu.ram[0x66F]
        partial = self.emu.ram[0x670]
        assert hc == 0xFF, f"Expected 0xFF, got {hc:#04x}"
        assert partial == 0xFF, f"Expected partial=$FF, got {partial:#04x}"

        # Should fire beams (c-1=15 == filled=15, partial=$FF)
        game2 = self.emu.game_state()
        assert game2.link._is_health_full_for_beams
