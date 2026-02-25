"""Area 3: Screen lock boundary tests.

Verifies that is_sword_screen_locked and get_sword_directions_allowed match
the NES assembly's BorderBounds table (Z_05.asm line 2728).

Assembly outer bounds:
  OW: left=$07, right=$E9, up=$45, down=$D6
  UW: left=$17, right=$D9, up=$55, down=$C6
"""

import pytest
from tests.conftest import ZeldaFixture
from tests.asm_addresses import BTN_A, BTN_UP, BTN_DOWN, BTN_LEFT, BTN_RIGHT, SWORD_STATE
from triforce.zelda_enums import Direction, Position


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _precise_sword_test(emu):
    """Test if sword fires at current position. Returns True if sword fires."""
    save = emu.save()
    emu.step()           # idle frame to flush buffered input
    emu.step([BTN_A])
    fired = emu.ram[SWORD_STATE] != 0
    emu.restore(save)
    return fired


def _walk_and_find_boundary(emu, direction_btn, coord_name, max_steps=60):
    """Walk in a direction and return (last_blocked_coord, first_allowed_coord).

    Starts from a position where sword is blocked and walks until it fires.
    """
    last_blocked = None
    for _ in range(max_steps):
        coord = emu.get(coord_name)
        fired = _precise_sword_test(emu)
        if fired:
            return last_blocked, coord
        last_blocked = coord
        emu.step([direction_btn])
    return last_blocked, None


def _wait_for_gameplay(emu, max_frames=100):
    """Wait for mode 5 (gameplay) after screen transitions."""
    for _ in range(max_frames):
        emu.step()
        if emu.get('mode') == 5:
            return
    raise RuntimeError("Never reached gameplay mode")


# ---------------------------------------------------------------------------
# UW boundary tests (empirical)
# ---------------------------------------------------------------------------

class TestUWBoundaries:
    """Test underworld screen lock boundaries by walking Link to edges."""

    def test_uw_right_boundary(self):
        """UW right boundary: sword blocked at X >= $D9."""
        # East entry starts in blocked zone, walk left to find transition
        emu = ZeldaFixture("1_44e.state")
        _wait_for_gameplay(emu)

        last_blocked, first_allowed = _walk_and_find_boundary(
            emu, BTN_LEFT, 'link_x')
        emu.close()

        assert first_allowed is not None, "Never found sword-allowed position"
        assert first_allowed <= 0xD8, f"Expected sword to fire at X <= $D8, got X={first_allowed:#04x}"
        assert last_blocked >= 0xD9, f"Expected sword blocked at X >= $D9, got X={last_blocked:#04x}"

    def test_uw_left_boundary(self):
        """UW left boundary: sword blocked at X < $17."""
        emu = ZeldaFixture("1_36w.state")
        _wait_for_gameplay(emu)

        last_blocked, first_allowed = _walk_and_find_boundary(
            emu, BTN_RIGHT, 'link_x')
        emu.close()

        assert first_allowed is not None, "Never found sword-allowed position"
        # Assembly outer left bound is $17: x < $17 blocked, x >= $17 allowed
        assert last_blocked <= 0x16, f"Expected blocked at X <= $16, got X={last_blocked:#04x}"
        assert first_allowed >= 0x17, f"Expected allowed at X >= $17, got X={first_allowed:#04x}"

    def test_uw_south_boundary(self):
        """UW south boundary: sword blocked at Y >= $C6."""
        emu = ZeldaFixture("1_23s.state")
        _wait_for_gameplay(emu)

        last_blocked, first_allowed = _walk_and_find_boundary(
            emu, BTN_UP, 'link_y')
        emu.close()

        assert first_allowed is not None, "Never found sword-allowed position"
        # Assembly outer down bound is $C6: y >= $C6 blocked, y < $C6 (y <= $C5) allowed
        assert last_blocked >= 0xC6, f"Expected blocked at Y >= $C6, got Y={last_blocked:#04x}"
        assert first_allowed <= 0xC5, f"Expected allowed at Y <= $C5, got Y={first_allowed:#04x}"

    def test_uw_north_boundary(self):
        """UW north boundary: sword blocked at Y < $55."""
        emu = ZeldaFixture("1_33n.state")
        _wait_for_gameplay(emu)

        last_blocked, first_allowed = _walk_and_find_boundary(
            emu, BTN_DOWN, 'link_y')
        emu.close()

        assert first_allowed is not None, "Never found sword-allowed position"
        # Assembly outer up bound is $55: y < $55 blocked, y >= $55 allowed
        assert last_blocked <= 0x54, f"Expected blocked at Y <= $54, got Y={last_blocked:#04x}"
        assert first_allowed >= 0x55, f"Expected allowed at Y >= $55, got Y={first_allowed:#04x}"


# ---------------------------------------------------------------------------
# OW boundary tests (empirical)
# ---------------------------------------------------------------------------

class TestOWBoundaries:
    """Test overworld screen lock boundaries."""

    def test_ow_right_boundary(self):
        """OW right boundary: sword blocked at X >= $E9."""
        emu = ZeldaFixture("debug_0_67_1772056964.state")
        _wait_for_gameplay(emu)

        # Walk east until sword is blocked
        prev_fired = True
        last_allowed = None
        first_blocked = None
        for _ in range(120):
            x = emu.get('link_x')
            fired = _precise_sword_test(emu)
            if prev_fired and not fired:
                first_blocked = x
                break
            if fired:
                last_allowed = x
            prev_fired = fired
            emu.step([BTN_RIGHT])
        emu.close()

        assert first_blocked is not None, "Never found sword-blocked position"
        assert first_blocked >= 0xE9, f"Expected blocked at X >= $E9, got X={first_blocked:#04x}"
        assert last_allowed is not None and last_allowed <= 0xE8, \
            f"Expected allowed at X <= $E8, got X={last_allowed:#04x}"


# ---------------------------------------------------------------------------
# Python model consistency tests (unit-style)
# ---------------------------------------------------------------------------

class TestBoundaryConstants:
    """Verify Python boundary values match assembly BorderBounds table."""

    def test_ow_is_sword_screen_locked(self):
        """OW lock boundaries match assembly outer OW bounds."""
        emu = ZeldaFixture("debug_0_67_1772056964.state")
        _wait_for_gameplay(emu)
        game = emu.game_state()
        link = game.link
        assert game.level == 0

        y_center = 0x80

        # Left bound: $07
        link.position = Position(0x06, y_center)
        assert link.is_sword_screen_locked
        link.position = Position(0x07, y_center)
        assert not link.is_sword_screen_locked

        # Right bound: $E9
        link.position = Position(0xE8, y_center)
        assert not link.is_sword_screen_locked
        link.position = Position(0xE9, y_center)
        assert link.is_sword_screen_locked

        # Up bound: $45
        link.position = Position(0x78, 0x44)
        assert link.is_sword_screen_locked
        link.position = Position(0x78, 0x45)
        assert not link.is_sword_screen_locked

        # Down bound: $D6
        link.position = Position(0x78, 0xD5)
        assert not link.is_sword_screen_locked
        link.position = Position(0x78, 0xD6)
        assert link.is_sword_screen_locked
        emu.close()

    def test_uw_is_sword_screen_locked(self):
        """UW lock boundaries match assembly outer UW bounds."""
        emu = ZeldaFixture("1_44e.state")
        _wait_for_gameplay(emu)
        game = emu.game_state()
        link = game.link
        assert game.level > 0

        y_center = 0x80

        # Left bound: $17
        link.position = Position(0x16, y_center)
        assert link.is_sword_screen_locked
        link.position = Position(0x17, y_center)
        assert not link.is_sword_screen_locked

        # Right bound: $D9
        link.position = Position(0xD8, y_center)
        assert not link.is_sword_screen_locked
        link.position = Position(0xD9, y_center)
        assert link.is_sword_screen_locked

        # Up bound: $55
        link.position = Position(0x78, 0x54)
        assert link.is_sword_screen_locked
        link.position = Position(0x78, 0x55)
        assert not link.is_sword_screen_locked

        # Down bound: $C6
        link.position = Position(0x78, 0xC5)
        assert not link.is_sword_screen_locked
        link.position = Position(0x78, 0xC6)
        assert link.is_sword_screen_locked
        emu.close()

    def test_ow_sword_directions_allowed(self):
        """OW directions match assembly outer OW bounds."""
        emu = ZeldaFixture("debug_0_67_1772056964.state")
        _wait_for_gameplay(emu)
        game = emu.game_state()
        link = game.link

        # Center: all directions
        link.position = Position(0x78, 0x80)
        dirs = link.get_sword_directions_allowed()
        assert set(dirs) == {Direction.E, Direction.W, Direction.N, Direction.S}

        # At X=$07 (left edge, but still in bounds): E/W allowed
        link.position = Position(0x07, 0x80)
        dirs = link.get_sword_directions_allowed()
        assert Direction.E in dirs and Direction.W in dirs

        # At X=$06: E/W NOT allowed
        link.position = Position(0x06, 0x80)
        dirs = link.get_sword_directions_allowed()
        assert Direction.E not in dirs and Direction.W not in dirs

        # At X=$E8 (right edge, still in bounds): E/W allowed
        link.position = Position(0xE8, 0x80)
        dirs = link.get_sword_directions_allowed()
        assert Direction.E in dirs and Direction.W in dirs

        # At X=$E9: E/W NOT allowed
        link.position = Position(0xE9, 0x80)
        dirs = link.get_sword_directions_allowed()
        assert Direction.E not in dirs and Direction.W not in dirs

        # At Y=$45 (top edge, still in bounds): N/S allowed
        link.position = Position(0x78, 0x45)
        dirs = link.get_sword_directions_allowed()
        assert Direction.N in dirs and Direction.S in dirs

        # At Y=$44: N/S NOT allowed
        link.position = Position(0x78, 0x44)
        dirs = link.get_sword_directions_allowed()
        assert Direction.N not in dirs and Direction.S not in dirs

        # At Y=$D5 (bottom edge, still in bounds): N/S allowed
        link.position = Position(0x78, 0xD5)
        dirs = link.get_sword_directions_allowed()
        assert Direction.N in dirs and Direction.S in dirs

        # At Y=$D6: N/S NOT allowed
        link.position = Position(0x78, 0xD6)
        dirs = link.get_sword_directions_allowed()
        assert Direction.N not in dirs and Direction.S not in dirs
        emu.close()

    def test_uw_sword_directions_allowed(self):
        """UW directions match assembly outer UW bounds."""
        emu = ZeldaFixture("1_44e.state")
        _wait_for_gameplay(emu)
        game = emu.game_state()
        link = game.link

        # Center: all directions
        link.position = Position(0x78, 0x80)
        dirs = link.get_sword_directions_allowed()
        assert set(dirs) == {Direction.E, Direction.W, Direction.N, Direction.S}

        # Left bound: $17
        link.position = Position(0x17, 0x80)
        dirs = link.get_sword_directions_allowed()
        assert Direction.E in dirs and Direction.W in dirs
        link.position = Position(0x16, 0x80)
        dirs = link.get_sword_directions_allowed()
        assert Direction.E not in dirs and Direction.W not in dirs

        # Right bound: $D9
        link.position = Position(0xD8, 0x80)
        dirs = link.get_sword_directions_allowed()
        assert Direction.E in dirs and Direction.W in dirs
        link.position = Position(0xD9, 0x80)
        dirs = link.get_sword_directions_allowed()
        assert Direction.E not in dirs and Direction.W not in dirs

        # Up bound: $55
        link.position = Position(0x78, 0x55)
        dirs = link.get_sword_directions_allowed()
        assert Direction.N in dirs and Direction.S in dirs
        link.position = Position(0x78, 0x54)
        dirs = link.get_sword_directions_allowed()
        assert Direction.N not in dirs and Direction.S not in dirs

        # Down bound: $C6
        link.position = Position(0x78, 0xC5)
        dirs = link.get_sword_directions_allowed()
        assert Direction.N in dirs and Direction.S in dirs
        link.position = Position(0x78, 0xC6)
        dirs = link.get_sword_directions_allowed()
        assert Direction.N not in dirs and Direction.S not in dirs
        emu.close()
