# pylint: disable=all
"""Layer 2: Verify beam/sword shot state machine against NES assembly.

Tests verify ObjState lifecycle for sword melee (slot $0D) and beam (slot $0E),
health requirements for beam firing, and Python constant correctness.

Requires debug_0_67_1772056964.state: overworld room $67, Link centered,
sword equipped, full health. A button fires sword (retro maps sword to A).
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from conftest import ZeldaFixture, RAMWatcher
from asm_addresses import *

BEAM_STATE_FILE = "debug_0_67_1772056964.state"


def _fire_beam(emu, idle_frames=10):
    """Set up and fire a beam. Returns frame number when beam first becomes 0x10."""
    for _ in range(idle_frames):
        emu.step()
    emu.step([BTN_A])
    # Step until beam activates
    for i in range(20):
        emu.step()
        if emu.ram[BEAM_STATE] == 0x10:
            return i
    return None


@pytest.fixture
def beam_emu():
    """Provide a ZeldaFixture with sword and full health, centered in room."""
    fixture = ZeldaFixture(BEAM_STATE_FILE)
    yield fixture
    fixture.close()


# --- T3.1: Beam ObjState lifecycle ---

class TestBeamLifecycle:
    """Verify beam slot ($0E) goes 0x00 → 0x10 → 0x11 → 0x00."""

    def test_beam_lifecycle_states(self, beam_emu):
        """Beam transitions through exactly INACTIVE → ACTIVE → HIT → INACTIVE."""
        _fire_beam(beam_emu)

        states_seen = {0x10}
        prev = 0x10
        for _ in range(80):
            beam_emu.step()
            bm = beam_emu.ram[BEAM_STATE]
            if bm != prev:
                states_seen.add(bm)
                prev = bm
            if bm == 0 and 0x11 in states_seen:
                break

        assert states_seen == {0x10, 0x11, 0x00}, f"Unexpected beam states: {states_seen}"

    def test_beam_active_is_0x10(self, beam_emu):
        """ANIMATION_BEAMS_ACTIVE = 16 (0x10) matches NES."""
        _fire_beam(beam_emu)
        assert beam_emu.ram[BEAM_STATE] == 0x10

    def test_beam_hit_is_0x11(self, beam_emu):
        """ANIMATION_BEAMS_HIT = 17 (0x11) matches NES."""
        _fire_beam(beam_emu)
        # Wait for transition to hit/spread
        for _ in range(40):
            beam_emu.step()
            if beam_emu.ram[BEAM_STATE] == 0x11:
                break
        assert beam_emu.ram[BEAM_STATE] == 0x11

    def test_beam_spread_duration_22_frames(self, beam_emu):
        """Spread phase (0x11) lasts exactly 22 frames (ObjDir $FE → $E8)."""
        _fire_beam(beam_emu)

        # Wait for spread phase
        for _ in range(40):
            beam_emu.step()
            if beam_emu.ram[BEAM_STATE] == 0x11:
                break

        assert beam_emu.ram[BEAM_STATE] == 0x11, "Never reached spread phase"

        # Count spread frames
        spread_frames = 0
        while beam_emu.ram[BEAM_STATE] == 0x11:
            spread_frames += 1
            beam_emu.step()
            if spread_frames > 30:
                break

        assert spread_frames == 22, f"Spread lasted {spread_frames} frames, expected 22"

    def test_beam_spread_uses_objdir_counter(self, beam_emu):
        """During spread, ObjDir[BEAM] decrements from $FE to $E8."""
        _fire_beam(beam_emu)

        # Wait for spread
        for _ in range(40):
            beam_emu.step()
            if beam_emu.ram[BEAM_STATE] == 0x11:
                break

        first_dir = beam_emu.ram[OBJ_DIR + SLOT_BEAM]
        assert first_dir == 0xFE, f"Spread starts with ObjDir=0x{first_dir:02X}, expected 0xFE"

        # Step through spread
        dirs = [first_dir]
        while beam_emu.ram[BEAM_STATE] == 0x11:
            beam_emu.step()
            dirs.append(beam_emu.ram[OBJ_DIR + SLOT_BEAM])

        # Should decrement by 1 each frame
        for i in range(1, len(dirs) - 1):
            assert dirs[i] == dirs[i-1] - 1, f"ObjDir not decrementing at frame {i}"


# --- T3.2: Sword melee state machine ---

class TestSwordMelee:
    """Verify sword slot ($0D) state sequence during melee swing."""

    def test_sword_state_sequence(self, beam_emu):
        """Sword melee goes through states 1→2→3→4→5→0."""
        for _ in range(10):
            beam_emu.step()

        beam_emu.step([BTN_A])
        states = [beam_emu.ram[SWORD_STATE]]

        for _ in range(20):
            beam_emu.step()
            s = beam_emu.ram[SWORD_STATE]
            if s != states[-1]:
                states.append(s)

        assert states == [0x01, 0x02, 0x03, 0x04, 0x05, 0x00], \
            f"Sword states: {[f'0x{s:02X}' for s in states]}"

    def test_sword_state2_duration_8_frames(self, beam_emu):
        """Sword state 2 lasts 8 frames per assembly ObjAnimCounter."""
        for _ in range(10):
            beam_emu.step()

        beam_emu.step([BTN_A])

        # Wait for state 2
        while beam_emu.ram[SWORD_STATE] != 0x02:
            beam_emu.step()

        frames = 0
        while beam_emu.ram[SWORD_STATE] == 0x02:
            frames += 1
            beam_emu.step()

        assert frames == 8, f"State 2 lasted {frames} frames, expected 8"

    def test_beam_fires_at_sword_state3(self, beam_emu):
        """MakeSwordShot fires beam when sword reaches state 3."""
        for _ in range(10):
            beam_emu.step()

        beam_emu.step([BTN_A])

        # Track when beam first appears
        beam_sword_state = None
        for _ in range(20):
            beam_emu.step()
            if beam_emu.ram[BEAM_STATE] == 0x10 and beam_sword_state is None:
                beam_sword_state = beam_emu.ram[SWORD_STATE]
                break

        assert beam_sword_state == 0x03, \
            f"Beam appeared at sword state 0x{beam_sword_state:02X}, expected 0x03"


# --- T3.3: Health check for beams ---

class TestBeamHealthCheck:
    """Verify beam firing requires filled == containers-1 AND partial >= 0x80."""

    def test_full_health_fires_beam(self, beam_emu):
        """containers-1 == filled, partial=0xFF → beam fires."""
        for _ in range(10):
            beam_emu.step()

        beam_emu.set('hearts_and_containers', 0x22)
        beam_emu.set('partial_hearts', 0xFF)
        beam_emu.step()

        beam_emu.step([BTN_A])
        for _ in range(20):
            beam_emu.step()
            if beam_emu.ram[BEAM_STATE] == 0x10:
                return  # pass
        pytest.fail("Beam didn't fire with full health")

    def test_partial_0x80_fires_beam(self, beam_emu):
        """Partial heart exactly 0x80 (half full) → beam fires."""
        for _ in range(10):
            beam_emu.step()

        beam_emu.set('hearts_and_containers', 0x22)
        beam_emu.set('partial_hearts', 0x80)
        beam_emu.step()

        beam_emu.step([BTN_A])
        for _ in range(20):
            beam_emu.step()
            if beam_emu.ram[BEAM_STATE] == 0x10:
                return  # pass
        pytest.fail("Beam didn't fire with partial=0x80")

    def test_partial_0x7F_no_beam(self, beam_emu):
        """Partial heart 0x7F (just below threshold) → no beam."""
        for _ in range(10):
            beam_emu.step()

        beam_emu.set('hearts_and_containers', 0x22)
        beam_emu.set('partial_hearts', 0x7F)
        beam_emu.step()

        beam_emu.step([BTN_A])
        for _ in range(20):
            beam_emu.step()
            if beam_emu.ram[BEAM_STATE] == 0x10:
                pytest.fail("Beam fired with partial=0x7F (should not)")

    def test_not_full_hearts_no_beam(self, beam_emu):
        """filled < containers-1 → no beam even with partial=0xFF."""
        for _ in range(10):
            beam_emu.step()

        beam_emu.set('hearts_and_containers', 0x21)  # containers-1=2, filled=1
        beam_emu.set('partial_hearts', 0xFF)
        beam_emu.step()

        beam_emu.step([BTN_A])
        for _ in range(20):
            beam_emu.step()
            if beam_emu.ram[BEAM_STATE] == 0x10:
                pytest.fail("Beam fired with filled < containers-1")

    def test_filled_equals_containers_no_beam(self, beam_emu):
        """BUG: filled == containers (not containers-1) → NES says NO beams.
        Assembly checks filled == containers_minus_one, not filled == containers.
        Python's is_health_full returns True here (health=3.0, max=3) which is wrong."""
        for _ in range(10):
            beam_emu.step()

        beam_emu.set('hearts_and_containers', 0x23)  # containers-1=2, filled=3
        beam_emu.set('partial_hearts', 0x00)
        beam_emu.step()

        beam_emu.step([BTN_A])
        for _ in range(20):
            beam_emu.step()
            if beam_emu.ram[BEAM_STATE] == 0x10:
                pytest.fail("Beam fired with filled==containers (NES should reject)")

    def test_filled_equals_containers_partial_full_no_beam(self, beam_emu):
        """filled == containers with partial=0x80 → still NO beams in NES."""
        for _ in range(10):
            beam_emu.step()

        beam_emu.set('hearts_and_containers', 0x23)  # containers-1=2, filled=3
        beam_emu.set('partial_hearts', 0x80)
        beam_emu.step()

        beam_emu.step([BTN_A])
        for _ in range(20):
            beam_emu.step()
            if beam_emu.ram[BEAM_STATE] == 0x10:
                pytest.fail("Beam fired with filled==containers even with partial=0x80")


# --- T3.8: Python constant verification ---

class TestBeamConstants:
    """Verify Python animation constants match NES values."""

    def test_animation_beams_active_is_16(self):
        """ANIMATION_BEAMS_ACTIVE should be 0x10 (16)."""
        from triforce.link import ANIMATION_BEAMS_ACTIVE
        assert ANIMATION_BEAMS_ACTIVE == 0x10

    def test_animation_beams_hit_is_17(self):
        """ANIMATION_BEAMS_HIT should be 0x11 (17)."""
        from triforce.link import ANIMATION_BEAMS_HIT
        assert ANIMATION_BEAMS_HIT == 0x11


# --- T3.9: Beam spread duration ---

class TestBeamSpreadDuration:
    """Verify beam spread phase lasts 22 frames.

    The beam spread phase (state $11) naturally lasts 22 frames as the ObjDir
    counter decrements from $FE to $E8. A previous 11-frame hack incorrectly
    reset beam_animation in the info dict mid-spread. That hack has been removed
    since the beam deactivates naturally.
    """

    def test_spread_duration_exceeds_sword_cooldown(self, beam_emu):
        """The natural spread (22 frames) exceeds sword cooldown (~15 frames).
        This means beams can still be spreading when the agent regains control."""
        _fire_beam(beam_emu)

        # Wait for spread
        for _ in range(40):
            beam_emu.step()
            if beam_emu.ram[BEAM_STATE] == 0x11:
                break

        spread_frames = 0
        while beam_emu.ram[BEAM_STATE] == 0x11:
            spread_frames += 1
            beam_emu.step()

        assert spread_frames > 15, \
            f"Spread is {spread_frames} frames, must exceed sword cooldown (~15)"
