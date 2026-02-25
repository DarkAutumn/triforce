"""Tests for frame skip animation-state polling (Area 12).

Verifies that the animation cooldown polling in frame_skip_wrapper correctly
waits for link_status==0 and sword_animation==0 before returning control,
and that the next action fires on the first possible NES frame.
"""

import pytest
import numpy as np
from tests.conftest import ZeldaFixture
from tests.asm_addresses import OBJ_STATE, BTN_A, BTN_B

LINK_STATUS = OBJ_STATE       # ObjState[0]
SWORD_ANIMATION = OBJ_STATE + 0x0D  # ObjState[SLOT_SWORD]
BEAM_ANIMATION = OBJ_STATE + 0x0E   # ObjState[SLOT_BEAM]


@pytest.fixture
def emu():
    """Emulator fixture with debug savestate (full health, center of room)."""
    fix = ZeldaFixture("debug_0_67_1772056964.state")
    yield fix
    fix.close()


class TestSwordCooldownTiming:
    """Verify sword melee link_status and sword_animation timing."""

    def test_link_status_nonzero_during_swing(self, emu):
        """link_status should be non-zero throughout the sword swing animation."""
        emu.set('hearts_and_containers', 0x20)  # low health, no beams
        emu.set('partial_hearts', 0x00)

        emu.step([BTN_A])
        assert emu.ram[LINK_STATUS] != 0, "link_status should be non-zero immediately after A press"

        # Should stay non-zero for several frames
        for _ in range(8):
            emu.step()
            assert emu.ram[LINK_STATUS] != 0

    def test_link_controllable_when_both_zero(self, emu):
        """Link can re-swing only when both link_status==0 AND sword_animation==0."""
        emu.set('hearts_and_containers', 0x20)
        emu.set('partial_hearts', 0x00)

        emu.step([BTN_A])

        # Poll until both are zero
        frames = 0
        for _ in range(60):
            emu.step()
            frames += 1
            if emu.ram[LINK_STATUS] == 0 and emu.ram[SWORD_ANIMATION] == 0:
                break

        assert frames == 15, f"Expected both zero at frame 15, got {frames}"

    def test_sword_refires_on_first_controllable_frame(self, emu):
        """Pressing A immediately after both are zero should fire a new sword."""
        emu.set('hearts_and_containers', 0x20)
        emu.set('partial_hearts', 0x00)

        emu.step([BTN_A])

        # Wait until controllable
        for _ in range(60):
            emu.step()
            if emu.ram[LINK_STATUS] == 0 and emu.ram[SWORD_ANIMATION] == 0:
                break

        # Immediate re-swing
        emu.step([BTN_A])
        emu.step()
        assert emu.ram[SWORD_ANIMATION] != 0, "Sword should fire on first controllable frame"
        assert emu.ram[LINK_STATUS] != 0, "link_status should be non-zero after new swing"

    def test_sword_blocked_one_frame_before_controllable(self, emu):
        """Pressing A one frame before controllable should NOT fire a new sword."""
        emu.set('hearts_and_containers', 0x20)
        emu.set('partial_hearts', 0x00)

        emu.step([BTN_A])

        # Wait until 1 frame BEFORE controllable (frame 14)
        for _ in range(14):
            emu.step()

        # At frame 14: link_status should be 0 but sword_animation should still be non-zero
        assert emu.ram[LINK_STATUS] == 0, "link_status should be 0 at frame 14"
        assert emu.ram[SWORD_ANIMATION] != 0, "sword_animation should still be non-zero at frame 14"

        # Press A — should NOT fire (sword_animation gate)
        emu.step([BTN_A])
        emu.step()
        # If sword fired, sword_animation would restart at 0x01
        assert emu.ram[SWORD_ANIMATION] == 0, "Sword should NOT fire while sword_animation != 0"


class TestItemCooldownTiming:
    """Verify item (bomb, boomerang) animation timing."""

    def test_bomb_link_controllable_at_frame_12(self, emu):
        """After placing a bomb, link_status returns to 0 at frame 12."""
        emu.set('bombs', 8)
        emu.set('selected_item', 1)

        emu.step([BTN_B])

        frames = 0
        for _ in range(60):
            emu.step()
            frames += 1
            if emu.ram[LINK_STATUS] == 0 and emu.ram[SWORD_ANIMATION] == 0:
                break

        assert frames == 12, f"Expected controllable at frame 12, got {frames}"

    def test_sword_fires_immediately_after_bomb(self, emu):
        """Pressing A immediately after bomb animation ends should fire sword."""
        emu.set('bombs', 8)
        emu.set('selected_item', 1)
        emu.set('hearts_and_containers', 0x20)
        emu.set('partial_hearts', 0x00)

        emu.step([BTN_B])

        for _ in range(60):
            emu.step()
            if emu.ram[LINK_STATUS] == 0 and emu.ram[SWORD_ANIMATION] == 0:
                break

        emu.step([BTN_A])
        emu.step()
        assert emu.ram[SWORD_ANIMATION] != 0, "Sword should fire on first frame after bomb"

    def test_boomerang_link_controllable_at_frame_12(self, emu):
        """After throwing boomerang, link_status returns to 0 at frame 12."""
        emu.set('regular_boomerang', 1)
        emu.set('selected_item', 0)

        emu.step([BTN_B])

        frames = 0
        for _ in range(60):
            emu.step()
            frames += 1
            if emu.ram[LINK_STATUS] == 0 and emu.ram[SWORD_ANIMATION] == 0:
                break

        assert frames == 12, f"Expected controllable at frame 12, got {frames}"

    def test_sword_fires_immediately_after_boomerang(self, emu):
        """Pressing A immediately after boomerang animation ends should fire sword."""
        emu.set('regular_boomerang', 1)
        emu.set('selected_item', 0)
        emu.set('hearts_and_containers', 0x20)
        emu.set('partial_hearts', 0x00)

        emu.step([BTN_B])

        for _ in range(60):
            emu.step()
            if emu.ram[LINK_STATUS] == 0 and emu.ram[SWORD_ANIMATION] == 0:
                break

        emu.step([BTN_A])
        emu.step()
        assert emu.ram[SWORD_ANIMATION] != 0, "Sword should fire on first frame after boomerang"


class TestBeamCooldownTiming:
    """Verify sword+beam timing matches pure melee timing."""

    def test_beam_sword_controllable_at_frame_15(self, emu):
        """With beams, link_status==0 AND sword_animation==0 at same frame as no-beams."""
        emu.set('hearts_and_containers', 0x22)  # full health
        emu.set('partial_hearts', 0xFF)

        emu.step([BTN_A])

        frames = 0
        for _ in range(60):
            emu.step()
            frames += 1
            if emu.ram[LINK_STATUS] == 0 and emu.ram[SWORD_ANIMATION] == 0:
                break

        assert frames == 15, f"Expected controllable at frame 15 with beams, got {frames}"

    def test_beam_fires_and_link_can_reswing(self, emu):
        """After beam fires, Link can immediately re-swing once controllable."""
        emu.set('hearts_and_containers', 0x22)
        emu.set('partial_hearts', 0xFF)

        emu.step([BTN_A])

        for _ in range(60):
            emu.step()
            if emu.ram[LINK_STATUS] == 0 and emu.ram[SWORD_ANIMATION] == 0:
                break

        # Beam should still be active (it lasts much longer than the sword animation)
        assert emu.ram[BEAM_ANIMATION] != 0, "Beam should still be active after sword cooldown"

        emu.step([BTN_A])
        emu.step()
        assert emu.ram[SWORD_ANIMATION] != 0, "Sword should re-fire while beam is still active"
