"""Area 8: Look-ahead simulation tests.

Verifies that _predict_future_effects in StateChange correctly:
- Saves and restores emulator state without RAM corruption
- Isolates weapons during multi-weapon look-ahead
- Does not corrupt beam state across action boundaries
"""

import numpy as np
import pytest
from conftest import ZeldaFixture
from asm_addresses import *


class TestStateRestore:
    """Verify em.set_state() fully restores RAM after look-ahead simulation."""

    def test_state_restore_after_weapon_disable(self, emu):
        """Disabling weapon slots + stepping + restoring should produce identical RAM."""
        for _ in range(5):
            emu.step()

        ram_before = emu.ram.copy()
        state = emu.save()

        # Simulate what _predict_future_effects does
        emu.set('beam_animation', 0)
        emu.set('bomb_or_flame_animation', 0)
        emu.set('bomb_or_flame_animation2', 0)
        emu.set('bait_or_boomerang_animation', 0)
        emu.set('arrow_magic_animation', 0)
        emu.set('hearts_and_containers', 0xFF)

        for _ in range(10):
            emu.step()

        emu.restore(state)
        ram_after = emu.ram.copy()

        assert np.array_equal(ram_before, ram_after), \
            f"{np.sum(ram_before != ram_after)} RAM bytes differ after restore"

    def test_state_restore_during_active_beam(self, beam_emu):
        """Restoring state while beam is active preserves beam state."""
        emu = beam_emu
        beam_before = emu.ram[BEAM_STATE]
        assert beam_before == 0x10, "Expected beam to be active"

        state = emu.save()

        # Simulate look-ahead: disable beam and step
        emu.set('beam_animation', 0)
        for _ in range(5):
            emu.step()
        assert emu.ram[BEAM_STATE] == 0, "Beam should be disabled"

        emu.restore(state)
        assert emu.ram[BEAM_STATE] == beam_before, "Beam state not restored"

    def test_health_override_does_not_persist(self, emu):
        """Setting hearts_and_containers to 0xFF during look-ahead is undone by restore."""
        for _ in range(5):
            emu.step()

        original_health = emu.get('hearts_and_containers')
        state = emu.save()

        emu.set('hearts_and_containers', 0xFF)
        for _ in range(5):
            emu.step()

        emu.restore(state)
        assert emu.get('hearts_and_containers') == original_health


class TestMagicRodShot:
    """Verify magic rod shot lifecycle: $80 (flying) -> $00 (deactivated)."""

    def test_rod_shot_lifecycle(self, emu):
        """Rod shot should go $00 -> $80 -> $00, never reaching $81."""
        emu.set('magic_rod', 1)
        emu.set('selected_item', 8)  # WAND

        # Move inward to avoid screen lock
        for _ in range(10):
            emu.step([BTN_LEFT])
        emu.step()

        emu.step([BTN_B])  # fire rod

        # Track all beam states observed
        seen_states = set()
        for _ in range(80):
            emu.step()
            seen_states.add(emu.ram[BEAM_STATE])

        assert 0x80 in seen_states, "Rod shot should reach state $80 (flying)"
        assert 0x81 not in seen_states, "Rod shot should never reach state $81"
        assert 0x00 in seen_states, "Rod shot should deactivate to $00"

    def test_rod_with_book_spawns_fire(self, emu):
        """Rod + book of magic should spawn fire in bomb/fire slot on wall hit."""
        emu.set('magic_rod', 1)
        emu.set('selected_item', 8)
        emu.set('book', 1)

        for _ in range(10):
            emu.step([BTN_LEFT])
        emu.step()

        emu.step([BTN_B])

        fire_spawned = False
        for _ in range(100):
            emu.step()
            # Fire spawns in bomb/fire slot ($10 or $11) at state $22
            if emu.ram[OBJ_STATE + 0x10] == 0x22 or emu.ram[OBJ_STATE + 0x11] == 0x22:
                fire_spawned = True
                break

        assert fire_spawned, "Rod + book should spawn fire in bomb/fire slot"

    def test_rod_deals_damage(self):
        """Rod shot should reduce enemy HP on hit."""
        emu = ZeldaFixture("1_44e.state")
        try:
            emu.set('magic_rod', 1)
            emu.set('selected_item', 8)

            # Walk toward enemies (they're to the west)
            for _ in range(10):
                emu.step([BTN_LEFT])

            # Record enemy health before
            hp_before = [emu.ram[OBJ_HP + s] for s in range(1, 4)]

            emu.step([BTN_B])
            for _ in range(80):
                emu.step()

            hp_after = [emu.ram[OBJ_HP + s] for s in range(1, 4)]
            total_damage = sum(b - a for b, a in zip(hp_before, hp_after) if b > a)

            assert total_damage > 0, "Rod shot should deal damage to at least one enemy"
        finally:
            emu.close()


class TestBeamSpreadNotStuck:
    """Verify beam spread naturally deactivates — no hack needed."""

    def test_beam_spread_deactivates_naturally(self, beam_emu):
        """Beam at state $11 should return to $00 within 22 frames."""
        emu = beam_emu

        # Step until beam enters spread state
        for _ in range(60):
            emu.step()
            if emu.ram[BEAM_STATE] == 0x11:
                break

        assert emu.ram[BEAM_STATE] == 0x11, "Beam never entered spread state"

        # Count how long it stays at $11
        spread_frames = 0
        for _ in range(30):
            emu.step()
            if emu.ram[BEAM_STATE] == 0x11:
                spread_frames += 1
            else:
                break

        assert emu.ram[BEAM_STATE] == 0x00, "Beam should deactivate after spreading"
        assert spread_frames == 21, f"Expected 21 more frames at $11, got {spread_frames}"

    def test_beam_not_stuck_across_attacks(self, emu):
        """Rapidly firing sword should not cause beam to get stuck at state $11."""
        # Move to center
        for _ in range(5):
            emu.step()

        for attack in range(4):
            emu.step([BTN_A])
            for _ in range(60):
                emu.step()
                if emu.get('link_status') == 0 and emu.get('sword_animation') == 0:
                    break

        # After 4 rapid attacks, beam should not be permanently stuck
        # Step enough frames for any spread to finish
        for _ in range(50):
            emu.step()

        assert emu.ram[BEAM_STATE] == 0x00, \
            f"Beam stuck at 0x{emu.ram[BEAM_STATE]:02X} after rapid attacks"


@pytest.fixture
def emu():
    """Overworld room with full health, centered Link."""
    fixture = ZeldaFixture("debug_0_67_1772056964.state")
    yield fixture
    fixture.close()


@pytest.fixture
def beam_emu():
    """Fixture with beams already active (state $10)."""
    fixture = ZeldaFixture("debug_0_67_1772056964.state")
    for _ in range(5):
        fixture.step()

    fixture.step([BTN_A])

    for _ in range(20):
        fixture.step()
        if fixture.ram[BEAM_STATE] == 0x10:
            break

    assert fixture.ram[BEAM_STATE] == 0x10, "Failed to activate beams"
    yield fixture
    fixture.close()
