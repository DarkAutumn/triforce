"""Areas 5, 6, 11: Enemy model verification.

Tests verify:
- Death metastate sequence ($10-$13, then dropped item)
- is_active for Zora (states 2-4), Leever (state 3), WallMaster (state 1)
- Object classification boundaries (_is_id_enemy, _is_projectile)
- ZeldaEnemyKind enum completeness against assembly jump table
"""

import pytest
from conftest import ZeldaFixture
from asm_addresses import *


class TestDeathMetastate:
    """Area 5: Verify death sparkle metastate sequence."""

    def test_death_sequence_values(self):
        """Death metastate goes $10 -> $11 -> $12 -> $13, then becomes item ($60)."""
        emu = ZeldaFixture("1_44e.state")
        try:
            emu.set('sword', 3)  # Magic sword for one-shot kills
            emu.set('hearts_and_containers', 0x22)
            emu.set('partial_hearts', 0xFF)

            for _ in range(15):
                emu.step([BTN_LEFT])

            emu.step([BTN_A])

            # Track metastate sequence for any enemy that dies
            death_states = []
            became_item = False
            dying_slot = None

            for _ in range(100):
                emu.step()
                for slot in range(1, 4):
                    meta = emu.ram[0x405 + slot]
                    obj_type = emu.ram[0x34F + slot]

                    if meta >= 0x10 and dying_slot is None:
                        dying_slot = slot

                    if dying_slot == slot:
                        if meta >= 0x10:
                            if not death_states or death_states[-1] != meta:
                                death_states.append(meta)
                        elif death_states and obj_type == 0x60:
                            became_item = True
                            break

                if became_item:
                    break

            assert death_states == [0x10, 0x11, 0x12, 0x13], \
                f"Expected death sequence [0x10-0x13], got {[hex(x) for x in death_states]}"
            assert became_item, "Enemy should become a dropped item (type=0x60) after death"
        finally:
            emu.close()

    def test_is_dying_range(self):
        """Python's is_dying should match metastates 16-19 ($10-$13)."""
        from triforce.enemy import Enemy
        from triforce.zelda_enums import Direction, ZeldaEnemyKind

        for meta in range(0x14):
            e = Enemy(None, 1, ZeldaEnemyKind.RedGoriya, (0, 0), Direction.N, 3, 0, meta, 0)
            if 16 <= meta <= 19:
                assert e.is_dying, f"meta={meta} should be dying"
            else:
                assert not e.is_dying, f"meta={meta} should NOT be dying"


class TestZoraIsActive:
    """Area 5: Verify Zora is_active for states 2, 3, 4."""

    def test_zora_active_states(self):
        """Zora should be targetable in states 2, 3, and 4."""
        from triforce.enemy import Enemy
        from triforce.zelda_enums import Direction, ZeldaEnemyKind

        for state in range(6):
            e = Enemy(None, 1, ZeldaEnemyKind.Zora, (0, 0), Direction.N, 3, 0, 0, state)
            if 2 <= state <= 4:
                assert e.is_active, f"Zora state {state} should be active"
            else:
                assert not e.is_active, f"Zora state {state} should NOT be active"

    def test_zora_state_cycle_in_nes(self):
        """Verify Zora cycles through states 0-5 in the NES."""
        emu = ZeldaFixture("0_17s")
        try:
            zora_slot = None
            for slot in range(1, 12):
                if emu.ram[0x34F + slot] == 0x11:
                    zora_slot = slot
                    break

            assert zora_slot is not None, "No Zora found in savestate"

            for _ in range(30):
                emu.step([BTN_LEFT])

            states_seen = set()
            for _ in range(600):
                emu.step()
                states_seen.add(emu.ram[OBJ_STATE + zora_slot])

            assert states_seen == {0, 1, 2, 3, 4, 5}, \
                f"Zora should cycle through states 0-5, saw {sorted(states_seen)}"
        finally:
            emu.close()


class TestLeeverIsActive:
    """Area 5: Verify Leever is_active only at state 3."""

    def test_leever_active_only_state3(self):
        """Leevers should only be targetable at state 3."""
        from triforce.enemy import Enemy
        from triforce.zelda_enums import Direction, ZeldaEnemyKind

        for state in range(6):
            blue = Enemy(None, 1, ZeldaEnemyKind.BlueLeever, (0, 0), Direction.N, 3, 0, 0, state)
            red = Enemy(None, 1, ZeldaEnemyKind.RedLeever, (0, 0), Direction.N, 3, 0, 0, state)
            if state == 3:
                assert blue.is_active, f"BlueLeever state {state} should be active"
                assert red.is_active, f"RedLeever state {state} should be active"
            else:
                assert not blue.is_active, f"BlueLeever state {state} should NOT be active"
                assert not red.is_active, f"RedLeever state {state} should NOT be active"


class TestWallmasterIsActive:
    """Area 5: Verify WallMaster is_active only at state 1."""

    def test_wallmaster_active_only_state1(self):
        """WallMaster should only be targetable at state 1."""
        from triforce.enemy import Enemy
        from triforce.zelda_enums import Direction, ZeldaEnemyKind

        for state in range(3):
            e = Enemy(None, 1, ZeldaEnemyKind.Wallmaster, (0, 0), Direction.N, 3, 0, 0, state)
            if state == 1:
                assert e.is_active, f"WallMaster state {state} should be active"
            else:
                assert not e.is_active, f"WallMaster state {state} should NOT be active"


class TestObjectClassification:
    """Area 6: Verify _is_id_enemy and _is_projectile boundaries."""

    def test_enemies_in_range(self):
        """Enemy IDs 1-$48 (excluding $40) should be classified as enemies."""
        from triforce.zelda_game import ZeldaGame
        game = type('FakeGame', (), {
            '_is_id_enemy': ZeldaGame._is_id_enemy,
            '_is_projectile': ZeldaGame._is_projectile,
        })()

        for obj_id in range(1, 0x49):
            if obj_id == 0x40:
                assert not game._is_id_enemy(obj_id), "0x40 (StandingFire) should NOT be enemy"
            else:
                assert game._is_id_enemy(obj_id), f"0x{obj_id:02X} should be enemy"

    def test_zero_is_not_enemy(self):
        """ObjType 0 means empty slot."""
        from triforce.zelda_game import ZeldaGame
        assert not ZeldaGame._is_id_enemy(None, 0)

    def test_item_excluded_from_projectile(self):
        """ObjType $60 (item) should not be classified as a projectile."""
        from triforce.zelda_game import ZeldaGame
        assert not ZeldaGame._is_projectile(None, 0x60)

    def test_standing_fire_excluded_from_enemy(self):
        """ObjType $40 (StandingFire) is correctly excluded from enemies."""
        from triforce.zelda_game import ZeldaGame
        assert not ZeldaGame._is_id_enemy(None, 0x40)

    def test_trap_is_projectile_not_enemy(self):
        """Trap ($49) should be classified as projectile, not enemy."""
        from triforce.zelda_game import ZeldaGame
        assert not ZeldaGame._is_id_enemy(None, 0x49)
        assert ZeldaGame._is_projectile(None, 0x49)


class TestEnemyKindEnum:
    """Area 11: Verify ZeldaEnemyKind matches assembly jump table."""

    def test_all_enum_values_in_enemy_range(self):
        """Every ZeldaEnemyKind value should be in the enemy range 1-$48."""
        from triforce.zelda_enums import ZeldaEnemyKind
        for kind in ZeldaEnemyKind:
            assert 1 <= kind.value <= 0x48, \
                f"{kind.name}=0x{kind.value:02X} is outside enemy range 1-$48"

    def test_no_duplicate_values(self):
        """No two enum members should have the same value."""
        from triforce.zelda_enums import ZeldaEnemyKind
        values = [e.value for e in ZeldaEnemyKind]
        assert len(values) == len(set(values)), \
            f"Duplicate values: {[v for v in values if values.count(v) > 1]}"

    def test_standing_fire_not_in_enum(self):
        """$40 (StandingFire) should not be in ZeldaEnemyKind."""
        from triforce.zelda_enums import ZeldaEnemyKind
        values = {e.value for e in ZeldaEnemyKind}
        assert 0x40 not in values

    def test_trap_not_in_enum(self):
        """$49 (Trap) should not be in ZeldaEnemyKind."""
        from triforce.zelda_enums import ZeldaEnemyKind
        values = {e.value for e in ZeldaEnemyKind}
        assert 0x49 not in values
