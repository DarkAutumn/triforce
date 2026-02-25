# Assembly Review Progress

**Specs**: Read [asm-review.md](asm-review.md) for review areas and [test-plan.md](test-plan.md) for test strategy.

## Workflow

1. `git checkout main && git pull`
2. Run `pytest tests/ -v --ignore=tests/ppo_test.py` — record baseline. All existing tests must pass. Skip `ppo_test.py` (slow, unrelated).
3. `git checkout -b asm-review` (branch only if making changes)
4. Work through areas below. For each area:
   - Investigate assembly vs Python
   - Add/update tests as appropriate
   - Fix bugs found
   - Run `pytest` — no regressions, no new failures
   - Run `pylint triforce/ evaluate.py run.py train.py` — clean
5. Commit after each area or logical group of changes.

**Not done until**: `pytest` passes fully (baseline + new tests) and `pylint` is clean.

**IMPORTANT**: Do not start work on a new area unless all prior changes are committed. If the
branch has uncommitted changes, commit or stash them first.

## Testing Approach

We can't see the game screen, but we can read every RAM byte. The ROM is the oracle — we
test by observing the NES's RAM reaction to inputs:

- **Boundary tests**: Set link_x/link_y, press B, check if `ObjState[SLOT_SWORD]` changed
  from 0. Changed → sword fired. Unchanged → screen-locked. Binary RAM check, no screen needed.
- **State machine tests**: Fire a weapon, read `ObjState[slot]` every frame via `RAMWatcher`.
  The trace gives the exact lifecycle and timing.
- **has_beams test**: Set health RAM to the edge case, press B, check if `ObjState[SLOT_BEAM]`
  changes. The NES tells us whether it actually fires.
- **Object model tests**: Load a savestate with known enemies, compare raw RAM bytes against
  what the Python classes report. Discrepancy = bug.
- **Look-ahead tests**: Snapshot all 10KB of RAM, run the look-ahead code, snapshot again.
  Byte-for-byte diff reveals any state corruption.

## Environment Notes

- Python 3.12 venv at `.venv` (stable-retro requires <3.13, system Python is 3.14)
- RAM is 10240 bytes (not 2KB — retro maps a larger NES address space)
- Read API is `data.lookup_value(name)`, write is `data.set_value(name, value)`
- `ZeldaGame` has an `__active` class variable — only the most recent instance can read RAM.
  Tests using `ZeldaFixture.game_state()` must not hold stale references.

---

## Test Infrastructure

- [x] `ZeldaFixture` — raw emulator wrapper, single-frame step, direct RAM r/w
- [x] `AssemblyAddresses` — constants from Variables.inc
- [x] `RAMWatcher` — track RAM changes across frames
- [x] `conftest.py` — shared pytest fixtures
- [x] Baseline existing tests still pass (22/22)

## Area 1: Memory Address Mapping (Medium)

- [ ] Cross-ref zelda_game_data.txt [memory] vs Variables.inc
- [ ] Cross-ref zelda_game_data.txt [tables] vs Variables.inc
- [ ] Verify room_kills/$34F vs ObjType overlap
- [ ] Verify item_timer/$3A8 vs ObjPosFrac union
- [ ] Verify triforce_of_power/$672 vs LastBossDefeated semantics
- [ ] Verify sound_pulse_1/$605 vs Tune0
- [ ] Verify weapon slot ObjState addresses ($B9-$BE)
- [ ] Tests: test_ram_mapping.py

## Area 2: Beam/Sword Shot State Machine (HIGH)

- [ ] Trace ObjState[$0E] lifecycle: $00→$10→$11→$00
- [ ] Verify ANIMATION_BEAMS_ACTIVE=16, ANIMATION_BEAMS_HIT=17
- [ ] Measure spread duration (expected ~10 frames)
- [ ] Trace magic rod shot: $80→$81 vs beam $10→$11
- [ ] Reproduce beam stuck-at-17 bug
- [ ] Determine if look-ahead causes stuck-at-17
- [ ] Validate 11-frame hack in frame_skip_wrapper.py
- [ ] Tests: test_weapons.py (T3.1-T3.3, T3.8-T3.9)

## Area 3: Sword Screen Lock Boundaries (HIGH)

- [ ] Trace MaskInputInBorder in Z_05.asm
- [ ] Identify which BorderBounds set controls A-button masking
- [ ] Compare assembly inner bounds ($BE,$54,$D1,$1F) vs Python OW/UW values
- [ ] Test actual NES behavior at boundary coordinates
- [ ] Fix is_sword_screen_locked if wrong
- [ ] Fix get_sword_directions_allowed to match
- [ ] Tests: test_boundaries.py (T4.1-T4.5)

## Area 4: Enemy Health Encoding (Medium)

- [ ] Verify ObjHP high nibble is health via >> 4
- [ ] Determine low nibble purpose
- [ ] Trace damage subtraction in Z_01.asm
- [ ] Verify initial HP assignment for several enemy types
- [ ] Tests: test_object_model.py (T2.1-T2.2)

## Area 5: Enemy is_dying / is_active (Medium)

- [ ] Trace death cloud metastate sequence in assembly
- [ ] Verify is_dying range 16-19 (or correct it)
- [ ] Verify Lever/Zora is_active: ObjState==3 means surfaced
- [ ] Verify WallMaster is_active: ObjState==1
- [ ] Verify default: spawn_state==0 means active
- [ ] Tests: test_object_model.py (T2.5-T2.8)

## Area 6: Object ID Classification (Medium)

- [ ] Identify ObjType $40 (excluded from enemies)
- [ ] Verify enemy range 1-$48
- [ ] Identify excluded projectile IDs ($63,$64,$68,$6A)
- [ ] Verify $48 boundary between enemies and projectiles
- [ ] Tests: test_object_model.py (T2.3-T2.4)

## Area 7: Link Health System (Medium)

- [ ] Verify hearts_and_containers nibble encoding
- [ ] Verify partial_hearts thresholds (0, 1-$7F, $80-$FF)
- [ ] Test 16-heart special case
- [ ] Verify health setter round-trip for all values
- [ ] Tests: test_health.py (T1.1-T1.2, T1.6)

## Area 8: Look-Ahead Simulation (HIGH)

- [ ] Test em.set_state() fully restores RAM after data.set_value changes
- [ ] Verify _disable_others doesn't persist after restore
- [ ] Verify health override (0xFF) doesn't persist
- [ ] Test multi-weapon look-ahead isolation
- [ ] Test damage attribution with discounts
- [ ] Test room transition clears discounts
- [ ] Verify trigger condition (INACTIVE→ACTIVE only)
- [ ] Tests: test_look_ahead.py (T5.1-T5.9)

## Area 9: Direction Encoding (Low)

- [ ] Confirm E=1, W=2, S=4, N=8 matches assembly
- [ ] Verify no object stores composite directions
- [ ] Tests: test_object_model.py (T2.9)

## Area 10: Tile Layout (Low)

- [ ] Verify $D30 maps to PlayAreaTiles (retro remapping)
- [ ] Verify reshape (32,22).T.swapaxes(0,1) gives correct x,y indexing
- [ ] Tests: test_object_model.py (T7.1-T7.2)

## Area 11: ZeldaEnemyKind IDs (Medium)

- [ ] Cross-ref enum values against assembly enemy type tables
- [ ] Resolve Octorok/OctorokFast duplicate (both 0x07)
- [ ] Tests: test_object_model.py (T2.10)

## Area 12: Frame Skip Cooldowns (Medium)

- [ ] Measure sword cooldown vs ATTACK_COOLDOWN=15
- [ ] Measure item cooldowns vs ITEM_COOLDOWN=10
- [ ] Investigate south/west movement asymmetry (WS_ADJUSTMENT_FRAMES=4)
- [ ] Tests: test_frame_skip.py (T6.1-T6.5)

## Area 13: Sound Bitmask Register (Low)

- [ ] Verify $605 is correct register for sound bitmasks
- [ ] Verify each SoundKind value against assembly
- [ ] Tests: test_object_model.py (T8.1-T8.2)

---

## Summary

| Area | Risk | Status |
|------|------|--------|
| 1. Address mapping | Medium | ⬜ |
| 2. Beam state machine | **HIGH** | ⬜ |
| 3. Screen lock bounds | **HIGH** | ⬜ |
| 4. Enemy health encoding | Medium | ⬜ |
| 5. Enemy dying/active | Medium | ⬜ |
| 6. Object ID classification | Medium | ⬜ |
| 7. Health system | Medium | ⬜ |
| 8. Look-ahead simulation | **HIGH** | ⬜ |
| 9. Direction encoding | Low | ⬜ |
| 10. Tile layout | Low | ⬜ |
| 11. Enemy kind IDs | Medium | ⬜ |
| 12. Frame skip cooldowns | Medium | ⬜ |
| 13. Sound bitmasks | Low | ⬜ |

Legend: ⬜ Not started · 🔄 In progress · ✅ Done · ❌ Blocked
