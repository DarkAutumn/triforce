# Assembly Review Progress

**Specs**: Read [asm-review.md](asm-review.md) for review areas and [test-plan.md](test-plan.md) for test strategy.

## Workflow

1. `git checkout main && git pull origin main` — ensure local main matches origin
2. Run `pytest tests/ -v --ignore=tests/ppo_test.py` — record baseline. All existing tests must pass. Skip `ppo_test.py` (slow, unrelated).
3. `git checkout -b <descriptive-branch-name>` — **always** branch from main, never commit to main
4. Work through areas below. For each area:
   - Investigate assembly vs Python
   - Add/update tests as appropriate
   - Fix bugs found
   - Run `pytest tests/ -v --ignore=tests/ppo_test.py` — no regressions, no new failures
   - Run `pylint triforce/ evaluate.py run.py train.py` — clean
5. Commit after each area or logical group of changes.
6. Push branch, open PR to merge to main. **Never push directly to main.**
7. After merge, update specs/docs with anything learned (see End-of-Area Checklist below).

**Not done until**: `pytest` passes fully (baseline + new tests) and `pylint` is clean.

**IMPORTANT**: Do not start work on a new area unless all prior changes are committed. If the
branch has uncommitted changes, commit or stash them first.

## Testing Approach

We can't see the game screen, but we can read every RAM byte. The ROM is the oracle — we
test by observing the NES's RAM reaction to inputs:

- **Boundary tests**: Walk Link toward each edge using controller inputs, check if `ObjState[SLOT_SWORD]`
  changes after pressing A. Changed → sword fired. Unchanged → screen-locked. Binary RAM check.
  **Do NOT modify RAM to set Link's position** — use inputs to reach positions naturally.
- **State machine tests**: Fire a weapon, read `ObjState[slot]` every frame via `RAMWatcher`.
  The trace gives the exact lifecycle and timing.
- **has_beams test**: Set health RAM to the edge case, press A, check if `ObjState[SLOT_BEAM]`
  changes. The NES tells us whether it actually fires.
- **Object model tests**: Load a savestate with known enemies, compare raw RAM bytes against
  what the Python classes report. Discrepancy = bug.
- **Look-ahead tests**: Snapshot all 10KB of RAM, run the look-ahead code, snapshot again.
  Byte-for-byte diff reveals any state corruption.

## Interactive Debugging

You can write ad-hoc Python scripts to load any savestate and interactively explore the
NES. Use `ZeldaFixture` from `tests/conftest.py` to load a state, poke RAM values, send
button inputs, and observe how the NES responds frame-by-frame. This is the primary way
to build understanding of game mechanics before writing formal tests.

Example workflow:
```python
from tests.conftest import ZeldaFixture
from tests.asm_addresses import *

emu = ZeldaFixture("1_44e.state")       # Load dungeon room
print(f"Link at ({emu.get('link_x')}, {emu.get('link_y')})")
print(f"Link direction: {emu.ram[OBJ_DIR]}")

# Send button presses and watch what happens
import numpy as np
buttons = np.zeros(9, dtype=np.int8)
buttons[BTN_B] = 1                      # Press B (sword)
emu.step(buttons)
print(f"Sword state: {emu.ram[SWORD_STATE]}")

# Step multiple frames, watch a value evolve
for i in range(20):
    emu.step()
    print(f"Frame {i}: beam={emu.ram[BEAM_STATE]}, sword={emu.ram[SWORD_STATE]}")

emu.close()
```

Use this to prototype and discover behavior, then codify findings as pytest tests.

### Savestate Naming Convention

Savestates follow the pattern `{level}_{room_hex}{suffix}.state`:
- **level**: 0 = overworld, 1 = dungeon 1, etc.
- **room_hex**: hex room ID (e.g., `44`, `72`)
- **suffix**: Link's starting position in the room
  - `n` = north side (entered from the south, can move north)
  - `s` = south side
  - `e` = east side
  - `w` = west side
  - `c` = center of room
  - `t` = game start (`start.state`)

Example: `1_44e.state` = dungeon 1, room $44, Link on the east side (entered from east).
This tells you which direction Link can move — `e` means Link is on the east edge and
would typically move west toward the room interior.

## Environment Notes

- Python 3.12 venv at `.venv` (stable-retro requires <3.13, system Python is 3.14)
- RAM is 10240 bytes (not 2KB — retro maps a larger NES address space)
- Read API is `data.lookup_value(name)`, write is `data.set_value(name, value)`
- `ZeldaGame` has an `__active` class variable — only the most recent instance can read RAM.
  Tests using `ZeldaFixture.game_state()` must not hold stale references.
- **Only one emulator instance per process** — must `close()` before creating another.
- **NES A button (retro index 8) fires sword**. B button (retro index 0) uses selected item.
  `action_space.py` correctly maps `ActionKind.SWORD` to `self.a`.
- **State files must be gzip compressed** for retro to load them. The F1 hotkey in zui saves
  gzipped states.
- Most savestates start at room edges where Link is screen-locked and cannot fire weapons.
  For weapon testing, use `debug_` prefixed states or move Link inward first.

## Requesting Developer Help

When automated investigation hits a wall (e.g., need a savestate with specific conditions like
full health + specific equipment + enemies nearby, or Link in a position that's hard to reach
programmatically), **ask the developer to create a savestate** using the F1 hotkey in the zui
debugger. Specify exactly what's needed:
- Room and level (e.g., "overworld room $67")
- Link's position (center, near enemies, etc.)
- Health state (full health for beams, partial for no-beams)
- Equipment (sword type, items in inventory)
- Enemies present and their state
- Any other RAM conditions

The developer can navigate to the right spot in-game and press F1 to drop a state file.
This is faster and more reliable than trying to manipulate RAM to create the right conditions.

## End-of-Area Checklist

After completing each area, update these docs with anything learned:
- New findings or corrections → add to the relevant area's notes in this file
- New environment quirks or API gotchas → add to Environment Notes above
- Test plan changes → update `test-plan.md`
- Architecture insights → update `asm-review.md` if the area description was wrong/incomplete

**MANDATORY: Every bug found must either be fixed in the same commit or tracked as a todo
below with a clear description, the affected file(s), and line numbers. No exceptions.
"Non-blocking" is not an excuse to skip tracking. If it's deferred, it must have a todo.**

## Bug Fix Backlog

Bugs found during review that need production code fixes. These are **not** test-only issues —
they affect the actual game state model used for training.

### BUG-1: `has_beams` health check disagrees with NES (Area 2)
- **File**: `triforce/link.py` lines 195-205
- **Problem**: `is_health_full` uses float comparison (`health == max_health`). NES assembly
  checks `filled == containers_minus_one AND partial >= 0x80`. When `filled == containers`
  with `partial = 0`, Python says full health → has_beams=True, but NES says no beams.
- **Fix**: Replace `is_health_full` check in `has_beams` with exact integer check on
  `hearts_and_containers` nibbles and `partial_hearts` byte.
- **Impact**: Training scenarios that override health via `per_reset`/`per_frame` could
  trigger this. The model may incorrectly predict beam availability.
- **Status**: Todo `fix-has-beams-health`

### BUG-2: 11-frame beam hack fires mid-spread (Area 2)
- **File**: `triforce/frame_skip_wrapper.py` lines 110-115
- **Problem**: Resets `beam_animation` in info dict after 11 consecutive frames at state 17.
  But the assembly's spread phase naturally lasts 22 frames. Hack triggers every single time,
  causing Python to report beam as inactive 11 frames early.
- **Root cause**: Hack was added because beams get "stuck" at 17. Likely caused by look-ahead
  simulation (Area 8) corrupting beam state.
- **Fix**: Investigate look-ahead first (Area 8 dependency), then either remove the hack
  (if root cause is fixed) or change threshold to >22.
- **Status**: Todo `fix-11-frame-hack` (blocked on `look-ahead-sim`)

### BUG-3: data.json obj_health_b/c off by 1 (Area 1)
- **File**: `triforce/zelda_game_data.txt` or data.json
- **Problem**: `obj_health_b` ($491) and `obj_health_c` ($492) are off by 1 from ObjHP table.
  Game code uses table reads (correct), but individual address mappings are wrong.
- **Impact**: Low — nothing currently reads these individual entries.
- **Status**: Todo `fix-obj-health-bc-offset`

---

## Test Infrastructure

- [x] `ZeldaFixture` — raw emulator wrapper, single-frame step, direct RAM r/w
- [x] `AssemblyAddresses` — constants from Variables.inc
- [x] `RAMWatcher` — track RAM changes across frames
- [x] `conftest.py` — shared pytest fixtures
- [x] Baseline existing tests still pass (22/22)

## Area 1: Memory Address Mapping (Medium) ✅

- [x] Cross-ref zelda_game_data.txt [memory] vs Variables.inc
- [x] Cross-ref zelda_game_data.txt [tables] vs Variables.inc
- [x] Verify room_kills/$34F vs ObjType overlap
- [x] Verify item_timer/$3A8 vs ObjPosFrac union
- [x] Verify triforce_of_power/$672 vs LastBossDefeated semantics
- [x] Verify sound_pulse_1/$605 vs Tune0
- [x] Verify weapon slot ObjState addresses ($B9-$BE)
- [x] Tests: test_ram_mapping.py (41 tests)
- **Finding**: data.json `obj_health_b` ($491) and `obj_health_c` ($492) are off by 1 from ObjHP table. Game code uses table read (correct), not individual addresses. Tracked as BUG-3 / todo `fix-obj-health-bc-offset`.

## Area 2: Beam/Sword Shot State Machine (HIGH) ✅

- [x] Trace ObjState[$0E] lifecycle: $00→$10→$11→$00 — confirmed
- [x] Verify ANIMATION_BEAMS_ACTIVE=16 ($10), ANIMATION_BEAMS_HIT=17 ($11) — correct
- [x] Measure spread duration — **22 frames** (not ~10 as originally estimated). ObjDir decrements from $FE to $E8.
- [x] Verify sword melee state sequence: 1→2→3→4→5→0 (state 2 = 8 frames, states 3/4/5 = 1 frame each)
- [x] Verify MakeSwordShot fires at sword state 3 (Z_07.asm line 4543)
- [x] Verify health check: filled == containers-1 AND partial >= $80 (Z_07.asm lines 4632-4648)
- [x] Validate 11-frame hack in frame_skip_wrapper.py — **hack is wrong**: it fires at 11 frames but natural spread is 22 frames. Only modifies info dict, not NES RAM, so NES is unaffected but Python thinks beam inactive 11 frames early.
- [x] Tests: test_weapons.py (17 tests covering lifecycle, timing, health edge cases)
- [ ] Trace magic rod shot: $80→$81 vs beam $10→$11 (deferred — no rod savestate available)
- [ ] Reproduce beam stuck-at-17 bug (deferred to Area 8 — likely caused by look-ahead simulation)
- [ ] Determine if look-ahead causes stuck-at-17 (deferred to Area 8)

### Area 2 Findings

1. **BUG: `is_health_full` disagrees with NES for beams** — When RAM has `filled == containers`
   (not `containers-1`) with partial=0, Python's float-based `is_health_full` returns True
   (health=3.0, max=3), but the NES assembly says no beams. This edge case can occur via
   `per_reset`/`per_frame` RAM overrides used in training scenarios. Fix should make `has_beams`
   use the assembly's exact check: `filled == containers_minus_one AND partial >= 0x80`.

2. **11-frame hack is incorrect** — `frame_skip_wrapper.py` resets `beam_animation` in the info
   dict after 11 consecutive frames at state 17. But the assembly's spread phase naturally lasts
   22 frames. The hack was added to work around beams "stuck" at 17, which is likely caused by
   the look-ahead simulation (Area 8). The hack masks the real bug.

3. **Sword state 1 lasts 4 frames** — Assembly timer for state 1 appears to be 1 frame, but
   observed behavior is 4 frames. Likely due to ObjAnimCounter initialization. Non-blocking
   for beam logic since the beam fires at state 3 regardless.

4. **BTN_A is sword, not BTN_B** — Fixed comment in `asm_addresses.py`. The NES A button (retro
   index 8) fires sword. B button (retro index 0) uses the selected item. `action_space.py`
   correctly maps `ActionKind.SWORD` to `self.a`.

5. **Beam active duration is variable** — Observed 25 frames in test, but this depends on distance
   to wall/edge. The spread duration (22 frames) is always fixed.

6. **`debug_0_67_1772056964.state`** — Critical savestate for beam testing: overworld room $67,
   Link centered, full health, sword equipped. Created via F1 hotkey in zui debugger.

## Area 3: Sword Screen Lock Boundaries (HIGH) ✅

- [x] Trace MaskInputInBorder in Z_05.asm
- [x] Identify which BorderBounds set controls A-button masking
- [x] Compare assembly outer bounds vs Python OW/UW values
- [x] Test actual NES behavior at boundary coordinates
- [x] Fix is_sword_screen_locked UW bounds (were wrong)
- [x] Fix get_sword_directions_allowed OW and UW bounds (were wrong)
- [x] Tests: test_boundaries.py (9 tests: 5 empirical NES + 4 Python model)

### Area 3 Findings

1. **Assembly structure**: `BorderBounds` (Z_05.asm:2728) has 3 sets of 4 bytes (down, up, right,
   left): OW outer ($D6,$45,$E9,$07), UW outer ($C6,$55,$D9,$17), Inner ($BE,$54,$D1,$1F).
   `Link_FilterInput` calls `MaskInputInBorder` twice: first with inner bounds + mask $80
   (keeps A button, blocks everything else), then with outer bounds + mask $00 (blocks all).

2. **Inner bounds do NOT block sword** — The inner check uses AND $80 which KEEPS bit 7 (A button)
   and clears everything else. Only the OUTER bounds (mask $00) clear the A button and block sword.

3. **BUG FIXED: UW boundaries were wrong in Python**:
   - `is_sword_screen_locked`: left was `x <= 0x10` (should be `x < 0x17`), up was `y <= 0x53`
     (should be `y < 0x55`), down was `y >= 0xC5` (should be `y >= 0xC6`). Right was correct.
   - `get_sword_directions_allowed`: Same UW value errors. OW values were also off by 1 on each
     side (e.g., `7 < x < 0xe8` should be `0x06 < x < 0xe9`).
   - **Impact**: At X=$11-$16 in UW, Python said sword available but NES blocked it. At Y=$C5
     in UW, Python said locked but NES allowed it. Wrong action masking during training.

4. **Sword lock is direction-dependent in NES** — The assembly only checks boundaries on the axis
   of Link's facing direction. At X=$D9 facing North, sword fires (only E/W are blocked). Python's
   `is_sword_screen_locked` is conservative (blocks all directions at any boundary). This is a
   known approximation, not a bug — documented in docstring.

5. **OW bounds are correct** — Python OW outer bounds exactly match assembly: left=$07, right=$E9,
   up=$45, down=$D6. Verified empirically for east boundary.

6. **Corrected boundary values**:
   - OW outer: left=$07, right=$E9, up=$45, down=$D6 (unchanged)
   - UW outer: left=$17, right=$D9, up=$55, down=$C6 (fixed left/up/down)

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
| 1. Address mapping | Medium | ✅ |
| 2. Beam state machine | **HIGH** | ✅ |
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
