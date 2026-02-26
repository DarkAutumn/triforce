# Assembly Verification and Fix Progress

**Specs**: Read [asm-review.md](asm-review.md) for review areas and [test-plan.md](test-plan.md) for test strategy.

## Workflow

1. `git checkout main && git pull origin main` — ensure local main matches origin
2. Run `pytest tests/ -v --ignore=tests/ppo_test.py` — record baseline. All existing tests must pass. Skip `ppo_test.py` (slow, unrelated).
3. `git checkout -b <descriptive-branch-name>` — **always** branch from main, never commit to main
4. Work through areas below. For each area:
   - Investigate assembly vs Python
   - **Annotate the assembly** in `zelda-asm/` with comments as you discover insights (see below)
   - Add/update tests as appropriate
   - **Fix bugs found** — the goal is to fix discrepancies, not just document them
   - Run `pytest tests/ -v --ignore=tests/ppo_test.py` — no regressions, no new failures
   - Run `pylint triforce/ evaluate.py run.py train.py` — clean
5. Commit after each area or logical group of changes (triforce and zelda-asm separately).
6. Push branch, open PR to merge to main. **Never push directly to main.**
7. After merge, update specs/docs with anything learned (see End-of-Area Checklist below).

## Assembly Annotation Rules

The `zelda-asm/` directory is a separate git repo (`DarkAutumn/zelda1-disassembly`). Origin
points to the user's fork — there is no upstream to track (it never changes).

**When you discover something about the assembly, annotate it in the source file.** Add
comments that explain what the code does in the context of the Triforce project:
- What Python code corresponds to this assembly routine
- What RAM addresses are read/written and what they mean
- Timing information (how many frames a state lasts, when transitions happen)
- Edge cases or surprising behavior that matters for the AI agent
- Use `; TRIFORCE:` prefix for annotations to distinguish from original comments

Example:
```asm
; TRIFORCE: This is the beam health check. Python: Link._is_health_full_for_beams
; TRIFORCE: filled == containers_minus_one AND partial >= $80
LDA HeartValues
AND #$0F
```

Commit assembly annotations to the `zelda-asm` repo on a branch, push, and merge via PR
(same workflow as triforce). The zelda-asm repo uses `master` as its default branch.

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
- **Tests must never pop up UI.** Always pass `render_mode=None` (or omit, since None is the
  default) when calling `retro.make()` in tests. CI runs headless; any attempt to open a
  window will fail.

## RAM Editing Rules

**Safe to edit via `data.set_value()`:**
- Inventory values: rupees, bombs, arrows, keys, subweapon availability
- Health values: hearts, partial hearts, containers (for beam/health tests)
- Equipment flags: sword type, items, rings
- RNG seeds

**NOT safe to edit:**
- **Link position** (`link_x`, `link_y`) — the game engine's internal state machine doesn't
  update properly. Use controller inputs (BTN_UP/DOWN/LEFT/RIGHT) to move Link naturally.
- **Link/enemy/object states** (ObjState, ObjMetastate, etc.) — the game may not re-read
  these at the frame boundary where tests observe them. Results are unreliable.
- **General rule:** If the NES code reads a value once per frame at a specific point in its
  update loop, writing it between frames may or may not take effect depending on timing.
  Inventory/flags are safe because they're read on demand. Position/state are unsafe because
  the engine has already committed to a state for this frame.

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

### BUG-1: `has_beams` health check disagrees with NES (Area 2) — ✅ FIXED
- **File**: `triforce/link.py`
- **Problem**: `is_health_full` uses float comparison (`health == max_health`). NES assembly
  checks `filled == containers_minus_one AND partial >= 0x80`. When `filled == containers`
  with `partial = 0`, Python says full health → has_beams=True, but NES says no beams.
- **Fix applied**:
  1. Added `_is_health_full_for_beams` property — exact NES integer check on nibbles + partial byte.
  2. Changed `has_beams` to use `_is_health_full_for_beams` instead of `is_health_full`.
  3. Fixed `health.setter` to produce NES-standard encoding for full health: `filled = c-1, partial = $FF`
     (was incorrectly producing `filled = containers, partial = 0`).
- **Verified**: 7 edge cases tested empirically on NES + Python, all match.
- **Tests**: `tests/test_health.py` (26 tests)

### BUG-2: 11-frame beam hack fires mid-spread (Area 2) — ✅ FIXED
- **File**: `triforce/frame_skip_wrapper.py` lines 109-115 (removed)
- **Problem**: Resets `beam_animation` in info dict after 11 consecutive frames at state 17.
  But the assembly's spread phase naturally lasts 22 frames. Hack triggered every single time,
  causing Python to report beam as inactive 11 frames early.
- **Root cause**: Beam is NOT stuck. Spread (22 frames) outlasts sword cooldown (~15 frames).
  The `_sword_count` counter accumulated across action boundaries. The hack masked normal
  behavior and caused: false `has_beams=True`, missed look-ahead triggers, corrupted
  INACTIVE→ACTIVE transition detection.
- **Fix**: Removed the hack entirely. Also removed `_sword_count` field from `ZeldaCooldownHandler`.
- **Tests**: test_look_ahead.py (8 tests), test_weapons.py (updated)

### BUG-3: data.json obj_health_b/c off by 1 (Area 1) — ✅ FIXED
- **File**: `triforce/zelda_game_data.txt` or data.json
- **Problem**: `obj_health_b` ($491) and `obj_health_c` ($492) are off by 1 from ObjHP table.
  Game code uses table reads (correct), but individual address mappings are wrong.
- **Impact**: Low — nothing currently reads these individual entries.
- **Status**: Todo `fix-obj-health-bc-offset`

### BUG-4: Magic rod shot invisible to animation system (Area 2) — ✅ FIXED
- **Files**: `triforce/link.py`, `triforce/state_change_wrapper.py`
- **Problem**: `get_animation_state(BEAMS)` only checked for sword beam states ($10/$11).
  Magic rod shot ($80) was completely ignored — the look-ahead simulation never triggered
  for rod shots, so rod damage was never predicted or rewarded during training.
- **Fix applied**:
  1. Added `ANIMATION_MAGIC_ROD_ACTIVE = 0x80` constant.
  2. Added `ZeldaAnimationKind.MAGIC` case to `get_animation_state` — returns ACTIVE for $80.
  3. Added MAGIC to `_detect_future_damage` look-ahead chain.
  4. Added MAGIC to `_disable_others` (shares beam_animation slot with BEAMS).
- **Verified**: Rod shot lifecycle confirmed via NES: $80 (flying) → $00 (deactivated).
  With book of magic, fire spawns in bomb/fire slot ($10) at state $22 on wall hit.
- **Tests**: All 148 existing tests pass.

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
- [x] Trace magic rod shot: $80→$00 (no $81 state). With book, fire spawns in bomb slot $10.
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

7. **Magic rod shot lifecycle** — Rod shot uses same beam slot ($0E) but with high bit set:
   $80 (flying) → $00 (deactivated on wall/enemy hit). State $81 never occurs — the assembly's
   `HandleShotBlocked` uses ASL to detect magic shots and branches to a different path. With
   `InvBook` ($661) set, fire spawns in bomb/fire slot ($10) at state $22 via `WieldCandle`.
   Rod melee animation uses slot $12 (arrow/rod slot), states $31→$32→$33→$34→$35→$00.
   **BUG-4 FIXED**: `get_animation_state` and look-ahead now handle MAGIC rod shots.

8. **Rod can be tested via RAM** — Setting `magic_rod=1`, `selected_item=8`, and optionally
   `book=1` via data API is sufficient to test rod mechanics. No savestate with rod needed.

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

## Area 4: Enemy Health Encoding ✅

- [x] Verify ObjHP high nibble is health via >> 4
- [x] Determine low nibble purpose (always 0 — HP and damage both multiples of $10)
- [x] Trace damage subtraction in Z_01.asm (DealDamage:5973)
- [x] Verify initial HP assignment for several enemy types
- [x] Decode ObjectTypeToHpPairs packed table (Z_07.asm:5279)
- [x] Trace ExtractHitPointValue nibble extraction (Z_04.asm:11002)
- [x] Identify 0-HP enemies: Gel(0x14,0x15), Keese(0x1B-0x1D)
- [x] Verify "health > 0 else 1" hack in state_change_wrapper is correct
- [x] Fix BUG-3: data.json obj_health_b/c off-by-1
- [x] Update ZeldaEnemyKind enum with complete enemy type table
- [x] Fix references: AquaMentus→Aquamentus, WallMaster→Wallmaster, etc.
- [x] Tests: test_enemy_health.py (23 tests)
- [x] Assembly annotations: ObjectTypeToHpPairs, ExtractHitPointValue, SwordDamagePoints, DealDamage

**Findings:**
- ObjHP stores HP as multiples of $10 (high nibble = HP count, low nibble always 0)
- ObjectTypeToHpPairs packs 2 enemies per byte; ExtractHitPointValue extracts correct nibble
- All damage values are multiples of $10: Wood sword=$10, White=$20, Magic=$40, Bomb=$40, Fire=$10
- Python's `>> 4` extraction is fully correct
- Keese/Gel have 0 HP from table — die in one hit, death tracked via ObjMetastate not ObjHP
- Rope overrides table HP ($10→$40) in quest 2 (InitRope, Z_04.asm:4537)

## Area 5: Enemy is_dying / is_active (Medium) ✅

- [x] Trace death cloud metastate sequence in assembly — $10(1f)→$11(6f)→$12(6f)→$13(6f)→$14 (item drop)
- [x] Verify is_dying range 16-19 ($10-$13) — correct, matches assembly UpdateMetaObject
- [x] Verify Zora is_active: states 2, 3, AND 4 — **BUG FIXED**: was only checking state 3
- [x] Verify Leever is_active: state 3 only — correct (non-Zora path in UpdateBurrower)
- [x] Verify WallMaster is_active: state 1 — correct
- [x] Verify default: spawn_state==0 means active — correct
- [x] Tests: test_enemy_model.py (TestDeathMetastate, TestZoraIsActive, TestLeeverIsActive, TestWallmasterIsActive)

### Area 5 Findings

1. **Death sparkle metastate**: UpdateMetaObject (Z_01.asm) handles sequence $10→$11→$12→$13→$14.
   States $10-$13 last 6 frames each (via ObjTimer). $14 converts to item (type=$60, metastate reset).
   Python's `is_dying` range of 16-19 ($10-$13) is correct — $14 is never observable as the object
   type changes immediately.

2. **BUG FIXED: Zora is_active** — Assembly `UpdateBurrower` (Z_04.asm) checks collisions for Zora
   at states 2, 3, AND 4. Python only returned True for state 3. Fixed to check states 2-4.
   State 3 is when Zora fires its projectile, but states 2 (rising) and 4 (sinking) are also
   vulnerable to player attacks.

3. **Zora state cycle**: 0→1→2→3→4→5→0. State 0 = hidden, 1 = choosing position, 2 = surfacing,
   3 = fully surfaced (fires projectile), 4 = sinking, 5 = delay before restart.

4. **Leever collisions**: Same UpdateBurrower routine but non-Zora path only checks state 3
   (fully surfaced). Confirmed correct in Python.

## Area 6: Object ID Classification (Medium) ✅

- [x] Identify ObjType $40 (StandingFire) — correctly excluded from enemies
- [x] Verify enemy range 1-$48 — correct, matches UpdateObject jump table
- [x] Identify $3F (GuardFire) — within range but not a real enemy, unlikely in enemy slots
- [x] Verify $48 boundary between enemies and projectiles — $49 (Trap) is first non-enemy
- [x] Verify Trap classified as projectile, not enemy — correct for training
- [x] Tests: test_enemy_model.py (TestObjectClassification)

### Area 6 Findings

1. **UpdateObject jump table**: $01-$48 are enemy update routines. $40=UpdateStandingFire is in
   the range but correctly excluded from `_is_id_enemy` (it's a hazard, not a targetable enemy).

2. **GuardFire ($3F)**: Within enemy range but unlikely to appear in enemy slots (it's spawned
   by guards as a projectile-like hazard). Not worth special-casing.

3. **Objects $49+**: Classified as "projectiles" in Python. Includes Trap ($49), UW persons,
   docks, rocks, etc. Functionally adequate for training — these are all non-targetable objects.

4. **$60 (item)**: Correctly excluded from projectiles. Items are dropped by dead enemies and
   handled separately.

## Area 7: Link Health System ✅

- [x] Verify hearts_and_containers nibble encoding
- [x] Verify partial_hearts thresholds (0, 1-$7F, $80-$FF)
- [x] Test 16-heart special case
- [x] Verify health setter round-trip for all values
- [x] **Fix BUG-1**: `has_beams` health check uses assembly's exact integer check
- [x] **Fix health setter**: produces NES-standard encoding (`filled=c-1, partial=$FF`) for full health
- [x] Verify `is_health_full` agrees with NES for all edge cases
- [x] Tests: test_health.py (26 tests)

**Findings**:
- NES beam check (Z_07.asm:4632-4648): `containers_minus_one == hearts_filled AND partial >= $80`
- Python's `is_health_full` float comparison was wrong for beams; added `_is_health_full_for_beams`
- Health setter was also producing incorrect encoding for full health (filled=containers instead of c-1)
- 16-heart case (0xFF) works correctly as a natural consequence of the general logic

## Area 8: Look-Ahead Simulation (HIGH) ✅

- [x] Test em.set_state() fully restores RAM after data.set_value changes — **confirmed, byte-for-byte identical**
- [x] Verify _disable_others doesn't persist after restore — **confirmed**
- [x] Verify health override (0xFF) doesn't persist — **confirmed**
- [x] Test multi-weapon look-ahead isolation — **confirmed, beam state preserved after bomb look-ahead**
- [x] Investigate beam stuck-at-17 root cause — **beam is NOT stuck, spread naturally lasts 22 frames**
- [x] **Fix BUG-2**: Removed 11-frame hack from frame_skip_wrapper.py (see findings below)
- [x] Tests: test_look_ahead.py (8 tests)
- [ ] Test damage attribution with discounts
- [ ] Test room transition clears discounts
- [ ] Verify trigger condition (INACTIVE→ACTIVE only)

### Area 8 Findings

1. **em.set_state() is complete** — After saving state, modifying weapon slots + health via
   data.set_value, stepping 10 frames, and restoring, the full 10KB RAM is byte-for-byte
   identical. No state leaks. The look-ahead's save/restore mechanism is sound.

2. **BUG-2 root cause: not a bug** — Beams were never "stuck" at state 17. The beam spread
   phase naturally lasts 22 frames, which exceeds the sword cooldown (~15 frames). When the
   agent fires rapidly, the beam from attack N is still spreading when the sword cooldown
   from attack N+1 finishes. The `_sword_count` counter in `_step_with_frame_capture` accumulated
   across action boundaries, reaching 11 and triggering the hack mid-spread every time.

3. **11-frame hack was harmful** — The hack modified `info['beam_animation']` to 0, which:
   - Made `get_animation_state(BEAMS)` return INACTIVE while beams were still active
   - Made `has_beams` return True (beams available) when the beam slot was occupied
   - Corrupted the INACTIVE→ACTIVE transition detection in look-ahead
   - Could cause the look-ahead to miss beam damage attribution
   **Fix**: Removed the hack entirely. Beam spread deactivates naturally after 22 frames.

4. **ZeldaGame.__active guard is safe** — The look-ahead creates new ZeldaGame instances in a
   loop, which changes `__active`. But the `start` reference's enemies/health data is accessed
   via `cached_property` which was already evaluated before the loop. No stale-reference bugs.

## Area 9: Direction Encoding (Low)

- [ ] Confirm E=1, W=2, S=4, N=8 matches assembly
- [ ] Verify no object stores composite directions
- [ ] Tests: test_object_model.py (T2.9)

## Area 10: Tile Layout (Low)

- [ ] Verify $D30 maps to PlayAreaTiles (retro remapping)
- [ ] Verify reshape (32,22).T.swapaxes(0,1) gives correct x,y indexing
- [ ] Tests: test_object_model.py (T7.1-T7.2)

## Area 11: ZeldaEnemyKind IDs (Medium) ✅

- [x] Cross-ref enum values against assembly enemy type tables — added 12 missing IDs
- [x] Removed Trap from enum (it's $49, outside enemy range)
- [x] Renamed Gohma→BlueGohma, Lamnola→BlueLamnola for consistency with color variants
- [x] Added: PatraChild1/2, Dodongo2, RedGohma, Digdogger2/3, RedLamnola, Gleeok2/3/4, GleeokHead, Patra2
- [x] Resolve Octorok/OctorokFast duplicate (both 0x07) — not a duplicate, OctorokFast is $0A
- [x] Tests: test_enemy_model.py (TestEnemyKindEnum)

### Area 11 Findings

1. **12 missing enemy IDs** added to `ZeldaEnemyKind` by cross-referencing the full UpdateObject
   jump table in assembly. These are higher-level variants and sub-enemies (e.g., PatraChild
   orbiting enemies, multi-headed Gleeok variants).

2. **Trap removed** — Trap is at $49, which is OUTSIDE the enemy range ($01-$48). It's correctly
   classified as a projectile/hazard, not an enemy. Removing it from the enum prevents incorrect
   enemy identification.

3. **Naming consistency** — Gohma was only the blue variant ($30). Added RedGohma ($32).
   Lamnola was only the blue variant ($3B). Added RedLamnola ($3C). Renamed originals to
   BlueGohma/BlueLamnola for clarity.

## Area 12: Frame Skip & Animation Tracking ✅ (partial — BUG-2 and WS_ADJUSTMENT deferred)

Replaced hardcoded `ATTACK_COOLDOWN=15` and `ITEM_COOLDOWN=10` with animation-state polling.
The NES blocks new actions until `link_status==0` AND `sword_animation==0`.

- [x] Measure sword cooldown: 15 frames (states 1→2→3→4→5→0). ATTACK_COOLDOWN=15 was correct.
- [x] Measure item cooldowns: 12 frames (link_status 0x11→0x31→0x00). ITEM_COOLDOWN=10 was 2 frames early.
- [x] Verify sword animation state tracking: states 1(4f)→2(8f)→3(1f)→4(1f)→5(1f)→0
- [x] Determine controllability signal: `link_status==0 AND sword_animation==0`
- [x] Replace hardcoded cooldowns with animation-state polling in `_act_attack_or_item`
- [x] Verify new code fires on first possible frame (sword, bomb, boomerang all confirmed)
- [x] Tests: test_frame_skip.py (10 tests)
- [ ] Investigate south/west movement asymmetry (WS_ADJUSTMENT_FRAMES=4) — deferred
- [ ] **Fix BUG-2**: 11-frame beam hack fires mid-spread (depends on Area 8 look-ahead) — deferred

**Findings**:
- `link_status` (ObjState[0]): 0x11 during animation, 0x31 near end, 0x00 when controllable
- Sword melee uses slot 0x0D (not 0x0B). States 1-5, timer in ObjAnimCounter.
- State 2 lasts 8 frames, all others 1 frame. State 3 triggers beam check (MakeSwordShot).
- Items (bomb, boomerang): link_status lock lasts 12 frames. No sword_state involvement.
- STUN_FLAG (0x40) in is_link_stunned does NOT detect animation lock (0x11 & 0x40 = 0).
- Old ITEM_COOLDOWN=10 was 2 frames short — compensated by accident via extra frame in
  _skip_uncontrollable_states. New polling removes this fragile dependency.

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
| 3. Screen lock bounds | **HIGH** | ✅ |
| 4. Enemy health encoding | Medium | ✅ |
| 5. Enemy dying/active | Medium | ✅ |
| 6. Object ID classification | Medium | ✅ |
| 7. Health system | Medium | ✅ |
| 8. Look-ahead simulation | **HIGH** | ✅ |
| 9. Direction encoding | Low | ⬜ |
| 10. Tile layout | Low | ⬜ |
| 11. Enemy kind IDs | Medium | ✅ |
| 12. Frame skip & animation | **HIGH** | ✅* |
| 13. Sound bitmasks | Low | ⬜ |

Legend: ⬜ Not started · 🔄 In progress · ✅ Done · ❌ Blocked
