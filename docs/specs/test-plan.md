# Comprehensive Test Plan

This document defines the testing strategy for verifying the Python game state model against the
NES assembly. It covers test infrastructure, organization, specific test cases, and methodology.

Companion document: [asm-review.md](asm-review.md) — the 13 review areas these tests validate.

## Principles

1. **The ROM is immutable truth.** The NES ROM and its assembly source are fixed and will never
   change. Every test can rely on deterministic behavior given the same savestate and RNG seed.

2. **Test from the bottom up.** RAM addresses → object model → game logic → state machines →
   wrapper chain. Lower layers must be verified before higher layers can be trusted.

3. **Raw RAM is the ground truth.** Tests should read raw RAM bytes and verify that our Python
   abstraction layer interprets them correctly. Never test Python against Python.

4. **One frame at a time.** For state machine and timing tests, we need single-NES-frame stepping
   without the FrameSkipWrapper. This is the only way to trace state transitions precisely.

5. **Tests prove bugs and prove fixes.** When we find a discrepancy, write a test that demonstrates
   the bug first, then fix the code and verify the test passes.

6. **Determinism is mandatory.** Every test must produce the same result every time. RNG values
   are set to fixed seeds. Savestates provide known starting points.

---

## Test Infrastructure

### Current Infrastructure

- **`ZeldaActionReplay`** (`tests/utilities.py`): Wraps the full wrapper chain
  (retro → FrameSkipWrapper → StateChangeWrapper → ZeldaActionSpace). Good for high-level
  integration tests (fire weapon, verify damage). Not suitable for low-level RAM or timing tests.

- **`CriticWrapper`**: Injects reward critics into the wrapper chain. Used for reward-specific
  tests.

- **Savestates**: 271 `.state` files in `triforce/custom_integrations/Zelda-NES/` covering
  overworld and dungeon rooms with various entry directions.

### New Infrastructure Needed

#### 1. `ZeldaFixture` — Low-level emulator access

A test utility class that wraps `retro.make()` directly, **without** the FrameSkipWrapper or
StateChangeWrapper. This gives tests:

- **Single-frame stepping**: `fixture.step()` advances exactly one NES frame
- **Raw RAM access**: `fixture.ram[address]` reads any byte (10240-byte array, not just 2KB)
- **Named variable access**: `fixture.get(name)` / `fixture.set(name, value)` via data.json
  (note: read API is `data.lookup_value`, write is `data.set_value`)
- **Object table reading**: `fixture.object_tables()` returns an `ObjectTables` from current RAM
- **Game state construction**: `fixture.game_state()` builds a `ZeldaGame` from current RAM
  (caution: `ZeldaGame` uses a class-level `__active` guard — only the latest instance can read RAM)
- **Savestate management**: `fixture.save()` / `fixture.restore()` for mid-test checkpoints
- **Button press helpers**: `fixture.step(buttons=[BTN_B])` to press buttons for one frame

```python
class ZeldaFixture:
    """Low-level test fixture with direct emulator access."""
    def __init__(self, savestate):
        self.env = retro.make(game='Zelda-NES', state=savestate,
                              inttype=retro.data.Integrations.CUSTOM_ONLY)
        self.env.reset()

    @property
    def ram(self):
        """Returns the full RAM as a read/write array."""
        return self.env.get_ram()

    def get(self, name):
        """Read a named variable via data.json."""
        return self.env.data.lookup_value(name)

    def set(self, name, value):
        """Write a named variable via data.json."""
        self.env.data.set_value(name, value)

    def step(self, buttons=None):
        """Advance exactly one NES frame."""
        action = np.zeros(9, dtype=bool)
        if buttons:
            for b in buttons:
                action[b] = True
        return self.env.step(action)

    def game_state(self):
        """Build a ZeldaGame from current emulator state."""
        ...

    def save(self):
        """Save emulator state."""
        return self.env.em.get_state()

    def restore(self, state):
        """Restore emulator state."""
        self.env.em.set_state(state)
```

#### 2. `AssemblyAddresses` — Assembly address reference

A module that defines constants from `Variables.inc` so tests can reference canonical NES
addresses without magic numbers:

```python
# From Variables.inc
OBJ_X      = 0x070   # ObjX, 12 bytes
OBJ_Y      = 0x084   # ObjY, 12 bytes
OBJ_DIR    = 0x098   # ObjDir, 12 bytes
OBJ_STATE  = 0x0AC   # ObjState, indexed by slot
OBJ_TYPE   = 0x34F   # ObjType, 12 bytes
OBJ_META   = 0x405   # ObjMetastate, 12 bytes
OBJ_HP     = 0x485   # ObjHP, 12 bytes
HEART_VAL  = 0x66F   # HeartValues
HEART_PART = 0x670   # HeartPartial
CUR_LEVEL  = 0x010   # CurLevel
ROOM_ID    = 0x0EB   # RoomId
GAME_MODE  = 0x012   # GameMode

# Weapon slots (ObjState offsets)
SLOT_SWORD = 0x0D    # Sword/Rod
SLOT_BEAM  = 0x0E    # Sword shot / Magic shot
SLOT_BOOM  = 0x0F    # Boomerang / Food
SLOT_BOMB1 = 0x10    # Bomb / Fire 1
SLOT_BOMB2 = 0x11    # Bomb / Fire 2
SLOT_ARROW = 0x12    # Rod / Arrow
```

#### 3. `RAMWatcher` — Track RAM changes across frames

A helper that records changes to specified RAM addresses over multiple frames, producing a
trace log for state machine verification:

```python
class RAMWatcher:
    def __init__(self, fixture, addresses):
        self.fixture = fixture
        self.addresses = addresses  # {name: address}
        self.trace = []

    def step(self, buttons=None):
        """Step one frame and record watched addresses."""
        self.fixture.step(buttons)
        snapshot = {name: self.fixture.ram[addr] for name, addr in self.addresses.items()}
        self.trace.append(snapshot)
        return snapshot

    def run(self, n_frames, buttons=None):
        """Run N frames, recording each."""
        for _ in range(n_frames):
            self.step(buttons)
        return self.trace
```

#### 4. Pytest fixtures

```python
@pytest.fixture
def emu():
    """Provide a ZeldaFixture with the start state, cleaned up after test."""
    fixture = ZeldaFixture("start.state")
    yield fixture
    fixture.close()

@pytest.fixture(params=["1_44e.state", "1_72e.state", "1_73s.state", "1_74w.state"])
def dungeon_emu(request):
    """Provide a ZeldaFixture for various dungeon rooms."""
    fixture = ZeldaFixture(request.param)
    yield fixture
    fixture.close()
```

### Button Constants

```python
# NES button mapping for retro's MultiBinary(9) action space
BTN_B      = 0  # B button (sword)
BTN_Y      = 1  # (unused on NES)
BTN_SELECT = 2
BTN_START  = 3
BTN_UP     = 4
BTN_DOWN   = 5
BTN_LEFT   = 6
BTN_RIGHT  = 7
BTN_A      = 8  # A button (use item)
```

---

## Test File Organization

```
tests/
├── conftest.py                  # Shared fixtures (ZeldaFixture, dungeon_emu, etc.)
├── asm_addresses.py             # Assembly address constants
├── utilities.py                 # Existing (kept for backward compat)
│
├── test_ram_mapping.py          # Layer 0: Address verification
├── test_health.py               # Layer 1: Health encoding/decoding
├── test_object_model.py         # Layer 1: Object model (enemies, items, projectiles)
├── test_weapons.py              # Layer 2: Weapon state machines
├── test_boundaries.py           # Layer 2: Screen lock and movement bounds
├── test_look_ahead.py           # Layer 3: Future damage prediction
├── test_frame_skip.py           # Layer 3: Frame skip timing
│
├── test_hit.py                  # Integration: existing hit tests (renamed from hit_test.py)
├── test_rewards.py              # Integration: existing reward tests (renamed from reward_test.py)
└── test_game_state.py           # Integration: existing game state tests (renamed)
```

Existing tests (`hit_test.py`, `reward_test.py`, etc.) remain as integration tests. The new
test files cover low-level correctness that the integration tests cannot reach.

---

## Layer 0: RAM Address Verification

**File**: `tests/test_ram_mapping.py`  
**Covers**: asm-review.md Area 1 (Memory Address Mapping)  
**Utilities**: `ZeldaFixture`, `AssemblyAddresses`

### Test Strategy

For each address in `zelda_game_data.txt`, verify it matches the assembly's `Variables.inc` by:
1. Loading a savestate where the value at that address is non-zero
2. Reading the raw RAM byte at the expected address
3. Reading the same value via `data.get_value(name)`
4. Asserting they match

### Specific Tests

#### T0.1: Named variable addresses match data.json

```
For every entry in zelda_game_data.txt [memory]:
  - Read raw RAM at the hex address
  - Read via data.get_value(name)
  - Assert they are equal
```

#### T0.2: Object table addresses and lengths

```
For every entry in zelda_game_data.txt [tables]:
  - Verify the table offset matches Variables.inc
  - Verify the table length covers the right number of slots
  - Read table[0] and verify it equals the named Link variable (e.g., obj_pos_x[0] == link_x)
```

#### T0.3: Link is always object slot 0

```
Load any savestate:
  - ram[OBJ_X + 0] == data.get_value('link_x')
  - ram[OBJ_Y + 0] == data.get_value('link_y')
  - ram[OBJ_DIR + 0] == data.get_value('link_direction')
  - ram[OBJ_STATE + 0] == data.get_value('link_status')
```

#### T0.4: room_kills / ObjType overlap

```
Load a savestate, kill some enemies:
  - Verify ram[0x34F] serves as both ObjType[0] (Link's type) and room_kills
  - Verify this address is read correctly by both code paths
```

#### T0.5: item_timer / ObjPosFrac union

```
Load a savestate with an active item:
  - Read ram[0x3A8 + index] for an item slot
  - Verify it matches the item timer value
Load a savestate with an active enemy:
  - Read ram[0x3A8 + index] for an enemy slot
  - Verify it is NOT interpreted as a timer
```

#### T0.6: Weapon slot ObjState addresses

```
Verify that each "animation" address is actually ObjState for the correct slot:
  - ram[0xAC + 0x0D] == ram[0xB9]  (sword_animation)
  - ram[0xAC + 0x0E] == ram[0xBA]  (beam_animation)
  - ram[0xAC + 0x0F] == ram[0xBB]  (bait_or_boomerang_animation)
  - ram[0xAC + 0x10] == ram[0xBC]  (bomb_or_flame_animation)
  - ram[0xAC + 0x11] == ram[0xBD]  (bomb_or_flame_animation2)
  - ram[0xAC + 0x12] == ram[0xBE]  (arrow_magic_animation)
```

#### T0.7: Individual enemy health addresses

```
Verify obj_health_1 through obj_health_c map to the correct ObjHP table offsets:
  - data.get_value('obj_health_1') == ram[0x486]  (ObjHP[1])
  - data.get_value('obj_health_2') == ram[0x487]  (ObjHP[2])
  - ... etc through obj_health_c
Note: Verify the gap — obj_health_a is at 0x48F and obj_health_b is at 0x491 (not 0x490).
Is slot $0A at 0x48F and slot $0B at 0x490 or 0x491?
```

---

## Layer 1: Health Encoding and Beams

**File**: `tests/test_health.py`  
**Covers**: asm-review.md Areas 2 (has_beams), 7 (health system)  
**Utilities**: `ZeldaFixture`, `AssemblyAddresses`

### Test Strategy

Directly manipulate `HeartValues` ($66F) and `HeartPartial` ($670) RAM bytes, then verify the
Python `Link` properties interpret them correctly. Also verify `has_beams` by comparing against
the assembly's `MakeSwordShot` check.

### Specific Tests

#### T1.1: Health getter — nibble decoding

```
For hearts_and_containers in [0x00, 0x01, 0x10, 0x11, 0x32, 0x55, 0xF0, 0xFF]:
  For partial_hearts in [0x00, 0x01, 0x7F, 0x80, 0xFF]:
    - Set ram[0x66F] and ram[0x670] directly
    - Build ZeldaGame / Link
    - Verify link.health matches manual calculation:
      filled = hearts_and_containers & 0x0F
      containers = (hearts_and_containers >> 4) + 1
      partial = 0 if val==0, 0.5 if 0<val<0x80, 1.0 if val>=0x80
      expected = min(filled + partial, containers)
```

#### T1.2: Health setter — round-trip consistency

```
For max_health in [1, 3, 8, 16]:
  For health in [0, 0.5, 1, 1.5, max_health-0.5, max_health]:
    - Set link.max_health, then link.health
    - Read back link.health
    - Assert it matches (within 0.5 quantization)
    - Read raw RAM and verify the bytes are consistent
```

#### T1.3: has_beams — assembly-accurate health check

This is the critical test for the `has_beams` discrepancy identified in asm-review.md Area 2.

```
# Case 1: Normal full health (both agree)
Set hearts_and_containers = 0x23 (3 containers, filled=3... wait, that's wrong)
Actually: containers_minus_one=2 means 3 containers, filled=3 means 3 full hearts
Hmm, but assembly checks filled == containers_minus_one.

Let's be precise:
  hearts_and_containers = 0x23 → containers_minus_one=2, filled=3
  Assembly check: filled(3) == containers_minus_one(2)? NO → no beams
  Python check: max_health=3, health=3.0+partial → depends on partial

Actually the correct full health state for 3 containers is:
  hearts_and_containers = 0x22 → containers_minus_one=2, filled=2
  partial_hearts = 0xFF
  Assembly: filled(2) == containers_minus_one(2) AND partial(0xFF) >= 0x80 → YES beams
  Python: health = 2 + 1.0 = 3.0, max_health = 3 → full → YES beams

# Case 2: Edge case (Python says full, assembly says no)
  hearts_and_containers = 0x23 → containers_minus_one=2, filled=3
  partial_hearts = 0x00
  Assembly: filled(3) == containers_minus_one(2)? NO → no beams
  Python: health = min(3 + 0, 3) = 3.0, max_health = 3 → full → YES beams (BUG!)

# Case 3: Half heart short
  hearts_and_containers = 0x22 → containers_minus_one=2, filled=2
  partial_hearts = 0x7F
  Assembly: filled(2) == containers_minus_one(2) AND partial(0x7F) >= 0x80? NO → no beams
  Python: health = 2 + 0.5 = 2.5, max_health = 3 → not full → no beams (agree)

Test: Set up each case, fire sword, verify whether beam actually fires in the emulator.
Compare against link.has_beams.
```

#### T1.4: has_beams — screen lock interaction

```
For several positions near screen edges:
  - Place Link, set full health, set sword
  - Verify has_beams returns False when at edge
  - Verify has_beams returns True when away from edge
```

#### T1.5: has_beams — beam already active

```
- Fire a beam
- While beam is in flight (beam_animation == 16), verify has_beams returns False
- While beam is spreading (beam_animation == 17), verify has_beams returns False
- After beam finishes (beam_animation == 0), verify has_beams returns True
```

#### T1.6: 16-heart special case

```
Set hearts_and_containers = 0xFF (15 containers_minus_one = 16 max, 15 filled)
Set partial_hearts = 0xFF
Verify link.max_health == 16
Verify link.health == 16.0
Verify is_health_full == True
```

---

## Layer 1: Object Model

**File**: `tests/test_object_model.py`  
**Covers**: asm-review.md Areas 4 (enemy health), 5 (enemy states), 6 (object classification),
9 (direction), 11 (enemy IDs)  
**Utilities**: `ZeldaFixture`, `AssemblyAddresses`

### Test Strategy

Load savestates with known enemies, verify the Python object model reads RAM correctly by
cross-referencing raw RAM bytes.

### Specific Tests

#### T2.1: Enemy health — high nibble extraction

```
Load a savestate with enemies:
  For each enemy slot 1-B:
    - Read raw ram[OBJ_HP + slot]
    - Verify enemy.health == ram_value >> 4
    - Also note the low nibble value to understand its purpose
```

#### T2.2: Enemy health — damage application

```
Load a savestate with enemies, give Link a sword:
  - Record enemy health before
  - Hit enemy with sword
  - Record enemy health after
  - Verify the health decreased by the expected sword damage
  - Verify the high nibble of ObjHP decreased accordingly
```

#### T2.3: Object ID classification — boundary values

```
For obj_id in range(0, 0x80):
  Manually classify using _is_id_enemy() and _is_projectile()
  Verify:
    - IDs 1-0x48 (except 0x40) → enemy
    - ID 0x40 → neither (verify what this actually is)
    - ID 0x60 → item (OBJ_ITEM_ID)
    - IDs 0x49-0x7F (except 0x60, 0x63, 0x64, 0x68, 0x6A) → projectile
    - Excluded projectile IDs (0x63, 0x64, 0x68, 0x6A) → neither (verify what these are)
```

#### T2.4: Object enumeration — slot range

```
Load a savestate with enemies:
  - Verify _enumerate_active_ids only checks slots 1-B (not 0, not C+)
  - Verify it skips slots with ObjType == 0
  - Verify each returned (index, obj_id) matches ram[OBJ_TYPE + index]
```

#### T2.5: Enemy is_dying — spawn state values

```
Load savestate, kill an enemy with a single lethal hit:
  Step frame by frame (using ZeldaFixture, not FrameSkipWrapper):
  - Record ObjMetastate[slot] each frame
  - Verify the death sequence: observe what values the metastate takes
  - Verify is_dying is True for exactly the right range
  - Determine the actual upper bound (is it 19? higher?)
```

#### T2.6: Enemy is_active — Lever/Zora cycle

```
Load a savestate with a Lever (RedLever or BlueLever) enemy:
  Step frame by frame:
  - Record ObjState[slot] each frame
  - Verify the cycle: surfacing → up (state 3) → submerging → hidden
  - Verify is_active is True only when state == 3
```

#### T2.7: Enemy is_active — WallMaster

```
Load a savestate with a WallMaster:
  Step frame by frame:
  - Record ObjState[slot] each frame
  - Verify is_active is True only when state == 1
```

#### T2.8: Enemy is_active — default (spawn_state based)

```
Load savestate with regular enemies (Stalfos, Keese, etc.):
  - Verify is_active when spawn_state == 0
  - Force spawn_state to non-zero, verify is_active is False
```

#### T2.9: Direction encoding

```
For slots 0-B:
  - Read ram[OBJ_DIR + slot]
  - Verify Direction.from_ram_value matches:
    1 → E, 2 → W, 4 → S, 8 → N, other → NONE
  - Step many frames across multiple savestates
  - Verify no slot ever contains a composite direction (3, 5, 6, 9, 10, 12)
```

#### T2.10: ZeldaEnemyKind — duplicate Octorok values

```
Verify ZeldaEnemyKind.Octorok.value == ZeldaEnemyKind.OctorokFast.value == 0x07
Determine from assembly whether this is intentional (same sprite, different behavior flag)
or if OctorokFast should have a different value.
```

#### T2.11: Position reading

```
Load savestate:
  For each active object slot:
    - Verify object.position.x == ram[OBJ_X + slot]
    - Verify object.position.y == ram[OBJ_Y + slot]
```

---

## Layer 2: Weapon State Machines

**File**: `tests/test_weapons.py`  
**Covers**: asm-review.md Area 2 (beam state machine), Area 12 (frame skip cooldowns)  
**Utilities**: `ZeldaFixture`, `RAMWatcher`, `AssemblyAddresses`

### Test Strategy

Use `ZeldaFixture` with `RAMWatcher` to trace weapon slot ObjState values frame-by-frame
after firing each weapon. Verify the state machine matches the assembly documentation.

### Specific Tests

#### T3.1: Sword shot lifecycle — full trace

```
Load dungeon savestate with enemies in beam range:
  - Set full health, wood sword
  - Set RNG to fixed seed
  - Fire sword (press B button facing enemy direction)
  - Watch ram[OBJ_STATE + SLOT_BEAM] (= ram[0xBA]) every frame
  - Expected trace:
    Frame 0: 0x00 (inactive)
    Frame N: 0x10 (flying) — beam appears
    Frame M: 0x11 (spreading) — beam hit something or reached max range
    Frame M+~10: 0x00 (inactive) — spread finished
  - Record exact frame counts for flying and spreading durations
```

#### T3.2: Sword shot spread duration

```
Same setup as T3.1:
  - Count exact frames where ObjState[$0E] == 0x11
  - Verify it matches assembly prediction (~10 frames from ObjDir countdown $FE→$F4)
  - This validates the 11-frame hack in frame_skip_wrapper.py
```

#### T3.3: Magic rod shot lifecycle

```
Give Link the magic rod:
  - Fire rod, watch ObjState[$0E] (same slot as beam)
  - Expected: 0x00 → 0x80 (magic shot flying) → 0x81 (magic shot spreading)
  - Verify these values don't collide with beam interpretation
  - Verify get_animation_state correctly distinguishes rod from beam
```

#### T3.4: Sword slash lifecycle

```
Load savestate, give Link a sword (not full health, no beams):
  - Press B, watch ObjState[$0D] (= ram[0xB9]) every frame
  - Expected: States 1→2→3→4→5→6→0 (from assembly)
  - Count frames in each state
  - Verify total matches ATTACK_COOLDOWN constant (15 frames)
```

#### T3.5: Bomb lifecycle

```
Give Link bombs, place a bomb:
  - Watch ObjState[$10] (= ram[0xBC]) every frame
  - Record the full state sequence
  - Verify ANIMATION_BOMBS_ACTIVE (18) and ANIMATION_BOMBS_EXPLODED (19, 20) match
```

#### T3.6: Arrow lifecycle

```
Give Link bow + arrows + rupees:
  - Fire arrow, watch ObjState[$12] (= ram[0xBE]) every frame
  - Record the full state sequence
  - Verify ANIMATION_ARROW_ACTIVE (10), ANIMATION_ARROW_HIT (20), ANIMATION_ARROW_END (21)
```

#### T3.7: Boomerang lifecycle

```
Give Link boomerang:
  - Throw boomerang, watch ObjState[$0F] (= ram[0xBB]) every frame
  - Record the full state sequence
  - Verify ANIMATION_BOOMERANG_MIN (10) through ANIMATION_BOOMERANG_MAX (57)
  - Verify the state increases during outbound and decreases during return
```

#### T3.8: Beam stuck-at-17 reproduction

```
This test should reproduce the known bug where beam_animation stays at 17:
  - Fire beams repeatedly in rapid succession
  - Run look-ahead prediction after each fire
  - Check if ObjState[$0E] gets stuck at 17 (0x11)
  - Compare behavior with and without look-ahead to isolate whether the look-ahead causes it

If the bug is in the look-ahead:
  - Fire beam, let look-ahead run
  - Check ObjState[$0E] after look-ahead restore
  - Verify it matches the pre-look-ahead value
```

#### T3.9: Animation state interpretation consistency

```
For each weapon type (BEAMS, BOMB_1, BOMB_2, ARROW, BOOMERANG):
  For each possible ObjState value (0-255):
    - Set the raw RAM byte
    - Call get_animation_state()
    - Verify the return value matches expected:
      INACTIVE for values outside the documented active/hit ranges
      ACTIVE for documented active values
      HIT for documented hit values
```

---

## Layer 2: Screen Lock Boundaries

**File**: `tests/test_boundaries.py`  
**Covers**: asm-review.md Area 3 (sword screen lock)  
**Utilities**: `ZeldaFixture`, `AssemblyAddresses`

### Test Strategy

Place Link at specific coordinates and verify both the Python `is_sword_screen_locked` and
the actual NES behavior (whether pressing B fires a sword or not).

### Specific Tests

#### T4.1: Screen lock — overworld boundary sweep

```
Load an overworld savestate:
  For x in range(0, 256):
    For y in [0x44, 0x45, 0x46, 0x80, 0xD4, 0xD5, 0xD6]:
      - Set link_x = x, link_y = y
      - Check is_sword_screen_locked
      - Press B (sword button)
      - Check if ObjState[$0D] changed (sword activated)
      - Verify Python matches NES reality
```

#### T4.2: Screen lock — underworld boundary sweep

```
Load a dungeon savestate:
  Same sweep as T4.1 but with dungeon-relevant coordinate ranges
  Compare against Python's UW bounds: x<=0x10, x>=0xD9, y<=0x53, y>=0xC5
```

#### T4.3: Screen lock — verify assembly inner bounds

```
The assembly's inner BorderBounds are: down=$BE, up=$54, right=$D1, left=$1F
These should be the A-button masking bounds.

Load savestate (try both OW and UW):
  Place Link at inner boundary coordinates:
    - (0x1F, 0x80): left inner bound
    - (0xD1, 0x80): right inner bound
    - (0x80, 0x54): top inner bound
    - (0x80, 0xBE): bottom inner bound
  Press A (item button):
    - Verify whether the A button is masked at these exact coordinates
    - This tells us the comparison operators (< vs <=)
```

#### T4.4: Screen lock — verify which bound set applies where

```
Test whether inner bounds apply to BOTH overworld and underworld or just underworld:
  Load OW savestate, test at inner bound coordinates
  Load UW savestate, test at inner bound coordinates
  Load OW savestate, test at outer OW bound coordinates
  Load UW savestate, test at outer UW bound coordinates
  Determine which set is used for A-button masking in each context
```

#### T4.5: get_sword_directions_allowed consistency

```
Verify get_sword_directions_allowed matches is_sword_screen_locked:
  - When is_sword_screen_locked is True, directions should be empty or restricted
  - When False, all 4 directions should be available
  - Test at boundary coordinates where only some directions are locked
```

---

## Layer 3: Look-Ahead Simulation

**File**: `tests/test_look_ahead.py`  
**Covers**: asm-review.md Area 8 (look-ahead simulation)  
**Utilities**: `ZeldaFixture`, `ZeldaActionReplay`

### Test Strategy

Test the `_predict_future_effects` method for correctness of state save/restore, damage
attribution, and weapon isolation. Uses both low-level (ZeldaFixture) and high-level
(ZeldaActionReplay) access.

### Specific Tests

#### T5.1: State restore completeness — RAM integrity

```
Using ZeldaFixture:
  - Save full RAM snapshot (all 2KB)
  - Call the same pattern as _predict_future_effects:
    savestate = em.get_state()
    data.set_value('beam_animation', 0)     # disable
    data.set_value('hearts_and_containers', 0xFF)  # override health
    step N frames
    em.set_state(savestate)
  - Save full RAM snapshot again
  - Compare byte-by-byte
  - Any differences indicate the restore is incomplete
```

#### T5.2: State restore completeness — weapon slot verification

```
Using ZeldaFixture:
  - Set beam_animation to some known non-zero value (e.g., 16)
  - Save state
  - Zero beam_animation via data.set_value
  - Restore state
  - Verify beam_animation is back to 16
  Repeat for all weapon slots
```

#### T5.3: Look-ahead doesn't corrupt active weapons

```
Using ZeldaActionReplay:
  - Fire beams and bombs simultaneously
  - After the step (which triggers look-ahead), verify:
    - Beam is still in expected state (not zeroed)
    - Bomb is still in expected state (not zeroed)
    - Enemy health changes are attributed correctly
```

#### T5.4: Look-ahead damage attribution — single weapon

```
Using ZeldaActionReplay:
  - Fire beams at an enemy
  - Verify state_change.enemies_hit attributes damage to the correct enemy
  - Verify state_change.damage_dealt matches the actual damage
  - Step forward, verify the damage isn't double-counted (discount system)
```

#### T5.5: Look-ahead damage attribution — multiple weapons

```
Using ZeldaActionReplay:
  - Fire beams in step 1 (look-ahead predicts beam damage)
  - Fire bomb in step 2 while beams still active (look-ahead predicts bomb damage)
  - Verify beam damage attributed to step 1, bomb damage attributed to step 2
  - Verify no cross-contamination between predictions
```

#### T5.6: Look-ahead with room transition

```
Using ZeldaActionReplay:
  - Fire beams near room exit
  - Walk through exit on next step
  - Verify the look-ahead doesn't carry over damage from old room
  - Verify discounts are cleared on room change
```

#### T5.7: Look-ahead health preservation

```
Using ZeldaFixture:
  - Set Link health to 1 heart (near death)
  - Simulate the look-ahead pattern (save, override health to 0xFF, step, restore)
  - Verify Link's health is back to 1 heart after restore
  - Verify the game doesn't think Link died during look-ahead
```

#### T5.8: _disable_others — verify all weapons covered

```
For each equipment type in [BEAMS, BOMB_1, BOMB_2, ARROW, BOOMERANG]:
  - Verify _disable_others zeros exactly the other 4 weapon slots
  - Verify it leaves the current weapon slot untouched
```

#### T5.9: Look-ahead trigger conditions

```
Verify _handle_future_effects only triggers when:
  - Previous animation was NOT ACTIVE and current IS ACTIVE (newly fired)
  Does NOT trigger when:
  - Both previous and current are ACTIVE (weapon still in flight from prior step)
  - Previous was ACTIVE, current is HIT (weapon transitioning)
  - Previous was HIT, current is INACTIVE (weapon finishing)
```

---

## Layer 3: Frame Skip Timing

**File**: `tests/test_frame_skip.py`  
**Covers**: asm-review.md Area 12 (frame skip cooldowns)  
**Utilities**: `ZeldaFixture`, `RAMWatcher`

### Test Strategy

Use `ZeldaFixture` with frame-by-frame stepping to measure exact timing of game actions,
then verify the FrameSkipWrapper constants are correct.

### Specific Tests

#### T6.1: Sword cooldown — frame count

```
Fire sword (no beams), count frames until Link can act again:
  - Watch ObjState[$0D] (sword slot)
  - Count frames from press to ObjState returning to 0
  - Compare against ATTACK_COOLDOWN = 15
  - Determine if 15 is exact, conservative, or too short
```

#### T6.2: Item cooldown — frame count for each item type

```
For each item (boomerang, bow, bomb, candle, etc.):
  - Use item, count frames until actionable
  - Compare against ITEM_COOLDOWN = 10
  - Identify items where 10 is insufficient
```

#### T6.3: Movement tile-snapping — symmetric timing

```
For each direction (N, S, E, W):
  - Move Link one tile in that direction
  - Count exact frames from button press to tile boundary crossing
  - Compare N vs S and E vs W
  - Verify WS_ADJUSTMENT_FRAMES = 4 compensates correctly for south/west
  - Determine the actual asymmetry cause (if any)
```

#### T6.4: Beam stuck-at-17 — with frame_skip_wrapper

```
Using ZeldaActionReplay (with FrameSkipWrapper active):
  - Fire beams repeatedly
  - After each fire, check beam_animation
  - Verify the 11-frame hack correctly clears stuck state
  - Verify it doesn't clear the state prematurely (cutting spread short)
```

#### T6.5: Movement — stuck detection

```
Move Link into a wall:
  - Verify MAX_MOVEMENT_FRAMES (16) is hit
  - Verify the stuck detection correctly identifies no position change
  - Verify Link's position is unchanged after stuck detection
```

---

## Tile Layout Tests

**File**: `tests/test_object_model.py` (or separate `tests/test_tiles.py`)  
**Covers**: asm-review.md Area 10

### Specific Tests

#### T7.1: Tile layout address and size

```
Verify tile_layout table: offset=0xD30, length=0x2C0 (704 bytes = 32×22)
Load savestate, read ram[0xD30:0xD30+0x2C0]
Compare against ZeldaGame.current_tiles (flattened)
```

#### T7.2: Tile layout reshape correctness

```
Load savestate with known tile pattern (e.g., dungeon room with walls):
  - Read tiles via current_tiles
  - Verify tiles[x][y] correctly represents the tile at screen position (x, y)
  - Check a few known positions: walls should be at edges, floor in center
```

---

## Sound Tests

**File**: `tests/test_object_model.py` (or `tests/test_sound.py`)  
**Covers**: asm-review.md Area 13

### Specific Tests

#### T8.1: Sound register identification

```
Trigger each sound event and verify ram[0x605] has the expected bitmask:
  - Block a projectile → verify ArrowDeflected (0x01) is set
  - Stun enemy with boomerang → verify BoomerangStun (0x02) is set
  - Pick up a key → verify KeyPickup (0x08) is set
  - Pick up a heart → verify SmallHeartPickup (0x10) is set
  - Place a bomb → verify SetBomb (0x20) is set
```

#### T8.2: is_sound_playing bitmask check

```
Set ram[0x605] to various values:
  - 0x01: only ArrowDeflected playing
  - 0x03: ArrowDeflected + BoomerangStun
  - 0xFF: all playing
  - 0x00: none playing
Verify is_sound_playing returns correct True/False for each SoundKind
```

---

## Integration Tests (Existing, to maintain)

The existing tests in `hit_test.py`, `reward_test.py`, `link_test.py`, `game_state_test.py`
serve as end-to-end integration tests. They should be kept (possibly renamed to `test_*.py`
for pytest discovery) but are not sufficient on their own because they:

- Don't verify raw RAM interpretation
- Don't test frame-level timing
- Can't isolate which layer caused a failure
- Don't cover edge cases or boundary conditions

---

## Creating New Savestates

Some tests may need savestates we don't currently have. To create a new savestate:

1. Run `run.py` with an appropriate model/scenario
2. Play to the desired game state
3. Save via the emulator's savestate function
4. Place the `.state` file in `triforce/custom_integrations/Zelda-NES/`

Savestates we may need:
- **Overworld with enemies**: For screen lock boundary tests (Level 0)
- **Room with Levers**: For `is_active` state cycle tests
- **Room with Zoras**: For `is_active` underwater cycle tests
- **Room with WallMasters**: For `is_active` wall hiding tests
- **Room with single enemy at known position**: For precise damage tests
- **Room near exit**: For look-ahead room transition tests

---

## Execution Order

Tests should be run in layer order. If Layer 0 fails, Layer 1+ results are unreliable:

1. **Layer 0**: RAM address mapping — if these fail, everything else is wrong
2. **Layer 1**: Health + Object model — verify our interpretation of RAM
3. **Layer 2**: Weapons + Boundaries — verify game logic understanding
4. **Layer 3**: Look-ahead + Frame skip — verify wrapper correctness
5. **Integration**: Existing tests — verify end-to-end behavior

Use pytest markers to enforce this:

```python
@pytest.mark.layer0
def test_ram_addresses(): ...

@pytest.mark.layer1
def test_health_encoding(): ...
```

---

## Test Naming Convention

Tests follow the pattern: `test_{area}_{specific_thing}`.

Examples:
- `test_health_nibble_decoding`
- `test_beam_lifecycle_flying_to_spreading`
- `test_screen_lock_overworld_x_boundary`
- `test_look_ahead_state_restore_integrity`
- `test_enemy_is_dying_metastate_range`
