# Assembly Verification Review Plan

This document catalogs every piece of the Python game state model (ZeldaGame, Link, Enemy, etc.)
that reads or interprets NES RAM, and maps it to the assembly source in `zelda-asm/src/` for
correctness verification. It also covers the look-ahead simulation in `_handle_future_effects`.

## Key Reference Files

| Python File | Assembly/Data Reference |
|---|---|
| `triforce/zelda_game_data.txt` | `zelda-asm/src/Variables.inc` (canonical addresses) |
| `triforce/link.py` | `Z_05.asm` (Link input/movement), `Z_07.asm` (weapons) |
| `triforce/enemy.py` | `Z_04.asm` (monster update), `Z_01.asm` (collision/damage) |
| `triforce/zelda_game.py` | `Variables.inc`, `Z_05.asm` (object enumeration) |
| `triforce/frame_skip_wrapper.py` | `Z_07.asm` (weapon state machines) |
| `triforce/state_change_wrapper.py` | `Z_01.asm` (damage), `Z_07.asm` (weapon lifecycle) |

## Understanding the Object Slot Layout

The assembly uses a flat array of object slots indexed by slot number. Each "table"
(ObjX, ObjY, ObjState, ObjHP, etc.) is a contiguous array where `table[slot]` gives
the value for that slot. Key slot assignments:

| Slot | Purpose | ObjState Address |
|---|---|---|
| 0 | Link | $AC |
| 1–B | Enemies/Items (up to 11 objects) | $AD–$B7 |
| C (12) | — | $B8 |
| D (13) | Sword / Rod | $B9 (`sword_animation`) |
| E (14) | Sword Shot / Magic Shot | $BA (`beam_animation`) |
| F (15) | Boomerang / Food | $BB (`bait_or_boomerang_animation`) |
| 10 (16) | Bomb / Fire 1 | $BC (`bomb_or_flame_animation`) |
| 11 (17) | Bomb / Fire 2 | $BD (`bomb_or_flame_animation2`) |
| 12 (18) | Rod / Arrow | $BE (`arrow_magic_animation`) |

This means what `zelda_game_data.txt` calls `beam_animation` (address $BA) is really
`ObjState[14]` — the state byte of the sword shot object in slot $0E. Similarly all
the "animation" fields are just ObjState for weapon slots.

---

## Area 1: Memory Address Mapping (`zelda_game_data.txt` vs `Variables.inc`)

Every address in `zelda_game_data.txt` needs to be verified against the assembly's
`Variables.inc`. This is mostly a mechanical 1:1 check, but some are non-obvious.

### Known mappings to verify

| Data File Entry | Address | Assembly Symbol | Notes |
|---|---|---|---|
| `level` | $010 | `CurLevel` | ✓ straightforward |
| `location` | $0EB | `RoomId` | ✓ |
| `mode` | $012 | `GameMode` | ✓ |
| `link_x` / `link_y` | $070 / $084 | `ObjX[0]` / `ObjY[0]` | Link is slot 0 |
| `link_status` | $0AC | `ObjState[0]` | Link's state byte |
| `link_direction` | $098 | `ObjDir[0]` | Link's direction |
| `sword_animation` | $0B9 | `ObjState[$0D]` | Sword/Rod slot |
| `beam_animation` | $0BA | `ObjState[$0E]` | Sword shot slot |
| `bait_or_boomerang_animation` | $0BB | `ObjState[$0F]` | Boomerang/Food slot |
| `bomb_or_flame_animation` | $0BC | `ObjState[$10]` | Bomb/Fire 1 slot |
| `bomb_or_flame_animation2` | $0BD | `ObjState[$11]` | Bomb/Fire 2 slot |
| `arrow_magic_animation` | $0BE | `ObjState[$12]` | Rod/Arrow slot |
| `hearts_and_containers` | $66F | `HeartValues` | ✓ |
| `partial_hearts` | $670 | `HeartPartial` | ✓ |
| `triforce` | $671 | `InvTriforce` | ✓ |
| `triforce_of_power` | $672 | `LastBossDefeated` | **Name mismatch — verify semantics** |
| `sound_pulse_1` | $605 | `Tune0` | **Verify this is the right sound register** |
| `kill_streak` | $627 | `WorldKillCount` | ✓ |
| `room_kills` | $34F | `ObjType[0]`? | **Conflicts with ObjType — verify** |
| `obj_stun_timer` table | $03D | `ObjStunTimer` | ✓ |
| `obj_id` table | $34F | `ObjType` | ✓ |
| `obj_spawn_state` table | $405 | `ObjMetastate` | ✓ |
| `obj_health` table | $485 | `ObjHP` | ✓ |
| `item_timer` table | $3A8 | `ObjPosFrac`? | **SUSPICIOUS — $3A8 is ObjPosFrac in Variables.inc, but CommonVars.inc has `Item_ObjItemLifetime := $3A8`** |
| All inventory items ($657–$676) | Various | `Items`, `InvBombs`, etc. | Spot-check a few |

### Specific concerns

1. **`room_kills` at $34F overlaps `ObjType`** — `ObjType := $34F` in Variables.inc.
   `room_kills` and `ObjType[0]` are the same address. Is this intentional
   (Link's "type" slot doubles as kill count)? Verify this is read at the right time.

2. **`item_timer` at $3A8** — In Variables.inc this is `ObjPosFrac`, but in
   `CommonVars.inc` it's `Item_ObjItemLifetime := $3A8`. This is a union — different
   object types use the same memory differently. Need to verify the item timer is only
   read for item slots and not for enemy slots.

3. **`triforce_of_power` at $672** — Assembly calls this `LastBossDefeated`. Verify
   the semantics are actually "has triforce of power" vs "defeated last boss".

4. **`sound_pulse_1` at $605** — Assembly calls this `Tune0`. Verify the `SoundKind`
   enum values (ArrowDeflected=0x01, etc.) correspond to the correct bitmask for
   this register.

---

## Area 2: Beam / Sword Shot State Machine

This is the highest-risk area and the one with known bugs.

### Assembly state machine (slot $0E = `ObjState[$0E]` = `beam_animation`)

From `Z_07.asm`, the sword shot lifecycle:

```
$00       → Inactive (no shot)
$10 (16)  → Sword shot is flying (MakeSwordShot sets this)
$11 (17)  → Sword shot is spreading/dissipating (SetShotSpreadingState: INC ObjState,X)
            The spread lasts ~10 frames (ObjDir counts down from $FE to $F4), then → $00
$80 (128) → Magic shot (rod) is flying
$81 (129) → Magic shot spreading/fire
```

### Python interpretation (`link.py`)

```python
ANIMATION_BEAMS_ACTIVE = 16   # $10 — matches
ANIMATION_BEAMS_HIT = 17      # $11 — matches (spreading state)
```

`get_animation_state(ZeldaAnimationKind.BEAMS)`:
- Returns `ACTIVE` if beam_animation == 16
- Returns `HIT` if beam_animation == 17
- Returns `INACTIVE` otherwise

### Known bug: Beam animation stuck at 17

In `frame_skip_wrapper.py` (lines 110-115), there's a workaround:

```python
if info['beam_animation'] == 17:
    self._sword_count += 1
    if self._sword_count >= 11:
        info['beam_animation'] = 0
```

This force-resets beam_animation to 0 after 11 consecutive frames at value 17. The
assembly's `SpreadShot` routine should naturally reset to 0 after ~10 frames of spread.

**Questions to investigate:**
1. Does the save/restore in `_predict_future_effects` corrupt the ObjState[$0E]? The
   method saves emulator state, runs the simulation forward, then restores. But it also
   calls `_disable_others` which sets other animation slots to 0 via `data.set_value`.
   If the emulator state restore doesn't undo `set_value` changes, this could leave
   stale values.
2. Is there a race condition where the beam state transitions from 16→17 during the
   look-ahead simulation, and the state restore doesn't properly reset it?
3. The SpreadShot routine in the assembly uses `ObjDir[$0E]` as a decrementing counter.
   If the look-ahead simulation modifies this and the restore is incomplete, the spread
   might never finish.

### `has_beams` check

Python (`link.py:200-205`):
```python
@property
def has_beams(self) -> bool:
    if self.sword == SwordKind.NONE or not self.is_health_full or self.is_sword_screen_locked:
        return False
    return self.get_animation_state(ZeldaAnimationKind.BEAMS) == AnimationState.INACTIVE
```

Assembly (`MakeSwordShot` in `Z_07.asm:4616-4656`):
```asm
; Check: filled hearts == containers - 1
LDA HeartValues
AND #$0F          ; filled_hearts (low nibble)
STA $00
LDA HeartValues   ; (reloaded via PLA after PHA)
LSR x4            ; containers_minus_one (high nibble)
CMP $00           ; filled_hearts == containers_minus_one?
BNE Exit          ; No → no beams

; Check: partial_hearts >= $80
LDA HeartPartial
CMP #$80
BCC Exit          ; < $80 → no beams
```

**Potential discrepancy**: The assembly checks `filled_hearts == containers_minus_one`
AND `partial >= $80`. The Python checks `is_health_full` which computes a float and
compares to `max_health`. In the normal full-health state the game stores
`filled = containers-1, partial = 0xFF`, so both agree. But if RAM were in the state
`filled = containers, partial = 0` (which Python treats as full health), the assembly
would say **no beams** while Python says **has beams**. Verify whether this state can
actually occur during gameplay or only via `per_reset`/`per_frame` RAM overrides.

---

## Area 3: Sword Screen Lock Boundaries

### Python (`link.py:213-220`)

```python
@property
def is_sword_screen_locked(self) -> bool:
    x, y = self.position
    if self.game.level == 0:
        return x < 0x7 or x > 0xe8 or y < 0x45 or y > 0xd5
    return x <= 0x10 or x >= 0xd9 or y <= 0x53 or y >= 0xc5
```

### Assembly (`Z_05.asm:2728-2730`)

```
BorderBounds:
    .BYTE $D6, $45, $E9, $07    ; OW outer: down, up, right, left
    .BYTE $C6, $55, $D9, $17    ; UW outer: down, up, right, left
    .BYTE $BE, $54, $D1, $1F    ; Inner bounds (used for A-button masking)
```

The assembly's `Link_FilterInput` uses these bounds to mask out the A button (attack)
when Link crosses the **inner** boundary. The **outer** boundary controls screen
scrolling. The **inner** boundary is what prevents sword use:
- Inner: down=$BE, up=$54, right=$D1, left=$1F

But `is_sword_screen_locked` in Python uses:
- OW: x<7, x>$E8, y<$45, y>$D5 — these look like the **outer OW** bounds
- UW: x<=$10, x>=$D9, y<=$53, y>=$C5 — mixes outer UW and inner bounds

**This needs careful verification.** The Python bounds should match the assembly's
inner bounds (the A-button masking set), not the screen-scroll bounds. The inner
bounds are: y_down=$BE, y_up=$54, x_right=$D1, x_left=$1F. But Python uses entirely
different values. Need to trace `MaskInputInBorder` to understand the exact comparison
operators (< vs <=) used in the assembly.

---

## Area 4: Enemy Health Encoding

### Python (`zelda_game.py:294`)

```python
health = int(tables.read("obj_health")[index] >> 4)
```

This reads `ObjHP` ($485+index) and shifts right by 4, treating the **high nibble**
as the health value.

### Assembly

Need to verify how `ObjHP` is structured. When damage is dealt in `Z_01.asm`,
the code does:
```asm
LDA ObjHP, X
; ... subtract damage ...
STA ObjHP, X
```

**Verify**: Is the high nibble always the HP? The low nibble might be used for damage
type or invincibility info. Check the initial HP assignment for various enemies and
the damage subtraction routine to confirm the >> 4 is correct.

---

## Area 5: Enemy `is_dying` and `is_active` State Machine

### `is_dying` (`enemy.py:31-36`)

```python
@property
def is_dying(self) -> bool:
    return 16 <= self.spawn_state <= 19
```

`spawn_state` reads from `ObjMetastate` ($405+index). In the assembly, metastate
values >= $10 (16) likely represent the death cloud animation. But the upper bound of
19 was determined by "trial and error" per the code comment.

**Verify**: What are the actual metastate values during the death sequence? The assembly
sets `ObjMetastate` to 1 for the initial cloud state (`Z_05.asm:1690`), then increments
it. What is the final value before the object is removed? Is 19 correct, or could it
be higher/lower?

### `is_active` (`enemy.py:39-54`)

```python
@property
def is_active(self) -> bool:
    status = self.status & 0xff
    if self.id in (RedLever, BlueLever, Zora):
        return status & 0xff == 3
    if self.id == WallMaster:
        return status == 1
    return not self.spawn_state
```

- **Levers/Zora**: ObjState == 3 means "surfaced". Verify against the Lever/Zora
  update routines in the assembly (they cycle through states like 0→1→2→3→2→1→0).
- **WallMaster**: ObjState == 1 means "active on screen". Verify against
  `Wallmaster_ObjStep` and the WallMaster update routine.
- **Default**: `spawn_state == 0` (ObjMetastate == 0) means fully spawned. This is
  consistent with the cloud spawn sequence, but verify there aren't other metastate
  values that also mean "active" for certain enemy types.

---

## Area 6: Object Classification (Enemy / Projectile / Item)

### Python (`zelda_game.py:278-283`)

```python
def _is_id_enemy(self, obj_id):
    return 1 <= obj_id <= 0x48 and obj_id != 0x40

def _is_projectile(self, obj_id):
    return obj_id > 0x48 and obj_id != 0x60 and obj_id != 0x63 \
           and obj_id != 0x64 and obj_id != 0x68 and obj_id != 0x6a
```

**Item detection**: `OBJ_ITEM_ID = 0x60` — items always have ObjType == $60.

**Verify**:
- What is ObjType $40? (Excluded from enemies — possibly a boss segment or special?)
- Are the projectile exclusions ($60, $63, $64, $68, $6A) correct? What are these IDs?
  $60 = item, but what are $63, $64, $68, $6A?
- Is the boundary of $48 between enemies and projectiles correct? Are there enemies
  above $48 or projectiles below $48?

---

## Area 7: Link's Health System (hearts_and_containers)

### Encoding (address $66F = `HeartValues`)

```
High nibble = containers_minus_one (i.e. max_hearts - 1)
Low nibble  = fully_filled_hearts
```

Address $670 = `HeartPartial`:
```
$00       = empty
$01-$7F   = half heart
$80-$FF   = full heart
```

### Python implementation (`link.py:44-134`)

The getter/setter for health and max_health need to be verified against the assembly
for edge cases:

1. **16-heart case**: Python special-cases `value >= 15.99` → `partial=0xFF, filled=15`.
   The encoding allows max 16 containers (nibble F+1), with filled=15 and partial=0xFF.
   Verify this is how the game actually represents 16 hearts.

2. **Setter consistency**: When setting health to exactly max, verify the stored
   `(filled, partial)` tuple matches what the game expects for "full health" so that
   `has_beams` works correctly.

---

## Area 8: The Look-Ahead Simulation (`_handle_future_effects`)

This is the second highest-risk area. `state_change_wrapper.py:146-219`.

### How it works

When beams/bombs/arrows/boomerang are newly fired (transition from INACTIVE → ACTIVE),
the code:
1. Saves the emulator state via `em.get_state()`
2. Disables all *other* weapon slots by zeroing their ObjState
3. Steps the emulator forward until the weapon deactivates (INACTIVE) or the room changes
4. Compares enemy health between start and end to attribute damage
5. Stores damage into both `self.__dict__` (current reward) and `discounts` (prevent double-counting)
6. Restores emulator state via `em.set_state()`

### Risk areas

1. **State restore completeness**: The comment says "We only touched data.values, so
   we should be clear of any modifications." But `_disable_others` calls
   `data.set_value()` to zero out other weapon slots. Does `em.set_state()` fully
   restore all RAM, or only the emulator's internal state? If `data.set_value` writes
   go through a different path than the emulator's RAM, they might persist after restore.

2. **Health override during simulation**: Line 179 sets `hearts_and_containers = 0xFF`
   every frame to prevent Link from dying during the look-ahead. This sets filled=15,
   containers=16. When the state is restored, this should be undone. But if the restore
   is incomplete, Link could end up with 16 heart containers.

3. **Disabled weapons persist**: If `_disable_others` zeros out a weapon's ObjState and
   the restore doesn't undo it, that weapon is permanently deactivated for the current
   game step. This could cause the 11-frame beam stuck bug.

4. **Boomerang/arrow item pickup**: The code checks for item pickups during boomerang
   and arrow look-ahead. But items might expire naturally during the simulation. The
   timer comparison `elapsed_frames < item.timer` might not account for the simulated
   frames correctly.

5. **Multiple simultaneous weapons**: If both a bomb and beams are active
   simultaneously, both get simulated independently. But each simulation disables the
   other. Does the order matter? Could bomb simulation accidentally attribute beam kills
   to the bomb?

6. **`_detect_future_damage` trigger condition**: The code only triggers look-ahead when
   `prev_ani != ACTIVE and curr_ani == ACTIVE`. But what if a weapon was already ACTIVE
   from a previous step and fires again? Or what if the weapon transitions from HIT back
   to ACTIVE?

---

## Area 9: Direction Encoding

### Python (`zelda_enums.py:200-226`)

```python
class Direction(Enum):
    NONE = 0
    E = 1
    W = 2
    S = 4
    N = 8
```

### Assembly

The assembly uses the same bit-flag encoding (1=E, 2=W, 4=S, 8=N) as seen in
`GetOppositeDir` and direction-indexed tables. The `from_ram_value` method only handles
exact matches (1,2,4,8) and returns NONE for anything else. Composite directions
(NE=9, NW=10, SE=5, SW=6) are defined in the enum but never read from RAM.

**Low risk** — the encoding matches. Just verify no object ever stores a composite
direction in RAM.

---

## Area 10: Tile Layout Reading

### Python (`zelda_game.py:237-242`)

```python
map_offset, map_len = zelda_game_data.tables['tile_layout']
tiles = self.ram[map_offset:map_offset+map_len]
tiles = tiles.reshape((32, 22)).T.swapaxes(0, 1)
```

`tile_layout` = address $D30, length $2C0 (704 bytes = 32 × 22).

**Verify**: The assembly stores tiles at $6530 (`PlayAreaTiles`) but the data file says
$D30. These might differ due to NES memory mirroring. Also verify the reshape/transpose
gives the correct (x, y) layout that matches how the assembly indexes tiles.

**Note**: $D30 may be the retro integration's remapped address, not the raw NES address.
The assembly address $6530 is in battery-backed SRAM space ($6000-$7FFF).

---

## Area 11: `ZeldaEnemyKind` ID Values

The Python enum needs to be checked against the assembly's object type assignments.
Some values look suspicious:

```python
Octorok : int = 0x07
OctorokFast : int = 0x7      # DUPLICATE of Octorok!
OctorokBlue : int = 0x8
```

`Octorok` and `OctorokFast` have the same value (0x07). This is either intentional
(fast variant uses same ID) or a bug. Verify against the assembly's enemy type list.

---

## Area 12: Frame Skip Wrapper Cooldowns

### Constants (`frame_skip_wrapper.py`)

```python
ATTACK_COOLDOWN = 15
ITEM_COOLDOWN = 10
MAX_MOVEMENT_FRAMES = 16
```

These control how many frames the agent waits after attacking or using an item.
Verify against the assembly's weapon state machine durations:

- Sword: States 1→2→3→4→5→6 (deactivate). State 2 lasts 8 frames, others last 1.
  Total = 5 + 8 = 13 frames. Python uses 15 — slightly conservative, verify if correct.
- Items: Various durations depending on type. 10 frames may be too few for some items.

### Movement tile snapping

The movement code waits until Link crosses a tile boundary (8-pixel grid), with extra
`WS_ADJUSTMENT_FRAMES = 4` for south/west movement. This asymmetry needs verification —
why do south and west need extra frames but north and east don't?

---

## Area 13: `is_sound_playing` and `SoundKind`

### Python

```python
def is_sound_playing(self, sound: SoundKind) -> bool:
    return bool(self.sound_pulse_1 & sound)
```

`sound_pulse_1` is at $605 which is `Tune0` in the assembly. The `SoundKind` values
are bitmasks:

```python
ArrowDeflected = 0x01
BoomerangStun = 0x02
MagicCast = 0x04
KeyPickup = 0x08
SmallHeartPickup = 0x10
SetBomb = 0x20
HeartWarning = 0x40
```

**Verify**: Does $605 (`Tune0`) contain bitmasks for active sound effects? The assembly
has separate sound systems (Song, Tune0, Tune1, Effect, Sample). Verify that these
`SoundKind` bitmasks correspond to `Tune0` and not to `Effect` ($606) or another
register.

---

## Summary of Review Areas

| # | Area | Risk | Key Files |
|---|---|---|---|
| 1 | Memory address mapping | Medium | `zelda_game_data.txt`, `Variables.inc` |
| 2 | Beam/sword shot state machine | **High** | `link.py`, `Z_07.asm`, `frame_skip_wrapper.py` |
| 3 | Sword screen lock boundaries | **High** | `link.py`, `Z_05.asm` |
| 4 | Enemy health encoding | Medium | `zelda_game.py`, `Z_01.asm` |
| 5 | Enemy is_dying / is_active | Medium | `enemy.py`, `Z_04.asm`, `Z_05.asm` |
| 6 | Object ID classification | Medium | `zelda_game.py`, assembly enemy tables |
| 7 | Link's health system | Medium | `link.py`, `Z_07.asm:MakeSwordShot` |
| 8 | Look-ahead simulation | **High** | `state_change_wrapper.py`, emulator API |
| 9 | Direction encoding | Low | `zelda_enums.py` |
| 10 | Tile layout reading | Low | `zelda_game.py`, assembly tile storage |
| 11 | ZeldaEnemyKind IDs | Medium | `zelda_enums.py`, assembly enemy types |
| 12 | Frame skip cooldowns | Medium | `frame_skip_wrapper.py`, `Z_07.asm` |
| 13 | Sound bitmask register | Low | `zelda_enums.py`, `Z_00.asm` |
