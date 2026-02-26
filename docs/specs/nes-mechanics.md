# NES Game Mechanics Reference

This document records verified NES assembly behavior for The Legend of Zelda, as it relates
to the Python game state model in `triforce/`.  Everything here was verified empirically
against the ROM and/or traced through the disassembly in `zelda-asm/src/`.

## Object Slot Layout

The NES uses flat arrays indexed by slot number.  Each table (`ObjX`, `ObjY`, `ObjState`,
`ObjHP`, etc.) is contiguous — `table[slot]` gives the value for that slot.

| Slot | Purpose | ObjState Address | Python Name |
|------|---------|-----------------|-------------|
| 0 | Link | $AC | `link_status` |
| 1–B | Enemies/Items (up to 11) | $AD–$B7 | via `obj_id` table |
| D (13) | Sword / Rod melee | $B9 | `sword_animation` |
| E (14) | Sword beam / Magic rod shot | $BA | `beam_animation` |
| F (15) | Boomerang / Food | $BB | `bait_or_boomerang_animation` |
| 10 (16) | Bomb / Fire 1 | $BC | `bomb_or_flame_animation` |
| 11 (17) | Bomb / Fire 2 | $BD | `bomb_or_flame_animation2` |
| 12 (18) | Arrow / Rod melee anim | $BE | `arrow_magic_animation` |

The Python "animation" fields in `zelda_game_data.txt` are just `ObjState` for weapon slots.

## Weapon State Machines

### Sword Melee (slot $0D)

```
$00 → $01 (4 frames) → $02 (8 frames) → $03 (1 frame, beam check here)
  → $04 (1 frame) → $05 (1 frame) → $00
Total: ~15 frames
```

`MakeSwordShot` (Z_07.asm:4616) fires at exactly state 3.

### Sword Beam (slot $0E)

```
$00 (inactive) → $10 (flying) → $11 (spreading, 22 frames) → $00
```

The spread phase counts `ObjDir` down from $FE to $E8 (22 frames).  Spread duration is
fixed; flight duration depends on distance to wall/enemy.

### Beam Health Check (Z_07.asm:4632-4648)

The NES checks two conditions to fire beams:
1. `HeartValues & 0x0F` (filled hearts) == `HeartValues >> 4` (containers minus one)
2. `HeartPartial >= $80`

This is NOT a float comparison.  Python uses `_is_health_full_for_beams` to match this exactly.

### Magic Rod Shot (slot $0E, shared with beam)

```
$80 (flying) → $00 (deactivated on hit)
```

- State $81 NEVER occurs.  `HandleShotBlocked` uses ASL to detect magic shots (bit 7 set)
  and branches differently than sword shots.
- Rod melee animation uses slot $12: states $31→$32→$33→$34→$35→$00.
- With `InvBook` ($661) set, fire spawns in bomb/flame slot ($10) at state $22 via
  `WieldCandle`.  Fire lasts ~79 frames at constant state $22.

### Bomb (slots $10/$11)

```
$12 (placed/active) → $13/$14 (exploding) → $00
```

### Arrow (slot $12)

States $0A through $15 are active.  $14 = hit, $15 = end.

### Boomerang (slot $0F)

States $0A through $39 are active (outbound and return).

## Enemy State Machines

### Death Metastate Sequence

`UpdateMetaObject` (Z_01.asm) handles:
```
$10 (1 frame) → $11 (6 frames) → $12 (6 frames) → $13 (6 frames) → $14 (becomes item)
```

States $10–$13 are the death sparkle animation.  Python's `is_dying` range of 16–19
($10–$13) is correct — $14 is never directly observable because the object type changes
to $60 (item) immediately.

### Spawn Metastate

Values 1–4 are the spawn cloud.  Value 0 = fully spawned and active (for most enemies).

### Zora (ObjType $11)

State cycle: 0→1→2→3→4→5→0
- State 0: hidden underwater
- State 1: choosing position
- State 2: surfacing (collision checks active)
- State 3: fully surfaced, fires projectile (collision checks active)
- State 4: sinking (collision checks active)
- State 5: delay before restart

`UpdateBurrower` checks collisions for Zora at states **2, 3, AND 4** (not just 3).

### Leever (ObjType $0C/$0D)

Same `UpdateBurrower` routine but non-Zora path — collision only at state **3** (fully surfaced).

### WallMaster (ObjType $27)

State 0 = hidden in wall.  State **1** = active on screen.

## Object Classification

From `UpdateObject_JumpTable`:
- **Enemy range**: ObjType $01–$48 (excluding $40 = StandingFire)
- **Projectile/hazard range**: ObjType $49+ (Trap=$49 is first non-enemy)
- **Item**: ObjType $60

$3F (GuardFire) is within enemy range but unlikely in enemy slots.
$40 (StandingFire) is explicitly excluded from `_is_id_enemy`.

## Health Encoding

Address $66F (`HeartValues`):
- High nibble = containers_minus_one (max_hearts - 1)
- Low nibble = fully_filled_hearts

Address $670 (`HeartPartial`):
- $00 = empty
- $01–$7F = half heart
- $80–$FF = full heart

Full health (NES standard): `filled = containers - 1`, `partial = $FF`.

## Screen Lock Boundaries

`BorderBounds` (Z_05.asm:2728) defines three sets of bounds (down, up, right, left):

| Set | Down | Up | Right | Left |
|-----|------|----|-------|------|
| OW outer | $D6 | $45 | $E9 | $07 |
| UW outer | $C6 | $55 | $D9 | $17 |
| Inner | $BE | $54 | $D1 | $1F |

`Link_FilterInput` calls `MaskInputInBorder` twice:
1. Inner bounds + mask $80 → keeps A button, blocks movement
2. Outer bounds + mask $00 → blocks everything including A button

Only the **outer** bounds block the sword.  The inner bounds still allow attacks.

## Direction Encoding

E=1, W=2, S=4, N=8.  The NES never stores composite directions — diagonal input resolves
to a single cardinal (vertical axis wins).  `from_ram_value` correctly returns NONE for
non-cardinal values.

## Tile Layout

`PlayAreaTiles` is 704 bytes (32 × 22) at address $D30 (retro-remapped from $6530).
Stored column-major: `reshape(32, 22)` gives `tiles[x, y]` indexing directly.

Door tile positions: north=`tiles[0xf, 2]`, south=`tiles[0xf, 0x13]`,
west=`tiles[2, 0xa]`, east=`tiles[0x1d, 0xa]`.

## Sound Register

`Tune0` ($605) contains incidental music bitmasks:
- $01 = Arrow Deflected
- $02 = Boomerang Stun
- $04 = Magic Cast
- $08 = Key Pickup
- $10 = Small Heart Pickup
- $20 = Set Bomb
- $40 = Heart Warning
- $80 = unknown/unused

$604 is the trigger register (write to request), $605 is the type register (what's playing).

## Enemy HP Encoding

`ObjHP` stores HP as multiples of $10 (high nibble = HP count, low nibble always 0).
`ObjectTypeToHpPairs` packs 2 enemies per byte; `ExtractHitPointValue` extracts the
correct nibble.

Damage values: Wood sword=$10, White=$20, Magic=$40, Bomb=$40, Fire=$10.
Python's `>> 4` extraction is correct.

Keese ($1B-$1D) and Gel ($14-$15) have 0 HP from the table — they die in one hit.
Death is tracked via ObjMetastate, not ObjHP reaching 0.

## Key RAM Addresses

| Address | Assembly | Python | Notes |
|---------|----------|--------|-------|
| $010 | CurLevel | `level` | 0=overworld, 1-9=dungeon |
| $012 | GameMode | `mode` | |
| $070 | ObjX[0] | `link_x` | Link is slot 0 |
| $084 | ObjY[0] | `link_y` | |
| $098 | ObjDir[0] | `link_direction` | |
| $0AC | ObjState[0] | `link_status` | |
| $0EB | RoomId | `location` | |
| $34F | ObjType | `obj_id` table | Also `room_kills` at [0] |
| $3A8 | ObjPosFrac / Item_ObjItemLifetime | `item_timer` table | Union — context-dependent |
| $405 | ObjMetastate | `obj_spawn_state` table | |
| $485 | ObjHP | `obj_health` table | |
| $605 | Tune0 | `sound_pulse_1` | |
| $627 | WorldKillCount | `kill_streak` | |
| $661 | InvBook | `book` | Enables rod fire |
| $66F | HeartValues | `hearts_and_containers` | |
| $670 | HeartPartial | `partial_hearts` | |
| $671 | InvTriforce | `triforce` | |
| $672 | LastBossDefeated | `triforce_of_power` | |
| $D30 | PlayAreaTiles (remapped) | `tile_layout` table | 704 bytes |
