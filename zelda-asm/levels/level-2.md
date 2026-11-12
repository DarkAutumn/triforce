# Level 2: the magic boomerang and Dodongo

Level 2 lies under overworld room `$3C`, in the forest in the east; its entrance is open from the start and needs
no item. From the start room `$7D` the required path is one column of rooms running north (`$7E` up to the boss
room `$0E`), with a second column to the east behind key doors and bombable walls. Link collects the magic
boomerang (`$4F`), a heart container from the boss and the second Triforce piece (`$0D`); four keys lie in
the level and the required route opens no key door, so Link leaves with more keys than he brought. The boss is
a single Dodongo, which only bombs can beat: the level hands Link four bombs in room `$1E`, one room before
the boss, so the one thing he must bring is a sword. There are no dark rooms, no water, no push blocks, no
stairs and no Wallmasters.

Contents (generated; `L` = line in this file):

<!-- toc -->
- L28 At a glance
- L71 What Link must bring
- L92 What Link gets here
- L142 Map
- L171 Rooms
- L198 Floor plans
- L340 Monsters and boss
- L378 Route
- L415 Optional detours
- L440 Hazards and rules that matter here
- L458 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 2 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-2 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | not in Q1, the secret works in Q2 only: $3C | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178), [`Flute_SecretRoomIdsOW`](../src/bank7/flute.asm#L36) |
| Start room | $7D | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Triforce room | $0D (room item: Triforce piece ($1B)); the field places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $0E (1 x Dodongo ($32)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | none | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 18: 18 on the submenu map | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (room and side that has it) | 6 ($2E E, $2F W, $5E E, $5F W, $6E E, $6F W) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (room and side that has it) | 10 ($1E E, $1F W, $2F S, $3F N, $3F S, $4F N, $4F S, $5F N, $5F S, $6F N) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (room and side that has it) | 6 ($0E W, $1E N, $1E S, $2E N, $4E S, $6D W) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (room and side that has it) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $0D E open / $0E shutter, $0E S open / $1E shutter, $4E S shutter / $5E open, $6C E open / $6D shutter | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $7D S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 0 (none) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 0 (none) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 4 ($3E, $4E, $6C, $7E) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | bomb ($00): $1E, $3F; 5 rupees ($0F): $2F; compass ($16): $6F; map ($17): $5F; heart container ($1A): $0E; Triforce piece ($1B): $0D; magic boomerang ($1E): $4F | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md) (rows with `lvl` = `2` or `c2`).
<!-- /generated:facts -->

Positions below are Link's `ObjX`,`ObjY` (left and top edge of his 16-pixel sprite). On a floor plan, column
`c` (0-11) is X = `$20` + `$10` * c and row `r` (0-6) is Y = `$5D` + `$10` * r; the doors are at X = `$78`
(north, south) and Y = `$8D` (west, east).

- **Entrance.** Overworld room `$3C` has cave index 2 (level 2) and no tile object: nothing has to be bombed, burned, pushed or played, the opening is part of the room's layout (`$0A`, the dungeon face in the middle of row 4). See the row `$3C` in [data/levelblock-ow.md](../data/levelblock-ow.md) and [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178).
- **Reaching it.** The border of `$3C` is closed except for one square at the bottom (column 7, X `$70`-`$7F`), so the room is entered from room `$4C` below it; the opening (tile `$24`) is the lower half of the square at column 7, row 4, straight up that same column, and Link goes in when he stands on it at X `$70`. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
- **Armos squares outside.** Eight squares of armos tiles (`$C0`-`$C3`) stand left and right of the entrance (columns 3, 5, 9, `$B` in rows 4 and 6); pushing into one wakes an Armos, so walk up the middle column. The room's own monster is one fast blue Octorok. [`Spawn_CheckTouchedArmosOrGravestone`](../src/bank1/armos-ghini-spawn.asm#L54)
- **Where Link stands on entry.** Room `$7D`, X `$78`, Y `$DD` ([`LevelInfo_StartY`](../src/Variables.inc#L717)), facing up; he is first controllable in the south doorway at (`$78`,`$CD`) (trace). [`Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink`](../src/bank5/mode3-mode6-init.asm#L129)
- **Walking out.** The south door of `$7D` has no room behind it: going through it loads the overworld and puts Link back at the entrance of `$3C`. [`Room_CalcNextRoomForDoor`](../src/bank5/next-room-calc.asm#L76), [`Mode_LeaveLevelToOW`](../src/bank5/next-room-calc.asm#L107)
- **What ends the level.** Taking the Triforce piece in `$0D` starts the end-of-level sequence: hearts refill to full, the curtain closes, and about 600 frames later (more when hearts have to refill) Link stands in overworld room `$3C` at (`$70`,`$7D`), the room's exit spot, facing down (trace). [`Mode12EndLevel_Update_Sub4LeaveLevel`](../src/bank5/mode12-lay-out-room.asm#L156), [`Room_EnterCommon`](../src/bank5/enter-room.asm#L143)
- **What is kept when Link leaves and comes back.** The saved room flags: key doors unlocked and walls bombed (open from both sides), room items taken, rooms visited, and the "all dead" mark of a cleared room. See "Room flags and saved state" in [mechanics.md](../mechanics.md) and [`Door_Update`](../src/bank5/update-doors.asm#L60).
- **What resets.** Shutters are never saved: every shutter is closed again on each visit of its room. Loading the level zeroes the per-room kill counts and the room history, so every room that was only partly cleared is full again, and a cleared room of Ropes, Goriyas, Gels or Keese refills too; only the Dodongo (`$0E`) and the Moldorm (`$3E`) stay dead for good. [`Mode2Load_Init`](../src/bank7/init-mode-dispatch.asm#L147), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- **Inside one visit.** A cleared room stays empty while it is among the last six different rooms entered; after that it refills. A room item hidden behind "all dead" (the keys of `$7E`, `$4E`, `$3E`, the boomerang, the bombs of `$1E`) that was not taken is hidden again when its room refills. [`RoomHistory_SaveKillCountUW`](../src/bank5/room-history-uw.asm#L110), [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- **Recorder.** Once Link holds this level's Triforce piece the recorder's whirlwind can set him down in room `$3C`. See "Recorder (flute, whistle)" in [items.md](../items.md) and [`Whirlwind_PrevRoomIds`](../src/bank1/whirlwind.asm#L81).

# What Link must bring

Short answer: a sword. Everything else the route needs is found inside the level.

| Thing | Needed? | Where and why |
| --- | --- | --- |
| Sword | yes, any grade | Shutters in `$2E` (8 Ropes) and `$1E` (5 red Goriyas) open only when every monster is dead; the cheapest Dodongo kill is one sword stab |
| Keys | 0 | The route passes no key door. 4 keys lie in the level, 3 key doors exist, all optional |
| Bombs | 0 | `$1E` (on the route, cleared anyway for its shutter) gives 4; the Dodongo needs 1 or 2 |
| Candle | no | No dark room |
| Ladder, raft | no | No water square in any layout; no stairs, no cellar |
| Recorder, food, bow and arrows | no | No Digdogger, no hungry Goriya, nothing that needs an arrow |
| Hearts | no minimum | Nothing tests hearts; the starting 3 containers leave little room in `$1E`, `$4F` and `$0E` (1 heart per Goriya boomerang or Dodongo touch) |

- **Sword.** Ropes (HP `$10`) and red Goriyas (HP `$30`) die to any sword; without a sword only bomb blasts could kill them, and Link enters with none. The wooden sword (`InvSword` = 1, damage `$10`) is enough for every monster of the level. [`Sword_DamagePoints`](../src/bank1/sword-bomb-arrow-collision.asm#L182), [`InvSword`](../src/Variables.inc#L556)
- **Keys.** Key doors: `$2E` E, `$5E` E, `$6E` E, each the same doorway as the west door of `$2F`, `$5F`, `$6F`. None is on the route. Keys lying in the level: `$7E`, `$4E`, `$3E` (all three on the route) and `$6C`. The key count is one counter for all levels, so what Link picks up here he carries on. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`InvKeys`](../src/Variables.inc#L584)
- **Bombs.** No wall on the route has to be bombed. The Dodongo cannot be hurt by anything but bombs; the 4 bombs of `$1E` appear when its five Goriyas are dead, the same condition that opens its north shutter. Arriving with bombs only adds spare tries. [`Dodongo_CheckBombHit`](../src/bank4/dodongo.asm#L400), [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337)
- **Candle.** No room of the level has the dark bit (array E bit 7), so [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) is never true here and nothing is drawn black.
- **Ladder and raft.** None of the twelve layouts used has a water square (`W` in the floor plans) and the level lists no cellar. [`Ladder_CheckAndDraw`](../src/bank5/ladder.asm#L44), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
- **Hearts.** Full health matters only for the sword beam (thrown at full hearts, see "Sword (Wooden, White, Magic) and sword beam" in [items.md](../items.md)): it lets Link kill Ropes from a distance. [`SwordShot_Make`](../src/bank7/sword-and-rod.asm#L327)

# What Link gets here

## Magic boomerang (room `$4F`)

- **Room.** `$4F`, one room east of `$4E` through an open door. The item (id `$1E`) lies hidden at (`$80`,`$90`) and appears when the three blue Goriyas are dead (secret trigger 7). [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- **Taking it.** Walk along the door row (Y = `$8D`) to X = `$78`: it was taken there on the way in from the west door (trace). A room item is tested against Link only, never against a thrown boomerang or the sword. [`Item_TryTakeRoomItem`](../src/bank1/take-item.asm#L173)
- **What it changes: use.** With the boomerang selected, B throws it from slot `$0F` in the held direction (diagonals included) or the facing direction, at 3 pixels per frame, at no cost; one at a time. [`Boomerang_Wield`](../src/bank5/wield-items.asm#L124)
- **What it changes: range.** `InvMagicBoomerang` = 1 sets the range limit to `$FF`, which is never reached: the magic boomerang flies until a room bound or a monster stops it, where the wooden one turns back after `$31` pixels. Thrown left from X `$7D` it reached X `$13` before returning (trace). [`Boomerang_RangeLimits`](../src/bank5/wield-items.asm#L85), [`InvMagicBoomerang`](../src/Variables.inc#L603)
- **What it changes: combat.** A hit deals damage 0 and stuns for about 150 frames: a stunned monster does not move or shoot and its touch does not hurt Link. Monsters with HP 0 (Gel, Keese) die from the hit. No knockback. [`Collision_CheckWeaponPointVsMonster`](../src/bank1/weapon-damage-death.asm#L82), [`Boomerang_CheckMonsterCollision`](../src/bank1/weapon-damage-death.asm#L48)
- **What it changes: pickups.** While it is out it collects dropped items (hearts, rupees, bombs, fairies) it touches, so drops can be fetched from across the room. [`Drop_UpdateItem`](../src/bank4/drop-items.asm#L297)
- **What it does not do.** It opens nothing: no door, wall, block or secret reads it. It does nothing to the Dodongo (parry sound, no stun), to Traps, or to statue fireballs. [`Dodongo_CheckCollisions`](../src/bank4/dodongo.asm#L296)
- **First use here.** From `$4E` north: stun the Moldorm segments of `$3E` while crossing (their touch is then harmless), freeze the Ropes of `$2E` and the red Goriyas of `$1E` before stabbing them, kill the Gels and Keese of the optional rooms from a distance. **Later:** the same against every monster not immune to it; see "Boomerang (wooden, magic)" in [items.md](../items.md).

## Bombs (room `$1E`; also `$3F`)

- **Room `$1E`.** One bomb item (id `$00`) at (`$D0`,`$60`), the top right floor square, hidden until the five red Goriyas are dead. From the south door go left to X `$60`, up to Y `$5D`, right to X `$D0`: taken on arrival (trace). [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- **Amount.** Every bomb item adds 4 to `InvBombs`, capped at `InvMaxBombs` (8 unless enlarged). [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337), [`InvMaxBombs`](../src/Variables.inc#L607)
- **What it changes.** With bombs selected, B puts a bomb `$10` pixels ahead of Link: fuse `$30` frames, blast `$18` frames (damage `$40` within `$18` pixels), cloud `$0C` frames. Bombs are what beats the Dodongo in the next room, and they open bombable walls (five in this level, all optional). See "Bombs" in [items.md](../items.md). [`Bomb_Wield`](../src/bank1/bomb-candle-wield.asm#L63), [`Bomb_StateTimes`](../src/bank7/bomb.asm#L55)
- **First needed.** Room `$0E`, the next room: 1 bomb (blast or cloud stuns the Dodongo, then one sword stab) or 2 bombs (both swallowed). **Later:** bombable walls and rock walls in the overworld and other levels. [`Dodongo_CheckBombHit`](../src/bank4/dodongo.asm#L400)
- **Room `$3F`** (optional): a second bomb item lies in the open at (`$80`,`$90`), guarded by four Traps and four Keese. Red Goriyas (`$5E`, `$1E`) also drop bombs. [`Drop_Group2Types`](../src/bank4/drop-items.asm#L111)

## Heart container (room `$0E`)

- **Room.** The boss room; the item (id `$1A`) appears at (`$80`,`$90`) when the Dodongo is dead, together with the opening of the west shutter. Taken at (`$78`,`$8D`), on the way to the west door (trace). [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- **What it changes.** `HeartValues` + `$11`: one more container and one more full heart (3 -> 4 containers on a first visit); the other hearts are not refilled by it. See "Hearts, Heart Containers and Fairies" in [items.md](../items.md). [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337), [`HeartValues`](../src/Variables.inc#L593)
- **Where it counts.** The white sword's cave asks for 5 containers, the magic sword's for 12; each container is also one more heart of damage Link survives. See "Old men and shopkeepers (cave and dungeon persons)" in [enemies.md](../enemies.md).
- **Do not skip it.** The Triforce piece ends the level; coming back for the container means walking the whole level again (the Dodongo stays dead and the item still appears). [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)

## Triforce piece (room `$0D`)

- **Room.** `$0D`, west of the boss room, no monsters. The piece lies at (`$78`,`$90`) inside a ring of blocks whose only gap is at the bottom (floor columns 5-6 of row 5). From the east door: left to X `$D0`, down to Y `$BD`, left to X `$78`, up; it is taken at Y `$95` (trace). [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- **What it changes.** Sets bit 1 of `InvTriforce`, refills all hearts, and returns Link to the overworld entrance. [`Item_TakeClass0PerLevel`](../src/bank1/take-item.asm#L421), [`InvTriforce`](../src/Variables.inc#L597)
- **Where it counts.** All eight pieces open the way past the first old man of level 9; a held piece also makes this level a recorder destination. See "Triforce pieces and the Triforce of Power" in [items.md](../items.md).

## Keys (rooms `$7E`, `$4E`, `$3E`, `$6C`)

- **`$7E`, `$4E`, `$3E`.** Each key (id `$19`) is hidden at (`$80`,`$90`) until the room's monsters are dead: 5 Ropes, 5 Ropes, the Moldorm. All three rooms are on the route; each key is taken by walking the door row to X `$78` (trace for `$7E` and `$4E`). [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- **`$6C`.** Lies in the open at (`$80`,`$90`) among 6 Ropes; the room is a dead end west of `$6D`, behind `$6D`'s west shutter. [`Item_TryTakeRoomItem`](../src/bank1/take-item.asm#L173)
- **What a key changes.** Pushing into a key door with `InvKeys` > 0 spends one key and opens the door for good, from both sides. Nothing in this level needs one; the keys are for the optional east rooms or for later levels. See "Keys and Magic Key" in [items.md](../items.md). [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)

## Map (room `$5F`) and compass (room `$6F`)

- **Map.** `$5F`, in the open at (`$80`,`$90`), 5 Gels. It only draws the level's shape on the status bar. [`Mode3Unfurl_Init_Sub6_CueStatusBarMap`](../src/bank5/mode3-mode6-init.asm#L102)
- **Compass.** `$6F`, in the open at (`$D0`,`$60`), 6 Gels. It only draws the blinking marker of room `$0D` on the status bar map. [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51)
- **Neither changes any rule** of movement, combat or access; both are off the route. See "Map and Compass" in [items.md](../items.md).

## 5 rupees (room `$2F`)

- **Room.** `$2F`, hidden at (`$80`,`$90`) until its 5 Gels are dead; worth 5 rupees, off the route. [`Item_Take5Rupees`](../src/bank1/take-item.asm#L328)

# Map

<!-- generated:map -->
```
         [0DT ]-/S[0EBi]
                   |/S
                  [1Ei ] X [1F  ]
                    S        |
                  [2E  ] K [2Fi ]
                    |        X
                  [3Ek ]---[3Fi ]
                    |        X
                  [4Ek ]---[4Fi ]
                   S/|       X
                  [5E  ] K [5Fi ]
                    |        X
[6Ck ]-/S[6D  ]---[6E  ] K [6Fi ]
           |        |
         [7DS ]---[7Ek ]

[73S] = room $73 and its marks: S start room, B boss room, T Triforce room,
        k a key lies here, i another room item lies here, h not shown on the submenu map.
Between two rooms, the door: --- or | open, K key door, S shutter, X bombable wall,
        F false wall (walk-through), nothing = wall. A/B = the two sides differ: the left
        or upper room's side first, . = wall on that side.
Doors on sides with no room of the level: $7D S open.
```
<!-- /generated:map -->

# Rooms

Generated from the level data by `tools/romdata.py`; do not edit between the markers.

<!-- generated:rooms -->
| room | N | S | W | E | monsters | item | secret | block | dark | layout | floor | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $0D Triforce | wall | wall | wall | open | none | Triforce piece ($1B) at $80,$90 | - | - | - | $29 | blocks, statues | - |
| $0E boss | wall | open | shutter | wall | 1 x Dodongo ($32) | heart container ($1A) at $80,$90 | all dead: shutters open, room item appears | - | - | $25 | sand | - |
| $1E | shutter | shutter | wall | bomb | 5 x Goriya ($06) | bomb ($00) at $D0,$60 | all dead: shutters open, room item appears | - | - | $0D | blocks | boss sound $20 |
| $1F | wall | open | bomb | wall | 1 x dungeon person ($4B) | - | - | - | - | $26 | blank | person $4B says: DODONGO DISLIKES SMOKE. |
| $2E | shutter | open | wall | key | 8 x Rope ($28) | - | all dead: shutters open | - | - | $1D | blocks | - |
| $2F | open | bomb | key | wall | 5 x Gel ($15) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $3E | open | open | wall | open | 1 x Moldorm ($41) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $25 | sand | - |
| $3F | bomb | bomb | open | wall | 1 x Trap ($4A), 3 x (empty) ($00), 4 x Keese blue ($1B) | bomb ($00) at $80,$90 | - | - | - | $00 | plain | - |
| $4E | open | shutter | wall | open | 5 x Rope ($28) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $02 | blocks | - |
| $4F | bomb | bomb | open | wall | 3 x Goriya ($05) | magic boomerang ($1E) at $80,$90 | all dead: shutters open, room item appears | - | - | $24 | statues | - |
| $5E | open | open | wall | key | 5 x Goriya ($06) | - | ringleader: first monster dead kills all | - | - | $1F | blocks | - |
| $5F | bomb | bomb | key | wall | 5 x Gel ($15) | map ($17) at $80,$90 | - | - | - | $03 | blocks | - |
| $6C | wall | wall | wall | open | 6 x Rope ($28) | key ($19) at $80,$90 | - | - | - | $00 | plain | - |
| $6D | wall | open | shutter | open | 5 x Rope ($28) | - | all dead: shutters open | - | - | $03 | blocks | - |
| $6E | open | open | open | key | 3 x Rope ($28) | - | - | - | - | $0D | blocks | - |
| $6F | bomb | wall | key | wall | 6 x Gel ($15) | compass ($16) at $D0,$60 | - | - | - | $1E | blocks | - |
| $7D start | open | open | wall | open | none | - | - | - | - | $21 | statues, sand | - |
| $7E | open | wall | open | wall | 5 x Rope ($28) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
<!-- /generated:rooms -->

# Floor plans

<!-- generated:floors -->
One character per floor square, 12 columns x 7 rows inside the walls (left column = west, top row =
north): `.` floor (tile `$74`), `B` block (tile `$B0`; a push block object is placed on one in a room
with a push block), `W` water (tile `$F4`), `S` statue (tiles `$94` and `$B4`), `>` stairs (tile `$70`),
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L63), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L73), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
Stairs that a secret makes appear are not part of the layout.

```
layout $00  rooms: $2F,$3F,$6C,$7E
............
............
............
............
............
............
............
```

```
layout $02  rooms: $4E
............
............
..B......B..
............
..B......B..
............
............
```

```
layout $03  rooms: $5F,$6D
............
..B......B..
............
............
............
..B......B..
............
```

```
layout $0D  rooms: $1E,$6E
............
.B.B.BB.B.B.
............
.B.B.BB.B.B.
............
.B.B.BB.B.B.
............
```

```
layout $1D  rooms: $2E
............
............
..BB....BB..
..BB....BB..
..BB....BB..
............
............
```

```
layout $1E  rooms: $6F
............
............
.....BB.....
.....BB.....
.....BB.....
............
............
```

```
layout $1F  rooms: $5E
............
.BB......BB.
............
.....BB.....
............
.BB......BB.
............
```

```
layout $21  rooms: $7D
............
.S..S..S..S.
............
.S..S..S..S.
.....::.....
.S.:S::S:.S.
...::::::...
```

```
layout $24  rooms: $4F
S..........S
............
............
............
............
............
S..........S
```

```
layout $25  rooms: $0E,$3E
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
```

```
layout $26  rooms: $1F
____________
____________
____________
____________
____________
____________
____________
```

```
layout $29  rooms: $0D
............
.BBBBBBBBBB.
.B..S..S..B.
.B.S....S.B.
.B........B.
.BBBB..BBBB.
............
```
<!-- /generated:floors -->

# Monsters and boss

HP is the raw `ObjHP` value; a wooden sword hit is `$10`, a bomb blast `$40`. Damage to Link is without a ring.

| Monster | Rooms (count) | HP | Dies to | Boomerang | Hurts Link | Section in [enemies.md](../enemies.md) |
| --- | --- | --- | --- | --- | --- | --- |
| Rope `$28` | `$7E` (5), `$6E` (3), `$4E` (5), `$2E` (8); off route `$6D` (5), `$6C` (6) | `$10` | 1 hit of anything | stuns | touch 1/2 heart | "Rope" |
| Red Goriya `$06` | `$5E` (5, ringleader), `$1E` (5) | `$30` | 3 wooden sword hits, 1 bomb | stuns | touch 1/2 heart; its boomerang 1 heart | "Goriya (blue, red)" |
| Blue Goriya `$05` | `$4F` (3) | `$50` | 5 wooden sword hits, 2 bombs | stuns | touch 1 heart; its boomerang 1 heart | "Goriya (blue, red)" |
| Moldorm `$41` | `$3E` (2 worms of 5 segments) | `$20` per segment | 2 wooden sword hits per segment, 10 segment kills | touch made harmless, keeps moving | touch 1/2 heart | "Moldorm" |
| Gel `$15` | off route `$2F` (5), `$5F` (5), `$6F` (6) | `$00` | any hit | kills | touch 1/2 heart | "Gel" |
| Blue Keese `$1B` | off route `$3F` (4) | `$00` | any hit | kills | touch 1/2 heart | "Keese (blue, red, black)" |
| Trap `$49` | off route `$3F` (4, corners) | none | nothing | nothing | touch 1 heart | "Trap" |
| Statue fireballs `$55` | `$4F` (4 statues) | none | nothing | nothing | 1/2 heart | "Statue fireballs" |
| Dodongo `$32` | `$0E` (1), the boss | `$F0`, not used | 2 swallowed bombs, or 1 bomb stun + 1 sword stab | nothing | touch 1 heart | "Dodongo" |
| Old man `$4B` | off route `$1F` | none | cannot die | nothing | touch 0 damage | "Old men and shopkeepers" |

- **Ropes** walk at 0.5 pixel per frame and rush at 1.5 when Link is within 8 pixels of their row or column. Meet them head on with the sword out, or stay 8 or more pixels off both their row and their column. Every Rope dies in one hit in Q1. [`Rope_Update`](../src/bank4/rope.asm#L51), [`Rope_Init`](../src/bank4/rope.asm#L32)
- **`$2E` (8 Ropes, shutter).** The largest fight of the route and it cannot be skipped: the north shutter opens only when all eight are dead. The south door behind Link stays open, so he can step out; the room keeps its kills while it is among the last six rooms. The four block pairs (floor columns 2-3 and 8-9, rows 2-4) stop rushes. [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- **`$1E` (5 red Goriyas, locked in).** Link comes in through a shutter that closes behind him; north is a shutter too. He cannot leave until all five are dead (or he bombs the east wall, which needs bombs he does not have yet). Three wooden sword hits each; a thrown boomerang costs a whole heart, and any shield blocks it when Link faces it without swinging. [`Walker_SetInputDirAndTryThrowBoomerang`](../src/bank4/goriya.asm#L156), [`Link_CheckMonsterTouch_NoReset`](../src/bank1/link-collision-harm.asm#L73)
- **`$5E` (ringleader).** Killing the Goriya in object slot 1 kills the other four at once (trace); no door depends on it, and Link can also just walk through. Deaths started this way do not raise the kill counters that force help drops. [`Secret_Trigger2Ringleader`](../src/bank5/underworld-secrets.asm#L176)
- **`$4F` (boomerang room).** Three blue Goriyas (5 wooden sword hits each, 1 heart per touch or boomerang) while the four corner statues shoot aimed fireballs the whole time; a statue does not shoot while Link is within `$18` pixels of it on both axes. Do not stay: with the Goriyas already gone, a Link who lingered in the room lost all 3 hearts to fireballs alone within about 800 frames (trace). Take the item and leave. [`Statue_Update`](../src/bank4/statues.asm#L86), [`Goriya_Update`](../src/bank4/goriya.asm#L45)
- **`$3E` (Moldorm).** Two worms fly through the room with no regard for tiles. Walking straight north took 138 frames and one half-heart touch (trace). Killing it (10 segment kills, 20 wooden sword hits) is optional and gives the key. [`Moldorm_Update`](../src/bank4/moldorm.asm#L59), [`Moldorm_Init`](../src/bank4/moldorm-init.asm#L38)
- **`$3E`, the short way to its key.** The Moldorm is a non-recurring type: once one worm's head has died (5 kills on that worm) and Link leaves, the room is saved as cleared. On the next entry it is empty and the key lies in the open (trace, with the head's death forced). [`RoomHistory_SaveKillCountUW`](../src/bank5/room-history-uw.asm#L110)
- **Red Goriyas drop bombs**: they are in drop group 2, whose table holds a bomb in three of ten kill-cycle columns. See "Drop tables" in [enemies.md](../enemies.md). [`Drop_Group2Types`](../src/bank4/drop-items.asm#L111), [`Drop_ItemTable`](../src/bank4/drop-items.asm#L132)

## The boss: Dodongo, room `$0E`

- **Room.** Entered from the south (`$1E`); the floor is all sand with no obstacle. One Dodongo walks at 0.5 pixel per frame, turning now and then; it shoots nothing and its touch costs 1 heart. [`Dodongo_Init`](../src/bank4/aquamentus-digdogger-dodongo-inits.asm#L118), [`Damage_ObjTypeDamageToLink`](../src/bank1/monster-collisions.asm#L200)
- **Nothing but a bomb starts the kill.** Sword, beam, boomerang and arrows only play the parry sound; a bomb blast does no ordinary damage either. [`Dodongo_CheckCollisions`](../src/bank4/dodongo.asm#L296)
- **Way 1, smoke and sword: 1 bomb.** A bomb that is not swallowed but whose blast or cloud (frames `$30`-`$54` after placing) is within 12 pixels of the Dodongo's centre stuns it for about 300 frames. One stab of any sword on the stunned Dodongo kills it, and that death drops a bomb, so the bomb comes back. The stunned Dodongo does not hurt on touch. [`Dodongo_CheckBombHit`](../src/bank4/dodongo.asm#L400), [`Dodongo_State2Stunned`](../src/bank4/dodongo.asm#L171)
- **Way 2, feeding: 2 bombs.** A bomb whose fuse is burning is swallowed when it lies in the front half of the walking Dodongo, on its own row or column; put it about 16 pixels in front of its nose. After the first it stands bloated for 97 frames and walks on; the second kills it 161 frames after it is swallowed. No sword needed. [`Dodongo_TryEatBomb`](../src/bank4/dodongo.asm#L469), [`Dodongo_State1Bloated_Wait`](../src/bank4/dodongo.asm#L211)
- **One bomb at a time.** Only the bomb in object slot `$10` counts, for eating and for the stun. A bomb placed while the previous one is still blasting or fading goes to slot `$11` and is wasted: wait until the first bomb is gone (`$54` frames) before placing the next. [`Bomb_Wield`](../src/bank1/bomb-candle-wield.asm#L63)
- **The least the room needs:** 1 bomb with a sword, 2 without. With the 4 bombs of `$1E` Link has four tries at the stun or two feedings. [`Dodongo_CheckBombHit`](../src/bank4/dodongo.asm#L400)
- **If Link runs out of bombs** the Dodongo cannot be killed and the west shutter stays shut. The south door is an open doorway: go back for bombs. Sources: the item of `$3F` (once), red Goriya drops in `$5E` and `$1E` (`$1E` refills only after it has left the six-room history or the level was re-entered), and the overworld. [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50), [`Drop_Group2Types`](../src/bank4/drop-items.asm#L111)
- **When it dies** the west shutter opens and the heart container appears at (`$80`,`$90`) (trace). The room is saved as cleared at once and never makes a Dodongo again; the level's boss roar (heard in `$1E`) stops. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42)
- **Drop.** Sword kill: a bomb, unless Link is hurt before the death sparkle ends. Bomb kill: the ordinary roll of drop group 3 (heart, fairy or rupee, 41%). See "Dodongo" in [enemies.md](../enemies.md).

# Route

How it was established: the room graph was searched from the level data (doors, keys, secrets), and the whole
route below was then walked in the running game from the start room to the overworld, room by room, with these
positions (trace). In that walk the monsters were removed without fighting; the fights themselves are described
from the code and from the sections of `enemies.md`. Twelve room entries, no key door, no bomb wall.

`$7D` -> `$7E` -> `$6E` -> `$5E` -> `$4E` -> `$4F` -> `$4E` -> `$3E` -> `$2E` -> `$1E` -> `$0E` -> `$0D`

"Keys" is the count after the room if Link entered with none and takes every key on the path.

| Step | Room | Entered from | What to do | Leave by | Keys | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `$7D` start | overworld; Link at (`$78`,`$CD`) | No monsters. Statues block the east door's row: go up to Y `$9D`, right to X `$D0`, up to Y `$8D`, then right | E, open | 0 | The north door (straight up X `$78`) is the alternative start |
| 2 | `$7E` | W, at (`$10`,`$8D`) | Kill the 5 Ropes: the key appears at (`$80`,`$90`); take it at X `$78` on the door row. Then up X `$78` | N, open | 1 | Empty floor. The key is optional |
| 3 | `$6E` | S, at (`$78`,`$CD`) | 3 Ropes, no need to kill. Blocks close the middle column: up to Y `$BD`, left to X `$60`, up to Y `$5D`, right to X `$78`, up | N, open | 1 | Key door on the E side (to the compass): leave it |
| 4 | `$5E` | S | 5 red Goriyas; killing the first one made kills all, or walk past. Blocks in the middle: up to Y `$9D`, left to X `$60`, up to Y `$7D`, right to X `$78`, up | N, open | 1 | Key door on the E side (to the map): leave it |
| 5 | `$4E` | S, through its shutter, at (`$78`,`$BD`) | The shutter closes behind Link (N and E stay open). Kill the 5 Ropes: key at (`$80`,`$90`), taken at (`$78`,`$8D`). Then right along Y `$8D` | E, open | 2 | The key is optional |
| 6 | `$4F` | W, at (`$10`,`$8D`) | Kill the 3 blue Goriyas under statue fire: the magic boomerang appears; take it at (`$78`,`$8D`) and go back at once | W, open | 2 | Bombable walls N and S (optional) |
| 7 | `$4E` | E, at (`$E0`,`$8D`) | Empty if it was cleared in step 5. Left to X `$78`, up | N, open | 2 | |
| 8 | `$3E` | S | Moldorm. Straight up X `$78` (sand, no obstacle). Optional: kill both worms for the key | N, open | 3 | 2 keys if the Moldorm is left alone |
| 9 | `$2E` | S | Kill all 8 Ropes: the north shutter opens. Straight up X `$78` | N, shutter | 3 | Key door on the E side (to the rupees and the old man): leave it |
| 10 | `$1E` | S, through its shutter, at (`$78`,`$BD`) | Locked in. Kill the 5 red Goriyas: both shutters open and the bombs appear. Left to X `$60`, up to Y `$5D`, right to X `$D0` (bombs taken), back left to X `$78`, up | N, shutter | 3 | Bombs 0 -> 4. Blocks close the middle column and the door row |
| 11 | `$0E` boss | S, at (`$78`,`$CD`) | Kill the Dodongo (1 bomb + sword, or 2 bombs). Take the heart container at (`$78`,`$8D`), then left along Y `$8D` | W, shutter | 3 | Containers 3 -> 4 |
| 12 | `$0D` Triforce | E, at (`$E0`,`$8D`) | Left to X `$D0`, down to Y `$BD`, left to X `$78`, up through the gap to Y `$95`: the piece is taken | (end of level) | 3 | Link comes out in overworld `$3C` with full hearts |

- **Key balance.** Used: 0. Found on the path: 3 (`$7E`, `$4E`, `$3E`), each behind an optional fight. Link leaves with the keys he brought plus 0 to 3; a fourth key lies in `$6C`. [`InvKeys`](../src/Variables.inc#L584)
- **Bomb balance.** Brought: 0. Gained: 4 in `$1E`, plus what red Goriyas drop. Used: 1 or 2 on the Dodongo; the sword kill returns one. [`InvBombs`](../src/Variables.inc#L558)
- **Alternative start, same length.** `$7D` north -> `$6D` -> east -> `$6E` -> north replaces `$7D` -> `$7E` -> `$6E`: it meets 5 Ropes in `$6D` instead of 5 in `$7E` and passes no key. In `$6E` the west door's row is closed by a block one square in: step to X `$20`, up to Y `$7D`, right to X `$60`, up to Y `$5D`, right to X `$78`, up (trace). [`Tile_GetAheadOfObj`](../src/bank7/tile-collision.asm#L65)
- **Alternative out of `$4F`.** Bomb its north wall (stand at (`$78`,`$6D`) facing up; the wall opened 79 frames after the bomb was placed, trace), go north into `$3F` (bombs, Traps, Keese) and west into `$3E`: the same number of room entries, one bomb spent and four gained, but Link has no bomb yet on a first visit unless he brought one. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136)
- **Order: `$4F` before the north.** The boomerang makes `$3E`, `$2E` and `$1E` easier; taking it later costs the walk back. [`InvMagicBoomerang`](../src/Variables.inc#L603)
- **Order: bombs before the boss, container before the Triforce.** The bombs of `$1E` can still be fetched from `$0E` (its south door is open and the cleared `$1E` opens its shutters again as soon as Link is inside, trace of the same case in `$4E`), but the Triforce piece ends the level. [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68)
- **Walking paths.** Link cannot turn sideways inside a doorway: after coming in he must first reach the floor (Y `$BD` from the south door, X `$20` from the west door) before stepping aside. [`Link_ModifyDirInDoorway`](../src/bank5/doorway.asm#L52)
- **After a death.** CONTINUE puts Link back in `$7D` at (`$78`,`$CD`) with 3 full hearts whatever his containers; inventory (keys, bombs, boomerang), room flags and banked kill counts are kept (trace). [`Mode8Continue_ChoiceModes`](../src/bank5/mode8-continue-mode10-stairs.asm#L49), [`Mode3Unfurl_Init_Sub1`](../src/bank7/init-mode-dispatch.asm#L294)
- **After a death, the rooms.** The room history is cleared, so every cleared room of ordinary monsters refills when it is entered again: `$7E` held 5 Ropes again after a continue (trace). Items already taken stay taken; a dead Dodongo or Moldorm stays dead; shutters are closed again, so `$2E` and `$1E` must be cleared a second time. [`RoomHistory_Clear`](../src/bank7/init-mode-dispatch.asm#L271), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- **Wallmasters.** None in this level: nothing carries Link back to the start. See the monster column of [data/levelblock-uw-1to6-q1.md](../data/levelblock-uw-1to6-q1.md).

# Optional detours

Bomb wall spots: the bomb must end its blast within `$18` pixels of the wall's hotspot. Standing on the door
line and facing the wall does it: north wall from (`$78`,`$6D`), south wall from (`$78`,`$AD`), west wall from
(`$30`,`$8D`), east wall from (`$C0`,`$8D`), where the floor plan leaves that square free (in `$1E` the east wall
is reached from (`$D0`,`$7D`) facing down). The wall is open about 80 frames after the bomb is placed; only the
north wall of `$4F` was bombed in the running game (trace).

| Detour | Rooms added | Keys | Bombs | Worth |
| --- | --- | --- | --- | --- |
| Bombs of `$3F` | `$3E` E -> `$3F` -> back W (+2 entries) | 0 | +4 | 4 bombs before the boss without clearing anything; 4 Traps and 4 Keese |
| Map of `$5F` | `$5E` E (key door) -> `$5F` -> back (+2), or from `$4F` through its S bomb wall and back (+2) | -1 or 0 | 0 or -1 | The status bar map; 5 Gels |
| Compass of `$6F` | `$6E` E (key door) -> `$6F` -> back (+2), or from `$5F` through its S bomb wall | -1 or 0 | 0 or -1 | The Triforce room marker; 6 Gels |
| 5 rupees of `$2F` | `$2E` E (key door) -> `$2F` -> back (+2), or from `$3F` through its N bomb wall | -1 or 0 | 0 or -1 | 5 rupees after 5 Gels |
| Old man of `$1F` | from `$2F` north (open door) (+2 more) | 0 | 0 | The hint "DODONGO DISLIKES SMOKE." |
| Key of `$6C` | `$6E` W -> `$6D` -> W shutter (kill the 5 Ropes of `$6D`) -> `$6C` -> back (+4; +2 on the alternative start) | +1 | 0 | One key among 6 Ropes, taken at (`$78`,`$8D`) (trace) |
| Around the 8 Ropes | `$2E` E (key door) -> `$2F` -> N -> `$1F` -> bomb the W wall -> `$1E` (+2) | -1 | -1 | Enters `$1E` from the east without clearing `$2E`; `$1E` must still be cleared for its north shutter |
| The whole east column | from `$4F`: S wall -> `$5F` -> S wall -> `$6F`, back; N wall -> `$3F` -> N wall -> `$2F` -> `$1F` -> W wall -> `$1E` | 0 | -5, +4 in `$3F` | Every item of the level with no key spent |

- **`$3F`.** The four Traps sit in the corners at (`$20`,`$5D`), (`$20`,`$BD`), (`$D0`,`$5D`), (`$D0`,`$BD`) and charge along the top and bottom rows and the left and right columns, 1 heart per touch. Coming in by the west door and walking straight along Y `$8D` to X `$78` took the bombs without a Trap hit (a Keese cost half a heart); do not linger, and do not enter by a north or south wall hole, which puts Link on a Trap row (trace). [`Trap_Update_Full`](../src/bank1/trap-rupee-stash.asm#L189), [`Trap_AllowedDirs`](../src/bank1/trap-rupee-stash.asm#L167)
- **Gels** (`$2F`, `$5F`, `$6F`) die to the boomerang and drop nothing. The map and the compass lie in the open: no Gel has to die for them. [`Gel_Update`](../src/bank4/zol-gel.asm#L244), [`Drop_NoDropTypes`](../src/bank4/drop-items.asm#L96)
- **Key doors and bomb walls stay open** once opened, from both sides and across visits, so a key or bomb is paid once per doorway. [`Door_Update`](../src/bank5/update-doors.asm#L60)
- **`$1F`** is a blank room with the old man between two fires; Link cannot walk above Y `$8D` there. [`Person_BlockLinkMovingUp`](../src/bank1/modeb-cave-save-sprites.asm#L289)
- **Is a key detour worth it?** Only the map and compass are behind key doors alone, and neither changes a rule; keys are better carried out of the level. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)

# Hazards and rules that matter here

- **Shutters** (`$0E` W, `$1E` N and S, `$2E` N, `$4E` S, `$6D` W) open only by their room's secret trigger, here always "all monsters dead". They are not saved: a shutter Link walks in through closes behind him. A room with nothing left to kill opens them again at once. See "Doorways, doors, keys, shutters and bomb walls" in [mechanics.md](../mechanics.md). [`Door_TouchShutter`](../src/bank5/touch-door.asm#L126), [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152)
- **One-way pairs.** Four doorways are an open door on one side and a shutter on the other. Free direction (out through the open side): `$5E` -> `$4E`, `$0E` -> `$1E`, `$0D` -> `$0E`, `$6C` -> `$6D`. The other direction (`$4E` -> `$5E`, `$1E` -> `$0E`, `$0E` -> `$0D`, `$6D` -> `$6C`) needs the room Link is leaving cleared. [`Door_SetEnteringDoorwayOpened`](../src/bank5/mode7-scroll-init.asm#L368)
- **"All dead" items.** In `$7E`, `$4E`, `$3E`, `$4F`, `$1E`, `$0E` and `$2F` the room item is hidden until no monster is left (trigger 7); in `$3F`, `$5F`, `$6C` and `$6F` it lies in the open. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- **Ringleader room `$5E`.** Trigger 2: when object slot 1 is empty, every monster left dies. It opens nothing and shows nothing. [`Secret_Trigger2Ringleader`](../src/bank5/underworld-secrets.asm#L176)
- **Blocks.** Link is stopped by a block only with his lower half, and tiles are tested only on his 8-pixel grid points; the door rows and the middle column are closed by blocks in `$6E`, `$1E` (layout `$0D`), `$5E` (layout `$1F`) and `$0D` (layout `$29`), and by statues in `$7D`. Use the paths of the route table. See "Tile collision and walking half into blocks" in [mechanics.md](../mechanics.md). [`Tile_GetAheadOfObj`](../src/bank7/tile-collision.asm#L65)
- **Push blocks, stairs, false walls, dark rooms, water:** none in this level; no block here moves. [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`Door_TouchFalseWall`](../src/bank5/touch-door.asm#L90)
- **Bombable walls** (`$1E` E/`$1F` W, `$2F` S/`$3F` N, `$3F` S/`$4F` N, `$4F` S/`$5F` N, `$5F` S/`$6F` N) look like plain wall. The test is made once, when the blast ends, with the bomb within `$18` pixels of the wall's hotspot on both axes. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Bomb_WallHotspotsX`](../src/bank7/bomb.asm#L60)
- **Statues that shoot.** Room `$4F` (layout `$24`) is the only one: four sources near the corners, each on its own timer of 81 to 241 frames, half a heart per fireball, blocked only by the magic shield. The statues of `$7D` and `$0D` (other layouts) never shoot. [`Statue_RoomLayoutIds`](../src/bank4/statues.asm#L45), [`Statue_FireballStartTimes`](../src/bank4/statues.asm#L54)
- **Traps** exist only in `$3F`: they cannot be hurt or stunned, sense Link within `$0E` pixels of their row or column, charge at up to 1.75 pixels per frame toward the room's middle line and creep back at 0.5. [`Trap_Update_Full`](../src/bank1/trap-rupee-stash.asm#L189), [`Trap_Init_Full`](../src/bank1/trap-rupee-stash.asm#L134)
- **Old man (`$1F`).** He says "DODONGO DISLIKES SMOKE." and does nothing else. Hitting him does not kill him: it switches on two fireball sources beside his fires for the rest of the visit. [`Person_CheckCollisions`](../src/bank1/underworld-person-update.asm#L89), [`Person_TextSelectorsLevels1257`](../src/bank1/underworld-person-init.asm#L67)
- **Keys.** A key door takes one key and stays open; no key is ever needed on the route. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- **Refill rule.** A cleared room stays empty only while it is among the last six different rooms entered; walking a long detour brings the Ropes and Goriyas back, and with them the closed shutters. [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- **Knockback.** A hit pushes Link `$20` pixels and gives him about 48 frames in which nothing can hurt him; in `$3F` and `$4F` a push can carry him onto a Trap row or into a Goriya. [`Shove_Begin`](../src/bank1/shove-knockback.asm#L65)
- **Bubbles, Wallmasters, Like-Likes, hungry Goriya:** none in this level. See the monster column of [data/levelblock-uw-1to6-q1.md](../data/levelblock-uw-1to6-q1.md).
- **Code that names this level.** Levels 1, 2, 5 and 7 share the old men's text table and the persons that do more than talk (none of those is placed here); nothing in the code tests a room id of level 2. [`Person_TextSelectorsLevels1257`](../src/bank1/underworld-person-init.asm#L67)

# Second quest

In Q2 the level with internal number 2 is built from another level block, [`Level_BlockUW1to6Q2`](../src/bank6/level-blocks-common-data.asm#L78)
(listing: [data/levelblock-uw-1to6-q2.md](../data/levelblock-uw-1to6-q2.md)), and its level info is patched by
[`Quest_Q2LevelInfoPatch2`](../src/bank6/quest2-replacements.asm#L96) (see "Level 2, Q2" in
[data/levelinfo.md](../data/levelinfo.md)): other rooms, items and boss, and the game shows it as LEVEL-3. Its
entrance is still overworld room `$3C`, with another layout. This document does not cover it.
