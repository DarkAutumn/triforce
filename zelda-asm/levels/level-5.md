# Level 5: the recorder and the Digdogger

Level 5 is entered from overworld room `$0B`, the mountain top that is reached only through the Lost Hills
maze (`$1B`). It holds the recorder (in a treasure cellar under room `$05`), a heart container and a Triforce
piece; its boss is a Digdogger, which no weapon can touch until the recorder is played in its room, so the
recorder has to be fetched first. Link must bring the ladder (the way to the boss crosses a full row of water
in room `$66`) and at least one bomb (two bombable walls lie on the way to the recorder; room `$65` between
them holds bombs); every key the route needs lies inside. What sets it apart: two push blocks that hide
stairs behind rooms of blue Darknuts (11 of them on
the route), a tunnel cellar that is the only way to the recorder, and nine rooms with water.

> **Written from the ROM only.** Unlike levels 1-4, nothing here was walked in the running game; the route, the
> requirements and the room notes come from the level data and the code. When you work on this level, check each
> step against the game and the code, and update this document with what you find (mark it `(trace)`).

Contents (generated; `L` = line in this file):

<!-- toc -->
- L32 At a glance
- L71 What Link must bring
- L83 What Link gets here
- L126 Map
- L159 Rooms
- L193 Floor plans
- L428 Monsters and boss
- L456 Route
- L501 Optional detours
- L512 Hazards and rules that matter here
- L531 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 5 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-5 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | $0B; not in Q1, the secret works in Q2 only: $1B (tile object: $62 rock at $C0,$90) | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178), [`Flute_SecretRoomIdsOW`](../src/bank7/flute.asm#L36) |
| Start room | $76 | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Triforce room | $14 (room item: Triforce piece ($1B)); the field places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $24 (1 x Digdogger ($39)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | $07, $04 | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 23: 22 on the submenu map, 1 the map does not show ($15) | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (room and side that has it) | 12 ($05 E, $06 W, $06 S, $16 N, $24 E, $25 W, $26 E, $27 W, $46 S, $56 N, $66 E, $67 W) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (room and side that has it) | 10 ($15 S, $25 N, $16 E, $17 W, $64 E, $65 W, $65 E, $66 W, $67 S, $77 N) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (room and side that has it) | 6 ($24 N, $34 N, $65 N, $55 E, $56 W, $66 N) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (room and side that has it) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $14 S open / $24 shutter, $24 S open / $34 shutter, $55 S open / $65 shutter, $56 S open / $66 shutter | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $76 S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 3 ($05, $06, $64) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 5 ($27, $37, $46, $57, $66) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 7 ($16, $26, $27, $47, $55, $66, $77) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | bomb ($00): $56, $65; flute ($05): $04 (cellar); 5 rupees ($0F): $57; compass ($16): $37; map ($17): $46; heart container ($1A): $24; Triforce piece ($1B): $14 | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md) (rows with `lvl` = `5` or `c5`).
<!-- /generated:facts -->

- **Entrance (Q1).** Overworld room `$0B` leads to level 5: its layout `$0A` has an open cave mouth at X `$70` (column 7, row 4; lower tiles `$24`), so no item and no secret is needed; Link enters by standing at (`$70`, `$7D`). [data/levelblock-ow.md](../data/levelblock-ow.md) (row `$0B`), [data/room-layouts.md](../data/room-layouts.md) (`OW $0A`), [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178)
- The Armos at (`$B0`, `$80`) in `$0B` leaves stairs when it is woken; they lead to the same level, because a room has one destination. [`Armos_SecretRoomIds`](../src/bank4/ghini-armos.asm#L138), [`Armos_SecretXs`](../src/bank4/ghini-armos.asm#L143)
- Room `$0B` is closed on three sides; it is entered from `$1B` below it, and `$1B` is the Lost Hills: Link must leave it upward four times in a row (the first three bring `$1B` up again), see [mechanics.md](../mechanics.md) "# Lost Woods and Lost Hills (mazes)". `$1B` itself is entered from `$1A` (west) or `$1C` (east). [`Maze_LostHillsDirs`](../src/bank1/mazes-lost-woods-hills.asm#L41), [`Maze_CheckStep`](../src/bank1/mazes-lost-woods-hills.asm#L56)
- Room `$1B` carries the same destination with a rock at (`$C0`, `$90`), but that secret belongs to Q2: in Q1 the rock ignores Link. [`Secret_IsForOtherQuest`](../src/bank4/rock-wall-tree-secrets.asm#L331)
- Once Link has this level's Triforce piece, the recorder's whirlwind can set him down in `$0B` at (`$80`, `$AD`). [`Whirlwind_PrevRoomIds`](../src/bank1/whirlwind.asm#L81), [`Whirlwind_DropOffYs`](../src/bank1/whirlwind.asm#L348)
- **On entry** Link stands in the start room `$76` at X `$78`, Y `$DD`, facing up, in the S doorway. [`LevelInfo_StartY`](../src/Variables.inc#L717), [`Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink`](../src/bank5/mode3-mode6-init.asm#L129)
- **Leaving on foot:** the S door of `$76` has no room behind it; walking out loads the overworld and puts Link at the exit spot of `$0B`, (`$70`, `$7D`). [`Room_CalcNextRoomForDoor`](../src/bank5/next-room-calc.asm#L76), [`Mode_LeaveLevelToOW`](../src/bank5/next-room-calc.asm#L107), [`LevelEntranceRoomId`](../src/Variables.inc#L463)
- **After the Triforce piece** (room `$14`): the end-of-level sequence refills all hearts and puts Link at the same spot of `$0B`; see [items.md](../items.md) "# Triforce pieces and the Triforce of Power". [`Mode12EndLevel_Update_Sub4LeaveLevel`](../src/bank5/mode12-lay-out-room.asm#L156)
- **Kept when Link leaves and comes back:** unlocked key doors and bombed walls (both sides), taken room items, the visited-room map, a dead Digdogger. [`RoomFlags_SetDoorOpened`](../src/bank5/shutters-door-flags.asm#L41), [`RoomFlags_SetItemTaken`](../src/bank1/take-item.asm#L140), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- **Reset:** loading the level zeroes the kill counts and the room history, so every room but the boss room is full again; shutters are closed again on every room entry; a push block is back in place on every room entry and the stairs of `$05` are hidden again. [`Mode2Load_Init`](../src/bank7/init-mode-dispatch.asm#L147), [`LevelKillCounts`](../src/Variables.inc#L480), [`Door_Update`](../src/bank5/update-doors.asm#L60), [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)

# What Link must bring

- **Ladder: required for the boss and the Triforce piece, not for the recorder.** Layout `$18` of room `$66` has water in every column of row 2 (Y `$80`); the S, W and E doors are below it and the N door, the only way on to `$56` and the boss, is above it. Room `$55` (the other room that touches `$56`) is split the same way by a water column. With the ladder the route search finds 23 room entries; without it there is no route to `$24`. [`InvLadder`](../src/Variables.inc#L570), [`Link_EndMoveAndAnimate`](../src/bank7/link-animate-ladder.asm#L150), [items.md](../items.md) "# Ladder"
- **Bombs: at least 1, better 4 or more.** Two bombable walls lie on the way to the recorder (`$66` W, then `$65` W). The first bombs inside the level (room `$65`, 4 bombs) are behind the first wall, so that wall needs a bomb carried in (or dropped by a Gibdo of `$66`). More bombs make the 11 blue Darknuts much cheaper. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`InvBombs`](../src/Variables.inc#L558)
- **Keys: none.** The route opens 3 key doors (`$06` W, `$27` W, `$24` E) and passes 4 keys (`$66`, `$47`, `$27`, `$26`); keys carry over between levels, so each key brought saves one room of Gibdos (see the key balance under "# Route"). [`InvKeys`](../src/Variables.inc#L584), [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- **Sword:** any grade works everywhere. The wooden sword needs 8 hits on a blue Darknut or the small Digdogger and splits every Zol; the white sword needs 4 and kills a Zol outright. The white sword's cave is overworld room `$0A`, next to the entrance. [`Sword_DamagePoints`](../src/bank1/sword-bomb-arrow-collision.asm#L182), [`Obj_TypeHpPairs`](../src/bank7/object-update-table.asm#L105)
- **Hearts and ring:** nothing in the code asks for a number. Gibdo, blue Darknut, Pols Voice and Digdogger take 2 hearts per touch before rings, and the route kills 11 blue Darknuts, at least 3 Gibdos and the Digdogger; the blue ring halves all of it. [`Damage_ObjTypeDamageToLink`](../src/bank1/monster-collisions.asm#L200), [`Link_BeHarmed`](../src/bank1/link-collision-harm.asm#L193)
- **Candle: not needed.** Rooms `$27`, `$37`, `$46`, `$57`, `$66` are dark, but the dark flag is read only by the palette fades of room entry and scrolling and by the candle's own light routine; walls, water, monsters and collisions are the same as in a lit room. A candle (or the book's fire) only shows the picture. [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124), [`Mode7Scroll_Init_Sub5`](../src/bank5/mode7-scroll-init.asm#L256), [`Candle_UpdateRoomLight`](../src/bank4/shoot-helpers-candle.asm#L132)
- **Bow and arrows: optional.** An arrow kills a Pols Voice outright (10 wooden sword hits otherwise) and costs 1 rupee; the route walks past the Pols Voices of `$25` and `$27` without having to kill them. [`Collision_CheckShotVsMonster`](../src/bank1/sword-bomb-arrow-collision.asm#L287), [`Arrow_Wield`](../src/bank5/wield-items.asm#L187)
- **Rupees: optional,** 100 for the bomb capacity offer in room `$17`. [`Person_ComplexState2SellBombCapacity`](../src/bank1/underworld-person-update.asm#L171)
- **Not needed:** raft, food, magic key, boomerang, rod. The recorder is found here. No room of the level has a false wall, a trap, a Bubble, a Wallmaster or a Like-Like. [`Level_BlockUW1to6Q1`](../src/bank6/level-blocks-common-data.asm#L70)

# What Link gets here

## Recorder (item `$05`), cellar `$04` under room `$05`

- **Where:** treasure cellar `$04`, item at (`$80`, `$90`); its stairs are in room `$05`, whose only door is the key door E to `$06`; `$06` is reached only through the tunnel cellar `$07` from `$64` (or through `$16`, which is itself only reached from `$06`). [data/levelblock-uw-1to6-q1.md](../data/levelblock-uw-1to6-q1.md), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
- **To make the stairs appear:** kill the 6 blue Darknuts of `$05`, then push the block at X `$70`, Y `$90` (column 5, row 3) for `$10` frames in any direction; stairs appear at (`$D0`, `$60`), the top right floor square. [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`Secret_Trigger5BlockStairs`](../src/bank5/underworld-secrets.asm#L226)
- **In the cellar:** Link comes down the left ladder (X `$30`); the walkway is at Y `$BD`, a second ladder at X `$B0` goes up to the ledge at Y `$8D`, and the recorder lies on it at X `$80` (from the cellar floor plan). Taking it halts Link while he holds it up. Both exits lead back to `$05`, where he appears at (`$90`, `$8D`). [`Mode9Cellar_Init_Sub8_EnterCellar`](../src/bank5/cave-cellar-enter.asm#L276), [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43), [`Item_Take`](../src/bank1/take-item.asm#L259)
- **What it changes, in a level:** played in a Digdogger's room (B button, selected slot 5; play freezes for `$98` frames) it makes the big Digdogger shrink for `$40` frames and split into small ones that every weapon hurts. Without it the big one cannot be affected at all. It must be played after entering that room: the flag is cleared on room entry. [`Flute_Wield`](../src/bank7/flute.asm#L49), [`Digdogger_HandleFlute`](../src/bank4/digdogger.asm#L91), [`FluteUsedState`](../src/Variables.inc#L439), [items.md](../items.md) "# Recorder (flute, whistle)"
- **First needed:** room `$24` of this level (the boss; the heart container and the N shutter to the Triforce room depend on its death). Later: the Digdoggers of level 7. [enemies.md](../enemies.md) "# Digdogger", [`Digdogger_Update`](../src/bank4/digdogger.asm#L202)
- **What it changes, on the overworld:** with at least one Triforce piece it calls the whirlwind, which carries Link to the entrance room of a level whose piece he holds (each use steps to the next such level, up or down by his facing); in overworld room `$42` (Q1) it drains the pond and uncovers stairs instead, which is how level 7 is entered. [`Whirlwind_Summon`](../src/bank1/whirlwind.asm#L243), [`Flute_SecretRoomIdsOW`](../src/bank7/flute.asm#L36)
- It costs nothing and can be played again as soon as the freeze ends. [`InvFlute`](../src/Variables.inc#L563)

## Heart container (item `$1A`), room `$24`

- **Where:** boss room `$24`, at (`$80`, `$70`); hidden until no monster is left in the room (secret trigger 7), that is until the small Digdogger is dead. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- **What it changes:** one more container and one full heart (the other hearts are not refilled). The container count is what the sword caves test (5 for the white sword, 12 for the magic sword), and the sword beam needs full health. [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337), [`HeartValues`](../src/Variables.inc#L593), [`SwordShot_Make`](../src/bank7/sword-and-rod.asm#L327), [items.md](../items.md) "# Hearts, Heart Containers and Fairies"

## Triforce piece (item `$1B`), room `$14`

- **Where:** room `$14`, at (`$78`, `$90`), inside a ring of blocks open at the bottom (columns 5 and 6 of row 5): walk straight up from the S door at X `$78`. The room has no monsters; its only door is S, to the boss room, whose N shutter opens when the Digdogger is dead. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152)
- **What it changes:** sets bit 4 of `InvTriforce`, refills the hearts and ends the level (Link comes out in overworld room `$0B`). The piece makes level 5 a whirlwind destination, and all eight are needed to pass the first old man of level 9. [`Item_TakeClass0PerLevel`](../src/bank1/take-item.asm#L421), [`InvTriforce`](../src/Variables.inc#L597), [`Mode12EndLevel_Init`](../src/bank5/mode12-lay-out-room.asm#L56), [items.md](../items.md) "# Triforce pieces and the Triforce of Power"

## Keys (item `$19`): 7 in the level

- `$66` (3 Gibdos, dark): not hidden. While a Gibdo is in object slot 1 the key rides on it and is taken by touching that Gibdo (2 hearts unless it is stunned) or by killing it and walking over the spot where it died. [`Item_MoveAndDrawRoomItem`](../src/bank7/room-item-draw.asm#L65), [`Item_TryTake`](../src/bank1/take-item.asm#L206)
- The carrier may start above the water row: of the nine spawn spots used when Link enters facing up, four are in rows 0 and 1, and a Gibdo cannot cross water. Then Link reaches it with the ladder, or re-enters the room (the spot cycle moves on from visit to visit). [`Spawn_PosListFacingUp`](../src/bank5/mode-inits-obj-lists.asm#L142), [`Spawn_AssignPositions`](../src/bank5/spawn-positions.asm#L64)
- `$66`, if no Gibdo carries it: the key lies at (`$C0`, `$80`), on the water row. Link at X `$C0` walking up stops at Y `$85` under the water square and the pickup test (|Link Y + 3 - item Y| < 9) passes there, so it can be taken from below without the ladder (from the tile rule and the pickup rule; unverified in play). [`Walker_TestTileAhead`](../src/bank7/walker-collision-screen-edge.asm#L138), [`Item_TryTake`](../src/bank1/take-item.asm#L206)
- `$47` (5 Gibdos) and `$26` (5 Gibdos): hidden; appear at (`$80`, `$A0`) when the room is empty. In `$26` that spot is inside the water ring: one ladder crossing. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- `$27` (dark; 2 Pols Voices, 2 Gibdos, 2 blue Keese): not hidden, at (`$80`, `$A0`), reached from the S door through the gap in the bottom of the water ring (columns 4-7 of row 5). No fight needed. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- `$77` (5 Pols Voices), `$55` (5 Zols), `$16` (8 blue Keese, not hidden, below the water row): off the route, see "# Optional detours". [data/levelblock-uw-1to6-q1.md](../data/levelblock-uw-1to6-q1.md)
- **What a key changes:** one key opens one key door for good, from both sides; the count is one byte shared by all levels, so keys left over here open doors in later levels. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`InvKeys`](../src/Variables.inc#L584), [items.md](../items.md) "# Keys and Magic Key"

## Bombs (item `$00`) and 5 rupees (item `$0F`)

- `$65`: a bomb item (4 bombs, up to the capacity), not hidden; it rides on the Gibdo in slot 1, else it lies at (`$C0`, `$80`). It is on the route, right behind the first bombed wall. [`Item_MoveAndDrawRoomItem`](../src/bank7/room-item-draw.asm#L65), [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337)
- `$56`: a bomb item at (`$80`, `$70`), hidden until the 3 Dodongos are dead. `$57`: 5 rupees at (`$80`, `$A0`), hidden until the 5 Zols are dead. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- **What bombs change:** they open the 5 bombable walls of the level, hurt Darknuts (`$40`, as much as the magic sword, from the side or behind) and are the only way to kill a Dodongo. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Bomb_CheckBombOrFireVsMonster`](../src/bank1/sword-bomb-arrow-collision.asm#L126), [items.md](../items.md) "# Bombs"

## Compass (item `$16`), room `$37`, and map (item `$17`), room `$46`

- Compass: hidden until the 3 red Darknuts of `$37` are dead; then at (`$80`, `$A0`), which is on the path through that room. It makes the status bar map show the Triforce room (`$14`). [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51), [`Inv_HasCompass`](../src/bank5/submenu-map.asm#L143)
- Map: at (`$80`, `$A0`) in `$46`, not hidden, no monsters; reachable only from the S key door (see "# Optional detours"). It draws the level's rooms on the status bar and submenu maps. [`Inv_HasMap`](../src/bank5/submenu-map.asm#L153), [items.md](../items.md) "# Map and Compass"

# Map

<!-- generated:map -->
```
         [05  ] K [06  ]
                    K
[14T ]   [15h ]   [16k ] X [17  ]
 |/S       X
[24Bi] K [25  ]---[26k ] K [27k ]
 |/S                         |
[34  ]                     [37i ]
                             |
                  [46i ]---[47k ]
                    K        |
         [55k ] S [56i ]---[57i ]
          |/S      |/S
[64  ] X [65i ] X [66k ] K [67  ]
                    |        X
                  [76S ]---[77k ]

[73S] = room $73 and its marks: S start room, B boss room, T Triforce room,
        k a key lies here, i another room item lies here, h not shown on the submenu map.
Between two rooms, the door: --- or | open, K key door, S shutter, X bombable wall,
        F false wall (walk-through), nothing = wall. A/B = the two sides differ: the left
        or upper room's side first, . = wall on that side.
Doors on sides with no room of the level: $76 S open.
```

- Cellar `$07` (tunnel cellar): joins `$64` (left exit) and `$06` (right exit), whose stairs lead here; item: none. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md)

- Cellar `$04` (treasure cellar): both exits lead to `$05`, whose stairs lead here; item: flute ($05). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md)
<!-- /generated:map -->

# Rooms

Generated from the level data by `tools/romdata.py`; do not edit between the markers.

<!-- generated:rooms -->
| room | N | S | W | E | monsters | item | secret | block | dark | layout | floor | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $05 | wall | wall | wall | key | 6 x Darknut blue ($0C) | - | push block: stairs appear | B | - | $22 | blocks | stairs lead to cellar $04 |
| $06 | wall | key | key | wall | none | - | - | B | - | $1A | blocks, stairs | stairs lead to cellar $07 |
| $14 Triforce | wall | open | wall | wall | none | Triforce piece ($1B) at $80,$90 | - | - | - | $29 | blocks, statues | - |
| $15 off-map | wall | bomb | wall | wall | 1 x dungeon person ($4D) | - | - | - | - | $26 | blank | person $4D says: SECRET POWER IS SAID / TO BE IN THE ARROW. |
| $16 | key | wall | wall | bomb | 8 x Keese blue ($1B) | key ($19) at $80,$A0 | - | - | - | $18 | water | - |
| $17 | wall | wall | bomb | wall | 1 x dungeon person ($4F) | - | - | - | - | $26 | blank | person $4F says: I BET YOU'D LIKE / TO HAVE MORE BOMBS. |
| $24 boss | shutter | open | wall | key | 1 x Digdogger ($39) | heart container ($1A) at $80,$70 | all dead: shutters open, room item appears | - | - | $24 | statues | - |
| $25 | bomb | wall | key | open | 5 x Pols Voice ($16) | - | - | - | - | $07 | blocks | boss sound $40 |
| $26 | wall | wall | open | key | 5 x Gibdo ($30) | key ($19) at $80,$A0 | all dead: shutters open, room item appears | - | - | $14 | water | - |
| $27 | wall | open | key | wall | 2 x Pols Voice ($16), 2 x Gibdo ($30), 2 x Keese blue ($1B) | key ($19) at $80,$A0 | - | - | D | $15 | water | - |
| $34 | shutter | wall | wall | wall | 6 x Gibdo ($30) | - | all dead: shutters open | - | - | $1E | blocks | boss sound $40 |
| $37 | open | open | wall | wall | 3 x Darknut red ($0B) | compass ($16) at $80,$A0 | all dead: shutters open, room item appears | - | D | $17 | water | - |
| $46 | wall | key | wall | open | none | map ($17) at $80,$A0 | - | - | D | $12 | water | - |
| $47 | open | open | open | wall | 5 x Gibdo ($30) | key ($19) at $80,$A0 | all dead: shutters open, room item appears | - | - | $15 | water | - |
| $55 | wall | open | wall | shutter | 5 x Zol ($13) | key ($19) at $80,$A0 | all dead: shutters open, room item appears | - | - | $13 | water | - |
| $56 | key | open | shutter | open | 3 x Dodongo ($31) | bomb ($00) at $80,$70 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $57 | open | wall | open | wall | 5 x Zol ($13) | 5 rupees ($0F) at $80,$A0 | all dead: shutters open, room item appears | - | D | $17 | water | - |
| $64 | wall | wall | wall | bomb | 5 x Darknut blue ($0C) | - | - | B | - | $1A | blocks, stairs | stairs lead to cellar $07 |
| $65 | shutter | wall | bomb | bomb | 5 x Gibdo ($30) | bomb ($00) at $C0,$80 | - | - | - | $02 | blocks | - |
| $66 | shutter | open | bomb | key | 3 x Gibdo ($30) | key ($19) at $C0,$80 | all dead: shutters open | - | D | $18 | water | - |
| $67 | wall | bomb | key | wall | 1 x dungeon person ($4E) | - | - | - | - | $26 | blank | person $4E says: DIGDOGGER HATES / CERTAIN KIND OF SOUND. |
| $76 start | open | open | wall | open | none | - | - | - | - | $21 | statues, sand | - |
| $77 | bomb | wall | open | wall | 5 x Pols Voice ($16) | key ($19) at $80,$A0 | all dead: shutters open, room item appears | - | - | $1D | blocks | - |
| $07 cellar | - | - | - | - | 4 x Keese blue ($1B) | - | - | - | - | $3E | tunnel cellar | left exit to $64, right exit to $06; Link comes out at $60,$9D |
| $04 cellar | - | - | - | - | 4 x Keese blue ($1B) | flute ($05) at $80,$90 | - | - | - | $3F | treasure cellar | left exit to $05, right exit to $05; Link comes out at $90,$8D |
<!-- /generated:rooms -->

# Floor plans

<!-- generated:floors -->
One character per floor square, 12 columns x 7 rows inside the walls (left column = west, top row =
north): `.` floor (tile `$74`), `B` block (tile `$B0`; a push block object is placed on one in a room
with a push block), `W` water (tile `$F4`), `S` statue (tiles `$94` and `$B4`), `>` stairs (tile `$70`),
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L63), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L73), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
Stairs that a secret makes appear are not part of the layout.

```
layout $00  rooms: $56
............
............
............
............
............
............
............
```

```
layout $02  rooms: $65
............
............
..B......B..
............
..B......B..
............
............
```

```
layout $07  rooms: $25
............
.BBBBBBBBBB.
............
.BBBBBBBBBB.
............
.BBBBBBBBBB.
............
```

```
layout $12  rooms: $46
............
.WWWWWWWWWW.
.WWWWWWWWWW.
.WW......WW.
.WW......WW.
.WWWW..WWWW.
.WWWW..WWWW.
```

```
layout $13  rooms: $55
........W...
........W...
........W...
........W...
........W...
........W...
........W...
```

```
layout $14  rooms: $26
............
.WWWWWWWWWW.
.W........W.
.W........W.
.W........W.
.WWWWWWWWWW.
............
```

```
layout $15  rooms: $27,$47
............
.WWW....WWW.
.W........W.
.W.WWWWWW.W.
.W........W.
.WWW....WWW.
............
```

```
layout $17  rooms: $37,$57
WWWWW..WWWWW
W...W....W.W
W.W.W.WW.W.W
..W.W..W.W..
W.W.WW.W.W.W
W.W....W...W
WWWWW..WWWWW
```

```
layout $18  rooms: $16,$66
............
............
WWWWWWWWWWWW
............
............
............
............
```

```
layout $1A  rooms: $06,$64
............
......B.....
.....B.B....
....B.>.B...
.....B.B....
......B.....
............
```

```
layout $1D  rooms: $77
............
............
..BB....BB..
..BB....BB..
..BB....BB..
............
............
```

```
layout $1E  rooms: $34
............
............
.....BB.....
.....BB.....
.....BB.....
............
............
```

```
layout $21  rooms: $76
............
.S..S..S..S.
............
.S..S..S..S.
.....::.....
.S.:S::S:.S.
...::::::...
```

```
layout $22  rooms: $05
............
............
............
.....B......
............
............
............
```

```
layout $24  rooms: $24
S..........S
............
............
............
............
............
S..........S
```

```
layout $26  rooms: $15,$17,$67
____________
____________
____________
____________
____________
____________
____________
```

```
layout $29  rooms: $14
............
.BBBBBBBBBB.
.B..S..S..B.
.B.S....S.B.
.B........B.
.BBBB..BBBB.
............
```

```
layout $3E (tunnel cellar)  rooms: $07
     c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 cA cB cC cD cE cF
r0   02 02 02 00 02 02 02 02 02 02 02 02 00 02 02 02
r1   03 03 03 01 03 03 03 03 03 03 03 03 01 03 03 03
r2   03 03 03 01 03 03 03 03 03 03 03 03 01 03 03 03
r3   03 03 03 01 03 03 03 03 03 03 03 03 01 03 03 03
r4   03 03 03 01 03 03 03 03 03 03 03 03 01 03 03 03
r5   03 03 03 01 03 03 03 03 03 03 03 03 01 03 03 03
r6   03 03 03 01 03 03 03 03 03 03 03 03 01 03 03 03
r7   03 03 02 01 02 02 02 02 02 02 02 02 01 02 03 03
r8   03 03 0C 01 0C 0C 0C 0C 0C 0C 0C 0C 01 0C 03 03
r9   03 03 03 03 03 03 03 03 03 03 03 03 03 03 03 03
r10  03 03 03 03 03 03 03 03 03 03 03 03 03 03 03 03
squares (first tile): 00 = $24, 01 = $6F, 02 = $F3, 03 = $FA, 0C = $F3
```

```
layout $3F (treasure cellar)  rooms: $04
     c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 cA cB cC cD cE cF
r0   02 02 02 00 02 02 02 02 02 02 02 02 02 02 02 02
r1   03 03 03 01 03 03 03 03 03 03 03 03 03 03 03 03
r2   03 03 03 01 03 03 03 03 03 03 03 03 03 03 03 03
r3   03 03 03 01 03 03 03 02 02 02 02 02 02 02 03 03
r4   03 03 03 01 03 03 03 02 02 02 02 02 02 02 03 03
r5   03 03 03 01 03 03 03 0C 0C 0C 0C 0C 0C 0C 03 03
r6   03 03 03 01 03 03 03 03 03 03 03 01 03 03 03 03
r7   03 03 02 01 02 02 02 02 02 02 02 01 02 02 03 03
r8   03 03 0C 01 0C 0C 0C 0C 0C 0C 0C 01 0C 0C 03 03
r9   03 03 03 03 03 03 03 03 03 03 03 03 03 03 03 03
r10  03 03 03 03 03 03 03 03 03 03 03 03 03 03 03 03
squares (first tile): 00 = $24, 01 = $6F, 02 = $F3, 03 = $FA, 0C = $F3
```

A cellar is laid out by the overworld routine: 16 columns x 11 rows of overworld square indexes,
as in [`room-layouts.md`](../data/room-layouts.md) (sections "Cave and cellar layouts" and "Overworld squares").
[`Mode9Cellar_Init_LayOutCellar`](../src/bank5/layout-room-ow.asm#L418), [`RoomLayout_SubroomLayoutAddrs`](../src/bank5/layout-room-ow.asm#L387)
<!-- /generated:floors -->

# Monsters and boss

| Monster (type) | Rooms (count) | HP | What kills or stops it | Touch (before rings) | [enemies.md](../enemies.md) section |
| --- | --- | --- | --- | --- | --- |
| Gibdo (`$30`) | `$66` (3), `$65` (5), `$47` (5), `$26` (5), `$34` (6), `$27` (2) | `$70` | everything: sword 7 / 4 / 2 hits (wooden / white / magic), bomb 2; boomerang stuns | 2 hearts | "# Gibdo" |
| Darknut blue (`$0C`) | `$64` (5), `$05` (6) | `$80` | sword, beam or bomb only, and not from the front: sword 8 / 4 / 2, bomb 2; never stunned | 2 hearts | "# Darknut (red, blue)" |
| Darknut red (`$0B`) | `$37` (3) | `$40` | as blue: sword 4 / 2 / 1, bomb 1 | 1 heart | "# Darknut (red, blue)" |
| Pols Voice (`$16`) | `$25` (5), `$77` (5), `$27` (2) | `$A0` | sword and beam only (10 / 5 / 3); any arrow kills at once; boomerang and bomb do nothing | 2 hearts | "# Pols Voice" |
| Zol (`$13`) | `$57` (5), `$55` (5) | `$20` | damage `$20` or more kills (white sword, bomb, arrow); the wooden sword splits it into 2 Gels; boomerang stuns | 1 heart | "# Zol" |
| Gel (`$14`, from a Zol) | where a Zol was split | 0 | any hit | half a heart | "# Gel" |
| Keese blue (`$1B`) | `$16` (8), `$27` (2), cellars `$07` and `$04` (4 each) | 0 | any hit, the boomerang included | half a heart | "# Keese (blue, red, black)" |
| Dodongo (`$31`) | `$56` (3) | - | two bombs swallowed, or one bomb's blast or cloud (stun) and then one sword hit | 1 heart | "# Dodongo" |
| Digdogger (`$39`) | `$24` (1, boss) | big: cannot be hit; small: `$80` | the recorder splits it; then sword 8 / 4 / 2, bomb 2, arrow 4 (silver 2) | 2 hearts | "# Digdogger" |
| Statue fireballs (`$55`) | `$24` (4 statues) | - | cannot be stopped; the magic shield blocks them | half a heart | "# Statue fireballs" |
| Old man (`$4D`, `$4E`, `$4F`) | `$15`, `$67`, `$17` | - | must not be hit | 0 | "# Old men and shopkeepers (cave and dungeon persons)" |

- **Blue Darknuts (`$64`: 5, `$05`: 6) are the level's real gate:** both push blocks stay fixed until their room is empty, so all 11 must die. A hit counts only when the weapon's direction is not exactly opposite to the Darknut's facing: strike from the side or from behind. [`Damage_HandleWeaponHit`](../src/bank1/weapon-damage-death.asm#L127), [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74)
- A bomb does `$40` to a Darknut under the same rule: the direction compared is the one Link faced when he put the bomb down, so do not place it while facing the Darknut head-on. Two unblocked bombs kill a blue Darknut. [`Bomb_CheckBombOrFireVsMonster`](../src/bank1/sword-bomb-arrow-collision.asm#L126), [`Darknut_Init`](../src/bank4/darknut-gibdo.asm#L44)
- A Darknut turns toward Link at a grid point about half the time when he is within 8 pixels of its row or column, so standing in line with its front invites it; it moves at 0.625 pixel per frame against Link's 1.5. [`Wanderer_MoveAndTurn`](../src/bank4/walker-wanderer-common.asm#L224), [`Darknut_Init`](../src/bank4/darknut-gibdo.asm#L44)
- **Gibdos** have no immunity; every hit knocks them back `$40` pixels. In `$66` only as many need to die as stand between Link and the key until the way N is wanted; then all of them, for the shutter. [`Gibdo_Update`](../src/bank4/darknut-gibdo.asm#L69), [`Shove_Begin`](../src/bank1/shove-knockback.asm#L65)
- **Pols Voices** (`$25`, `$27` on the route) need not be killed. They hop over blocks and water, do not chase Link and move 0.5 pixel per frame. [`PolsVoice_Update`](../src/bank4/pols-voice.asm#L82)
- **Dodongos (`$56`)** need not be killed either: the route enters by the S door and leaves by the open E door. They ignore the sword unless stunned. [`Dodongo_CheckBombHit`](../src/bank4/dodongo.asm#L400), [`Dodongo_CheckCollisions`](../src/bank4/dodongo.asm#L296)
- **Zols (`$57`)** need not be killed; with the wooden sword each hit makes two Gels instead of a kill. [`Zol_CheckCollisions`](../src/bank4/zol-gel.asm#L170), [`Zol_State2Split`](../src/bank4/zol-gel.asm#L113)
- **Boss, room `$24`:** one big Digdogger (type `$39`: it splits into one). Enter by the E key door, play the recorder anywhere in the room, wait out the freeze (`$98` frames) and the shrink (`$40` frames, touch is tested only on three of them), then kill the small one: HP `$80`, speed rising to 1.25-1.5 pixels per frame, 2 hearts per touch, no knockback when hit. [`Digdogger_InitWith1Child`](../src/bank4/aquamentus-digdogger-dodongo-inits.asm#L99), [`Digdogger_HandleFlute`](../src/bank4/digdogger.asm#L91), [`Digdogger_Update`](../src/bank4/digdogger.asm#L202)
- While the recorder's freeze runs nothing moves, Link included. With the clock held, or while the Digdogger is stunned, the shrink neither starts nor goes on. [`Mode5Play_Update`](../src/bank7/mode5-play-update.asm#L59), [`Digdogger_Update`](../src/bank4/digdogger.asm#L202)
- The four corner statues of `$24` (layout `$24`) shoot fireballs at Link all through the fight, half a heart each; a statue does not shoot while Link is within `$18` pixels of it on both axes. [`Statue_RoomLayoutIds`](../src/bank4/statues.asm#L45), [`Statue_Update`](../src/bank4/statues.asm#L86)
- **When the small Digdogger dies:** the heart container appears at (`$80`, `$70`) and the N shutter opens (trigger 7); the kill is banked at once and the room never makes a Digdogger again; the boss roar heard in `$25` and `$34` stops. The small one's own drop is the ordinary random one. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`RoomHistory_SaveKillCountUW`](../src/bank5/room-history-uw.asm#L110), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42)

# Route

Established from the level data and the code; not walked in the running game. The path is the result of a search over
the room graph (door types, in-room water with and without the ladder, keys, secret triggers, the two cellars) for the
fewest room entries, ties broken by fewer monsters killed: 23 entries counting cellars, 3 keys used, 2 walls bombed.
Coordinates are Link's `ObjX`, `ObjY`; floor columns 0-11 and rows 0-6 as in "# Floor plans" (X = `$20` + `$10` x column,
Link's Y = `$5D` + `$10` x row).

| Step | Room | Entered from | What to do | Leave by | Keys after | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `$76` start | entrance (S) | nothing here; walk up at X `$78` | N (open) | 0 | statues do not shoot (layout `$21`) |
| 2 | `$66` | S | take the key from the Gibdo that carries it (kill it, or touch it); bomb the W wall: put a bomb within `$18` pixels of (`$20`, `$8D`), e.g. from (`$30`, `$8D`) facing W | W (bombed) | 1 | dark; 3 Gibdos; the carrier may be above the water row (ladder); the N shutter is for step 14 |
| 3 | `$65` | E | take the bombs from the Gibdo that carries them (+4); bomb the W wall at (`$20`, `$8D`) | W (bombed) | 1 | 5 Gibdos, none has to die; the N shutter never opens from this side |
| 4 | `$64` | E | kill all 5 blue Darknuts; go round by row 6 to (`$60`, `$9D`), below the block at column 4, row 3; hold Up `$10` frames; when the block has moved, go up to Y `$8D` and right to the stairs at (`$80`, `$8D`) | stairs | 1 | never push that block from the W (see below) |
| 5 | cellar `$07` | left ladder (X `$30`) | go down, right along the walkway (Y `$BD`) to X `$C0`, up the right ladder and out the top | right exit | 1 | 4 blue Keese |
| 6 | `$06` | cellar; Link appears at (`$60`, `$9D`) | no monsters; walk to the W door (row 3) and use the key | W (key) | 0 | leave the block alone for now |
| 7 | `$05` | E | kill all 6 blue Darknuts; from (`$80`, `$8D`) hold Left `$10` frames against the block at (`$70`, `$90`) (any side works); stairs appear at (`$D0`, `$60`): stand at (`$D0`, `$5D`) | stairs | 0 | |
| 8 | cellar `$04` | left ladder | down, right along Y `$BD` to X `$B0`, up that ladder, left along Y `$8D` to the recorder at X `$80`; leave by the way Link came | either exit | 0 | 4 blue Keese |
| 9 | `$05` | cellar; Link appears at (`$90`, `$8D`) | nothing | E (open now) | 0 | the Darknuts are not back (room still in the history) |
| 10 | `$06` | W | walk to (`$60`, `$9D`), hold Up `$10` frames, then up and right to the stairs at (`$80`, `$8D`) | stairs | 0 | no monsters, so the block moves at once |
| 11 | cellar `$07` | right ladder (X `$C0`) | walkway to X `$30`, up the left ladder | left exit | 0 | |
| 12 | `$64` | cellar; Link appears at (`$60`, `$9D`) | walk round to the E door | E (open) | 0 | empty if all 5 Darknuts were killed on step 4 |
| 13 | `$65` | W | walk through | E (open) | 0 | the Gibdos not killed on step 3 are here |
| 14 | `$66` | W | kill every Gibdo in the room (the N shutter opens); at X `$78` walk up from row 3: the ladder is laid over the water row; go on to the N door | N (shutter) | 0 | dark; needs the ladder |
| 15 | `$56` | S | walk to the E door (row 3); the 3 Dodongos need not be fought | E (open) | 0 | |
| 16 | `$57` | W | follow the path between the water (below), or cross it with the ladder, to the N door | N (open) | 0 | dark; 5 Zols |
| 17 | `$47` | S | kill all 5 Gibdos; the key appears at (`$80`, `$A0`): take it | N (open) | 1 | skip the fight if a spare key was brought |
| 18 | `$37` | S | path: from the S door up to row 5, right to X `$80`, up to row 3, left to X `$70`, up to row 1, then to X `$78` and out | N (open) | 1 | dark; 3 red Darknuts need not die (the compass appears if they do) |
| 19 | `$27` | S | take the key at (`$80`, `$A0`) through the gap above the S door; go back to row 6, along it to column 0, up to the W door; use a key | W (key) | 1 | dark; 2 Pols Voices, 2 Gibdos, 2 Keese |
| 20 | `$26` | E | walk round the water ring by row 0 or row 6 to the W door | W (open) | 1 | 5 Gibdos need not die |
| 21 | `$25` | E | go up one row, along row 2 to column 0, down to the W door; use the key | W (key) | 0 | 5 Pols Voices; boss roar |
| 22 | `$24` boss | E | play the recorder; kill the small Digdogger; take the heart container at (`$80`, `$70`) | N (shutter, open now) | 0 | 4 statues shoot fireballs |
| 23 | `$14` Triforce | S | walk up at X `$78` through the gap in the blocks to the piece at (`$78`, `$90`) | (level ends) | 0 | no monsters |

- **Key balance:** keys used 3 (`$06` W on step 6, `$27` W on step 19, `$24` E on step 21); keys taken on the route 3 (`$66`, `$47`, `$27`); the count never goes below 0 and is 0 at the end. The level holds 7 keys and 6 key doorways: the three used, plus `$46` S / `$56` N (the map), `$06` S / `$16` N and `$66` E / `$67` W. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`InvKeys`](../src/Variables.inc#L584)
- **Alternative, same length:** leave the Gibdos of `$47` alone and take the key of `$26` on step 20 instead (kill its 5 Gibdos, cross one water square into the ring with the ladder). With one key brought from outside neither fight is needed: 15 monsters killed instead of 20. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Ladder_CheckAndDraw`](../src/bank5/ladder.asm#L44)
- **Order matters, recorder before boss:** `$24` cannot be cleared without the recorder, and its N shutter and heart container wait for the kill. Going to the boss first only costs keys. [`Digdogger_HandleFlute`](../src/bank4/digdogger.asm#L91), [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152)
- **Order matters, a key before `$06`:** the W door of `$06` is the first key door, and the only keys Link can reach before it are those of `$66` (on the way) and `$77` (a detour). With no key he stands in `$06` in front of two locked doors and must go back through the cellar. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- **The blocks of `$64` and `$06`:** the push block is the first block of row 3 from the left, at column 4 (X `$60`); the stairs are at column 6, behind the floor square of column 5. Pushed up or down it frees the way; pushed right (from the W side) it lands on column 5 and closes the way again. A block moves once per room entry: after a wrong push, leave the room and come back. [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`PushBlock_State1Moving`](../src/bank4/push-block.asm#L165)
- **`$66` on step 14:** a room cleared earlier stays empty only while it is among the last six rooms entered; by step 14 the rooms `$65`, `$64`, `$07`, `$06`, `$05`, `$04` have pushed `$66` out, so three Gibdos stand there again if all three were killed on step 2 (if only some were killed, the rest remain). They can start on either side of the water. The shutter needs the room empty. [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50), [`Room_RunEntryTasks`](../src/bank7/init-mode5-play.asm#L169)
- **`$57`, the path without the ladder:** from the W door go right one square, up column 1 to row 1, right to column 3, down column 3 to row 5, right to column 6, up column 6 to row 3, left to column 5, up column 5 to row 1, then to X `$78` and out N. With the ladder: straight along row 3 over the water of columns 2 and 4 (two crossings), then up column 5. [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L73), [`Ladder_CheckAndDraw`](../src/bank5/ladder.asm#L44)
- **After a death:** CONTINUE puts Link back in `$76` with 3 hearts. Opened doors and bombed walls, taken items (the recorder, keys, the heart container) and the kill counts stay; the room history is cleared, so rooms that were fully cleared refill, except the boss room. From `$76` the recorder side is two open walls away and the boss side is step 14 onward. [`Mode8Continue_ChoiceModes`](../src/bank5/mode8-continue-mode10-stairs.asm#L49), [`Mode8Continue_Init`](../src/bank5/mode-inits-obj-lists.asm#L49), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- **Wallmasters:** the level has none, so nothing carries Link back to the start. [data/levelblock-uw-1to6-q1.md](../data/levelblock-uw-1to6-q1.md)
- **Nothing on the route locks Link in:** every room is entered through an open door, a key door or a bombed wall, which stay open, or from a cellar. Shutters close behind Link only in `$55` and `$34` (see "# Optional detours"). [`Door_TouchShutter`](../src/bank5/touch-door.asm#L126), [`Room_EnterCommon`](../src/bank5/enter-room.asm#L143)

# Optional detours

- **Compass, `$37` (on the route):** kill the 3 red Darknuts on step 18; it appears on the path at (`$80`, `$A0`). No extra room, no key. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- **5 rupees, `$57` (on the route):** kill the 5 Zols on step 16; the rupees appear at (`$80`, `$A0`), which the path without the ladder passes. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- **Map, `$46`:** from `$56` (step 15) through the N key door: 2 extra room entries, 1 key spent, none gained (so a fourth key is needed: `$26` or one of the keys below). The S door opens into a pocket enclosed by water two squares wide; the map lies in it at (`$80`, `$A0`). The E door to `$47` is on the other side of that water and cannot be reached from the pocket, nor the map from `$47`: the ladder spans one square. [`Ladder_CheckAndDraw`](../src/bank5/ladder.asm#L44), [items.md](../items.md) "# Ladder"
- **Key of `$77` and the old man of `$67`:** `$76` E (open) to `$77`: kill the 5 Pols Voices (arrows do it in one each) and the key appears at (`$80`, `$A0`). From `$77` a bomb on the N wall (bomb within `$18` pixels of (`$78`, `$5D`)) opens `$67`, whose man says "DIGDOGGER HATES CERTAIN KIND OF SOUND."; its W key door leads into `$66` below the water. 2 extra entries for the key alone (+1 key); through `$67` 1 bomb, 1 key spent and gained. [`Bomb_WallHotspotsX`](../src/bank7/bomb.asm#L60), [`Person_TextAddrs`](../src/bank1/cave-init-person-text.asm#L47)
- **Key of `$55` and the bombs of `$56`:** kill the 3 Dodongos of `$56` (one bomb each by blast and sword, two each when swallowed): the bomb item appears at (`$80`, `$70`) and the W shutter opens. In `$55` Link arrives E of the water column; the E shutter closes behind him. Kill the 5 Zols: the key appears at (`$80`, `$A0`), W of the water (one ladder crossing) and the shutter reopens. `$55` S is open and leads down into `$65`, whose N side is a shutter that never opens from below. +1 key; 2 or 3 extra entries. [`Dodongo_TryEatBomb`](../src/bank4/dodongo.asm#L469), [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Door_TouchShutter`](../src/bank5/touch-door.asm#L126)
- **More bombs, `$17`, and the key of `$16`:** from `$06` (step 6 or 10) through the S key door into `$16` (8 blue Keese); Link arrives above the water row, the key (`$80`, `$A0`) and the E wall are below it: one ladder crossing, then a bomb within `$18` pixels of (`$D0`, `$8D`). In `$17` the man offers "I BET YOU'D LIKE TO HAVE MORE BOMBS." for 100 rupees: stand at X `$78`, Y `$98` with 100 rupees; the capacity grows by 4 and the bombs are refilled. 4 extra entries, 1 key spent and 1 gained, 1 bomb, 100 rupees. [`Person_ComplexState2SellBombCapacity`](../src/bank1/underworld-person-update.asm#L171), [`InvMaxBombs`](../src/Variables.inc#L607)
- **Old man of `$15`** (not on the submenu map): a bomb on the N wall of `$25` (within `$18` pixels of (`$78`, `$5D`)); he says "SECRET POWER IS SAID TO BE IN THE ARROW." 2 extra entries, 1 bomb, nothing gained. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Person_TextAddrs`](../src/bank1/cave-init-person-text.asm#L47)
- **`$34`, below the boss room: nothing to gain.** 6 Gibdos, no item; its N side is a shutter that opens only when all six are dead, so walking in means fighting out. [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152)

# Hazards and rules that matter here

- **Shutters** open only on the room's own trigger: "no monsters left" in `$24`, `$34`, `$55`, `$56`, `$66`. `$65` has a shutter (N) and no trigger: it never opens from that side, and `$14` S, `$24` S, `$55` S, `$56` S are plain open doors on their own side. An opened shutter is not saved and is closed again on the next entry. [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68), [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152), [`Door_Update`](../src/bank5/update-doors.asm#L60)
- Entering a room through a shutter side (into `$55` from `$56`, into `$34` from `$24`) closes it behind Link until the room's trigger fires. [`Room_EnterCommon`](../src/bank5/enter-room.asm#L143), [`Door_TouchShutter`](../src/bank5/touch-door.asm#L126)
- **Push blocks** (`$05`, `$06`, `$64`): the block is the first block tile of the row at Y `$90`; it will not move while a monster is in the room; Link must be exactly in line with it, within `$10` pixels, holding only the direction toward it for `$10` frames in a row; it moves once per entry. In `$05` the push makes stairs at (`$D0`, `$60`); in `$06` and `$64` it only clears the way to stairs that are part of the layout. [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`Secret_Trigger5BlockStairs`](../src/bank5/underworld-secrets.asm#L226)
- While a block slides Link cannot move if he is within `$10` pixels of it on both axes (`$20` frames). [`Link_CheckSlidingTileObjBlocking`](../src/bank1/tile-block-power-triforce.asm#L45)
- **Bomb walls** (`$15` S / `$25` N, `$16` E / `$17` W, `$64` E / `$65` W, `$65` E / `$66` W, `$67` S / `$77` N): they look like plain wall; the bomb itself must lie within `$18` pixels of the wall's hotspot (N (`$78`, `$5D`), S (`$78`, `$BD`), W (`$20`, `$8D`), E (`$D0`, `$8D`)) when its blast ends; a bomb is put `$10` pixels ahead of Link. The hole stays, on both sides. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Bomb_WallHotspotsX`](../src/bank7/bomb.asm#L60), [`Link_PlaceWeaponAhead`](../src/bank1/bomb-candle-wield.asm#L154)
- **Key doors** stay open from both sides once unlocked; a key cannot be used while another door is opening or closing. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`Door_Update`](../src/bank5/update-doors.asm#L60)
- **Water and the ladder:** water (tile `$F4`) stops Link and every walking monster. With the ladder Link crosses one square: he must walk straight at the water from a grid point with the direction held; the square behind the water must be walkable. Two squares in a row (`$46`) cannot be crossed. A free monster slot is needed, and the ladder is not laid in a doorway. [`Link_EndMoveAndAnimate`](../src/bank7/link-animate-ladder.asm#L150), [`Ladder_CheckAndDraw`](../src/bank5/ladder.asm#L44), [mechanics.md](../mechanics.md) "# Ladder and raft"
- Rooms with water: `$16` and `$66` (a full row), `$55` (a full column), `$26`, `$27`, `$47` (rings with gaps or without), `$37` and `$57` (a maze that can be walked dry), `$46` (the map pocket). [data/room-layouts.md](../data/room-layouts.md)
- **Dark rooms** (`$27`, `$37`, `$46`, `$57`, `$66`): only the palette is dark. Entering one fades the picture out; a candle's standing fire fades it in for that visit, and play is frozen during that fade. Nothing else reads the flag. [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124), [`Candle_UpdateRoomLight`](../src/bank4/shoot-helpers-candle.asm#L132), [items.md](../items.md) "# Candle (blue, red)"
- **Room items on Gibdos:** in `$65` and `$66` the visible room item is carried by the Gibdo in object slot 1 and stays where that Gibdo dies. A hidden item (`$26`, `$47`) is not carried and appears at its own spot. The slot-1 Gibdo never drops anything else. [`Item_MoveAndDrawRoomItem`](../src/bank7/room-item-draw.asm#L65), [`Drop_SetUpItem`](../src/bank4/drop-items.asm#L157)
- **Statues:** only layout `$24` (the boss room) shoots. The statues of `$76` and `$14` are scenery. [`Statue_RoomLayoutIds`](../src/bank4/statues.asm#L45)
- **Old men** (`$15`, `$17`, `$67`): Link cannot walk above Y `$8E` in their rooms. A weapon hit does not kill the man; it turns on two fireball sources beside the fires until Link leaves the room. [`Person_BlockLinkMovingUp`](../src/bank1/modeb-cave-save-sprites.asm#L289), [`Person_CheckCollisions`](../src/bank1/underworld-person-update.asm#L89)
- The man of `$17` (type `$4F`) is the only one who does more than talk; once paid he is gone for good (room flag `$10`). [`Person_UpdateUnderworld_Full`](../src/bank1/underworld-person-update.asm#L44), [`Person_ComplexState2SellBombCapacity`](../src/bank1/underworld-person-update.asm#L171)
- **Stairs and cellars:** stairs take Link when he stands exactly on them on a grid point (X a multiple of `$10`, Y low nibble `$D`). A cellar always holds four blue Keese; it is left by walking up off the top of a ladder, and the room Link returns to depends on which half of the cellar he is in (X below `$80` or not). He comes out at a fixed spot: (`$60`, `$9D`) from `$07`, (`$90`, `$8D`) from `$04`. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43)
- **Refilling rooms:** a cleared room stays empty only while it is one of the last six rooms entered (cellars count); a partly cleared one keeps its reduced number until the level is loaded again. The Digdogger's room never refills. [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50), [`RoomHistory_SaveKillCountUW`](../src/bank5/room-history-uw.asm#L110)
- **Not in this level:** false walls, traps, Bubbles, Wallmasters, Like-Likes, the hungry Goriya, a life-or-money man. No code path tests a room id of level 5; the only per-level tests that apply are the person routines chosen for levels 1, 2, 5, 7. [`Person_InitUnderworld_JumpTable`](../src/bank1/underworld-person-init.asm#L53), [`Person_UpdateUnderworld_Full`](../src/bank1/underworld-person-update.asm#L44)

# Second quest

- In Q2 the internal level 5 loads another level block ([data/levelblock-uw-1to6-q2.md](../data/levelblock-uw-1to6-q2.md)) and a patched level info (start room `$7D`, Triforce room `$4F`, boss room `$5F`) and is shown as LEVEL-4; the dungeon shown as LEVEL-5 is internal level 4 ([data/levelinfo.md](../data/levelinfo.md)). [`Level_BlockAddrsQ2`](../src/bank6/level-load-mode2.asm#L89), [`Quest_Q2LevelInfoPatch5`](../src/bank6/quest2-replacements.asm#L131), [`Quest_Q2LevelInfoPatch4`](../src/bank6/quest2-replacements.asm#L119)
- Nothing in this document (rooms, items, route) applies to either of them. [`Level_BlockUW1to6Q2`](../src/bank6/level-blocks-common-data.asm#L78)
