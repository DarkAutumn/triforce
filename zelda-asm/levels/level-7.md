# Level 7: the red candle, the hungry Goriya and a second Aquamentus

Level 7 lies under the pond of overworld room `$42`; the recorder drains the pond and uncovers the stairs. Link
collects the red candle (cellar `$4A` under the unmapped room `$1A`), a heart container from Aquamentus in room
`$2A` and the Triforce piece in room `$2B`. He must bring the recorder (the entrance, and the Digdogger of room
`$1C`), the food (the Grumble Goriya of room `$28` blocks the only way north), the ladder (water in rooms `$49`
and `$18`) and bombs for eight walls: the candle room and the boss room are reached only through bombed walls
and a tunnel cellar. The level holds four keys and the route below spends two, so no key has to be brought.

> **Written from the ROM only.** Unlike levels 1-4, nothing here was walked in the running game; the route, the
> requirements and the room notes come from the level data and the code. When you work on this level, check each
> step against the game and the code, and update this document with what you find (mark it `(trace)`).

Contents (generated; `L` = line in this file):

<!-- toc -->
- L30 At a glance
- L67 What Link must bring
- L85 What Link gets here
- L138 Map
- L171 Rooms
- L215 Floor plans
- L494 Monsters and boss
- L530 Route
- L578 Optional detours
- L590 Hazards and rules that matter here
- L614 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 7 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-7 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | $42 (the recorder's Q1 secret); not in Q1, the secret works in Q2 only: $19 (tile object: $63 rock wall at $40,$70) | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178), [`Flute_SecretRoomIdsOW`](../src/bank7/flute.asm#L36) |
| Start room | $79 | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Triforce room | $2B (room item: Triforce piece ($1B)); the field places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $2A (1 x Aquamentus ($3D)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | $7B, $4A | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 33: 31 on the submenu map, 2 the map does not show ($08, $1A) | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (room and side that has it) | 10 ($0B S, $1B N, $18 E, $19 W, $1B E, $1C W, $28 S, $38 N, $48 S, $58 N) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (room and side that has it) | 20 ($08 S, $18 N, $08 E, $09 W, $0C E, $0D W, $19 E, $1A W, $1A E, $1B W, $29 E, $2A W, $39 E, $3A W, $59 S, $69 N, $5A S, $6A N, $68 E, $69 W) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (room and side that has it) | 11 ($09 S, $19 N, $09 E, $0A W, $1C N, $18 S, $2A E, $38 E, $49 N, $59 E, $6B N) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (room and side that has it) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $0C S open / $1C shutter, $18 S shutter / $28 open, $2A E shutter / $2B open, $38 E shutter / $39 open, $39 S open / $49 shutter, $59 E shutter / $5A open, $5B S open / $6B shutter | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $79 S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 3 ($0D, $1A, $29) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 3 ($18, $5A, $6A) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 4 ($0A, $3A, $6D, $78) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | bomb ($00): $0B, $0C, $1B, $68, $69, $6C, $7A; red candle ($07): $4A (cellar); 5 rupees ($0F): $09, $38, $58; compass ($16): $5A; map ($17): $18; heart container ($1A): $2A; Triforce piece ($1B): $2B | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md) (rows with `lvl` = `7` or `c7`).
<!-- /generated:facts -->

- Entrance: overworld room `$42` (cave index 7 = level 7). The stairs are under the pond; playing the recorder in that room in Q1 starts the pond secret instead of the whirlwind, and the stairs appear at X `$60`, Y `$90` 81 to 88 frames after the recorder's freeze ends. [`Flute_SecretRoomIdsOW`](../src/bank7/flute.asm#L36), [`FluteSecret_RevealPondStairs`](../src/bank7/flute-secret-pond.asm#L157), [items.md](../items.md) "# Recorder (flute, whistle)"
- The secret needs a free object slot in `$01`-`$09`; room `$42` makes one Moblin. Once revealed, room flag `$80` keeps the stairs open on every later visit, so the recorder is needed there once. [`Flute_Wield`](../src/bank7/flute.asm#L49), [`Secret_RevealAndFlagStairsObj`](../src/bank4/rock-wall-tree-secrets.asm#L283), [data/levelblock-ow.md](../data/levelblock-ow.md)
- Overworld room `$19` also carries cave index 7, behind a rock wall whose secret belongs to Q2 only: in Q1 it cannot be opened, and `$42` is the only entrance. [`Secret_IsForOtherQuest`](../src/bank4/rock-wall-tree-secrets.asm#L331), [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178)
- On entry Link stands in room `$79` at X `$78`, Y `$DD`, facing up; the room has no monsters and its statues do not shoot (layout `$21`). [`Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink`](../src/bank5/mode3-mode6-init.asm#L129), [`LevelInfo_StartY`](../src/Variables.inc#L717), [`Statue_RoomLayoutIds`](../src/bank4/statues.asm#L45)
- Walking out through the S door of `$79` leaves the level: the next room id is off the map, the overworld is loaded and Link comes out in room `$42`. [`Room_CalcNextRoomForDoor`](../src/bank5/next-room-calc.asm#L76), [`Mode_LeaveLevelToOW`](../src/bank5/next-room-calc.asm#L107), [`LevelEntranceRoomId`](../src/Variables.inc#L463)
- Taking the Triforce piece in `$2B` ends the level: hearts refill to full, the curtain closes and Link comes out of the entrance in overworld room `$42` (exit spot `$60`,`$5D` in the overworld listing). [`Mode12EndLevel_Update_Sub4LeaveLevel`](../src/bank5/mode12-lay-out-room.asm#L156), [`Mode12EndLevel_StartFillingHearts`](../src/bank5/mode12-lay-out-room.asm#L118), [data/levelblock-ow.md](../data/levelblock-ow.md)
- Kept when Link leaves and comes back: unlocked key doors and bombed walls (both sides), taken room items, the fed Grumble Goriya, and the rooms of non-recurring monsters once killed (Aquamentus, Digdoggers, Moldorms, the rupee stash). [`Door_Update`](../src/bank5/update-doors.asm#L60), [`RoomFlags_SetItemTaken`](../src/bank1/take-item.asm#L140), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- Reset: shutters close on every room entry; push blocks return to their square on every room entry; the per-room kill counts are zeroed when the level is loaded again, so every other cleared room is full again. [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`Mode2Load_Init`](../src/bank7/init-mode-dispatch.asm#L147)

# What Link must bring

Proved from the door types, floor plans and monster lists above and the code that reads them.

- **Recorder: required.** It opens the entrance (overworld `$42`), and in room `$1C` the big Digdogger cannot be hit by anything until the recorder is played in that room; the N shutter of `$1C`, the only way on to `$0C`, opens only when the room is empty. [`InvFlute`](../src/Variables.inc#L563), [`Digdogger_HandleFlute`](../src/bank4/digdogger.asm#L91), [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152)
- **Food: required**, for room `$28`. While the Grumble Goriya stands there Link cannot move up past Y `$8D`, and the N door of `$28` is the only way into the upper half of the level (rooms `$18`, `$08`-`$0D`, `$19`-`$1C`). Giving the food uses it up (`InvFood` = 0). [`InvFood`](../src/Variables.inc#L564), [`Person_BlockLinkMovingUp`](../src/bank1/modeb-cave-save-sprites.asm#L289), [`Grumble_State3TakeFood`](../src/bank1/grumble-goriya.asm#L109)
- Food is sold in the overworld shops for 100 and 60 rupees (see "# Caves" in the overworld listing). [data/levelblock-ow.md](../data/levelblock-ow.md), [items.md](../items.md) "# Food (bait, meat)"
- **Ladder: required.** Room `$49` (layout `$18`) has a full row of water (floor row 2) between the S door and the N door; room `$18` (layout `$16`) has one water square between the S door and the middle (row 5), one between the middle and the N door (row 1) and one between the middle and the E door (column 9). Each is one square wide, which is what the ladder spans. [`InvLadder`](../src/Variables.inc#L570), [`Link_EndMoveAndAnimate@SetUpLadder`](../src/bank7/link-animate-ladder.asm#L246), [items.md](../items.md) "# Ladder"
- **Bombs: required**, 8 walls on the route with no keys brought: `$69` N, `$39` E, `$18` N, `$08` E, `$19` E, `$1A` E, `$0C` E, `$29` E (5 with three keys brought, see Route). The boss room `$2A` has no door other than the bombable W wall and the shutter to `$2B`. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Door_TouchBombable`](../src/bank5/touch-door.asm#L108)
- Bomb supply: a new file holds at most 8 (`InvMaxBombs`). Room bombs on the route give 4 each: `$69` (kill 5 Goriya) and `$1B` (kill 6 Goriya); so 8 bombs with none wasted, or at least 2 on arrival plus both pickups. [`InvMaxBombs`](../src/Variables.inc#L607), [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337), [items.md](../items.md) "# Bombs"
- **Keys: none needed.** The level holds 4 keys (`$0A`, `$3A`, `$6D`, `$78`) and has 5 key doors; the route opens 2 (`$38` N, `$1B` E) and takes 2 (`$3A`, `$0A`). Every key brought shortens it: with 2 keys the E door of `$18` replaces the loop through `$08`, `$09`, `$0A`; with 3 the Moldorm room `$3A` is skipped too. The magic key does the same as 3 keys. [`InvKeys`](../src/Variables.inc#L584), [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`InvMagicKey`](../src/Variables.inc#L571)
- **Candle: not needed.** Rooms `$18`, `$5A` and `$6A` are dark; the dark flag is read only by the palette fades on room change and by the candle's own lighting routine. Monsters, collision, doors and items are the same in the dark; only the picture is black. [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124), [`Mode7Scroll_Init_Sub5`](../src/bank5/mode7-scroll-init.asm#L256), [`Candle_State0Begin`](../src/bank4/shoot-helpers-candle.asm#L147)
- **Raft, bow, arrows: not needed.** No monster of the level needs an arrow, so no rupees are spent on arrows. [enemies.md](../enemies.md) "# Aquamentus", "# Digdogger", "# Dodongo"
- **Sword: any grade.** Aquamentus (HP `$60`) takes 6 wooden, 3 white or 2 magic sword hits; a small Digdogger (HP `$80`) 8, 4 or 2; a blue Goriya (HP `$50`) 5, 3 or 2. The route forces at least 20 Goriya kills, so the white sword or better is worth having. [`Sword_DamagePoints`](../src/bank1/sword-bomb-arrow-collision.asm#L182), [`Obj_TypeHpPairs`](../src/bank7/object-update-table.asm#L105)
- **Hearts: no gate.** Nothing in the level tests the heart count. The hardest hits are 2 hearts from a Digdogger (either size) and 1 heart from a blue Goriya, its boomerang, a Dodongo, a trap or Aquamentus, before rings. [`Damage_ObjTypeDamageToLink`](../src/bank1/monster-collisions.asm#L200), [mechanics.md](../mechanics.md) "# Hearts, rings and death"
- **Magic shield: optional.** It parries the type `$55` fireballs of the statues and of Aquamentus; any shield parries a Goriya's boomerang. [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](../src/bank1/link-collision-harm.asm#L138), [items.md](../items.md) "# Shield (small, magic)"
- **Rupees: optional**, 100 for the bomb capacity man in `$48` (see Optional detours). [`Person_ComplexState2SellBombCapacity`](../src/bank1/underworld-person-update.asm#L171)

# What Link gets here

## Red candle (item `$07`), cellar `$4A` under room `$1A`

- Room `$1A` is not drawn on the level's map and has no door: it is entered through the bombable E wall of `$19` or the bombable W wall of `$1B`. [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729), [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136)
- The stairs are part of the layout (floor column 6, row 3), closed in by a diamond of blocks. The push block is the block at column 4, row 3 (X `$60`, Y `$90`): the first block tile from the left in that row. [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L73)
- The block moves only when the room has no monsters left: kill the 3 blue and 3 red Goriya first. [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124)
- Push it down from above (Link at X `$60`, Y `$7D`, holding Down) or up from below (X `$60`, Y `$9D`, holding Up) for `$10` frames in a row. It slides one square and the square at column 4, row 3 becomes floor; from there columns 5 and 6 of row 3 are open. A push to the right (from column 3) moves the block onto column 5 and leaves the stairs closed until the room is entered again. [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`PushBlock_State1Moving`](../src/bank4/push-block.asm#L165), [`Tile_ChangeTileObjTiles`](../src/bank7/tile-obj-tiles.asm#L57)
- The room's secret trigger is 0: the push itself opens nothing; it only clears the way to stairs that are already there. [`Secret_RunTrigger_JumpTable`](../src/bank5/underworld-secrets.asm#L105)
- Stairs: Link must stand on the stairs square with X a multiple of `$10` (X `$80`) and Y `$8D`, grid offset 0. He enters cellar `$4A` at X `$30`, walks down to Y `$5D`; four blue Keese are made. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`Mode9Cellar_Init_Sub8_EnterCellar`](../src/bank5/cave-cellar-enter.asm#L276), [`Spawn_AssignPositions@AssignSpecialPositions`](../src/bank5/spawn-positions.asm#L128)
- The candle lies at X `$80`, Y `$90`. From the cellar's floor plan: down the ladder at X `$30`, east along the passage, up the short ladder at X `$B0`, west along the upper ledge (the walk inside the cellar is read from the layout grid only, unverified). [`Mode9Cellar_Init_LayOutCellar`](../src/bank5/layout-room-ow.asm#L418), [`Item_TryTakeRoomItem`](../src/bank1/take-item.asm#L173)
- Leave by climbing off the top (Y < `$40` with Up held): both exits lead back to `$1A`, where Link appears at X `$60`, Y `$9D`, outside the diamond; the room is then empty because the stairs banked its kills. [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43), [`RoomHistory_SaveKillCount`](../src/bank5/warps-stairs-caves.asm#L151)

**What it changes**

- `InvCandle` becomes 2. The blue candle makes one fire per room entry (`HasUsedCandle`); the red candle skips that test, so the only limit is the two bomb-or-fire slots `$10` and `$11`: two fires at once, as often as wanted. [`Candle_Wield@NotInUse`](../src/bank1/bomb-candle-wield.asm#L233), [`InvCandle`](../src/Variables.inc#L562), [items.md](../items.md) "# Candle (blue, red)"
- A fire is a weapon: damage `$10` (one wooden sword hit) to a monster within `$0E` pixels, moving or standing, and half a heart to Link himself. With the red candle it can be used over and over in one room. [`Bomb_CheckBombOrFireVsMonster`](../src/bank1/sword-bomb-arrow-collision.asm#L126), [`Fire_Update@DrawAndCheckCollisions`](../src/bank7/fire.asm#L92)
- A standing fire lights a dark dungeon room until the room is left. In this level that is `$18` (crossed before the candle is taken, and again after a restart), `$5A` and `$6A`; the light changes the picture only. [`Fire_Update@StandingFire`](../src/bank7/fire.asm#L75), [`Candle_UpdateRoomLight`](../src/bank4/shoot-helpers-candle.asm#L132)
- Later: either candle burns the trees that hide overworld stairs (a standing fire resting on the tree); the entrance of level 8 is behind such a tree in overworld room `$6D`. The red candle allows a second try without leaving the room. [`Tree_Update`](../src/bank4/rock-wall-tree-secrets.asm#L249), [data/levelblock-ow.md](../data/levelblock-ow.md)
- Aquamentus ignores fire (invincibility mask `$E2`); Dodongos and big Digdoggers ignore everything. [`Aquamentus_Init`](../src/bank4/aquamentus-digdogger-dodongo-inits.asm#L40), [enemies.md](../enemies.md) "# Dodongo"

## Heart container (item `$1A`), room `$2A`

- It is the boss room's item, hidden until no monster is left (secret trigger 7): it appears at X `$80`, Y `$90` when Aquamentus dies. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- **What it changes:** `HeartValues` + `$11`: one more container and one more full heart (the other hearts are not refilled), up to 16 containers. The white sword cave needs 5 containers and the magic sword cave 12. [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337), [`HeartValues`](../src/Variables.inc#L593), [items.md](../items.md) "# Hearts, Heart Containers and Fairies"

## Triforce piece (item `$1B`), room `$2B`

- Room `$2B` is entered only from `$2A` through the E shutter, which opens when Aquamentus is dead. The piece lies at X `$78`, Y `$90` inside a ring of blocks (layout `$29`) whose only opening is in the south side (row 5, columns 5 and 6): walk down the west side to row 6, east to X `$78`, then up. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152)
- **What it changes:** bit 6 (`$40`) of `InvTriforce` is set, the level ends (mode `$12`: full heal, exit to overworld `$42`), the compass marker stops blinking. The recorder's whirlwind can now carry Link to level 7's landing room `$42`. All eight pieces are needed to pass the old man of level 9. [`Item_TakeClass0PerLevel`](../src/bank1/take-item.asm#L421), [`InvTriforce`](../src/Variables.inc#L597), [`Whirlwind_PrevRoomIds`](../src/bank1/whirlwind.asm#L81), [`Person_InitUnderworldLevel9`](../src/bank1/underworld-person-init.asm#L172)

## Map (item `$17`), room `$18`

- Lies in the open at X `$80`, Y `$90`, the middle of the dark room `$18`, on the route's line from the S door to the N wall (Link at X `$78` is within pickup range). [`Item_TryTake`](../src/bank1/take-item.asm#L206), [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- **What it changes:** the status bar draws the level's map. Rooms `$08` and `$1A` are not on it. Nothing else reads the map. [`Inv_HasMap`](../src/bank5/submenu-map.asm#L153), [items.md](../items.md) "# Map and Compass"

## Compass (item `$16`), room `$5A` (off the route)

- Lies in the open at X `$80`, Y `$90` in the dark room `$5A`, between its two rows of water; reached from `$59` through the E shutter (kill the 6 Goriya of `$59`). [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68)
- **What it changes:** a marker on the status bar map at the Triforce room `$2B`; nothing else. [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51), [`Inv_HasCompass`](../src/bank5/submenu-map.asm#L143)

## Keys (item `$19`): `$3A`, `$0A`, `$78`, `$6D`

- `$3A`: hidden until both Moldorm worms are dead (secret trigger 7); the room is entered through the bombable E wall of `$39`. On the route. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Moldorm_Update`](../src/bank4/moldorm.asm#L59)
- `$0A`: lies in the open at X `$80`, Y `$90`; the W shutter Link came in by reopens only when the 5 blue Goriya are dead. On the route. [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152), [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- `$78` (5 Ropes) and `$6D` (8 Stalfos): both lie in the open and both rooms have only open doors. In `$6D` the key (floor column 6) is west of the water column (column 8), on the side of the door. Off the route. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L73)
- **What it changes:** `InvKeys` + 1; one key opens one key door for good, from both sides; keys carry over to other levels. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`Door_Update`](../src/bank5/update-doors.asm#L60), [items.md](../items.md) "# Keys and Magic Key"

## Bombs and rupees

- Room bombs (item `$00`, +4 each, capped at `InvMaxBombs`), all hidden until the room is empty: `$69`, `$1B`, `$0C` on the route; `$68`, `$7A`, `$6C`, `$0B` off it. [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337), [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- Rupees: the stash of room `$08` (10 single rupees, on the route) and the 5-rupee items of `$09` (on the route, appears when the room is cleared), `$38` and `$58`. [`RupeeStash_Update_Full`](../src/bank1/trap-rupee-stash.asm#L338), [enemies.md](../enemies.md) "# Rupee stash"

# Map

<!-- generated:map -->
```
[08h ] X [09i ] S [0Ak ]   [0Bi ]   [0Ci ] X [0D  ]
  X        S                 K       |/S
[18i ] K [19  ] X [1Ah ] X [1Bi ] K [1C  ]
 S/|
[28  ]   [29  ] X [2ABi]S/-[2BT ]
  K
[38i ]S/-[39  ] X [3Ak ]
          |/S
[48  ]   [49  ]
  K        |
[58i ]---[59  ]S/-[5Ai ]   [5B  ]
  |        X        X       |/S
[68i ] X [69i ]---[6A  ]---[6B  ]---[6Ci ]---[6Dk ]
  |        |        |
[78k ]   [79S ]---[7Ai ]

[73S] = room $73 and its marks: S start room, B boss room, T Triforce room,
        k a key lies here, i another room item lies here, h not shown on the submenu map.
Between two rooms, the door: --- or | open, K key door, S shutter, X bombable wall,
        F false wall (walk-through), nothing = wall. A/B = the two sides differ: the left
        or upper room's side first, . = wall on that side.
Doors on sides with no room of the level: $79 S open.
```

- Cellar `$7B` (tunnel cellar): joins `$29` (left exit) and `$0D` (right exit), whose stairs lead here; item: none. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)

- Cellar `$4A` (treasure cellar): both exits lead to `$1A`, whose stairs lead here; item: red candle ($07). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)
<!-- /generated:map -->

# Rooms

Generated from the level data by `tools/romdata.py`; do not edit between the markers.

<!-- generated:rooms -->
| room | N | S | W | E | monsters | item | secret | block | dark | layout | floor | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $08 off-map | wall | bomb | wall | bomb | 1 x Rupee stash ($35) | - | - | - | - | $26 | blank | - |
| $09 | wall | shutter | bomb | shutter | 3 x Goriya ($05), 3 x Goriya ($06) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $0A | blocks | - |
| $0A | wall | wall | shutter | wall | 5 x Goriya ($05) | key ($19) at $80,$90 | all dead: shutters open | - | - | $24 | statues | - |
| $0B | wall | key | wall | wall | 1 x Moldorm ($41) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $25 | sand | - |
| $0C | wall | open | wall | bomb | 3 x Dodongo ($31) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $23 | statues | - |
| $0D | wall | wall | bomb | wall | 3 x Bubble ($2B), 5 x Wallmaster ($27) | - | push block: stairs appear | B | - | $08 | blocks | stairs lead to cellar $7B |
| $18 | bomb | shutter | wall | key | 2 x Bubble ($2B), 3 x Goriya ($05), 3 x Keese blue ($1B) | map ($17) at $80,$90 | all dead: shutters open | - | D | $16 | water | - |
| $19 | shutter | wall | key | bomb | 6 x Goriya ($05) | - | all dead: shutters open | - | - | $1F | blocks | - |
| $1A off-map | wall | wall | bomb | bomb | 3 x Goriya ($05), 3 x Goriya ($06) | - | - | B | - | $1A | blocks, stairs | stairs lead to cellar $4A; boss sound $10 |
| $1B | key | wall | bomb | key | 6 x Goriya ($05) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $1C | shutter | wall | key | wall | 1 x Digdogger ($38) | - | all dead: shutters open | - | - | $24 | statues | - |
| $28 | open | key | wall | wall | 1 x Grumble ($36) | - | - | - | - | $26 | blank | person $36 says: GRUMBLE,GRUMBLE... |
| $29 | wall | wall | wall | bomb | 3 x Goriya ($05), 3 x Goriya ($06) | - | - | B | - | $1A | blocks, stairs | stairs lead to cellar $7B; boss sound $10 |
| $2A boss | wall | wall | bomb | shutter | 1 x Aquamentus ($3D) | heart container ($1A) at $80,$90 | all dead: shutters open, room item appears | - | - | $04 | blocks | - |
| $2B Triforce | wall | wall | open | wall | none | Triforce piece ($1B) at $80,$90 | - | - | - | $29 | blocks, statues | - |
| $38 | key | wall | wall | shutter | 3 x Goriya ($05), 3 x Goriya ($06) | 5 rupees ($0F) at $D0,$60 | all dead: shutters open, room item appears | - | - | $0D | blocks | - |
| $39 | wall | open | open | bomb | 1 x Digdogger ($38) | - | - | - | - | $24 | statues | - |
| $3A | wall | wall | bomb | wall | 1 x Moldorm ($41) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $25 | sand | boss sound $10 |
| $48 | wall | key | wall | wall | 1 x dungeon person ($4F) | - | - | - | - | $26 | blank | person $4F says: I BET YOU'D LIKE / TO HAVE MORE BOMBS. |
| $49 | shutter | open | wall | wall | 2 x Bubble ($2B), 3 x Goriya ($05), 3 x Keese blue ($1B) | - | all dead: shutters open | - | - | $18 | water | - |
| $58 | key | open | wall | open | 3 x Dodongo ($31) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $59 | open | bomb | open | shutter | 3 x Goriya ($05), 3 x Goriya ($06) | - | all dead: shutters open | - | - | $02 | blocks | - |
| $5A | wall | bomb | open | wall | 8 x Stalfos ($2A) | compass ($16) at $80,$90 | - | - | D | $19 | water | - |
| $5B | wall | open | wall | wall | 1 x dungeon person ($50) | - | - | - | - | $26 | blank | person $50 says: THERE'S A SECRET IN / THE TIP OF THE NOSE. |
| $68 | open | open | wall | bomb | 1 x Trap ($4A), 3 x (empty) ($00), 4 x Keese blue ($1B) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $69 | bomb | open | bomb | open | 5 x Goriya ($05) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $23 | statues | - |
| $6A | bomb | open | open | open | 8 x Keese blue ($1B) | - | ringleader: first monster dead kills all | - | D | $09 | blocks | - |
| $6B | shutter | wall | open | open | 6 x Goriya ($05) | - | all dead: shutters open | - | - | $1E | blocks | - |
| $6C | wall | wall | open | open | 1 x Digdogger ($39) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $24 | statues | - |
| $6D | wall | wall | open | wall | 8 x Stalfos ($2A) | key ($19) at $80,$90 | - | - | - | $13 | water | - |
| $78 | open | wall | wall | wall | 5 x Rope ($28) | key ($19) at $80,$90 | - | - | - | $23 | statues | - |
| $79 start | open | open | wall | open | none | - | - | - | - | $21 | statues, sand | - |
| $7A | open | wall | open | wall | 1 x Moldorm ($41) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $1D | blocks | - |
| $7B cellar | - | - | - | - | 4 x Keese blue ($1B) | - | - | - | - | $3E | tunnel cellar | left exit to $29, right exit to $0D; Link comes out at $60,$9D |
| $4A cellar | - | - | - | - | 4 x Keese blue ($1B) | red candle ($07) at $80,$90 | - | - | - | $3F | treasure cellar | left exit to $1A, right exit to $1A; Link comes out at $60,$9D |
<!-- /generated:rooms -->

# Floor plans

<!-- generated:floors -->
One character per floor square, 12 columns x 7 rows inside the walls (left column = west, top row =
north): `.` floor (tile `$74`), `B` block (tile `$B0`; a push block object is placed on one in a room
with a push block), `W` water (tile `$F4`), `S` statue (tiles `$94` and `$B4`), `>` stairs (tile `$70`),
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L63), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L73), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
Stairs that a secret makes appear are not part of the layout.

```
layout $00  rooms: $1B,$58,$68
............
............
............
............
............
............
............
```

```
layout $02  rooms: $59
............
............
..B......B..
............
..B......B..
............
............
```

```
layout $04  rooms: $2A
.......BBBBB
.........BBB
..........BB
............
..........BB
.........BBB
.......BBBBB
```

```
layout $08  rooms: $0D
............
.BBBBBBBBBB.
..........B.
..........B.
..........B.
.BBBBBBBBBB.
............
```

```
layout $09  rooms: $6A
............
..BBBBBBBB..
..B......B..
..B......B..
..B......B..
..BBBBBBBB..
............
```

```
layout $0A  rooms: $09
............
............
............
....B..B....
............
............
............
```

```
layout $0D  rooms: $38
............
.B.B.BB.B.B.
............
.B.B.BB.B.B.
............
.B.B.BB.B.B.
............
```

```
layout $13  rooms: $6D
........W...
........W...
........W...
........W...
........W...
........W...
........W...
```

```
layout $16  rooms: $18
WWWWW..WWWWW
W...WWWW...W
W.WWW..WWW.W
..W......W..
W.WWW..WWW.W
W...WWWW...W
WWWWW..WWWWW
```

```
layout $18  rooms: $49
............
............
WWWWWWWWWWWW
............
............
............
............
```

```
layout $19  rooms: $5A
............
WWWWWWWWWWWW
............
............
............
WWWWWWWWWWWW
............
```

```
layout $1A  rooms: $1A,$29
............
......B.....
.....B.B....
....B.>.B...
.....B.B....
......B.....
............
```

```
layout $1D  rooms: $7A
............
............
..BB....BB..
..BB....BB..
..BB....BB..
............
............
```

```
layout $1E  rooms: $6B
............
............
.....BB.....
.....BB.....
.....BB.....
............
............
```

```
layout $1F  rooms: $19
............
.BB......BB.
............
.....BB.....
............
.BB......BB.
............
```

```
layout $21  rooms: $79
............
.S..S..S..S.
............
.S..S..S..S.
.....::.....
.S.:S::S:.S.
...::::::...
```

```
layout $23  rooms: $0C,$69,$78
............
............
............
....S..S....
............
............
............
```

```
layout $24  rooms: $0A,$1C,$39,$6C
S..........S
............
............
............
............
............
S..........S
```

```
layout $25  rooms: $0B,$3A
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
```

```
layout $26  rooms: $08,$28,$48,$5B
____________
____________
____________
____________
____________
____________
____________
```

```
layout $29  rooms: $2B
............
.BBBBBBBBBB.
.B..S..S..B.
.B.S....S.B.
.B........B.
.BBBB..BBBB.
............
```

```
layout $3E (tunnel cellar)  rooms: $7B
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
layout $3F (treasure cellar)  rooms: $4A
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

HP in the units of the damage table: wooden sword `$10`, white sword `$20`, magic sword or bomb `$40` per hit.

| Monster (type) | Rooms (count) | HP | Killed or stopped by | Does to Link (no ring) | Section in [enemies.md](../enemies.md) |
| --- | --- | --- | --- | --- | --- |
| Blue Goriya (`$05`) | `$09` (3), `$0A` (5), `$18` (3), `$19` (6), `$1A` (3), `$1B` (6), `$29` (3), `$38` (3), `$49` (3), `$59` (3), `$69` (5), `$6B` (6) | `$50` | any weapon; boomerang stuns | touch 1 heart, boomerang 1 heart | "# Goriya (blue, red)" |
| Red Goriya (`$06`) | `$09`, `$1A`, `$29`, `$38`, `$59` (3 each) | `$30` | any weapon; boomerang stuns | touch half a heart, boomerang 1 heart | "# Goriya (blue, red)" |
| Blue Keese (`$1B`) | `$18` (3), `$49` (3), `$68` (4), `$6A` (8), each cellar (4) | 0 | any hit, the boomerang included | half a heart | "# Keese (blue, red, black)" |
| Bubble (`$2B`) | `$0D` (3), `$18` (2), `$49` (2) | - | nothing | no damage; sword blocked about 160 frames | "# Bubble (flashing, red, blue)" |
| Wallmaster (`$27`) | `$0D` (5) | `$20` | any weapon while it is out of the wall; boomerang stuns | half a heart and carries Link to `$79` | "# Wallmaster" |
| Stalfos (`$2A`) | `$5A` (8), `$6D` (8) | `$20` | any weapon; boomerang stuns | half a heart | "# Stalfos" |
| Rope (`$28`) | `$78` (5) | `$10` | any weapon; boomerang stuns | half a heart | "# Rope" |
| Moldorm (`$41`) | `$3A`, `$0B`, `$7A` (two worms of 5 segments each) | `$20` per segment | any weapon; 10 kills clear the room | half a heart per segment | "# Moldorm" |
| Dodongo (`$31`) | `$0C` (3), `$58` (3) | - | two swallowed bombs, or one sword hit while stunned by a bomb's blast or cloud | 1 heart | "# Dodongo" |
| Digdogger (`$38`, `$39`) | `$1C` (splits into 3), `$39` (3), `$6C` (1) | big: none; small `$80` | recorder, then any weapon on the small ones | 2 hearts | "# Digdogger" |
| Trap (`$4A`) | `$68` (4, one in each corner) | - | nothing | 1 heart | "# Trap" |
| Statues | `$0A`, `$1C`, `$39`, `$6C` (4 each); `$0C`, `$69`, `$78` (2 each) | - | nothing | fireball half a heart | "# Statue fireballs" |
| Aquamentus (`$3D`) | `$2A` (1) | `$60` | sword, beam, rod, arrow, bomb; not boomerang or fire | touch 1 heart, fireball half a heart | "# Aquamentus" |
| Grumble Goriya (`$36`) | `$28` | - | the food | nothing; blocks the way up | "# Grumble Goriya" |
| Old men (`$4F`, `$50`) | `$48`, `$5B` | - | nothing | nothing unless attacked | "# Old men and shopkeepers (cave and dungeon persons)" |
| Rupee stash (`$35`) | `$08` | - | - | nothing | "# Rupee stash" |

- Goriya rooms are the bulk of the level. A Goriya throws only when Link is within `$51` pixels on the longer axis, then stands still until its boomerang returns: that is the moment to hit it. Facing the boomerang with any shield and not swinging parries it. [`Goriya_Update`](../src/bank4/goriya.asm#L45), [`Walker_SetInputDirAndTryThrowBoomerang`](../src/bank4/goriya.asm#L156), [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](../src/bank1/link-collision-harm.asm#L138)
- Rooms the route must clear: `$49` (3 blue Goriya, 3 Keese), `$3A` (Moldorm), `$09` (3 blue and 3 red Goriya), `$0A` (5 blue Goriya), `$1A` (3 blue and 3 red Goriya), `$1C` (Digdogger), `$0D` (5 Wallmasters), `$2A` (Aquamentus). Every other room on the route can be crossed without a kill. [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152), [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74)
- `$49`: Bubbles do not count toward "all dead"; Link crosses the water row with the ladder at any column; the Goriya are walkers bound by the tile test and stay on the side where they appear (unverified), the Keese fly without a tile test. [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124), [`Ladder_CheckAndDraw`](../src/bank5/ladder.asm#L44)
- `$3A`: any segment hit removes that worm's tail; 5 kills per worm, 10 in all, white sword or better one hit each. The key appears when the second head dies. [`Moldorm_Update`](../src/bank4/moldorm.asm#L59), [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- `$1C`: play the recorder once inside the room; 152 frames of freeze, `$40` frames of shrinking, then three small Digdoggers (HP `$80`, 1.25-1.5 pixels per frame at full speed, 2 hearts a touch). Four statues shoot meanwhile. The N shutter opens when the third is dead; the room then stays empty for good. [`Digdogger_HandleFlute`](../src/bank4/digdogger.asm#L91), [`Flute_Wield@FlagFluteUsed`](../src/bank7/flute.asm#L127), [`Statue_Update`](../src/bank4/statues.asm#L86)
- `$39` and `$6C` also hold a big Digdogger, but their doors are open: walk round it; nothing is gained by splitting it. [`Digdogger_InitWith3Children`](../src/bank4/aquamentus-digdogger-dodongo-inits.asm#L65), [`Digdogger_InitWith1Child`](../src/bank4/aquamentus-digdogger-dodongo-inits.asm#L99)
- `$0C` and `$58` (three Dodongos each): the route only crosses `$0C` to its E wall. Killing them is needed only for the room bomb (`$0C`) or the 5 rupees (`$58`); smoke-then-sword costs one bomb per Dodongo and returns a bomb drop. [`Dodongo_CheckBombHit`](../src/bank4/dodongo.asm#L400), [`Dodongo_CheckCollisions`](../src/bank4/dodongo.asm#L296)
- `$0D`: the five Wallmasters stay in the wall, unhittable, until Link stands exactly against a wall (X `$20` or `$D0` with Y `$6D`-`$B4`, or Y `$5D` or `$BD` with X `$29`-`$C7`); one emerges per `$60` frames and walks a fixed path of 7 squares. Stand at the wall, step away, hit the hand as it passes (one hit of the white sword or better). Three Bubbles roam the room and block the sword for about 160 frames on touch. [`Wallmaster_Update`](../src/bank4/wallmaster.asm#L91), [`Wallmaster_State1Trip`](../src/bank4/wallmaster.asm#L231), [`Bubble_Update`](../src/bank4/bubble-keese.asm#L93)
- **Boss: Aquamentus, room `$2A`.** It stands at X `$87`-`$C8`, Y `$80` in the pocket between the blocks on the east side and fires three fireballs every `$70`-`$7F` or `$F0`-`$FF` frames. Only its head box (X+8, Y+8) can be hit; boomerang and fire do nothing. Two bomb blasts, 2 magic, 3 white or 6 wooden sword hits kill it (the beam counts as the sword). [`Aquamentus_Update`](../src/bank4/aquamentus.asm#L54), [`Aquamentus_Shoot`](../src/bank4/aquamentus.asm#L166), [`Sword_DamagePoints`](../src/bank1/sword-bomb-arrow-collision.asm#L182)
- Link enters `$2A` from the W wall with the whole room between him and the boss; the floor is open west of column 7, and the blocks lie only on the east side around the boss's row. [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L73), [`Aquamentus_Init`](../src/bank4/aquamentus-digdogger-dodongo-inits.asm#L40)
- When it dies: the heart container appears at X `$80`, Y `$90`, the E shutter to `$2B` opens, and the room is saved as empty for good. Its own drop is the ordinary random one. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`RoomHistory_SaveKillCountUW`](../src/bank5/room-history-uw.asm#L110), [`Drop_SetUpItem`](../src/bank4/drop-items.asm#L157)
- The boss roar is heard in rooms `$1A`, `$29` and `$3A` while Aquamentus is alive. [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42)

# Route

How it was established: from the level data and the code; not walked in the running game. A search over the
room graph (rooms, door types per side, keys, cellars; cost = room entries, then forced room clears) gave the
path below for a Link who arrives with no keys: 25 room entries after the start room, 2 keys taken and 2
spent, 8 walls bombed. Floor positions are (column, row) of the floor plans; a doorway is at X `$78` (N, S)
or Y `$8D` (W, E).

| Step | Room | Entered from | What to do | Leave by | Keys after | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `$79` | overworld | nothing; no monsters | N (open) | 0 | start, X `$78`, Y `$DD` |
| 1 | `$69` | S | bomb the N wall (bomb within `$18` pixels of X `$78`, Y `$5D`); optional: kill the 5 blue Goriya for the room bomb | N (bombed) | 0 | two statues shoot |
| 2 | `$59` | S | cross; 3 blue and 3 red Goriya | N (open) | 0 | |
| 3 | `$49` | S | cross the water row (row 2) with the ladder; kill 3 blue Goriya and 3 Keese so the N shutter opens | N (shutter) | 0 | ladder; 2 Bubbles |
| 4 | `$39` | S | avoid the big Digdogger; bomb the E wall (bomb near X `$D0`, Y `$8D`) | E (bombed) | 0 | four statues shoot |
| 5 | `$3A` | W | kill both Moldorm worms (10 kills); take the key at X `$80`, Y `$90` | W | 1 | sand floor |
| 6 | `$39` | E | cross to the W door | W (open) | 1 | the Digdogger is back |
| 7 | `$38` | E (shutter shuts behind) | go north along column 11 to row 0, west to X `$78`, unlock the N door; 3 blue and 3 red Goriya need not die | N (key) | 0 | rows 1, 3, 5 are block rows |
| 8 | `$28` | S | wait for the text, put the food down (B); `$40` frames later the Goriya is gone | N (open) | 0 | food used up |
| 9 | `$18` | S (shutter shuts behind) | straight north at X `$78`: ladder over row 5, map at row 3, ladder over row 1; bomb the N wall (X `$78`, Y `$5D`) | N (bombed) | 0 | dark; 2 Bubbles, 3 blue Goriya, 3 Keese |
| 10 | `$08` | S | take the 10 rupees; bomb the E wall | E (bombed) | 0 | not on the map |
| 11 | `$09` | W | kill 3 blue and 3 red Goriya: S and E shutters open, 5 rupees appear | E (shutter) | 0 | |
| 12 | `$0A` | W (shutter shuts behind) | take the key at X `$80`, Y `$90`; kill the 5 blue Goriya so the W shutter opens | W (shutter) | 1 | four statues shoot |
| 13 | `$09` | E | empty now; the S shutter opens by itself | S (shutter) | 1 | |
| 14 | `$19` | N (shutter shuts behind) | bomb the E wall; the 6 blue Goriya need not die | E (bombed) | 1 | |
| 15 | `$1A` | W | kill 3 blue and 3 red Goriya; push the block at (4, 3) down from (4, 2) or up from (4, 4); walk to the stairs at (6, 3) | stairs | 1 | not on the map |
| 16 | `$4A` | stairs, X `$30` | take the red candle at X `$80`, Y `$90`; climb out at the top | top | 1 | cellar; 4 Keese |
| 17 | `$1A` | cellar, at X `$60`, Y `$9D` | room is empty; bomb the E wall | E (bombed) | 1 | |
| 18 | `$1B` | W | unlock the E door; optional: kill the 6 blue Goriya for the room bomb | E (key) | 0 | |
| 19 | `$1C` | W | play the recorder; kill the 3 small Digdoggers so the N shutter opens | N (shutter) | 0 | four statues shoot |
| 20 | `$0C` | S | bomb the E wall; the 3 Dodongos need not die | E (bombed) | 0 | two statues shoot |
| 21 | `$0D` | W | kill the 5 Wallmasters; push the block at (10, 3) west from (11, 3), or east from (9, 3); stairs appear at (11, 0), X `$D0`, Y `$60` | stairs | 0 | 3 Bubbles |
| 22 | `$7B` | stairs, X `$C0` | down the right ladder, west along the passage, up the left ladder (X `$30`), off the top | top, X < `$80` | 0 | tunnel; 4 Keese |
| 23 | `$29` | cellar, at X `$60`, Y `$9D` | bomb the E wall; the 3 blue and 3 red Goriya need not die | E (bombed) | 0 | boss roar |
| 24 | `$2A` | W | kill Aquamentus; take the heart container | E (shutter) | 0 | boss |
| 25 | `$2B` | W | round the block ring to its south opening; take the Triforce piece | level ends | 0 | |

- Key balance: found 4 (`$3A`, `$0A`, `$78`, `$6D`), taken 2 (`$3A`, `$0A`), spent 2 (`$38` N, `$1B` E), 0 left; three key doors stay locked (`$18` E, `$0B` S, `$48` S). [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`InvKeys`](../src/Variables.inc#L584)
- With keys brought (same search): 1 key, skip `$0A` (step 11 leaves `$09` by the S shutter): 23 entries; 2 keys, unlock the E door of `$18` and enter `$19` from the W, skipping `$08`, `$09`, `$0A` and two walls: 21 entries; 3 keys or the magic key, skip `$3A` as well: 19 entries, 5 walls. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`InvMagicKey`](../src/Variables.inc#L571)
- Alternative for the first key, 2 entries longer and without the Moldorm fight: from `$69` bomb the W wall instead of the N wall, `$68`, S to `$78` (key in the open, 5 Ropes), back through `$68`, N to `$58`, E to `$59`. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [enemies.md](../enemies.md) "# Trap"
- Order matters in `$39`: the key of `$3A` must be taken before the W door, because the E shutter of `$38` shuts behind Link and reopens only when the 6 Goriya of `$38` are dead. [`Room_EnterCommon@Method2`](../src/bank5/enter-room.asm#L211), [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68)
- Order matters in `$1A`: the candle (kill all, push, stairs) before the E wall is only a matter of economy; in `$0D` the Wallmasters must all be dead before the block will move. [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124)
- Entering a room through a shutter, key door or bombable wall always works from the open side: the side Link comes in by is marked open in the new room, then a shutter there closes behind him. [`Door_SetEnteringDoorwayOpened`](../src/bank5/mode7-scroll-init.asm#L368), [`Room_EnterCommon@Method2`](../src/bank5/enter-room.asm#L211)
- Bomb walls: the bomb's own position must be less than `$18` pixels on both axes from the wall's hotspot (N `$78`,`$5D`; S `$78`,`$BD`; W `$20`,`$8D`; E `$D0`,`$8D`) when its blast ends; the hole is passable 8 frames later and is saved for both rooms. [`Bomb_WallHotspotsX`](../src/bank7/bomb.asm#L60), [`Bomb_WallHotspotsY`](../src/bank7/bomb.asm#L64), [`Door_Update`](../src/bank5/update-doors.asm#L60)
- After a death, CONTINUE restarts Link in `$79` with 3 hearts. Unlocked doors, bombed walls, taken items (keys, candle, map), the fed Goriya and dead Digdoggers, Moldorms and Aquamentus stay as they are; shutters are shut again. [`Mode8Continue_ChoiceModes`](../src/bank5/mode8-continue-mode10-stairs.asm#L49), [`Mode3Unfurl_Init_Sub1`](../src/bank7/init-mode-dispatch.asm#L294), [mechanics.md](../mechanics.md) "# Hearts, rings and death"
- A Wallmaster (room `$0D` only) also puts Link back in `$79`, with his hearts as they were. Both restarts clear the room history, so cleared Goriya and Keese rooms on the way back are full again and the shutter rooms `$49` and `$09` must be cleared again; `$1C` stays empty and its shutter opens at once. [`Wallmaster_State1Trip`](../src/bank4/wallmaster.asm#L231), [`RoomHistory_Clear`](../src/bank7/init-mode-dispatch.asm#L271), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- The way back from `$79` to `$0D` once everything is open: `$69`, `$59`, `$49`, `$39`, `$38`, `$28`, `$18`, `$08`, `$09`, `$19`, `$1A`, `$1B`, `$1C`, `$0C`, `$0D` (15 entries; the ladder is needed again, the food and the recorder are not). [`Door_AddRoomFlagsToOpenedDoors`](../src/bank5/door-flags-dark-room.asm#L84), [`Grumble_Init_Full`](../src/bank1/underworld-person-init.asm#L216)

# Optional detours

- Bomb capacity, room `$48`: from `$59` go W to `$58` (3 Dodongos, open doors), unlock its N door (1 key). The old man says "I BET YOU'D LIKE TO HAVE MORE BOMBS." and shows "-100": stand at X `$78`, Y `$98` (within 5 pixels) with 100 rupees or more. `InvMaxBombs` + 4, bombs refilled to the new maximum, the man is gone for good. Adds 4 room entries (`$58`, `$48`, `$58`, `$59`), costs 1 key and 100 rupees; worth it for the eight walls and the Dodongos. [`Person_ComplexState2SellBombCapacity`](../src/bank1/underworld-person-update.asm#L171), [`InvMaxBombs`](../src/Variables.inc#L607)
- Compass, room `$5A`: kill the 6 Goriya of `$59` so its E shutter opens; the compass lies in the open among 8 Stalfos in the dark. 2 entries, no key. [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152), [items.md](../items.md) "# Map and Compass"
- Key of `$78` (5 Ropes): from `$69` through the bombable W wall or from `$58` S, then S of `$68`. 4 entries when it replaces the N wall of `$69` (see Route), 1 bomb, no key spent. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [enemies.md](../enemies.md) "# Rope"
- Key of `$6D` (8 Stalfos): `$69` E to `$6A` (dark, 8 Keese), `$6B` (6 blue Goriya), `$6C` (big Digdogger, statues), `$6D`, all through open doors, and back: 8 entries, no forced fight. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [enemies.md](../enemies.md) "# Stalfos"
- Room bombs (+4 each, when the room is empty): `$68` (4 Keese; the traps do not count), `$7A` (Moldorm, E of the start room), `$6C` (Digdogger, needs the recorder), `$0B` (Moldorm, behind the key door N of `$1B`: a key for 4 bombs). [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124), [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- Rupees: `$38` (5 rupees at X `$D0`, Y `$60` when its 6 Goriya are dead; the route crosses the room) and `$58` (5 rupees when its 3 Dodongos are dead). [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [items.md](../items.md) "# Rupees"
- Hint, room `$5B`: kill the 6 Goriya of `$6B` so its N shutter opens; the old man says "THERE'S A SECRET IN THE TIP OF THE NOSE." (it fits the push block room `$0D` at the north-east tip of the map; unverified). 2 entries from `$6B`. [`Person_TextSelectorsLevels1257`](../src/bank1/underworld-person-init.asm#L67), [data/person-text.md](../data/person-text.md)
- Ringleader room `$6A` (dark, 8 Keese): killing the Keese in object slot `$01` kills the rest; nothing is gained but drops. [`Secret_Trigger2Ringleader`](../src/bank5/underworld-secrets.asm#L176)
- Shortcut out: the tunnel `$7B` works both ways (`$29` to `$0D`), but `$29` has no other way in than the boss room's W wall, so it shortens nothing. [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43)

# Hazards and rules that matter here

- Shutters (`$49` N, `$09` S and E, `$0A` W, `$19` N, `$1C` N, `$2A` E, `$38` E, `$18` S, `$59` E, `$6B` N): they open only while the room is empty of counting monsters, one per 8 frames, and are never saved. Bubbles, traps and persons do not count. [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68), [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124), [mechanics.md](../mechanics.md) "# Doorways, doors, keys, shutters and bomb walls"
- Seven doorways are a shutter on one side and open on the other (`$1C`/`$0C`, `$18`/`$28`, `$2A`/`$2B`, `$38`/`$39`, `$49`/`$39`, `$59`/`$5A`, `$6B`/`$5B`): free to enter from the open side, shut behind Link. Going back needs the room cleared. [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55), [`Room_EnterCommon@Method2`](../src/bank5/enter-room.asm#L211)
- Ten bombable walls, drawn as plain wall until opened; the two unmapped rooms `$08` and `$1A` and the boss room are behind them. No false walls. [`Door_LayOut`](../src/bank5/lay-out-doors.asm#L131), [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136)
- A bomb wall cannot be opened while another door is opening or closing (one door command at a time); after entering a room, Link's entry door command is finished before a bomb's 72-frame fuse and blast end. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`TriggeredDoorCmd`](../src/Variables.inc#L85)
- Push blocks (`$0D`, `$1A`, `$29`): only with the room empty; Link exactly in line, at most `$10` pixels away, holding only the direction toward the block for `$10` frames; one move per room entry. [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [mechanics.md](../mechanics.md) "# Push blocks and room secrets"
- `$0D` (trigger 5): the stairs always appear at X `$D0`, Y `$60`, the north-east corner square, whichever way the block went. From (11, 3) it is three squares straight up. [`Secret_Trigger5BlockStairs`](../src/bank5/underworld-secrets.asm#L226)
- `$1A` and `$29` (trigger 0, same layout): the push only moves the block; in `$29` nothing requires it, since Link arrives from the tunnel outside the diamond at X `$60`, Y `$9D` and leaves by the E wall. [`Secret_RunTrigger_JumpTable`](../src/bank5/underworld-secrets.asm#L105), [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43)
- Stairs take Link only when he stands exactly on the square (X a multiple of `$10`, Y ending in `$D`, grid offset 0); after coming out of a cellar he must complete one grid step before stairs take him again. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`UndergroundExitType`](../src/Variables.inc#L99)
- Tunnel `$7B`: coming from `$0D` Link starts at X `$C0`; he leaves at the top with Up held, to `$29` if his X is below `$80`, else back to `$0D`. Bombs open nothing in a cellar. [`Mode9Cellar_LinkStartXs`](../src/bank5/cave-cellar-enter.asm#L262), [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43)
- Water (`$18`, `$49`, `$5A`, `$6D`): the ladder is laid when Link walks into a water tile from a grid point, outside a doorway, with a free monster slot; it spans one square and is put away when he steps off. Whether walking monsters are stopped by water was not read in their code (unverified). [`Link_EndMoveAndAnimate@SetUpLadder`](../src/bank7/link-animate-ladder.asm#L246), [`Ladder_CheckAndDraw@StashLadder`](../src/bank5/ladder.asm#L110), [mechanics.md](../mechanics.md) "# Ladder and raft"
- No ladder without a free slot in `$01`-`$0B`: in a room with many monsters and shots the ladder can fail to appear. [`Obj_FindEmptyMonsterSlot`](../src/bank7/object-stubs-destroy-monster.asm#L169)
- Dark rooms (`$18`, `$5A`, `$6A`): black picture only; a standing candle fire lights the room until it is left. [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124), [`Candle_UpdateRoomLight`](../src/bank4/shoot-helpers-candle.asm#L132)
- Grumble Goriya (`$28`): Link is halted while "GRUMBLE,GRUMBLE..." is typed; after that the food only has to be out (object slot `$0F`), anywhere in the room. It cannot be killed; hitting it turns on fireballs from beside the two fires until the room is left. [`Grumble_State1WaitForFood`](../src/bank1/grumble-goriya.asm#L73), [`Person_CheckCollisions`](../src/bank1/underworld-person-update.asm#L89), [enemies.md](../enemies.md) "# Grumble Goriya"
- The boomerang shares slot `$0F` with the food: throwing it removes food on the ground, and food cannot be placed while the boomerang is out. [`Food_Wield`](../src/bank5/wield-items.asm#L247), [`Boomerang_Wield`](../src/bank5/wield-items.asm#L124)
- Old men (`$48`, `$5B`): Link is halted while the text is typed and cannot move up past Y `$8D`; attacking one starts the same fireballs. [`Person_BlockLinkMovingUp`](../src/bank1/modeb-cave-save-sprites.asm#L289), [`Person_CheckCollisions`](../src/bank1/underworld-person-update.asm#L89)
- Statues shoot in layouts `$24` (four sources: `$0A`, `$1C`, `$39`, `$6C`) and `$23` (two: `$0C`, `$69`, `$78`); not in `$79` or `$2B`. No shot comes from a statue within `$18` pixels of Link. [`Statue_RoomLayoutIds`](../src/bank4/statues.asm#L45), [`Statue_Update@LoopAxis`](../src/bank4/statues.asm#L194)
- Traps (`$68`): one in each corner; a trap charges at 1.75 pixels per frame along the wall toward the middle when Link is within `$0E` pixels of its row or column, so the two traps of a side meet in front of that side's door. Keep walking through the door lanes; never stop there. [`Trap_Update_Full`](../src/bank1/trap-rupee-stash.asm#L189), [`Trap_AllowedDirs`](../src/bank1/trap-rupee-stash.asm#L167)
- Bubbles (`$0D`, `$18`, `$49`): a touch does no damage but blocks the A button for 151 to 160 frames; the B item still works, so bombs, the boomerang and the candle remain. [`Bubble_Update`](../src/bank4/bubble-keese.asm#L93), [`SwordBlockedLongTimer`](../src/Variables.inc#L62), [`Link_HandleInput`](../src/bank5/link-handle-input.asm#L68)
- Wallmasters (`$0D`): a capture needs a touch that harms Link, so none happens while his invincibility timer runs; the hand cannot be hit while it carries him. [`Wallmaster_State1Trip`](../src/bank4/wallmaster.asm#L231), [`Link_CheckMonsterTouch`](../src/bank1/link-collision-harm.asm#L48)
- The recorder freezes play for 152 frames on every use, in any room. [`Flute_Wield`](../src/bank7/flute.asm#L49)
- No code tests a room id of this level by number; everything above comes from the level data. [`Level_BlockUW7to9Q1`](../src/bank6/level-blocks-common-data.asm#L74), [mechanics.md](../mechanics.md) "# Special-cased rooms"

# Second quest

- In Q2 the level blocks are different data: the dungeon shown as LEVEL-7 is internal level 8 (start room `$79`, Triforce room `$1B`), and internal level 7 is shown as LEVEL-8; both are in the Q2 block for levels 7-9. [`Level_BlockUW7to9Q2`](../src/bank6/level-blocks-common-data.asm#L82), [data/levelblock-uw-7to9-q2.md](../data/levelblock-uw-7to9-q2.md), [data/levelinfo.md](../data/levelinfo.md)
- In Q2 the recorder does not drain the pond of overworld room `$42` (the room calls the whirlwind instead), and the rock wall of room `$19` becomes a working secret. This document covers Q1 only. [`Flute_SecretRoomIdsOW`](../src/bank7/flute.asm#L36), [`Secret_IsForOtherQuest`](../src/bank4/rock-wall-tree-secrets.asm#L331)
