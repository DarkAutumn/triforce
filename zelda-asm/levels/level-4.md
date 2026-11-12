# Level 4: the ladder and a two-headed Gleeok

Level 4 lies under overworld room `$45`, an island that only the raft reaches (from the dock of room `$55`); the entrance itself is an open cave mouth. It holds the ladder (in cellar `$60`, under room `$32`), a heart container and a Triforce piece; the boss is a Gleeok with two heads in room `$13`, which only a sword (or the rod's swing) can hurt. Link must bring the raft and a sword; two bombs save a detour through a second Manhandla, and the level's own keys pay for its key doors. Seven of its twenty rooms are dark, which changes the picture and nothing else. Six rooms have water, and that is what the level is about: from the moment Link holds the ladder he walks across one square of water, and the way from the ladder to the boss needs that at once, in room `$30`.

Contents (generated; `L` = line in this file):

<!-- toc -->
- L21 At a glance
- L60 What Link must bring
- L81 What Link gets here
- L127 Map
- L158 Rooms
- L188 Floor plans
- L406 Monsters and boss
- L443 Route
- L482 Optional detours
- L493 Hazards and rules that matter here
- L517 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 4 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-4 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | $45 | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178), [`Flute_SecretRoomIdsOW`](../src/bank7/flute.asm#L36) |
| Start room | $71 | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Triforce room | $03 (room item: Triforce piece ($1B)); the field places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $13 (1 x Gleeok ($43)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | $60 | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 20: 19 on the submenu map, 1 the map does not show ($11) | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (room and side that has it) | 10 ($00 E, $01 W, $20 S, $30 N, $21 S, $31 N, $30 E, $31 W, $61 E, $62 W) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (room and side that has it) | 8 ($01 S, $11 N, $10 E, $11 W, $11 S, $21 N, $11 E, $12 W) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (room and side that has it) | 6 ($10 N, $13 N, $10 S, $12 E, $13 W, $31 E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (room and side that has it) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $00 S open / $10 shutter, $03 S open / $13 shutter, $10 S shutter / $20 open, $31 E shutter / $32 open | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $71 S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 2 ($12, $32) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 7 ($01, $02, $21, $30, $40, $50, $62) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 4 ($01, $40, $51, $70) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | ladder ($0D): $60 (cellar); compass ($16): $62; map ($17): $21; heart container ($1A): $13; Triforce piece ($1B): $03 | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md) (rows with `lvl` = `4` or `c4`).
<!-- /generated:facts -->

- **Coordinates used below.** A floor square (column c, row r of the floor plans) covers X = `$20` + `$10` x c, Y = `$60` + `$10` x r; Link stands on it at X = `$20` + `$10` x c, Y = `$5D` + `$10` x r. The door spots are N (`$78`, `$5D`), S (`$78`, `$BD`), W (`$20`, `$8D`), E (`$D0`, `$8D`): a door is entered only with X = `$78` (N, S) or Y = `$8D` (W, E) exact. [`Doorway_RequiredCoords`](../src/bank5/doorway.asm#L93)
- **Entrance.** Overworld room `$45` has cave index 4, which loads level 4. The entrance is the opening tile (`$24`) in the square at column 8, row 4 of the room's layout (OW `$41` in [data/room-layouts.md](../data/room-layouts.md)); the level loads when Link stands at X = `$80`, Y = `$7D`. Nothing has to be opened: the room has no bomb, candle or recorder secret, and one Peahat. [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178), [data/levelblock-ow.md](../data/levelblock-ow.md)
- **The island.** Layout OW `$41` is ringed by rock, with water in columns 0 and 15; its only gap is the pier in column 8 of the two bottom rows, and the room below it, `$55`, has water along its whole top except the dock square (column 8, row 4). So Link gets in only by raft: in room `$55`, holding the raft, he stands at X = `$80`, Y = `$7D` and the ride starts by itself; one pixel off in X it does not. [`Dock_Update`](../src/bank4/dock-raft.asm#L39), [items.md](../items.md) `# Raft`
- (trace) Room `$55` entered from the S screen edge at X = `$B0`: up to Y = `$8D`, left to X = `$80`, up to Y = `$7D`. Link is halted there and stands in room `$45` at (`$80`, `$DD`) 149 frames later; 86 more frames walking up bring him to (`$80`, `$7D`), and 183 frames after that he can move in room `$71`. Without the raft he walks on to Y = `$75` and nothing happens. [`Dock_Update@HandleOtherStates`](../src/bank4/dock-raft.asm#L94), [`InvRaft`](../src/Variables.inc#L567)
- **Second way in.** Once Link holds this level's Triforce piece, the recorder's whirlwind can set him down in room `$45` at X = `$80`, Y = `$8D`; before that the whirlwind skips level 4. [`Whirlwind_PrevRoomIds`](../src/bank1/whirlwind.asm#L81), [`Whirlwind_DropOffYs`](../src/bank1/whirlwind.asm#L348), [`Whirlwind_Summon@TestLevelPiece`](../src/bank1/whirlwind.asm#L258)
- **On entry** Link is in room `$71` at X = `$78`, Y = `$DD` (the level's start Y), facing up inside the S doorway; he is first movable at Y = `$CD` (trace). The room has no monsters, and its statues do not shoot. [`Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink`](../src/bank5/mode3-mode6-init.asm#L129), [`LevelInfo_StartY`](../src/Variables.inc#L717), [`Statue_RoomLayoutIds`](../src/bank4/statues.asm#L45)
- **Way out.** The S door of room `$71` leads off the map, which loads the overworld: Link stands in room `$45` at (`$80`, `$7D`) facing down (trace: 185 frames of holding Down from the entry spot). From there he walks down the pier at X = `$80` and off the bottom edge; in room `$55` he arrives at Y = `$3D` with X = the dock's X, so the raft carries him down to Y = `$7F` (trace: 176 frames to room `$55`, about 70 more for the ride). [`Room_CalcNextRoomForDoor`](../src/bank5/next-room-calc.asm#L76), [`Mode_LeaveLevelToOW`](../src/bank5/next-room-calc.asm#L107), [`Dock_Update@State1`](../src/bank4/dock-raft.asm#L117)
- **After the Triforce piece** the end-of-level sequence refills the hearts and puts Link on the overworld at the same spot, room `$45` (`$80`, `$7D`), facing down (trace: 644 frames after he touches the piece, with one heart to refill). [`Mode12EndLevel_Update_Sub4LeaveLevel`](../src/bank5/mode12-lay-out-room.asm#L156), [`Mode12EndLevel_StartFillingHearts`](../src/bank5/mode12-lay-out-room.asm#L118)
- **What stays when Link leaves and comes back.** Saved in the room flags: every unlocked key door and bombed wall (on both of its sides), every item taken, and the death of Gleeok, of the Manhandla of room `$10` and of a touched rupee stash. The key counter is one for all levels. [`RoomFlags_SetDoorOpened`](../src/bank5/shutters-door-flags.asm#L41), [`RoomFlags_SetItemTaken`](../src/bank1/take-item.asm#L140), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50), [`InvKeys`](../src/Variables.inc#L584)
- **What resets.** Loading the level zeroes the banked kills and the room history, so every other room is full of monsters again; shutters are closed again on every visit of a room; a push block is back in its place and the stairs of room `$32` are gone on every entry of the room, not only after leaving the level. [`Mode2Load_Init`](../src/bank7/init-mode-dispatch.asm#L147), [`LevelKillCounts`](../src/Variables.inc#L480), [`Door_Update`](../src/bank5/update-doors.asm#L60), [`PushBlock_State2Done`](../src/bank4/push-block.asm#L191)

# What Link must bring

Required: the raft and a sword. Useful: two bombs. Everything else the level asks for lies inside it.

- **Raft: required**, to reach the entrance (room `$55` to room `$45`, see "At a glance"). Inside the level it does nothing. [`Dock_Update`](../src/bank4/dock-raft.asm#L39), [`InvRaft`](../src/Variables.inc#L567)
- **A sword: required**, for Gleeok in room `$13`: its invincibility mask leaves only the sword, the sword beam and the rod's swing; bombs, arrows, fire, the boomerang and the magic shot do nothing. Sword hits on its heads for the kill: 16 wooden, 8 white, 5 magic. [`Gleeok_Init`](../src/bank4/gleeok-manhandla-gohma-inits.asm#L59), [enemies.md](../enemies.md) `# Gleeok`
- **Keys: none.** The route opens two key doors, both in room `$30` (E, then N), and passes two keys lying in the open, in rooms `$51` and `$40`, before the first of them. A Link who leaves those keys lying needs two of his own. The whole level has 4 keys and 5 key doors. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`InvKeys`](../src/Variables.inc#L584)
- **Bombs: two for the route**, one for the N wall of room `$21` and one for the E wall of room `$11`; a bomb opens a wall only at the end of its blast and only next to the wall's middle. With no bombs the boss is still reachable through rooms `$10`, `$00`, `$01`, `$02` (see "Route"), at the price of a Manhandla, six traps and one more key. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Door_TouchBombable`](../src/bank5/touch-door.asm#L108)
- Bombs are also the cheap way to kill a Vire outright (blast damage `$40` = its HP), so that it does not split, and to kill the hands of Manhandla. [`Bomb_CheckBombOrFireVsMonster`](../src/bank1/sword-bomb-arrow-collision.asm#L126), [enemies.md](../enemies.md) `# Vire`
- **Candle: not needed.** A dark room is a room with bit 7 of its attribute byte E set. The only code that reads the bit is the palette fade when a room is entered or left, a redraw of the play area after the scroll, and the candle's own lighting routine; monsters, doors, tiles, items and collisions are the same as in a lit room. Dark rooms on the route: `$50`, `$40`, `$30` (twice), `$21`. [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124), [`Mode7Scroll_Init_Sub5`](../src/bank5/mode7-scroll-init.asm#L256), [`Mode7Scroll_Update_Sub6`](../src/bank5/mode7-scroll-update.asm#L227), [`Mode4Enter_Init`](../src/bank5/enter-room.asm#L65)
- With a candle, a fire that has come to rest lights the room until Link leaves it; while the picture fades in (4 steps, 10 frames apart) nothing else in the room is updated. The blue candle makes one fire per room entry. [`Candle_State0Begin`](../src/bank4/shoot-helpers-candle.asm#L147), [`Fire_Update@StandingFire`](../src/bank7/fire.asm#L75), [items.md](../items.md) `# Candle (blue, red)`
- **Ladder: not brought**, it is this level's item; everything north of room `$30` needs it (see "What Link gets here"). [`InvLadder`](../src/Variables.inc#L570)
- **Recorder: not needed.** No Digdogger lives here; in a dungeon the recorder only plays its tune and freezes the room for `$98` frames. [`Flute_Wield`](../src/bank7/flute.asm#L49)
- **Food: not needed** (no hungry Goriya). It can be used as a lure: in a room whose first monster is a Vire (`$61`, `$50`, `$31`, `$20`, `$12`, `$62`) or a blue Keese (`$51`, `$70`, `$01`) food on the floor becomes the monsters' chase target; not in room `$30`, whose first object is a Bubble, nor in the Zol, Gel and Like-Like rooms. [`Boomerang_UpdateOrFood@AttractMonsters`](../src/bank7/boomerang-arrow-food.asm#L107), [items.md](../items.md) `# Food (bait, meat)`
- **Bow and arrows: not needed**, and useless against the boss. Each shot costs one rupee; a silver arrow (`$40`) kills a Vire outright, a wooden one (`$20`) splits it. [`Arrow_Wield`](../src/bank5/wield-items.asm#L187), [items.md](../items.md) `# Arrows and Bow (wooden, silver)`
- **Boomerang: not needed, helpful.** It kills Keese and Gels (HP 0) and stuns Vires, Zols and Like-Likes for about 160 frames without splitting them. [`Collision_CheckWeaponPointVsMonster`](../src/bank1/weapon-damage-death.asm#L82)
- **Hearts: no test.** Nothing in the level reads the heart count. What costs hearts: Vire, Zol, Like-Like, trap, Manhandla and Gleeok (neck base, flying head, fireball) take 1 heart per touch, a Gleeok head on its neck 3/8, Keese and Gels half a heart; a ring halves or quarters all of it. [`Damage_ObjTypeDamageToLink`](../src/bank1/monster-collisions.asm#L200), [mechanics.md](../mechanics.md) `# Hearts, rings and death`
- Refills are scarce: no fairy lives here, Keese, Gels and Like-Likes never drop anything, and a Vire or Zol that is split instead of killed drops nothing either. With a wooden or white sword the only monsters that roll for a drop are Zols killed outright (white sword), Manhandla and Gleeok. [`Drop_NoDropTypes`](../src/bank4/drop-items.asm#L96), [`Vire_Update@SplitUp`](../src/bank4/vire.asm#L61), [enemies.md](../enemies.md) `# Drop tables`
- **Magic shield: at risk.** The two Like-Likes of room `$32` (on the route) take it for good once one of them has held Link for 95 frames. [`LikeLike_Update@IncCaptureTime`](../src/bank4/like-like.asm#L111), [`InvMagicShield`](../src/Variables.inc#L604)
- **Rupees: none.** The level's one person only talks; nothing is sold or charged. [`Person_UpdateUnderworld_Full`](../src/bank1/underworld-person-update.asm#L44)

# What Link gets here

## Ladder (cellar `$60`, under room `$32`)

- **Where.** The ladder (item `$0D`) is the room item of cellar `$60`, at (`$80`, `$90`) on the cellar's upper walkway. The cellar's only entrance is a flight of stairs in room `$32`, and those stairs do not exist until the room's push block has been pushed (secret trigger 5). [`Secret_Trigger5BlockStairs`](../src/bank5/underworld-secrets.asm#L226), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
- **Clear the room first.** The block cannot move while a monster is left: the 2 Zols and 2 Like-Likes must die (and the Gels of a Zol that was split). The 2 Bubbles do not count and cannot be killed. [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124)
- **Push the block.** It is the left one of the two blocks, at (`$60`, `$90`) (column 4, row 3), on the line of the W door. Link stands at (`$50`, `$8D`) and holds Right alone for `$10` frames; the block then slides one square. Any of the four sides works (same X, or Link's Y + 3 = `$90`, at most `$10` pixels away, holding only the direction toward the block). [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`PushBlock_State1Moving`](../src/bank4/push-block.asm#L165)
- **The stairs** appear when the slide ends, always at (`$D0`, `$60`): column 11, row 0, the top right corner of the floor (trace: 47 frames after the first frame of the push). Link takes them by standing at (`$D0`, `$5D`). [`Secret_Trigger5BlockStairs`](../src/bank5/underworld-secrets.asm#L226), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
- **In the cellar** Link starts at (`$30`, `$5D`) on top of the left ladder; four blue Keese fly there whatever the room data says. (trace) Down to Y = `$BD` (72 frames), right to X = `$B0` (82), up the right ladder to Y = `$8D` (35), left along the walkway: the ladder is taken when Link reaches X = `$88` (30 frames) and he is halted `$80` frames while he holds it up. Back the same way: right to X = `$B0`, down to Y = `$BD`, left to X = `$30`, up and out of the top. [`Mode9Cellar_Init_Sub8_EnterCellar`](../src/bank5/cave-cellar-enter.asm#L276), [`Spawn_AssignPositions`](../src/bank5/spawn-positions.asm#L64), [`Item_Take`](../src/bank1/take-item.asm#L259), [`ItemLiftTimer`](../src/Variables.inc#L416)
- **Coming out.** Both exits of the cellar lead to room `$32`, where Link appears at (`$C0`, `$BD`) facing down. The room has been laid out again: the block is back at (`$60`, `$90`) and the stairs are gone (trace); to go down again the block must be pushed again. [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43), [`PushBlock_State2Done`](../src/bank4/push-block.asm#L191)

**What it changes.** Before the ladder a water tile is a wall. With it, Link lays the ladder by walking into water and crosses **one square**; no button, no cost, nothing to select. With the grid-point skip (one frame of a perpendicular direction at the right moment) one press crosses two or three squares: room `$21` below.

- **The rule.** After each move, when Link stands on a grid point in a dungeon room (any room; on the overworld only six rooms), is not in a doorway and not halted, has no ladder out, and the tile ahead of him in his facing direction is water (`$F4`), the ladder is laid one square ahead, provided a direction is held that equals his facing direction and one of the object slots `$01`-`$0B` is free. [`Link_EndMoveAndAnimate@CheckLadder`](../src/bank7/link-animate-ladder.asm#L209), [`Link_EndMoveAndAnimate@SetUpLadder`](../src/bank7/link-animate-ladder.asm#L246), [`Obj_FindEmptyMonsterSlot`](../src/bank7/object-stubs-destroy-monster.asm#L169)
- **One square, not two.** On the ladder Link may go on only if the tile behind the water square is walkable, or go back the way he came; the ladder is put away when he is `$10` pixels past it and can be laid again at the next water tile, so several single squares in a row can be crossed but a strip two squares wide cannot, unless Link skips the ladder's one tile test with a perpendicular press (the ladder clip: [mechanics.md](../mechanics.md) `# Grid-point skip: block clip, ladder clip and their run-up`). [`Ladder_CheckAndDraw@HandleInput`](../src/bank5/ladder.asm#L125), [`Ladder_CheckAndDraw@StashLadder`](../src/bank5/ladder.asm#L110), [items.md](../items.md) `# Ladder`
- **First use, room `$31` (the very next room; a shortcut, not required).** On the line of its W and E doors (row 3, Y = `$8D`) four single water squares lie between the doors, at X = `$B0`, `$90`, `$60` and `$40`. With the ladder Link holds Left from the E door and walks straight to the W door, laying the ladder four times (trace: 165 frames to the W screen edge); without it he stops at X = `$C0` and must take the winding land path (353 frames, see "Route"). [`Ladder_CheckAndDraw`](../src/bank5/ladder.asm#L44), [`LadderSlot`](../src/Variables.inc#L104)
- **First required use, room `$30` (the third room entered after the cellar).** Row 2 of its floor (tiles Y `$80`-`$8F`) is water from wall to wall. The S and E doors are south of it and the N door north of it; the way to the boss is the N door. Link stands at (`$78`, `$8D`) and holds Up: the ladder appears at (`$78`, `$80`) when he reaches Y = `$85`, he is across at Y = `$6D` 18 frames later and at the N door spot (`$78`, `$5D`) 37 frames after starting (trace). Without the ladder he stops at Y = `$85` and stays there (trace, same room and input with `InvLadder` = 0). [`Link_EndMoveAndAnimate@SetUpLadder`](../src/bank7/link-animate-ladder.asm#L246), [`InvLadder`](../src/Variables.inc#L570)
- **Without it the level cannot be finished.** The rooms north of row 3 (`$20`, `$21`, `$10`, `$11`, `$12`, `$13`, `$00`-`$03`) join the rest only through the N door of room `$30` (behind the water row) and the N door of room `$31`, which leads onto an island in room `$21` (next bullet). [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55), [data/levelblock-uw-1to6-q1.md](../data/levelblock-uw-1to6-q1.md)
- **Where it is not enough: room `$21`.** Its S door (the key door from room `$31`) opens onto an island (rows 3-6 in the middle) with water two or more squares wide on every side. Link, ladder in hand, gets one square out and stops: at Y = `$7D` going up from (`$78`, `$8D`), at X = `$40` going left, at X = `$B0` going right (trace). The map, the N wall and the W door of the room are on the outer ring, reached from the W door. With the ladder clip he does get off the island: from (`$78`, `$8D`) Up 8 frames, Left (or Right) 1 frame, Up 34 frames take him over both squares to (`$78`, `$55`) on the line of the N wall (trace; see "Route", "Shortcut with the ladder clip"). [`Ladder_CheckAndDraw@HandleInput`](../src/bank5/ladder.asm#L125)
- **Room `$01` (off the route)** is all single squares: without the ladder none of its three doors is joined to another or to its key; with it Link holds Right from the W door and comes out at the E door, crossing at X = `$40` and `$B0` (trace: he stops at X = `$30` without it). [`Link_EndMoveAndAnimate@CheckLadder`](../src/bank7/link-animate-ladder.asm#L209)
- **Later in the game.** On the overworld the ladder works only in rooms `$17`, `$18`, `$19`, `$27`, `$4F` and `$5F`; room `$5F` holds a heart container at (`$C0`, `$90`). In dungeons it works in every room: the rooms with water are `$23`, `$33` (level 1), `$2B` (level 3), and those of levels 5 to 9 listed in the `floor` column of their rooms tables ([level-5.md](level-5.md), [level-6.md](level-6.md), [level-7.md](level-7.md), [level-8.md](level-8.md), [level-9.md](level-9.md)). [`Ladder_RoomIdsOW`](../src/bank7/link-animate-ladder.asm#L76), [mechanics.md](../mechanics.md) `# Ladder and raft`

## Heart container (room `$13`)

- It is the boss room's item and is hidden until no monster is left there (secret trigger 7): when Gleeok dies it appears at (`$D0`, `$C0`), the bottom right corner of the floor, and Link takes it at (`$D0`, `$BD`). [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- **What it changes.** `HeartValues` + `$11`: one more container and one more full heart, nothing refilled (trace: `$65` to `$76`). The container count is what the white sword's cave (5 containers) and the magic sword's cave (12) test, and a larger total means more hearts must be full before the sword beam fires. [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337), [`Cave_State2TalkOrShopOrDoorCharge@TryToTake`](../src/bank1/cave-shop.asm#L161), [`SwordShot_Make`](../src/bank7/sword-and-rod.asm#L327), [items.md](../items.md) `# Hearts, Heart Containers and Fairies`

## Triforce piece (room `$03`)

- It lies at (`$78`, `$90`) inside a ring of blocks that is open at the bottom (columns 5 and 6); from the S door Link walks straight up at X = `$78` and takes it at Y = `$95` (trace: 43 frames from the entry spot). The room has no monsters and its statues do not shoot. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`Statue_RoomLayoutIds`](../src/bank4/statues.asm#L45)
- **What it changes.** Bit 3 (`$08`) of `InvTriforce` is set and the level ends: full hearts, then the overworld at room `$45`. From then on the recorder's whirlwind can carry Link to room `$45`, and the compass marker of this level stops blinking; all eight pieces are what the old man at the start of level 9 tests. [`Item_TakeClass0PerLevel`](../src/bank1/take-item.asm#L421), [`InvTriforce`](../src/Variables.inc#L597), [`Person_InitUnderworldLevel9`](../src/bank1/underworld-person-init.asm#L172), [items.md](../items.md) `# Triforce pieces and the Triforce of Power`

## Keys (four)

- Room `$51` (route): at (`$90`, `$90`), in the open from the start; Link takes it at (`$90`, `$8D`), two squares right of the line between the S and W doors. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- Room `$40` (route): at (`$80`, `$80`), inside the water ring. The ring is open at the top (row 1, columns 4-7), so no ladder is needed: from the S door right along the bottom row to X = `$D0`, up to Y = `$5D`, left to X = `$80`, down to Y = `$7D` (trace). [`Item_TryTake`](../src/bank1/take-item.asm#L206)
- Room `$70` (one room W of the start): hidden until all 8 blue Keese are dead (secret trigger 7), then at (`$90`, `$90`). [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- Room `$01`: at (`$80`, `$80`) on the middle island; needs the ladder (above). [`Link_EndMoveAndAnimate@CheckLadder`](../src/bank7/link-animate-ladder.asm#L209)
- **What a key changes.** Pushing into a key door spends one key and opens the door for good, from both sides; keys not spent here go on to the later levels. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [items.md](../items.md) `# Keys and Magic Key`

## Map (room `$21`), compass (room `$62`), rupees (room `$11`)

- Map: at (`$D0`, `$C0`), the bottom right corner of room `$21`, on the outer ring (from the W door: up the left column, along the top row, down the right column). It makes the status bar draw this level's map and changes nothing else. [`Mode3Unfurl_Init_Sub6_CueStatusBarMap`](../src/bank5/mode3-mode6-init.asm#L102), [`Inv_HasMap`](../src/bank5/submenu-map.asm#L153), [items.md](../items.md) `# Map and Compass`
- Compass: at (`$80`, `$80`) in room `$62`, behind the E key door of room `$61`. It puts a marker on the status bar map at the Triforce room `$03` and changes nothing else. [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51), [`Inv_HasCompass`](../src/bank5/submenu-map.asm#L143)
- Rupees: room `$11` (not on the submenu map, bomb walls on all four sides, on the route) holds ten rupees worth 1 each in a diamond around (`$78`, `$90`). Walking the route's line through the room (up at X = `$78` to Y = `$8D`, then right) collects six of them (trace). [`RupeeStash_Init_Full`](../src/bank1/trap-rupee-stash.asm#L75), [`RupeeStash_Update_Full`](../src/bank1/trap-rupee-stash.asm#L338)

# Map

<!-- generated:map -->
```
[00  ] K [01k ]---[02  ]   [03T ]
 |/S       X        |       |/S
[10  ] X [11h ] X [12  ] S [13Bi]
 S/|       X
[20  ]---[21i ]
  K        K
[30  ] K [31  ]S/-[32  ]
  |
[40k ]
  |
[50  ]---[51k ]
           |
         [61  ] K [62i ]
           |
[70k ]---[71S ]

[73S] = room $73 and its marks: S start room, B boss room, T Triforce room,
        k a key lies here, i another room item lies here, h not shown on the submenu map.
Between two rooms, the door: --- or | open, K key door, S shutter, X bombable wall,
        F false wall (walk-through), nothing = wall. A/B = the two sides differ: the left
        or upper room's side first, . = wall on that side.
Doors on sides with no room of the level: $71 S open.
```

- Cellar `$60` (treasure cellar): both exits lead to `$32`, whose stairs lead here; item: ladder ($0D). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md)
<!-- /generated:map -->

# Rooms

Generated from the level data by `tools/romdata.py`; do not edit between the markers.

<!-- generated:rooms -->
| room | N | S | W | E | monsters | item | secret | block | dark | layout | floor | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $00 | wall | open | wall | key | 1 x dungeon person ($4E) | - | - | - | - | $26 | blank | person $4E says: WALK INTO THE / WATERFALL. |
| $01 | wall | bomb | key | open | 8 x Keese blue ($1B) | key ($19) at $80,$80 | - | - | D | $16 | water | - |
| $02 | wall | open | open | wall | 1 x Trap ($49) | - | - | - | D | $01 | blocks | - |
| $03 Triforce | wall | open | wall | wall | none | Triforce piece ($1B) at $80,$90 | - | - | - | $29 | blocks, statues | - |
| $10 | shutter | shutter | wall | bomb | 1 x Manhandla ($3C) | - | all dead: shutters open | - | - | $06 | blocks, statues | - |
| $11 off-map | bomb | bomb | bomb | bomb | 1 x Rupee stash ($35) | - | - | - | - | $26 | blank | - |
| $12 | open | wall | bomb | shutter | 5 x Vire ($12) | - | push block: shutters open | B | - | $0A | blocks | boss sound $10 |
| $13 boss | shutter | wall | shutter | wall | 1 x Gleeok ($43) | heart container ($1A) at $D0,$C0 | all dead: shutters open, room item appears | - | - | $05 | blocks | - |
| $20 | open | key | wall | open | 5 x Vire ($12) | - | - | - | - | $15 | water | - |
| $21 | bomb | key | open | wall | 5 x Gel ($15) | map ($17) at $D0,$C0 | - | - | D | $12 | water | - |
| $30 | key | open | wall | key | 2 x Bubble ($2B), 3 x Vire ($12) | - | - | - | D | $18 | water | - |
| $31 | key | wall | key | shutter | 5 x Vire ($12) | - | all dead: shutters open | - | - | $17 | water | - |
| $32 | wall | wall | open | wall | 2 x Bubble ($2B), 2 x Zol ($13), 2 x Like-Like ($17) | - | push block: stairs appear | B | - | $0A | blocks | stairs lead to cellar $60 |
| $40 | open | open | wall | wall | 5 x Zol ($13) | key ($19) at $80,$80 | - | - | D | $15 | water | - |
| $50 | open | wall | wall | open | 5 x Vire ($12) | - | - | - | D | $11 | blocks | - |
| $51 | wall | open | open | wall | 8 x Keese blue ($1B) | key ($19) at $90,$90 | - | - | - | $03 | blocks | - |
| $61 | open | open | wall | key | 3 x Vire ($12) | - | - | - | - | $02 | blocks | - |
| $62 | wall | wall | key | wall | 5 x Vire ($12) | compass ($16) at $80,$80 | - | - | D | $0C | blocks | - |
| $70 | wall | wall | wall | open | 8 x Keese blue ($1B) | key ($19) at $90,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $71 start | open | open | open | wall | none | - | - | - | - | $21 | statues, sand | - |
| $60 cellar | - | - | - | - | 4 x Keese blue ($1B) | ladder ($0D) at $80,$90 | - | - | - | $3F | treasure cellar | left exit to $32, right exit to $32; Link comes out at $C0,$BD |
<!-- /generated:rooms -->

# Floor plans

<!-- generated:floors -->
One character per floor square, 12 columns x 7 rows inside the walls (left column = west, top row =
north): `.` floor (tile `$74`), `B` block (tile `$B0`; a push block object is placed on one in a room
with a push block), `W` water (tile `$F4`), `S` statue (tiles `$94` and `$B4`), `>` stairs (tile `$70`),
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L63), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L73), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
Stairs that a secret makes appear are not part of the layout.

```
layout $00  rooms: $70
............
............
............
............
............
............
............
```

```
layout $01  rooms: $02
............
............
..B......B..
.B........B.
..B......B..
............
............
```

```
layout $02  rooms: $61
............
............
..B......B..
............
..B......B..
............
............
```

```
layout $03  rooms: $51
............
..B......B..
............
............
............
..B......B..
............
```

```
layout $05  rooms: $13
BBBBB..BBBBB
BBBB....BBBB
BB........BB
............
............
..B......B..
............
```

```
layout $06  rooms: $10
............
............
............
.B........B.
............
...B....B...
S..........S
```

```
layout $0A  rooms: $12,$32
............
............
............
....B..B....
............
............
............
```

```
layout $0C  rooms: $62
.B.B........
.B.BBB.BBB..
.B...B...BBB
.B.B.B...B..
.BBB.B...B..
.....B.B.B.B
..B....B...B
```

```
layout $11  rooms: $50
..B........B
.B...B....B.
....B....B..
...B....B...
..B....B....
.B....B...B.
B........B..
```

```
layout $12  rooms: $21
............
.WWWWWWWWWW.
.WWWWWWWWWW.
.WW......WW.
.WW......WW.
.WWWW..WWWW.
.WWWW..WWWW.
```

```
layout $15  rooms: $20,$40
............
.WWW....WWW.
.W........W.
.W.WWWWWW.W.
.W........W.
.WWW....WWW.
............
```

```
layout $16  rooms: $01
WWWWW..WWWWW
W...WWWW...W
W.WWW..WWW.W
..W......W..
W.WWW..WWW.W
W...WWWW...W
WWWWW..WWWWW
```

```
layout $17  rooms: $31
WWWWW..WWWWW
W...W....W.W
W.W.W.WW.W.W
..W.W..W.W..
W.W.WW.W.W.W
W.W....W...W
WWWWW..WWWWW
```

```
layout $18  rooms: $30
............
............
WWWWWWWWWWWW
............
............
............
............
```

```
layout $21  rooms: $71
............
.S..S..S..S.
............
.S..S..S..S.
.....::.....
.S.:S::S:.S.
...::::::...
```

```
layout $26  rooms: $00,$11
____________
____________
____________
____________
____________
____________
____________
```

```
layout $29  rooms: $03
............
.BBBBBBBBBB.
.B..S..S..B.
.B.S....S.B.
.B........B.
.BBBB..BBBB.
............
```

```
layout $3F (treasure cellar)  rooms: $60
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

| Kind (type) | Rooms and count | HP | What kills or stuns it | What it does to Link | [enemies.md](../enemies.md) section |
| --- | --- | --- | --- | --- | --- |
| Vire (`$12`) | route: `$61` x3, `$50` x5, `$30` x3, `$31` x5, `$20` x5, `$12` x5; elsewhere: `$62` x5 | `$40` | killed by a hit of `$40`: magic sword, bomb, silver arrow. Any weaker hit (wooden or white sword, rod, wooden arrow, fire) turns it into 2 red Keese. Boomerang: stun | 1 heart | `# Vire` |
| Red Keese (`$1C`) | up to 2 from every split Vire | 0 | any hit, the boomerang included | half a heart | `# Keese (blue, red, black)` |
| Blue Keese (`$1B`) | route: `$51` x8, cellar `$60` x4; elsewhere: `$70` x8, `$01` x8 | 0 | any hit, the boomerang included | half a heart | `# Keese (blue, red, black)` |
| Zol (`$13`) | route: `$40` x5, `$32` x2 | `$20` | killed by white or magic sword, bomb, arrow. Wooden sword or fire splits it into 2 Gels. Boomerang: stun | 1 heart | `# Zol` |
| Gel (`$15`, child `$14`) | route: `$21` x5; 2 from every split Zol | 0 | any hit, the boomerang included | half a heart | `# Gel` |
| Like-Like (`$17`) | route: `$32` x2 | `$90` | sword 9 / 5 / 3 hits (wooden / white / magic), bomb 3. Boomerang: stun | 1 heart, then holds Link in place; eats the magic shield after 95 frames | `# Like-Like` |
| Bubble (`$2B`) | route: `$30` x2, `$32` x2 | - | nothing | no damage, knockback; the sword is gone for about 160 frames | `# Bubble (flashing, red, blue)` |
| Trap (`$49`) | elsewhere: `$02`, six | - | nothing | 1 heart | `# Trap` |
| Manhandla (`$3C`) | elsewhere: `$10` | `$40` per hand, four hands | sword 4 / 2 / 1 hits per hand, bomb blast 1; boomerang and fire do nothing | 1 heart (touch or fireball) | `# Manhandla` |
| Gleeok, two heads (`$43`) | boss: `$13` | `$A0` for the first head, `$60` for the second | sword, sword beam or rod swing only: 10 + 6 wooden, 5 + 3 white, 3 + 2 magic | 1 heart (neck base, flying head, fireball); 3/8 heart (head on its neck) | `# Gleeok` |

Not monsters: the rupee stash (`$35`) of room `$11` and the old man (`$4E`) of room `$00`.

- **Where a fight is required.** On the route only four rooms must be cleared: `$31` (first visit, for the E shutter), `$32` (for the push block), `$12` (for the push block) and `$13` (the boss). Every other room is left through an open doorway, a key door or a bombed wall, with its monsters alive. [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152), [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74)
- **"Cleared" counts everything except Bubbles.** The test looks at the object slots: any type `$01`-`$2A` or `$2E`-`$48` still there keeps the room uncleared, so the red Keese of split Vires and the Gels of split Zols must die too. [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124)
- **Vires split.** A hit that does not kill a Vire sends it flying in the weapon's direction until a tile or the room bound stops it, and there it becomes two red Keese; while it flies it cannot be hit and cannot hurt. With a wooden or white sword a room of 5 Vires is therefore 5 + 10 kills. [`Vire_CheckCollisions`](../src/bank4/vire.asm#L154), [`Vire_Update@SplitUp`](../src/bank4/vire.asm#L61)
- **Split Vires come back.** A split is not a kill and red Keese are not counted for the room, so a room cleared by splitting is banked with 0 kills: (trace, room `$31`, white sword) 15 swings cleared it and opened the shutter, the kill count stayed 0, and when Link came back from room `$32` all 5 Vires were there again. With the magic sword the same room banked 5 kills and was empty on the way back. [`Obj_EndMetastate`](../src/bank7/object-update-table.asm#L308), [`RoomHistory_SaveKillCountUW`](../src/bank5/room-history-uw.asm#L110)
- **So on the way back** (rooms `$31` and `$30`, second visits) expect the Vires again unless they were killed outright. Neither room needs clearing then: in `$31` the W key door is already open, and in `$30` Link only crosses to the N door. [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- **Room `$32`.** A Like-Like that touches Link jumps onto him and holds him; A and B still work while he is held, and it lets go when it dies or when the room is cleared. Kill it within 95 frames of the capture to keep the magic shield. A Bubble's touch removes the sword for about 160 frames (the B item still works), and Bubbles cannot be killed or stunned: they stay in the room, but they do not stop the block. [`LikeLike_Update`](../src/bank4/like-like.asm#L46), [`Bubble_Update`](../src/bank4/bubble-keese.asm#L93), [`SwordBlockedLongTimer`](../src/Variables.inc#L62)
- (trace, room `$32`) White sword: both Zols died in one hit each, 4 kills were banked, and after the cellar the room held only its 2 Bubbles. Wooden sword: the Zols split, 2 kills were banked, and after the cellar the 2 Zols were back with the Bubbles. [`Zol_CheckCollisions`](../src/bank4/zol-gel.asm#L170), [`RoomHistory_SaveKillCountUW`](../src/bank5/room-history-uw.asm#L110)
- **Keese rooms.** Blue Keese need `$A1` frames to reach their top speed and fly over water and blocks; in room `$51` the key lies in the open, so Link can take it and leave before they are fast. In the cellar the four Keese start at Y = `$9D`, X = `$20`, `$60`, `$90`, `$D0`, and cost half a heart per touch. [`Keese_Update`](../src/bank4/bubble-keese.asm#L163), [`Mode9Cellar_KeeseXs`](../src/bank5/spawn-positions.asm#L44)
- **Every monster starts in a cloud**: its update and its collisions do not run until the cloud ends (object slot number + 12 frames after the room starts), so the first moments in a room are free. [mechanics.md](../mechanics.md) `# Monster spawning, edge spawns and respawn (room history)`

**The boss: Gleeok with two heads, room `$13`**

- **The room locks.** Link enters from the W through the shutter that the block of room `$12` opened; it closes behind him, and the only other door (N, to the Triforce room) is a shutter too. Nothing opens until Gleeok is dead. [`Room_EnterCommon@Method2`](../src/bank5/enter-room.asm#L211), [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- **What is where.** The body is drawn at X `$74`-`$84`, Y `$57`-`$67` (top middle, in front of the N door); it cannot be hit and does not hurt. Both necks start at (`$7C`, `$6F`) and hang down; a head moves 1 pixel on each axis every 4 frames and stays within `$0B` pixels per neck segment of the one before it. [`Gleeok_Init`](../src/bank4/gleeok-manhandla-gohma-inits.asm#L59), [`Gleeok_MoveHead`](../src/bank4/gleeok-head-body.asm#L285), [`Gleeok_CalcSegmentLimits`](../src/bank4/gleeok.asm#L456)
- **How it dies.** Hit the heads with the sword (the beam and the rod's swing count as well); nothing else in Link's inventory does anything. The heads share one HP byte: the first falls after `$A0` damage, whichever head took the hits, the second after `$60` more. After a hit no head can be hit, or hurts, for about 32 frames. [`Gleeok_CheckSegmentCollisions`](../src/bank4/gleeok-head-body.asm#L128), [`Sword_DamagePoints`](../src/bank1/sword-bomb-arrow-collision.asm#L182)
- **The cut-off head** flies around, chases Link most of the time, costs 1 heart per touch, shoots, and cannot be destroyed; it disappears when the second head falls. One Gleeok fireball exists at a time (1 heart, no shield stops it). [`Gleeok_UpdateFlyingHead`](../src/bank4/gleeok.asm#L78), [`Gleeok_Update`](../src/bank4/gleeok.asm#L175)
- (trace) White sword, Link standing at (`$78`, `$9D`) facing up and pressing A every other frame, walking back to that spot whenever he was knocked off it: the first head fell after 5 hits (frame 245), the second after 3 more (frame 397); Link was hit 5 times in that time. [`Gleeok_CheckSegmentCollisions`](../src/bank4/gleeok-head-body.asm#L128)
- **When it dies** the flying head is removed (a fireball in flight is not), the N and W shutters open, the heart container appears at (`$D0`, `$C0`), and the death rolls one drop. Gleeok does not come back, in this visit or a later one. [`Gleeok_CheckSegmentCollisions@BossDied`](../src/bank4/gleeok-head-body.asm#L250), [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- **The other boss.** The Manhandla of room `$10` guards nothing: no item, only the shutters to rooms `$00` and `$20`. It is on the route only for a Link without bombs. [`Manhandla_Init`](../src/bank4/gleeok-manhandla-gohma-inits.asm#L155), [enemies.md](../enemies.md) `# Manhandla`

# Route

How this route was established: from the level data and the code (door types, secret triggers, floor plans; a search over rooms, keys, bombs and the ladder for the fewest room entries), then walked in the running game from the entry spot of room `$71` to the overworld, starting with 0 keys and 2 bombs. That walk had the monsters taken out of each room on entry, so it shows doors, keys, blocks, walls, water and paths but no fights: 3,232 frames to the ladder, 7,163 to the Triforce piece (trace). The required fights and the ladder crossings were traced separately (see "Monsters and boss" and "What Link gets here").

Keys are counted from 0: add the number Link brings. Positions are Link's X, Y (see "At a glance").

| Step | Room | Entered from | What to do | Leave by | Keys after | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `$71` | the overworld (S doorway) | nothing to fight; straight up at X = `$78` | N, open | 0 | W leads to the key room `$70` |
| 2 | `$61` | S | 3 Vires, no need to fight; straight up at X = `$78` | N, open | 0 | E is the key door to the compass |
| 3 | `$51` | S | 8 blue Keese; take the **key**: up to Y = `$8D`, right to X = `$90`; then left along Y = `$8D` | W, open | 1 | |
| 4 | `$50` | E | 5 Vires, no need to fight; diagonal rows of blocks: left to X = `$B0`, down to Y = `$AD`, left to X = `$90`, down to Y = `$BD`, left to X = `$70`, up to Y = `$7D`, right to X = `$80`, up to Y = `$5D`, left to X = `$78`, up | N, open | 1 | dark |
| 5 | `$40` | S | 5 Zols; take the **key**: right along the bottom row to X = `$D0`, up to Y = `$5D`, left to X = `$80`, down to Y = `$7D`; back up to Y = `$5D`, left to X = `$78`, up | N, open | 2 | dark; water ring, open at the top: no ladder needed |
| 6 | `$30` | S | 3 Vires, 2 Bubbles, no need to fight; up to Y = `$8D`, right to X = `$D0`, push Right: **a key opens the E door** | E, key door | 1 | dark; the water row cuts off the N door |
| 7 | `$31` | W | **kill the 5 Vires** and their Keese: the E shutter opens. Land path W to E: right to X = `$30`, up to Y = `$6D`, right to X = `$50`, down to Y = `$AD`, right to X = `$80`, up to Y = `$8D`, left to X = `$70`, up to Y = `$6D`, right to X = `$A0`, down to Y = `$AD`, right to X = `$C0`, up to Y = `$8D`, right | E, shutter | 1 | water maze; leave the N key door alone |
| 8 | `$32` | W | **kill the 2 Zols and 2 Like-Likes**; at (`$50`, `$8D`) hold Right `$10` frames: the block slides, stairs appear at the top right; up to Y = `$5D`, right to X = `$D0` | the stairs | 1 | 2 Bubbles stay |
| 9 | cellar `$60` | the stairs of `$32` | 4 blue Keese; down to Y = `$BD`, right to X = `$B0`, up to Y = `$8D`, left: **the ladder** at X = `$88`; back the same way and up the left ladder | the top of the left ladder | 1 | from here on Link has the ladder |
| 10 | `$32` | the cellar, at (`$C0`, `$BD`) | nothing required; left along the bottom row to X = `$20`, up to Y = `$8D`, left | W, open | 1 | |
| 11 | `$31` | E (the shutter closes behind) | hold Left along Y = `$8D`: the ladder crosses the four water squares | W, key door (open) | 1 | **ladder** (shortcut); Vires may be back |
| 12 | `$30` | E | left to X = `$78`, then hold Up: **the ladder crosses the water row**; push Up: **a key opens the N door** | N, key door | 0 | **ladder required**; dark |
| 13 | `$20` | S | 5 Vires, no need to fight; up to Y = `$BD`, right along the bottom row to X = `$D0`, up to Y = `$8D`, right | E, open | 0 | water ring, no ladder needed; N leads to Manhandla |
| 14 | `$21` | W | 5 Gels, no need to fight; right to X = `$20`, up to Y = `$5D`, right to X = `$78`, face up, **bomb**; the wall is open 81 frames later | N, bomb wall | 0 | dark; the map is in the bottom right corner |
| 15 | `$11` | S | ten rupees, no monsters; up to Y = `$8D` (5 rupees), right to X = `$D0` (1 more), face right, **bomb** | E, bomb wall | 0 | not on the submenu map |
| 16 | `$12` | W | **kill the 5 Vires** and their Keese; at (`$50`, `$8D`) hold Right `$10` frames: the block slides and the E shutter opens; down to Y = `$9D`, right to X = `$D0`, up to Y = `$8D`, right | E, shutter | 0 | the boss is heard here |
| 17 | `$13` | W (the shutter closes behind) | **kill Gleeok**; take the **heart container** at (`$D0`, `$BD`); then to X = `$78` and up | N, shutter | 0 | no way out before he is dead |
| 18 | `$03` | S | straight up at X = `$78`: the **Triforce piece** at Y = `$95` | the level ends | 0 | Link comes out in overworld room `$45` |

As a list: `$71`, `$61`, `$51`, `$50`, `$40`, `$30`, `$31`, `$32`, cellar `$60`, `$32`, `$31`, `$30`, `$20`, `$21`, `$11`, `$12`, `$13`, `$03`.

- **Key balance.** Found on the route: 2 (rooms `$51`, `$40`). Spent: 2 (room `$30`, E door at step 6 and N door at step 12). Link leaves with the keys he came with; the keys of rooms `$70` and `$01` are extra (see "Optional detours"). [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`InvKeys`](../src/Variables.inc#L584)
- **Bomb balance.** Spent: 2 (steps 14 and 15). Link stands on the door spot facing the wall; the bomb lies `$10` pixels ahead of him, inside the `$18`-pixel range of the wall's hotspot, and the wall is passable 81 frames after the button (trace, both walls). Bombed walls stay open. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Bomb_WallHotspotsX`](../src/bank7/bomb.asm#L60)
- **Order that matters.** The ladder (step 9) must come before the N door of room `$30` (step 12). The Vires of room `$31` must die on the first visit (step 7), not on the second. A push block moves only once per room entry and only in a cleared room. Entering room `$13` (or room `$10`) locks Link in with the boss. [`PushBlock_State2Done`](../src/bank4/push-block.asm#L191), [`Door_TouchShutter`](../src/bank5/touch-door.asm#L126)
- **The N key door of room `$31`** looks like a shortcut to room `$21` and the bomb wall, one room shorter than going back through `$30`. With the ladder alone it is not: it lets Link onto the island of room `$21`, from which the plain ladder cannot reach the rest of the room (trace: key spent, Link stuck one square out). With the ladder clip it is the shortcut of the next bullet. [`Ladder_CheckAndDraw@HandleInput`](../src/bank5/ladder.asm#L125)
- **Shortcut with the ladder clip** (2 room entries fewer; same keys and bombs). After the cellar: `$32`, `$31` (from the E door hold Left along Y = `$8D` to X = `$78`, the ladder crossing the four single squares; Up through the N key door: the key that step 12 would spend on `$30` N), `$21` on the island, `$11`. In `$21` walk up to (`$78`, `$8D`) and hold Up: the ladder is laid at Y = `$80` when Link reaches Y = `$85`; on the frame that starts at Y = `$82` press Left or Right for exactly one frame, then hold Up: one press crosses both water squares and Link stops at (`$78`, `$55`), 8 pixels into the top wall (43 frames from (`$78`, `$8D`) at rest). Down 5 frames to the N wall spot (`$78`, `$5D`), Up 1 frame to face the wall, **bomb** (the bomb of step 14), Up into `$11` and on as from step 15. It skips the second visits of `$30` and `$20`. Needs: the ladder, a key, a bomb, a free monster slot, and a walk up into Y = `$85` from Y = `$8D` or below (arriving on `$85` any other way spoils the timing). A miss leaves Link on the ladder at Y = `$7D`: Down 11, Up 8, press again (trace, room by room; the 5 Gels of `$21` were absent, and with them present the recipe still worked but mostly cost half a heart or more). Exact rule, decision table and failure cases: [mechanics.md](../mechanics.md) `# Grid-point skip: block clip, ladder clip and their run-up`. [`Ladder_CheckAndDraw@HandleInput`](../src/bank5/ladder.asm#L125), [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136)
- **Without bombs** (two rooms longer, one more key): from room `$20` go N instead of E. `$10` (entered from S; the shutter closes; kill Manhandla, both shutters open), N to `$00` (the old man; walk up to Y = `$8D`, right to X = `$D0`), E through a key door to `$01` (hold Right: the ladder crosses at X = `$40` and `$B0`; the key is one square above the door row at X = `$80`), E to `$02` (six traps, see "Hazards"), S to `$12` (entered from N: push the block from any side), E to `$13`. Keys: 3 spent, 3 found (`$51`, `$40`, `$01`), but the third is found behind the third door, so Link must bring one or take the key of room `$70` first (trace of this walk from room `$20`, monsters taken out, 1 key at the start). [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152), [`Trap_Update_Full`](../src/bank1/trap-rupee-stash.asm#L189)
- **After a death.** CONTINUE restarts Link in room `$71` with 3 hearts without reloading the level: his items, keys, the opened key doors and bombed walls, and the kills banked for each room are kept; the room history is cleared, so a room that was banked as fully cleared fills again while a partly cleared room keeps its reduced count. Dead bosses stay dead and taken items stay taken. [`Mode8Continue_Init`](../src/bank5/mode-inits-obj-lists.asm#L49), [`Mode8Continue_ChoiceModes`](../src/bank5/mode8-continue-mode10-stairs.asm#L49), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- **What a death costs.** Before the ladder: the fight of room `$31` again (its shutter is closed and its Vires are back, whichever sword was used), while room `$32` keeps its banked kills (only the Bubbles are left if the Zols were killed outright). With the ladder: the walk back to room `$30` is through open doorways only. [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50), [`RoomFlags_SetDoorOpened`](../src/bank5/shutters-door-flags.asm#L41)
- **Wallmasters: none in this level**, so nothing carries Link back to the start room. [enemies.md](../enemies.md) `# Wallmaster`

# Optional detours

- **Key of room `$70`** (+2 room entries, +1 key, no cost). In room `$71` a statue stands on the W door's row, so go up to Y = `$9D`, left to X = `$20`, up to Y = `$8D` and left. All 8 blue Keese must die; then the key appears at (`$90`, `$90`) (secret trigger 7) and is taken at (`$90`, `$8D`) (trace; 721 frames of walking for the round trip, fight not counted). Worth it when keys are short later, or for the route without bombs. The room has no water, no blocks and is not dark. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- **Compass, room `$62`** (+2 room entries, 1 key spent, nothing gained but the marker). The E door of room `$61` is a key door; room `$62` is dark, holds 5 Vires and a maze of blocks. From its W door: down the left column to Y = `$AD`, right to X = `$60`, down to Y = `$BD`, right to X = `$80`, up to Y = `$7D`, where the compass is taken (trace). [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51), [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- **Map, room `$21`** (no extra room). It is in a room of the route but not on its line: from the N wall spot go on right along the top row to X = `$D0` and down the right column to Y = `$BD`, where it is taken (trace: 141 frames each way). Five Gels wander on the same ring. [`Inv_HasMap`](../src/bank5/submenu-map.asm#L153), [`Item_TryTake`](../src/bank1/take-item.asm#L206)
- **The other four rupees of room `$11`** (no extra room): they lie at (`$60`, `$90`), (`$70`, `$80`), (`$80`, `$80`) and (`$78`, `$70`), within 9 pixels of Link on both axes to be taken. Once one rupee has been taken the room counts as cleared when Link leaves, and the rest are gone for good. [`RupeeStash_Update_Full`](../src/bank1/trap-rupee-stash.asm#L338), [`RoomHistory_SaveKillCountUW`](../src/bank5/room-history-uw.asm#L110)
- **Key of room `$01`** (+2 room entries, 1 bomb, +1 key; needs the ladder). From room `$11` bomb the N wall (stand at (`$78`, `$5D`) facing up) and enter `$01` from the S: holding Up, the ladder crosses the one water square above the door (without it Link stops at Y = `$B5`); from (`$78`, `$8D`) step right to X = `$80` and up to Y = `$7D` for the key at (`$80`, `$80`); back the same way (trace). Eight blue Keese; dark. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Link_EndMoveAndAnimate@SetUpLadder`](../src/bank7/link-animate-ladder.asm#L246)
- **Old man, room `$00`** (on the route without bombs only): his text is "WALK INTO THE WATERFALL." Nothing is given or sold; Link is halted while the text is typed (trace: 151 frames). [`Person_TextSelectorsLevels3468`](../src/bank1/underworld-person-init.asm#L134), [data/person-text.md](../data/person-text.md)
- **Manhandla, room `$10`, and the traps of room `$02`**: no item in either room; they are passed only on the route without bombs. [`Manhandla_Init`](../src/bank4/gleeok-manhandla-gohma-inits.asm#L155), [`Trap_Init_Full`](../src/bank1/trap-rupee-stash.asm#L134)
- **The N key door of room `$31`** costs a key and leads onto the island of room `$21`, which the plain ladder cannot leave; it is worth taking only with the ladder clip (see "Route", "Shortcut with the ladder clip"). [`Ladder_CheckAndDraw@HandleInput`](../src/bank5/ladder.asm#L125)

# Hazards and rules that matter here

- **Shutters** (5 doorways). Room `$31` E and room `$10` N and S open when the room is cleared; room `$12` E opens when the block has been pushed; room `$13` W and N open when Gleeok is dead. An opened shutter is not saved: it is closed again on the next visit, and a shutter Link walks in through closes behind him as he enters. [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68), [`Secret_Trigger4BlockShutters`](../src/bank5/underworld-secrets.asm#L215), [`Door_Update`](../src/bank5/update-doors.asm#L60)
- **Doorways that differ by side.** Room `$10` can be walked into from `$00` and `$20` through plain doorways whose `$10` side is a shutter, and room `$13` from `$03` likewise; room `$32` leaves W through a plain doorway whose `$31` side is a shutter. Walking in is free; the way back needs the room's secret. [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55), [`Door_SetEnteringDoorwayOpened`](../src/bank5/mode7-scroll-init.asm#L368)
- **Push blocks** (rooms `$32`, `$12`). In both the block is the left one of the two blocks of layout `$0A`, at (`$60`, `$90`). It moves only while the room is cleared, after `$10` frames in a row of Link pushing exactly toward it from an adjacent, exactly aligned spot, and once per room entry; pushing it with monsters alive does nothing (trace). [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74)
- **What the push does** depends on the room's secret trigger: stairs at (`$D0`, `$60`) in room `$32` (trigger 5), the E shutter in room `$12` (trigger 4; open 7 frames after the block arrives, trace). The pushed block stays where it stopped as a block: go round it. [`Secret_Trigger5BlockStairs`](../src/bank5/underworld-secrets.asm#L226), [`Secret_Trigger4BlockShutters`](../src/bank5/underworld-secrets.asm#L215), [`PushBlock_State1Moving`](../src/bank4/push-block.asm#L165)
- **False walls: none.** [data/levelblock-uw-1to6-q1.md](../data/levelblock-uw-1to6-q1.md)
- **Bomb walls** (4 doorways, all of them walls of room `$11`: to `$01`, `$10`, `$12`, `$21`). They look like plain wall until opened. The bomb's own position must be less than `$18` pixels on each axis from the wall's hotspot: N (`$78`, `$5D`), S (`$78`, `$BD`), W (`$20`, `$8D`), E (`$D0`, `$8D`); the test is made once, when the blast ends. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Bomb_WallHotspotsX`](../src/bank7/bomb.asm#L60), [`Door_LayOut`](../src/bank5/lay-out-doors.asm#L131)
- **Dark rooms** (`$01`, `$02`, `$21`, `$30`, `$40`, `$50`, `$62`): the picture is black, the rooms work as any other. Entering one adds a 22-frame redraw of the play area, and a fade to black (4 steps, 10 frames apart) when the room before it was lit. [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124), [`Mode7Scroll_Init_Sub5`](../src/bank5/mode7-scroll-init.asm#L256), [`Mode7Scroll_Update_Sub6`](../src/bank5/mode7-scroll-update.asm#L227)
- **Water** (rooms `$01`, `$20`, `$21`, `$30`, `$31`, `$40`). Water tiles (`$F4`) are unwalkable like blocks; going up, Link stops with his upper half over the water (Y = water top + 5), which is where the ladder is laid. What needs the ladder: `$30` S or E to N; anything in `$01`. What does not: `$31` W to E (land path), `$20` and `$40` (rings open at the top and bottom), `$21` W to N. What the plain ladder cannot do: leave the island of `$21`; that takes the ladder clip (next bullet). [`Tile_GetAheadOfObj`](../src/bank7/tile-collision.asm#L65), [`RoomFirstUnwalkableTile`](../src/Variables.inc#L292), [`Link_EndMoveAndAnimate@CheckLadder`](../src/bank7/link-animate-ladder.asm#L209)
- **Ladder clip off the island of `$21`** (trace). North: the recipe of "Route", "Shortcut with the ladder clip"; Link ends at (`$78`, `$55`), past the room bound, from where Down reaches the N wall spot and Left walks him along Y = `$55` to X = `$20` (then Down to Y = `$8D` and Left: the W door). West, rows Y = `$8D`, `$95`, `$9D`: from X = `$50` Left 8 frames, Up 1, Left: he ends at X = `$20`, and on Y = `$8D` walks straight on through the W door into `$20`. East, same rows: from X = `$A0` Right 8, Up 1, Right: he ends at X = `$D0` (the map's column; the E side is a wall). A failed press leaves him on the ladder over the water, from where the way back (Down, or the opposite direction) always works; there is no softlock. [`Ladder_CheckAndDraw@HandleInput`](../src/bank5/ladder.asm#L125), [mechanics.md](../mechanics.md) `# Grid-point skip: block clip, ladder clip and their run-up`
- **The ladder needs room.** It is an object in a monster slot: with all of `$01`-`$0B` in use it is not laid. It is put away on every room change, and while it is out Link can only go on or back along it. [`Obj_FindEmptyMonsterSlot`](../src/bank7/object-stubs-destroy-monster.asm#L169), [`Obj_ResetWeaponSlotsAndLadder`](../src/bank5/mode3-mode6-init.asm#L263), [`Ladder_CheckAndDraw`](../src/bank5/ladder.asm#L44)
- **Bubbles** (rooms `$30`, `$32`): the flashing kind only; a touch takes the A button away for `$10` timer steps of 10 frames and knocks Link back, without damage; nothing harms them. [`Bubble_Update`](../src/bank4/bubble-keese.asm#L93), [`Link_HandleInput`](../src/bank5/link-handle-input.asm#L68)
- **Like-Likes** (room `$32`): see "Monsters and boss"; the capture itself costs 1 heart, and nothing else touches Link while he is held. [`LikeLike_Update`](../src/bank4/like-like.asm#L46), [`Link_CheckMonsterTouch_NoReset`](../src/bank1/link-collision-harm.asm#L73)
- **Traps** (room `$02` only, off the route): six, at (`$20`, `$5D`), (`$20`, `$BD`), (`$D0`, `$5D`), (`$D0`, `$BD`), (`$40`, `$8D`), (`$B0`, `$8D`). A trap charges along a row or column when Link is within `$0E` pixels of its line, only toward the room's middle, at up to 1.75 pixels per frame, and creeps back at 0.5. The W door is flanked by the two left corner traps and faces the side trap at (`$40`, `$8D`), which never moves toward the door. [`Trap_StartXs`](../src/bank1/trap-rupee-stash.asm#L112), [`Trap_AllowedDirs`](../src/bank1/trap-rupee-stash.asm#L167), [`Trap_Update_Full`](../src/bank1/trap-rupee-stash.asm#L189)
- (trace, room `$02` entered from the W, leaving S) Holding Right out of the doorway and then Down along the left column: Link is hit after 26 frames and again every 48 frames while he stays in that column, and dies from 7 hearts at frame 344. Safe: in the doorway step right to X = `$13` (both left corner traps charge) and back to X = `$10` (they pass without touching him); wait until the lower trap has crept back to Y >= `$AD` and the upper to Y <= `$78` (70 frames); then right to X = `$20`, down to Y = `$9D`, right to X = `$30`, down to Y = `$AD`, right to X = `$78`, down through the S door without stopping: no damage. [`Trap_Update_Full`](../src/bank1/trap-rupee-stash.asm#L189)
- **Wallmasters, statues that shoot, the hungry Goriya: none.** The statues of rooms `$71`, `$10` and `$03` are scenery: only layouts `$24` and `$23` shoot. [`Statue_RoomLayoutIds`](../src/bank4/statues.asm#L45), [data/levelblock-uw-1to6-q1.md](../data/levelblock-uw-1to6-q1.md)
- **The old man** (room `$00`, person `$4E`) only talks: "WALK INTO THE WATERFALL." While he stands there Link cannot go higher than Y = `$8D` (trace). A weapon hit does not kill him; it makes two fireball sources beside the fires, at (`$48`, `$82`) and (`$A8`, `$86`), shoot at Link until he leaves the room. [`Person_BlockLinkMovingUp`](../src/bank1/modeb-cave-save-sprites.asm#L289), [`Person_CheckCollisions`](../src/bank1/underworld-person-update.asm#L89), [`Statue_Update`](../src/bank4/statues.asm#L86)
- **Stairs and the cellar.** One cellar (`$60`, a treasure cellar, both exits to room `$32`); its stairs exist only after the push and must be made again on every visit of room `$32`. Taking stairs banks the room's kills first; in the cellar there are no doors, bombs open nothing, and four blue Keese are made on every entry. [`Warp_StartStairsMode`](../src/bank5/warps-stairs-caves.asm#L131), [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43), [`Spawn_AssignPositions`](../src/bank5/spawn-positions.asm#L64)
- **The rupee room** (`$11`) is a stash object, not a monster room: touching any rupee sets the room's object count to 0, so the room is banked as cleared on leaving and never makes the stash again; a stash left untouched is made again in full. [`RupeeStash_Update_Full`](../src/bank1/trap-rupee-stash.asm#L338), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- **A key door that opens by itself.** While a shutter is opening (8 frames) and Link's own timer is 0, a closed key door lets him through unpaid. The only room of this level with a shutter and a closed key door is `$31`, and that door is the N one, which leads anywhere only with the ladder clip (see "Route"). [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`Door_Touch_BlockWhileTimer`](../src/bank5/touch-door.asm#L193), [mechanics.md](../mechanics.md) `# Doorways, doors, keys, shutters and bomb walls`
- **Boss roar.** Room `$12`, next to the boss room, repeats a roar sample until the flags of room `$13` say the boss is dead. [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42)
- **Special cases in the code for this level:** none by room id. By level number: the person's text and talk-only behaviour (levels 3, 4, 6, 8 share one table), and the sprite palette row chosen for Gleeok's object type. [`Person_TextSelectorsLevels3468`](../src/bank1/underworld-person-init.asm#L134), [`Person_UpdateUnderworld_Full`](../src/bank1/underworld-person-update.asm#L44), [mechanics.md](../mechanics.md) `# Special-cased rooms`

# Second quest

In Q2 the entrance in overworld room `$45` still loads internal level 4, but its rooms come from the Q2 level block ([`Level_BlockUW1to6Q2`](../src/bank6/level-blocks-common-data.asm#L78), listed in [data/levelblock-uw-1to6-q2.md](../data/levelblock-uw-1to6-q2.md)) and its level info is patched ([`Quest_Q2LevelInfoPatch4`](../src/bank6/quest2-replacements.asm#L119): start room `$72`, Triforce room `$00`, boss room `$10`); the game shows it as LEVEL-5. The dungeon the game shows as LEVEL-4 in Q2 is internal level 5 ([`Quest_Q2LevelInfoPatch5`](../src/bank6/quest2-replacements.asm#L131), [data/levelinfo.md](../data/levelinfo.md)).

Neither is covered here: this document describes the first quest only.
