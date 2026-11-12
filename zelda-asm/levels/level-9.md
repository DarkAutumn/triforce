# Level 9: the silver arrow, the red ring, Ganon and Zelda

Level 9 is the last level of Q1: 57 rooms and 8 cellars behind a rock wall in overworld room `$05` that a bomb
opens. The man in its second room (`$66`) lets Link through only with all eight Triforce pieces. There is no
Triforce piece and no heart container here: Link collects the silver arrow (cellar `$4F` under room `$10`) and,
optionally, the red ring (cellar `$00` under room `$07`), then fights Ganon in room `$42`, takes the Triforce of
Power he leaves, and walks north to Zelda in room `$32`, which ends the game. Link must bring the eight pieces,
a sword, the bow and at least one rupee, six bombs, and either the magic key or three keys; the ladder saves two
rooms. What makes the level different: the route runs through four tunnel cellars whose stairs are closed in by
blocks (a room must be cleared before its block moves), several doors are one-way shutters, and Ganon's room
cannot be left once entered without the silver arrow.

> **Written from the ROM only.** Unlike levels 1-4, nothing here was walked in the running game; the route, the
> requirements and the room notes come from the level data and the code. When you work on this level, check each
> step against the game and the code, and update this document with what you find (mark it `(trace)`).

Contents (generated; `L` = line in this file):

<!-- toc -->
- L33 At a glance
- L70 What Link must bring
- L82 What Link gets here
- L120 Map
- L165 Rooms
- L239 Floor plans
- L573 Monsters and boss
- L615 Route
- L671 Optional detours
- L682 Hazards and rules that matter here
- L700 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 9 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-9 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | $05 (tile object: $63 rock wall at $50,$A0); not in Q1, the secret works in Q2 only: $00 (tile object: $63 rock wall at $A0,$80) | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178), [`Flute_SecretRoomIdsOW`](../src/bank7/flute.asm#L36) |
| Start room | $76 | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Compass marker room (the Triforce room field) | $32 (no Triforce piece there; monsters: 1 x Zelda ($37)); the field only places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $42 (1 x Ganon ($3E)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | $60, $70, $72, $75, $67, $77, $00, $4F | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 57: 54 on the submenu map, 3 the map does not show ($07, $32, $35) | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (room and side that has it) | 32 ($01 E, $02 W, $06 S, $16 N, $12 S, $22 N, $12 E, $13 W, $13 S, $23 N, $14 E, $15 W, $22 E, $23 W, $23 S, $33 N, $25 E, $26 W, $26 S, $36 N, $30 S, $40 N, $40 S, $50 N, $53 S, $63 N, $62 E, $63 W, $63 S, $73 N, $73 E, $74 W) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (room and side that has it) | 38 ($03 E, $04 W, $05 E, $06 W, $07 S, $17 N, $10 S, $20 N, $15 S, $25 N, $17 S, $27 N, $21 E, $22 W, $23 E, $24 W, $25 S, $35 N, $26 E, $27 W, $27 S, $37 N, $30 E, $31 W, $33 E, $34 W, $36 E, $37 W, $43 S, $53 N, $45 S, $55 N, $54 E, $55 W, $55 S, $65 N, $55 E, $56 W) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (room and side that has it) | 14 ($01 S, $12 N, $11 S, $21 S, $31 S, $32 S, $42 N, $41 S, $42 S, $52 N, $50 E, $51 W, $66 N, $66 W) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (room and side that has it) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $01 S shutter / $11 open, $02 S open / $12 shutter, $11 S shutter / $21 open, $21 S shutter / $31 open, $31 S shutter / $41 open, $41 S shutter / $51 open, $56 S open / $66 shutter, $65 E open / $66 shutter | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $76 S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 11 ($03, $04, $05, $07, $10, $20, $30, $37, $55, $61, $74) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 10 ($01, $13, $15, $24, $33, $41, $42, $50, $57, $65) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 4 ($47, $56, $57, $61) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | bomb ($00): $11, $16, $23, $25, $37; silver arrow ($09): $4F (cellar); Triforce of Power ($0E): $42; 5 rupees ($0F): $12, $15, $26, $34, $40, $44, $62; red ring ($13): $00 (cellar); compass ($16): $35; map ($17): $27 | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md) (rows with `lvl` = `9` or `c9`).
<!-- /generated:facts -->

- Entrance: overworld room `$05` has a rock wall tile object at (`$50`,`$A0`) whose secret works in Q1 only; a bomb in its blast state with its midpoint less than `$10` pixels from the wall's on both axes turns it into a cave mouth, and room flag `$80` keeps it open for good ([`levelblock-ow.md`](../data/levelblock-ow.md) row `$05`). [`RockWall_Update`](../src/bank4/rock-wall-tree-secrets.asm#L214), [`Secret_IsForOtherQuest`](../src/bank4/rock-wall-tree-secrets.asm#L331), [`Secret_RevealAndFlagTileObj`](../src/bank4/rock-wall-tree-secrets.asm#L287)
- Room `$05` makes six monsters (two blue Lynels, two red Lynels, a Blue Leever, a Red Leever); standing on the opened mouth with X = `$50`, Y = `$9D` loads level 9 (cave index 9). [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
- Link starts in room `$76` at X = `$78`, Y = `$DD`, facing up. [`LevelInfo_StartY`](../src/Variables.inc#L717), [`Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink`](../src/bank5/mode3-mode6-init.asm#L129)
- The south door of `$76` leads off the map: walking through it loads the overworld, and Link appears in room `$05` at (`$50`,`$9D`). [`Room_CalcNextRoomForDoor`](../src/bank5/next-room-calc.asm#L76), [`Mode_LeaveLevelToOW`](../src/bank5/next-room-calc.asm#L107)
- Nothing in this level sends Link back out: there is no Triforce piece, so the end-of-level sequence never runs. The game ends in room `$32`: when Link's X is `$70`-`$80` and his Y is exactly `$95`, he is halted, and `$80` frames later the game mode becomes `$13` (win game). [`Zelda_Update`](../src/bank4/zelda-guard-fire.asm#L42), [`Zelda_State1WaitForEnding`](../src/bank4/zelda-guard-fire.asm#L90), [`Mode13WinGame_Update`](../src/bank2/mode13-win-game.asm#L51)
- The compass marker points at Zelda's room `$32`, and in level 9 it always blinks. [`StatusBar_UpdatePositionMarker`](../src/bank1/map-marker-curtain.asm#L72)
- Leaving and coming back: unlocked key doors and bombed walls stay open and taken room items stay taken (room flags); shutters are closed again and every push block is back in place (both are reset on each room entry). [`Door_AddRoomFlagsToOpenedDoors`](../src/bank5/door-flags-dark-room.asm#L84), [`RoomFlags_SetItemTaken`](../src/bank1/take-item.asm#L140), [`Room_EnterCommon`](../src/bank5/enter-room.asm#L143)
- Leaving and coming back, monsters: the per-room kill counts are zeroed when the level is loaded, so every room is full again, except that a killed Patra or Ganon (non-recurring types, flagged "all dead") never returns; details in [mechanics.md](../mechanics.md) "Monster spawning, edge spawns and respawn (room history)". [`Mode2Load_Init`](../src/bank7/init-mode-dispatch.asm#L147), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)

# What Link must bring

- **All eight Triforce pieces**, for room `$66`: the man there (person type `$4B`) opens the room's two shutters and disappears only when [`InvTriforce`](../src/Variables.inc#L597) = `$FF`; otherwise he stays, the shutters stay shut, and Link cannot walk above Y = `$8D`. [`Person_InitUnderworldLevel9`](../src/bank1/underworld-person-init.asm#L172), [`Walker_Move@CheckPersonBlocking`](../src/bank7/walker-move.asm#L149)
- **Bombs: 6** (one for the overworld rock wall, five for the bombable walls of the route: `$65` N, `$06` W, `$20` N, `$31` W, `$04` W). No monster of the level needs a bomb. A new file holds at most 8; the route passes one bomb pickup (+4) in room `$16` if its Patra is killed. The red ring detour costs 3 more. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`InvMaxBombs`](../src/Variables.inc#L607), [items.md](../items.md) "Bombs"
- **The magic key, or 3 keys.** The route opens three key doors (`$14` E, `$16` N, `$63` W) before it reaches its first key (room `$61`). With the magic key (level 8, see [level-8.md](level-8.md)) no key is spent. With 2 keys the variant through room `$56` works (it yields a key before the first door). With no key at all the level can still be done with the ladder by a longer way (see "Route", key balance). [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`InvMagicKey`](../src/Variables.inc#L571), [`InvKeys`](../src/Variables.inc#L584)
- **Bow, and at least 1 rupee per arrow**, for Ganon in room `$42`: the arrow is refused without [`InvBow`](../src/Variables.inc#L560) or with 0 rupees, and each shot costs 1 rupee. The bow is in level 1 ([level-1.md](level-1.md)); the silver arrow itself is found here. [`Arrow_Wield`](../src/bank5/wield-items.asm#L187), [`InvRupees`](../src/Variables.inc#L583)
- **A sword** (any grade): Ganon must first be hit 15 times with the wooden sword, 8 with the white or 4 with the magic sword; Patras take only sword damage; Wizzrobes take only the sword, the sword beam and bomb blasts. [`Ganon_CheckCollisions`](../src/bank4/ganon-draw-collisions.asm#L360), [`Sword_DamagePoints`](../src/bank1/sword-bomb-arrow-collision.asm#L182), [`Wizzrobe_DrawAndCheckCollisions`](../src/bank4/wizzrobe.asm#L683)
- **Ladder: used once, not strictly required.** In room `$65` a full row of water lies between the doors and the north wall; from below the water a bomb lands at Y = `$75`, `$18` pixels from the wall's hotspot at Y = `$5D`, and the test needs less than `$18`. Without the ladder go `$65` -> `$64` -> `$54` and bomb the east wall of `$54` instead (two more rooms). The ladder is the item of level 4 ([level-4.md](level-4.md)). [`Link_EndMoveAndAnimate@SetUpLadder`](../src/bank7/link-animate-ladder.asm#L246), [`Bomb_WallHotspotsY`](../src/bank7/bomb.asm#L64), [`Link_PlaceWeaponAhead`](../src/bank1/bomb-candle-wield.asm#L154)
- **Candle: not needed.** Ten rooms are dark (on the route `$65`, `$41`, `$42`), but the dark flag is read only by the palette fades of the room change and by the candle's room light; no movement, collision or monster code reads it. Ganon's room is lit by his own entry scene. [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124), [`Candle_UpdateRoomLight`](../src/bank4/shoot-helpers-candle.asm#L132), [`Ganon_ScenePhase0_LiftInDark`](../src/bank4/ganon.asm#L71)
- **Raft, recorder, food: not needed.** The level has no hungry Goriya and no Digdogger, and its room list holds only the monsters of the table under "Monsters and boss". [`Level_BlockUW7to9Q1`](../src/bank6/level-blocks-common-data.asm#L74)
- **Hearts and protection: no rule demands any, but the damage is the highest in the game** (before rings): Ganon's touch and a Red Wizzrobe's shot 4 hearts; Blue Wizzrobe (touch and shot), Patra and Lanmola 2 hearts. The blue ring halves it, the red ring quarters it; the magic shield blocks Wizzrobe shots while Link faces them and is not attacking, and a Like-Like removes it after holding Link 95 frames. [`Damage_ObjTypeDamageToLink`](../src/bank1/monster-collisions.asm#L200), [`Link_BeHarmed`](../src/bank1/link-collision-harm.asm#L193), [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](../src/bank1/link-collision-harm.asm#L138), [`LikeLike_Update@IncCaptureTime`](../src/bank4/like-like.asm#L111)

# What Link gets here

## Silver arrow (cellar `$4F`, under room `$10`)

- Where: item id `$09`, the room item of treasure cellar `$4F`, at (`$80`,`$90`); the cellar's stairs are in room `$10`, whose only door is the bombable south wall shared with room `$20`. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
- How: in `$10` kill the 3 Blue and 2 Red Wizzrobes (the 3 Bubbles do not count), then push the block at X = `$C0`, Y = `$90` (stand at X = `$D0`, Y = `$8D` and hold Left for `$10` frames); stairs appear at (`$D0`,`$60`); stand on them at X = `$D0`, Y = `$5D`. [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`Secret_Trigger5BlockStairs`](../src/bank5/underworld-secrets.asm#L226), [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124)
- **What it changes:** [`InvArrow`](../src/Variables.inc#L559) becomes 2 and is never lowered again. An arrow now does damage `$40` instead of `$20`, and it is the only thing that kills Ganon: his arrow test runs only in his brown phase and only when `InvArrow` = 2. Shooting still needs the bow and 1 rupee per arrow, with the arrow selected as the B item. See [items.md](../items.md) "Arrows and Bow (wooden, silver)". [`Item_TakeClass2Grade`](../src/bank1/take-item.asm#L536), [`Arrow_CheckArrowOrRodVsMonster`](../src/bank1/sword-bomb-arrow-collision.asm#L249), [`Ganon_CheckCollisions@CheckArrowCollision`](../src/bank4/ganon-draw-collisions.asm#L427)
- First needed: in this level, room `$42` (Ganon). Nothing later in Q1 needs it: the game ends two rooms on. [`Ganon_CheckCollisions`](../src/bank4/ganon-draw-collisions.asm#L360)

## Red ring (cellar `$00`, under room `$07`)

- Where: item id `$13`, the room item of treasure cellar `$00`, at (`$80`,`$90`); its stairs are in room `$07` (not shown on the map), reached only through the bombable walls `$26` E, `$27` N and `$17` N. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`Door_TouchBombable`](../src/bank5/touch-door.asm#L108)
- How: in `$07` kill the 3 Blue and 2 Red Wizzrobes (3 Bubbles do not count), push the block at X = `$60`, Y = `$90` up or down (stand at X = `$60`, Y = `$9D`, hold Up), walk in from the left to the stairs at X = `$80`, Y = `$8D`. [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234)
- **What it changes:** [`InvRing`](../src/Variables.inc#L569) becomes 2: every amount of damage to Link is shifted right twice, a quarter of the listed value (Ganon's touch 1 heart instead of 4, a Red Wizzrobe's shot 1 instead of 4, a Patra 1/2 instead of 2). Knockback and the invincibility time are unchanged. See [items.md](../items.md) "Ring (blue, red)". [`Link_BeHarmed`](../src/bank1/link-collision-harm.asm#L193), [`Item_TakeClass2Grade`](../src/bank1/take-item.asm#L536)
- First useful: at once, in every room after it. It opens nothing and no code tests for it apart from the damage shift and the tunic colour; it is not on the route (see "Optional detours"). [`Link_BeHarmed`](../src/bank1/link-collision-harm.asm#L193)

## Triforce of Power (room `$42`)

- Where: item id `$0E`, the room item of Ganon's room, hidden until Ganon dies; it appears at his ashes `$A0` frames after the silver arrow hits. [`Ganon_ActivateRoomItem`](../src/bank4/ganon-draw-collisions.asm#L541), [`Ganon_UpdateDying`](../src/bank4/ganon.asm#L271)
- **What it changes:** taking it halts Link for `$C0` frames of fanfare; when they end [`InvPowerTriforce`](../src/Variables.inc#L598) = 1, and the secret trigger of room `$42` (type 3) opens both of its shutters. Only then can Link leave the room, north to Zelda. It refills no hearts. See [items.md](../items.md) "Triforce pieces and the Triforce of Power". [`Item_UpdatePowerTriforceFanfare`](../src/bank1/tile-block-power-triforce.asm#L99), [`Secret_Trigger3PowerTriforce`](../src/bank5/underworld-secrets.asm#L255)

## Keys (4)

- Room `$61` (on the route): lies in the open at (`$D0`,`$60`), the top right corner; no kill needed. Room `$47`: the same spot, also in the open. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- Room `$56`: appears at (`$80`,`$90`) when the 2 Zols and 2 Like-Likes are dead; the spot lies between two water rows (ladder from the north or south door, none from the west wall). Room `$57` (dark): appears at (`$80`,`$90`) inside a ring of water when the 3 Blue and 2 Red Wizzrobes are dead; the ladder is needed. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Link_EndMoveAndAnimate@SetUpLadder`](../src/bank7/link-animate-ladder.asm#L246)
- **What it changes:** each key adds 1 to [`InvKeys`](../src/Variables.inc#L584), which opens one key door (16 doorways in this level); with the magic key they are never spent. See [items.md](../items.md) "Keys and Magic Key". [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337), [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)

## Map (room `$27`) and compass (room `$35`)

- Map: appears at (`$80`,`$90`) in `$27` when its Patra is dead. Compass: appears at (`$80`,`$90`) in `$35` (not shown on the map) when its 3 Blue and 2 Red Wizzrobes are dead. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- **What it changes:** they set [`InvMap9`](../src/Variables.inc#L577) and [`InvCompass9`](../src/Variables.inc#L576) (level 9 has its own bytes): the status bar and submenu maps are drawn, and the marker for room `$32` is shown. Nothing else reads them. See [items.md](../items.md) "Map and Compass". [`Item_TakeClass0PerLevel`](../src/bank1/take-item.asm#L421)

## Bombs and rupees

- Bomb room items (+4 each, up to the capacity): `$16` (on the route; appears when the Patra is dead), `$11`, `$25`, `$37` (each appears when the room is cleared), `$23` (in the open at (`$D0`,`$60`)). [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337)
- 5-rupee room items, all appearing when the room is cleared: `$62` (on the route: 8 Keese) and `$15` (on the route, but inside a ring of water: ladder), `$12`, `$26`, `$34`, `$40`, `$44`. Worth it only for arrows: one rupee is one shot at Ganon. [`Arrow_Wield`](../src/bank5/wield-items.asm#L187)
- No heart container lies in level 9, and Ganon drops nothing but the Triforce of Power. [`Ganon_UpdateDying`](../src/bank4/ganon.asm#L271)

# Map

<!-- generated:map -->
```
         [01  ] K [02  ]   [03  ] X [04  ]   [05  ] X [06  ]   [07h ]
          S/|      |/S                                  K        X
[10  ]   [11i ]   [12i ] K [13  ]   [14  ] K [15i ]---[16i ]   [17  ]
  X       S/|       K        K        |        X        |        X
[20  ]   [21  ] X [22  ] K [23i ] X [24  ]   [25i ] K [26i ] X [27i ]
          S/|                K        |        X        K        X
[30  ] X [31  ]   [32Th]   [33  ] X [34i ]   [35hi]   [36  ] X [37i ]
  K       S/|       S                                            |
[40i ]   [41  ]   [42Bi]   [43  ]   [44i ]---[45  ]---[46  ]   [47k ]
  K       S/|       S        X        |        X        |        |
[50  ] S [51  ]   [52  ]   [53  ]   [54  ] X [55  ] X [56k ]   [57k ]
           |                 K        |        X       |/S
         [61k ]---[62i ] K [63  ]   [64  ]---[65  ]-/S[66  ]
           |                 K                          |
         [71  ]            [73  ] K [74  ]            [76S ]

[73S] = room $73 and its marks: S start room, B boss room, T room the compass marks (no Triforce piece),
        k a key lies here, i another room item lies here, h not shown on the submenu map.
Between two rooms, the door: --- or | open, K key door, S shutter, X bombable wall,
        F false wall (walk-through), nothing = wall. A/B = the two sides differ: the left
        or upper room's side first, . = wall on that side.
Doors on sides with no room of the level: $76 S open.
```

- Cellar `$60` (tunnel cellar): joins `$14` (left exit) and `$55` (right exit), whose stairs lead here; item: none. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)

- Cellar `$70` (tunnel cellar): joins `$63` (left exit) and `$05` (right exit), whose stairs lead here; item: none. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)

- Cellar `$72` (tunnel cellar): joins `$71` (left exit) and `$74` (right exit), whose stairs lead here; item: none. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)

- Cellar `$75` (tunnel cellar): joins `$20` (left exit) and `$61` (right exit), whose stairs lead here; item: none. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)

- Cellar `$67` (tunnel cellar): joins `$30` (left exit) and `$04` (right exit), whose stairs lead here; item: none. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)

- Cellar `$77` (tunnel cellar): joins `$52` (left exit) and `$03` (right exit), whose stairs lead here; item: none. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)

- Cellar `$00` (treasure cellar): both exits lead to `$07`, whose stairs lead here; item: red ring ($13). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)

- Cellar `$4F` (treasure cellar): both exits lead to `$10`, whose stairs lead here; item: silver arrow ($09). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)
<!-- /generated:map -->

# Rooms

Generated from the level data by `tools/romdata.py`; do not edit between the markers.

<!-- generated:rooms -->
| room | N | S | W | E | monsters | item | secret | block | dark | layout | floor | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $01 | wall | shutter | wall | key | 3 x Bubble ($2B), 2 x Zol ($13), 3 x Keese blue ($1B) | - | - | - | D | $13 | water | - |
| $02 | wall | open | key | wall | 1 x dungeon person ($4E) | - | - | - | - | $26 | blank | person $4E says: EYES OF SKULL / HAS A SECRET. |
| $03 | wall | wall | wall | bomb | 2 x Bubble ($2B), 2 x Zol ($13), 2 x Like-Like ($17) | - | - | B | - | $1A | blocks, stairs | stairs lead to cellar $77 |
| $04 | wall | wall | bomb | wall | 1 x Trap ($4A), 3 x (empty) ($00), 2 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | - | B | - | $1A | blocks, stairs | stairs lead to cellar $67 |
| $05 | wall | wall | wall | bomb | 3 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | push block: stairs appear | B | - | $0A | blocks | stairs lead to cellar $70 |
| $06 | wall | key | bomb | wall | 1 x dungeon person ($4D) | - | - | - | - | $26 | blank | person $4D says: GO TO THE NEXT ROOM. |
| $07 off-map | wall | bomb | wall | wall | 3 x Bubble ($2B), 3 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | - | B | - | $1A | blocks, stairs | stairs lead to cellar $00 |
| $10 | wall | bomb | wall | wall | 3 x Bubble ($2B), 3 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | push block: stairs appear | B | - | $08 | blocks | stairs lead to cellar $4F |
| $11 | open | shutter | wall | wall | 1 x Lanmola ($3B) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $25 | sand | - |
| $12 | shutter | key | wall | key | 1 x Lanmola ($3A) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $26 | blank | - |
| $13 | wall | key | key | wall | 2 x Bubble ($2B), 2 x Zol ($13), 2 x Like-Like ($17) | - | - | - | D | $00 | plain | - |
| $14 | wall | open | wall | key | 5 x Like-Like ($17) | - | - | - | - | $1C | blocks, stairs | stairs lead to cellar $60 |
| $15 | wall | bomb | key | open | 3 x Blue Wizzrobe ($23) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | D | $14 | water | - |
| $16 | key | open | open | wall | 1 x Patra ($48) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $0A | blocks | - |
| $17 | bomb | bomb | wall | wall | 1 x Bubble ($2B), 3 x Like-Like ($17), 2 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | - | - | - | $1F | blocks | - |
| $20 | bomb | wall | wall | wall | 3 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | - | B | - | $1A | blocks, stairs | stairs lead to cellar $75 |
| $21 | open | shutter | wall | bomb | 1 x Patra ($47) | - | - | - | - | $25 | sand | - |
| $22 | key | wall | bomb | key | 2 x Bubble ($2B), 2 x Zol ($13), 2 x Like-Like ($17) | - | - | - | - | $1D | blocks | - |
| $23 | key | key | key | bomb | 8 x Gel ($15) | bomb ($00) at $D0,$60 | - | - | - | $12 | water | - |
| $24 | open | open | bomb | wall | 5 x Vire ($12) | - | - | - | D | $17 | water | - |
| $25 | bomb | bomb | wall | key | 2 x Bubble ($2B), 2 x Zol ($13), 2 x Like-Like ($17) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $0E | blocks | - |
| $26 | open | key | key | bomb | 8 x Gel ($15) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $03 | blocks | - |
| $27 | bomb | bomb | bomb | wall | 1 x Patra ($48) | map ($17) at $80,$90 | all dead: shutters open, room item appears | - | - | $0A | blocks | - |
| $30 | wall | key | wall | bomb | 1 x Trap ($4A), 3 x (empty) ($00), 2 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | push block: stairs appear | B | - | $0A | blocks | stairs lead to cellar $67 |
| $31 | open | shutter | bomb | wall | 1 x Bubble ($2B), 3 x Like-Like ($17), 2 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | - | - | - | $25 | sand | - |
| $32 compass marker off-map | wall | shutter | wall | wall | 1 x Zelda ($37) | - | all dead: shutters open | - | - | $27 | blocks, statues, blank | - |
| $33 | key | wall | wall | bomb | 8 x Gel ($15) | - | - | - | D | $19 | water | - |
| $34 | open | wall | bomb | wall | 8 x Keese blue ($1B) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $14 | water | - |
| $35 off-map | bomb | wall | wall | wall | 3 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | compass ($16) at $80,$90 | all dead: shutters open, room item appears | - | - | $26 | blank | - |
| $36 | key | wall | wall | bomb | 3 x Bubble ($2B), 2 x Zol ($13), 3 x Keese blue ($1B) | - | - | - | - | $00 | plain | - |
| $37 | bomb | open | bomb | wall | 6 x Vire ($12) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | B | - | $23 | statues | - |
| $40 | key | key | wall | wall | 3 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $16 | water | - |
| $41 | open | shutter | wall | wall | 1 x Trap ($4A), 3 x (empty) ($00), 4 x Like-Like ($17) | - | - | - | D | $00 | plain | boss sound $10 |
| $42 boss | shutter | shutter | wall | wall | 1 x Ganon ($3E) | Triforce of Power ($0E) at $80,$90 | last boss dead: shutters open | - | D | $28 | statues, blank | - |
| $43 | wall | bomb | wall | wall | 1 x dungeon person ($4C) | - | - | - | - | $26 | blank | person $4C says: PATRA HAS THE MAP.; boss sound $10 |
| $44 | wall | open | wall | open | 6 x Red Wizzrobe ($24) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $45 | wall | bomb | open | open | 1 x Bubble ($2B), 3 x Like-Like ($17), 2 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | - | - | - | $00 | plain | - |
| $46 | wall | open | open | wall | 6 x Blue Wizzrobe ($23) | - | - | - | - | $25 | sand | - |
| $47 | open | open | wall | wall | 3 x Bubble ($2B), 3 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | key ($19) at $D0,$60 | - | - | - | $1E | blocks | - |
| $50 | key | wall | wall | shutter | 5 x Zol ($13) | - | all dead: shutters open | - | D | $14 | water | - |
| $51 | open | open | shutter | wall | 6 x Like-Like ($17) | - | all dead: shutters open | - | - | $11 | blocks | - |
| $52 | shutter | wall | wall | wall | 1 x Patra ($47) | - | all dead: shutters open | - | - | $1B | blocks, stairs | stairs lead to cellar $77; boss sound $10 |
| $53 | bomb | key | wall | wall | 1 x Bubble ($2B), 3 x Like-Like ($17), 2 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | - | - | - | $23 | statues | - |
| $54 | open | open | wall | bomb | 3 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | - | - | - | $0C | blocks | - |
| $55 | bomb | bomb | bomb | bomb | 1 x Lanmola ($3A) | - | - | B | - | $1A | blocks, stairs | stairs lead to cellar $60 |
| $56 | open | open | bomb | wall | 2 x Bubble ($2B), 2 x Zol ($13), 2 x Like-Like ($17) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $19 | water | - |
| $57 | open | wall | wall | wall | 3 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | D | $14 | water | - |
| $61 | open | open | wall | open | 1 x Patra ($47) | key ($19) at $D0,$60 | - | B | - | $1A | blocks, stairs | stairs lead to cellar $75 |
| $62 | wall | wall | open | key | 8 x Keese blue ($1B) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $26 | blank | - |
| $63 | key | key | key | wall | 5 x Zol ($13) | - | - | - | - | $1B | blocks, stairs | stairs lead to cellar $70 |
| $64 | open | wall | wall | open | 1 x Bubble ($2B), 3 x Like-Like ($17), 2 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | - | - | - | $1F | blocks | - |
| $65 | bomb | wall | open | open | 3 x Bubble ($2B), 3 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | - | - | D | $18 | water | - |
| $66 | shutter | open | shutter | wall | 1 x dungeon person ($4B) | - | - | - | - | $26 | blank | person $4B says: ONES WHO DOES NOT HAVE / TRIFORCE CAN'T GO IN. |
| $71 | open | wall | wall | wall | none | - | - | - | - | $1B | blocks, stairs | stairs lead to cellar $72 |
| $73 | key | wall | wall | key | 3 x Bubble ($2B), 3 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | - | - | - | $1F | blocks | - |
| $74 | wall | wall | key | wall | 1 x Lanmola ($3B) | - | - | B | - | $1A | blocks, stairs | stairs lead to cellar $72 |
| $76 start | open | open | wall | wall | none | - | - | - | - | $21 | statues, sand | - |
| $60 cellar | - | - | - | - | 4 x Keese blue ($1B) | - | - | - | - | $3E | tunnel cellar | left exit to $14, right exit to $55; Link comes out at $60,$9D |
| $70 cellar | - | - | - | - | 4 x Keese blue ($1B) | - | - | - | - | $3E | tunnel cellar | left exit to $63, right exit to $05; Link comes out at $A0,$9D |
| $72 cellar | - | - | - | - | 4 x Keese blue ($1B) | - | - | - | - | $3E | tunnel cellar | left exit to $71, right exit to $74; Link comes out at $60,$9D |
| $75 cellar | - | - | - | - | 4 x Keese blue ($1B) | - | - | - | - | $3E | tunnel cellar | left exit to $20, right exit to $61; Link comes out at $60,$9D |
| $67 cellar | - | - | - | - | 4 x Keese blue ($1B) | - | - | - | - | $3E | tunnel cellar | left exit to $30, right exit to $04; Link comes out at $C0,$6D |
| $77 cellar | - | - | - | - | 4 x Keese blue ($1B) | - | - | - | - | $3E | tunnel cellar | left exit to $52, right exit to $03; Link comes out at $60,$9D |
| $00 cellar | - | - | - | - | 4 x Keese blue ($1B) | red ring ($13) at $80,$90 | - | - | - | $3F | treasure cellar | left exit to $07, right exit to $07; Link comes out at $60,$9D |
| $4F cellar | - | - | - | - | 4 x Keese blue ($1B) | silver arrow ($09) at $80,$90 | - | - | - | $3F | treasure cellar | left exit to $10, right exit to $10; Link comes out at $60,$9D |
<!-- /generated:rooms -->

# Floor plans

<!-- generated:floors -->
One character per floor square, 12 columns x 7 rows inside the walls (left column = west, top row =
north): `.` floor (tile `$74`), `B` block (tile `$B0`; a push block object is placed on one in a room
with a push block), `W` water (tile `$F4`), `S` statue (tiles `$94` and `$B4`), `>` stairs (tile `$70`),
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L63), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L73), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
Stairs that a secret makes appear are not part of the layout.

```
layout $00  rooms: $13,$36,$41,$44,$45
............
............
............
............
............
............
............
```

```
layout $03  rooms: $26
............
..B......B..
............
............
............
..B......B..
............
```

```
layout $08  rooms: $10
............
.BBBBBBBBBB.
..........B.
..........B.
..........B.
.BBBBBBBBBB.
............
```

```
layout $0A  rooms: $05,$16,$27,$30
............
............
............
....B..B....
............
............
............
```

```
layout $0C  rooms: $54
.B.B........
.B.BBB.BBB..
.B...B...BBB
.B.B.B...B..
.BBB.B...B..
.....B.B.B.B
..B....B...B
```

```
layout $0E  rooms: $25
....B..B....
....B..B....
....B..B....
....B..B....
....B..B....
....B..B....
....B..B....
```

```
layout $11  rooms: $51
..B........B
.B...B....B.
....B....B..
...B....B...
..B....B....
.B....B...B.
B........B..
```

```
layout $12  rooms: $23
............
.WWWWWWWWWW.
.WWWWWWWWWW.
.WW......WW.
.WW......WW.
.WWWW..WWWW.
.WWWW..WWWW.
```

```
layout $13  rooms: $01
........W...
........W...
........W...
........W...
........W...
........W...
........W...
```

```
layout $14  rooms: $15,$34,$50,$57
............
.WWWWWWWWWW.
.W........W.
.W........W.
.W........W.
.WWWWWWWWWW.
............
```

```
layout $16  rooms: $40
WWWWW..WWWWW
W...WWWW...W
W.WWW..WWW.W
..W......W..
W.WWW..WWW.W
W...WWWW...W
WWWWW..WWWWW
```

```
layout $17  rooms: $24
WWWWW..WWWWW
W...W....W.W
W.W.W.WW.W.W
..W.W..W.W..
W.W.WW.W.W.W
W.W....W...W
WWWWW..WWWWW
```

```
layout $18  rooms: $65
............
............
WWWWWWWWWWWW
............
............
............
............
```

```
layout $19  rooms: $33,$56
............
WWWWWWWWWWWW
............
............
............
WWWWWWWWWWWW
............
```

```
layout $1A  rooms: $03,$04,$07,$20,$55,$61,$74
............
......B.....
.....B.B....
....B.>.B...
.....B.B....
......B.....
............
```

```
layout $1B  rooms: $52,$63,$71
........BBBB
........BBBB
.........BBB
...........>
.........BBB
........BBBB
........BBBB
```

```
layout $1C  rooms: $14
..........B.
.BBBBBBBB.B.
.B.....>B.B.
.B...BBBB.B.
.B........B.
.BBBBBBBBBB.
............
```

```
layout $1D  rooms: $22
............
............
..BB....BB..
..BB....BB..
..BB....BB..
............
............
```

```
layout $1E  rooms: $47
............
............
.....BB.....
.....BB.....
.....BB.....
............
............
```

```
layout $1F  rooms: $17,$64,$73
............
.BB......BB.
............
.....BB.....
............
.BB......BB.
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
layout $23  rooms: $37,$53
............
............
............
....S..S....
............
............
............
```

```
layout $25  rooms: $11,$21,$31,$46
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
```

```
layout $26  rooms: $02,$06,$12,$35,$43,$62,$66
____________
____________
____________
____________
____________
____________
____________
```

```
layout $27  rooms: $32
BBBBBBBBBBBB
B...B__B...B
B.BB____BB.B
B.BB____BB.B
B...S..S...B
B..........B
B.S......S.B
```

```
layout $28  rooms: $42
S__......__S
__.._.._..__
__.__..__.__
__.__..__.__
__...__...__
____....____
S___....___S
```

```
layout $3E (tunnel cellar)  rooms: $60,$70,$72,$75,$67,$77
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
layout $3F (treasure cellar)  rooms: $00,$4F
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

HP is in the units of weapon damage: wooden sword `$10`, white `$20`, magic `$40`, bomb `$40`, silver arrow `$40`.
Damage to Link is before rings. Sections named are in [enemies.md](../enemies.md).

| Monster (type) | Rooms (count) | HP | What kills or stuns it | What it does to Link | Section |
| --- | --- | --- | --- | --- | --- |
| Blue Wizzrobe (`$23`) | `$04` 2, `$05` 3, `$07` 3, `$10` 3, `$15` 3, `$17` 2, `$20` 3, `$30` 2, `$31` 2, `$35` 3, `$40` 3, `$45` 2, `$46` 6, `$47` 3, `$53` 2, `$54` 3, `$57` 3, `$64` 2, `$65` 3, `$73` 3 | `$A0` | sword (10 / 5 / 3 hits), sword beam, bomb blast (3); nothing else, no stun, no knockback | touch 2 hearts; magic shot 2 hearts along its row or column (magic shield blocks) | Wizzrobe (blue, red) |
| Red Wizzrobe (`$24`) | `$04`, `$05`, `$07`, `$10`, `$17`, `$20`, `$30`, `$31`, `$35`, `$40`, `$45`, `$47`, `$53`, `$54`, `$57`, `$64`, `$65`, `$73` 2 each; `$44` 6 | `$40` | sword (4 / 2 / 1), sword beam, bomb blast (1); only while solid or fading | touch 1 heart; magic shot 4 hearts (magic shield blocks); appears 2-5 squares from Link, facing him | Wizzrobe (blue, red) |
| Like-Like (`$17`) | `$03` 2, `$13` 2, `$14` 5, `$17` 3, `$22` 2, `$25` 2, `$31` 3, `$41` 4, `$45` 3, `$51` 6, `$53` 3, `$56` 2, `$64` 3 | `$90` | any weapon (sword 9 / 5 / 3, bomb 3); boomerang stuns | touch 1 heart, then holds Link; magic shield gone after 95 frames held | Like-Like |
| Zol (`$13`) | `$01` 2, `$03` 2, `$13` 2, `$22` 2, `$25` 2, `$36` 2, `$50` 5, `$56` 2, `$63` 5 | `$20` | damage `$20` or more kills; the wooden sword splits it into 2 Gels; boomerang stuns | touch 1 heart | Zol |
| Gel (`$15`) | `$23` 8, `$26` 8, `$33` 8 | 0 | any hit, the boomerang included | touch 1/2 heart | Gel |
| Keese, blue (`$1B`) | `$01` 3, `$34` 8, `$36` 3, `$62` 8; 4 in every cellar | 0 | any hit, the boomerang included | touch 1/2 heart | Keese (blue, red, black) |
| Vire (`$12`) | `$24` 5, `$37` 6 | `$40` | magic sword, bomb or silver arrow kill outright; a weaker hit splits it into 2 red Keese | touch 1 heart | Vire |
| Bubble (`$2B`) | `$01` 3, `$03` 2, `$07` 3, `$10` 3, `$13` 2, `$17` 1, `$22` 2, `$25` 2, `$31` 1, `$36` 3, `$45` 1, `$47` 3, `$53` 1, `$56` 2, `$64` 1, `$65` 3, `$73` 3 | - | nothing | no damage; knockback, and the sword is dead for 151-160 frames | Bubble (flashing, red, blue) |
| Trap (`$4A`: four, one per corner) | `$04`, `$30`, `$41` | - | nothing | touch 1 heart; charges along its row or column when Link lines up | Trap |
| Lanmola, red (`$3A`) | `$12`, `$55` (two worms of 5 segments each) | `$20` per segment | any weapon; 5 segment kills per worm (wooden sword 2 hits each) | touch 2 hearts; 1 pixel per frame | Lanmola (red, blue) |
| Lanmola, blue (`$3B`) | `$11`, `$74` (two worms) | `$20` per segment | as red | touch 2 hearts; 2 pixels per frame | Lanmola (red, blue) |
| Patra (`$47`, children `$25`) | `$21`, `$52`, `$61` | body `$B0`, 8 children `$60` each | sword and sword beam only; body cannot be hit while a child lives | touch of body or child 2 hearts | Patra |
| Patra (`$48`, children `$26`) | `$16`, `$27` | as above | as above | as above; smaller orbit that widens | Patra |
| Statue fireballs (`$55`) | `$37`, `$53` (layout `$23`, two statues) | - | nothing | 1/2 heart; magic shield blocks | Statue fireballs |
| Dungeon persons (`$4B`-`$4E`) | `$66`, `$06`, `$02`, `$43` | - | cannot die; a hit turns on fireballs | none | Old men and shopkeepers |
| Ganon (`$3E`) | `$42` | `$F0` sword count, then the silver arrow | see below | touch 4 hearts; fireball 1 heart, never blocked | Ganon |
| Zelda (`$37`) and 4 guard fires (`$3F`) | `$32` | fires `$10` | any damaging weapon kills a fire | fire touch 1/2 heart | Zelda, Guard fire |

- Rooms the route must clear (their block does not move until no counted monster is left; Bubbles and Traps are not counted): `$55` (both Lanmola worms), `$05` (3 Blue + 2 Red Wizzrobes), `$61` (Patra), `$20` (3 Blue + 2 Red Wizzrobes), `$10` (3 Blue + 2 Red), `$30` (2 Blue + 2 Red), `$03` (2 Zols and their Gels, 2 Like-Likes), and `$52` (Patra, for its shutter). [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124), [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74)
- Wizzrobes: arrows, boomerang and the rod pass through them; only the sword, the sword beam (full hearts) and a bomb blast hurt. A Blue Wizzrobe shoots only along the row or column it shares with Link while facing him; a Red one shoots once per appearance, 16 frames after it turns solid, in the direction it faces. Stay off their lines, or face the shot with the magic shield while not swinging the sword or using an item. [`Wizzrobe_DrawAndCheckCollisions`](../src/bank4/wizzrobe.asm#L683), [`Wizzrobe_Blue_TryShooting`](../src/bank4/wizzrobe.asm#L469), [`Wizzrobe_Red_State2Solid`](../src/bank4/wizzrobe.asm#L667), [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](../src/bank1/link-collision-harm.asm#L138)
- Patra (`$61`, `$52`; `$16` only passed through): it starts at (`$80`,`$70`); the children appear one at a time and cannot be hit until the last one is out; kill all eight children with the sword, then the body (11 / 6 / 3 hits). A dead Patra never comes back. [`Patra_Init`](../src/bank4/patra-ganon-inits.asm#L42), [`Patra_Update@LoopChildren`](../src/bank4/patra.asm#L77), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- Lanmola (`$55`): both worms start at (`$40`,`$8D`); every kill removes the tail and the head dies last, so ten segment kills clear the room. The room is never saved as cleared: on every later entry the worms are back. [`Lanmola_Init`](../src/bank4/zelda-lanmola-inits.asm#L75), [`Lanmola_Update@FindTail`](../src/bank4/lanmola.asm#L116), [`RoomHistory_SaveKillCountUW`](../src/bank5/room-history-uw.asm#L110)
- Like-Likes (`$14`, `$51`, `$41`, `$31`, `$03`): a touch that harms Link also holds him; he can still use the sword and the B item. Kill the holder within 95 frames to keep the magic shield. [`LikeLike_Update`](../src/bank4/like-like.asm#L46), [`LikeLike_Update@IncCaptureTime`](../src/bank4/like-like.asm#L111)
- Bubbles: a touch costs no hearts but disables the A button for about 160 frames; in the Wizzrobe rooms that leaves only bombs. They never die and do not keep a block from moving. [`Bubble_Update`](../src/bank4/bubble-keese.asm#L93), [`Link_HandleInput`](../src/bank5/link-handle-input.asm#L68)
- Zols in `$03`: with the wooden sword each Zol becomes two Gels, which also must die before the block moves. [`Zol_CheckCollisions`](../src/bank4/zol-gel.asm#L170)

## Ganon (room `$42`)

- Entry scene: Link is halted for `$40` frames in the dark, the room fades to light, and control returns `$C0` frames later. Both shutters of the room are shut and stay shut until the Triforce of Power is taken. [`Ganon_Init`](../src/bank4/patra-ganon-inits.asm#L96), [`Ganon_ScenePhase0_LiftInDark`](../src/bank4/ganon.asm#L71), [`Ganon_ScenePhase1_LiftInLight`](../src/bank4/ganon.asm#L127)
- Sword phase (blue): Ganon is invisible, moves 1 pixel per frame turning toward Link at intervals, and shoots a fireball at Link every 64 frames. The sword hits him only while he is invisible and only at full extension; each hit shows him for `$40` frames, during which he cannot be hurt, then he jumps to Y = `$A0`, X = `$30` or `$B0` and vanishes again. [`Ganon_MoveAndShoot`](../src/bank4/ganon.asm#L195), [`Ganon_RandomizeLocation`](../src/bank4/ganon.asm#L180), [`Ganon_CheckCollisions@CheckHarmed`](../src/bank4/ganon-draw-collisions.asm#L408)
- Hits needed: 15 wooden, 8 white, 4 magic. The sword beam, bombs, the rod and every other weapon are never tested against him. His touch (within 9 pixels of his middle at X + `$10`, Y + `$10`) costs 4 hearts and is tested while he is invisible too. [`Ganon_CheckCollisions`](../src/bank4/ganon-draw-collisions.asm#L360), [`Sword_DamagePoints`](../src/bank1/sword-bomb-arrow-collision.asm#L182)
- Arrow phase (brown): after the last sword hit he stands still for about 510 frames (flickering for the last 96), does not shoot, and takes no sword hit. An arrow in flight that overlaps him kills him if [`InvArrow`](../src/Variables.inc#L559) = 2; a wooden arrow passes through. If the time runs out he turns blue again and the whole sword count starts over. [`Ganon_UpdateBrownState`](../src/bank4/ganon.asm#L234), [`Ganon_CheckCollisions@CheckArrowCollision`](../src/bank4/ganon-draw-collisions.asm#L427)
- After the arrow: `$50` frames later he bursts into ashes, at `$A0` frames the Triforce of Power appears there; taking it (`$C0` frames) opens the north and south shutters. He leaves no heart container and no drop. [`Ganon_UpdateDying`](../src/bank4/ganon.asm#L271), [`Ganon_ActivateRoomItem`](../src/bank4/ganon-draw-collisions.asm#L541), [`Secret_Trigger3PowerTriforce`](../src/bank5/underworld-secrets.asm#L255)
- Dead end: without the bow, the silver arrow and a rupee, nothing in room `$42` can open its shutters; Link can only die there (or use the controller 2 exit of the submenu). [`Arrow_Wield`](../src/bank5/wield-items.asm#L187), [`Door_TouchShutter`](../src/bank5/touch-door.asm#L126), [`Submenu_UpdateActive`](../src/bank5/submenu-states.asm#L301)

# Route

Established from the level data and the code; not walked in the running game. A search over the room graph
(door types, keys, shutter triggers, stairs and cellar links, floor plans with and without the ladder) gave the
path with the fewest room entries: 32 entries after the start room, 5 bombs inside the level, 3 key doors.
Positions are Link's X, Y. "Clear" means kill every monster except Bubbles and Traps. The keys column assumes 3
keys brought and no magic key.

| Step | Room | Entered from | What to do | Leave by | Keys after | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `$76` | start (`$78`,`$DD`) | walk up | N | 3 | no monsters; statues here do not shoot |
| 1 | `$66` | S | the man is gone (eight pieces) and both shutters open; keep clear of the two fires at (`$48`,`$80`) and (`$A8`,`$80`) | W shutter | 3 | without all pieces: dead end |
| 2 | `$65` | E | walk to X = `$78`, go up across the water row with the ladder; from Y = `$6D` or `$5D` face up and place a bomb; the wall opens 8 frames after the blast ends | N bombed wall | 3 | dark; 3 Bubbles, 3 Blue + 2 Red Wizzrobes (no kill needed); bomb 1 |
| 3 | `$55` | S | kill both red Lanmola worms; stand at (`$60`,`$9D`) and hold Up `$10` frames: the block slides up; walk up to Y = `$8D`, then right to the stairs at (`$80`,`$8D`) | stairs | 3 | |
| 4 | cellar `$60` | right ladder (X = `$C0`) | walk left, go up the left ladder (X = `$30`) | left ladder | 3 | 4 blue Keese |
| 5 | `$14` | cellar, at (`$60`,`$9D`) | Link is inside a spiral of blocks: east along Y = `$9D` to X = `$B0`, up to Y = `$5D`, west along the top row to X = `$20`, down the left lane to Y = `$BD`, east along the bottom row to X = `$D0`, up to Y = `$8D`, push east into the key door | E key door | 2 | 5 Like-Likes |
| 6 | `$15` | W | go round the ring of water along the top or bottom lane to the east door | E | 2 | dark; 3 Blue Wizzrobes |
| 7 | `$16` | W | cross to the north door (X = `$78`); the Patra starts just below it at (`$80`,`$70`); killing it makes a bomb pickup appear at (`$80`,`$90`) | N key door | 1 | Patra `$48` |
| 8 | `$06` | S | the man's text halts Link; then go left below the fire (Y = `$9D`) to X = `$30`, up to Y = `$8D`, face left, place a bomb | W bombed wall | 1 | Link cannot go above Y = `$8D` here; bomb 2 |
| 9 | `$05` | E | clear (3 Blue + 2 Red Wizzrobes); push the block at (`$60`,`$90`) from (`$60`,`$9D`) holding Up; stairs appear at (`$D0`,`$60`): stand at (`$D0`,`$5D`) | stairs | 1 | |
| 10 | cellar `$70` | right ladder | walk left, up the left ladder | left ladder | 1 | 4 blue Keese |
| 11 | `$63` | cellar, at (`$A0`,`$9D`) | walk to the west door | W key door | 0 | 5 Zols |
| 12 | `$62` | E | walk west; clearing the 8 Keese makes 5 rupees appear at (`$80`,`$90`) | W | 0 | |
| 13 | `$61` | E | take the key at (`$D0`,`$60`) (stand at (`$D0`,`$5D`)); kill the Patra; push the block from (`$60`,`$9D`) holding Up; stairs at (`$80`,`$8D`) | stairs | 1 | Patra `$47` |
| 14 | cellar `$75` | right ladder | walk left, up the left ladder | left ladder | 1 | 4 blue Keese |
| 15 | `$20` | cellar, at (`$60`,`$9D`) | clear now (3 Blue + 2 Red Wizzrobes) so the block moves on the way back; stand at (`$78`,`$5D`), face up, place a bomb | N bombed wall | 1 | bomb 3 |
| 16 | `$10` | S | clear (3 Blue + 2 Red Wizzrobes); go up the right-hand lane to (`$D0`,`$8D`), hold Left `$10` frames: the block at (`$C0`,`$90`) slides; stairs appear at (`$D0`,`$60`): stand at (`$D0`,`$5D`) | stairs | 1 | 3 Bubbles |
| 17 | cellar `$4F` | ladder | take the silver arrow at (`$80`,`$90`) | either ladder | 1 | 4 blue Keese; both ladders lead to `$10` |
| 18 | `$10` | cellar, at (`$60`,`$9D`) | Link is inside the C of blocks: leave it by its open west side, go to the south door | S | 1 | |
| 19 | `$20` | N | if the room is clear: stand at (`$60`,`$9D`), hold Up; stairs at (`$80`,`$8D`) | stairs | 1 | |
| 20 | cellar `$75` | left ladder | walk right, up the right ladder | right ladder | 1 | |
| 21 | `$61` | cellar, at (`$60`,`$9D`) | walk to the north door | N | 1 | the Patra stays dead |
| 22 | `$51` | S | walk north between the diagonal rows of blocks | N | 1 | 6 Like-Likes; have 2 bombs before going on |
| 23 | `$41` | S | walk straight north without stopping: the bottom and the top pair of Traps charge toward the middle as Link crosses their rows | N | 1 | dark; 4 Traps, 4 Like-Likes; the south shutter never opens again |
| 24 | `$31` | S | go to (`$30`,`$8D`), face left, place a bomb | W bombed wall | 1 | Bubble, 3 Like-Likes, 2 Blue + 2 Red Wizzrobes; bomb 4 |
| 25 | `$30` | E | move west at once (the east doorway is in the right-hand Traps' column); clear the 2 Blue + 2 Red Wizzrobes; push the block at (`$60`,`$90`) from (`$60`,`$9D`) holding Up; stairs appear at (`$D0`,`$60`), where the top right Trap rests: lure it out and step on at (`$D0`,`$5D`) while it creeps back | stairs | 1 | 4 Traps |
| 26 | cellar `$67` | left ladder | walk right, up the right ladder | right ladder | 1 | 4 blue Keese |
| 27 | `$04` | cellar, at (`$C0`,`$6D`) | cross to the west half along the top or bottom row (the Traps' rows: let them charge, pass while they creep back), go to (`$30`,`$8D`), face left, place a bomb; the west doorway is in the left-hand Traps' column | W bombed wall | 1 | 4 Traps, 2 Blue + 2 Red Wizzrobes; bomb 5 |
| 28 | `$03` | E | clear (2 Zols, 2 Like-Likes); go round to (`$60`,`$9D`), hold Up; stairs at (`$80`,`$8D`) | stairs | 1 | 2 Bubbles |
| 29 | cellar `$77` | right ladder | walk left, up the left ladder | left ladder | 1 | 4 blue Keese |
| 30 | `$52` | cellar, at (`$60`,`$9D`) | kill the Patra: the north shutter opens | N shutter | 1 | Patra `$47`; the only other way out is the stairs at (`$D0`,`$8D`) |
| 31 | `$42` | S | Ganon (see "Monsters and boss"): sword hits while he is invisible, then one silver arrow; take the Triforce of Power | N shutter | 1 | dark until the scene lights it |
| 32 | `$32` | S | kill the two fires at (`$70`,`$9D`) and (`$80`,`$9D`), walk up the middle (X = `$78`) to Y = `$95` | - | 1 | the game ends |

- Key balance: 3 doors on the route (`$14` E at step 5, `$16` N at step 7, `$63` W at step 11), 1 key on it (`$61`, step 13, after the last door). With the magic key nothing is spent. An opened door stays open from both sides. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`Door_Update`](../src/bank5/update-doors.asm#L60)
- Variant with 2 keys (same length, needs the ladder): from `$66` go north instead of west, cross the water row of `$56`, clear it (2 Zols, 2 Like-Likes) and take the key at (`$80`,`$90`), bomb its west wall from (`$30`,`$8D`), and enter `$55` from the east. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136)
- Variant without the ladder (2 more rooms): `$66` -> `$65` (west door) -> `$64` (north door) -> `$54`; there go round the blocks to (`$C0`,`$8D`), face right, bomb the east wall and enter `$55` from the west. [`Bomb_WallHotspotsX`](../src/bank7/bomb.asm#L60), [`Link_PlaceWeaponAhead`](../src/bank1/bomb-candle-wield.asm#L154)
- With no key and no magic key (ladder needed, 42 entries, 7 bombs): take the key of `$56` as above for `$14` E; from `$16` go `$26` -> `$27` (bomb `$26` E) -> `$37` (bomb `$27` S) -> `$47` (key in the open) -> `$57` (clear it, key inside the water ring) and back to `$16`; those two keys open `$16` N and `$63` W. Without the ladder the same search finds no route with 0 keys. [`InvKeys`](../src/Variables.inc#L584), [`Door_TouchBombable`](../src/bank5/touch-door.asm#L108)
- Variant for steps 23-25 with the magic key or 2 spare keys, and the ladder: in `$51` kill the 6 Like-Likes (its west shutter opens), then `$50` -> north key door -> `$40` -> north key door -> `$30`; same length, one bomb less, a water maze in `$40`. [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152), [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- Order matters, silver arrow: steps 15-19 must come before step 31; room `$42` cannot be left without it. [`Ganon_CheckCollisions@CheckArrowCollision`](../src/bank4/ganon-draw-collisions.asm#L427)
- Order matters, bombs: from `$41` on there is no way back south (the south shutters of `$41`, `$31`, `$21` and `$01` have no trigger). A Link in `$31` without a bomb can only go north through `$21`, `$11`, `$01`, and gets out of that column only with a bomb (`$21` E) or a key (`$01` E). Have 2 bombs when leaving `$61` north. [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68), [`Door_TouchShutter`](../src/bank5/touch-door.asm#L126)
- Order matters, blocks: a block moves only once per room entry and only in a cleared room, and a tunnel's stairs in rooms `$55`, `$61`, `$20`, `$03` are closed in by blocks until the left one is pushed up or down. Pushing it sideways leaves the stairs closed; leave and re-enter the room to try again. [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`PushBlock_State1Moving`](../src/bank4/push-block.asm#L165)
- Room `$20` is entered twice; its kills are banked when Link leaves it and the room is still among the last six rooms entered when he comes back, so clearing it on either visit is enough. [`LevelKillCounts`](../src/Variables.inc#L480), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- After a death: CONTINUE puts Link back in room `$76` with 3 hearts; the level is not reloaded, so opened doors, bombed walls, taken items and the silver arrow are kept and dead Patras stay dead, but the room history is cleared: cleared rooms of other monsters refill, every block is back, and the Lanmolas of `$55` are alive again. From `$76` the way to Ganon is steps 1-13 (leaving `$61` by its north door) and 22-31. [`Mode8Continue_Update_Full@HandleActivated`](../src/bank5/mode8-continue-mode10-stairs.asm#L143), [`Mode8Continue_ChoiceModes`](../src/bank5/mode8-continue-mode10-stairs.asm#L49), [`Link_BeHarmed@HandleDied`](../src/bank1/link-collision-harm.asm#L254), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- Wallmasters: none in this level, so nothing carries Link back to the start. [`Level_BlockUW7to9Q1`](../src/bank6/level-blocks-common-data.asm#L74)

# Optional detours

- **Red ring** (from `$16`, step 7): `$16` S -> `$26` -> bomb E -> `$27` -> bomb N -> `$17` -> bomb N -> `$07` -> clear, push the block, stairs -> cellar `$00` -> back the same way to `$16`. 10 room entries, 3 bombs, no key; one forced fight (`$07`: 3 Blue + 2 Red Wizzrobes, 3 Bubbles). Worth a quarter of all later damage, Ganon's included. [`Item_TakeClass2Grade`](../src/bank1/take-item.asm#L536), [`Link_BeHarmed`](../src/bank1/link-collision-harm.asm#L193)
- **Map** (room `$27`): on the red ring detour; kill its Patra (`$48`) and take it at (`$80`,`$90`). Alone from `$16`: 4 entries, 1 bomb. [`InvMap9`](../src/Variables.inc#L577)
- **Compass** (room `$35`): from `$15` bomb the south wall, cross the middle strip of `$25` (two full-height rows of blocks cut that room into three strips; the north and south walls open into the middle one), bomb its south wall, clear `$35` (3 Blue + 2 Red Wizzrobes). 4 entries, 2 bombs. It only marks room `$32`. [`InvCompass9`](../src/Variables.inc#L576)
- **Key of `$47`** (in the open) and **key of `$57`** (clear the room, ladder): from `$16` go `$26` -> `$27` (bomb `$26` E) -> `$37` (bomb `$27` S; 6 Vires, statues that shoot) -> `$47` -> `$57`. 8 entries and 2 bombs for `$47` and back, 2 more entries for `$57`. Needed only by a Link who came with fewer than 2 keys. [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337)
- **Bomb pickups**: `$16` on the route (kill the Patra); `$37` and `$25` on the detours above; `$23` in the open but behind `$24`'s west wall; `$11` behind the one-way column north of `$31`. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- **Hints**: the men in `$02` ("EYES OF SKULL HAS A SECRET."), `$43` ("PATRA HAS THE MAP.", behind `$53`'s north wall) and `$06` (on the route) only talk; texts are in [`person-text.md`](../data/person-text.md). [`Person_TextSelectorsLevel9`](../src/bank1/underworld-person-init.asm#L162)
- The other tunnel, `$72` (`$71` <-> `$74`), joins a room below `$61` to a room behind the key doors of `$63` and `$73`; it leads to nothing the route needs. [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43)
- This level has no bomb-capacity man and no money or life-or-money room. [`Person_UpdateUnderworld_Full`](../src/bank1/underworld-person-update.asm#L44)

# Hazards and rules that matter here

- **The man of `$66`**: type `$4B` says "ONES WHO DOES NOT HAVE TRIFORCE CAN'T GO IN." unless all eight pieces are held; then he is removed during the room's setup and the shutters open, on every entry. His two fires stay and burn for half a heart. [`Person_InitUnderworldLevel9`](../src/bank1/underworld-person-init.asm#L172), [enemies.md](../enemies.md) "Standing fire"
- **The other men** (`$06`, `$02`, `$43`) only talk: Link is halted until the text is typed (6 frames per character) and cannot walk above Y = `$8D` in their rooms. Hitting one does not kill him; it turns on fireballs from beside the fires until Link leaves the room. [`Person_UpdateUnderworld_Full`](../src/bank1/underworld-person-update.asm#L44), [`Person_CheckCollisions`](../src/bank1/underworld-person-update.asm#L89), [`Person_BlockLinkMovingUp`](../src/bank1/modeb-cave-save-sprites.asm#L289)
- **Shutters with a trigger**: `$11`, `$12`, `$50`, `$51`, `$52`, `$32` open when the room is cleared; `$42` when the Triforce of Power is held; `$66` by the man. An opened shutter is not saved and closes behind Link when he enters through it. [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152), [`Secret_Trigger3PowerTriforce`](../src/bank5/underworld-secrets.asm#L255), [`Room_EnterCommon@Method2`](../src/bank5/enter-room.asm#L211)
- **Shutters without a trigger** (one-way doors): the south sides of `$01`, `$21`, `$31` and `$41`. The rooms below them have an open doorway on that side, so Link passes northward (the side he enters by is opened for him and closes behind him) and never southward. `$11` S, `$12` N and `$66` N and W are also shutters seen from inside only, but have a trigger. [`Door_SetEnteringDoorwayOpened`](../src/bank5/mode7-scroll-init.asm#L368), [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68)
- **Bombable walls** look like plain walls. The bomb itself must lie less than `$18` pixels on both axes from the wall's hotspot: N (`$78`,`$5D`), S (`$78`,`$BD`), W (`$20`,`$8D`), E (`$D0`,`$8D`); a bomb appears `$10` pixels ahead of Link. The test is made once, when the blast ends; the wall is passable 8 frames later and stays open from both sides. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Bomb_WallHotspotsX`](../src/bank7/bomb.asm#L60), [`Door_Update`](../src/bank5/update-doors.asm#L60)
- **Push blocks**: 11 rooms have one, on the first block of the floor row at Y = `$90`: X = `$60` in layouts `$1A` and `$0A`, X = `$C0` in `$10`. Link must be in line with it, at most `$10` pixels away, holding only the direction toward it for `$10` frames, in a cleared room. In `$05`, `$10`, `$30` the push makes stairs appear at (`$D0`,`$60`); in the layout `$1A` rooms it opens the ring of blocks round the stairs. The block of `$37` has no effect. [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`Secret_Trigger5BlockStairs`](../src/bank5/underworld-secrets.asm#L226)
- **Stairs and tunnels**: stairs take Link when he stands on them on a grid point. A tunnel cellar has two ladders; Link enters at X = `$30` when he comes from the cellar's left-exit room, else at X = `$C0`, and leaves by walking up a ladder: left of X = `$80` to the left-exit room, else to the right-exit room. Bombs open no wall in a cellar. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`Mode9Cellar_Init_Sub8_EnterCellar`](../src/bank5/cave-cellar-enter.asm#L276), [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43)
- **The six tunnels** (left exit <-> right exit): `$60`: `$14` <-> `$55`; `$70`: `$63` <-> `$05`; `$72`: `$71` <-> `$74`; `$75`: `$20` <-> `$61`; `$67`: `$30` <-> `$04`; `$77`: `$52` <-> `$03`. Link comes out at (`$60`,`$9D`), except (`$A0`,`$9D`) from `$70` and (`$C0`,`$6D`) from `$67`; in the layout `$1A` rooms that spot is right under the push block. [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)
- **Water and the ladder**: the ladder bridges one square of water in the direction Link faces. Route: `$65` (one row to cross). Elsewhere: `$56` and `$33` (two rows), `$15`, `$34`, `$50`, `$57` (a ring round the middle: the doors are joined by the outer lane without it), `$01` (one column), `$23`, `$24`, `$40` (mazes). See [items.md](../items.md) "Ladder". [`Link_EndMoveAndAnimate@SetUpLadder`](../src/bank7/link-animate-ladder.asm#L246), [`Ladder_CheckAndDraw`](../src/bank5/ladder.asm#L44)
- **Dark rooms** (`$01`, `$13`, `$15`, `$24`, `$33`, `$41`, `$42`, `$50`, `$57`, `$65`): only the palette is dark; everything behaves as in a lit room. A candle's or the book's standing fire lights the room until Link leaves it. [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124), [`Candle_UpdateRoomLight`](../src/bank4/shoot-helpers-candle.asm#L132)
- **Traps** (`$04`, `$30`, `$41`): four, starting at (`$20`,`$5D`), (`$20`,`$BD`), (`$D0`,`$5D`), (`$D0`,`$BD`). One charges when Link is within `$0E` pixels of its row or column, at up to 1.75 pixels per frame toward the room's middle, and creeps back at 0.5; it senses nothing until it is home again. The top and bottom floor rows and the columns of the west and east doorways (X = `$20`, `$D0`) are trap lines; in `$30` the stairs appear on the home square of the top right Trap. [`Trap_Update_Full`](../src/bank1/trap-rupee-stash.asm#L189), [`Trap_StartXs`](../src/bank1/trap-rupee-stash.asm#L112), [`Trap_StartYs`](../src/bank1/trap-rupee-stash.asm#L117)
- **Statues**: only layout `$23` (rooms `$37`, `$53`) shoots fireballs; the statues of `$76`, `$32` and `$42` do nothing. [`Statue_RoomLayoutIds`](../src/bank4/statues.asm#L45), [`Statue_Update`](../src/bank4/statues.asm#L86)
- **Boss roar**: rooms `$41`, `$43` and `$52` repeat the roar sample until Ganon's room is flagged cleared. [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42)
- **Keyless pass**: while a shutter or a bombed wall is opening and Link's timer is 0, a locked door lets him through unopened and unpaid; the rule is in [mechanics.md](../mechanics.md) "Doorways, doors, keys, shutters and bomb walls". [`Door_Touch_BlockWhileTimer`](../src/bank5/touch-door.asm#L193)
- **Not in this level**: Wallmasters, false walls, a hungry Goriya, a heart container, a Triforce piece. [`Level_BlockUW7to9Q1`](../src/bank6/level-blocks-common-data.asm#L74)

# Second quest

In Q2 the game loads a different level block for level 9 ([`levelblock-uw-7to9-q2.md`](../data/levelblock-uw-7to9-q2.md), rows with `lvl` = `9`)
and patches the level info (start room `$74`, boss room `$17`, five cellars; [`levelinfo.md`](../data/levelinfo.md) "Level 9, Q2"); its
overworld entrance is the rock wall of room `$00`, whose secret works in Q2 only. This document covers the Q1
level only.
