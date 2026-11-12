# Level 6: the magic rod, Wizzrobes and Gohma

Level 6 lies under overworld room `$22` (Q1) and is the Wizzrobe dungeon: 25 rooms, most of them holding Blue or Red
Wizzrobes, which only the sword, its beam and bombs can hurt, mixed with Like-Likes that take the magic shield and
Bubbles that switch the sword off. Link collects the magic rod (cellar `$75`, under room `$09`), a heart container
(boss room `$1C`) and a Triforce piece (room `$0C`). The boss is a Gohma of type `$34`, which dies to one arrow shot
upward into its fully open eye and to nothing else, so the bow, an arrow and rupees must be brought. The ladder must
be brought too: the only way to the boss crosses a column of water in room `$39`. The level's five keys pay for its
five key doors, but only one key lies in front of the first door, and three rooms (`$38`, `$09`, `$3A`) give up their
exit or their stairs only after a block is pushed in a room that has first been cleared of Wizzrobes.

> **Written from the ROM only.** Unlike levels 1-4, nothing here was walked in the running game; the route, the
> requirements and the room notes come from the level data and the code. When you work on this level, check each
> step against the game and the code, and update this document with what you find (mark it `(trace)`).

Contents (generated; `L` = line in this file):

<!-- toc -->
- L32 At a glance
- L71 What Link must bring
- L84 What Link gets here
- L128 Map
- L161 Rooms
- L197 Floor plans
- L454 Monsters and boss
- L493 Route
- L539 Optional detours
- L551 Hazards and rules that matter here
- L573 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 6 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-6 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | $22; not in Q1, the secret works in Q2 only: $30 | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178), [`Flute_SecretRoomIdsOW`](../src/bank7/flute.asm#L36) |
| Start room | $79 | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Triforce room | $0C (room item: Triforce piece ($1B)); the field places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $1C (1 x Gohma ($34)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | $08, $75 | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 25: 25 on the submenu map | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (room and side that has it) | 10 ($09 S, $19 N, $19 S, $29 N, $1C S, $2C N, $6A S, $7A N, $78 E, $79 W) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (room and side that has it) | 6 ($09 E, $0A W, $0A E, $0B W, $28 E, $29 W) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (room and side that has it) | 12 ($0A S, $1B N, $1C N, $18 S, $18 E, $28 S, $38 N, $39 N, $3C N, $38 S, $39 E, $58 N) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (room and side that has it) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $0A S shutter / $1A open, $0B S open / $1B shutter, $0C S open / $1C shutter, $18 S shutter / $28 open, $18 E shutter / $19 open, $29 S open / $39 shutter, $2C S open / $3C shutter, $38 S shutter / $48 open, $39 E shutter / $3A open, $48 S open / $58 shutter | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $79 S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 3 ($09, $38, $3A) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 6 ($1A, $29, $2D, $39, $3C, $7A) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 5 ($1A, $29, $2D, $58, $7A) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | bomb ($00): $3C; 5 rupees ($0F): $28; rod ($10): $75 (cellar); compass ($16): $68; map ($17): $19; heart container ($1A): $1C; Triforce piece ($1B): $0C | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md) (rows with `lvl` = `6` or `c6`).
<!-- /generated:facts -->

- Entrance (Q1): overworld room `$22`, an open cave mouth two squares wide in the middle of the room (layout `$21`, squares `c7`/`c8` of row `r4`, tile `$24` under Link at Y `$7D`); no item or secret is needed. See `# Rooms (first quest)` in [data/levelblock-ow.md](../data/levelblock-ow.md) and layout `OW $21` in [data/room-layouts.md](../data/room-layouts.md); the entrance test is [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178).
- Room `$22` is the one overworld room where the entrance test accepts Link's X at any multiple of 8 (elsewhere `$10`), so X `$70`, `$78` and `$80` all enter. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
- The room's rock border has one gap, the steps at X `$70` in the bottom two rows, so it is walked into from room `$32` below; it holds one red Lynel (`$02`) and twelve Armos statues. The Armos at X `$30`, Y `$80` leaves stairs when it wakes, and those stairs lead into the level as well. [`Armos_SecretRoomIds`](../src/bank4/ghini-armos.asm#L138), [`Armos_SecretXs`](../src/bank4/ghini-armos.asm#L143)
- The facts table also names overworld room `$30`: its level block byte carries the same destination, but its layout (`OW $2F` in [data/room-layouts.md](../data/room-layouts.md)) has no entrance tile, and the recorder uncovers stairs there only in Q2. In Q1 room `$22` is the only way in. [`Flute_SecretRoomIdsOW`](../src/bank7/flute.asm#L36)
- The recorder's whirlwind sets Link down in room `$22` only once he already holds this level's Triforce piece, so it is no help for the first visit. [`Whirlwind_PrevRoomIds`](../src/bank1/whirlwind.asm#L81), [`Whirlwind_Summon`](../src/bank1/whirlwind.asm#L243)
- On entry Link stands in room `$79` at X `$78`, Y `$DD` (the bottom doorway), facing up; the room has no monsters. [`Mode3Unfurl_Init_Sub8_LayOutRoomPlaceLink`](../src/bank5/mode3-mode6-init.asm#L129), [`LevelInfo_StartY`](../src/Variables.inc#L717)
- Walking down out of room `$79` leaves the level: the next room id is off the map, the overworld is loaded and Link stands at the exit spot of room `$22` (X `$70`, Y `$7D`). [`Room_CalcNextRoomForDoor`](../src/bank5/next-room-calc.asm#L76), [`Mode_LeaveLevelToOW`](../src/bank5/next-room-calc.asm#L107)
- Taking the Triforce piece in room `$0C` ends the level: mode `$12` refills the hearts, then Link is put on the overworld at the same exit spot of room `$22`. [`Mode12EndLevel_Update_Sub4LeaveLevel`](../src/bank5/mode12-lay-out-room.asm#L156), [`LevelEntranceRoomId`](../src/Variables.inc#L463)
- Kept when Link leaves and comes back: unlocked key doors and bombed walls (room flag bits 0-3, set in both rooms), taken room items (bit `$10`), a dead Gohma or Gleeok (non-recurring). [`RoomFlags_SetDoorOpened`](../src/bank5/shutters-door-flags.asm#L41), [`RoomFlags_SetItemTaken`](../src/bank1/take-item.asm#L140), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- Reset when the level is loaded again: the RAM kill counts and the room history, so partly cleared rooms are full again and fully cleared rooms refill; shutters are shut on every room entry; a pushed block is back and its stairs are gone. [`Mode2Load_Init`](../src/bank7/init-mode-dispatch.asm#L147), [`LevelKillCounts`](../src/Variables.inc#L480), [`Door_Update`](../src/bank5/update-doors.asm#L60), [`Secret_Trigger5BlockStairs`](../src/bank5/underworld-secrets.asm#L226)

# What Link must bring

Needed from outside, with the room that needs it. Everything else on the list of usual dungeon gear is not needed.

- **Bow, an arrow, rupees** (required): Gohma in room `$1C` refuses every damage type but the arrow. A shot needs `InvBow` and at least 1 rupee, and costs 1 rupee whether it hits or not; the wooden arrow (`$20`) is enough for this Gohma's HP `$20`. Bring a few dozen rupees for misses. [`Arrow_Wield`](../src/bank5/wield-items.asm#L187), [`Gohma_HandleWeaponCollision`](../src/bank4/gohma.asm#L367), [`InvRupees`](../src/Variables.inc#L583)
- **Ladder** (required): room `$39` (layout `$13`) has a full-height column of water at X `$A0`; Link comes in by the N door (X `$78`) and must leave by the E door, and there is no other way to rooms `$3A`, `$1D`, `$2D`, `$2C` and the boss. The ladder is also what reaches the key inside the water ring of room `$29`. [`InvLadder`](../src/Variables.inc#L570), [`Link_EndMoveAndAnimate@SetUpLadder`](../src/bank7/link-animate-ladder.asm#L246)
- **A sword, the stronger the better** (required in practice): Wizzrobes test only the sword, the sword beam and bomb blasts, and the route must empty three Wizzrobe rooms (`$38`, `$09`, `$3A`) before their push blocks move, plus room `$7A` for the first key. A Blue Wizzrobe (HP `$A0`) takes 10 wooden, 5 white or 3 magic sword hits; a Like-Like (HP `$90`) 9, 5 or 3. [`Wizzrobe_DrawAndCheckCollisions`](../src/bank4/wizzrobe.asm#L683), [`Sword_DamagePoints`](../src/bank1/sword-bomb-arrow-collision.asm#L182), [`Obj_TypeHpPairs`](../src/bank7/object-update-table.asm#L105)
- **Bombs** (1 for the route as given; more are useful): one bomb opens the wall between rooms `$28` and `$29` and saves the Gleeok room; without bombs take the alternative under `# Route`. Bombs are also the second weapon that hurts Wizzrobes: a blast (`$40`) kills a Red Wizzrobe outright and a Blue one in three. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Bomb_CheckBombOrFireVsMonster`](../src/bank1/sword-bomb-arrow-collision.asm#L126)
- **Keys**: none. The route finds four keys (`$7A`, `$58`, `$29`, `$2D`) and opens four key doors (`$79` W, `$29` N, `$19` N, `$1C` S), never going below zero. One key brought in saves the fight in dark room `$7A` (two room entries); the magic key saves all four. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`InvKeys`](../src/Variables.inc#L584), [`InvMagicKey`](../src/Variables.inc#L571)
- **Candle**: not needed. Six rooms are dark (`$1A`, `$29`, `$2D`, `$39`, `$3C`, `$7A`; four are on the route), but the dark flag is read only by the code that fades the palettes and redraws the room on entry; monsters, collision, doors and items do not test it. A candle fire, or the book's fire, lights the room until Link leaves it. [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124), [`Mode7Scroll_Init_Sub5`](../src/bank5/mode7-scroll-init.asm#L256), [`Candle_UpdateRoomLight`](../src/bank4/shoot-helpers-candle.asm#L132)
- **Hearts, ring, shield** (not tested by any code, but this is where they pay): a Red Wizzrobe's shot costs 4 hearts and a Blue one's 2 before rings, a Blue Wizzrobe's touch 2, Gohma's touch 2; each ring grade halves it. Only the magic shield parries the Wizzrobes' shots, and the Like-Likes of rooms `$38` and `$3A` take it away. See `# Wizzrobe (blue, red)` and `# Like-Like` in [enemies.md](../enemies.md). [`Damage_ObjTypeDamageToLink`](../src/bank1/monster-collisions.asm#L200), [`Link_BeHarmed`](../src/bank1/link-collision-harm.asm#L193)
- **Raft, recorder, food**: not needed; the level has no dock, no Digdogger and no hungry Goriya (rooms table below). [`Level_BlockUW1to6Q1`](../src/bank6/level-blocks-common-data.asm#L70)

# What Link gets here

## Magic rod (item `$10`): treasure cellar `$75`, under room `$09`

- Room `$09` is reached from room `$19` through its N key door (1 key). The stairs to the cellar do not exist until the room's push block is moved: kill the 3 Blue and 2 Red Wizzrobes, then push the block at X `$60`, Y `$90` (the left one of the two) for `$10` frames; the stairs appear at X `$D0`, Y `$60` (top right corner square). [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`Secret_Trigger5BlockStairs`](../src/bank5/underworld-secrets.asm#L226)
- In the cellar Link appears at X `$30`, walks down the left ladder, and the rod lies at X `$80`, Y `$90`; four blue Keese fly there. Both ways out (Y below `$40` with Up held) lead back to room `$09`, where he comes out at X `$C0`, Y `$8D`. [`Mode9Cellar_Init_Sub8_EnterCellar`](../src/bank5/cave-cellar-enter.asm#L276), [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43), [`Item_TryTakeRoomItem`](../src/bank1/take-item.asm#L173)
- **What it changes.** The rod becomes a B-button item (selected-slot index 8): B starts a swing in object slot `$12` that hits like a white sword (damage `$20`, the sword's damage type) and, 12 frames in, sends a magic shot from slot `$0E` (damage `$20`, 2.5 pixels per frame, ends on the first monster or the room bound). Unlike the sword beam it needs no full health and unlike the arrow it costs nothing. See `# Magic Rod and Book` in [items.md](../items.md). [`Rod_Wield`](../src/bank5/wield-items.asm#L294), [`Sword_UpdateSwordOrRod@MakeMagicShot`](../src/bank7/sword-and-rod.asm#L258), [`SwordShot_CheckMonsterCollision`](../src/bank1/sword-bomb-arrow-collision.asm#L64)
- **Where it does not work here.** Wizzrobes never test slot `$12` (the swing passes through) and mask the magic shot; Gohma masks both; Bubbles and Traps take no weapon at all. So the rod does not make the rest of this level easier against its main monsters. [`Wizzrobe_DrawAndCheckCollisions`](../src/bank4/wizzrobe.asm#L683), [`Gohma_Init`](../src/bank4/gleeok-manhandla-gohma-inits.asm#L221)
- **Where it does work here.** Like-Likes, Zols (killed outright, no split), Keese and Vires take both the swing and the shot; a Vire hit for `$20` splits into two red Keese instead of dying. The swing also counts against Gleeok's heads. [`Arrow_CheckArrowOrRodVsMonster`](../src/bank1/sword-bomb-arrow-collision.asm#L249), [`Vire_CheckCollisions`](../src/bank4/vire.asm#L154), [`Gleeok_CheckSegmentCollisions`](../src/bank4/gleeok-head-body.asm#L128)
- **Limits and later use.** The rod shares slot `$12` with the arrow (no arrow while a swing runs, no swing while an arrow flies) and slot `$0E` with the sword beam. No door, secret or room of this level tests `InvRod`; later, with the book, each magic shot leaves a standing fire where it ends, which also lights dark rooms. [`InvRod`](../src/Variables.inc#L566), [`SwordShot_HandleBlocked`](../src/bank7/sword-shot-magic-shot.asm#L172), [`InvBook`](../src/Variables.inc#L568)

## Heart container (item `$1A`): boss room `$1C`

- Hidden until Gohma is dead (secret trigger 7), then it appears at X `$80`, Y `$90`, the middle of the room, and the N shutter opens. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- **What it changes.** `HeartValues` + `$11`: one more container and one more full heart; the other hearts are not refilled (the Triforce piece next door does that). The count of containers is what the sword caves test: 5 for the white sword, 12 for the magic sword. See `# Hearts, Heart Containers and Fairies` in [items.md](../items.md). [`Item_Take_CheckClass1@TakeHeartContainer`](../src/bank1/take-item.asm#L392), [`Cave_State2TalkOrShopOrDoorCharge@TryToTake`](../src/bank1/cave-shop.asm#L161)

## Triforce piece (item `$1B`): room `$0C`

- Room `$0C` has one door, S, onto the boss room, whose N shutter opens only when Gohma is dead. The piece lies at X `$78`, Y `$90` inside a ring of blocks open at the bottom middle: walk straight up from the door at X `$78`. No monsters. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152)
- **What it changes.** Sets bit 5 (`$20`) of `InvTriforce`, ends the level (mode `$12`: full heart refill, then the overworld at room `$22`), and makes room `$22` a destination of the recorder's whirlwind. All eight pieces are what the old man at the start of level 9 tests. See `# Triforce pieces and the Triforce of Power` in [items.md](../items.md). [`Item_TakeClass0PerLevel`](../src/bank1/take-item.asm#L421), [`InvTriforce`](../src/Variables.inc#L597), [`Person_InitUnderworldLevel9`](../src/bank1/underworld-person-init.asm#L172)

## Keys (item `$19`): rooms `$7A`, `$58`, `$29`, `$2D`, `$1A`

| room | how the key is had | on the route |
| --- | --- | --- |
| `$7A` | appears at `$80`,`$90` when the 5 Red Wizzrobes are dead; dark room | yes: the only key in front of the first key door |
| `$58` | appears at `$80`,`$90` when the 8 blue Keese are dead (the same kills open the N shutter) | yes |
| `$29` | lies in the open at `$80`,`$90`, inside a one-square ring of water: ladder; dark room, nothing need be killed | yes |
| `$2D` | appears at `$80`,`$90` when the 3 Vires and every red Keese they split into are dead (the 2 Bubbles do not count); dark room | yes |
| `$1A` | lies in the open at `$80`,`$90`, south of the room's water row; dark room with 6 Red Wizzrobes, nothing need be killed | no (spare) |

- **What it changes.** Each key adds 1 to `InvKeys` (one counter for all levels) and is spent by pushing into a key door; an unlocked door stays open from both sides for good. See `# Keys and Magic Key` in [items.md](../items.md) and `# Doorways, doors, keys, shutters and bomb walls` in [mechanics.md](../mechanics.md). [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [`Door_Update`](../src/bank5/update-doors.asm#L60)
- An item with "room item appears" in the rooms table is absent until no counting monster is left; an item without it is there from the start. [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51), [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124)

## Map (`$17`, room `$19`) and compass (`$16`, room `$68`)

- Both rooms are on the route and both items appear at X `$80`, Y `$90` only when the room is cleared: room `$68` 5 Zols; room `$19` 2 Zols and 2 Like-Likes (its 2 Bubbles do not count). In room `$19` that spot is west of the water, on the side of the N and S doors. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124)
- **What they change.** Only the status bar: the map draws the level's room plan, the compass a blinking marker on room `$0C`. Neither changes doors, monsters or room contents. See `# Map and Compass` in [items.md](../items.md). [`Inv_HasMap`](../src/bank5/submenu-map.asm#L153), [`Inv_HasCompass`](../src/bank5/submenu-map.asm#L143), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51)

## Bombs (`$00`, room `$3C`) and 5 rupees (`$0F`, room `$28`)

- Room `$3C` (dark, south of `$2C`): the bomb item appears at X `$80`, Y `$90` when its 3 Blue and 2 Red Wizzrobes are dead; it adds 4 bombs up to the capacity. The room's only door is a shutter that shuts behind Link and opens on the same kills, so entering commits him to the fight. [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337), [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152)
- Room `$28` (on the route): 5 rupees appear at X `$D0`, Y `$60` when its 3 Blue and 2 Red Wizzrobes are dead. Five rupees are five arrows. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`Arrow_Wield`](../src/bank5/wield-items.asm#L187)

# Map

<!-- generated:map -->
```
         [09  ] X [0A  ] X [0B  ]   [0CT ]
           K       S/|      |/S      |/S
[18  ]S/-[19i ]---[1Ak ]---[1B  ]   [1CBi]   [1D  ]
 S/|       K                          K        |
[28i ] X [29k ]                     [2C  ]---[2Dk ]
  S       |/S                        |/S
[38  ]   [39  ]S/-[3A  ]            [3Ci ]
 S/|
[48  ]
 |/S
[58k ]
  |
[68i ]            [6A  ]
  |                 K
[78  ] K [79S ]---[7Ak ]

[73S] = room $73 and its marks: S start room, B boss room, T Triforce room,
        k a key lies here, i another room item lies here, h not shown on the submenu map.
Between two rooms, the door: --- or | open, K key door, S shutter, X bombable wall,
        F false wall (walk-through), nothing = wall. A/B = the two sides differ: the left
        or upper room's side first, . = wall on that side.
Doors on sides with no room of the level: $79 S open.
```

- Cellar `$08` (tunnel cellar): joins `$3A` (left exit) and `$1D` (right exit), whose stairs lead here; item: none. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md)

- Cellar `$75` (treasure cellar): both exits lead to `$09`, whose stairs lead here; item: rod ($10). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md)
<!-- /generated:map -->

# Rooms

Generated from the level data by `tools/romdata.py`; do not edit between the markers.

<!-- generated:rooms -->
| room | N | S | W | E | monsters | item | secret | block | dark | layout | floor | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $09 | wall | key | wall | bomb | 3 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | push block: stairs appear | B | - | $0A | blocks | stairs lead to cellar $75 |
| $0A | wall | shutter | bomb | bomb | 2 x Bubble ($2B), 2 x Zol ($13), 2 x Like-Like ($17) | - | all dead: shutters open | - | - | $0A | blocks | - |
| $0B | wall | open | bomb | wall | 1 x dungeon person ($4C) | - | - | - | - | $26 | blank | person $4C says: THERE ARE SECRETS WHERE / FAIRIES DON'T LIVE. |
| $0C Triforce | wall | open | wall | wall | none | Triforce piece ($1B) at $80,$90 | - | - | - | $29 | blocks, statues | - |
| $18 | wall | shutter | wall | shutter | 1 x Gleeok ($44) | - | all dead: shutters open | - | - | $05 | blocks | - |
| $19 | key | key | open | open | 2 x Bubble ($2B), 2 x Zol ($13), 2 x Like-Like ($17) | map ($17) at $80,$90 | all dead: shutters open, room item appears | - | - | $13 | water | - |
| $1A | open | wall | open | open | 6 x Red Wizzrobe ($24) | key ($19) at $80,$90 | - | - | D | $18 | water | - |
| $1B | shutter | wall | open | wall | 1 x Bubble ($2B), 3 x Like-Like ($17), 2 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | all dead: shutters open | - | - | $03 | blocks | boss sound $20 |
| $1C boss | shutter | key | wall | wall | 1 x Gohma ($34) | heart container ($1A) at $80,$90 | all dead: shutters open, room item appears | - | - | $06 | blocks, statues | - |
| $1D | wall | open | wall | wall | 2 x Bubble ($2B), 2 x Zol ($13), 2 x Like-Like ($17) | - | - | - | - | $1B | blocks, stairs | stairs lead to cellar $08; boss sound $20 |
| $28 | open | shutter | wall | bomb | 3 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | 5 rupees ($0F) at $D0,$60 | all dead: shutters open, room item appears | - | - | $0D | blocks | - |
| $29 | key | open | bomb | wall | 3 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | key ($19) at $80,$90 | - | - | D | $14 | water | - |
| $2C | key | open | wall | open | 1 x Trap ($4A), 3 x (empty) ($00), 2 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | - | - | - | $00 | plain | boss sound $20 |
| $2D | open | wall | open | wall | 2 x Bubble ($2B), 3 x Vire ($12) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | D | $13 | water | - |
| $38 | shutter | shutter | wall | wall | 1 x Bubble ($2B), 3 x Like-Like ($17), 2 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | push block: shutters open | B | - | $0A | blocks | - |
| $39 | shutter | wall | wall | shutter | 5 x Vire ($12) | - | all dead: shutters open | - | D | $13 | water | - |
| $3A | wall | wall | open | wall | 1 x Bubble ($2B), 3 x Like-Like ($17), 2 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | - | push block: stairs appear | B | - | $22 | blocks | stairs lead to cellar $08 |
| $3C | shutter | wall | wall | wall | 3 x Blue Wizzrobe ($23), 2 x Red Wizzrobe ($24) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | D | $17 | water | - |
| $48 | open | open | wall | wall | 1 x Trap ($49) | - | - | - | - | $01 | blocks | - |
| $58 | shutter | open | wall | wall | 8 x Keese blue ($1B) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $24 | statues | - |
| $68 | open | open | wall | wall | 5 x Zol ($13) | compass ($16) at $80,$90 | all dead: shutters open, room item appears | - | - | $1D | blocks | - |
| $6A | wall | key | wall | wall | 1 x dungeon person ($4D) | - | - | - | - | $26 | blank | person $4D says: AIM AT THE EYES / OF GOHMA. |
| $78 | open | wall | wall | key | 5 x Red Wizzrobe ($24) | - | ringleader: first monster dead kills all | - | - | $1F | blocks | - |
| $79 start | wall | open | key | open | none | - | - | - | - | $21 | statues, sand | - |
| $7A | key | wall | open | wall | 5 x Red Wizzrobe ($24) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | D | $03 | blocks | - |
| $08 cellar | - | - | - | - | 4 x Keese blue ($1B) | - | - | - | - | $3E | tunnel cellar | left exit to $3A, right exit to $1D; Link comes out at $60,$9D |
| $75 cellar | - | - | - | - | 4 x Keese blue ($1B) | rod ($10) at $80,$90 | - | - | - | $3F | treasure cellar | left exit to $09, right exit to $09; Link comes out at $C0,$8D |
<!-- /generated:rooms -->

# Floor plans

<!-- generated:floors -->
One character per floor square, 12 columns x 7 rows inside the walls (left column = west, top row =
north): `.` floor (tile `$74`), `B` block (tile `$B0`; a push block object is placed on one in a room
with a push block), `W` water (tile `$F4`), `S` statue (tiles `$94` and `$B4`), `>` stairs (tile `$70`),
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L63), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L73), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
Stairs that a secret makes appear are not part of the layout.

```
layout $00  rooms: $2C
............
............
............
............
............
............
............
```

```
layout $01  rooms: $48
............
............
..B......B..
.B........B.
..B......B..
............
............
```

```
layout $03  rooms: $1B,$7A
............
..B......B..
............
............
............
..B......B..
............
```

```
layout $05  rooms: $18
BBBBB..BBBBB
BBBB....BBBB
BB........BB
............
............
..B......B..
............
```

```
layout $06  rooms: $1C
............
............
............
.B........B.
............
...B....B...
S..........S
```

```
layout $0A  rooms: $09,$0A,$38
............
............
............
....B..B....
............
............
............
```

```
layout $0D  rooms: $28
............
.B.B.BB.B.B.
............
.B.B.BB.B.B.
............
.B.B.BB.B.B.
............
```

```
layout $13  rooms: $19,$2D,$39
........W...
........W...
........W...
........W...
........W...
........W...
........W...
```

```
layout $14  rooms: $29
............
.WWWWWWWWWW.
.W........W.
.W........W.
.W........W.
.WWWWWWWWWW.
............
```

```
layout $17  rooms: $3C
WWWWW..WWWWW
W...W....W.W
W.W.W.WW.W.W
..W.W..W.W..
W.W.WW.W.W.W
W.W....W...W
WWWWW..WWWWW
```

```
layout $18  rooms: $1A
............
............
WWWWWWWWWWWW
............
............
............
............
```

```
layout $1B  rooms: $1D
........BBBB
........BBBB
.........BBB
...........>
.........BBB
........BBBB
........BBBB
```

```
layout $1D  rooms: $68
............
............
..BB....BB..
..BB....BB..
..BB....BB..
............
............
```

```
layout $1F  rooms: $78
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
layout $22  rooms: $3A
............
............
............
.....B......
............
............
............
```

```
layout $24  rooms: $58
S..........S
............
............
............
............
............
S..........S
```

```
layout $26  rooms: $0B,$6A
____________
____________
____________
____________
____________
____________
____________
```

```
layout $29  rooms: $0C
............
.BBBBBBBBBB.
.B..S..S..B.
.B.S....S.B.
.B........B.
.BBBB..BBBB.
............
```

```
layout $3E (tunnel cellar)  rooms: $08
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
layout $3F (treasure cellar)  rooms: $75
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

HP and damage in the units of [enemies.md](../enemies.md): a wooden sword hit is `$10`, white `$20`, magic `$40`, a bomb blast `$40`; damage to Link is before rings.

| monster | rooms (count) | HP | what hurts it | what it does to Link | section in enemies.md |
| --- | --- | --- | --- | --- | --- |
| Red Wizzrobe `$24` | `$78` (5), `$7A` (5), `$1A` (6); 2 each in `$09`, `$28`, `$29`, `$3C`, `$1B`, `$38`, `$3A`, `$2C` | `$40` | sword, sword beam, bomb only; no stun, no knockback | appears 2-5 squares from Link facing him, one shot per 208-frame cycle: 4 hearts; touch 1 heart | `# Wizzrobe (blue, red)` |
| Blue Wizzrobe `$23` | 3 each in `$09`, `$28`, `$29`, `$3C`; 2 each in `$1B`, `$38`, `$3A`, `$2C` | `$A0` | sword, sword beam, bomb only | walks and teleports (through blocks and water), shoots along its row or column every 32 frames when facing Link: 2 hearts; touch 2 hearts | `# Wizzrobe (blue, red)` |
| Like-Like `$17` | 2 each in `$0A`, `$19`, `$1D`; 3 each in `$1B`, `$38`, `$3A` | `$90` | every weapon; boomerang stuns | touch 1 heart and holds Link; after 95 frames of holding the magic shield is gone | `# Like-Like` |
| Bubble `$2B` | 2 each in `$0A`, `$19`, `$1D`, `$2D`; 1 each in `$1B`, `$38`, `$3A` | cannot die | nothing | touch: no hearts, knockback, sword (A button) off for 151-160 frames | `# Bubble (flashing, red, blue)` |
| Zol `$13` | `$68` (5); 2 each in `$0A`, `$19`, `$1D` | `$20` | everything; a `$10` hit splits it into 2 Gels | touch 1 heart | `# Zol` |
| Vire `$12` | `$39` (5), `$2D` (3) | `$40` | everything; a hit under `$40` splits it into 2 red Keese | touch 1 heart (Keese half a heart) | `# Vire` |
| Keese, blue `$1B` | `$58` (8); 4 in each cellar | 0 | any hit, the boomerang included | touch half a heart | `# Keese (blue, red, black)` |
| Trap `$49` | `$48` (6 traps), `$2C` (4 traps, with 2 Blue and 2 Red Wizzrobes) | cannot die | nothing | charges along Link's row or column; touch 1 heart | `# Trap` |
| Statues | `$58` (4 sources) | - | nothing | aimed fireballs, half a heart; magic shield parries | `# Statue fireballs` |
| Gleeok `$44`, 3 heads | `$18` | `$A0` first head, `$60` each later one | sword, sword beam, rod swing | head touch 3/8 heart, flying head and fireball 1 heart | `# Gleeok` |
| Gohma `$34` | `$1C` (boss) | `$20` | an arrow flying up into the fully open eye | touch 2 hearts; fireball 1 heart, no shield parries it | `# Gohma (blue, red)` |
| Old man `$4C`, `$4D` | `$0B`, `$6A` | - | a hit turns on two fireball sources | blocks the way up past Y `$8D` | `# Old men and shopkeepers (cave and dungeon persons)` |

**Getting through the hard rooms**

- What "the room is cleared" means for shutters, hidden items and push blocks: no object of type `$01`-`$2A` or `$2E`-`$48` in slots `$01`-`$0C`. Bubbles, Traps and old men never count; Gels and Keese made by a split do. [`Secret_UpdateRoomAllDead`](../src/bank5/underworld-secrets.asm#L124)
- Red Wizzrobe: it can be hit on its 64 solid frames and on every other frame while fading; it fires on the 16th solid frame along the line it faces, which is the line to Link when it appeared. Step off that row or column after it appears, then hit it from the side. One magic sword hit, two white, four wooden, or one bomb. [`Wizzrobe_Red_Update`](../src/bank4/wizzrobe.asm#L550), [`Wizzrobe_Red_State2Solid`](../src/bank4/wizzrobe.asm#L667)
- Blue Wizzrobe: it shoots only when it already faces Link on a shared 16-pixel row, or on a column with Link's X equal to its X rounded down to `$10`; staying off its row and column and striking from the side avoids every shot. It takes no knockback and is immune for about 32 frames after each hit. [`Wizzrobe_Blue_TryShooting`](../src/bank4/wizzrobe.asm#L469), [`Wizzrobe_DrawAndCheckCollisions`](../src/bank4/wizzrobe.asm#L683)
- Like-Like: kill it within 95 frames of being caught and the magic shield stays; while held Link cannot walk or turn but A and B still work, and nothing else can touch him. A Like-Like held off with the boomerang (stun) cannot catch him. [`LikeLike_Update@IncCaptureTime`](../src/bank4/like-like.asm#L111), [`Link_Update`](../src/bank7/mode5-play-update.asm#L330)
- Bubble: in rooms `$38` and `$3A` a touch takes the sword away for up to 160 frames in a room where the sword is the main weapon; bombs still work meanwhile. [`Bubble_Update`](../src/bank4/bubble-keese.asm#L93), [`Link_HandleInput`](../src/bank5/link-handle-input.asm#L68)
- Rooms `$38` and `$3A` (1 Bubble, 3 Like-Likes, 2 Blue and 2 Red Wizzrobes) are the hardest on the route and both must be emptied of everything but the Bubble before their block moves. In `$38` Link is shut in until then. [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`Secret_Trigger4BlockShutters`](../src/bank5/underworld-secrets.asm#L215)
- Vires (rooms `$39`, `$2D`): the magic sword, its beam or a bomb kills one outright; anything weaker turns it into two red Keese, which must die too before the shutter opens or the key appears. Any hit kills a Keese. [`Vire_Update@SplitUp`](../src/bank4/vire.asm#L61), [`Keese_Update`](../src/bank4/bubble-keese.asm#L163)
- Room `$78` is a ringleader room: when the Red Wizzrobe in object slot `$01` dies the other four die with it. Nothing in the room depends on it; the route only walks through. [`Secret_Trigger2Ringleader`](../src/bank5/underworld-secrets.asm#L176)

**The boss: Gohma (type `$34`), room `$1C`**

- Link enters by the S key door. Gohma starts at X `$80`, Y `$70` with no spawn cloud and paces `$20` pixels out and back, left, right or down, always returning to that spot; it fires an aimed fireball every 65 frames. [`Gohma_Init`](../src/bank4/gleeok-manhandla-gohma-inits.asm#L221), [`Gohma_Update`](../src/bank4/gohma.asm#L63)
- It dies to one arrow (wooden or silver) that is flying up, overlaps the eye part, and arrives while the eye state is 3. The eye starts to open on Gohma's first update and is fully open for 96 frames of each opening; openings start 384 to 510 frames apart. [`Gohma_HandleWeaponCollision`](../src/bank4/gohma.asm#L367), [`Gohma_Update@AnimateEye`](../src/bank4/gohma.asm#L152), [`Gohma_Update@DecNextOpenEyeCounter`](../src/bank4/gohma.asm#L167)
- Aim: stand below it, face up, with Link's X from Gohma's `ObjX` - `$0A` to `ObjX` + 5 (the arrow starts 3 pixels right of Link and must be within `ObjX` - 7 to `ObjX` + 8); an arrow further left sparks on a leg and is wasted, and every arrow that touches any part is used up. The eye state is readable at [`Gohma_ObjEyeState`](../src/CommonVars.inc#L17). [`Gohma_CheckCollisions`](../src/bank4/gohma.asm#L322), [`Arrow_Wield`](../src/bank5/wield-items.asm#L187)
- Sword, beam, rod, magic shot and boomerang only make the parry sound; bombs and fire do nothing. With no arrow, no bow or no rupees the room cannot be cleared. [`Gohma_HandleWeaponCollision`](../src/bank4/gohma.asm#L367)
- When it dies the N shutter opens (to the Triforce room) and the heart container appears at X `$80`, Y `$90`; the kill is saved, so Gohma does not come back. The statues of this room's layout (`$06`) do not shoot. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61), [`RoomHistory_SaveKillCountUW`](../src/bank5/room-history-uw.asm#L110), [`Statue_RoomLayoutIds`](../src/bank4/statues.asm#L45)
- The boss roar is heard in rooms `$1B`, `$1D` and `$2C` while Gohma lives; it marks neighbours of the boss room, not danger in those rooms. [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42)

# Route

Established from the level data and the code; not walked in the running game. The path was found by a search over
the room graph of the rooms table (key doors cost a key once, shutters open by their room's secret, a bombable wall
costs a bomb, stairs as listed under the map) for the fewest room entries that take the rod, kill the boss and reach
the Triforce room. Positions are Link's `ObjX`, `ObjY`; floor row `r` of the floor plans is Link's Y `$5D` + `$10` x `r`,
floor column `c` is X `$20` + `$10` x `c`; the N and S doors are at X `$78`, the W and E doors at Y `$8D` ("the door
row" below is that floor row, row 3).

| step | room | entered from | what to do | leave by | keys after | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `$79` | overworld | nothing (no monsters) | E, open | 0 | start room |
| 2 | `$7A` | W | kill the 5 Red Wizzrobes; take the key that appears at `$80`,`$90` | W, open | 1 | dark; do not open the N key door |
| 3 | `$79` | E | walk across to the W door and unlock it | W, key door | 0 | |
| 4 | `$78` | E | walk through: west along the door row to X `$90`, up one row, west to X `$78`, up (blocks stand in the middle of the door row) | N, open | 0 | 5 Red Wizzrobes, need not be killed |
| 5 | `$68` | S | walk straight up at X `$78` | N, open | 0 | 5 Zols; compass if they are killed |
| 6 | `$58` | S | kill the 8 blue Keese: the N shutter opens and the key appears at `$80`,`$90`; take it | N, shutter | 1 | four statues shoot fireballs |
| 7 | `$48` | S | walk straight up at X `$78` without stopping | N, open | 1 | six Traps |
| 8 | `$38` | S (shuts behind) | kill the 3 Like-Likes, 2 Blue and 2 Red Wizzrobes; push the block at `$60`,`$90` (from `$50`,`$8D` pushing right, or from `$60`,`$9D` pushing up) for `$10` frames: both shutters open | N, shutter | 1 | 1 Bubble |
| 9 | `$28` | S (shuts behind) | walk east along the bottom floor row to X `$D0`, up to `$D0`,`$9D`, face up and place a bomb: it lies on the E wall's hotspot `$D0`,`$8D`; the wall is open about `$50` frames later; step up to `$D0`,`$8D` and walk east | E, bombed wall | 1 | 3 Blue and 2 Red Wizzrobes, need not be killed; needs 1 bomb |
| 10 | `$29` | W | from the door walk east: the ladder crosses the ring of water; take the key at `$80`,`$90`; leave the ring (north across the water at X `$80`, or back west the way he came and up the west edge), go to X `$78` on the top floor row, unlock the N door | N, key door | 2, then 1 | dark; needs the ladder for the key; 3 Blue and 2 Red Wizzrobes, need not be killed |
| 11 | `$19` | S | walk straight up at X `$78` and unlock the N door | N, key door | 0 | 2 Bubbles, 2 Zols, 2 Like-Likes; the doors are west of the water; map if cleared |
| 12 | `$09` | S | kill the 3 Blue and 2 Red Wizzrobes; push the block at `$60`,`$90` for `$10` frames; stairs appear at `$D0`,`$60`: stand on them (`$D0`,`$5D`) | stairs | 0 | |
| 13 | `$75` cellar | stairs | go down the ladder; take the rod at `$80`,`$90`; climb back out with Up | ladder | 0 | 4 blue Keese |
| 14 | `$09` | cellar, at `$C0`,`$8D` | walk to the S door | S, open now | 0 | the Wizzrobes stay dead |
| 15 | `$19` | N | walk straight down at X `$78` | S, open now | 0 | |
| 16 | `$29` | N | walk round the water ring by the outer floor (west or east edge) to the S door | S, open | 0 | dark |
| 17 | `$39` | N (shuts behind) | kill the 5 Vires and the red Keese they split into: the E shutter opens; on the door row walk east: the ladder crosses the water column at X `$A0` | E, shutter | 0 | dark; needs the ladder |
| 18 | `$3A` | W | kill the 3 Like-Likes, 2 Blue and 2 Red Wizzrobes; push the block at `$70`,`$90` (from `$60`,`$8D` pushing right) for `$10` frames; stairs appear at `$D0`,`$60` | stairs | 0 | 1 Bubble; the room has no other exit |
| 19 | `$08` cellar | stairs, at X `$30` | go down, walk right along the tunnel, climb the right-hand ladder (X `$C0`) with Up | right ladder | 0 | 4 blue Keese |
| 20 | `$1D` | cellar, at `$60`,`$9D` | walk to the S door | S, open | 0 | 2 Bubbles, 2 Zols, 2 Like-Likes, need not be killed |
| 21 | `$2D` | N | kill the 3 Vires and their Keese; take the key that appears at `$80`,`$90` | W, open | 1 | dark; 2 Bubbles; doors and key are west of the water |
| 22 | `$2C` | E | move off the east wall at once, walk to X `$78` and up; unlock the N door | N, key door | 0 | four corner Traps, 2 Blue and 2 Red Wizzrobes, need not be killed |
| 23 | `$1C` | S | kill Gohma with an arrow into the open eye; take the heart container at `$80`,`$90` | N, shutter | 0 | boss |
| 24 | `$0C` | S | walk straight up at X `$78` to the Triforce piece at `$78`,`$90` | level ends | 0 | no monsters |

- Key balance: found 4 (`$7A`, `$58`, `$29`, `$2D`), spent 4 (doors `$79` W, `$29` N, `$19` N, `$1C` S), never below 0, 0 left. The level's fifth key (`$1A`) and fifth key door (`$7A` N, to the old man of `$6A`) are not used. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160), [data/levelblock-uw-1to6-q1.md](../data/levelblock-uw-1to6-q1.md)
- Order that matters, 1: the key of `$7A` is the only key Link can reach before the W door of `$79`. Spending it on the N door of `$7A` leaves him with no way on except a key from outside (shops sell them, see `# Keys and Magic Key` in [items.md](../items.md)). [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- Order that matters, 2: take the ring key of `$29` on the first pass. Link arrives there with one key and needs two before the next key (`$2D`): the N door of `$29` and the N door of `$19`. [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- Order that matters, 3: the rod first, then the east side. Rooms `$39` onward are a one-way run to the boss (shutter behind in `$39`, stairs in `$3A`), and the Triforce piece ends the level. Coming back for the rod later means the whole route again. [`Item_TakeClass0PerLevel`](../src/bank1/take-item.asm#L421)
- Bomb wall detail: the test is made once, on the frame the blast ends (`$48` frames after the bomb is placed), with the bomb's own X and Y each less than `$18` from the hotspot (`$D0`,`$8D` for an E wall); the wall can be walked through 8 frames later and stays open from both sides. A bomb is placed `$10` pixels ahead of Link. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Bomb_WallHotspotsX`](../src/bank7/bomb.asm#L60), [`Link_PlaceWeaponAhead`](../src/bank1/bomb-candle-wield.asm#L154)
- Alternative without bombs (2 entries more, 25 after the start, and the Gleeok fight): from `$28` go N into `$18`, cut off Gleeok's three heads so that the E shutter opens, E into `$19`, unlock its S door, S into `$29` for the key, back N into `$19`, unlock its N door, then on as from step 12. Keys: the same four found and four spent (door `$19` S in place of the bombed wall's `$29` N; they are the same doorway). [`Gleeok_CheckSegmentCollisions`](../src/bank4/gleeok-head-body.asm#L128), [`Secret_Trigger1AllDead`](../src/bank5/underworld-secrets.asm#L152)
- With one key brought from outside, steps 2 and 3 fall away (21 entries after the start): go W from `$79` at once. [`InvKeys`](../src/Variables.inc#L584)
- After a death: the CONTINUE choice restarts Link in room `$79` with 3 hearts, without reloading the level. Unlocked doors, the bombed wall, taken keys and the rod are kept; shutters are shut again, blocks must be pushed again; a partly cleared room stays reduced, but because the room history is cleared a room that had been emptied is full again when he returns to it (a dead Gohma or Gleeok stays dead). [`Mode8Continue_Update_Full@HandleActivated`](../src/bank5/mode8-continue-mode10-stairs.asm#L143), [`RoomHistory_Clear`](../src/bank7/init-mode-dispatch.asm#L271), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)
- So after a death with the rod in hand the way back is steps 4 to 9 again, with the fights of `$58` and `$38` but no key or bomb to spend, then `$29` and S into `$39`. This level has no Wallmaster. [`Level_BlockUW1to6Q1`](../src/bank6/level-blocks-common-data.asm#L70)

# Optional detours

- Map, room `$19` (on the route, no extra room): kill its 2 Zols and 2 Like-Likes; costs a fight beside two Bubbles, changes only the status bar. [`Inv_HasMap`](../src/bank5/submenu-map.asm#L153)
- Compass, room `$68` (on the route, no extra room): kill its 5 Zols; a white or magic sword, the rod or a bomb kills a Zol without a split. [`Zol_CheckCollisions`](../src/bank4/zol-gel.asm#L170), [`Inv_HasCompass`](../src/bank5/submenu-map.asm#L143)
- 5 rupees, room `$28` (on the route, no extra room): kill its 3 Blue and 2 Red Wizzrobes; the same kills open the S shutter, which the route does not need. [`Secret_CheckUnderworld`](../src/bank5/underworld-secrets.asm#L61)
- Spare key, room `$1A` (2 entries from `$19`): cross the water column of `$19` with the ladder on the door row, E into dark `$1A`, take the key lying at `$80`,`$90` on the door row, and return; its 6 Red Wizzrobes need not be killed. Gains 1 key that this level has no use for; it carries to later levels. [`InvKeys`](../src/Variables.inc#L584), [`Link_EndMoveAndAnimate@SetUpLadder`](../src/bank7/link-animate-ladder.asm#L246)
- Bombs, room `$3C` (2 entries from `$2C`, S door): +4 bombs for killing 3 Blue and 2 Red Wizzrobes in a dark room of water channels, with the only door shut until they are dead. The item spot is reached from the door without the ladder (down the middle channel). Worth it only if bombs are short. [`Item_Take_CheckClass1`](../src/bank1/take-item.asm#L337), [`Door_TouchShutter`](../src/bank5/touch-door.asm#L126)
- Old man, room `$6A` (2 entries from `$7A`, costs 1 key): only the hint "AIM AT THE EYES OF GOHMA". The key is the one the route needs for the W door of `$79`: do not take this detour without a spare. [`Person_TextSelectorsLevels3468`](../src/bank1/underworld-person-init.asm#L134), [data/person-text.md](../data/person-text.md)
- Old man, room `$0B` (from `$19`: E across the water to `$1A`, E to `$1B`, clear `$1B` to open its N shutter, N): only the hint "THERE ARE SECRETS WHERE FAIRIES DON'T LIVE". Room `$1B` holds 1 Bubble, 3 Like-Likes, 2 Blue and 2 Red Wizzrobes. [`Person_UpdateUnderworld_Full`](../src/bank1/underworld-person-update.asm#L44)
- Rooms `$0A` and `$18` hold no item. `$0A` joins `$1A` (N door, across that room's water row: ladder), `$09` and `$0B` by bombable walls; it is a second way to the rod that costs a bomb and the fight of `$0A` instead of a key. `$18` is the Gleeok room of the no-bomb alternative. [`Door_TouchBombable`](../src/bank5/touch-door.asm#L108)
- The stairs of room `$1D` (X `$D0`, Y `$8D`, in the layout from the start) lead down to cellar `$08` and so back to `$3A`: the only way back west from the boss side, since `$3A`'s own stairs need its block pushed again. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)

# Hazards and rules that matter here

- Shutters open only by their room's secret and are shut again on every entry; nothing about a shutter is saved. Rooms with a shutter side: `$0A`, `$18`, `$1B`, `$1C`, `$28`, `$39`, `$3C`, `$58` (secret: all dead) and `$38` (secret: push block). See `# Doorways, doors, keys, shutters and bomb walls` in [mechanics.md](../mechanics.md). [`Shutter_OpenTriggered`](../src/bank5/shutters-door-flags.asm#L68), [`Room_EnterCommon@Method2`](../src/bank5/enter-room.asm#L211)
- Ten doorways are a shutter on one side and an open door on the other (list in the facts table). From the open side Link walks in freely and the shutter then shuts behind him; from the shutter side he needs the room's secret. On the route he walks freely into `$38` (from `$48`), `$28` (from `$38`, once its block is pushed) and `$39` (from `$29`), and must solve `$58`, `$38`, `$39` and `$1C` to go on. [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55), [`Door_TouchShutter`](../src/bank5/touch-door.asm#L126)
- Push blocks (`$09`, `$38`, `$3A`): the block is the leftmost block square of the middle floor row (X `$60` in `$09` and `$38`, X `$70` in `$3A`). It moves only when the room is cleared, with Link exactly in line (same X, or his Y + 3 = block Y), at most `$10` pixels away, holding exactly the one direction toward it for `$10` frames in a row; it slides `$10` pixels in `$20` frames, once per room entry. See `# Push blocks and room secrets` in [mechanics.md](../mechanics.md). [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L234), [`PushBlock_State0Idle`](../src/bank4/push-block.asm#L74), [`PushBlock_State1Moving`](../src/bank4/push-block.asm#L165)
- Block stairs (`$09`, `$3A`) always appear at X `$D0`, Y `$60`, whichever way the block was pushed; Link takes them standing at X `$D0`, Y `$5D` at rest on the grid. They are not saved. [`Secret_Trigger5BlockStairs`](../src/bank5/underworld-secrets.asm#L226), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
- Cellars: tunnel `$08` joins `$3A` (left ladder, X below `$80`) and `$1D` (right ladder); Link arrives at X `$30` coming from `$3A` and at X `$C0` coming from `$1D`, and comes out at `$60`,`$9D` in either room. Treasure cellar `$75` returns to `$09` at `$C0`,`$8D` by either ladder. Each cellar holds four blue Keese whatever its data says. See `# Stairs, cellars and caves` in [mechanics.md](../mechanics.md). [`Mode9Cellar_LinkStartXs`](../src/bank5/cave-cellar-enter.asm#L262), [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43), [`Spawn_AssignPositions`](../src/bank5/spawn-positions.asm#L64)
- Bomb walls: `$09` E / `$0A` W, `$0A` E / `$0B` W, `$28` E / `$29` W. They look like plain wall until opened. The bomb must lie within `$18` pixels of the wall's hotspot (W `$20`,`$8D`; E `$D0`,`$8D`) when its blast ends. [`Bomb_TryOpenWall`](../src/bank7/bomb.asm#L136), [`Bomb_WallHotspotsY`](../src/bank7/bomb.asm#L64)
- False walls: none in this level. [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55)
- Water and the ladder: water is tile `$F4`; with `InvLadder` Link lays the ladder by walking into it from a grid point outside a doorway, holding the direction he faces, and it spans one square. It needs a free object slot among `$01`-`$0B`. Rooms: `$19`, `$2D`, `$39` a column at X `$A0` (doors N, S, W on the west side, E on the east); `$1A` a row at Y `$80` (N door beyond it); `$29` a ring one square wide; `$3C` channels. See `# Ladder` in [items.md](../items.md). [`Link_EndMoveAndAnimate@CheckLadder`](../src/bank7/link-animate-ladder.asm#L209), [`Ladder_CheckAndDraw`](../src/bank5/ladder.asm#L44), [`Obj_FindEmptyMonsterSlot`](../src/bank7/object-stubs-destroy-monster.asm#L169)
- Water does not stop this level's monsters much: Blue Wizzrobes teleport through water and blocks, Red Wizzrobes appear on any walkable square near Link, Keese fly, and magic shots pass over everything. [`Wizzrobe_Blue_MoveAndCheckTile`](../src/bank4/wizzrobe.asm#L174), [`Wizzrobe_Red_AppearOffsetsX`](../src/bank4/wizzrobe.asm#L577), [`Wizzrobe_ShootMagicShot`](../src/bank4/wizzrobe.asm#L534)
- Dark rooms (`$1A`, `$29`, `$2D`, `$39`, `$3C`, `$7A`): only the palettes are black; coordinates, tiles and monsters are as in the floor plans. A standing candle or book fire starts a fade to light during which the whole play update is frozen; the room is dark again on the next entry. [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124), [`Candle_State0Begin`](../src/bank4/shoot-helpers-candle.asm#L147), [`Mode5Play_Update`](../src/bank7/mode5-play-update.asm#L59)
- Traps, room `$48` (six: four corners plus `$40`,`$8D` and `$B0`,`$8D`): a trap charges at up to 1.75 pixels per frame toward the middle when Link is within `$0E` pixels of its row (or, failing that, its column) and creeps back at 0.5; it cannot be hurt. Walking the middle column X `$78` crosses three trap rows: keep moving, or step into a row, step back, and pass while the pair creeps home. [`Trap_Update_Full`](../src/bank1/trap-rupee-stash.asm#L189), [`Trap_StartXs`](../src/bank1/trap-rupee-stash.asm#L112), [`Trap_AllowedDirs`](../src/bank1/trap-rupee-stash.asm#L167)
- Traps, room `$2C` (four corners): Link comes in by the E door, in the column of the two east traps, which then charge down and up toward the door row; move west at once. The N door is a key door in the top traps' row: the unlock holds Link there for 8 frames, so bait the top pair first. [`Trap_Init_Full`](../src/bank1/trap-rupee-stash.asm#L134), [`Door_TouchKey`](../src/bank5/touch-door.asm#L160)
- Statues shoot only in room `$58` (layout `$24`, four sources near the corners): aimed fireballs, half a heart, at random intervals of 81 to 241 frames per statue, none while Link is within `$18` pixels of the source. The statues of `$79`, `$1C` and `$0C` are scenery. [`Statue_RoomLayoutIds`](../src/bank4/statues.asm#L45), [`Statue_Update`](../src/bank4/statues.asm#L86)
- Bubbles (`$0A`, `$19`, `$1B`, `$1D`, `$2D`, `$38`, `$3A`; all type `$2B`): a touch costs no hearts but knocks Link back, zeroes the help-drop counters and disables the A button for 151-160 frames; it wears off by itself. [`Bubble_Update`](../src/bank4/bubble-keese.asm#L93), [`SwordBlockedLongTimer`](../src/Variables.inc#L62)
- Like-Likes (`$0A`, `$19`, `$1B`, `$1D`, `$38`, `$3A`): the catch needs a touch that harms Link; 95 frames later the magic shield is gone for good. The route must fight the three of `$38` and the three of `$3A`; those of `$19` and `$1D` can be walked past. [`Collision_CheckMonsterVsWeaponsAndLink@CheckCaptors`](../src/bank1/monster-collisions.asm#L151), [`LikeLike_Update@IncCaptureTime`](../src/bank4/like-like.asm#L111)
- Wizzrobe shots (types `$58`, `$59`) are parried only by the magic shield, with Link facing the shot and not swinging or using an item; they fly 3 pixels per frame over blocks and water. [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](../src/bank1/link-collision-harm.asm#L138), [`InvMagicShield`](../src/Variables.inc#L604)
- Old men (`$0B`: "THERE ARE SECRETS WHERE FAIRIES DON'T LIVE."; `$6A`: "AIM AT THE EYES OF GOHMA."): in level 6 they only talk. Link is halted while the text types and cannot walk up past Y `$8D` in their room. Hitting one does not kill him: it turns on two fireball sources (`$48`,`$82` and `$A8`,`$86`) until Link leaves the room. [`Person_UpdateUnderworld_Full`](../src/bank1/underworld-person-update.asm#L44), [`Person_CheckCollisions`](../src/bank1/underworld-person-update.asm#L89), [`Walker_Move@CheckPersonBlocking`](../src/bank7/walker-move.asm#L149)
- Not in this level: Wallmasters, the hungry Goriya, false walls, the life-or-money man, the more-bombs man. No room of level 6 is tested by number in the code; the only number test near it is overworld room `$22`'s entrance X rule. [`Level_BlockUW1to6Q1`](../src/bank6/level-blocks-common-data.asm#L70), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
- Kill counts: leaving a room banks its kills; while the room stays in the six-room history, or is only partly cleared, it comes back with that many fewer monsters, which is why `$09` is still empty when Link climbs out of the cellar. See `# Monster spawning, edge spawns and respawn (room history)` in [mechanics.md](../mechanics.md). [`RoomHistory_SaveKillCountUW`](../src/bank5/room-history-uw.asm#L110), [`RoomHistory_ModifyObjCountUW`](../src/bank5/room-history-uw.asm#L50)

# Second quest

- In Q2 the dungeons load a different level block, so internal level 6 (still shown as LEVEL-6) is another dungeon: start room `$74`, Triforce room `$16`, boss room `$26`; its rooms are the rows with `lvl` = `6` in [data/levelblock-uw-1to6-q2.md](../data/levelblock-uw-1to6-q2.md) and its level info is `# Level 6, Q2` in [data/levelinfo.md](../data/levelinfo.md). [`Level_BlockAddrsQ2`](../src/bank6/level-load-mode2.asm#L89), [`Quest_Q2LevelInfoPatch6`](../src/bank6/quest2-replacements.asm#L145)
- In Q2 overworld room `$22` is patched to lead to a rupee cave (`# Rooms patched for the second quest` in [data/levelblock-ow.md](../data/levelblock-ow.md)); room `$30`, a Q2 recorder secret room, keeps the level 6 destination. [`Quest_Q2OverworldPatchValues`](../src/bank6/quest2-replacements.asm#L64), [`Flute_SecretRoomIdsOW`](../src/bank7/flute.asm#L36)
- This document covers the first quest only. [`Level_BlockAddrsQ1`](../src/bank6/level-load-mode2.asm#L57)
