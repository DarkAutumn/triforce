# Level 5

(one-paragraph summary to be written)

Contents (generated; `L` = line in this file):

<!-- toc -->
- L21 At a glance
- L49 What Link must bring
- L51 What Link gets here
- L53 Map
- L86 Rooms
- L120 Floor plans
- L355 Monsters and boss
- L357 Route
- L359 Optional detours
- L361 Hazards and rules that matter here
- L363 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 5 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-5 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | $0B; $1B (tile object: $62 rock at $C0,$90, secret works in Q2 only) | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L175) |
| Start room | $76 | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Triforce room | $14 (room item: Triforce piece ($1B)); the field places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $24 (1 x Digdogger ($39)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | $07, $04 | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 23: 22 on the submenu map, 1 the map does not show ($15) | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (doorways; room and side) | 6 ($05 E, $06 S, $24 E, $26 E, $46 S, $66 E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (doorways; room and side) | 5 ($15 S, $16 E, $64 E, $65 E, $67 S) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (doorways; room and side) | 5 ($14 S, $24 S, $55 S, $55 E, $56 S) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (doorways; room and side) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $14 S open / $24 shutter, $24 S open / $34 shutter, $55 S open / $65 shutter, $56 S open / $66 shutter | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $76 S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 3 ($05, $06, $64) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 5 ($27, $37, $46, $57, $66) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 7 ($16, $26, $27, $47, $55, $66, $77) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | bomb ($00): $56, $65; flute ($05): $04 (cellar); 5 rupees ($0F): $57; compass ($16): $37; map ($17): $46; heart container ($1A): $24; Triforce piece ($1B): $14 | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md) (rows with `lvl` = `5` or `c5`).
<!-- /generated:facts -->

# What Link must bring

# What Link gets here

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
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L57), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L67), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L228), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
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

# Route

# Optional detours

# Hazards and rules that matter here

# Second quest
