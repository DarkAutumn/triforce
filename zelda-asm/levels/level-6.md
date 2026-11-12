# Level 6

(one-paragraph summary to be written)

Contents (generated; `L` = line in this file):

<!-- toc -->
- L21 At a glance
- L49 What Link must bring
- L51 What Link gets here
- L53 Map
- L86 Rooms
- L122 Floor plans
- L379 Monsters and boss
- L381 Route
- L383 Optional detours
- L385 Hazards and rules that matter here
- L387 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 6 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-6 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | $22; $30 | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L175) |
| Start room | $79 | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Triforce room | $0C (room item: Triforce piece ($1B)); the field places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $1C (1 x Gohma ($34)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | $08, $75 | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 25: 25 on the submenu map | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (doorways; room and side) | 5 ($09 S, $19 S, $1C S, $6A S, $78 E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (doorways; room and side) | 3 ($09 E, $0A E, $28 E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (doorways; room and side) | 11 ($0A S, $0B S, $0C S, $18 S, $18 E, $28 S, $29 S, $2C S, $38 S, $39 E, $48 S) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (doorways; room and side) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $0A S shutter / $1A open, $0B S open / $1B shutter, $0C S open / $1C shutter, $18 S shutter / $28 open, $18 E shutter / $19 open, $29 S open / $39 shutter, $2C S open / $3C shutter, $38 S shutter / $48 open, $39 E shutter / $3A open, $48 S open / $58 shutter | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $79 S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 3 ($09, $38, $3A) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 6 ($1A, $29, $2D, $39, $3C, $7A) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 5 ($1A, $29, $2D, $58, $7A) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | bomb ($00): $3C; 5 rupees ($0F): $28; rod ($10): $75 (cellar); compass ($16): $68; map ($17): $19; heart container ($1A): $1C; Triforce piece ($1B): $0C | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md) (rows with `lvl` = `6` or `c6`).
<!-- /generated:facts -->

# What Link must bring

# What Link gets here

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
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L57), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L67), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L228), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
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

# Route

# Optional detours

# Hazards and rules that matter here

# Second quest
