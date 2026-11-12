# Level 4

(one-paragraph summary to be written)

Contents (generated; `L` = line in this file):

<!-- toc -->
- L21 At a glance
- L49 What Link must bring
- L51 What Link gets here
- L53 Map
- L84 Rooms
- L114 Floor plans
- L332 Monsters and boss
- L334 Route
- L336 Optional detours
- L338 Hazards and rules that matter here
- L340 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 4 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-4 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | $45 | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L175) |
| Start room | $71 | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Triforce room | $03 (room item: Triforce piece ($1B)); the field places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $13 (1 x Gleeok ($43)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | $60 | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 20: 19 on the submenu map, 1 the map does not show ($11) | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (doorways; room and side) | 5 ($00 E, $20 S, $21 S, $30 E, $61 E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (doorways; room and side) | 4 ($01 S, $10 E, $11 S, $11 E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (doorways; room and side) | 5 ($00 S, $03 S, $10 S, $12 E, $31 E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (doorways; room and side) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $00 S open / $10 shutter, $03 S open / $13 shutter, $10 S shutter / $20 open, $31 E shutter / $32 open | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $71 S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 2 ($12, $32) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 7 ($01, $02, $21, $30, $40, $50, $62) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 4 ($01, $40, $51, $70) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | ladder ($0D): $60 (cellar); compass ($16): $62; map ($17): $21; heart container ($1A): $13; Triforce piece ($1B): $03 | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md) (rows with `lvl` = `4` or `c4`).
<!-- /generated:facts -->

# What Link must bring

# What Link gets here

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
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L57), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L67), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L228), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
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

# Route

# Optional detours

# Hazards and rules that matter here

# Second quest
