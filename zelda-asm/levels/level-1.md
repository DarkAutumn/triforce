# Level 1

(one-paragraph summary to be written)

Contents (generated; `L` = line in this file):

<!-- toc -->
- L21 At a glance
- L49 What Link must bring
- L51 What Link gets here
- L53 Map
- L80 Rooms
- L107 Floor plans
- L303 Monsters and boss
- L305 Route
- L307 Optional detours
- L309 Hazards and rules that matter here
- L311 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 1 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-1 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | $37 | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L175) |
| Start room | $73 | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Triforce room | $36 (room item: Triforce piece ($1B)); the field places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $35 (1 x Aquamentus ($3D)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | $7F | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 17: 17 on the submenu map | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (doorways; room and side) | 6 ($22 E, $23 S, $35 S, $42 S, $43 E, $63 S) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (doorways; room and side) | 2 ($43 S, $44 S) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (doorways; room and side) | 3 ($35 E, $41 E, $52 E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (doorways; room and side) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $35 E shutter / $36 open, $41 E open / $42 shutter, $52 E shutter / $53 open | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $73 S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 2 ($22, $42) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 0 (none) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 6 ($23, $33, $45, $53, $72, $74) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | bow ($0A): $7F (cellar); compass ($16): $54; map ($17): $43; heart container ($1A): $35; Triforce piece ($1B): $36; wooden boomerang ($1D): $44 | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md) (rows with `lvl` = `1` or `c1`).
<!-- /generated:facts -->

# What Link must bring

# What Link gets here

# Map

<!-- generated:map -->
```
         [22  ] K [23k ]
                    K
                  [33k ]            [35Bi]S/-[36T ]
                    |                 K
[41  ]-/S[42  ]---[43i ] K [44i ]---[45k ]
           K        X        X
         [52  ]S/-[53k ]---[54i ]
                    |
                  [63  ]
                    K
         [72k ]---[73S ]---[74k ]

[73S] = room $73 and its marks: S start room, B boss room, T Triforce room,
        k a key lies here, i another room item lies here, h not shown on the submenu map.
Between two rooms, the door: --- or | open, K key door, S shutter, X bombable wall,
        F false wall (walk-through), nothing = wall. A/B = the two sides differ: the left
        or upper room's side first, . = wall on that side.
Doors on sides with no room of the level: $73 S open.
```

- Cellar `$7F` (treasure cellar): both exits lead to `$22`, whose stairs lead here; item: bow ($0A). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md)
<!-- /generated:map -->

# Rooms

Generated from the level data by `tools/romdata.py`; do not edit between the markers.

<!-- generated:rooms -->
| room | N | S | W | E | monsters | item | secret | block | dark | layout | floor | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $22 | wall | wall | wall | key | 1 x Trap ($4A) | - | - | B | - | $1A | blocks, stairs | stairs lead to cellar $7F |
| $23 | wall | key | key | wall | 3 x Goriya ($06) | key ($19) at $80,$70 | all dead: shutters open, room item appears | - | - | $15 | water | - |
| $33 | key | open | wall | wall | 3 x Stalfos ($2A) | key ($19) at $80,$90 | - | - | - | $17 | water | - |
| $35 boss | wall | key | wall | shutter | 1 x Aquamentus ($3D) | heart container ($1A) at $C0,$90 | all dead: shutters open, room item appears | - | - | $04 | blocks | - |
| $36 Triforce | wall | wall | open | wall | none | Triforce piece ($1B) at $80,$90 | - | - | - | $29 | blocks, statues | - |
| $41 | wall | wall | wall | open | 1 x dungeon person ($4C) | - | - | - | - | $26 | blank | person $4C says: EASTMOST PENNINSULA / IS THE SECRET. |
| $42 | wall | key | shutter | open | 3 x Gel ($15) | - | push block: shutters open | B | - | $22 | blocks | - |
| $43 | open | bomb | open | key | 5 x Gel ($15) | map ($17) at $C0,$90 | - | - | - | $1F | blocks | - |
| $44 | wall | bomb | key | open | 3 x Goriya ($06) | wooden boomerang ($1D) at $80,$70 | all dead: shutters open, room item appears | - | - | $02 | blocks | - |
| $45 | key | wall | open | wall | 8 x Wallmaster ($27) | key ($19) at $A0,$C0 | - | - | - | $0D | blocks | boss sound $10 |
| $52 | key | wall | wall | shutter | 6 x Keese blue ($1B) | - | all dead: shutters open | - | - | $1E | blocks | - |
| $53 | bomb | open | open | open | 5 x Stalfos ($2A) | key ($19) at $80,$70 | all dead: shutters open, room item appears | - | - | $02 | blocks | - |
| $54 | bomb | wall | open | wall | 8 x Keese blue ($1B) | compass ($16) at $C0,$90 | - | - | - | $03 | blocks | - |
| $63 | open | key | wall | wall | 3 x Stalfos ($2A) | - | - | - | - | $1E | blocks | - |
| $72 | wall | wall | wall | open | 3 x Keese blue ($1B) | key ($19) at $A0,$C0 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $73 start | key | open | open | open | none | - | - | - | - | $21 | statues, sand | - |
| $74 | wall | wall | open | wall | 5 x Stalfos ($2A) | key ($19) at $C0,$90 | - | - | - | $1D | blocks | - |
| $7F cellar | - | - | - | - | 4 x Keese blue ($1B) | bow ($0A) at $80,$90 | - | - | - | $3F | treasure cellar | left exit to $22, right exit to $22; Link comes out at $60,$AD |
<!-- /generated:rooms -->

# Floor plans

<!-- generated:floors -->
One character per floor square, 12 columns x 7 rows inside the walls (left column = west, top row =
north): `.` floor (tile `$74`), `B` block (tile `$B0`; a push block object is placed on one in a room
with a push block), `W` water (tile `$F4`), `S` statue (tiles `$94` and `$B4`), `>` stairs (tile `$70`),
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L57), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L67), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L228), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
Stairs that a secret makes appear are not part of the layout.

```
layout $00  rooms: $72
............
............
............
............
............
............
............
```

```
layout $02  rooms: $44,$53
............
............
..B......B..
............
..B......B..
............
............
```

```
layout $03  rooms: $54
............
..B......B..
............
............
............
..B......B..
............
```

```
layout $04  rooms: $35
.......BBBBB
.........BBB
..........BB
............
..........BB
.........BBB
.......BBBBB
```

```
layout $0D  rooms: $45
............
.B.B.BB.B.B.
............
.B.B.BB.B.B.
............
.B.B.BB.B.B.
............
```

```
layout $15  rooms: $23
............
.WWW....WWW.
.W........W.
.W.WWWWWW.W.
.W........W.
.WWW....WWW.
............
```

```
layout $17  rooms: $33
WWWWW..WWWWW
W...W....W.W
W.W.W.WW.W.W
..W.W..W.W..
W.W.WW.W.W.W
W.W....W...W
WWWWW..WWWWW
```

```
layout $1A  rooms: $22
............
......B.....
.....B.B....
....B.>.B...
.....B.B....
......B.....
............
```

```
layout $1D  rooms: $74
............
............
..BB....BB..
..BB....BB..
..BB....BB..
............
............
```

```
layout $1E  rooms: $52,$63
............
............
.....BB.....
.....BB.....
.....BB.....
............
............
```

```
layout $1F  rooms: $43
............
.BB......BB.
............
.....BB.....
............
.BB......BB.
............
```

```
layout $21  rooms: $73
............
.S..S..S..S.
............
.S..S..S..S.
.....::.....
.S.:S::S:.S.
...::::::...
```

```
layout $22  rooms: $42
............
............
............
.....B......
............
............
............
```

```
layout $26  rooms: $41
____________
____________
____________
____________
____________
____________
____________
```

```
layout $29  rooms: $36
............
.BBBBBBBBBB.
.B..S..S..B.
.B.S....S.B.
.B........B.
.BBBB..BBBB.
............
```

```
layout $3F (treasure cellar)  rooms: $7F
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
