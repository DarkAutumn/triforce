# Level 3

(one-paragraph summary to be written)

Contents (generated; `L` = line in this file):

<!-- toc -->
- L21 At a glance
- L49 What Link must bring
- L51 What Link gets here
- L53 Map
- L80 Rooms
- L108 Floor plans
- L293 Monsters and boss
- L295 Route
- L297 Optional detours
- L299 Hazards and rules that matter here
- L301 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 3 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-3 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | $74 | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L175) |
| Start room | $7C | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Triforce room | $3D (room item: Triforce piece ($1B)); the field places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $4D (1 x Manhandla ($3C)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | $0F (not in the cellar list; the search that has no end test finds it 72 bytes after the list's start) | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 18: 18 on the submenu map | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (doorways; room and side) | 4 ($3B S, $4A E, $4B E, $59 E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (doorways; room and side) | 2 ($4C E, $5B E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (doorways; room and side) | 6 ($2A E, $2B S, $3D S, $4D S, $59 S, $5C E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (doorways; room and side) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $2A E shutter / $2B open, $2B S open / $3B shutter, $3D S open / $4D shutter, $59 S shutter / $69 open | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $7C S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 1 ($3B) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 0 (none) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 5 ($2A, $49, $4B, $6B, $7B) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | bomb ($00): $4A, $5B, $69; raft ($0C): $0F (cellar); 5 rupees ($0F): $5D; compass ($16): $5A; map ($17): $4C; heart container ($1A): $4D; Triforce piece ($1B): $3D | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md) (rows with `lvl` = `3` or `c3`).
<!-- /generated:facts -->

# What Link must bring

# What Link gets here

# Map

<!-- generated:map -->
```
         [2Ak ]S/-[2B  ]
                   |/S
                  [3B  ]            [3DT ]
                    K                |/S
[49k ]---[4Ai ] K [4Bk ] K [4Ci ] X [4DBi]
  |        |        |        |        S
[59  ] K [5Ai ]---[5Bi ] X [5C  ] S [5Di ]
 S/|                |
[69i ]            [6Bk ]
                    |
                  [7Bk ]---[7CS ]

[73S] = room $73 and its marks: S start room, B boss room, T Triforce room,
        k a key lies here, i another room item lies here, h not shown on the submenu map.
Between two rooms, the door: --- or | open, K key door, S shutter, X bombable wall,
        F false wall (walk-through), nothing = wall. A/B = the two sides differ: the left
        or upper room's side first, . = wall on that side.
Doors on sides with no room of the level: $7C S open.
```

- Cellar `$0F` (treasure cellar): both exits lead to `$69`, whose stairs lead here; not in the level's cellar list; item: raft ($0C). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md)
<!-- /generated:map -->

# Rooms

Generated from the level data by `tools/romdata.py`; do not edit between the markers.

<!-- generated:rooms -->
| room | N | S | W | E | monsters | item | secret | block | dark | layout | floor | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $2A | wall | wall | wall | shutter | 8 x Keese blue ($1B) | key ($19) at $80,$90 | all dead: shutters open | - | - | $1D | blocks | - |
| $2B | wall | open | open | wall | 1 x dungeon person ($4B) | - | - | - | - | $26 | blank | person $4B says: DID YOU GET THE SWORD / FROM THE OLD MAN ON / TOP OF THE WATERFALL? |
| $3B | shutter | key | wall | wall | 1 x Trap ($4A), 3 x (empty) ($00), 2 x Zol ($13) | - | push block: shutters open | B | - | $0A | blocks | - |
| $3D Triforce | wall | open | wall | wall | none | Triforce piece ($1B) at $80,$90 | - | - | - | $29 | blocks, statues | - |
| $49 | wall | open | wall | open | 3 x Bubble ($2B), 2 x Zol ($13), 3 x Keese blue ($1B) | key ($19) at $80,$90 | - | - | - | $00 | plain | - |
| $4A | wall | open | open | key | 5 x Keese blue ($1B) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $0F | blocks | - |
| $4B | key | open | key | key | 3 x Zol ($13) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $08 | blocks | - |
| $4C | wall | open | key | bomb | 1 x Trap ($4A), 3 x (empty) ($00), 2 x Zol ($13) | map ($17) at $80,$90 | - | - | - | $00 | plain | boss sound $40 |
| $4D boss | shutter | shutter | bomb | wall | 1 x Manhandla ($3C) | heart container ($1A) at $80,$90 | all dead: shutters open, room item appears | - | - | $25 | sand | - |
| $59 | open | shutter | wall | key | 5 x Darknut red ($0B) | - | all dead: shutters open | - | - | $1D | blocks | - |
| $5A | open | wall | key | open | 1 x Trap ($4A), 3 x (empty) ($00), 4 x Keese blue ($1B) | compass ($16) at $80,$90 | - | - | - | $00 | plain | - |
| $5B | open | open | open | bomb | 3 x Darknut red ($0B) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $5C | open | wall | bomb | shutter | 3 x Darknut red ($0B) | - | all dead: shutters open | - | - | $1F | blocks | - |
| $5D | shutter | wall | shutter | wall | 3 x Bubble ($2B), 2 x Zol ($13), 3 x Keese blue ($1B) | 5 rupees ($0F) at $20,$60 | all dead: shutters open, room item appears | - | - | $1E | blocks | boss sound $40 |
| $69 | open | wall | wall | wall | 8 x Darknut red ($0B) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $1B | blocks, stairs | stairs lead to cellar $0F |
| $6B | open | open | wall | wall | 5 x Zol ($13) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $11 | blocks | - |
| $7B | open | wall | wall | open | 6 x Zol ($13) | key ($19) at $80,$90 | - | - | - | $03 | blocks | - |
| $7C start | wall | open | open | wall | none | - | - | - | - | $21 | statues, sand | - |
| $0F cellar | - | - | - | - | 4 x Keese blue ($1B) | raft ($0C) at $80,$90 | - | - | - | $3F | treasure cellar | left exit to $69, right exit to $69; Link comes out at $60,$8D |
<!-- /generated:rooms -->

# Floor plans

<!-- generated:floors -->
One character per floor square, 12 columns x 7 rows inside the walls (left column = west, top row =
north): `.` floor (tile `$74`), `B` block (tile `$B0`; a push block object is placed on one in a room
with a push block), `W` water (tile `$F4`), `S` statue (tiles `$94` and `$B4`), `>` stairs (tile `$70`),
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L57), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L67), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L228), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
Stairs that a secret makes appear are not part of the layout.

```
layout $00  rooms: $49,$4C,$5A,$5B
............
............
............
............
............
............
............
```

```
layout $03  rooms: $7B
............
..B......B..
............
............
............
..B......B..
............
```

```
layout $08  rooms: $4B
............
.BBBBBBBBBB.
..........B.
..........B.
..........B.
.BBBBBBBBBB.
............
```

```
layout $0A  rooms: $3B
............
............
............
....B..B....
............
............
............
```

```
layout $0F  rooms: $4A
............
............
BBBBBBBBBBBB
............
BBBBBBBBBBBB
............
............
```

```
layout $11  rooms: $6B
..B........B
.B...B....B.
....B....B..
...B....B...
..B....B....
.B....B...B.
B........B..
```

```
layout $1B  rooms: $69
........BBBB
........BBBB
.........BBB
...........>
.........BBB
........BBBB
........BBBB
```

```
layout $1D  rooms: $2A,$59
............
............
..BB....BB..
..BB....BB..
..BB....BB..
............
............
```

```
layout $1E  rooms: $5D
............
............
.....BB.....
.....BB.....
.....BB.....
............
............
```

```
layout $1F  rooms: $5C
............
.BB......BB.
............
.....BB.....
............
.BB......BB.
............
```

```
layout $21  rooms: $7C
............
.S..S..S..S.
............
.S..S..S..S.
.....::.....
.S.:S::S:.S.
...::::::...
```

```
layout $25  rooms: $4D
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
```

```
layout $26  rooms: $2B
____________
____________
____________
____________
____________
____________
____________
```

```
layout $29  rooms: $3D
............
.BBBBBBBBBB.
.B..S..S..B.
.B.S....S.B.
.B........B.
.BBBB..BBBB.
............
```

```
layout $3F (treasure cellar)  rooms: $0F
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
