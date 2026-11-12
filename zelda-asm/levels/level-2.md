# Level 2

(one-paragraph summary to be written)

Contents (generated; `L` = line in this file):

<!-- toc -->
- L21 At a glance
- L49 What Link must bring
- L51 What Link gets here
- L53 Map
- L82 Rooms
- L109 Floor plans
- L251 Monsters and boss
- L253 Route
- L255 Optional detours
- L257 Hazards and rules that matter here
- L259 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 2 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-2 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | $3C | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L175) |
| Start room | $7D | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Triforce room | $0D (room item: Triforce piece ($1B)); the field places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $0E (1 x Dodongo ($32)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | none | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 18: 18 on the submenu map | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (doorways; room and side) | 3 ($2E E, $5E E, $6E E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (doorways; room and side) | 5 ($1E E, $2F S, $3F S, $4F S, $5F S) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (doorways; room and side) | 5 ($0D E, $0E S, $1E S, $4E S, $6C E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (doorways; room and side) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $0D E open / $0E shutter, $0E S open / $1E shutter, $4E S shutter / $5E open, $6C E open / $6D shutter | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $7D S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 0 (none) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 0 (none) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 4 ($3E, $4E, $6C, $7E) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | bomb ($00): $1E, $3F; 5 rupees ($0F): $2F; compass ($16): $6F; map ($17): $5F; heart container ($1A): $0E; Triforce piece ($1B): $0D; magic boomerang ($1E): $4F | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-1to6-q1.md`](../data/levelblock-uw-1to6-q1.md) (rows with `lvl` = `2` or `c2`).
<!-- /generated:facts -->

# What Link must bring

# What Link gets here

# Map

<!-- generated:map -->
```
         [0DT ]-/S[0EBi]
                   |/S
                  [1Ei ] X [1F  ]
                    S        |
                  [2E  ] K [2Fi ]
                    |        X
                  [3Ek ]---[3Fi ]
                    |        X
                  [4Ek ]---[4Fi ]
                   S/|       X
                  [5E  ] K [5Fi ]
                    |        X
[6Ck ]-/S[6D  ]---[6E  ] K [6Fi ]
           |        |
         [7DS ]---[7Ek ]

[73S] = room $73 and its marks: S start room, B boss room, T Triforce room,
        k a key lies here, i another room item lies here, h not shown on the submenu map.
Between two rooms, the door: --- or | open, K key door, S shutter, X bombable wall,
        F false wall (walk-through), nothing = wall. A/B = the two sides differ: the left
        or upper room's side first, . = wall on that side.
Doors on sides with no room of the level: $7D S open.
```
<!-- /generated:map -->

# Rooms

Generated from the level data by `tools/romdata.py`; do not edit between the markers.

<!-- generated:rooms -->
| room | N | S | W | E | monsters | item | secret | block | dark | layout | floor | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $0D Triforce | wall | wall | wall | open | none | Triforce piece ($1B) at $80,$90 | - | - | - | $29 | blocks, statues | - |
| $0E boss | wall | open | shutter | wall | 1 x Dodongo ($32) | heart container ($1A) at $80,$90 | all dead: shutters open, room item appears | - | - | $25 | sand | - |
| $1E | shutter | shutter | wall | bomb | 5 x Goriya ($06) | bomb ($00) at $D0,$60 | all dead: shutters open, room item appears | - | - | $0D | blocks | boss sound $20 |
| $1F | wall | open | bomb | wall | 1 x dungeon person ($4B) | - | - | - | - | $26 | blank | person $4B says: DODONGO DISLIKES SMOKE. |
| $2E | shutter | open | wall | key | 8 x Rope ($28) | - | all dead: shutters open | - | - | $1D | blocks | - |
| $2F | open | bomb | key | wall | 5 x Gel ($15) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $3E | open | open | wall | open | 1 x Moldorm ($41) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $25 | sand | - |
| $3F | bomb | bomb | open | wall | 1 x Trap ($4A), 3 x (empty) ($00), 4 x Keese blue ($1B) | bomb ($00) at $80,$90 | - | - | - | $00 | plain | - |
| $4E | open | shutter | wall | open | 5 x Rope ($28) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $02 | blocks | - |
| $4F | bomb | bomb | open | wall | 3 x Goriya ($05) | magic boomerang ($1E) at $80,$90 | all dead: shutters open, room item appears | - | - | $24 | statues | - |
| $5E | open | open | wall | key | 5 x Goriya ($06) | - | ringleader: first monster dead kills all | - | - | $1F | blocks | - |
| $5F | bomb | bomb | key | wall | 5 x Gel ($15) | map ($17) at $80,$90 | - | - | - | $03 | blocks | - |
| $6C | wall | wall | wall | open | 6 x Rope ($28) | key ($19) at $80,$90 | - | - | - | $00 | plain | - |
| $6D | wall | open | shutter | open | 5 x Rope ($28) | - | all dead: shutters open | - | - | $03 | blocks | - |
| $6E | open | open | open | key | 3 x Rope ($28) | - | - | - | - | $0D | blocks | - |
| $6F | bomb | wall | key | wall | 6 x Gel ($15) | compass ($16) at $D0,$60 | - | - | - | $1E | blocks | - |
| $7D start | open | open | wall | open | none | - | - | - | - | $21 | statues, sand | - |
| $7E | open | wall | open | wall | 5 x Rope ($28) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
<!-- /generated:rooms -->

# Floor plans

<!-- generated:floors -->
One character per floor square, 12 columns x 7 rows inside the walls (left column = west, top row =
north): `.` floor (tile `$74`), `B` block (tile `$B0`; a push block object is placed on one in a room
with a push block), `W` water (tile `$F4`), `S` statue (tiles `$94` and `$B4`), `>` stairs (tile `$70`),
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L57), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L67), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L228), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
Stairs that a secret makes appear are not part of the layout.

```
layout $00  rooms: $2F,$3F,$6C,$7E
............
............
............
............
............
............
............
```

```
layout $02  rooms: $4E
............
............
..B......B..
............
..B......B..
............
............
```

```
layout $03  rooms: $5F,$6D
............
..B......B..
............
............
............
..B......B..
............
```

```
layout $0D  rooms: $1E,$6E
............
.B.B.BB.B.B.
............
.B.B.BB.B.B.
............
.B.B.BB.B.B.
............
```

```
layout $1D  rooms: $2E
............
............
..BB....BB..
..BB....BB..
..BB....BB..
............
............
```

```
layout $1E  rooms: $6F
............
............
.....BB.....
.....BB.....
.....BB.....
............
............
```

```
layout $1F  rooms: $5E
............
.BB......BB.
............
.....BB.....
............
.BB......BB.
............
```

```
layout $21  rooms: $7D
............
.S..S..S..S.
............
.S..S..S..S.
.....::.....
.S.:S::S:.S.
...::::::...
```

```
layout $24  rooms: $4F
S..........S
............
............
............
............
............
S..........S
```

```
layout $25  rooms: $0E,$3E
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
```

```
layout $26  rooms: $1F
____________
____________
____________
____________
____________
____________
____________
```

```
layout $29  rooms: $0D
............
.BBBBBBBBBB.
.B..S..S..B.
.B.S....S.B.
.B........B.
.BBBB..BBBB.
............
```
<!-- /generated:floors -->

# Monsters and boss

# Route

# Optional detours

# Hazards and rules that matter here

# Second quest
