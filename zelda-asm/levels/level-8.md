# Level 8

(one-paragraph summary to be written)

Contents (generated; `L` = line in this file):

<!-- toc -->
- L21 At a glance
- L49 What Link must bring
- L51 What Link gets here
- L53 Map
- L88 Rooms
- L125 Floor plans
- L327 Monsters and boss
- L329 Route
- L331 Optional detours
- L333 Hazards and rules that matter here
- L335 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 8 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-8 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | $6C (tile object: $64 tree at $50,$60, secret works in Q2 only); $6D (tile object: $64 tree at $A0,$60, secret works in Q1 only) | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L175) |
| Start room | $7E | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Triforce room | $2C (room item: Triforce piece ($1B)); the field places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $3C (1 x Gleeok ($45)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | $2F, $0F, $6F | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 25: 24 on the submenu map, 1 the map does not show ($2D) | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (doorways; room and side) | 4 ($1E S, $3E S, $4D E, $5E E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (doorways; room and side) | 6 ($1D E, $2D E, $2E S, $3B E, $3C S, $5E S) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (doorways; room and side) | 9 ($0E S, $1E E, $2C S, $3D S, $3E E, $4B E, $4E S, $5D E, $7C E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (doorways; room and side) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $1E E shutter / $1F open, $2C S open / $3C shutter, $3D S open / $4D shutter, $4E S open / $5E shutter, $5D E open / $5E shutter, $7C E open / $7D shutter | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $7E S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 2 ($1F, $7C) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 4 ($4B, $5D, $5F, $7F) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 6 ($4B, $4C, $5C, $5D, $5E, $7F) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | bomb ($00): $0E, $1D, $3F; magic key ($0B): $0F (cellar); 5 rupees ($0F): $4E, $6E, $7D; book ($11): $6F (cellar); compass ($16): $5F; map ($17): $2E; heart container ($1A): $3C; Triforce piece ($1B): $2C | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md) (rows with `lvl` = `8` or `c8`).
<!-- /generated:facts -->

# What Link must bring

# What Link gets here

# Map

<!-- generated:map -->
```
                           [0Ei ]
                             S
                  [1Di ] X [1E  ]S/-[1F  ]
                    |        K
         [2CT ]   [2Dh ] X [2Ei ]
          |/S                X
[3B  ] X [3CBi]   [3D  ]   [3E  ] S [3Fi ]
  |        X       |/S       K
[4Bk ] S [4Ck ]   [4D  ] K [4Ei ]
                    |       |/S
         [5Ck ]---[5Dk ]-/S[5Ek ] K [5Fi ]
                             X
                           [6Ei ]
                             |
         [7C  ]-/S[7Di ]---[7ES ]---[7Fk ]

[73S] = room $73 and its marks: S start room, B boss room, T Triforce room,
        k a key lies here, i another room item lies here, h not shown on the submenu map.
Between two rooms, the door: --- or | open, K key door, S shutter, X bombable wall,
        F false wall (walk-through), nothing = wall. A/B = the two sides differ: the left
        or upper room's side first, . = wall on that side.
Doors on sides with no room of the level: $7E S open.
```

- Cellar `$2F` (tunnel cellar): joins `$4C` (left exit) and `$3F` (right exit), whose stairs lead here; item: none. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)

- Cellar `$0F` (treasure cellar): both exits lead to `$1F`, whose stairs lead here; item: magic key ($0B). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)

- Cellar `$6F` (treasure cellar): both exits lead to `$7C`, whose stairs lead here; item: book ($11). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)
<!-- /generated:map -->

# Rooms

Generated from the level data by `tools/romdata.py`; do not edit between the markers.

<!-- generated:rooms -->
| room | N | S | W | E | monsters | item | secret | block | dark | layout | floor | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $0E | wall | shutter | wall | wall | 5 x Darknut red ($0B) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $1D | wall | open | wall | bomb | 5 x Pols Voice ($16) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $11 | blocks | - |
| $1E | shutter | key | bomb | shutter | 1 x Gohma ($33) | - | all dead: shutters open | - | - | $24 | statues | - |
| $1F | wall | wall | open | wall | 2 x Pols Voice ($16), 2 x Darknut blue ($0C), 2 x Darknut red ($0B) | - | all dead: shutters open | B | - | $1A | blocks, stairs | stairs lead to cellar $0F |
| $2C Triforce | wall | open | wall | wall | none | Triforce piece ($1B) at $80,$90 | - | - | - | $29 | blocks, statues | - |
| $2D off-map | open | wall | wall | bomb | 1 x Rupee stash ($35) | - | - | - | - | $26 | blank | - |
| $2E | key | bomb | bomb | wall | 1 x Manhandla ($3C) | map ($17) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $3B | wall | open | wall | bomb | 1 x dungeon person ($50) | - | - | - | - | $26 | blank | person $50 says: 10TH ENEMY HAS THE BOMB.; boss sound $10 |
| $3C boss | shutter | bomb | bomb | wall | 1 x Gleeok ($45) | heart container ($1A) at $20,$C0 | all dead: shutters open, room item appears | - | - | $05 | blocks | - |
| $3D | wall | open | wall | wall | 1 x dungeon person ($4F) | - | - | - | - | $26 | blank | person $4F says: SPECTACLE ROCK IS / AN ENTRANCE TO DEATH.; boss sound $10 |
| $3E | bomb | key | wall | shutter | 6 x Darknut blue ($0C) | - | all dead: shutters open | - | - | $23 | statues | - |
| $3F | wall | wall | shutter | wall | 2 x Bubble ($2B), 1 x Darknut blue ($0C), 2 x Darknut red ($0B), 3 x Gibdo ($30) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $1B | blocks, stairs | stairs lead to cellar $2F |
| $4B | open | wall | wall | shutter | 6 x Darknut red ($0B) | key ($19) at $80,$90 | all dead: shutters open | - | D | $23 | statues | - |
| $4C | bomb | wall | shutter | wall | 8 x Pols Voice ($16) | key ($19) at $D0,$60 | all dead: shutters open | - | - | $1C | blocks, stairs | stairs lead to cellar $2F; boss sound $10 |
| $4D | shutter | open | wall | key | 1 x Gohma ($33) | - | all dead: shutters open | - | - | $25 | sand | - |
| $4E | key | open | key | wall | 2 x Bubble ($2B), 1 x Darknut blue ($0C), 2 x Darknut red ($0B), 3 x Gibdo ($30) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $5C | wall | wall | wall | open | 3 x Darknut red ($0B) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $24 | statues | - |
| $5D | open | wall | open | open | 2 x Pols Voice ($16), 2 x Gibdo ($30), 2 x Keese blue ($1B) | key ($19) at $80,$90 | - | - | D | $11 | blocks | - |
| $5E | shutter | bomb | shutter | key | 5 x Darknut blue ($0C) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $24 | statues | - |
| $5F | wall | wall | key | wall | 8 x Pols Voice ($16) | compass ($16) at $80,$90 | - | - | D | $18 | water | - |
| $6E | bomb | open | wall | wall | 1 x Manhandla ($3C) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $25 | sand | - |
| $7C | wall | wall | wall | open | 2 x Bubble ($2B), 1 x Darknut blue ($0C), 2 x Darknut red ($0B), 3 x Gibdo ($30) | - | - | B | - | $1A | blocks, stairs | stairs lead to cellar $6F |
| $7D | wall | wall | shutter | open | 1 x Manhandla ($3C) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $23 | statues | - |
| $7E start | open | open | open | open | none | - | - | - | - | $21 | statues, sand | - |
| $7F | wall | wall | open | wall | 3 x Bubble ($2B), 2 x Zol ($13), 3 x Keese blue ($1B) | key ($19) at $80,$90 | - | - | D | $16 | water | - |
| $2F cellar | - | - | - | - | 4 x Keese blue ($1B) | - | - | - | - | $3E | tunnel cellar | left exit to $4C, right exit to $3F; Link comes out at $70,$7D |
| $0F cellar | - | - | - | - | 4 x Keese blue ($1B) | magic key ($0B) at $80,$90 | - | - | - | $3F | treasure cellar | left exit to $1F, right exit to $1F; Link comes out at $60,$9D |
| $6F cellar | - | - | - | - | 4 x Keese blue ($1B) | book ($11) at $80,$90 | - | - | - | $3F | treasure cellar | left exit to $7C, right exit to $7C; Link comes out at $60,$9D |
<!-- /generated:rooms -->

# Floor plans

<!-- generated:floors -->
One character per floor square, 12 columns x 7 rows inside the walls (left column = west, top row =
north): `.` floor (tile `$74`), `B` block (tile `$B0`; a push block object is placed on one in a room
with a push block), `W` water (tile `$F4`), `S` statue (tiles `$94` and `$B4`), `>` stairs (tile `$70`),
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L57), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L67), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L228), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
Stairs that a secret makes appear are not part of the layout.

```
layout $00  rooms: $0E,$2E,$4E
............
............
............
............
............
............
............
```

```
layout $05  rooms: $3C
BBBBB..BBBBB
BBBB....BBBB
BB........BB
............
............
..B......B..
............
```

```
layout $11  rooms: $1D,$5D
..B........B
.B...B....B.
....B....B..
...B....B...
..B....B....
.B....B...B.
B........B..
```

```
layout $16  rooms: $7F
WWWWW..WWWWW
W...WWWW...W
W.WWW..WWW.W
..W......W..
W.WWW..WWW.W
W...WWWW...W
WWWWW..WWWWW
```

```
layout $18  rooms: $5F
............
............
WWWWWWWWWWWW
............
............
............
............
```

```
layout $1A  rooms: $1F,$7C
............
......B.....
.....B.B....
....B.>.B...
.....B.B....
......B.....
............
```

```
layout $1B  rooms: $3F
........BBBB
........BBBB
.........BBB
...........>
.........BBB
........BBBB
........BBBB
```

```
layout $1C  rooms: $4C
..........B.
.BBBBBBBB.B.
.B.....>B.B.
.B...BBBB.B.
.B........B.
.BBBBBBBBBB.
............
```

```
layout $21  rooms: $7E
............
.S..S..S..S.
............
.S..S..S..S.
.....::.....
.S.:S::S:.S.
...::::::...
```

```
layout $23  rooms: $3E,$4B,$7D
............
............
............
....S..S....
............
............
............
```

```
layout $24  rooms: $1E,$5C,$5E
S..........S
............
............
............
............
............
S..........S
```

```
layout $25  rooms: $4D,$6E
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
```

```
layout $26  rooms: $2D,$3B,$3D
____________
____________
____________
____________
____________
____________
____________
```

```
layout $29  rooms: $2C
............
.BBBBBBBBBB.
.B..S..S..B.
.B.S....S.B.
.B........B.
.BBBB..BBBB.
............
```

```
layout $3E (tunnel cellar)  rooms: $2F
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
layout $3F (treasure cellar)  rooms: $0F,$6F
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
