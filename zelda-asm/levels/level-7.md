# Level 7

(one-paragraph summary to be written)

Contents (generated; `L` = line in this file):

<!-- toc -->
- L21 At a glance
- L49 What Link must bring
- L51 What Link gets here
- L53 Map
- L86 Rooms
- L130 Floor plans
- L409 Monsters and boss
- L411 Route
- L413 Optional detours
- L415 Hazards and rules that matter here
- L417 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 7 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-7 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | $19 (tile object: $63 rock wall at $40,$70, secret works in Q2 only); $42 | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L175) |
| Start room | $79 | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Triforce room | $2B (room item: Triforce piece ($1B)); the field places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $2A (1 x Aquamentus ($3D)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | $7B, $4A | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 33: 31 on the submenu map, 2 the map does not show ($08, $1A) | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (doorways; room and side) | 5 ($0B S, $18 E, $1B E, $28 S, $48 S) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (doorways; room and side) | 10 ($08 S, $08 E, $0C E, $19 E, $1A E, $29 E, $39 E, $59 S, $5A S, $68 E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (doorways; room and side) | 9 ($09 S, $09 E, $0C S, $18 S, $2A E, $38 E, $39 S, $59 E, $5B S) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (doorways; room and side) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $0C S open / $1C shutter, $18 S shutter / $28 open, $2A E shutter / $2B open, $38 E shutter / $39 open, $39 S open / $49 shutter, $59 E shutter / $5A open, $5B S open / $6B shutter | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $79 S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 3 ($0D, $1A, $29) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 3 ($18, $5A, $6A) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 4 ($0A, $3A, $6D, $78) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | bomb ($00): $0B, $0C, $1B, $68, $69, $6C, $7A; red candle ($07): $4A (cellar); 5 rupees ($0F): $09, $38, $58; compass ($16): $5A; map ($17): $18; heart container ($1A): $2A; Triforce piece ($1B): $2B | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md) (rows with `lvl` = `7` or `c7`).
<!-- /generated:facts -->

# What Link must bring

# What Link gets here

# Map

<!-- generated:map -->
```
[08h ] X [09i ] S [0Ak ]   [0Bi ]   [0Ci ] X [0D  ]
  X        S                 K       |/S
[18i ] K [19  ] X [1Ah ] X [1Bi ] K [1C  ]
 S/|
[28  ]   [29  ] X [2ABi]S/-[2BT ]
  K
[38i ]S/-[39  ] X [3Ak ]
          |/S
[48  ]   [49  ]
  K        |
[58i ]---[59  ]S/-[5Ai ]   [5B  ]
  |        X        X       |/S
[68i ] X [69i ]---[6A  ]---[6B  ]---[6Ci ]---[6Dk ]
  |        |        |
[78k ]   [79S ]---[7Ai ]

[73S] = room $73 and its marks: S start room, B boss room, T Triforce room,
        k a key lies here, i another room item lies here, h not shown on the submenu map.
Between two rooms, the door: --- or | open, K key door, S shutter, X bombable wall,
        F false wall (walk-through), nothing = wall. A/B = the two sides differ: the left
        or upper room's side first, . = wall on that side.
Doors on sides with no room of the level: $79 S open.
```

- Cellar `$7B` (tunnel cellar): joins `$29` (left exit) and `$0D` (right exit), whose stairs lead here; item: none. [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)

- Cellar `$4A` (treasure cellar): both exits lead to `$1A`, whose stairs lead here; item: red candle ($07). [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55), [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md)
<!-- /generated:map -->

# Rooms

Generated from the level data by `tools/romdata.py`; do not edit between the markers.

<!-- generated:rooms -->
| room | N | S | W | E | monsters | item | secret | block | dark | layout | floor | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $08 off-map | wall | bomb | wall | bomb | 1 x Rupee stash ($35) | - | - | - | - | $26 | blank | - |
| $09 | wall | shutter | bomb | shutter | 3 x Goriya ($05), 3 x Goriya ($06) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $0A | blocks | - |
| $0A | wall | wall | shutter | wall | 5 x Goriya ($05) | key ($19) at $80,$90 | all dead: shutters open | - | - | $24 | statues | - |
| $0B | wall | key | wall | wall | 1 x Moldorm ($41) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $25 | sand | - |
| $0C | wall | open | wall | bomb | 3 x Dodongo ($31) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $23 | statues | - |
| $0D | wall | wall | bomb | wall | 3 x Bubble ($2B), 5 x Wallmaster ($27) | - | push block: stairs appear | B | - | $08 | blocks | stairs lead to cellar $7B |
| $18 | bomb | shutter | wall | key | 2 x Bubble ($2B), 3 x Goriya ($05), 3 x Keese blue ($1B) | map ($17) at $80,$90 | all dead: shutters open | - | D | $16 | water | - |
| $19 | shutter | wall | key | bomb | 6 x Goriya ($05) | - | all dead: shutters open | - | - | $1F | blocks | - |
| $1A off-map | wall | wall | bomb | bomb | 3 x Goriya ($05), 3 x Goriya ($06) | - | - | B | - | $1A | blocks, stairs | stairs lead to cellar $4A; boss sound $10 |
| $1B | key | wall | bomb | key | 6 x Goriya ($05) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $1C | shutter | wall | key | wall | 1 x Digdogger ($38) | - | all dead: shutters open | - | - | $24 | statues | - |
| $28 | open | key | wall | wall | 1 x Grumble ($36) | - | - | - | - | $26 | blank | person $36 says: GRUMBLE,GRUMBLE... |
| $29 | wall | wall | wall | bomb | 3 x Goriya ($05), 3 x Goriya ($06) | - | - | B | - | $1A | blocks, stairs | stairs lead to cellar $7B; boss sound $10 |
| $2A boss | wall | wall | bomb | shutter | 1 x Aquamentus ($3D) | heart container ($1A) at $80,$90 | all dead: shutters open, room item appears | - | - | $04 | blocks | - |
| $2B Triforce | wall | wall | open | wall | none | Triforce piece ($1B) at $80,$90 | - | - | - | $29 | blocks, statues | - |
| $38 | key | wall | wall | shutter | 3 x Goriya ($05), 3 x Goriya ($06) | 5 rupees ($0F) at $D0,$60 | all dead: shutters open, room item appears | - | - | $0D | blocks | - |
| $39 | wall | open | open | bomb | 1 x Digdogger ($38) | - | - | - | - | $24 | statues | - |
| $3A | wall | wall | bomb | wall | 1 x Moldorm ($41) | key ($19) at $80,$90 | all dead: shutters open, room item appears | - | - | $25 | sand | boss sound $10 |
| $48 | wall | key | wall | wall | 1 x dungeon person ($4F) | - | - | - | - | $26 | blank | person $4F says: I BET YOU'D LIKE / TO HAVE MORE BOMBS. |
| $49 | shutter | open | wall | wall | 2 x Bubble ($2B), 3 x Goriya ($05), 3 x Keese blue ($1B) | - | all dead: shutters open | - | - | $18 | water | - |
| $58 | key | open | wall | open | 3 x Dodongo ($31) | 5 rupees ($0F) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $59 | open | bomb | open | shutter | 3 x Goriya ($05), 3 x Goriya ($06) | - | all dead: shutters open | - | - | $02 | blocks | - |
| $5A | wall | bomb | open | wall | 8 x Stalfos ($2A) | compass ($16) at $80,$90 | - | - | D | $19 | water | - |
| $5B | wall | open | wall | wall | 1 x dungeon person ($50) | - | - | - | - | $26 | blank | person $50 says: THERE'S A SECRET IN / THE TIP OF THE NOSE. |
| $68 | open | open | wall | bomb | 1 x Trap ($4A), 3 x (empty) ($00), 4 x Keese blue ($1B) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $00 | plain | - |
| $69 | bomb | open | bomb | open | 5 x Goriya ($05) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $23 | statues | - |
| $6A | bomb | open | open | open | 8 x Keese blue ($1B) | - | ringleader: first monster dead kills all | - | D | $09 | blocks | - |
| $6B | shutter | wall | open | open | 6 x Goriya ($05) | - | all dead: shutters open | - | - | $1E | blocks | - |
| $6C | wall | wall | open | open | 1 x Digdogger ($39) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $24 | statues | - |
| $6D | wall | wall | open | wall | 8 x Stalfos ($2A) | key ($19) at $80,$90 | - | - | - | $13 | water | - |
| $78 | open | wall | wall | wall | 5 x Rope ($28) | key ($19) at $80,$90 | - | - | - | $23 | statues | - |
| $79 start | open | open | wall | open | none | - | - | - | - | $21 | statues, sand | - |
| $7A | open | wall | open | wall | 1 x Moldorm ($41) | bomb ($00) at $80,$90 | all dead: shutters open, room item appears | - | - | $1D | blocks | - |
| $7B cellar | - | - | - | - | 4 x Keese blue ($1B) | - | - | - | - | $3E | tunnel cellar | left exit to $29, right exit to $0D; Link comes out at $60,$9D |
| $4A cellar | - | - | - | - | 4 x Keese blue ($1B) | red candle ($07) at $80,$90 | - | - | - | $3F | treasure cellar | left exit to $1A, right exit to $1A; Link comes out at $60,$9D |
<!-- /generated:rooms -->

# Floor plans

<!-- generated:floors -->
One character per floor square, 12 columns x 7 rows inside the walls (left column = west, top row =
north): `.` floor (tile `$74`), `B` block (tile `$B0`; a push block object is placed on one in a room
with a push block), `W` water (tile `$F4`), `S` statue (tiles `$94` and `$B4`), `>` stairs (tile `$70`),
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L57), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L67), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L228), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
Stairs that a secret makes appear are not part of the layout.

```
layout $00  rooms: $1B,$58,$68
............
............
............
............
............
............
............
```

```
layout $02  rooms: $59
............
............
..B......B..
............
..B......B..
............
............
```

```
layout $04  rooms: $2A
.......BBBBB
.........BBB
..........BB
............
..........BB
.........BBB
.......BBBBB
```

```
layout $08  rooms: $0D
............
.BBBBBBBBBB.
..........B.
..........B.
..........B.
.BBBBBBBBBB.
............
```

```
layout $09  rooms: $6A
............
..BBBBBBBB..
..B......B..
..B......B..
..B......B..
..BBBBBBBB..
............
```

```
layout $0A  rooms: $09
............
............
............
....B..B....
............
............
............
```

```
layout $0D  rooms: $38
............
.B.B.BB.B.B.
............
.B.B.BB.B.B.
............
.B.B.BB.B.B.
............
```

```
layout $13  rooms: $6D
........W...
........W...
........W...
........W...
........W...
........W...
........W...
```

```
layout $16  rooms: $18
WWWWW..WWWWW
W...WWWW...W
W.WWW..WWW.W
..W......W..
W.WWW..WWW.W
W...WWWW...W
WWWWW..WWWWW
```

```
layout $18  rooms: $49
............
............
WWWWWWWWWWWW
............
............
............
............
```

```
layout $19  rooms: $5A
............
WWWWWWWWWWWW
............
............
............
WWWWWWWWWWWW
............
```

```
layout $1A  rooms: $1A,$29
............
......B.....
.....B.B....
....B.>.B...
.....B.B....
......B.....
............
```

```
layout $1D  rooms: $7A
............
............
..BB....BB..
..BB....BB..
..BB....BB..
............
............
```

```
layout $1E  rooms: $6B
............
............
.....BB.....
.....BB.....
.....BB.....
............
............
```

```
layout $1F  rooms: $19
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
layout $23  rooms: $0C,$69,$78
............
............
............
....S..S....
............
............
............
```

```
layout $24  rooms: $0A,$1C,$39,$6C
S..........S
............
............
............
............
............
S..........S
```

```
layout $25  rooms: $0B,$3A
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
::::::::::::
```

```
layout $26  rooms: $08,$28,$48,$5B
____________
____________
____________
____________
____________
____________
____________
```

```
layout $29  rooms: $2B
............
.BBBBBBBBBB.
.B..S..S..B.
.B.S....S.B.
.B........B.
.BBBB..BBBB.
............
```

```
layout $3E (tunnel cellar)  rooms: $7B
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
layout $3F (treasure cellar)  rooms: $4A
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
