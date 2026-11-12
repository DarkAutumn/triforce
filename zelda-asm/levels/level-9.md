# Level 9

(one-paragraph summary to be written)

Contents (generated; `L` = line in this file):

<!-- toc -->
- L21 At a glance
- L49 What Link must bring
- L51 What Link gets here
- L53 Map
- L98 Rooms
- L172 Floor plans
- L506 Monsters and boss
- L508 Route
- L510 Optional detours
- L512 Hazards and rules that matter here
- L514 Second quest
<!-- /toc -->

# At a glance

<!-- generated:facts -->
| Field | Value | Read by |
| --- | --- | --- |
| Internal level number | 9 | [`CurLevel`](../src/Variables.inc#L12) |
| Number shown on screen | LEVEL-9 | [`LevelInfo_LevelNumber`](../src/Variables.inc#L725) |
| Overworld entrance room (Q1) | $00 (tile object: $63 rock wall at $A0,$80, secret works in Q2 only); $05 (tile object: $63 rock wall at $50,$A0, secret works in Q1 only) | [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L175) |
| Start room | $76 | [`LevelInfo_StartRoomId`](../src/Variables.inc#L722) |
| Compass marker room (the Triforce room field) | $32 (no Triforce piece there; monsters: 1 x Zelda ($37)); the field only places the compass marker | [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L723), [`StatusBar_UpdateTriforceMarker`](../src/bank7/next-room-room-flags.asm#L51) |
| Boss room | $42 (1 x Ganon ($3E)) | [`LevelInfo_BossRoomId`](../src/Variables.inc#L728), [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42) |
| Cellars | $60, $70, $72, $75, $67, $77, $00, $4F | [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55) |
| Rooms | 57: 54 on the submenu map, 3 the map does not show ($07, $32, $35) | [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729) |
| Monster counts (index = level block C bits 7-6) | 3, 5, 6, 8 | [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) |
| Key doors (doorways; room and side) | 16 ($01 E, $06 S, $12 S, $12 E, $13 S, $14 E, $22 E, $23 S, $25 E, $26 S, $30 S, $40 S, $53 S, $62 E, $63 S, $73 E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Bombable walls (doorways; room and side) | 19 ($03 E, $05 E, $07 S, $10 S, $15 S, $17 S, $21 E, $23 E, $25 S, $26 E, $27 S, $30 E, $33 E, $36 E, $43 S, $45 S, $54 E, $55 S, $55 E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Shutters (doorways; room and side) | 11 ($01 S, $02 S, $11 S, $21 S, $31 S, $32 S, $41 S, $42 S, $50 E, $56 S, $65 E) | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| False walls (doorways; room and side) | 0 | [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62) |
| Doorways whose two sides differ | $01 S shutter / $11 open, $02 S open / $12 shutter, $11 S shutter / $21 open, $21 S shutter / $31 open, $31 S shutter / $41 open, $41 S shutter / $51 open, $56 S open / $66 shutter, $65 E open / $66 shutter | [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55) |
| Doors on sides with no room of the level | $76 S open | [`Room_CalcNext`](../src/bank7/next-room-room-flags.asm#L72) |
| Push blocks | 11 ($03, $04, $05, $07, $10, $20, $30, $37, $55, $61, $74) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Dark rooms | 10 ($01, $13, $15, $24, $33, $41, $42, $50, $57, $65) | [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124) |
| Keys (room item) | 4 ($47, $56, $57, $61) | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |
| Other room items | bomb ($00): $11, $16, $23, $25, $37; silver arrow ($09): $4F (cellar); Triforce of Power ($0E): $42; 5 rupees ($0F): $12, $15, $26, $34, $40, $44, $62; red ring ($13): $00 (cellar); compass ($16): $35; map ($17): $27 | [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51) |

Room by room data: [`levelblock-uw-7to9-q1.md`](../data/levelblock-uw-7to9-q1.md) (rows with `lvl` = `9` or `c9`).
<!-- /generated:facts -->

# What Link must bring

# What Link gets here

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
`:` sand (tile `$68`), `_` blank square (tile `$24`). [`RoomLayout_UWSquareTiles`](../src/bank5/uw-floor-push-block.asm#L57), [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L67), [`PushBlock_FindAndCreate`](../src/bank5/uw-floor-push-block.asm#L228), [`Warp_CheckUnderLink`](../src/bank5/warps-stairs-caves.asm#L55)
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

# Route

# Optional detours

# Hazards and rules that matter here

# Second quest
