# Underworld level block, levels 1-6, Q1

How to read this: one row per room `$00`-`$7F` of [`Level_BlockUW1to6Q1`](../src/bank6/level-blocks-common-data.asm#L70) (`dat/LevelBlockUW1Q1.dat`), the level block that levels 1, 2, 3, 4, 5, 6 share in
the first quest. Each row gives the six raw attribute bytes A-F, then the fields the game reads from
them. The formats are in `README.md`. Find a room with `grep '^\$32 '`.

Columns:

- `lvl`: the level whose submenu map shows the room; `cN` = a cellar listed by level N; `-` = neither
  (the room is drawn on no level's map). Levels are internal level numbers ([`CurLevel`](../src/Variables.inc#L12)). [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729), [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726)
- `N`, `S`, `W`, `E`: door type per side (A bits 7-5, A bits 4-2, B bits 7-5, B bits 4-2): `open`,
  `wall`, `false`/`false2` (walk-through wall), `bomb` (bombable wall), `key`/`key2` (key door),
  `shutter`. [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55), [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62)
- `po`, `pi`: outer and inner palette selectors (A bits 1-0, B bits 1-0). [`Room_FillPlayAreaAttrs`](../src/bank5/mode7-scroll-init.asm#L379)
- `cnt`: monster count selected by C bits 7-6 from [`LevelInfo_FoeCounts`](../src/Variables.inc#L716); `list`: monster list id (C bits 5-0, D bit 7).
  The `monsters` text applies the rules of `objlists.md`. [`Room_EnterCommon`](../src/bank5/enter-room.asm#L137)
- `blk`: `B` = the room has a push block (D bit 6). [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- `lay`: room layout id (D bits 5-0), see `room-layouts.md`; `$3E`/`$3F` = cellar. [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L67)
- `dk`: `D` = dark room (E bit 7). [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124)
- `snd`: boss sound request heard in the room (E bits 6-5 through [`Sample_BossRoarRequests`](../src/bank5/boss-sound-door-face.asm#L30)). [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42)
- `item`: room item (E bits 4-0; `$03` = none, shown as `-`). [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- `pos`: position index (F bits 5-4) and the item's X,Y from the owning level's [`LevelInfo_ShortcutOrItemPosArray`](../src/Variables.inc#L718) (`?` when the room
  has no `lvl`). [`Room_GetShortcutOrItemXYForRoomId`](../src/bank1/bomb-candle-wield.asm#L280)
- `secret`: secret trigger (F bits 2-0). [`Secret_RunTrigger_JumpTable`](../src/bank5/underworld-secrets.asm#L105)
- `monsters; cellar`: the monsters made on entry (before the kill count is subtracted). For a cellar:
  the rooms its left and right exits lead to (bytes A and B) and where Link appears there (byte C);
  its door columns are `-` and its monster list is not used. [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43), [`Room_EnterCommon`](../src/bank5/enter-room.asm#L137)

Levels of this block:

```
level  shown as  start  Triforce  boss  cellars  counts           item positions
1      LEVEL-1   $73    $36       $35   $7F      $03 $05 $06 $08  $C0,$90 $A0,$C0 $80,$90 $80,$70
2      LEVEL-2   $7D    $0D       $0E   -        $03 $05 $06 $08  $80,$90 $D0,$60 $20,$60 $20,$C0
3      LEVEL-3   $7C    $3D       $4D   -        $03 $05 $06 $08  $80,$90 $D0,$60 $20,$60 $20,$C0
4      LEVEL-4   $71    $03       $13   $60      $03 $05 $06 $08  $D0,$C0 $90,$90 $80,$80 $80,$90
5      LEVEL-5   $76    $14       $24   $07,$04  $03 $05 $06 $08  $80,$70 $C0,$80 $80,$A0 $80,$90
6      LEVEL-6   $79    $0C       $1C   $08,$75  $03 $05 $06 $08  $80,$90 $D0,$60 $20,$60 $20,$C0
```

# Rooms

```
room  A    B    C    D    E    F    lvl  N        S        W        E        po  pi  cnt  list  blk  lay  dk  snd  item                  pos        secret                                      monsters; cellar
$00   $22  $34  $0E  $A6  $03  $00  4    wall     open     wall     key      2   0   3    $4E   -    $26  -   -    -                     0:$D0,$C0  -                                           $4E dungeon person x1
$01   $32  $A3  $DB  $16  $99  $20  4    wall     bomb     key      open     2   3   8    $1B   -    $16  D   -    $19 key               2:$80,$80  -                                           $1B Keese blue x8
$02   $22  $07  $09  $81  $83  $00  4    wall     open     open     wall     2   3   3    $49   -    $01  D   -    -                     0:$D0,$C0  -                                           $49 Trap x1
$03   $22  $26  $00  $29  $1B  $30  4    wall     open     wall     wall     2   2   3    $00   -    $29  -   -    $1B Triforce piece    3:$80,$90  -                                           none
$04   $05  $05  $98  $3F  $05  $30  c5   -        -        -        -        -   -   -    -     -    $3F  -   -    $05 flute             3:$80,$90  -                                           treasure cellar: left exit to $05, right exit to $05, Link appears at $90,$8D
$05   $26  $36  $8C  $62  $03  $05  5    wall     wall     wall     key      2   2   6    $0C   B    $22  -   -    -                     0:$80,$70  push block: stairs appear                   $0C Darknut blue x6
$06   $36  $A6  $00  $5A  $03  $00  5    wall     key      key      wall     2   2   3    $00   B    $1A  -   -    -                     0:$80,$70  -                                           none
$07   $64  $06  $69  $3E  $03  $00  c5   -        -        -        -        -   -   -    -     -    $3E  -   -    -                     0:$80,$70  -                                           tunnel cellar: left exit to $64, right exit to $06, Link appears at $60,$9D
$08   $3A  $1D  $69  $3E  $03  $00  c6   -        -        -        -        -   -   -    -     -    $3E  -   -    -                     0:$80,$90  -                                           tunnel cellar: left exit to $3A, right exit to $1D, Link appears at $60,$9D
$09   $36  $32  $7B  $CA  $03  $05  6    wall     key      wall     bomb     2   2   5    $7B   B    $0A  -   -    -                     0:$80,$90  push block: stairs appear                   list $7B: $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$0A   $3E  $92  $B3  $8A  $03  $01  6    wall     shutter  bomb     bomb     2   2   6    $73   -    $0A  -   -    -                     0:$80,$90  all dead: shutters open                     list $73: $2B Bubble x2, $13 Zol x2, $17 Like-Like x2
$0B   $22  $84  $0C  $A6  $03  $00  6    wall     open     bomb     wall     2   0   3    $4C   -    $26  -   -    -                     0:$80,$90  -                                           $4C dungeon person x1
$0C   $22  $26  $00  $29  $1B  $00  6    wall     open     wall     wall     2   2   3    $00   -    $29  -   -    $1B Triforce piece    0:$80,$90  -                                           none
$0D   $26  $22  $00  $29  $1B  $00  2    wall     wall     wall     open     2   2   3    $00   -    $29  -   -    $1B Triforce piece    0:$80,$90  -                                           none
$0E   $22  $E6  $32  $25  $1A  $07  2    wall     open     shutter  wall     2   2   3    $32   -    $25  -   -    $1A heart container   0:$80,$90  all dead: shutters open, room item appears  $32 Dodongo x1
$0F   $69  $69  $68  $3F  $0C  $00  -    -        -        -        -        -   -   -    -     -    $3F  -   -    $0C raft              0:?        -                                           treasure cellar: left exit to $69, right exit to $69, Link appears at $60,$8D
$10   $FE  $32  $3C  $06  $03  $01  4    shutter  shutter  wall     bomb     2   2   3    $3C   -    $06  -   -    -                     0:$D0,$C0  all dead: shutters open                     $3C Manhandla x1
$11   $92  $92  $35  $26  $03  $00  -    bomb     bomb     bomb     bomb     2   2   3    $35   -    $26  -   -    -                     0:?        -                                           $35 Rupee stash x1
$12   $06  $9E  $52  $4A  $23  $04  4    open     wall     bomb     shutter  2   2   5    $12   B    $0A  -   $10  -                     0:$D0,$C0  push block: shutters open                   $12 Vire x5
$13   $E6  $E6  $03  $85  $1A  $07  4    shutter  wall     shutter  wall     2   2   3    $43   -    $05  -   -    $1A heart container   0:$D0,$C0  all dead: shutters open, room item appears  $43 Gleeok x1
$14   $22  $26  $00  $29  $1B  $30  5    wall     open     wall     wall     2   2   3    $00   -    $29  -   -    $1B Triforce piece    3:$80,$90  -                                           none
$15   $32  $24  $0D  $A6  $03  $00  -    wall     bomb     wall     wall     2   0   3    $4D   -    $26  -   -    -                     0:?        -                                           $4D dungeon person x1
$16   $A6  $33  $DB  $18  $19  $20  5    key      wall     wall     bomb     2   3   8    $1B   -    $18  -   -    $19 key               2:$80,$A0  -                                           $1B Keese blue x8
$17   $26  $84  $0F  $A6  $03  $00  5    wall     wall     bomb     wall     2   0   3    $4F   -    $26  -   -    -                     0:$80,$70  -                                           $4F dungeon person x1
$18   $3E  $3E  $04  $85  $03  $01  6    wall     shutter  wall     shutter  2   2   3    $44   -    $05  -   -    -                     0:$80,$90  all dead: shutters open                     $44 Gleeok x1
$19   $B6  $03  $B3  $93  $17  $07  6    key      key      open     open     2   3   6    $73   -    $13  -   -    $17 map               0:$80,$90  all dead: shutters open, room item appears  list $73: $2B Bubble x2, $13 Zol x2, $17 Like-Like x2
$1A   $06  $03  $A4  $18  $99  $00  6    open     wall     open     open     2   3   6    $24   -    $18  D   -    $19 key               0:$80,$90  -                                           $24 Red Wizzrobe x6
$1B   $E6  $06  $FC  $83  $43  $01  6    shutter  wall     open     wall     2   2   8    $7C   -    $03  -   $20  -                     0:$80,$90  all dead: shutters open                     list $7C: $2B Bubble x1, $17 Like-Like x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$1C   $F6  $26  $34  $06  $1A  $07  6    shutter  key      wall     wall     2   2   3    $34   -    $06  -   -    $1A heart container   0:$80,$90  all dead: shutters open, room item appears  $34 Gohma x1
$1D   $22  $26  $B3  $9B  $43  $00  6    wall     open     wall     wall     2   2   6    $73   -    $1B  -   $20  -                     0:$80,$90  -                                           list $73: $2B Bubble x2, $13 Zol x2, $17 Like-Like x2
$1E   $FE  $32  $46  $0D  $40  $17  2    shutter  shutter  wall     bomb     2   2   5    $06   -    $0D  -   $20  $00 bomb              1:$D0,$60  all dead: shutters open, room item appears  $06 Goriya x5
$1F   $22  $84  $0B  $A6  $03  $00  2    wall     open     bomb     wall     2   0   3    $4B   -    $26  -   -    -                     0:$80,$90  -                                           $4B dungeon person x1
$20   $16  $23  $52  $15  $03  $00  4    open     key      wall     open     2   3   5    $12   -    $15  -   -    -                     0:$D0,$C0  -                                           $12 Vire x5
$21   $96  $07  $55  $12  $97  $00  4    bomb     key      open     wall     2   3   5    $15   -    $12  D   -    $17 map               0:$D0,$C0  -                                           $15 Gel x5
$22   $26  $36  $0A  $DA  $03  $20  1    wall     wall     wall     key      2   2   3    $4A   B    $1A  -   -    -                     2:$80,$90  -                                           $4A Trap x1
$23   $36  $A7  $06  $15  $19  $37  1    wall     key      key      wall     2   3   3    $06   -    $15  -   -    $19 key               3:$80,$70  all dead: shutters open, room item appears  $06 Goriya x3
$24   $E2  $36  $39  $24  $1A  $07  5    shutter  open     wall     key      2   2   3    $39   -    $24  -   -    $1A heart container   0:$80,$70  all dead: shutters open, room item appears  $39 Digdogger x1
$25   $86  $A2  $56  $07  $63  $00  5    bomb     wall     key      open     2   2   5    $16   -    $07  -   $40  -                     0:$80,$70  -                                           $16 Pols Voice x5
$26   $26  $17  $70  $14  $19  $27  5    wall     wall     open     key      2   3   5    $30   -    $14  -   -    $19 key               2:$80,$A0  all dead: shutters open, room item appears  $30 Gibdo x5
$27   $22  $A7  $B0  $95  $99  $20  5    wall     open     key      wall     2   3   6    $70   -    $15  D   -    $19 key               2:$80,$A0  -                                           list $70: $16 Pols Voice x2, $30 Gibdo x2, $1B Keese blue x2
$28   $1E  $32  $7B  $8D  $0F  $17  6    open     shutter  wall     bomb     2   2   5    $7B   -    $0D  -   -    $0F 5 rupees          1:$D0,$60  all dead: shutters open, room item appears  list $7B: $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$29   $A2  $87  $7B  $94  $99  $00  6    key      open     bomb     wall     2   3   5    $7B   -    $14  D   -    $19 key               0:$80,$90  -                                           list $7B: $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$2A   $26  $3E  $DB  $1D  $19  $01  3    wall     wall     wall     shutter  2   2   8    $1B   -    $1D  -   -    $19 key               0:$80,$90  all dead: shutters open                     $1B Keese blue x8
$2B   $22  $04  $0B  $A6  $03  $00  3    wall     open     open     wall     2   0   3    $4B   -    $26  -   -    -                     0:$80,$90  -                                           $4B dungeon person x1
$2C   $A2  $22  $F7  $80  $43  $00  6    key      open     wall     open     2   2   8    $77   -    $00  -   $20  -                     0:$80,$90  -                                           list $77: $4A Trap x1, $00 (empty) x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$2D   $06  $07  $72  $93  $99  $07  6    open     wall     open     wall     2   3   5    $72   -    $13  D   -    $19 key               0:$80,$90  all dead: shutters open, room item appears  list $72: $2B Bubble x2, $12 Vire x3
$2E   $E2  $36  $E8  $1D  $03  $01  2    shutter  open     wall     key      2   2   8    $28   -    $1D  -   -    -                     0:$80,$90  all dead: shutters open                     $28 Rope x8
$2F   $12  $A6  $55  $00  $0F  $07  2    open     bomb     key      wall     2   2   5    $15   -    $00  -   -    $0F 5 rupees          0:$80,$90  all dead: shutters open, room item appears  $15 Gel x5
$30   $A2  $37  $72  $98  $83  $10  4    key      open     wall     key      2   3   5    $72   -    $18  D   -    -                     1:$90,$90  -                                           list $72: $2B Bubble x2, $12 Vire x3
$31   $A6  $BF  $52  $17  $03  $01  4    key      wall     key      shutter  2   3   5    $12   -    $17  -   -    -                     0:$D0,$C0  all dead: shutters open                     $12 Vire x5
$32   $26  $06  $B3  $CA  $03  $05  4    wall     wall     open     wall     2   2   6    $73   B    $0A  -   -    -                     0:$D0,$C0  push block: stairs appear                   list $73: $2B Bubble x2, $13 Zol x2, $17 Like-Like x2
$33   $A2  $27  $2A  $17  $19  $20  1    key      open     wall     wall     2   3   3    $2A   -    $17  -   -    $19 key               2:$80,$90  -                                           $2A Stalfos x3
$34   $E6  $26  $B0  $1E  $63  $01  5    shutter  wall     wall     wall     2   2   6    $30   -    $1E  -   $40  -                     0:$80,$70  all dead: shutters open                     $30 Gibdo x6
$35   $36  $3E  $3D  $04  $1A  $07  1    wall     key      wall     shutter  2   2   3    $3D   -    $04  -   -    $1A heart container   0:$C0,$90  all dead: shutters open, room item appears  $3D Aquamentus x1
$36   $26  $06  $00  $29  $1B  $20  1    wall     wall     open     wall     2   2   3    $00   -    $29  -   -    $1B Triforce piece    2:$80,$90  -                                           none
$37   $02  $27  $0B  $17  $96  $27  5    open     open     wall     wall     2   3   3    $0B   -    $17  D   -    $16 compass           2:$80,$A0  all dead: shutters open, room item appears  $0B Darknut red x3
$38   $FE  $26  $FC  $CA  $03  $04  6    shutter  shutter  wall     wall     2   2   8    $7C   B    $0A  -   -    -                     0:$80,$90  push block: shutters open                   list $7C: $2B Bubble x1, $17 Like-Like x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$39   $E6  $3F  $52  $13  $83  $01  6    shutter  wall     wall     shutter  2   3   5    $12   -    $13  D   -    -                     0:$80,$90  all dead: shutters open                     $12 Vire x5
$3A   $26  $06  $FC  $E2  $03  $05  6    wall     wall     open     wall     2   2   8    $7C   B    $22  -   -    -                     0:$80,$90  push block: stairs appear                   list $7C: $2B Bubble x1, $17 Like-Like x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$3B   $F6  $26  $AD  $CA  $03  $04  3    shutter  key      wall     wall     2   2   6    $6D   B    $0A  -   -    -                     0:$80,$90  push block: shutters open                   list $6D: $4A Trap x1, $00 (empty) x3, $13 Zol x2
$3C   $E6  $27  $7B  $97  $80  $07  6    shutter  wall     wall     wall     2   3   5    $7B   -    $17  D   -    $00 bomb              0:$80,$90  all dead: shutters open, room item appears  list $7B: $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$3D   $22  $26  $00  $29  $1B  $00  3    wall     open     wall     wall     2   2   3    $00   -    $29  -   -    $1B Triforce piece    0:$80,$90  -                                           none
$3E   $02  $21  $01  $A5  $19  $07  2    open     open     wall     open     2   1   3    $41   -    $25  -   -    $19 key               0:$80,$90  all dead: shutters open, room item appears  $41 Moldorm x1
$3F   $92  $06  $EE  $80  $00  $00  2    bomb     bomb     open     wall     2   2   8    $6E   -    $00  -   -    $00 bomb              0:$80,$90  -                                           list $6E: $4A Trap x1, $00 (empty) x3, $1B Keese blue x4
$40   $02  $27  $53  $15  $99  $20  4    open     open     wall     wall     2   3   5    $13   -    $15  D   -    $19 key               2:$80,$80  -                                           $13 Zol x5
$41   $26  $20  $0C  $A6  $03  $00  1    wall     wall     wall     open     2   0   3    $4C   -    $26  -   -    -                     0:$C0,$90  -                                           $4C dungeon person x1
$42   $36  $E2  $15  $62  $03  $04  1    wall     key      shutter  open     2   2   3    $15   B    $22  -   -    -                     0:$C0,$90  push block: shutters open                   $15 Gel x3
$43   $12  $16  $55  $1F  $17  $00  1    open     bomb     open     key      2   2   5    $15   -    $1F  -   -    $17 map               0:$C0,$90  -                                           $15 Gel x5
$44   $32  $A2  $06  $02  $1D  $37  1    wall     bomb     key      open     2   2   3    $06   -    $02  -   -    $1D wooden boomerang  3:$80,$70  all dead: shutters open, room item appears  $06 Goriya x3
$45   $A6  $06  $E7  $0D  $39  $10  1    key      wall     open     wall     2   2   8    $27   -    $0D  -   $10  $19 key               1:$A0,$C0  -                                           $27 Wallmaster x8
$46   $36  $23  $00  $12  $97  $20  5    wall     key      wall     open     2   3   3    $00   -    $12  D   -    $17 map               2:$80,$A0  -                                           none
$47   $02  $07  $70  $15  $19  $27  5    open     open     open     wall     2   3   5    $30   -    $15  -   -    $19 key               2:$80,$A0  all dead: shutters open, room item appears  $30 Gibdo x5
$48   $02  $26  $09  $81  $03  $00  6    open     open     wall     wall     2   2   3    $49   -    $01  -   -    -                     0:$80,$90  -                                           $49 Trap x1
$49   $22  $22  $EF  $80  $19  $00  3    wall     open     wall     open     2   2   8    $6F   -    $00  -   -    $19 key               0:$80,$90  -                                           list $6F: $2B Bubble x3, $13 Zol x2, $1B Keese blue x3
$4A   $22  $16  $5B  $0F  $00  $07  3    wall     open     open     key      2   2   5    $1B   -    $0F  -   -    $00 bomb              0:$80,$90  all dead: shutters open, room item appears  $1B Keese blue x5
$4B   $A2  $B6  $13  $08  $19  $07  3    key      open     key      key      2   2   3    $13   -    $08  -   -    $19 key               0:$80,$90  all dead: shutters open, room item appears  $13 Zol x3
$4C   $22  $B2  $AD  $80  $77  $00  3    wall     open     key      bomb     2   2   6    $6D   -    $00  -   $40  $17 map               0:$80,$90  -                                           list $6D: $4A Trap x1, $00 (empty) x3, $13 Zol x2
$4D   $FE  $85  $3C  $25  $1A  $07  3    shutter  shutter  bomb     wall     2   1   3    $3C   -    $25  -   -    $1A heart container   0:$80,$90  all dead: shutters open, room item appears  $3C Manhandla x1
$4E   $1E  $22  $68  $02  $19  $07  2    open     shutter  wall     open     2   2   5    $28   -    $02  -   -    $19 key               0:$80,$90  all dead: shutters open, room item appears  $28 Rope x5
$4F   $92  $06  $05  $24  $1E  $07  2    bomb     bomb     open     wall     2   2   3    $05   -    $24  -   -    $1E magic boomerang   0:$80,$90  all dead: shutters open, room item appears  $05 Goriya x3
$50   $06  $23  $52  $11  $83  $00  4    open     wall     wall     open     2   3   5    $12   -    $11  D   -    -                     0:$D0,$C0  -                                           $12 Vire x5
$51   $22  $06  $DB  $03  $19  $10  4    wall     open     open     wall     2   2   8    $1B   -    $03  -   -    $19 key               1:$90,$90  -                                           $1B Keese blue x8
$52   $A6  $3E  $9B  $1E  $03  $01  1    key      wall     wall     shutter  2   2   6    $1B   -    $1E  -   -    -                     0:$C0,$90  all dead: shutters open                     $1B Keese blue x6
$53   $82  $02  $6A  $02  $19  $37  1    bomb     open     open     open     2   2   5    $2A   -    $02  -   -    $19 key               3:$80,$70  all dead: shutters open, room item appears  $2A Stalfos x5
$54   $86  $06  $DB  $03  $16  $00  1    bomb     wall     open     wall     2   2   8    $1B   -    $03  -   -    $16 compass           0:$C0,$90  -                                           $1B Keese blue x8
$55   $22  $3F  $53  $13  $19  $27  5    wall     open     wall     shutter  2   3   5    $13   -    $13  -   -    $19 key               2:$80,$A0  all dead: shutters open, room item appears  $13 Zol x5
$56   $A2  $E2  $31  $00  $00  $07  5    key      open     shutter  open     2   2   3    $31   -    $00  -   -    $00 bomb              0:$80,$70  all dead: shutters open, room item appears  $31 Dodongo x3
$57   $06  $07  $53  $17  $8F  $27  5    open     wall     open     wall     2   3   5    $13   -    $17  D   -    $0F 5 rupees          2:$80,$A0  all dead: shutters open, room item appears  $13 Zol x5
$58   $E2  $27  $DB  $24  $19  $07  6    shutter  open     wall     wall     2   3   8    $1B   -    $24  -   -    $19 key               0:$80,$90  all dead: shutters open, room item appears  $1B Keese blue x8
$59   $1E  $36  $4B  $1D  $03  $01  3    open     shutter  wall     key      2   2   5    $0B   -    $1D  -   -    -                     0:$80,$90  all dead: shutters open                     $0B Darknut red x5
$5A   $06  $A2  $EE  $80  $16  $00  3    open     wall     key      open     2   2   8    $6E   -    $00  -   -    $16 compass           0:$80,$90  -                                           list $6E: $4A Trap x1, $00 (empty) x3, $1B Keese blue x4
$5B   $02  $12  $0B  $00  $00  $07  3    open     open     open     bomb     2   2   3    $0B   -    $00  -   -    $00 bomb              0:$80,$90  all dead: shutters open, room item appears  $0B Darknut red x3
$5C   $06  $9E  $0B  $1F  $03  $11  3    open     wall     bomb     shutter  2   2   3    $0B   -    $1F  -   -    -                     1:$D0,$60  all dead: shutters open                     $0B Darknut red x3
$5D   $E6  $E6  $EF  $9E  $6F  $27  3    shutter  wall     shutter  wall     2   2   8    $6F   -    $1E  -   $40  $0F 5 rupees          2:$20,$60  all dead: shutters open, room item appears  list $6F: $2B Bubble x3, $13 Zol x2, $1B Keese blue x3
$5E   $02  $36  $46  $1F  $03  $02  2    open     open     wall     key      2   2   5    $06   -    $1F  -   -    -                     0:$80,$90  ringleader: first monster dead kills all    $06 Goriya x5
$5F   $92  $A6  $55  $03  $17  $00  2    bomb     bomb     key      wall     2   2   5    $15   -    $03  -   -    $17 map               0:$80,$90  -                                           $15 Gel x5
$60   $32  $32  $CB  $3F  $0D  $30  c4   -        -        -        -        -   -   -    -     -    $3F  -   -    $0D ladder            3:$80,$90  -                                           treasure cellar: left exit to $32, right exit to $32, Link appears at $C0,$BD
$61   $02  $36  $12  $02  $03  $00  4    open     open     wall     key      2   2   3    $12   -    $02  -   -    -                     0:$D0,$C0  -                                           $12 Vire x3
$62   $26  $A7  $52  $0C  $96  $20  4    wall     wall     key      wall     2   3   5    $12   -    $0C  D   -    $16 compass           2:$80,$80  -                                           $12 Vire x5
$63   $16  $26  $2A  $1E  $03  $00  1    open     key      wall     wall     2   2   3    $2A   -    $1E  -   -    -                     0:$C0,$90  -                                           $2A Stalfos x3
$64   $26  $32  $4C  $5A  $03  $00  5    wall     wall     wall     bomb     2   2   5    $0C   B    $1A  -   -    -                     0:$80,$70  -                                           $0C Darknut blue x5
$65   $E6  $92  $70  $02  $00  $10  5    shutter  wall     bomb     bomb     2   2   5    $30   -    $02  -   -    $00 bomb              1:$C0,$80  -                                           $30 Gibdo x5
$66   $E2  $97  $30  $18  $99  $11  5    shutter  open     bomb     key      2   3   3    $30   -    $18  D   -    $19 key               1:$C0,$80  all dead: shutters open                     $30 Gibdo x3
$67   $32  $A4  $0E  $A6  $03  $00  5    wall     bomb     key      wall     2   0   3    $4E   -    $26  -   -    -                     0:$80,$70  -                                           $4E dungeon person x1
$68   $02  $26  $53  $1D  $16  $07  6    open     open     wall     wall     2   2   5    $13   -    $1D  -   -    $16 compass           0:$80,$90  all dead: shutters open, room item appears  $13 Zol x5
$69   $06  $26  $CB  $1B  $00  $07  3    open     wall     wall     wall     2   2   8    $0B   -    $1B  -   -    $00 bomb              0:$80,$90  all dead: shutters open, room item appears  $0B Darknut red x8
$6A   $36  $24  $0D  $A6  $03  $00  6    wall     key      wall     wall     2   0   3    $4D   -    $26  -   -    -                     0:$80,$90  -                                           $4D dungeon person x1
$6B   $02  $26  $53  $11  $19  $07  3    open     open     wall     wall     2   2   5    $13   -    $11  -   -    $19 key               0:$80,$90  all dead: shutters open, room item appears  $13 Zol x5
$6C   $26  $22  $A8  $00  $19  $00  2    wall     wall     wall     open     2   2   6    $28   -    $00  -   -    $19 key               0:$80,$90  -                                           $28 Rope x6
$6D   $22  $E2  $68  $03  $03  $01  2    wall     open     shutter  open     2   2   5    $28   -    $03  -   -    -                     0:$80,$90  all dead: shutters open                     $28 Rope x5
$6E   $02  $16  $28  $0D  $03  $00  2    open     open     open     key      2   2   3    $28   -    $0D  -   -    -                     0:$80,$90  -                                           $28 Rope x3
$6F   $86  $A6  $95  $1E  $16  $10  2    bomb     wall     key      wall     2   2   6    $15   -    $1E  -   -    $16 compass           1:$D0,$60  -                                           $15 Gel x6
$70   $26  $22  $DB  $00  $19  $17  4    wall     wall     wall     open     2   2   8    $1B   -    $00  -   -    $19 key               1:$90,$90  all dead: shutters open, room item appears  $1B Keese blue x8
$71   $02  $07  $00  $21  $03  $00  4    open     open     open     wall     2   3   3    $00   -    $21  -   -    -                     0:$D0,$C0  -                                           none
$72   $26  $22  $1B  $00  $19  $17  1    wall     wall     wall     open     2   2   3    $1B   -    $00  -   -    $19 key               1:$A0,$C0  all dead: shutters open, room item appears  $1B Keese blue x3
$73   $A2  $03  $00  $21  $03  $00  1    key      open     open     open     2   3   3    $00   -    $21  -   -    -                     0:$C0,$90  -                                           none
$74   $26  $06  $6A  $1D  $19  $00  1    wall     wall     open     wall     2   2   5    $2A   -    $1D  -   -    $19 key               0:$C0,$90  -                                           $2A Stalfos x5
$75   $09  $09  $C8  $3F  $10  $00  c6   -        -        -        -        -   -   -    -     -    $3F  -   -    $10 rod               0:$80,$90  -                                           treasure cellar: left exit to $09, right exit to $09, Link appears at $C0,$8D
$76   $02  $23  $00  $21  $03  $00  5    open     open     wall     open     2   3   3    $00   -    $21  -   -    -                     0:$80,$70  -                                           none
$77   $86  $06  $56  $1D  $19  $27  5    bomb     wall     open     wall     2   2   5    $16   -    $1D  -   -    $19 key               2:$80,$A0  all dead: shutters open, room item appears  $16 Pols Voice x5
$78   $06  $36  $64  $1F  $03  $02  6    open     wall     wall     key      2   2   5    $24   -    $1F  -   -    -                     0:$80,$90  ringleader: first monster dead kills all    $24 Red Wizzrobe x5
$79   $22  $A3  $00  $21  $03  $00  6    wall     open     key      open     2   3   3    $00   -    $21  -   -    -                     0:$80,$90  -                                           none
$7A   $A6  $07  $64  $03  $99  $07  6    key      wall     open     wall     2   3   5    $24   -    $03  D   -    $19 key               0:$80,$90  all dead: shutters open, room item appears  $24 Red Wizzrobe x5
$7B   $06  $22  $93  $03  $19  $00  3    open     wall     wall     open     2   2   6    $13   -    $03  -   -    $19 key               0:$80,$90  -                                           $13 Zol x6
$7C   $22  $07  $00  $21  $03  $00  3    wall     open     open     wall     2   3   3    $00   -    $21  -   -    -                     0:$80,$90  -                                           none
$7D   $02  $23  $00  $21  $03  $00  2    open     open     wall     open     2   3   3    $00   -    $21  -   -    -                     0:$80,$90  -                                           none
$7E   $06  $06  $68  $00  $19  $07  2    open     wall     open     wall     2   2   5    $28   -    $00  -   -    $19 key               0:$80,$90  all dead: shutters open, room item appears  $28 Rope x5
$7F   $22  $22  $6A  $3F  $0A  $20  c1   -        -        -        -        -   -   -    -     -    $3F  -   -    $0A bow               2:$80,$90  -                                           treasure cellar: left exit to $22, right exit to $22, Link appears at $60,$AD
```
