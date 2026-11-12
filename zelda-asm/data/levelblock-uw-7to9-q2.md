# Underworld level block, levels 7-9, Q2

How to read this: one row per room `$00`-`$7F` of [`Level_BlockUW7to9Q2`](../src/bank6/level-blocks-common-data.asm#L82) (`dat/LevelBlockUW2Q2.dat`), the level block that levels 7, 8, 9 share in
the second quest. Each row gives the six raw attribute bytes A-F, then the fields the game reads from
them. The formats are in `README.md`. Find a room with `grep '^\$32 '`.

Columns:

- `lvl`: the level whose submenu map shows the room; `cN` = a cellar listed by level N; `-` = neither
  (the room is drawn on no level's map). Levels are internal level numbers ([`CurLevel`](../src/Variables.inc#L12)). [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L729), [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L726)
- `N`, `S`, `W`, `E`: door type per side (A bits 7-5, A bits 4-2, B bits 7-5, B bits 4-2): `open`,
  `wall`, `false`/`false2` (walk-through wall), `bomb` (bombable wall), `key`/`key2` (key door),
  `shutter`. [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55), [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62)
- `po`, `pi`: outer and inner palette selectors (A bits 1-0, B bits 1-0). [`Room_FillPlayAreaAttrs`](../src/bank5/mode7-scroll-init.asm#L393)
- `cnt`: monster count selected by C bits 7-6 from [`LevelInfo_FoeCounts`](../src/Variables.inc#L716); `list`: monster list id (C bits 5-0, D bit 7).
  The `monsters` text applies the rules of `objlists.md`. [`Room_EnterCommon`](../src/bank5/enter-room.asm#L143)
- `blk`: `B` = the room has a push block (D bit 6). [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- `lay`: room layout id (D bits 5-0), see `room-layouts.md`; `$3E`/`$3F` = cellar. [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L73)
- `dk`: `D` = dark room (E bit 7). [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124)
- `snd`: boss sound request heard in the room (E bits 6-5 through [`Sample_BossRoarRequests`](../src/bank5/boss-sound-door-face.asm#L30)). [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42)
- `item`: room item (E bits 4-0; `$03` = none, shown as `-`). [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- `pos`: position index (F bits 5-4) and the item's X,Y from the owning level's [`LevelInfo_ShortcutOrItemPosArray`](../src/Variables.inc#L718) (`?` when the room
  has no `lvl`). [`Room_GetShortcutOrItemXYForRoomId`](../src/bank1/bomb-candle-wield.asm#L280)
- `secret`: secret trigger (F bits 2-0). [`Secret_RunTrigger_JumpTable`](../src/bank5/underworld-secrets.asm#L105)
- `monsters; cellar`: the monsters made on entry (before the kill count is subtracted). For a cellar:
  the rooms its left and right exits lead to (bytes A and B) and where Link appears there (byte C);
  its door columns are `-` and its monster list is not used. [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43), [`Room_EnterCommon`](../src/bank5/enter-room.asm#L143)

Levels of this block:

```
level  shown as  start  Triforce  boss  cellars                              counts           item positions
7      LEVEL-8   $7F    $2D       $3D   $02,$03,$04,$05,$20,$21,$26,$2B,$2C  $03 $05 $06 $08  $C0,$90 $A0,$C0 $80,$90 $70,$90
8      LEVEL-7   $79    $1B       $1C   $27,$30,$37,$60,$67,$70              $03 $05 $06 $08  $C0,$90 $A0,$C0 $80,$90 $50,$70
9      LEVEL-9   $74    $07       $17   $71,$72,$75,$76,$77                  $03 $05 $06 $08  $C0,$90 $A0,$C0 $80,$90 $B0,$60
```

# Rooms

```
room  A    B    C    D    E    F    lvl  N        S        W        E        po  pi  cnt  list  blk  lay  dk  snd  item                   pos        secret                                      monsters; cellar
$00   $26  $2A  $3B  $4A  $03  $05  9    wall     wall     wall     false    2   2   3    $3B   B    $0A  -   -    -                      0:$C0,$90  push block: stairs appear                   $3B Lanmola x1
$01   $22  $46  $07  $CA  $03  $05  9    wall     open     false    wall     2   2   3    $47   B    $0A  -   -    -                      0:$C0,$90  push block: stairs appear                   $47 Patra x1
$02   $18  $6F  $69  $3E  $03  $00  c7   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$C0,$90  -                                           tunnel cellar: left exit to $18, right exit to $6F, Link appears at $60,$9D
$03   $29  $5F  $69  $3E  $03  $00  c7   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$C0,$90  -                                           tunnel cellar: left exit to $29, right exit to $5F, Link appears at $60,$9D
$04   $19  $4F  $69  $3E  $03  $00  c7   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$C0,$90  -                                           tunnel cellar: left exit to $19, right exit to $4F, Link appears at $60,$9D
$05   $49  $0A  $69  $3E  $03  $00  c7   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$C0,$90  -                                           tunnel cellar: left exit to $49, right exit to $0A, Link appears at $60,$9D
$06   $26  $26  $EC  $62  $03  $05  9    wall     wall     wall     wall     2   2   8    $2C   B    $22  -   -    -                      0:$C0,$90  push block: stairs appear                   $2C Bubble x8
$07   $3E  $26  $37  $27  $03  $20  9    wall     shutter  wall     wall     2   2   3    $37   -    $27  -   -    -                      2:$80,$90  -                                           $37 Zelda x1
$08   $22  $32  $01  $DA  $03  $00  7    wall     open     wall     bomb     2   2   3    $41   B    $1A  -   -    -                      0:$C0,$90  -                                           $41 Moldorm x1
$09   $26  $87  $31  $24  $19  $27  7    wall     wall     bomb     wall     2   3   3    $31   -    $24  -   -    $19 key                2:$80,$90  all dead: shutters open, room item appears  $31 Dodongo x3
$0A   $26  $22  $AD  $5F  $19  $15  7    wall     wall     wall     open     2   2   6    $2D   B    $1F  -   -    $19 key                1:$A0,$C0  push block: stairs appear                   $2D Bubble x6
$0B   $26  $F7  $31  $24  $03  $01  7    wall     wall     shutter  key      2   3   3    $31   -    $24  -   -    -                      0:$C0,$90  all dead: shutters open                     $31 Dodongo x3
$0C   $26  $B3  $FF  $95  $80  $17  7    wall     wall     key      bomb     2   3   8    $7F   -    $15  D   -    $00 bomb               1:$A0,$C0  all dead: shutters open, room item appears  list $7F: $2D Bubble x3, $2C Bubble x1, $27 Wallmaster x4
$0D   $26  $9F  $38  $24  $0F  $17  7    wall     wall     bomb     shutter  2   3   3    $38   -    $24  -   -    $0F 5 rupees           1:$A0,$C0  all dead: shutters open, room item appears  $38 Digdogger x1
$0E   $26  $FE  $85  $0C  $03  $01  7    wall     wall     shutter  shutter  2   2   6    $05   -    $0C  -   -    -                      0:$C0,$90  all dead: shutters open                     $05 Goriya x6
$0F   $22  $E7  $45  $23  $19  $27  7    wall     open     shutter  wall     2   3   5    $05   -    $23  -   -    $19 key                2:$80,$90  all dead: shutters open, room item appears  $05 Goriya x5
$10   $26  $3F  $72  $95  $0F  $17  9    wall     wall     wall     shutter  2   3   5    $72   -    $15  -   -    $0F 5 rupees           1:$A0,$C0  all dead: shutters open, room item appears  list $72: $2B Bubble x2, $12 Vire x3
$11   $06  $E3  $F1  $A3  $03  $01  9    open     wall     shutter  open     2   3   8    $71   -    $23  -   -    -                      0:$C0,$90  all dead: shutters open                     list $71: $2B Bubble x3, $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$12   $26  $0B  $F6  $99  $03  $00  9    wall     wall     open     false    2   3   8    $76   -    $19  -   -    -                      0:$C0,$90  -                                           list $76: $4A Trap x1, $00 (empty) x3, $17 Like-Like x4
$13   $3E  $48  $0D  $A6  $03  $00  9    wall     shutter  false    false    2   0   3    $4D   -    $26  -   -    -                      0:$C0,$90  -                                           $4D dungeon person x1
$14   $22  $22  $ED  $00  $03  $00  9    wall     open     wall     open     2   2   8    $2D   -    $00  -   -    -                      0:$C0,$90  -                                           $2D Bubble x8
$15   $22  $12  $7B  $CE  $03  $05  9    wall     open     open     bomb     2   2   5    $7B   B    $0E  -   -    -                      0:$C0,$90  push block: stairs appear                   list $7B: $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$16   $46  $9F  $08  $A3  $23  $01  9    false    wall     bomb     shutter  2   3   3    $48   -    $23  -   $10  -                      0:$C0,$90  all dead: shutters open                     $48 Patra x1
$17   $E6  $E7  $3E  $28  $8E  $03  9    shutter  wall     shutter  wall     2   3   3    $3E   -    $28  D   -    $0E Triforce of Power  0:$C0,$90  last boss dead: shutters open               $3E Ganon x1
$18   $EA  $22  $86  $62  $03  $05  7    shutter  false    wall     open     2   2   6    $06   B    $22  -   -    -                      0:$C0,$90  push block: stairs appear                   $06 Goriya x6
$19   $26  $E6  $6C  $4A  $03  $05  7    wall     wall     shutter  wall     2   2   5    $2C   B    $0A  -   -    -                      0:$C0,$90  push block: stairs appear                   $2C Bubble x5
$1A   $22  $26  $DB  $5A  $03  $00  8    wall     open     wall     wall     2   2   8    $1B   B    $1A  -   -    -                      0:$C0,$90  -                                           $1B Keese blue x8
$1B   $26  $22  $00  $29  $1B  $20  8    wall     wall     wall     open     2   2   3    $00   -    $29  -   -    $1B Triforce piece     2:$80,$90  -                                           none
$1C   $26  $FE  $05  $85  $1A  $17  8    wall     wall     shutter  shutter  2   2   3    $45   -    $05  -   -    $1A heart container    1:$A0,$C0  all dead: shutters open, room item appears  $45 Gleeok x1
$1D   $26  $F7  $4C  $24  $20  $27  8    wall     wall     shutter  key      2   3   5    $0C   -    $24  -   $10  $00 bomb               2:$80,$90  all dead: shutters open, room item appears  $0C Darknut blue x5
$1E   $26  $A6  $FE  $9B  $03  $00  8    wall     wall     key      wall     2   2   8    $7E   -    $1B  -   -    -                      0:$C0,$90  -                                           list $7E: $2D Bubble x3, $2C Bubble x1, $0C Darknut blue x2, $0B Darknut red x2
$1F   $02  $27  $EE  $96  $96  $37  7    open     open     wall     wall     2   3   8    $6E   -    $16  D   -    $16 compass            3:$70,$90  all dead: shutters open, room item appears  list $6E: $4A Trap x1, $00 (empty) x3, $1B Keese blue x4
$20   $08  $58  $69  $3E  $03  $00  c7   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$C0,$90  -                                           tunnel cellar: left exit to $08, right exit to $58, Link appears at $60,$9D
$21   $6A  $4D  $39  $3E  $03  $20  c7   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      2:$80,$90  -                                           tunnel cellar: left exit to $6A, right exit to $4D, Link appears at $30,$9D
$22   $32  $37  $A3  $23  $03  $00  9    wall     bomb     wall     key      2   3   6    $23   -    $23  -   -    -                      0:$C0,$90  -                                           $23 Blue Wizzrobe x6
$23   $02  $A3  $FC  $96  $03  $10  9    open     open     key      open     2   3   8    $7C   -    $16  -   -    -                      1:$A0,$C0  -                                           list $7C: $2B Bubble x1, $17 Like-Like x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$24   $02  $E3  $00  $60  $83  $00  9    open     open     shutter  open     2   3   3    $00   B    $20  D   -    -                      0:$C0,$90  -                                           none
$25   $02  $06  $00  $60  $03  $00  9    open     open     open     wall     2   2   3    $00   B    $20  -   -    -                      0:$C0,$90  -                                           none
$26   $69  $3F  $C5  $3E  $03  $00  c7   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$C0,$90  -                                           tunnel cellar: left exit to $69, right exit to $3F, Link appears at $C0,$5D
$27   $78  $3A  $39  $3E  $03  $00  c8   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$C0,$90  -                                           tunnel cellar: left exit to $78, right exit to $3A, Link appears at $30,$9D
$28   $56  $23  $31  $23  $03  $00  7    false    key      wall     open     2   3   3    $31   -    $23  -   -    -                      0:$C0,$90  -                                           $31 Dodongo x3
$29   $32  $E6  $F5  $CD  $03  $05  7    wall     bomb     shutter  wall     2   2   8    $75   B    $0D  -   -    -                      0:$C0,$90  push block: stairs appear                   list $75: $2B Bubble x2, $05 Goriya x3, $1B Keese blue x3
$2A   $06  $26  $B8  $DF  $19  $05  8    open     wall     wall     wall     2   2   6    $78   B    $1F  -   -    $19 key                0:$C0,$90  push block: stairs appear                   list $78: $16 Pols Voice x2, $0C Darknut blue x2, $0B Darknut red x2
$2B   $38  $38  $39  $3F  $10  $20  c7   -        -        -        -        -   -   -    -     -    $3F  -   -    $10 rod                2:$80,$90  -                                           treasure cellar: left exit to $38, right exit to $38, Link appears at $30,$9D
$2C   $48  $48  $69  $3F  $0B  $20  c7   -        -        -        -        -   -   -    -     -    $3F  -   -    $0B magic key          2:$80,$90  -                                           treasure cellar: left exit to $48, right exit to $48, Link appears at $60,$9D
$2D   $22  $26  $00  $29  $1B  $20  7    wall     open     wall     wall     2   2   3    $00   -    $29  -   -    $1B Triforce piece     2:$80,$90  -                                           none
$2E   $22  $26  $EE  $C1  $19  $25  8    wall     open     wall     wall     2   2   8    $6E   B    $01  -   -    $19 key                2:$80,$90  push block: stairs appear                   list $6E: $4A Trap x1, $00 (empty) x3, $1B Keese blue x4
$2F   $02  $24  $36  $26  $03  $00  7    open     open     wall     wall     2   0   3    $36   -    $26  -   -    -                      0:$C0,$90  -                                           $36 Grumble x1
$30   $5B  $2E  $6A  $3E  $03  $00  c8   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$C0,$90  -                                           tunnel cellar: left exit to $5B, right exit to $2E, Link appears at $60,$AD
$31   $3E  $22  $F7  $81  $03  $01  9    wall     shutter  wall     open     2   2   8    $77   -    $01  -   -    -                      0:$C0,$90  all dead: shutters open                     list $77: $4A Trap x1, $00 (empty) x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$32   $92  $03  $FD  $A3  $19  $27  9    bomb     bomb     open     open     2   3   8    $7D   -    $23  -   -    $19 key                2:$80,$90  all dead: shutters open, room item appears  list $7D: $2D Bubble x3, $2C Bubble x1, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$33   $F6  $E2  $00  $60  $03  $00  9    shutter  key      shutter  open     2   2   3    $00   B    $20  -   -    -                      0:$C0,$90  -                                           none
$34   $12  $02  $00  $60  $03  $00  9    open     bomb     open     open     2   2   3    $00   B    $20  -   -    -                      0:$C0,$90  -                                           none
$35   $06  $26  $2D  $4A  $03  $05  9    open     wall     wall     wall     2   2   3    $2D   B    $0A  -   -    -                      0:$C0,$90  push block: stairs appear                   $2D Bubble x3
$36   $3E  $06  $72  $88  $03  $00  9    wall     shutter  open     wall     2   2   5    $72   -    $08  -   -    -                      0:$C0,$90  -                                           list $72: $2B Bubble x2, $12 Vire x3
$37   $5C  $1E  $69  $3E  $03  $00  c8   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$C0,$90  -                                           tunnel cellar: left exit to $5C, right exit to $1E, Link appears at $60,$9D
$38   $A6  $26  $FF  $C8  $03  $05  7    key      wall     wall     wall     2   2   8    $7F   B    $08  -   -    -                      0:$C0,$90  push block: stairs appear                   list $7F: $2D Bubble x3, $2C Bubble x1, $27 Wallmaster x4
$39   $86  $27  $45  $23  $17  $27  7    bomb     wall     wall     wall     2   3   5    $05   -    $23  -   -    $17 map                2:$80,$90  all dead: shutters open, room item appears  $05 Goriya x5
$3A   $3E  $3E  $2D  $62  $03  $05  8    wall     shutter  wall     shutter  2   2   3    $2D   B    $22  -   -    -                      0:$C0,$90  push block: stairs appear                   $2D Bubble x3
$3B   $3E  $04  $11  $A6  $03  $06  8    wall     shutter  open     wall     2   0   3    $51   -    $26  -   -    -                      0:$C0,$90  life-or-money person gone: shutters open    $51 life-or-money person x1
$3C   $3E  $27  $3C  $24  $0F  $07  8    wall     shutter  wall     wall     2   3   3    $3C   -    $24  -   -    $0F 5 rupees           0:$C0,$90  all dead: shutters open, room item appears  $3C Manhandla x1
$3D   $F6  $27  $31  $24  $1A  $07  7    shutter  key      wall     wall     2   3   3    $31   -    $24  -   -    $1A heart container    0:$C0,$90  all dead: shutters open, room item appears  $31 Dodongo x3
$3E   $E2  $27  $33  $23  $03  $01  8    shutter  open     wall     wall     2   3   3    $33   -    $23  -   -    -                      0:$C0,$90  all dead: shutters open                     $33 Gohma x1
$3F   $02  $26  $38  $46  $03  $05  7    open     open     wall     wall     2   2   3    $38   B    $06  -   -    -                      0:$C0,$90  push block: stairs appear                   $38 Digdogger x1
$40   $22  $32  $AD  $23  $03  $00  9    wall     open     wall     bomb     2   2   6    $2D   -    $23  -   -    -                      0:$C0,$90  -                                           $2D Bubble x6
$41   $16  $9E  $00  $60  $03  $00  9    open     key      bomb     shutter  2   2   3    $00   B    $20  -   -    -                      0:$C0,$90  -                                           none
$42   $8A  $02  $53  $26  $16  $17  -    bomb     false    open     open     2   2   5    $13   -    $26  -   -    $16 compass            1:?        all dead: shutters open, room item appears  $13 Zol x5
$43   $A2  $3F  $A4  $12  $03  $01  9    key      open     wall     shutter  2   3   6    $24   -    $12  -   -    -                      0:$C0,$90  all dead: shutters open                     $24 Red Wizzrobe x6
$44   $82  $E7  $08  $A3  $0F  $27  9    bomb     open     shutter  wall     2   3   3    $48   -    $23  -   -    $0F 5 rupees           2:$80,$90  all dead: shutters open, room item appears  $48 Patra x1
$45   $52  $2A  $7B  $E2  $03  $05  -    false    bomb     wall     false    2   2   5    $7B   B    $22  -   -    -                      0:?        push block: stairs appear                   list $7B: $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$46   $12  $53  $07  $A4  $17  $27  9    open     bomb     false    bomb     2   3   3    $47   -    $24  -   -    $17 map                2:$80,$90  all dead: shutters open, room item appears  $47 Patra x1
$47   $2A  $87  $B3  $A3  $03  $00  9    wall     false    bomb     wall     2   3   6    $73   -    $23  -   -    -                      0:$C0,$90  -                                           list $73: $2B Bubble x2, $13 Zol x2, $17 Like-Like x2
$48   $26  $36  $AA  $51  $03  $05  7    wall     wall     wall     key      2   2   6    $2A   B    $11  -   -    -                      0:$C0,$90  push block: stairs appear                   $2A Stalfos x6
$49   $22  $A6  $FF  $C8  $03  $05  7    wall     open     key      wall     2   2   8    $7F   B    $08  -   -    -                      0:$C0,$90  push block: stairs appear                   list $7F: $2D Bubble x3, $2C Bubble x1, $27 Wallmaster x4
$4A   $16  $3E  $F0  $4A  $03  $05  8    open     key      wall     shutter  2   2   8    $30   B    $0A  -   -    -                      0:$C0,$90  push block: stairs appear                   $30 Gibdo x8
$4B   $1E  $02  $00  $60  $03  $00  8    open     shutter  open     open     2   2   3    $00   B    $20  -   -    -                      0:$C0,$90  -                                           none
$4C   $1E  $E4  $11  $A6  $03  $06  8    open     shutter  shutter  wall     2   0   3    $51   -    $26  -   -    -                      0:$C0,$90  life-or-money person gone: shutters open    $51 life-or-money person x1
$4D   $B2  $26  $01  $C7  $43  $05  7    key      bomb     wall     wall     2   2   3    $41   B    $07  -   $20  -                      0:$C0,$90  push block: stairs appear                   $41 Moldorm x1
$4E   $1E  $27  $8C  $23  $03  $01  8    open     shutter  wall     wall     2   3   6    $0C   -    $23  -   -    -                      0:$C0,$90  all dead: shutters open                     $0C Darknut blue x6
$4F   $02  $27  $85  $18  $03  $00  7    open     open     wall     wall     2   3   6    $05   -    $18  -   -    -                      0:$C0,$90  -                                           $05 Goriya x6
$50   $06  $37  $3A  $24  $00  $17  9    open     wall     wall     key      2   3   3    $3A   -    $24  -   -    $00 bomb               1:$A0,$C0  all dead: shutters open, room item appears  $3A Lanmola x1
$51   $B6  $A7  $DD  $0D  $83  $00  9    key      key      key      wall     2   3   8    $1D   -    $0D  D   -    -                      0:$C0,$90  -                                           $1D Keese black x8
$52   $22  $26  $F1  $83  $03  $00  9    wall     open     wall     wall     2   2   8    $71   -    $03  -   -    -                      0:$C0,$90  -                                           list $71: $2B Bubble x3, $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$53   $02  $27  $F7  $94  $03  $00  9    open     open     wall     wall     2   3   8    $77   -    $14  -   -    -                      0:$C0,$90  -                                           list $77: $4A Trap x1, $00 (empty) x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$54   $02  $2B  $AD  $96  $83  $00  9    open     open     wall     false    2   3   6    $6D   -    $16  D   -    -                      0:$C0,$90  -                                           list $6D: $4A Trap x1, $00 (empty) x3, $13 Zol x2
$55   $82  $4B  $DD  $0F  $8F  $37  9    bomb     open     false    false    2   3   8    $1D   -    $0F  D   -    $0F 5 rupees           3:$B0,$60  all dead: shutters open, room item appears  $1D Keese black x8
$56   $82  $5E  $7B  $8C  $03  $00  9    bomb     open     false    shutter  2   2   5    $7B   -    $0C  -   -    -                      0:$C0,$90  -                                           list $7B: $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$57   $46  $06  $FD  $DF  $03  $05  9    false    wall     open     wall     2   2   8    $7D   B    $1F  -   -    -                      0:$C0,$90  push block: stairs appear                   list $7D: $2D Bubble x3, $2C Bubble x1, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$58   $2A  $3E  $E8  $62  $03  $05  7    wall     false    wall     shutter  2   2   8    $28   B    $22  -   -    -                      0:$C0,$90  push block: stairs appear                   $28 Rope x8
$59   $EA  $06  $00  $60  $03  $00  7    shutter  false    open     wall     2   2   3    $00   B    $20  -   -    -                      0:$C0,$90  -                                           none
$5A   $A6  $27  $8C  $23  $00  $27  8    key      wall     wall     wall     2   3   6    $0C   -    $23  -   -    $00 bomb               2:$80,$90  all dead: shutters open, room item appears  $0C Darknut blue x6
$5B   $06  $26  $CC  $5F  $03  $05  8    open     wall     wall     wall     2   2   8    $0C   B    $1F  -   -    -                      0:$C0,$90  push block: stairs appear                   $0C Darknut blue x8
$5C   $06  $26  $8C  $51  $03  $05  8    open     wall     wall     wall     2   2   6    $0C   B    $11  -   -    -                      0:$C0,$90  push block: stairs appear                   $0C Darknut blue x6
$5D   $82  $26  $BA  $9E  $0F  $17  7    bomb     open     wall     wall     2   2   6    $7A   -    $1E  -   -    $0F 5 rupees           1:$A0,$C0  all dead: shutters open, room item appears  list $7A: $05 Goriya x3, $06 Goriya x3
$5E   $E2  $27  $4C  $24  $0F  $27  8    shutter  open     wall     wall     2   3   5    $0C   -    $24  -   -    $0F 5 rupees           2:$80,$90  all dead: shutters open, room item appears  $0C Darknut blue x5
$5F   $02  $26  $E8  $51  $03  $05  7    open     open     wall     wall     2   2   8    $28   B    $11  -   -    -                      0:$C0,$90  push block: stairs appear                   $28 Rope x8
$60   $4A  $7D  $69  $3E  $03  $00  c8   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$C0,$90  -                                           tunnel cellar: left exit to $4A, right exit to $7D, Link appears at $60,$9D
$61   $A6  $23  $52  $23  $0F  $17  9    key      wall     wall     open     2   3   5    $12   -    $23  -   -    $0F 5 rupees           1:$A0,$C0  all dead: shutters open, room item appears  $12 Vire x5
$62   $06  $1F  $F6  $A3  $03  $01  9    open     wall     open     shutter  2   3   8    $76   -    $23  -   -    -                      0:$C0,$90  all dead: shutters open                     list $76: $4A Trap x1, $00 (empty) x3, $17 Like-Like x4
$63   $12  $E3  $3B  $1D  $80  $27  9    open     bomb     shutter  open     2   3   3    $3B   -    $1D  D   -    $00 bomb               2:$80,$90  all dead: shutters open, room item appears  $3B Lanmola x1
$64   $E2  $FC  $0B  $A6  $03  $00  9    shutter  open     shutter  shutter  2   0   3    $4B   -    $26  -   -    -                      0:$C0,$90  -                                           $4B dungeon person x1
$65   $06  $03  $7B  $95  $83  $00  9    open     wall     open     open     2   3   5    $7B   -    $15  D   -    -                      0:$C0,$90  -                                           list $7B: $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$66   $26  $07  $FC  $A3  $0F  $27  9    wall     wall     open     wall     2   3   8    $7C   -    $23  -   -    $0F 5 rupees           2:$80,$90  all dead: shutters open, room item appears  list $7C: $2B Bubble x1, $17 Like-Like x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$67   $1A  $7C  $69  $3E  $03  $00  c8   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$C0,$90  -                                           tunnel cellar: left exit to $1A, right exit to $7C, Link appears at $60,$9D
$68   $46  $23  $F5  $96  $0F  $37  7    false    wall     wall     open     2   3   8    $75   -    $16  -   -    $0F 5 rupees           3:$70,$90  all dead: shutters open, room item appears  list $75: $2B Bubble x2, $05 Goriya x3, $1B Keese blue x3
$69   $46  $16  $DB  $0F  $03  $00  7    false    wall     open     key      2   2   8    $1B   -    $0F  -   -    -                      0:$C0,$90  -                                           $1B Keese blue x8
$6A   $26  $A6  $85  $5A  $00  $17  7    wall     wall     key      wall     2   2   6    $05   B    $1A  -   -    $00 bomb               1:$A0,$C0  all dead: shutters open, room item appears  $05 Goriya x6
$6B   $26  $20  $0F  $A6  $03  $00  7    wall     wall     wall     open     2   0   3    $4F   -    $26  -   -    -                      0:$C0,$90  -                                           $4F dungeon person x1
$6C   $26  $E2  $FF  $C8  $03  $04  7    wall     wall     shutter  open     2   2   8    $7F   B    $08  -   -    -                      0:$C0,$90  push block: shutters open                   list $7F: $2D Bubble x3, $2C Bubble x1, $27 Wallmaster x4
$6D   $06  $E6  $3D  $04  $03  $01  7    open     wall     shutter  wall     2   2   3    $3D   -    $04  -   -    -                      0:$C0,$90  all dead: shutters open                     $3D Aquamentus x1
$6E   $02  $27  $FE  $98  $97  $37  8    open     open     wall     wall     2   3   8    $7E   -    $18  D   -    $17 map                3:$50,$70  all dead: shutters open, room item appears  list $7E: $2D Bubble x3, $2C Bubble x1, $0C Darknut blue x2, $0B Darknut red x2
$6F   $02  $26  $AD  $4A  $03  $05  7    open     open     wall     wall     2   2   6    $2D   B    $0A  -   -    -                      0:$C0,$90  push block: stairs appear                   $2D Bubble x6
$70   $2A  $2A  $39  $3F  $07  $20  c8   -        -        -        -        -   -   -    -     -    $3F  -   -    $07 red candle         2:$80,$90  -                                           treasure cellar: left exit to $2A, right exit to $2A, Link appears at $30,$9D
$71   $73  $35  $38  $3E  $03  $00  c9   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$C0,$90  -                                           tunnel cellar: left exit to $73, right exit to $35, Link appears at $30,$8D
$72   $01  $15  $C5  $3E  $03  $00  c9   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$C0,$90  -                                           tunnel cellar: left exit to $01, right exit to $15, Link appears at $C0,$5D
$73   $86  $22  $FD  $9D  $03  $00  9    bomb     wall     wall     open     2   2   8    $7D   -    $1D  -   -    -                      0:$C0,$90  -                                           list $7D: $2D Bubble x3, $2C Bubble x1, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$74   $02  $27  $00  $21  $03  $00  9    open     open     wall     wall     2   3   3    $00   -    $21  -   -    -                      0:$C0,$90  -                                           none
$75   $06  $57  $B5  $3E  $03  $00  c9   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$C0,$90  -                                           tunnel cellar: left exit to $06, right exit to $57, Link appears at $B0,$5D
$76   $45  $45  $CA  $3F  $09  $20  c9   -        -        -        -        -   -   -    -     -    $3F  -   -    $09 silver arrow       2:$80,$90  -                                           treasure cellar: left exit to $45, right exit to $45, Link appears at $C0,$AD
$77   $00  $00  $46  $3F  $13  $20  c9   -        -        -        -        -   -   -    -     -    $3F  -   -    $13 red ring           2:$80,$90  -                                           treasure cellar: left exit to $00, right exit to $00, Link appears at $40,$6D
$78   $26  $22  $AD  $DF  $16  $15  8    wall     wall     wall     open     2   2   6    $6D   B    $1F  -   -    $16 compass            1:$A0,$C0  push block: stairs appear                   list $6D: $4A Trap x1, $00 (empty) x3, $13 Zol x2
$79   $22  $03  $00  $21  $03  $00  8    wall     open     open     open     2   3   3    $00   -    $21  -   -    -                      0:$C0,$90  -                                           none
$7A   $26  $03  $B8  $A3  $03  $00  8    wall     wall     open     open     2   3   6    $78   -    $23  -   -    -                      0:$C0,$90  -                                           list $78: $16 Pols Voice x2, $0C Darknut blue x2, $0B Darknut red x2
$7B   $26  $E3  $FE  $93  $83  $01  8    wall     wall     shutter  open     2   3   8    $7E   -    $13  D   -    -                      0:$C0,$90  all dead: shutters open                     list $7E: $2D Bubble x3, $2C Bubble x1, $0C Darknut blue x2, $0B Darknut red x2
$7C   $26  $02  $F4  $D1  $03  $05  8    wall     wall     open     open     2   2   8    $74   B    $11  -   -    -                      0:$C0,$90  push block: stairs appear                   list $74: $2B Bubble x2, $0C Darknut blue x1, $0B Darknut red x2, $30 Gibdo x3
$7D   $26  $FE  $3C  $46  $03  $04  8    wall     wall     shutter  shutter  2   2   3    $3C   B    $06  -   -    -                      0:$C0,$90  push block: shutters open                   $3C Manhandla x1
$7E   $06  $07  $FE  $95  $83  $00  8    open     wall     open     wall     2   3   8    $7E   -    $15  D   -    -                      0:$C0,$90  -                                           list $7E: $2D Bubble x3, $2C Bubble x1, $0C Darknut blue x2, $0B Darknut red x2
$7F   $02  $27  $00  $21  $03  $00  7    open     open     wall     wall     2   3   3    $00   -    $21  -   -    -                      0:$C0,$90  -                                           none
```
