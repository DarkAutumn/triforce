# Underworld level block, levels 7-9, Q1

How to read this: one row per room `$00`-`$7F` of [`Level_BlockUW7to9Q1`](../src/bank6/level-blocks-common-data.asm#L74) (`dat/LevelBlockUW2Q1.dat`), the level block that levels 7, 8, 9 share in
the first quest. Each row gives the six raw attribute bytes A-F, then the fields the game reads from
them. The formats are in `README.md`. Find a room with `grep '^\$32 '`.

Columns:

- `lvl`: the level whose submenu map shows the room; `cN` = a cellar listed by level N; `-` = neither
  (the room is drawn on no level's map). Levels are internal level numbers ([`CurLevel`](../src/Variables.inc#L12)). [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L728), [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L725)
- `N`, `S`, `W`, `E`: door type per side (A bits 7-5, A bits 4-2, B bits 7-5, B bits 4-2): `open`,
  `wall`, `false`/`false2` (walk-through wall), `bomb` (bombable wall), `key`/`key2` (key door),
  `shutter`. [`Door_FindTypeByDirBit`](../src/bank5/fill-walls.asm#L55), [`Door_Touch_JumpTable`](../src/bank5/touch-door.asm#L62)
- `po`, `pi`: outer and inner palette selectors (A bits 1-0, B bits 1-0). [`Room_FillPlayAreaAttrs`](../src/bank5/mode7-scroll-init.asm#L379)
- `cnt`: monster count selected by C bits 7-6 from [`LevelInfo_FoeCounts`](../src/Variables.inc#L715); `list`: monster list id (C bits 5-0, D bit 7).
  The `monsters` text applies the rules of `objlists.md`. [`Room_EnterCommon`](../src/bank5/enter-room.asm#L137)
- `blk`: `B` = the room has a push block (D bit 6). [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- `lay`: room layout id (D bits 5-0), see `room-layouts.md`; `$3E`/`$3F` = cellar. [`RoomLayout_LayOutUWFloor`](../src/bank5/uw-floor-push-block.asm#L67)
- `dk`: `D` = dark room (E bit 7). [`Room_IsDark_Bank5`](../src/bank5/door-flags-dark-room.asm#L124)
- `snd`: boss sound request heard in the room (E bits 6-5 through [`Sample_BossRoarRequests`](../src/bank5/boss-sound-door-face.asm#L30)). [`Sample_SetBossRoarUW`](../src/bank5/boss-sound-door-face.asm#L42)
- `item`: room item (E bits 4-0; `$03` = none, shown as `-`). [`Room_SetUpItemAndPushBlock`](../src/bank5/create-room-objects.asm#L51)
- `pos`: position index (F bits 5-4) and the item's X,Y from the owning level's [`LevelInfo_ShortcutOrItemPosArray`](../src/Variables.inc#L717) (`?` when the room
  has no `lvl`). [`Room_GetShortcutOrItemXYForRoomId`](../src/bank1/bomb-candle-wield.asm#L280)
- `secret`: secret trigger (F bits 2-0). [`Secret_RunTrigger_JumpTable`](../src/bank5/underworld-secrets.asm#L105)
- `monsters; cellar`: the monsters made on entry (before the kill count is subtracted). For a cellar:
  the rooms its left and right exits lead to (bytes A and B) and where Link appears there (byte C);
  its door columns are `-` and its monster list is not used. [`Warp_CheckSubroomExit`](../src/bank5/cave-cellar-exits.asm#L43), [`Room_EnterCommon`](../src/bank5/enter-room.asm#L137)

Levels of this block:

```
level  shown as  start  Triforce  boss  cellars                          counts           item positions
7      LEVEL-7   $79    $2B       $2A   $7B,$4A                          $03 $05 $06 $08  $80,$90 $D0,$60 $20,$60 $20,$C0
8      LEVEL-8   $7E    $2C       $3C   $2F,$0F,$6F                      $03 $05 $06 $08  $80,$90 $D0,$60 $20,$60 $20,$C0
9      LEVEL-9   $76    $32       $42   $60,$70,$72,$75,$67,$77,$00,$4F  $03 $05 $06 $08  $80,$90 $D0,$60 $20,$60 $20,$C0
```

# Rooms

```
room  A    B    C    D    E    F    lvl  N        S        W        E        po  pi  cnt  list  blk  lay  dk  snd  item                   pos        secret                                      monsters; cellar
$00   $07  $07  $69  $3F  $13  $00  c9   -        -        -        -        -   -   -    -     -    $3F  -   -    $13 red ring           0:$80,$90  -                                           treasure cellar: left exit to $07, right exit to $07, Link appears at $60,$9D
$01   $3E  $37  $EF  $93  $83  $00  9    wall     shutter  wall     key      2   3   8    $6F   -    $13  D   -    -                      0:$80,$90  -                                           list $6F: $2B Bubble x3, $13 Zol x2, $1B Keese blue x3
$02   $22  $A4  $0E  $A6  $03  $00  9    wall     open     key      wall     2   0   3    $4E   -    $26  -   -    -                      0:$80,$90  -                                           $4E dungeon person x1
$03   $26  $32  $B3  $DA  $03  $00  9    wall     wall     wall     bomb     2   2   6    $73   B    $1A  -   -    -                      0:$80,$90  -                                           list $73: $2B Bubble x2, $13 Zol x2, $17 Like-Like x2
$04   $26  $86  $F7  $DA  $03  $00  9    wall     wall     bomb     wall     2   2   8    $77   B    $1A  -   -    -                      0:$80,$90  -                                           list $77: $4A Trap x1, $00 (empty) x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$05   $26  $32  $7B  $CA  $03  $05  9    wall     wall     wall     bomb     2   2   5    $7B   B    $0A  -   -    -                      0:$80,$90  push block: stairs appear                   list $7B: $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$06   $36  $84  $0D  $A6  $03  $00  9    wall     key      bomb     wall     2   0   3    $4D   -    $26  -   -    -                      0:$80,$90  -                                           $4D dungeon person x1
$07   $32  $26  $F1  $DA  $03  $00  -    wall     bomb     wall     wall     2   2   8    $71   B    $1A  -   -    -                      0:?        -                                           list $71: $2B Bubble x3, $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$08   $32  $32  $35  $26  $03  $00  -    wall     bomb     wall     bomb     2   2   3    $35   -    $26  -   -    -                      0:?        -                                           $35 Rupee stash x1
$09   $3E  $9E  $BA  $8A  $0F  $07  7    wall     shutter  bomb     shutter  2   2   6    $7A   -    $0A  -   -    $0F 5 rupees           0:$80,$90  all dead: shutters open, room item appears  list $7A: $05 Goriya x3, $06 Goriya x3
$0A   $26  $E6  $45  $24  $19  $01  7    wall     wall     shutter  wall     2   2   5    $05   -    $24  -   -    $19 key                0:$80,$90  all dead: shutters open                     $05 Goriya x5
$0B   $36  $25  $01  $A5  $00  $07  7    wall     key      wall     wall     2   1   3    $41   -    $25  -   -    $00 bomb               0:$80,$90  all dead: shutters open, room item appears  $41 Moldorm x1
$0C   $22  $33  $31  $23  $00  $07  7    wall     open     wall     bomb     2   3   3    $31   -    $23  -   -    $00 bomb               0:$80,$90  all dead: shutters open, room item appears  $31 Dodongo x3
$0D   $26  $86  $F9  $C8  $03  $05  7    wall     wall     bomb     wall     2   2   8    $79   B    $08  -   -    -                      0:$80,$90  push block: stairs appear                   list $79: $2B Bubble x3, $27 Wallmaster x5
$0E   $3E  $26  $4B  $00  $00  $07  8    wall     shutter  wall     wall     2   2   5    $0B   -    $00  -   -    $00 bomb               0:$80,$90  all dead: shutters open, room item appears  $0B Darknut red x5
$0F   $1F  $1F  $69  $3F  $0B  $00  c8   -        -        -        -        -   -   -    -     -    $3F  -   -    $0B magic key          0:$80,$90  -                                           treasure cellar: left exit to $1F, right exit to $1F, Link appears at $60,$9D
$10   $32  $26  $F1  $C8  $03  $05  9    wall     bomb     wall     wall     2   2   8    $71   B    $08  -   -    -                      0:$80,$90  push block: stairs appear                   list $71: $2B Bubble x3, $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$11   $1E  $26  $3B  $25  $00  $07  9    open     shutter  wall     wall     2   2   3    $3B   -    $25  -   -    $00 bomb               0:$80,$90  all dead: shutters open, room item appears  $3B Lanmola x1
$12   $F6  $36  $3A  $26  $0F  $07  9    shutter  key      wall     key      2   2   3    $3A   -    $26  -   -    $0F 5 rupees           0:$80,$90  all dead: shutters open, room item appears  $3A Lanmola x1
$13   $36  $A7  $B3  $80  $83  $00  9    wall     key      key      wall     2   3   6    $73   -    $00  D   -    -                      0:$80,$90  -                                           list $73: $2B Bubble x2, $13 Zol x2, $17 Like-Like x2
$14   $22  $36  $57  $1C  $03  $00  9    wall     open     wall     key      2   2   5    $17   -    $1C  -   -    -                      0:$80,$90  -                                           $17 Like-Like x5
$15   $32  $A3  $23  $14  $8F  $07  9    wall     bomb     key      open     2   3   3    $23   -    $14  D   -    $0F 5 rupees           0:$80,$90  all dead: shutters open, room item appears  $23 Blue Wizzrobe x3
$16   $A2  $06  $08  $8A  $00  $07  9    key      open     open     wall     2   2   3    $48   -    $0A  -   -    $00 bomb               0:$80,$90  all dead: shutters open, room item appears  $48 Patra x1
$17   $92  $26  $FC  $9F  $03  $10  9    bomb     bomb     wall     wall     2   2   8    $7C   -    $1F  -   -    -                      1:$D0,$60  -                                           list $7C: $2B Bubble x1, $17 Like-Like x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$18   $9E  $37  $F5  $96  $97  $01  7    bomb     shutter  wall     key      2   3   8    $75   -    $16  D   -    $17 map                0:$80,$90  all dead: shutters open                     list $75: $2B Bubble x2, $05 Goriya x3, $1B Keese blue x3
$19   $E6  $B2  $85  $1F  $03  $01  7    shutter  wall     key      bomb     2   2   6    $05   -    $1F  -   -    -                      0:$80,$90  all dead: shutters open                     $05 Goriya x6
$1A   $26  $92  $BA  $DA  $23  $00  -    wall     wall     bomb     bomb     2   2   6    $7A   B    $1A  -   $10  -                      0:?        -                                           list $7A: $05 Goriya x3, $06 Goriya x3
$1B   $A6  $96  $85  $00  $00  $07  7    key      wall     bomb     key      2   2   6    $05   -    $00  -   -    $00 bomb               0:$80,$90  all dead: shutters open, room item appears  $05 Goriya x6
$1C   $E6  $A6  $38  $24  $03  $01  7    shutter  wall     key      wall     2   2   3    $38   -    $24  -   -    -                      0:$80,$90  all dead: shutters open                     $38 Digdogger x1
$1D   $22  $32  $56  $11  $00  $07  8    wall     open     wall     bomb     2   2   5    $16   -    $11  -   -    $00 bomb               0:$80,$90  all dead: shutters open, room item appears  $16 Pols Voice x5
$1E   $F6  $9F  $33  $24  $03  $01  8    shutter  key      bomb     shutter  2   3   3    $33   -    $24  -   -    -                      0:$80,$90  all dead: shutters open                     $33 Gohma x1
$1F   $26  $06  $B8  $DA  $03  $01  8    wall     wall     open     wall     2   2   6    $78   B    $1A  -   -    -                      0:$80,$90  all dead: shutters open                     list $78: $16 Pols Voice x2, $0C Darknut blue x2, $0B Darknut red x2
$20   $86  $26  $7B  $DA  $03  $00  9    bomb     wall     wall     wall     2   2   5    $7B   B    $1A  -   -    -                      0:$80,$90  -                                           list $7B: $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$21   $1E  $32  $07  $A5  $03  $20  9    open     shutter  wall     bomb     2   2   3    $47   -    $25  -   -    -                      2:$20,$60  -                                           $47 Patra x1
$22   $A6  $96  $B3  $9D  $03  $00  9    key      wall     bomb     key      2   2   6    $73   -    $1D  -   -    -                      0:$80,$90  -                                           list $73: $2B Bubble x2, $13 Zol x2, $17 Like-Like x2
$23   $B6  $B3  $D5  $12  $00  $10  9    key      key      key      bomb     2   3   8    $15   -    $12  -   -    $00 bomb               1:$D0,$60  -                                           $15 Gel x8
$24   $02  $87  $52  $17  $83  $00  9    open     open     bomb     wall     2   3   5    $12   -    $17  D   -    -                      0:$80,$90  -                                           $12 Vire x5
$25   $92  $36  $B3  $8E  $00  $07  9    bomb     bomb     wall     key      2   2   6    $73   -    $0E  -   -    $00 bomb               0:$80,$90  all dead: shutters open, room item appears  list $73: $2B Bubble x2, $13 Zol x2, $17 Like-Like x2
$26   $16  $B2  $D5  $03  $0F  $07  9    open     key      key      bomb     2   2   8    $15   -    $03  -   -    $0F 5 rupees           0:$80,$90  all dead: shutters open, room item appears  $15 Gel x8
$27   $92  $86  $08  $8A  $17  $07  9    bomb     bomb     bomb     wall     2   2   3    $48   -    $0A  -   -    $17 map                0:$80,$90  all dead: shutters open, room item appears  $48 Patra x1
$28   $16  $24  $36  $26  $03  $00  7    open     key      wall     wall     2   0   3    $36   -    $26  -   -    -                      0:$80,$90  -                                           $36 Grumble x1
$29   $26  $32  $BA  $DA  $23  $00  7    wall     wall     wall     bomb     2   2   6    $7A   B    $1A  -   $10  -                      0:$80,$90  -                                           list $7A: $05 Goriya x3, $06 Goriya x3
$2A   $26  $9E  $3D  $04  $1A  $07  7    wall     wall     bomb     shutter  2   2   3    $3D   -    $04  -   -    $1A heart container    0:$80,$90  all dead: shutters open, room item appears  $3D Aquamentus x1
$2B   $26  $06  $00  $29  $1B  $00  7    wall     wall     open     wall     2   2   3    $00   -    $29  -   -    $1B Triforce piece     0:$80,$90  -                                           none
$2C   $22  $26  $00  $29  $1B  $00  8    wall     open     wall     wall     2   2   3    $00   -    $29  -   -    $1B Triforce piece     0:$80,$90  -                                           none
$2D   $06  $32  $35  $26  $03  $00  -    open     wall     wall     bomb     2   2   3    $35   -    $26  -   -    -                      0:?        -                                           $35 Rupee stash x1
$2E   $B2  $86  $3C  $00  $17  $07  8    key      bomb     bomb     wall     2   2   3    $3C   -    $00  -   -    $17 map                0:$80,$90  all dead: shutters open, room item appears  $3C Manhandla x1
$2F   $4C  $3F  $77  $3E  $03  $00  c8   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$80,$90  -                                           tunnel cellar: left exit to $4C, right exit to $3F, Link appears at $70,$7D
$30   $36  $32  $F7  $CA  $03  $05  9    wall     key      wall     bomb     2   2   8    $77   B    $0A  -   -    -                      0:$80,$90  push block: stairs appear                   list $77: $4A Trap x1, $00 (empty) x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$31   $1E  $86  $FC  $A5  $03  $00  9    open     shutter  bomb     wall     2   2   8    $7C   -    $25  -   -    -                      0:$80,$90  -                                           list $7C: $2B Bubble x1, $17 Like-Like x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$32   $3E  $26  $37  $27  $03  $01  -    wall     shutter  wall     wall     2   2   3    $37   -    $27  -   -    -                      0:?        all dead: shutters open                     $37 Zelda x1
$33   $A6  $33  $D5  $19  $83  $00  9    key      wall     wall     bomb     2   3   8    $15   -    $19  D   -    -                      0:$80,$90  -                                           $15 Gel x8
$34   $06  $87  $DB  $14  $0F  $07  9    open     wall     bomb     wall     2   3   8    $1B   -    $14  -   -    $0F 5 rupees           0:$80,$90  all dead: shutters open, room item appears  $1B Keese blue x8
$35   $86  $26  $7B  $A6  $16  $07  -    bomb     wall     wall     wall     2   2   5    $7B   -    $26  -   -    $16 compass            0:?        all dead: shutters open, room item appears  list $7B: $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$36   $A6  $32  $EF  $80  $03  $00  9    key      wall     wall     bomb     2   2   8    $6F   -    $00  -   -    -                      0:$80,$90  -                                           list $6F: $2B Bubble x3, $13 Zol x2, $1B Keese blue x3
$37   $82  $86  $92  $63  $00  $07  9    bomb     open     bomb     wall     2   2   6    $12   B    $23  -   -    $00 bomb               0:$80,$90  all dead: shutters open, room item appears  $12 Vire x6
$38   $A6  $3E  $BA  $8D  $0F  $17  7    key      wall     wall     shutter  2   2   6    $7A   -    $0D  -   -    $0F 5 rupees           1:$D0,$60  all dead: shutters open, room item appears  list $7A: $05 Goriya x3, $06 Goriya x3
$39   $22  $13  $38  $24  $03  $00  7    wall     open     open     bomb     2   3   3    $38   -    $24  -   -    -                      0:$80,$90  -                                           $38 Digdogger x1
$3A   $26  $86  $01  $A5  $39  $07  7    wall     wall     bomb     wall     2   2   3    $41   -    $25  -   $10  $19 key                0:$80,$90  all dead: shutters open, room item appears  $41 Moldorm x1
$3B   $22  $30  $10  $A6  $23  $00  8    wall     open     wall     bomb     2   0   3    $50   -    $26  -   $10  -                      0:$80,$90  -                                           $50 dungeon person x1
$3C   $F2  $86  $05  $85  $1A  $37  8    shutter  bomb     bomb     wall     2   2   3    $45   -    $05  -   -    $1A heart container    3:$20,$C0  all dead: shutters open, room item appears  $45 Gleeok x1
$3D   $22  $24  $0F  $A6  $23  $00  8    wall     open     wall     wall     2   0   3    $4F   -    $26  -   $10  -                      0:$80,$90  -                                           $4F dungeon person x1
$3E   $96  $3F  $8C  $23  $03  $01  8    bomb     key      wall     shutter  2   3   6    $0C   -    $23  -   -    -                      0:$80,$90  all dead: shutters open                     $0C Darknut blue x6
$3F   $26  $E6  $F4  $9B  $00  $07  8    wall     wall     shutter  wall     2   2   8    $74   -    $1B  -   -    $00 bomb               0:$80,$90  all dead: shutters open, room item appears  list $74: $2B Bubble x2, $0C Darknut blue x1, $0B Darknut red x2, $30 Gibdo x3
$40   $B6  $27  $7B  $96  $0F  $07  9    key      key      wall     wall     2   3   5    $7B   -    $16  -   -    $0F 5 rupees           0:$80,$90  all dead: shutters open, room item appears  list $7B: $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$41   $1E  $27  $F6  $80  $A3  $00  9    open     shutter  wall     wall     2   3   8    $76   -    $00  D   $10  -                      0:$80,$90  -                                           list $76: $4A Trap x1, $00 (empty) x3, $17 Like-Like x4
$42   $FE  $27  $3E  $28  $8E  $03  9    shutter  shutter  wall     wall     2   3   3    $3E   -    $28  D   -    $0E Triforce of Power  0:$80,$90  last boss dead: shutters open               $3E Ganon x1
$43   $32  $24  $0C  $A6  $23  $00  9    wall     bomb     wall     wall     2   0   3    $4C   -    $26  -   $10  -                      0:$80,$90  -                                           $4C dungeon person x1
$44   $22  $22  $A4  $00  $0F  $07  9    wall     open     wall     open     2   2   6    $24   -    $00  -   -    $0F 5 rupees           0:$80,$90  all dead: shutters open, room item appears  $24 Red Wizzrobe x6
$45   $32  $02  $FC  $80  $03  $00  9    wall     bomb     open     open     2   2   8    $7C   -    $00  -   -    -                      0:$80,$90  -                                           list $7C: $2B Bubble x1, $17 Like-Like x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$46   $22  $06  $A3  $25  $03  $00  9    wall     open     open     wall     2   2   6    $23   -    $25  -   -    -                      0:$80,$90  -                                           $23 Blue Wizzrobe x6
$47   $02  $26  $F1  $9E  $19  $10  9    open     open     wall     wall     2   2   8    $71   -    $1E  -   -    $19 key                1:$D0,$60  -                                           list $71: $2B Bubble x3, $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$48   $36  $24  $0F  $A6  $03  $10  7    wall     key      wall     wall     2   0   3    $4F   -    $26  -   -    -                      1:$D0,$60  -                                           $4F dungeon person x1
$49   $E2  $27  $F5  $98  $03  $01  7    shutter  open     wall     wall     2   3   8    $75   -    $18  -   -    -                      0:$80,$90  all dead: shutters open                     list $75: $2B Bubble x2, $05 Goriya x3, $1B Keese blue x3
$4A   $1A  $1A  $69  $3F  $07  $00  c7   -        -        -        -        -   -   -    -     -    $3F  -   -    $07 red candle         0:$80,$90  -                                           treasure cellar: left exit to $1A, right exit to $1A, Link appears at $60,$9D
$4B   $06  $3F  $8B  $23  $99  $01  8    open     wall     wall     shutter  2   3   6    $0B   -    $23  D   -    $19 key                0:$80,$90  all dead: shutters open                     $0B Darknut red x6
$4C   $86  $E6  $D6  $1C  $39  $11  8    bomb     wall     shutter  wall     2   2   8    $16   -    $1C  -   $10  $19 key                1:$D0,$60  all dead: shutters open                     $16 Pols Voice x8
$4D   $E2  $36  $33  $25  $03  $01  8    shutter  open     wall     key      2   2   3    $33   -    $25  -   -    -                      0:$80,$90  all dead: shutters open                     $33 Gohma x1
$4E   $A2  $A6  $F4  $80  $0F  $07  8    key      open     key      wall     2   2   8    $74   -    $00  -   -    $0F 5 rupees           0:$80,$90  all dead: shutters open, room item appears  list $74: $2B Bubble x2, $0C Darknut blue x1, $0B Darknut red x2, $30 Gibdo x3
$4F   $10  $10  $69  $3F  $09  $00  c9   -        -        -        -        -   -   -    -     -    $3F  -   -    $09 silver arrow       0:$80,$90  -                                           treasure cellar: left exit to $10, right exit to $10, Link appears at $60,$9D
$50   $A6  $3F  $53  $14  $83  $01  9    key      wall     wall     shutter  2   3   5    $13   -    $14  D   -    -                      0:$80,$90  all dead: shutters open                     $13 Zol x5
$51   $02  $E6  $97  $11  $03  $01  9    open     open     shutter  wall     2   2   6    $17   -    $11  -   -    -                      0:$80,$90  all dead: shutters open                     $17 Like-Like x6
$52   $E6  $26  $07  $9B  $23  $01  9    shutter  wall     wall     wall     2   2   3    $47   -    $1B  -   $10  -                      0:$80,$90  all dead: shutters open                     $47 Patra x1
$53   $96  $27  $FC  $A3  $03  $00  9    bomb     key      wall     wall     2   3   8    $7C   -    $23  -   -    -                      0:$80,$90  -                                           list $7C: $2B Bubble x1, $17 Like-Like x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$54   $02  $32  $7B  $8C  $03  $00  9    open     open     wall     bomb     2   2   5    $7B   -    $0C  -   -    -                      0:$80,$90  -                                           list $7B: $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$55   $92  $92  $3A  $5A  $03  $00  9    bomb     bomb     bomb     bomb     2   2   3    $3A   B    $1A  -   -    -                      0:$80,$90  -                                           $3A Lanmola x1
$56   $02  $87  $B3  $99  $19  $07  9    open     open     bomb     wall     2   3   6    $73   -    $19  -   -    $19 key                0:$80,$90  all dead: shutters open, room item appears  list $73: $2B Bubble x2, $13 Zol x2, $17 Like-Like x2
$57   $06  $27  $7B  $94  $99  $07  9    open     wall     wall     wall     2   3   5    $7B   -    $14  D   -    $19 key                0:$80,$90  all dead: shutters open, room item appears  list $7B: $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$58   $A2  $22  $31  $00  $0F  $07  7    key      open     wall     open     2   2   3    $31   -    $00  -   -    $0F 5 rupees           0:$80,$90  all dead: shutters open, room item appears  $31 Dodongo x3
$59   $12  $1E  $BA  $82  $03  $01  7    open     bomb     open     shutter  2   2   6    $7A   -    $02  -   -    -                      0:$80,$90  all dead: shutters open                     list $7A: $05 Goriya x3, $06 Goriya x3
$5A   $32  $07  $EA  $19  $96  $00  7    wall     bomb     open     wall     2   3   8    $2A   -    $19  D   -    $16 compass            0:$80,$90  -                                           $2A Stalfos x8
$5B   $22  $24  $10  $A6  $03  $00  7    wall     open     wall     wall     2   0   3    $50   -    $26  -   -    -                      0:$80,$90  -                                           $50 dungeon person x1
$5C   $26  $22  $0B  $24  $19  $07  8    wall     wall     wall     open     2   2   3    $0B   -    $24  -   -    $19 key                0:$80,$90  all dead: shutters open, room item appears  $0B Darknut red x3
$5D   $06  $03  $B0  $91  $99  $00  8    open     wall     open     open     2   3   6    $70   -    $11  D   -    $19 key                0:$80,$90  -                                           list $70: $16 Pols Voice x2, $30 Gibdo x2, $1B Keese blue x2
$5E   $F2  $F7  $4C  $24  $19  $07  8    shutter  bomb     shutter  key      2   3   5    $0C   -    $24  -   -    $19 key                0:$80,$90  all dead: shutters open, room item appears  $0C Darknut blue x5
$5F   $26  $A7  $D6  $18  $96  $00  8    wall     wall     key      wall     2   3   8    $16   -    $18  D   -    $16 compass            0:$80,$90  -                                           $16 Pols Voice x8
$60   $14  $55  $69  $3E  $03  $00  c9   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$80,$90  -                                           tunnel cellar: left exit to $14, right exit to $55, Link appears at $60,$9D
$61   $02  $22  $07  $DA  $19  $10  9    open     open     wall     open     2   2   3    $47   B    $1A  -   -    $19 key                1:$D0,$60  -                                           $47 Patra x1
$62   $26  $16  $DB  $26  $0F  $07  9    wall     wall     open     key      2   2   8    $1B   -    $26  -   -    $0F 5 rupees           0:$80,$90  all dead: shutters open, room item appears  $1B Keese blue x8
$63   $B6  $A6  $53  $1B  $03  $00  9    key      key      key      wall     2   2   5    $13   -    $1B  -   -    -                      0:$80,$90  -                                           $13 Zol x5
$64   $06  $22  $FC  $9F  $03  $00  9    open     wall     wall     open     2   2   8    $7C   -    $1F  -   -    -                      0:$80,$90  -                                           list $7C: $2B Bubble x1, $17 Like-Like x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$65   $86  $03  $F1  $98  $83  $00  9    bomb     wall     open     open     2   3   8    $71   -    $18  D   -    -                      0:$80,$90  -                                           list $71: $2B Bubble x3, $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$66   $E2  $E4  $0B  $A6  $03  $00  9    shutter  open     shutter  wall     2   0   3    $4B   -    $26  -   -    -                      0:$80,$90  -                                           $4B dungeon person x1
$67   $30  $04  $C6  $3E  $03  $00  c9   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$80,$90  -                                           tunnel cellar: left exit to $30, right exit to $04, Link appears at $C0,$6D
$68   $02  $32  $EE  $80  $00  $07  7    open     open     wall     bomb     2   2   8    $6E   -    $00  -   -    $00 bomb               0:$80,$90  all dead: shutters open, room item appears  list $6E: $4A Trap x1, $00 (empty) x3, $1B Keese blue x4
$69   $82  $83  $45  $23  $00  $07  7    bomb     open     bomb     open     2   3   5    $05   -    $23  -   -    $00 bomb               0:$80,$90  all dead: shutters open, room item appears  $05 Goriya x5
$6A   $82  $03  $DB  $09  $83  $02  7    bomb     open     open     open     2   3   8    $1B   -    $09  D   -    -                      0:$80,$90  ringleader: first monster dead kills all    $1B Keese blue x8
$6B   $E6  $02  $85  $1E  $03  $01  7    shutter  wall     open     open     2   2   6    $05   -    $1E  -   -    -                      0:$80,$90  all dead: shutters open                     $05 Goriya x6
$6C   $26  $02  $39  $24  $00  $07  7    wall     wall     open     open     2   2   3    $39   -    $24  -   -    $00 bomb               0:$80,$90  all dead: shutters open, room item appears  $39 Digdogger x1
$6D   $26  $07  $EA  $13  $19  $00  7    wall     wall     open     wall     2   3   8    $2A   -    $13  -   -    $19 key                0:$80,$90  -                                           $2A Stalfos x8
$6E   $82  $26  $3C  $25  $0F  $07  8    bomb     open     wall     wall     2   2   3    $3C   -    $25  -   -    $0F 5 rupees           0:$80,$90  all dead: shutters open, room item appears  $3C Manhandla x1
$6F   $7C  $7C  $69  $3F  $11  $00  c8   -        -        -        -        -   -   -    -     -    $3F  -   -    $11 book               0:$80,$90  -                                           treasure cellar: left exit to $7C, right exit to $7C, Link appears at $60,$9D
$70   $63  $05  $A9  $3E  $03  $00  c9   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$80,$90  -                                           tunnel cellar: left exit to $63, right exit to $05, Link appears at $A0,$9D
$71   $06  $26  $00  $1B  $03  $00  9    open     wall     wall     wall     2   2   3    $00   -    $1B  -   -    -                      0:$80,$90  -                                           none
$72   $71  $74  $69  $3E  $03  $00  c9   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$80,$90  -                                           tunnel cellar: left exit to $71, right exit to $74, Link appears at $60,$9D
$73   $A6  $36  $F1  $9F  $03  $00  9    key      wall     wall     key      2   2   8    $71   -    $1F  -   -    -                      0:$80,$90  -                                           list $71: $2B Bubble x3, $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$74   $26  $A6  $3B  $5A  $03  $00  9    wall     wall     key      wall     2   2   3    $3B   B    $1A  -   -    -                      0:$80,$90  -                                           $3B Lanmola x1
$75   $20  $61  $69  $3E  $03  $00  c9   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$80,$90  -                                           tunnel cellar: left exit to $20, right exit to $61, Link appears at $60,$9D
$76   $02  $26  $00  $21  $03  $00  9    open     open     wall     wall     2   2   3    $00   -    $21  -   -    -                      0:$80,$90  -                                           none
$77   $52  $03  $69  $3E  $03  $00  c9   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$80,$90  -                                           tunnel cellar: left exit to $52, right exit to $03, Link appears at $60,$9D
$78   $06  $26  $68  $23  $19  $00  7    open     wall     wall     wall     2   2   5    $28   -    $23  -   -    $19 key                0:$80,$90  -                                           $28 Rope x5
$79   $02  $23  $00  $21  $03  $00  7    open     open     wall     open     2   3   3    $00   -    $21  -   -    -                      0:$80,$90  -                                           none
$7A   $06  $06  $01  $9D  $00  $07  7    open     wall     open     wall     2   2   3    $41   -    $1D  -   -    $00 bomb               0:$80,$90  all dead: shutters open, room item appears  $41 Moldorm x1
$7B   $29  $0D  $69  $3E  $03  $00  c7   -        -        -        -        -   -   -    -     -    $3E  -   -    -                      0:$80,$90  -                                           tunnel cellar: left exit to $29, right exit to $0D, Link appears at $60,$9D
$7C   $26  $22  $F4  $DA  $03  $10  8    wall     wall     wall     open     2   2   8    $74   B    $1A  -   -    -                      1:$D0,$60  -                                           list $74: $2B Bubble x2, $0C Darknut blue x1, $0B Darknut red x2, $30 Gibdo x3
$7D   $26  $E3  $3C  $23  $0F  $07  8    wall     wall     shutter  open     2   3   3    $3C   -    $23  -   -    $0F 5 rupees           0:$80,$90  all dead: shutters open, room item appears  $3C Manhandla x1
$7E   $02  $03  $00  $21  $03  $00  8    open     open     open     open     2   3   3    $00   -    $21  -   -    -                      0:$80,$90  -                                           none
$7F   $26  $07  $EF  $96  $99  $00  8    wall     wall     open     wall     2   3   8    $6F   -    $16  D   -    $19 key                0:$80,$90  -                                           list $6F: $2B Bubble x3, $13 Zol x2, $1B Keese blue x3
```
