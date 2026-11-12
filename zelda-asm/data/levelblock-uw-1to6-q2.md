# Underworld level block, levels 1-6, Q2

How to read this: one row per room `$00`-`$7F` of [`LevelBlockUW1Q2`](../src/bank6/level-blocks-common-data.asm#L78) (`dat/LevelBlockUW1Q2.dat`), the level block that levels 1, 2, 3, 4, 5, 6 share in
the second quest. Each row gives the six raw attribute bytes A-F, then the fields the game reads from
them. The formats are in `README.md`. Find a room with `grep '^\$32 '`.

Columns:

- `lvl`: the level whose submenu map shows the room; `cN` = a cellar listed by level N; `-` = neither
  (the room is drawn on no level's map). Levels are internal level numbers ([`CurLevel`](../src/Variables.inc#L12)). [`LevelInfo_SubmenuMapMask`](../src/Variables.inc#L676), [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L673)
- `N`, `S`, `W`, `E`: door type per side (A bits 7-5, A bits 4-2, B bits 7-5, B bits 4-2): `open`,
  `wall`, `false`/`false2` (walk-through wall), `bomb` (bombable wall), `key`/`key2` (key door),
  `shutter`. [`FindDoorTypeByDoorBit`](../src/bank5/fill-walls.asm#L55), [`TouchDoor_JumpTable`](../src/bank5/touch-door.asm#L62)
- `po`, `pi`: outer and inner palette selectors (A bits 1-0, B bits 1-0). [`FillPlayAreaAttrs`](../src/bank5/mode7-scroll-init.asm#L379)
- `cnt`: monster count selected by C bits 7-6 from [`LevelInfo_FoeCounts`](../src/Variables.inc#L663); `list`: monster list id (C bits 5-0, D bit 7).
  The `monsters` text applies the rules of `objlists.md`. [`InitMode_EnterRoom`](../src/bank5/enter-room.asm#L137)
- `blk`: `B` = the room has a push block (D bit 6). [`CreateRoomObjects`](../src/bank5/create-room-objects.asm#L51)
- `lay`: room layout id (D bits 5-0), see `room-layouts.md`; `$3E`/`$3F` = cellar. [`LayoutUWFloor`](../src/bank5/uw-floor-push-block.asm#L66)
- `dk`: `D` = dark room (E bit 7). [`IsDarkRoom_Bank5`](../src/bank5/door-flags-dark-room.asm#L124)
- `snd`: boss sound request heard in the room (E bits 6-5 through [`BossSoundEffects`](../src/bank5/boss-sound-door-face.asm#L30)). [`CheckBossSoundEffectUW`](../src/bank5/boss-sound-door-face.asm#L42)
- `item`: room item (E bits 4-0; `$03` = none, shown as `-`). [`CreateRoomObjects`](../src/bank5/create-room-objects.asm#L51)
- `pos`: position index (F bits 5-4) and the item's X,Y from the owning level's [`LevelInfo_ShortcutOrItemPosArray`](../src/Variables.inc#L665) (`?` when the room
  has no `lvl`). [`GetShortcutOrItemXYForRoom`](../src/bank1/bomb-candle-wield.asm#L280)
- `secret`: secret trigger (F bits 2-0). [`CheckSecretTrigger_JumpTable`](../src/bank5/underworld-secrets.asm#L105)
- `monsters; cellar`: the monsters made on entry (before the kill count is subtracted). For a cellar:
  the rooms its left and right exits lead to (bytes A and B) and where Link appears there (byte C);
  its door columns are `-` and its monster list is not used. [`CheckSubroom`](../src/bank5/cave-cellar-exits.asm#L43), [`InitMode_EnterRoom`](../src/bank5/enter-room.asm#L137)

Levels of this block:

```
level  shown as  start  Triforce  boss  cellars      counts           item positions
1      LEVEL-1   $77    $08       $07   $28          $03 $05 $06 $08  $C0,$90 $A0,$C0 $80,$90 $B0,$70
2      LEVEL-3   $75    $20       $30   $56          $03 $05 $06 $08  $C0,$90 $A0,$C0 $80,$90 $80,$70
3      LEVEL-2   $79    $1B       $2B   $09,$0B      $03 $05 $06 $08  $C0,$90 $A0,$C0 $80,$90 $30,$70
4      LEVEL-5   $72    $00       $10   $21,$58,$7A  $03 $05 $06 $08  $C0,$90 $A0,$C0 $80,$90 $80,$60
5      LEVEL-4   $7D    $4F       $5F   $0F,$6A,$7F  $03 $05 $06 $08  $C0,$90 $A0,$C0 $80,$90 $80,$70
6      LEVEL-6   $74    $16       $26   $03,$73,$46  $03 $05 $06 $08  $40,$90 $70,$90 $80,$90 $50,$60
```

# Rooms

```
room  A    B    C    D    E    F    lvl  N        S        W        E        po  pi  cnt  list  blk  lay  dk  snd  item                  pos        secret                                      monsters; cellar
$00   $22  $26  $00  $29  $1B  $20  4    wall     open     wall     wall     2   2   3    $00   -    $29  -   -    $1B Triforce piece    2:$80,$90  -                                           none
$01   $26  $3E  $63  $5C  $03  $04  4    wall     wall     wall     shutter  2   2   5    $23   B    $1C  -   -    -                     0:$C0,$90  push block: shutters open                   $23 Blue Wizzrobe x5
$02   $22  $06  $33  $06  $19  $17  4    wall     open     open     wall     2   2   3    $33   -    $06  -   -    $19 key               1:$A0,$C0  all dead: shutters open, room item appears  $33 Gohma x1
$03   $04  $24  $69  $3E  $03  $00  c6   -        -        -        -        -   -   -    -     -    $3E  -   -    -                     0:$40,$90  -                                           tunnel cellar: left exit to $04, right exit to $24, Link appears at $60,$9D
$04   $32  $36  $DB  $1C  $03  $00  6    wall     bomb     wall     key      2   2   8    $1B   -    $1C  -   -    -                     0:$40,$90  -                                           $1B Keese blue x8
$05   $26  $BE  $03  $85  $0F  $07  6    wall     wall     key      shutter  2   2   3    $43   -    $05  -   -    $0F 5 rupees          0:$40,$90  all dead: shutters open, room item appears  $43 Gleeok x1
$06   $26  $06  $6D  $46  $03  $05  6    wall     wall     open     wall     2   2   5    $2D   B    $06  -   -    -                     0:$40,$90  push block: stairs appear                   $2D Bubble x5
$07   $32  $3E  $3D  $04  $1A  $07  1    wall     bomb     wall     shutter  2   2   3    $3D   -    $04  -   -    $1A heart container   0:$C0,$90  all dead: shutters open, room item appears  $3D Aquamentus x1
$08   $26  $06  $00  $29  $1B  $20  1    wall     wall     open     wall     2   2   3    $00   -    $29  -   -    $1B Triforce piece    2:$80,$90  -                                           none
$09   $3B  $7B  $68  $3E  $03  $00  c3   -        -        -        -        -   -   -    -     -    $3E  -   -    -                     0:$C0,$90  -                                           tunnel cellar: left exit to $3B, right exit to $7B, Link appears at $60,$8D
$0A   $36  $27  $AD  $A3  $17  $27  3    wall     key      wall     wall     2   3   6    $6D   -    $23  -   -    $17 map               2:$80,$90  all dead: shutters open, room item appears  list $6D: $4A Trap x1, $00 (empty) x3, $13 Zol x2
$0B   $3A  $3A  $68  $3F  $05  $20  c3   -        -        -        -        -   -   -    -     -    $3F  -   -    $05 flute             2:$80,$90  -                                           treasure cellar: left exit to $3A, right exit to $3A, Link appears at $60,$8D
$0C   $2A  $28  $12  $A6  $03  $00  5    wall     false    wall     false    2   0   3    $52   -    $26  -   -    -                     0:$C0,$90  -                                           $52 dungeon person x1
$0D   $32  $5C  $11  $A6  $03  $06  5    wall     bomb     false    shutter  2   0   3    $51   -    $26  -   -    -                     0:$C0,$90  life-or-money person gone: shutters open    $51 life-or-money person x1
$0E   $26  $06  $2C  $5F  $03  $05  5    wall     wall     open     wall     2   2   3    $2C   B    $1F  -   -    -                     0:$C0,$90  push block: stairs appear                   $2C Bubble x3
$0F   $0E  $6F  $6B  $3E  $03  $00  c5   -        -        -        -        -   -   -    -     -    $3E  -   -    -                     0:$C0,$90  -                                           tunnel cellar: left exit to $0E, right exit to $6F, Link appears at $60,$BD
$10   $E6  $36  $04  $85  $1A  $07  4    shutter  wall     wall     key      2   2   3    $44   -    $05  -   -    $1A heart container   0:$C0,$90  all dead: shutters open, room item appears  $44 Gleeok x1
$11   $26  $A6  $52  $1B  $23  $02  4    wall     wall     key      wall     2   2   5    $12   -    $1B  -   $10  -                     0:$C0,$90  ringleader: first monster dead kills all    $12 Vire x5
$12   $06  $26  $2D  $1B  $03  $00  4    open     wall     wall     wall     2   2   3    $2D   -    $1B  -   -    -                     0:$C0,$90  -                                           $2D Bubble x3
$13   $22  $3F  $7B  $A4  $00  $27  6    wall     open     wall     shutter  2   3   5    $7B   -    $24  -   -    $00 bomb              2:$80,$90  all dead: shutters open, room item appears  list $7B: $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$14   $82  $E7  $7B  $8B  $19  $31  6    bomb     open     shutter  wall     2   3   5    $7B   -    $0B  -   -    $19 key               3:$50,$60  all dead: shutters open                     list $7B: $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$15   $36  $26  $86  $03  $1E  $37  2    wall     key      wall     wall     2   2   6    $06   -    $03  -   -    $1E magic boomerang   3:$80,$70  all dead: shutters open, room item appears  $06 Goriya x6
$16   $22  $26  $00  $29  $1B  $20  6    wall     open     wall     wall     2   2   3    $00   -    $29  -   -    $1B Triforce piece    2:$80,$90  -                                           none
$17   $86  $37  $05  $24  $23  $00  1    bomb     wall     wall     key      2   3   3    $05   -    $24  -   $10  -                     0:$C0,$90  -                                           $05 Goriya x3
$18   $26  $A6  $F9  $9B  $19  $27  1    wall     wall     key      wall     2   2   8    $79   -    $1B  -   -    $19 key               2:$80,$90  all dead: shutters open, room item appears  list $79: $2B Bubble x3, $27 Wallmaster x5
$19   $22  $22  $3C  $00  $19  $27  3    wall     open     wall     open     2   2   3    $3C   -    $00  -   -    $19 key               2:$80,$90  all dead: shutters open, room item appears  $3C Manhandla x1
$1A   $A2  $06  $4B  $02  $03  $02  3    key      open     open     wall     2   2   5    $0B   -    $02  -   -    -                     0:$C0,$90  ringleader: first monster dead kills all    $0B Darknut red x5
$1B   $22  $26  $00  $29  $1B  $20  3    wall     open     wall     wall     2   2   3    $00   -    $29  -   -    $1B Triforce piece    2:$80,$90  -                                           none
$1C   $52  $2B  $FE  $95  $83  $00  5    false    bomb     wall     false    2   3   8    $7E   -    $15  D   -    -                     0:$C0,$90  -                                           list $7E: $2D Bubble x3, $2C Bubble x1, $0C Darknut blue x2, $0B Darknut red x2
$1D   $82  $47  $DB  $13  $99  $07  5    bomb     open     false    wall     2   3   8    $1B   -    $13  D   -    $19 key               0:$C0,$90  all dead: shutters open, room item appears  $1B Keese blue x8
$1E   $22  $27  $8C  $23  $17  $27  5    wall     open     wall     wall     2   3   6    $0C   -    $23  -   -    $17 map               2:$80,$90  all dead: shutters open, room item appears  $0C Darknut blue x6
$1F   $22  $06  $B8  $CC  $03  $05  5    wall     open     open     wall     2   2   6    $78   B    $0C  -   -    -                     0:$C0,$90  push block: stairs appear                   list $78: $16 Pols Voice x2, $0C Darknut blue x2, $0B Darknut red x2
$20   $22  $26  $00  $29  $1B  $20  2    wall     open     wall     wall     2   2   3    $00   -    $29  -   -    $1B Triforce piece    2:$80,$90  -                                           none
$21   $01  $71  $68  $3E  $03  $00  c4   -        -        -        -        -   -   -    -     -    $3E  -   -    -                     0:$C0,$90  -                                           tunnel cellar: left exit to $01, right exit to $71, Link appears at $60,$8D
$22   $22  $26  $DD  $1C  $03  $02  4    wall     open     wall     wall     2   2   8    $1D   -    $1C  -   -    -                     0:$C0,$90  ringleader: first monster dead kills all    $1D Keese black x8
$23   $02  $2B  $52  $19  $03  $00  6    open     open     wall     false    2   3   5    $12   -    $19  -   -    -                     0:$40,$90  -                                           $12 Vire x5
$24   $02  $46  $2C  $47  $03  $05  6    open     open     false    wall     2   2   3    $2C   B    $07  -   -    -                     0:$40,$90  push block: stairs appear                   $2C Bubble x3
$25   $A2  $24  $36  $26  $03  $00  2    key      open     wall     wall     2   0   3    $36   -    $26  -   -    -                     0:$C0,$90  -                                           $36 Grumble x1
$26   $FE  $27  $33  $24  $1A  $37  6    shutter  shutter  wall     wall     2   3   3    $33   -    $24  -   -    $1A heart container   3:$50,$60  all dead: shutters open, room item appears  $33 Gohma x1
$27   $3E  $26  $01  $9D  $03  $01  1    wall     shutter  wall     wall     2   2   3    $41   -    $1D  -   -    -                     0:$C0,$90  all dead: shutters open                     $41 Moldorm x1
$28   $18  $68  $6A  $3E  $03  $00  c1   -        -        -        -        -   -   -    -     -    $3E  -   -    -                     0:$C0,$90  -                                           tunnel cellar: left exit to $18, right exit to $68, Link appears at $60,$AD
$29   $E2  $33  $DB  $23  $03  $01  3    shutter  open     wall     bomb     2   3   8    $1B   -    $23  -   -    -                     0:$C0,$90  all dead: shutters open                     $1B Keese blue x8
$2A   $0A  $86  $DB  $0E  $39  $30  3    open     false    bomb     wall     2   2   8    $1B   -    $0E  -   $10  $19 key               3:$30,$70  -                                           $1B Keese blue x8
$2B   $F6  $26  $03  $85  $1A  $07  3    shutter  key      wall     wall     2   2   3    $43   -    $05  -   -    $1A heart container   0:$C0,$90  all dead: shutters open, room item appears  $43 Gleeok x1
$2C   $82  $23  $53  $17  $80  $27  5    bomb     open     wall     open     2   3   5    $13   -    $17  D   -    $00 bomb              2:$80,$90  all dead: shutters open, room item appears  $13 Zol x5
$2D   $02  $12  $00  $60  $03  $00  5    open     open     open     bomb     2   2   3    $00   B    $20  -   -    -                     0:$C0,$90  -                                           none
$2E   $E6  $87  $32  $24  $03  $01  5    shutter  wall     bomb     wall     2   3   3    $32   -    $24  -   -    -                     0:$C0,$90  all dead: shutters open                     $32 Dodongo x1
$2F   $FE  $27  $31  $24  $03  $01  5    shutter  shutter  wall     wall     2   3   3    $31   -    $24  -   -    -                     0:$C0,$90  all dead: shutters open                     $31 Dodongo x3
$30   $E6  $26  $86  $1B  $19  $01  2    shutter  wall     wall     wall     2   2   6    $06   -    $1B  -   -    $19 key               0:$C0,$90  all dead: shutters open                     $06 Goriya x6
$31   $22  $23  $FC  $98  $19  $30  4    wall     open     wall     open     2   3   8    $7C   -    $18  -   -    $19 key               3:$80,$60  -                                           list $7C: $2B Bubble x1, $17 Like-Like x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$32   $E6  $05  $3C  $25  $0F  $07  4    shutter  wall     open     wall     2   1   3    $3C   -    $25  -   -    $0F 5 rupees          0:$C0,$90  all dead: shutters open, room item appears  $3C Manhandla x1
$33   $0A  $33  $FD  $93  $03  $00  6    open     false    wall     bomb     2   3   8    $7D   -    $13  -   -    -                     0:$40,$90  -                                           list $7D: $2D Bubble x3, $2C Bubble x1, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$34   $12  $86  $DB  $0E  $16  $00  6    open     bomb     bomb     wall     2   2   8    $1B   -    $0E  -   -    $16 compass           0:$40,$90  -                                           $1B Keese blue x8
$35   $02  $27  $AA  $23  $19  $27  2    open     open     wall     wall     2   3   6    $2A   -    $23  -   -    $19 key               2:$80,$90  all dead: shutters open, room item appears  $2A Stalfos x6
$36   $06  $26  $3C  $4A  $43  $05  6    open     wall     wall     wall     2   2   3    $3C   B    $0A  -   $20  -                     0:$40,$90  push block: stairs appear                   $3C Manhandla x1
$37   $02  $36  $D5  $1E  $0F  $07  1    open     open     wall     key      2   2   8    $15   -    $1E  -   -    $0F 5 rupees          0:$C0,$90  all dead: shutters open, room item appears  $15 Gel x8
$38   $3E  $A6  $32  $25  $03  $01  1    wall     shutter  key      wall     2   2   3    $32   -    $25  -   -    -                     0:$C0,$90  all dead: shutters open                     $32 Dodongo x1
$39   $1E  $26  $70  $07  $03  $01  3    open     shutter  wall     wall     2   2   5    $30   -    $07  -   -    -                     0:$C0,$90  all dead: shutters open                     $30 Gibdo x5
$3A   $46  $46  $00  $1C  $03  $00  -    false    wall     false    wall     2   2   3    $00   -    $1C  -   -    -                     0:?        -                                           none
$3B   $A6  $26  $0B  $1B  $20  $27  3    key      wall     wall     wall     2   2   3    $0B   -    $1B  -   $10  $00 bomb              2:$80,$90  all dead: shutters open, room item appears  $0B Darknut red x3
$3C   $1E  $23  $AD  $95  $03  $00  5    open     shutter  wall     open     2   3   6    $6D   -    $15  -   -    -                     0:$C0,$90  -                                           list $6D: $4A Trap x1, $00 (empty) x3, $13 Zol x2
$3D   $06  $12  $B0  $83  $19  $27  5    open     wall     open     bomb     2   2   6    $70   -    $03  -   -    $19 key               2:$80,$90  all dead: shutters open, room item appears  list $70: $16 Pols Voice x2, $30 Gibdo x2, $1B Keese blue x2
$3E   $26  $86  $35  $26  $03  $06  -    wall     wall     bomb     wall     2   2   3    $35   -    $26  -   -    -                     0:?        life-or-money person gone: shutters open    $35 Rupee stash x1
$3F   $EA  $27  $0B  $23  $03  $01  5    shutter  false    wall     wall     2   3   3    $0B   -    $23  -   -    -                     0:$C0,$90  all dead: shutters open                     $0B Darknut red x3
$40   $3E  $22  $F7  $8D  $03  $01  4    wall     shutter  wall     open     2   2   8    $77   -    $0D  -   -    -                     0:$C0,$90  all dead: shutters open                     list $77: $4A Trap x1, $00 (empty) x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$41   $06  $E6  $53  $26  $16  $27  4    open     wall     shutter  wall     2   2   5    $13   -    $26  -   -    $16 compass           2:$80,$90  all dead: shutters open, room item appears  $13 Zol x5
$42   $36  $30  $12  $A6  $03  $00  6    wall     key      wall     bomb     2   0   3    $52   -    $26  -   -    -                     0:$40,$90  -                                           $52 dungeon person x1
$43   $46  $83  $F6  $94  $17  $17  6    false    wall     bomb     open     2   3   8    $76   -    $14  -   -    $17 map               1:$70,$90  all dead: shutters open, room item appears  list $76: $4A Trap x1, $00 (empty) x3, $17 Like-Like x4
$44   $92  $07  $F1  $98  $03  $00  6    bomb     bomb     open     wall     2   3   8    $71   -    $18  -   -    -                     0:$40,$90  -                                           list $71: $2B Bubble x3, $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$45   $E2  $27  $45  $24  $17  $21  2    shutter  open     wall     wall     2   3   5    $05   -    $24  -   -    $17 map               2:$80,$90  all dead: shutters open                     $05 Goriya x5
$46   $51  $51  $69  $3F  $0D  $20  c6   -        -        -        -        -   -   -    -     -    $3F  -   -    $0D ladder            2:$80,$90  -                                           treasure cellar: left exit to $51, right exit to $51, Link appears at $60,$9D
$47   $02  $23  $05  $15  $00  $17  1    open     open     wall     open     2   3   3    $05   -    $15  -   -    $00 bomb              1:$A0,$C0  all dead: shutters open, room item appears  $05 Goriya x3
$48   $E6  $07  $EE  $98  $17  $31  1    shutter  wall     open     wall     2   3   8    $6E   -    $18  -   -    $17 map               3:$B0,$70  all dead: shutters open                     list $6E: $4A Trap x1, $00 (empty) x3, $1B Keese blue x4
$49   $02  $37  $DB  $23  $19  $20  3    open     open     wall     key      2   3   8    $1B   -    $23  -   -    $19 key               2:$80,$90  -                                           $1B Keese blue x8
$4A   $32  $A2  $B0  $8D  $19  $17  3    wall     bomb     key      open     2   2   6    $70   -    $0D  -   -    $19 key               1:$A0,$C0  all dead: shutters open, room item appears  list $70: $16 Pols Voice x2, $30 Gibdo x2, $1B Keese blue x2
$4B   $22  $06  $4B  $0A  $00  $27  3    wall     open     open     wall     2   2   5    $0B   -    $0A  -   -    $00 bomb              2:$80,$90  all dead: shutters open, room item appears  $0B Darknut red x5
$4C   $1E  $36  $8C  $06  $19  $27  5    open     shutter  wall     key      2   2   6    $0C   -    $06  -   -    $19 key               2:$80,$90  all dead: shutters open, room item appears  $0C Darknut blue x6
$4D   $26  $AA  $4B  $0A  $00  $27  5    wall     wall     key      false    2   2   5    $0B   -    $0A  -   -    $00 bomb              2:$80,$90  all dead: shutters open, room item appears  $0B Darknut red x5
$4E   $26  $44  $0F  $A6  $03  $00  -    wall     wall     false    wall     2   0   3    $4F   -    $26  -   -    -                     0:?        -                                           $4F dungeon person x1
$4F   $42  $46  $00  $29  $1B  $20  5    false    open     false    wall     2   2   3    $00   -    $29  -   -    $1B Triforce piece    2:$80,$90  -                                           none
$50   $1E  $27  $FD  $95  $83  $01  4    open     shutter  wall     wall     2   3   8    $7D   -    $15  D   -    -                     0:$C0,$90  all dead: shutters open                     list $7D: $2D Bubble x3, $2C Bubble x1, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$51   $26  $22  $09  $C1  $0F  $35  6    wall     wall     wall     open     2   2   3    $49   B    $01  -   -    $0F 5 rupees          3:$50,$60  push block: stairs appear                   $49 Trap x1
$52   $A6  $17  $FC  $A3  $0F  $27  6    key      wall     open     key      2   3   8    $7C   -    $23  -   -    $0F 5 rupees          2:$80,$90  all dead: shutters open, room item appears  list $7C: $2B Bubble x1, $17 Like-Like x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$53   $22  $A3  $72  $95  $03  $00  6    wall     open     key      open     2   3   5    $72   -    $15  -   -    -                     0:$40,$90  -                                           list $72: $2B Bubble x2, $12 Vire x3
$54   $82  $07  $FC  $A4  $03  $00  6    bomb     open     open     wall     2   3   8    $7C   -    $24  -   -    -                     0:$40,$90  -                                           list $7C: $2B Bubble x1, $17 Like-Like x3, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$55   $02  $26  $A8  $11  $00  $27  2    open     open     wall     wall     2   2   6    $28   -    $11  -   -    $00 bomb              2:$80,$90  all dead: shutters open, room item appears  $28 Rope x6
$56   $30  $76  $69  $3E  $03  $00  c2   -        -        -        -        -   -   -    -     -    $3E  -   -    -                     0:$C0,$90  -                                           tunnel cellar: left exit to $30, right exit to $76, Link appears at $60,$9D
$57   $02  $26  $6A  $1F  $16  $17  1    open     open     wall     wall     2   2   5    $2A   -    $1F  -   -    $16 compass           1:$A0,$C0  all dead: shutters open, room item appears  $2A Stalfos x5
$58   $11  $22  $68  $3E  $03  $00  c4   -        -        -        -        -   -   -    -     -    $3E  -   -    -                     0:$C0,$90  -                                           tunnel cellar: left exit to $11, right exit to $22, Link appears at $60,$8D
$59   $1E  $36  $13  $4A  $03  $04  3    open     shutter  wall     key      2   2   3    $13   B    $0A  -   -    -                     0:$C0,$90  push block: shutters open                   $13 Zol x3
$5A   $86  $B2  $0B  $1E  $16  $17  3    bomb     wall     key      bomb     2   2   3    $0B   -    $1E  -   -    $16 compass           1:$A0,$C0  all dead: shutters open, room item appears  $0B Darknut red x3
$5B   $02  $87  $EE  $A3  $19  $27  3    open     open     bomb     wall     2   3   8    $6E   -    $23  -   -    $19 key               2:$80,$90  all dead: shutters open, room item appears  list $6E: $4A Trap x1, $00 (empty) x3, $1B Keese blue x4
$5C   $E2  $37  $FE  $9F  $83  $01  5    shutter  open     wall     key      2   3   8    $7E   -    $1F  D   -    -                     0:$C0,$90  all dead: shutters open                     list $7E: $2D Bubble x3, $2C Bubble x1, $0C Darknut blue x2, $0B Darknut red x2
$5D   $26  $BE  $3D  $04  $03  $01  5    wall     wall     key      shutter  2   2   3    $3D   -    $04  -   -    -                     0:$C0,$90  all dead: shutters open                     $3D Aquamentus x1
$5E   $46  $06  $B0  $CD  $63  $05  5    false    wall     open     wall     2   2   6    $70   B    $0D  -   $40  -                     0:$C0,$90  push block: stairs appear                   list $70: $16 Pols Voice x2, $30 Gibdo x2, $1B Keese blue x2
$5F   $F6  $27  $38  $24  $1A  $07  5    shutter  key      wall     wall     2   3   3    $38   -    $24  -   -    $1A heart container   0:$C0,$90  all dead: shutters open, room item appears  $38 Digdogger x1
$60   $06  $23  $72  $97  $97  $27  4    open     wall     wall     open     2   3   5    $72   -    $17  D   -    $17 map               2:$80,$90  all dead: shutters open, room item appears  list $72: $2B Bubble x2, $12 Vire x3
$61   $36  $02  $F6  $9D  $0F  $07  4    wall     key      open     open     2   2   8    $76   -    $1D  -   -    $0F 5 rupees          0:$C0,$90  all dead: shutters open, room item appears  list $76: $4A Trap x1, $00 (empty) x3, $17 Like-Like x4
$62   $22  $07  $AD  $A3  $83  $00  4    wall     open     open     wall     2   3   6    $6D   -    $23  D   -    -                     0:$C0,$90  -                                           list $6D: $4A Trap x1, $00 (empty) x3, $13 Zol x2
$63   $06  $23  $FD  $96  $99  $10  6    open     wall     wall     open     2   3   8    $7D   -    $16  D   -    $19 key               1:$70,$90  -                                           list $7D: $2D Bubble x3, $2C Bubble x1, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$64   $02  $07  $7B  $8C  $80  $27  6    open     open     open     wall     2   3   5    $7B   -    $0C  D   -    $00 bomb              2:$80,$90  all dead: shutters open, room item appears  list $7B: $23 Blue Wizzrobe x3, $24 Red Wizzrobe x2
$65   $02  $36  $86  $0D  $03  $00  2    open     open     wall     key      2   2   6    $06   -    $0D  -   -    -                     0:$C0,$90  -                                           $06 Goriya x6
$66   $36  $A6  $31  $25  $1A  $17  2    wall     key      key      wall     2   2   3    $31   -    $25  -   -    $1A heart container   1:$A0,$C0  all dead: shutters open, room item appears  $31 Dodongo x3
$67   $02  $32  $06  $23  $19  $27  1    open     open     wall     bomb     2   2   3    $06   -    $23  -   -    $19 key               2:$80,$90  all dead: shutters open, room item appears  $06 Goriya x3
$68   $32  $86  $9B  $5A  $03  $00  1    wall     bomb     bomb     wall     2   2   6    $1B   B    $1A  -   -    -                     0:$C0,$90  -                                           $1B Keese blue x6
$69   $02  $26  $70  $10  $19  $17  3    open     open     wall     wall     2   2   5    $30   -    $10  -   -    $19 key               1:$A0,$C0  all dead: shutters open, room item appears  $30 Gibdo x5
$6A   $1F  $1F  $6A  $3F  $0C  $20  c5   -        -        -        -        -   -   -    -     -    $3F  -   -    $0C raft              2:$80,$90  -                                           treasure cellar: left exit to $1F, right exit to $1F, Link appears at $60,$AD
$6B   $FE  $26  $13  $11  $19  $17  3    shutter  shutter  wall     wall     2   2   3    $13   -    $11  -   -    $19 key               1:$A0,$C0  all dead: shutters open, room item appears  $13 Zol x3
$6C   $06  $22  $B8  $9D  $16  $27  5    open     wall     wall     open     2   2   6    $78   -    $1D  -   -    $16 compass           2:$80,$90  all dead: shutters open, room item appears  list $78: $16 Pols Voice x2, $0C Darknut blue x2, $0B Darknut red x2
$6D   $22  $E3  $38  $23  $03  $01  5    wall     open     shutter  open     2   3   3    $38   -    $23  -   -    -                     0:$C0,$90  all dead: shutters open                     $38 Digdogger x1
$6E   $22  $06  $6C  $1F  $03  $00  5    wall     open     open     wall     2   2   5    $2C   -    $1F  -   -    -                     0:$C0,$90  -                                           $2C Bubble x5
$6F   $A6  $26  $2D  $1B  $63  $00  5    key      wall     wall     wall     2   2   3    $2D   -    $1B  -   $40  -                     0:$C0,$90  -                                           $2D Bubble x3
$70   $26  $2B  $DD  $00  $99  $27  4    wall     wall     wall     false    2   3   8    $1D   -    $00  D   -    $19 key               2:$80,$90  all dead: shutters open, room item appears  $1D Keese black x8
$71   $A6  $46  $FD  $9B  $03  $00  4    key      wall     false    wall     2   2   8    $7D   -    $1B  -   -    -                     0:$C0,$90  -                                           list $7D: $2D Bubble x3, $2C Bubble x1, $23 Blue Wizzrobe x2, $24 Red Wizzrobe x2
$72   $02  $27  $00  $21  $03  $00  4    open     open     wall     wall     2   3   3    $00   -    $21  -   -    -                     0:$C0,$90  -                                           none
$73   $06  $36  $6A  $3E  $03  $00  c6   -        -        -        -        -   -   -    -     -    $3E  -   -    -                     0:$40,$90  -                                           tunnel cellar: left exit to $06, right exit to $36, Link appears at $60,$AD
$74   $02  $27  $00  $21  $03  $00  6    open     open     wall     wall     2   3   3    $00   -    $21  -   -    -                     0:$40,$90  -                                           none
$75   $02  $27  $C0  $21  $03  $00  2    open     open     wall     wall     2   3   8    $00   -    $21  -   -    -                     0:$C0,$90  -                                           none
$76   $A6  $26  $E7  $48  $16  $25  2    key      wall     wall     wall     2   2   8    $27   B    $08  -   -    $16 compass           2:$80,$90  push block: stairs appear                   $27 Wallmaster x8
$77   $02  $23  $00  $21  $03  $00  1    open     open     wall     open     2   3   3    $00   -    $21  -   -    -                     0:$C0,$90  -                                           none
$78   $86  $06  $46  $02  $1D  $27  1    bomb     wall     open     wall     2   2   5    $06   -    $02  -   -    $1D wooden boomerang  2:$80,$90  all dead: shutters open, room item appears  $06 Goriya x5
$79   $02  $27  $00  $21  $03  $00  3    open     open     wall     wall     2   3   3    $00   -    $21  -   -    -                     0:$C0,$90  -                                           none
$7A   $12  $12  $69  $3F  $0A  $20  c4   -        -        -        -        -   -   -    -     -    $3F  -   -    $0A bow               2:$80,$90  -                                           treasure cellar: left exit to $12, right exit to $12, Link appears at $60,$9D
$7B   $06  $26  $70  $1B  $0F  $27  3    open     wall     wall     wall     2   2   5    $30   -    $1B  -   -    $0F 5 rupees          2:$80,$90  all dead: shutters open, room item appears  $30 Gibdo x5
$7C   $26  $23  $EF  $91  $99  $27  5    wall     wall     wall     open     2   3   8    $6F   -    $11  D   -    $19 key               2:$80,$90  all dead: shutters open, room item appears  list $6F: $2B Bubble x3, $13 Zol x2, $1B Keese blue x3
$7D   $02  $03  $00  $21  $03  $00  5    open     open     open     open     2   3   3    $00   -    $21  -   -    -                     0:$C0,$90  -                                           none
$7E   $06  $06  $6D  $00  $03  $00  5    open     wall     open     wall     2   2   5    $2D   -    $00  -   -    -                     0:$C0,$90  -                                           $2D Bubble x5
$7F   $5E  $5E  $6A  $3F  $11  $20  c5   -        -        -        -        -   -   -    -     -    $3F  -   -    $11 book              2:$80,$90  -                                           treasure cellar: left exit to $5E, right exit to $5E, Link appears at $60,$AD
```
