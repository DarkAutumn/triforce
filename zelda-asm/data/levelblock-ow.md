# Overworld level block

How to read this: one row per overworld room `$00`-`$7F` of [`Level_BlockOW`](../src/bank6/level-blocks-common-data.asm#L66) (`dat/LevelBlockOW.dat`), first quest. Each row gives the
six raw attribute bytes A-F, then the fields the game reads from them. The formats are in `README.md`.
The second-quest overworld is the same block with a few bytes patched; those rooms are listed again at
the end. Find a room with `grep '^\$77 '`.

Columns:

- `exit`: X,Y where Link appears when he comes out of a cave or level (`A & $F0`, `(F & 7) * 16 + $4D`). [`Room_EnterCommon`](../src/bank5/enter-room.asm#L143)
- `zo`: `Z` = a Zora lives here (A bit 3). `se`: `S` = sea sound (A bit 2). [`Zora_TrySpawn`](../src/bank4/zora-ow-inits.asm#L77), [`Mode5Play_UpdateWorld`](../src/bank7/mode5-play-update.asm#L129)
- `po`, `pi`: outer and inner palette selectors (A bits 1-0, B bits 1-0). [`Room_FillPlayAreaAttrs`](../src/bank5/mode7-scroll-init.asm#L393)
- `dest`: where a stairs or cave entrance tile of the room leads (B bits 7-2). [`Warp_HandleOW`](../src/bank5/warps-stairs-caves.asm#L178)
  `level n` is the internal level number ([`CurLevel`](../src/Variables.inc#L12)); in Q2 the number shown on screen differs for some
  levels (see `levelinfo.md`). [`LevelInfo_LevelNumber`](../src/Variables.inc#L725)
- `cnt`: monster count selected by C bits 7-6 from [`LevelInfo_FoeCounts`](../src/Variables.inc#L716) ($01 $04 $05 $06); `list`: monster list id (C bits 5-0, D bit 7).
  The `monsters` text applies the rules of `objlists.md`. [`Room_EnterCommon`](../src/bank5/enter-room.asm#L143)
- `lay`: room layout id (D bits 6-0), see `room-layouts.md`. [`RoomLayout_LayOutRoomOW`](../src/bank5/layout-room-ow.asm#L106)
- `qs`: quest in which the room's tile-object secret works (F bits 7-6): `-` both, `Q1`, `Q2`. [`Secret_IsForOtherQuest`](../src/bank4/rock-wall-tree-secrets.asm#L331)
- `stairs`: position index (F bits 5-4) and the X,Y it selects in [`LevelInfo_ShortcutOrItemPosArray`](../src/Variables.inc#L718), where secret stairs appear. [`Room_GetShortcutOrItemXYForRoomId`](../src/bank1/bomb-candle-wield.asm#L280)
- `ed`: `E` = monsters enter from the screen edges (F bit 3). [`Spawn_AssignPositions`](../src/bank5/spawn-positions.asm#L64)
- `tileobj`: the tile object the room's layout creates (type, X, Y), or the dock. [`RoomLayout_CheckTileObj`](../src/bank5/layout-room-ow.asm#L267), [`Room_SetUpTileObjOW`](../src/bank5/enter-room.asm#L429)
- `monsters; cave`: the monsters made on entry (before the kill count is subtracted), then for a cave
  room the person's object type and wares (`@n` = price). [`Room_EnterCommon`](../src/bank5/enter-room.asm#L143), [`Cave_Init_SetUpTextAndWares`](../src/bank1/cave-init-person-text.asm#L145)
- Raw byte E is not a room attribute in the overworld (cave wares, table below). [`Cave_Init_SetUpTextAndWares`](../src/bank1/cave-init-person-text.asm#L145)

# Rooms (first quest)

```
room  A    B    C    D    E    F    exit     zo  se  po  pi  dest               cnt  list  lay  qs  stairs     ed  tileobj                    monsters; cave
$00   $A3  $27  $00  $00  $3F  $83  $A0,$7D  -   -   3   3   level 9            1    $00   $00  Q2  0:$50,$70  -   $63 rock wall at $A0,$80   none
$01   $93  $5F  $42  $01  $01  $00  $90,$4D  -   -   3   3   cave $17           4    $02   $01  -   0:$50,$70  -   $63 rock wall at $90,$50   $02 Lynel x4; person $71: no wares
$02   $63  $6B  $42  $02  $7F  $83  $60,$7D  -   -   3   3   cave $1A           4    $02   $02  Q2  0:$50,$70  -   $63 rock wall at $60,$80   $02 Lynel x4; person $74: $1F blue potion @40, $20 red potion @68
$03   $73  $5F  $1F  $03  $20  $03  $70,$7D  -   -   3   3   cave $17           1    $1F   $03  -   0:$50,$70  -   $63 rock wall at $70,$80   $1F boulder spawner x1; person $71: no wares
$04   $C3  $6B  $C1  $04  $3F  $00  $C0,$4D  -   -   3   3   cave $1A           6    $01   $04  -   0:$50,$70  -   -                          $01 Lynel x6; person $74: $1F blue potion @40, $20 red potion @68
$05   $53  $27  $E6  $85  $5A  $45  $50,$9D  -   -   3   3   level 9            6    $66   $05  Q1  0:$50,$70  -   $63 rock wall at $50,$A0   list $66: $01 Lynel x2, $0F Blue Leever x1, $02 Lynel x2, $10 Red Leever x1
$06   $B3  $47  $E4  $86  $7F  $84  $B0,$8D  -   -   3   3   cave $11           6    $64   $06  Q2  0:$50,$70  -   -                          list $64: $1A Peahat x2, $02 Lynel x2, $01 Lynel x2; person $6B: $20 red potion, $1A heart container
$07   $A3  $5F  $02  $07  $02  $03  $A0,$7D  -   -   3   3   cave $17           1    $02   $07  -   0:$50,$70  -   $63 rock wall at $A0,$80   $02 Lynel x1; person $71: no wares
$08   $03  $03  $1F  $06  $7F  $00  $00,$4D  -   -   3   3   -                  1    $1F   $06  -   0:$50,$70  -   -                          $1F boulder spawner x1
$09   $93  $4F  $00  $08  $7F  $A4  $90,$8D  -   -   3   3   cave $13           1    $00   $08  Q2  2:$90,$90  -   $62 rock at $40,$90        none; person $6D: $03 magic sword
$0A   $2B  $4B  $01  $09  $03  $00  $20,$4D  Z   -   3   3   cave $12           1    $01   $09  -   0:$50,$70  -   -                          $01 Lynel x1; person $6C: $02 white sword
$0B   $73  $17  $10  $0A  $7F  $03  $70,$7D  -   -   3   3   level 5            1    $10   $0A  -   0:$50,$70  -   -                          $10 Red Leever x1
$0C   $83  $7B  $CE  $0B  $3F  $00  $80,$4D  -   -   3   3   cave $1E           6    $0E   $0B  -   0:$50,$70  -   -                          $0E Tektite x6; person $78: $1C magic shield @160, $19 key @100, $06 blue candle @60
$0D   $93  $6B  $CE  $0C  $3F  $00  $90,$4D  -   -   3   3   cave $1A           6    $0E   $0C  -   0:$50,$70  -   $63 rock wall at $90,$50   $0E Tektite x6; person $74: $1F blue potion @40, $20 red potion @68
$0E   $57  $63  $00  $0D  $3F  $0B  $50,$7D  -   S   3   3   cave $18           1    $00   $0D  -   0:$50,$70  E   -                          none; person $72: $15 letter
$0F   $87  $8B  $00  $0E  $3F  $03  $80,$7D  -   S   3   3   cave $22           1    $00   $0E  -   0:$50,$70  -   -                          none; person $7C: $18 rupee @100
$10   $93  $5B  $41  $0F  $3F  $00  $90,$4D  -   -   3   3   cave $16           4    $01   $0F  -   0:$50,$70  -   $63 rock wall at $90,$50   $01 Lynel x4; person $70: $18 rupee @10, $18 rupee @10, $18 rupee @10
$11   $53  $63  $E4  $90  $3F  $A4  $50,$8D  -   -   3   3   cave $18           6    $64   $10  Q2  2:$90,$90  -   $62 rock at $80,$90        list $64: $1A Peahat x2, $02 Lynel x2, $01 Lynel x2; person $72: $15 letter
$12   $83  $7F  $C1  $11  $98  $00  $80,$4D  -   -   3   3   cave $1F           6    $01   $11  -   0:$50,$70  -   $63 rock wall at $80,$50   $01 Lynel x6; person $79: $1C magic shield @90, $04 food @100, $22 heart @10
$13   $23  $87  $65  $92  $98  $00  $20,$4D  -   -   3   3   cave $21           4    $65   $12  -   0:$50,$70  -   $63 rock wall at $20,$50   list $65: $01 Lynel x2, $02 Lynel x2; person $7B: $18 rupee @30
$14   $C3  $5F  $42  $13  $D8  $00  $C0,$4D  -   -   3   3   cave $17           4    $02   $13  -   0:$50,$70  -   $63 rock wall at $C0,$50   $02 Lynel x4; person $71: no wares
$15   $C3  $7B  $E4  $94  $3F  $80  $C0,$4D  -   -   3   3   cave $1E           6    $64   $14  Q2  0:$50,$70  -   $63 rock wall at $C0,$50   list $64: $1A Peahat x2, $02 Lynel x2, $01 Lynel x2; person $78: $1C magic shield @160, $19 key @100, $06 blue candle @60
$16   $63  $5B  $1F  $15  $3F  $00  $60,$4D  -   -   3   3   cave $16           1    $1F   $15  -   0:$50,$70  -   $63 rock wall at $60,$50   $1F boulder spawner x1; person $70: $18 rupee @10, $18 rupee @10, $18 rupee @10
$17   $0B  $03  $1F  $16  $3F  $00  $00,$4D  Z   -   3   3   -                  1    $1F   $16  -   0:$50,$70  -   -                          $1F boulder spawner x1
$18   $CB  $6B  $1F  $17  $3F  $82  $C0,$6D  Z   -   3   3   cave $1A           1    $1F   $17  Q2  0:$50,$70  -   $63 rock wall at $C0,$70   $1F boulder spawner x1; person $74: $1F blue potion @40, $20 red potion @68
$19   $4B  $1F  $1F  $18  $15  $82  $40,$6D  Z   -   3   3   level 7            1    $1F   $18  Q2  0:$50,$70  -   $63 rock wall at $40,$70   $1F boulder spawner x1
$1A   $6B  $6F  $CE  $19  $7F  $03  $60,$7D  Z   -   3   3   cave $1B           6    $0E   $19  -   0:$50,$70  -   -                          $0E Tektite x6; person $75: $18 rupee @5, $18 rupee @10, $18 rupee @20
$1B   $93  $17  $00  $1A  $3F  $A4  $90,$8D  -   -   3   3   level 5            1    $00   $1A  Q2  2:$90,$90  -   $62 rock at $C0,$90        none
$1C   $33  $57  $00  $1B  $3F  $02  $30,$6D  -   -   3   3   cave $15           1    $00   $1B  -   0:$50,$70  -   -                          none; person $6F: no wares
$1D   $27  $53  $DA  $1C  $3F  $02  $20,$6D  -   S   3   3   shortcut cave $14  6    $1A   $1C  -   0:$50,$70  -   $62 rock at $30,$90        $1A Peahat x6; person $6E: no wares
$1E   $CF  $5F  $CE  $1D  $1F  $00  $C0,$4D  Z   S   3   3   cave $17           6    $0E   $1D  -   0:$50,$70  -   $63 rock wall at $C0,$50   $0E Tektite x6; person $71: no wares
$1F   $67  $5A  $DA  $1E  $3F  $03  $60,$7D  -   S   3   2   cave $16           6    $1A   $1E  -   0:$50,$70  -   -                          $1A Peahat x6; person $70: $18 rupee @10, $18 rupee @10, $18 rupee @10
$20   $50  $44  $21  $1F  $E0  $B3  $50,$7D  -   -   0   0   cave $11           1    $21   $1F  Q2  3:$60,$90  -   $65 gravestone at $60,$90  $21 Ghini x1; person $6B: $20 red potion, $1A heart container
$21   $50  $4C  $21  $20  $98  $62  $50,$6D  -   -   0   0   cave $13           1    $21   $20  Q1  2:$90,$90  -   $65 gravestone at $90,$90  $21 Ghini x1; person $6D: $03 magic sword
$22   $73  $18  $02  $21  $58  $03  $70,$7D  -   -   3   0   level 6            1    $02   $21  -   0:$50,$70  -   -                          $02 Lynel x1
$23   $43  $53  $42  $22  $D8  $11  $40,$5D  -   -   3   3   shortcut cave $14  4    $02   $22  -   1:$40,$90  -   $62 rock at $30,$90        $02 Lynel x4; person $6E: no wares
$24   $03  $03  $00  $23  $98  $00  $00,$4D  -   -   3   3   -                  1    $00   $23  -   0:$50,$70  -   -                          none
$25   $A3  $77  $5A  $24  $58  $00  $A0,$4D  -   -   3   3   cave $1D           4    $1A   $24  -   0:$50,$70  -   -                          $1A Peahat x4; person $77: $1C magic shield @130, $00 bomb @20, $08 wooden arrow @80
$26   $3B  $7F  $DA  $25  $D8  $00  $30,$4D  Z   -   3   3   cave $1F           6    $1A   $25  -   0:$50,$70  -   $63 rock wall at $30,$50   $1A Peahat x6; person $79: $1C magic shield @90, $04 food @100, $22 heart @10
$27   $EB  $6B  $DA  $26  $1C  $40  $E0,$4D  Z   -   3   3   cave $1A           6    $1A   $26  Q1  0:$50,$70  -   $63 rock wall at $E0,$50   $1A Peahat x6; person $74: $1F blue potion @40, $20 red potion @68
$28   $EB  $86  $DA  $27  $00  $05  $E0,$9D  Z   -   3   2   cave $21           6    $1A   $27  -   0:$50,$70  -   $64 tree at $D0,$A0        $1A Peahat x6; person $7B: $18 rupee @30
$29   $B3  $6B  $50  $28  $C8  $84  $B0,$8D  -   -   3   3   cave $1A           4    $10   $28  Q2  0:$50,$70  -   -                          $10 Red Leever x4; person $74: $1F blue potion @40, $20 red potion @68
$2A   $03  $03  $CF  $29  $1C  $00  $00,$4D  -   -   3   3   -                  6    $0F   $29  -   0:$50,$70  -   -                          $0F Blue Leever x6
$2B   $B3  $8F  $E7  $AA  $19  $84  $B0,$8D  -   -   3   3   cave $23           6    $67   $2A  Q2  0:$50,$70  -   -                          list $67: $0F Blue Leever x2, $1A Peahat x3, $10 Red Leever x1; person $7D: $18 rupee @10
$2C   $93  $47  $4E  $2B  $C6  $45  $90,$9D  -   -   3   3   cave $11           4    $0E   $2B  Q1  0:$50,$70  -   $63 rock wall at $90,$A0   $0E Tektite x4; person $6B: $20 red potion, $1A heart container
$2D   $5F  $87  $AA  $AC  $1C  $00  $50,$4D  Z   S   3   3   cave $21           5    $6A   $2C  -   0:$50,$70  -   $63 rock wall at $50,$50   list $6A: $08 Octorok fast x3, $09 blue Octorok slow x2; person $7B: $18 rupee @30
$2E   $0F  $03  $49  $2D  $04  $00  $00,$4D  Z   S   3   3   -                  4    $09   $2D  -   0:$50,$70  -   -                          $09 blue Octorok slow x4
$2F   $60  $44  $00  $2E  $E2  $03  $60,$7D  -   -   0   0   cave $11           1    $00   $2E  -   0:$50,$70  -   -                          none; person $6B: $20 red potion, $1A heart container
$30   $70  $18  $21  $2F  $19  $84  $70,$8D  -   -   0   0   level 6            1    $21   $2F  Q2  0:$50,$70  -   -                          $21 Ghini x1
$31   $70  $00  $21  $30  $12  $04  $70,$8D  -   -   0   0   -                  1    $21   $30  -   0:$50,$70  -   -                          $21 Ghini x1
$32   $03  $03  $E4  $B1  $C4  $00  $00,$4D  -   -   3   3   -                  6    $64   $31  -   0:$50,$70  -   -                          list $64: $1A Peahat x2, $02 Lynel x2, $01 Lynel x2
$33   $A3  $68  $00  $32  $3F  $00  $A0,$4D  -   -   3   0   cave $1A           1    $00   $32  -   0:$50,$70  -   $63 rock wall at $A0,$50   none; person $74: $1F blue potion @40, $20 red potion @68
$34   $43  $83  $4F  $33  $98  $02  $40,$6D  -   -   3   3   cave $20           4    $0F   $33  -   0:$50,$70  -   -                          $0F Blue Leever x4; person $7A: $19 key @80, $12 blue ring @250, $04 food @60
$35   $0B  $03  $00  $34  $7F  $00  $00,$4D  Z   -   3   3   -                  1    $00   $34  -   0:$50,$70  -   -                          none
$36   $0B  $03  $00  $35  $3F  $00  $00,$4D  Z   -   3   3   -                  1    $00   $35  -   0:$50,$70  -   -                          none
$37   $73  $07  $08  $36  $98  $03  $70,$7D  -   -   3   3   level 1            1    $08   $36  -   0:$50,$70  -   -                          $08 Octorok fast x1
$38   $0B  $03  $E8  $B7  $7F  $00  $00,$4D  Z   -   3   3   -                  6    $68   $37  -   0:$50,$70  -   -                          list $68: $09 blue Octorok slow x1, $08 Octorok fast x5
$39   $03  $02  $2F  $38  $3F  $80  $00,$4D  -   -   3   2   -                  1    $2F   $38  Q2  0:$50,$70  -   -                          $2F Pond fairy x1
$3A   $C3  $47  $E7  $B9  $98  $84  $C0,$8D  -   -   3   3   cave $11           6    $67   $39  Q2  0:$50,$70  -   -                          list $67: $0F Blue Leever x2, $1A Peahat x3, $10 Red Leever x1; person $6B: $20 red potion, $1A heart container
$3B   $63  $03  $4F  $3A  $7F  $02  $60,$6D  -   -   3   3   -                  4    $0F   $3A  -   0:$50,$70  -   -                          $0F Blue Leever x4
$3C   $72  $0A  $0A  $0A  $00  $03  $70,$7D  -   -   2   2   level 2            1    $0A   $0A  -   0:$50,$70  -   -                          $0A blue Octorok fast x1
$3D   $92  $86  $43  $3B  $00  $02  $90,$6D  -   -   2   2   cave $21           4    $03   $3B  -   0:$50,$70  -   -                          $03 Moblin x4; person $7B: $18 rupee @30
$3E   $0F  $03  $AA  $BC  $00  $00  $00,$4D  Z   S   3   3   -                  5    $6A   $3C  -   0:$50,$70  -   -                          list $6A: $08 Octorok fast x3, $09 blue Octorok slow x2
$3F   $0F  $03  $09  $3D  $00  $00  $00,$4D  Z   S   3   3   -                  1    $09   $3D  -   0:$50,$70  -   $61 dock                   $09 blue Octorok slow x1
$40   $00  $00  $21  $3E  $00  $00  $00,$4D  -   -   0   0   -                  1    $21   $3E  -   0:$50,$70  -   -                          $21 Ghini x1
$41   $00  $00  $21  $3F  $00  $00  $00,$4D  -   -   0   0   -                  1    $21   $3F  -   0:$50,$70  -   -                          $21 Ghini x1
$42   $63  $1F  $04  $38  $00  $01  $60,$5D  -   -   3   3   level 7            1    $04   $38  -   0:$50,$70  -   -                          $04 Moblin x1
$43   $03  $02  $2F  $38  $00  $00  $00,$4D  -   -   3   2   -                  1    $2F   $38  -   0:$50,$70  -   -                          $2F Pond fairy x1
$44   $4B  $76  $47  $40  $00  $00  $40,$4D  Z   -   3   2   cave $1D           4    $07   $40  -   0:$50,$70  -   -                          $07 Octorok slow x4; person $77: $1C magic shield @130, $00 bomb @20, $08 wooden arrow @80
$45   $83  $12  $1A  $41  $00  $03  $80,$7D  -   -   3   2   level 4            1    $1A   $41  -   0:$50,$70  -   -                          $1A Peahat x1
$46   $8A  $7E  $00  $42  $00  $06  $80,$AD  Z   -   2   2   cave $1F           1    $00   $42  -   0:$50,$70  -   $64 tree at $90,$B0        none; person $79: $1C magic shield @90, $04 food @100, $22 heart @10
$47   $BA  $46  $00  $43  $00  $45  $B0,$9D  Z   -   2   2   cave $11           1    $00   $43  Q1  0:$50,$70  -   $64 tree at $B0,$B0        none; person $6B: $20 red potion, $1A heart container
$48   $BA  $86  $50  $44  $00  $02  $B0,$6D  Z   -   2   2   cave $21           4    $10   $44  -   0:$50,$70  -   $64 tree at $D0,$60        $10 Red Leever x4; person $7B: $18 rupee @30
$49   $32  $52  $E8  $C5  $00  $02  $30,$6D  -   -   2   2   shortcut cave $14  6    $68   $45  -   0:$50,$70  -   $62 rock at $30,$90        list $68: $09 blue Octorok slow x1, $08 Octorok fast x5; person $6E: no wares
$4A   $B2  $76  $CD  $46  $00  $00  $B0,$4D  -   -   2   2   cave $1D           6    $0D   $46  -   0:$50,$70  -   -                          $0D Tektite x6; person $77: $1C magic shield @130, $00 bomb @20, $08 wooden arrow @80
$4B   $C2  $6A  $C4  $47  $00  $09  $C0,$5D  -   -   2   2   cave $1A           6    $04   $47  -   0:$50,$70  E   $64 tree at $B0,$60        $04 Moblin x6; person $74: $1F blue potion @40, $20 red potion @68
$4C   $02  $02  $AA  $C8  $00  $00  $00,$4D  -   -   2   2   -                  5    $6A   $48  -   0:$50,$70  -   -                          list $6A: $08 Octorok fast x3, $09 blue Octorok slow x2
$4D   $C2  $7E  $43  $49  $00  $0C  $C0,$8D  -   -   2   2   cave $1F           4    $03   $49  -   0:$50,$70  E   $64 tree at $D0,$A0        $03 Moblin x4; person $79: $1C magic shield @90, $04 food @100, $22 heart @10
$4E   $72  $8E  $43  $4A  $0A  $0A  $70,$6D  -   -   2   2   cave $23           4    $03   $4A  -   0:$50,$70  E   -                          $03 Moblin x4; person $7D: $18 rupee @10
$4F   $0F  $03  $AB  $CB  $0A  $00  $00,$4D  Z   S   3   3   -                  5    $6B   $4B  -   0:$50,$70  -   -                          list $6B: $0A blue Octorok fast x2, $07 Octorok slow x3
$50   $00  $00  $82  $4C  $0A  $00  $00,$4D  -   -   0   0   -                  5    $02   $4C  -   0:$50,$70  -   -                          $02 Lynel x5
$51   $A3  $8F  $83  $4D  $00  $05  $A0,$9D  -   -   3   3   cave $23           5    $03   $4D  -   0:$50,$70  -   $64 tree at $90,$A0        $03 Moblin x5; person $7D: $18 rupee @10
$52   $03  $03  $63  $CE  $00  $08  $00,$4D  -   -   3   3   -                  4    $63   $4E  -   0:$50,$70  E   -                          list $63: $03 Moblin x2, $04 Moblin x2
$53   $83  $8A  $A2  $CF  $00  $85  $80,$9D  -   -   3   2   cave $22           5    $62   $4F  Q2  0:$50,$70  -   $64 tree at $60,$A0        list $62: $03 Moblin x3, $04 Moblin x2; person $7C: $18 rupee @100
$54   $0A  $02  $69  $D0  $00  $00  $00,$4D  Z   -   2   2   -                  4    $69   $50  -   0:$50,$70  -   -                          list $69: $08 Octorok fast x2, $07 Octorok slow x2
$55   $0A  $02  $07  $51  $00  $00  $00,$4D  Z   -   2   2   -                  1    $07   $51  -   0:$50,$70  -   $61 dock                   $07 Octorok slow x1
$56   $BA  $8E  $47  $52  $00  $06  $B0,$AD  Z   -   2   2   cave $23           4    $07   $52  -   0:$50,$70  -   $64 tree at $A0,$A0        $07 Octorok slow x4; person $7D: $18 rupee @10
$57   $02  $02  $69  $D3  $00  $08  $00,$4D  -   -   2   2   -                  4    $69   $53  -   0:$50,$70  E   -                          list $69: $08 Octorok fast x2, $07 Octorok slow x2
$58   $C2  $86  $69  $D4  $00  $8C  $C0,$8D  -   -   2   2   cave $21           4    $69   $54  Q2  0:$50,$70  E   -                          list $69: $08 Octorok fast x2, $07 Octorok slow x2; person $7B: $18 rupee @30
$59   $0A  $02  $5A  $55  $00  $00  $00,$4D  Z   -   2   2   -                  4    $1A   $55  -   0:$50,$70  -   -                          $1A Peahat x4
$5A   $0A  $02  $47  $56  $28  $00  $00,$4D  Z   -   2   2   -                  4    $07   $56  -   0:$50,$70  -   -                          $07 Octorok slow x4
$5B   $32  $8E  $63  $D7  $00  $0D  $30,$9D  -   -   2   2   cave $23           4    $63   $57  -   0:$50,$70  E   $64 tree at $20,$A0        list $63: $03 Moblin x2, $04 Moblin x2; person $7D: $18 rupee @10
$5C   $02  $02  $43  $58  $44  $08  $00,$4D  -   -   2   2   -                  4    $03   $58  -   0:$50,$70  E   -                          $03 Moblin x4
$5D   $02  $02  $43  $59  $05  $08  $00,$4D  -   -   2   2   -                  4    $03   $59  -   0:$50,$70  E   -                          $03 Moblin x4
$5E   $72  $7A  $83  $5A  $0A  $00  $70,$4D  -   -   2   2   cave $1E           5    $03   $5A  -   0:$50,$70  -   -                          $03 Moblin x5; person $78: $1C magic shield @160, $19 key @100, $06 blue candle @60
$5F   $0F  $03  $AA  $CB  $14  $00  $00,$4D  Z   S   3   3   -                  5    $6A   $4B  -   0:$50,$70  -   -                          list $6A: $08 Octorok fast x3, $09 blue Octorok slow x2
$60   $C3  $5B  $E4  $DB  $0A  $84  $C0,$8D  -   -   3   3   cave $16           6    $64   $5B  Q2  0:$50,$70  -   -                          list $64: $1A Peahat x2, $02 Lynel x2, $01 Lynel x2; person $70: $18 rupee @10, $18 rupee @10, $18 rupee @10
$61   $03  $03  $83  $5C  $1E  $08  $00,$4D  -   -   3   3   -                  5    $03   $5C  -   0:$50,$70  E   -                          $03 Moblin x5
$62   $63  $8B  $83  $5D  $32  $4B  $60,$7D  -   -   3   3   cave $22           5    $03   $5D  Q1  0:$50,$70  E   $64 tree at $80,$60        $03 Moblin x5; person $7C: $18 rupee @100
$63   $73  $5F  $EC  $DE  $82  $05  $70,$9D  -   -   3   3   cave $17           6    $6C   $5E  -   0:$50,$70  -   $64 tree at $60,$A0        list $6C: $03 Moblin x1, $0A blue Octorok fast x2, $04 Moblin x3; person $71: no wares
$64   $72  $6A  $AA  $DF  $14  $00  $70,$4D  -   -   2   2   cave $1A           5    $6A   $5F  -   0:$50,$70  -   -                          list $6A: $08 Octorok fast x3, $09 blue Octorok slow x2; person $74: $1F blue potion @40, $20 red potion @68
$65   $0A  $03  $69  $E0  $50  $00  $00,$4D  Z   -   2   3   -                  4    $69   $60  -   0:$50,$70  -   -                          list $69: $08 Octorok fast x2, $07 Octorok slow x2
$66   $72  $7B  $69  $E1  $A0  $00  $70,$4D  -   -   2   3   cave $1E           4    $69   $61  -   0:$50,$70  -   -                          list $69: $08 Octorok fast x2, $07 Octorok slow x2; person $78: $1C magic shield @160, $19 key @100, $06 blue candle @60
$67   $72  $86  $47  $62  $64  $40  $70,$4D  -   -   2   2   cave $21           4    $07   $62  Q1  0:$50,$70  -   $63 rock wall at $70,$50   $07 Octorok slow x4; person $7B: $18 rupee @30
$68   $32  $5E  $47  $63  $3C  $0C  $30,$8D  -   -   2   2   cave $17           4    $07   $63  -   0:$50,$70  E   $64 tree at $20,$A0        $07 Octorok slow x4; person $71: no wares
$69   $0A  $02  $47  $64  $5A  $00  $00,$4D  Z   -   2   2   -                  4    $07   $64  -   0:$50,$70  -   -                          $07 Octorok slow x4
$6A   $DA  $5E  $69  $E5  $64  $05  $D0,$9D  Z   -   2   2   cave $17           4    $69   $65  -   0:$50,$70  -   $64 tree at $C0,$A0        list $69: $08 Octorok fast x2, $07 Octorok slow x2; person $71: no wares
$6B   $52  $8A  $EC  $E6  $0A  $4D  $50,$9D  -   -   2   2   cave $22           6    $6C   $66  Q1  0:$50,$70  E   $64 tree at $80,$A0        list $6C: $03 Moblin x1, $0A blue Octorok fast x2, $04 Moblin x3; person $7C: $18 rupee @100
$6C   $42  $22  $44  $67  $50  $89  $40,$5D  -   -   2   2   level 8            4    $04   $67  Q2  0:$50,$70  E   $64 tree at $50,$60        $04 Moblin x4
$6D   $62  $22  $44  $68  $FA  $49  $60,$5D  -   -   2   2   level 8            4    $04   $68  Q1  0:$50,$70  E   $64 tree at $A0,$60        $04 Moblin x4
$6E   $C2  $8E  $EC  $E9  $3C  $84  $C0,$8D  -   -   2   2   cave $23           6    $6C   $69  Q2  0:$50,$70  -   -                          list $6C: $03 Moblin x1, $0A blue Octorok fast x2, $04 Moblin x3; person $7D: $18 rupee @10
$6F   $3F  $77  $A8  $EA  $00  $00  $30,$4D  Z   S   3   3   cave $1D           5    $68   $6A  -   0:$50,$70  -   -                          list $68: $09 blue Octorok slow x1, $08 Octorok fast x4; person $77: $1C magic shield @130, $00 bomb @20, $08 wooden arrow @80
$70   $B3  $73  $5A  $6B  $1E  $00  $B0,$4D  -   -   3   3   cave $1C           4    $1A   $6B  -   0:$50,$70  -   -                          $1A Peahat x4; person $76: $18 rupee @10, $18 rupee @30, $18 rupee @50
$71   $53  $87  $83  $6C  $00  $48  $50,$4D  -   -   3   3   cave $21           5    $03   $6C  Q1  0:$50,$70  E   $63 rock wall at $50,$50   $03 Moblin x5; person $7B: $18 rupee @30
$72   $43  $5F  $62  $ED  $00  $8C  $40,$8D  -   -   3   3   cave $17           4    $62   $6D  Q2  0:$50,$70  E   -                          list $62: $03 Moblin x3, $04 Moblin x1; person $71: no wares
$73   $03  $03  $43  $6E  $64  $08  $00,$4D  -   -   3   3   -                  4    $03   $6E  -   0:$50,$70  E   -                          $03 Moblin x4
$74   $82  $0F  $0E  $6F  $00  $03  $80,$7D  -   -   2   3   level 3            1    $0E   $6F  -   0:$50,$70  -   -                          $0E Tektite x1
$75   $2A  $66  $E7  $F0  $00  $00  $20,$4D  Z   -   2   2   cave $19           6    $67   $70  -   0:$50,$70  -   -                          list $67: $0F Blue Leever x2, $1A Peahat x3, $10 Red Leever x1; person $73: no wares
$76   $62  $5A  $4E  $71  $0A  $00  $60,$4D  -   -   2   2   cave $16           4    $0E   $71  -   0:$50,$70  -   $63 rock wall at $60,$50   $0E Tektite x4; person $70: $18 rupee @10, $18 rupee @10, $18 rupee @10
$77   $42  $42  $00  $72  $00  $00  $40,$4D  -   -   2   2   cave $10           1    $00   $72  -   0:$50,$70  -   -                          none; person $6A: $01 wooden sword
$78   $52  $6A  $47  $73  $FF  $0C  $50,$8D  -   -   2   2   cave $1A           4    $07   $73  -   0:$50,$70  E   $64 tree at $40,$A0        $07 Octorok slow x4; person $74: $1F blue potion @40, $20 red potion @68
$79   $63  $53  $8D  $74  $FF  $22  $60,$6D  -   -   3   3   shortcut cave $14  5    $0D   $74  -   2:$90,$90  -   $62 rock at $80,$90        $0D Tektite x5; person $6E: no wares
$7A   $03  $03  $4D  $06  $FF  $00  $00,$4D  -   -   3   3   -                  4    $0D   $06  -   0:$50,$70  -   -                          $0D Tektite x4
$7B   $9F  $47  $D0  $75  $FF  $40  $90,$4D  Z   S   3   3   cave $11           6    $10   $75  Q1  0:$50,$70  -   $63 rock wall at $90,$50   $10 Red Leever x6; person $6B: $20 red potion, $1A heart container
$7C   $6F  $5B  $D0  $76  $FF  $00  $60,$4D  Z   S   3   3   cave $16           6    $10   $76  -   0:$50,$70  -   $63 rock wall at $60,$50   $10 Red Leever x6; person $70: $18 rupee @10, $18 rupee @10, $18 rupee @10
$7D   $6F  $5F  $49  $76  $FF  $00  $60,$4D  Z   S   3   3   cave $17           4    $09   $76  -   0:$50,$70  -   $63 rock wall at $60,$50   $09 blue Octorok slow x4; person $71: no wares
$7E   $0F  $03  $48  $77  $FF  $00  $00,$4D  Z   S   3   3   -                  4    $08   $77  -   0:$50,$70  -   -                          $08 Octorok fast x4
$7F   $0F  $03  $09  $78  $FF  $00  $00,$4D  Z   S   3   3   -                  1    $09   $78  -   0:$50,$70  -   -                          $09 blue Octorok slow x1
```

# Caves

One row per cave index (B bits 7-2 of a room): the person's object type, the text it says, its cave
flags and its three wares with prices, from overworld array E and [`Cave_PersonTextSelectors`](../src/bank1/cave-init-person-text.asm#L71). Flags are explained in
`README.md`. `rooms` are the first-quest rooms that lead to the cave. [`Cave_Init_SetUpTextAndWares`](../src/bank1/cave-init-person-text.asm#L145)

```
cave  person  sel  text  ware bytes   price bytes  flags                                               wares                                                        says                                      rooms
$10   $6A     $40  $00   $3F $01 $7F  $00 $00 $00  pick-up,show-items                                  - | $01 wooden sword | -                                     IT'S DANGEROUS TO GO / ALONE! TAKE THIS.  Q1 $77 Q2 $77
$11   $6B     $60  $20   $20 $3F $5A  $00 $00 $00  pick-up,show-items                                  $20 red potion | - | $1A heart container                     TAKE ANY ONE YOU WANT.                    Q1 $06,$20,$2C,$2F,$3A,$47,$7B Q2 $06,$20,$2C,$2F,$3A,$47,$7B
$12   $6C     $42  $02   $7F $02 $7F  $00 $00 $00  pick-up,show-items,heart-requirement                - | $02 white sword | -                                      MASTER USING IT AND / YOU CAN HAVE THIS.  Q1 $0A Q2 $0A
$13   $6D     $42  $02   $7F $03 $7F  $00 $00 $00  pick-up,show-items,heart-requirement                - | $03 magic sword | -                                      MASTER USING IT AND / YOU CAN HAVE THIS.  Q1 $09,$21 Q2 $09,$21
$14   $6E     $04  $04   $3F $3F $3F  $00 $00 $00  -                                                   - | - | -                                                    TAKE ANY ROAD YOU WANT.                   Q1 $1D,$23,$49,$79 Q2 $1D,$23,$49,$79
$15   $6F     $06  $06   $3F $3F $3F  $00 $00 $00  -                                                   - | - | -                                                    SECRET IS IN THE TREE / AT THE DEAD-END.  Q1 $1C Q2 $1C
$16   $70     $48  $08   $98 $98 $D8  $0A $0A $0A  pick-up,show-items,show-prices,money,minus-sign     $18 rupee @10 | $18 rupee @10 | $18 rupee @10                LET'S PLAY MONEY / MAKING GAME.           Q1 $10,$16,$1F,$60,$76,$7C Q2 $10,$16,$1F,$60,$76,$7C
$17   $71     $0A  $0A   $3F $3F $3F  $00 $00 $00  -                                                   - | - | -                                                    PAY ME FOR THE DOOR / REPAIR CHARGE.      Q1 $01,$03,$07,$14,$1E,$63,$68,$6A,$72,$7D Q2 $01,$03,$07,$14,$1E,$63,$68,$6A,$72,$7D
$18   $72     $4C  $0C   $3F $15 $7F  $00 $00 $00  pick-up,show-items                                  - | $15 letter | -                                           SHOW THIS TO THE / OLD WOMAN.             Q1 $0E,$11 Q2 $11
$19   $73     $0E  $0E   $3F $3F $3F  $00 $00 $00  -                                                   - | - | -                                                    MEET THE OLD MAN / AT THE GRAVE.          Q1 $75 Q2 $75
$1A   $74     $D0  $10   $1F $3F $E0  $28 $00 $44  pick-up,pay,show-items,show-prices                  $1F blue potion @40 | - | $20 red potion @68                 BUY MEDICINE BEFORE / YOU GO.             Q1 $02,$04,$0D,$18,$27,$29,$33,$4B,$64,$78 Q2 $02,$04,$0D,$18,$27,$29,$33,$4B,$64,$78
$1B   $75     $D2  $12   $98 $58 $D8  $05 $0A $14  pick-up,pay,show-items,show-prices,hint,minus-sign  $18 rupee @5 | $18 rupee @10 | $18 rupee @20                 PAY ME AND I'LL TALK.                     Q1 $1A Q2 $1A
$1C   $76     $D2  $12   $98 $58 $D8  $0A $1E $32  pick-up,pay,show-items,show-prices,hint,minus-sign  $18 rupee @10 | $18 rupee @30 | $18 rupee @50                PAY ME AND I'LL TALK.                     Q1 $70 Q2 $70
$1D   $77     $DC  $1C   $1C $00 $C8  $82 $14 $50  pick-up,pay,show-items,show-prices                  $1C magic shield @130 | $00 bomb @20 | $08 wooden arrow @80  BUY SOMETHIN' WILL YA!                    Q1 $25,$44,$4A,$6F Q2 $25,$44,$4A,$6F
$1E   $78     $DC  $1C   $1C $19 $C6  $A0 $64 $3C  pick-up,pay,show-items,show-prices                  $1C magic shield @160 | $19 key @100 | $06 blue candle @60   BUY SOMETHIN' WILL YA!                    Q1 $0C,$15,$5E,$66 Q2 $0C,$0E,$15,$5E,$66,$74
$1F   $79     $DE  $1E   $1C $04 $E2  $5A $64 $0A  pick-up,pay,show-items,show-prices                  $1C magic shield @90 | $04 food @100 | $22 heart @10         BOY, THIS IS / REALLY EXPENSIVE!          Q1 $12,$26,$46,$4D Q2 $12,$26,$46,$4D
$20   $7A     $DE  $1E   $19 $12 $C4  $50 $FA $3C  pick-up,pay,show-items,show-prices                  $19 key @80 | $12 blue ring @250 | $04 food @60              BOY, THIS IS / REALLY EXPENSIVE!          Q1 $34 Q2 $0F
$21   $7B     $62  $22   $3F $98 $7F  $00 $1E $00  pick-up,show-items,money                            - | $18 rupee @30 | -                                        IT'S A SECRET / TO EVERYBODY.             Q1 $13,$28,$2D,$3D,$48,$58,$67,$71 Q2 $13,$22,$28,$2D,$3D,$48,$58,$67,$71
$22   $7C     $62  $22   $3F $98 $7F  $00 $64 $00  pick-up,show-items,money                            - | $18 rupee @100 | -                                       IT'S A SECRET / TO EVERYBODY.             Q1 $0F,$53,$62,$6B Q2 $53,$62,$6B
$23   $7D     $62  $22   $3F $98 $7F  $00 $0A $00  pick-up,show-items,money                            - | $18 rupee @10 | -                                        IT'S A SECRET / TO EVERYBODY.             Q1 $2B,$4E,$51,$56,$5B,$6E Q2 $2B,$4E,$51,$56,$5B,$6E
```

# Rooms patched for the second quest

The rooms whose bytes the second-quest patch changes, decoded with the patched values. [`Mode2Load_Update_Full`](../src/bank6/level-load-mode2.asm#L297)

```
room  A    B    C    D    E    F    exit     zo  se  po  pi  dest      cnt  list  lay  qs  stairs     ed  tileobj  monsters; cave
$0B   $73  $17  $2F  $7B  $7F  $03  $70,$7D  -   -   3   3   level 5   1    $2F   $7B  -   0:$50,$70  -   -        $2F Pond fairy x1
$0E   $57  $7B  $00  $0D  $3F  $0B  $50,$7D  -   S   3   3   cave $1E  1    $00   $0D  -   0:$50,$70  E   -        none; person $78: $1C magic shield @160, $19 key @100, $06 blue candle @60
$0F   $87  $83  $00  $0E  $3F  $03  $80,$7D  -   S   3   3   cave $20  1    $00   $0E  -   0:$50,$70  -   -        none; person $7A: $19 key @80, $12 blue ring @250, $04 food @60
$22   $73  $84  $02  $21  $58  $03  $70,$7D  -   -   3   0   cave $21  1    $02   $21  -   0:$50,$70  -   -        $02 Lynel x1; person $7B: $18 rupee @30
$34   $43  $0F  $4F  $33  $98  $02  $40,$6D  -   -   3   3   level 3   4    $0F   $33  -   0:$50,$70  -   -        $0F Blue Leever x4
$3C   $72  $0B  $0A  $7B  $00  $01  $70,$5D  -   -   2   3   level 2   1    $0A   $7B  -   0:$50,$70  -   -        $0A blue Octorok fast x1
$45   $83  $12  $1A  $41  $00  $03  $80,$7D  -   -   3   2   level 4   1    $1A   $41  -   0:$50,$70  -   -        $1A Peahat x1
$74   $72  $7A  $0E  $5A  $00  $00  $70,$4D  -   -   2   2   cave $1E  1    $0E   $5A  -   0:$50,$70  -   -        $0E Tektite x1; person $78: $1C magic shield @160, $19 key @100, $06 blue candle @60
```
