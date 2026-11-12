# Level info

How to read this: one section per level (`0` = overworld) and quest, decoded from the level info blocks
([`LevelInfoAddrs`](../src/bank6/level-load-mode2.asm#L71)) with the second-quest tables applied for Q2 ([`UpdateMode2Load_Full`](../src/bank6/level-load-mode2.asm#L297)). Every line starts with the RAM symbol of the field;
the field formats are in `README.md`. A position is written X,Y. The overworld block is the same in
both quests. Palettes and transfer records are given as raw bytes.

# Summary

```
quest  level  shown as  start  Triforce  boss  startY  flags  cellars
Q1     0      -         $77    $2A       $2A   $8D     $067F  $1D,$23,$49,$79
Q1     1      LEVEL-1   $73    $36       $35   $DD     $06FF  $7F
Q1     2      LEVEL-2   $7D    $0D       $0E   $DD     $06FF  -
Q1     3      LEVEL-3   $7C    $3D       $4D   $DD     $06FF  -
Q1     4      LEVEL-4   $71    $03       $13   $DD     $06FF  $60
Q1     5      LEVEL-5   $76    $14       $24   $DD     $06FF  $07,$04
Q1     6      LEVEL-6   $79    $0C       $1C   $DD     $06FF  $08,$75
Q1     7      LEVEL-7   $79    $2B       $2A   $DD     $077F  $7B,$4A
Q1     8      LEVEL-8   $7E    $2C       $3C   $DD     $077F  $2F,$0F,$6F
Q1     9      LEVEL-9   $76    $32       $42   $DD     $077F  $60,$70,$72,$75,$67,$77,$00,$4F
```

```
quest  level  shown as  start  Triforce  boss  startY  flags  cellars
Q2     1      LEVEL-1   $77    $08       $07   $DD     $06FF  $28
Q2     2      LEVEL-3   $75    $20       $30   $DD     $06FF  $56
Q2     3      LEVEL-2   $79    $1B       $2B   $DD     $06FF  $09,$0B
Q2     4      LEVEL-5   $72    $00       $10   $DD     $06FF  $21,$58,$7A
Q2     5      LEVEL-4   $7D    $4F       $5F   $DD     $06FF  $0F,$6A,$7F
Q2     6      LEVEL-6   $74    $16       $26   $DD     $06FF  $03,$73,$46
Q2     7      LEVEL-8   $7F    $2D       $3D   $DD     $077F  $02,$03,$04,$05,$20,$21,$26,$2B,$2C
Q2     8      LEVEL-7   $79    $1B       $1C   $DD     $077F  $27,$30,$37,$60,$67,$70
Q2     9      LEVEL-9   $74    $07       $17   $DD     $077F  $71,$72,$75,$76,$77
```

- `level`: the internal level number ([`CurLevel`](../src/Variables.inc#L12)), which selects the block; `shown as`: the number drawn on
  screen ([`LevelInfo_LevelNumber`](../src/Variables.inc#L672)). They differ for some levels in Q2. [`InitMode3_Sub7`](../src/bank5/mode3-mode6-init.asm#L114)
- `start`, `Triforce`, `boss`: room ids ([`LevelInfo_StartRoomId`](../src/Variables.inc#L669), [`LevelInfo_TriforceRoomId`](../src/Variables.inc#L670), [`LevelInfo_BossRoomId`](../src/Variables.inc#L675)). In the overworld only the start room is read.
- `cellars`: cellar rooms of the level; for the overworld, the four rooms the shortcut cave joins. [`LevelInfo_CellarRoomIdArray`](../src/Variables.inc#L673)

# Overworld (level 0), Q1

Block [`LevelInfoOW`](../src/bank6/level-blocks-common-data.asm#L86) (`dat/LevelInfoOW.dat`).

```
field                                     value
LevelInfo_StartRoomId                     $77
LevelInfo_TriforceRoomId                  $2A
LevelInfo_BossRoomId                      $2A
LevelInfo_LevelNumber                     $00
LevelInfo_StartY                          $8D
LevelInfo_FoeCounts                       $01 $04 $05 $06
LevelInfo_ShortcutOrItemPosArray          $57 $49 $99 $69  = 0:$50,$70  1:$40,$90  2:$90,$90  3:$60,$90
LevelInfo_CellarRoomIdArray               $1D $23 $49 $79 $FF $FF $FF $FF $FF $FF
LevelInfo_WorldFlagsAddr                  $067F
LevelInfo_SubmenuMapRotation              $00
LevelInfo_StatusBarMapXOffset             $00
LevelInfo_SubmenuMapMask                  $FF $FF $FF $FF $FF $FF $FF $FF $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_StatusBarMapTransferBuf         $20 $62 $48 $F5 $20 $82 $48 $F5 $20 $A2 $48 $F5 $20 $C2 $48 $F5 $FF
  record                                  $2062 8: $F5 repeated
  record                                  $2082 8: $F5 repeated
  record                                  $20A2 8: $F5 repeated
  record                                  $20C2 8: $F5 repeated
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $1A $37 $12 $0F $17 $37 $12 $0F $29 $27 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $0C $1C $2C $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $1A $37 $12
  palette row 3                           $0F $17 $37 $12
  palette row 4                           $0F $29 $27 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $0C $1C $2C
LevelInfo_PaletteCycles cycle $00 step 0  $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_PaletteCycles cycle $00 step 1  $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_PaletteCycles cycle $00 step 2  $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_PaletteCycles cycle $00 step 3  $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_PaletteCycles cycle $20 step 0  $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_PaletteCycles cycle $20 step 1  $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_PaletteCycles cycle $20 step 2  $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_PaletteCycles cycle $20 step 3  $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_PaletteCycles cycle $40 step 0  $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_PaletteCycles cycle $40 step 1  $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_PaletteCycles cycle $40 step 2  $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_PaletteCycles cycle $40 step 3  $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

# Level 1, Q1 (shown as LEVEL-1)

Block [`LevelInfoUW1`](../src/bank6/level-blocks-common-data.asm#L90) (`dat/LevelInfoUW1.dat`).

```
field                                     value
LevelInfo_StartRoomId                     $73
LevelInfo_TriforceRoomId                  $36
LevelInfo_BossRoomId                      $35
LevelInfo_LevelNumber                     $01
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $C9 $AC $89 $87  = 0:$C0,$90  1:$A0,$C0  2:$80,$90  3:$80,$70
LevelInfo_CellarRoomIdArray               $7F $FF $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_WorldFlagsAddr                  $06FF
LevelInfo_SubmenuMapRotation              $04
LevelInfo_StatusBarMapXOffset             $00
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $00 $08 $2D $3F $0D $18 $10 $00 $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $84 $05 $67 $FF $24 $FB $FB $20 $A3 $05 $67 $FF $FF $FF $67 $20 $C4 $03 $FB $FF $FB $FF
  record                                  $2084 5: $67 $FF $24 $FB $FB
  record                                  $20A3 5: $67 $FF $FF $FF $67
  record                                  $20C4 3: $FB $FF $FB
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $0C $1C $2C $0F $12 $1C $2C $0F $29 $27 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $0C $1C $2C $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $0C $1C $2C
  palette row 3                           $0F $12 $1C $2C
  palette row 4                           $0F $29 $27 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $0C $1C $2C
LevelInfo_PaletteCycles cycle $00 step 0  $0F $0C $1C $2C $0F $12 $1C $2C
LevelInfo_PaletteCycles cycle $00 step 1  $0F $0C $0C $1C $0F $11 $0C $1C
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $0C $0C $0F $02 $0C $0C
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $0C $1C $2C $0F $12 $1C $2C
LevelInfo_PaletteCycles cycle $40 step 1  $0F $0C $0C $1C $0F $11 $0C $1C
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $0C $0C $0F $02 $0C $0C
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $0C $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  . . . . . . . . . . . . . . . .
$10  . . . . . . . . . . . . . . . .
$20  . . # # . . . . . . . . . . . .
$30  . . . # . B T . . . . . . . . .
$40  . # # # # # . . . . . . . . . .
$50  . . # # # . . . . . . . . . . .
$60  . . . # . . . . . . . . . . . .
$70  . . # S # . . . . . . . . . . .
```

# Level 2, Q1 (shown as LEVEL-2)

Block [`LevelInfoUW2`](../src/bank6/level-blocks-common-data.asm#L94) (`dat/LevelInfoUW2.dat`).

```
field                                     value
LevelInfo_StartRoomId                     $7D
LevelInfo_TriforceRoomId                  $0D
LevelInfo_BossRoomId                      $0E
LevelInfo_LevelNumber                     $02
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $89 $D6 $26 $2C  = 0:$80,$90  1:$D0,$60  2:$20,$60  3:$20,$C0
LevelInfo_CellarRoomIdArray               $FF $FF $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_WorldFlagsAddr                  $06FF
LevelInfo_SubmenuMapRotation              $0A
LevelInfo_StatusBarMapXOffset             $B0
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $00 $00 $02 $83 $FF $7E $00 $00 $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $65 $03 $67 $FF $FB $20 $86 $02 $FF $FF $20 $A6 $02 $FF $FF $20 $C4 $04 $67 $FF $FF $67 $FF
  record                                  $2065 3: $67 $FF $FB
  record                                  $2086 2: $FF $FF
  record                                  $20A6 2: $FF $FF
  record                                  $20C4 4: $67 $FF $FF $67
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $02 $12 $22 $0F $16 $12 $22 $0F $29 $27 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $02 $12 $22 $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $02 $12 $22
  palette row 3                           $0F $16 $12 $22
  palette row 4                           $0F $29 $27 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $02 $12 $22
LevelInfo_PaletteCycles cycle $00 step 0  $0F $02 $12 $22 $0F $16 $12 $22
LevelInfo_PaletteCycles cycle $00 step 1  $0F $01 $02 $12 $0F $17 $02 $12
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $01 $02 $0F $06 $01 $02
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $02 $12 $22 $0F $16 $12 $22
LevelInfo_PaletteCycles cycle $40 step 1  $0F $01 $02 $12 $0F $17 $02 $12
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $01 $02 $0F $06 $01 $02
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $01 $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  . . . . . . . . . . . . . T B .
$10  . . . . . . . . . . . . . . # #
$20  . . . . . . . . . . . . . . # #
$30  . . . . . . . . . . . . . . # #
$40  . . . . . . . . . . . . . . # #
$50  . . . . . . . . . . . . . . # #
$60  . . . . . . . . . . . . # # # #
$70  . . . . . . . . . . . . . S # .
```

# Level 3, Q1 (shown as LEVEL-3)

Block [`LevelInfoUW3`](../src/bank6/level-blocks-common-data.asm#L98) (`dat/LevelInfoUW3.dat`).

```
field                                     value
LevelInfo_StartRoomId                     $7C
LevelInfo_TriforceRoomId                  $3D
LevelInfo_BossRoomId                      $4D
LevelInfo_LevelNumber                     $03
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $89 $D6 $26 $2C  = 0:$80,$90  1:$D0,$60  2:$20,$60  3:$20,$C0
LevelInfo_CellarRoomIdArray               $FF $FF $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_WorldFlagsAddr                  $06FF
LevelInfo_SubmenuMapRotation              $0C
LevelInfo_StatusBarMapXOffset             $C0
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $00 $0E $2C $3F $0D $1C $00 $00 $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $84 $04 $67 $FF $24 $FB $20 $A3 $05 $FF $FF $FF $FF $FF $20 $C3 $04 $67 $24 $FF $FB $FF
  record                                  $2084 4: $67 $FF $24 $FB
  record                                  $20A3 5: $FF $FF $FF $FF $FF
  record                                  $20C3 4: $67 $24 $FF $FB
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $0B $1B $2B $0F $16 $1B $2B $0F $29 $37 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $0B $1B $2B $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $0B $1B $2B
  palette row 3                           $0F $16 $1B $2B
  palette row 4                           $0F $29 $37 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $0B $1B $2B
LevelInfo_PaletteCycles cycle $00 step 0  $0F $0B $1B $2B $0F $16 $1B $2B
LevelInfo_PaletteCycles cycle $00 step 1  $0F $0B $0B $1B $0F $17 $0B $1B
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $0B $0B $0F $06 $0B $0B
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $0B $1B $2B $0F $16 $1B $2B
LevelInfo_PaletteCycles cycle $40 step 1  $0F $0B $0B $1B $0F $17 $0B $1B
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $0B $0B $0F $06 $0B $0B
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $0B $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  . . . . . . . . . . . . . . . .
$10  . . . . . . . . . . . . . . . .
$20  . . . . . . . . . . # # . . . .
$30  . . . . . . . . . . . # . T . .
$40  . . . . . . . . . # # # # B . .
$50  . . . . . . . . . # # # # # . .
$60  . . . . . . . . . # . # . . . .
$70  . . . . . . . . . . . # S . . .
```

# Level 4, Q1 (shown as LEVEL-4)

Block [`LevelInfoUW4`](../src/bank6/level-blocks-common-data.asm#L102) (`dat/LevelInfoUW4.dat`).

```
field                                     value
LevelInfo_StartRoomId                     $71
LevelInfo_TriforceRoomId                  $03
LevelInfo_BossRoomId                      $13
LevelInfo_LevelNumber                     $04
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $DC $99 $88 $89  = 0:$D0,$C0  1:$90,$90  2:$80,$80  3:$80,$90
LevelInfo_CellarRoomIdArray               $60 $FF $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_WorldFlagsAddr                  $06FF
LevelInfo_SubmenuMapRotation              $06
LevelInfo_StatusBarMapXOffset             $10
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $00 $00 $FD $B7 $D2 $C0 $00 $00 $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $64 $04 $FF $67 $FF $FF $20 $84 $03 $FF $FF $FB $20 $A4 $02 $FF $FB $20 $C4 $03 $FB $FF $67 $FF
  record                                  $2064 4: $FF $67 $FF $FF
  record                                  $2084 3: $FF $FF $FB
  record                                  $20A4 2: $FF $FB
  record                                  $20C4 3: $FB $FF $67
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $08 $18 $28 $0F $12 $18 $28 $0F $29 $27 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $0F $18 $28 $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $08 $18 $28
  palette row 3                           $0F $12 $18 $28
  palette row 4                           $0F $29 $27 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $0F $18 $28
LevelInfo_PaletteCycles cycle $00 step 0  $0F $08 $18 $28 $0F $12 $18 $28
LevelInfo_PaletteCycles cycle $00 step 1  $0F $08 $08 $18 $0F $11 $08 $18
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $08 $08 $0F $02 $08 $08
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $08 $18 $28 $0F $12 $18 $28
LevelInfo_PaletteCycles cycle $40 step 1  $0F $08 $08 $18 $0F $11 $08 $18
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $08 $08 $0F $02 $08 $08
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $08 $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  # # # T . . . . . . . . . . . .
$10  # . # B . . . . . . . . . . . .
$20  # # . . . . . . . . . . . . . .
$30  # # # . . . . . . . . . . . . .
$40  # . . . . . . . . . . . . . . .
$50  # # . . . . . . . . . . . . . .
$60  . # # . . . . . . . . . . . . .
$70  # S . . . . . . . . . . . . . .
```

# Level 5, Q1 (shown as LEVEL-5)

Block [`LevelInfoUW5`](../src/bank6/level-blocks-common-data.asm#L106) (`dat/LevelInfoUW5.dat`).

```
field                                     value
LevelInfo_StartRoomId                     $76
LevelInfo_TriforceRoomId                  $14
LevelInfo_BossRoomId                      $24
LevelInfo_LevelNumber                     $05
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $87 $C8 $8A $89  = 0:$80,$70  1:$C0,$80  2:$80,$A0  3:$80,$90
LevelInfo_CellarRoomIdArray               $07 $04 $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_WorldFlagsAddr                  $06FF
LevelInfo_SubmenuMapRotation              $02
LevelInfo_StatusBarMapXOffset             $F0
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $00 $00 $72 $A6 $EF $7F $00 $00 $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $64 $04 $FB $67 $FF $FB $20 $84 $04 $FF $67 $67 $FF $20 $A5 $03 $FB $FF $FF $20 $C4 $04 $67 $67 $FF $FF $FF
  record                                  $2064 4: $FB $67 $FF $FB
  record                                  $2084 4: $FF $67 $67 $FF
  record                                  $20A5 3: $FB $FF $FF
  record                                  $20C4 4: $67 $67 $FF $FF
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $0A $1A $2A $0F $16 $1A $2A $0F $29 $27 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $0A $1A $2A $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $0A $1A $2A
  palette row 3                           $0F $16 $1A $2A
  palette row 4                           $0F $29 $27 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $0A $1A $2A
LevelInfo_PaletteCycles cycle $00 step 0  $0F $0A $1A $2A $0F $16 $1A $2A
LevelInfo_PaletteCycles cycle $00 step 1  $0F $0A $0A $1A $0F $17 $0A $1A
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $0A $0A $0F $06 $0A $0A
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $0A $1A $2A $0F $16 $1A $2A
LevelInfo_PaletteCycles cycle $40 step 1  $0F $0A $0A $1A $0F $17 $0A $1A
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $0A $0A $0F $06 $0A $0A
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $0A $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  . . . . . # # . . . . . . . . .
$10  . . . . T . # # . . . . . . . .
$20  . . . . B # # # . . . . . . . .
$30  . . . . # . . # . . . . . . . .
$40  . . . . . . # # . . . . . . . .
$50  . . . . . # # # . . . . . . . .
$60  . . . . # # # # . . . . . . . .
$70  . . . . . . S # . . . . . . . .
```

# Level 6, Q1 (shown as LEVEL-6)

Block [`LevelInfoUW6`](../src/bank6/level-blocks-common-data.asm#L110) (`dat/LevelInfoUW6.dat`).

```
field                                     value
LevelInfo_StartRoomId                     $79
LevelInfo_TriforceRoomId                  $0C
LevelInfo_BossRoomId                      $1C
LevelInfo_LevelNumber                     $06
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $89 $D6 $26 $2C  = 0:$80,$90  1:$D0,$60  2:$20,$60  3:$20,$C0
LevelInfo_CellarRoomIdArray               $08 $75 $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_WorldFlagsAddr                  $06FF
LevelInfo_SubmenuMapRotation              $0D
LevelInfo_StatusBarMapXOffset             $C8
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $00 $7F $F1 $D3 $C0 $F0 $60 $00 $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $63 $06 $FB $FF $FF $FF $FF $FB $20 $83 $06 $FF $FF $FB $24 $FF $67 $20 $A3 $01 $FF $20 $C3 $03 $FF $FB $FF $FF
  record                                  $2063 6: $FB $FF $FF $FF $FF $FB
  record                                  $2083 6: $FF $FF $FB $24 $FF $67
  record                                  $20A3 1: $FF
  record                                  $20C3 3: $FF $FB $FF
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $08 $18 $28 $0F $16 $18 $28 $0F $29 $27 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $08 $18 $28 $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $08 $18 $28
  palette row 3                           $0F $16 $18 $28
  palette row 4                           $0F $29 $27 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $08 $18 $28
LevelInfo_PaletteCycles cycle $00 step 0  $0F $08 $18 $28 $0F $16 $18 $28
LevelInfo_PaletteCycles cycle $00 step 1  $0F $08 $08 $18 $0F $17 $08 $18
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $08 $08 $0F $06 $08 $08
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $08 $18 $28 $0F $16 $18 $28
LevelInfo_PaletteCycles cycle $40 step 1  $0F $08 $08 $18 $0F $17 $08 $18
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $08 $08 $0F $06 $08 $08
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $08 $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  . . . . . . . . . # # # T . . .
$10  . . . . . . . . # # # # B # . .
$20  . . . . . . . . # # . . # # . .
$30  . . . . . . . . # # # . # . . .
$40  . . . . . . . . # . . . . . . .
$50  . . . . . . . . # . . . . . . .
$60  . . . . . . . . # . # . . . . .
$70  . . . . . . . . # S # . . . . .
```

# Level 7, Q1 (shown as LEVEL-7)

Block [`LevelInfoUW7`](../src/bank6/level-blocks-common-data.asm#L114) (`dat/LevelInfoUW7.dat`).

```
field                                     value
LevelInfo_StartRoomId                     $79
LevelInfo_TriforceRoomId                  $2B
LevelInfo_BossRoomId                      $2A
LevelInfo_LevelNumber                     $07
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $89 $D6 $26 $2C  = 0:$80,$90  1:$D0,$60  2:$20,$60  3:$20,$C0
LevelInfo_CellarRoomIdArray               $7B $4A $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_WorldFlagsAddr                  $077F
LevelInfo_SubmenuMapRotation              $0D
LevelInfo_StatusBarMapXOffset             $C8
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $00 $7F $FF $B7 $E6 $C2 $82 $00 $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $63 $06 $FB $FF $67 $FF $FF $67 $20 $83 $04 $FF $FF $FF $67 $20 $A3 $04 $FF $FF $FB $FB $20 $C3 $06 $FF $FF $FF $67 $67 $67 $FF
  record                                  $2063 6: $FB $FF $67 $FF $FF $67
  record                                  $2083 4: $FF $FF $FF $67
  record                                  $20A3 4: $FF $FF $FB $FB
  record                                  $20C3 6: $FF $FF $FF $67 $67 $67
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $0A $1A $2A $0F $12 $1A $2A $0F $29 $27 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $0A $1A $2A $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $0A $1A $2A
  palette row 3                           $0F $12 $1A $2A
  palette row 4                           $0F $29 $27 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $0A $1A $2A
LevelInfo_PaletteCycles cycle $00 step 0  $0F $0A $1A $2A $0F $12 $1A $2A
LevelInfo_PaletteCycles cycle $00 step 1  $0F $0A $0A $1A $0F $11 $0A $1A
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $0A $0A $0F $02 $0A $0A
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $0A $1A $2A $0F $12 $1A $2A
LevelInfo_PaletteCycles cycle $40 step 1  $0F $0A $0A $1A $0F $11 $0A $1A
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $0A $0A $0F $02 $0A $0A
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $0A $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  . . . . . . . . . # # # # # . .
$10  . . . . . . . . # # . # # . . .
$20  . . . . . . . . # # B T . . . .
$30  . . . . . . . . # # # . . . . .
$40  . . . . . . . . # # . . . . . .
$50  . . . . . . . . # # # # . . . .
$60  . . . . . . . . # # # # # # . .
$70  . . . . . . . . # S # . . . . .
```

# Level 8, Q1 (shown as LEVEL-8)

Block [`LevelInfoUW8`](../src/bank6/level-blocks-common-data.asm#L118) (`dat/LevelInfoUW8.dat`).

```
field                                     value
LevelInfo_StartRoomId                     $7E
LevelInfo_TriforceRoomId                  $2C
LevelInfo_BossRoomId                      $3C
LevelInfo_LevelNumber                     $08
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $89 $D6 $26 $2C  = 0:$80,$90  1:$D0,$60  2:$20,$60  3:$20,$C0
LevelInfo_CellarRoomIdArray               $2F $0F $6F $FF $FF $FF $FF $FF $FF $FF
LevelInfo_WorldFlagsAddr                  $077F
LevelInfo_SubmenuMapRotation              $0A
LevelInfo_StatusBarMapXOffset             $B0
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $00 $18 $3D $5D $FF $55 $00 $00 $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $65 $03 $FB $FF $FB $20 $83 $05 $FB $FF $FB $FF $FB $20 $A3 $05 $67 $FF $FF $FF $FB $20 $C4 $04 $FB $FB $FF $FB $FF
  record                                  $2065 3: $FB $FF $FB
  record                                  $2083 5: $FB $FF $FB $FF $FB
  record                                  $20A3 5: $67 $FF $FF $FF $FB
  record                                  $20C4 4: $FB $FB $FF $FB
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $00 $10 $30 $0F $22 $10 $30 $0F $29 $27 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $00 $10 $30 $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $00 $10 $30
  palette row 3                           $0F $22 $10 $30
  palette row 4                           $0F $29 $27 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $00 step 0  $0F $00 $10 $30 $0F $22 $10 $30
LevelInfo_PaletteCycles cycle $00 step 1  $0F $00 $00 $10 $0F $12 $00 $10
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $00 $00 $0F $02 $00 $00
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $00 $10 $30 $0F $22 $10 $30
LevelInfo_PaletteCycles cycle $40 step 1  $0F $00 $00 $10 $0F $12 $00 $10
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $00 $00 $0F $02 $00 $00
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $00 $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  . . . . . . . . . . . . . . # .
$10  . . . . . . . . . . . . . # # #
$20  . . . . . . . . . . . . T . # .
$30  . . . . . . . . . . . # B # # #
$40  . . . . . . . . . . . # # # # .
$50  . . . . . . . . . . . . # # # #
$60  . . . . . . . . . . . . . . # .
$70  . . . . . . . . . . . . # # S #
```

# Level 9, Q1 (shown as LEVEL-9)

Block [`LevelInfoUW9`](../src/bank6/level-blocks-common-data.asm#L122) (`dat/LevelInfoUW9.dat`).

```
field                                     value
LevelInfo_StartRoomId                     $76
LevelInfo_TriforceRoomId                  $32
LevelInfo_BossRoomId                      $42
LevelInfo_LevelNumber                     $09
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $89 $D6 $26 $2C  = 0:$80,$90  1:$D0,$60  2:$20,$60  3:$20,$C0
LevelInfo_CellarRoomIdArray               $60 $70 $72 $75 $67 $77 $00 $4F $FF $FF
LevelInfo_WorldFlagsAddr                  $077F
LevelInfo_SubmenuMapRotation              $04
LevelInfo_StatusBarMapXOffset             $00
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $7C $FF $EE $FF $FF $EE $FF $7C $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $62 $08 $FB $FF $FF $FF $FF $FF $FF $FB $20 $82 $08 $FF $FF $67 $FF $FF $67 $FF $FF $20 $A2 $48 $FF $20 $C3 $06 $FF $67 $FF $FF $67 $FF $FF
  record                                  $2062 8: $FB $FF $FF $FF $FF $FF $FF $FB
  record                                  $2082 8: $FF $FF $67 $FF $FF $67 $FF $FF
  record                                  $20A2 8: $FF repeated
  record                                  $20C3 6: $FF $67 $FF $FF $67 $FF
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $00 $10 $30 $0F $16 $10 $30 $0F $29 $27 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $0F $10 $30 $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $00 $10 $30
  palette row 3                           $0F $16 $10 $30
  palette row 4                           $0F $29 $27 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $0F $10 $30
LevelInfo_PaletteCycles cycle $00 step 0  $0F $00 $10 $30 $0F $16 $10 $30
LevelInfo_PaletteCycles cycle $00 step 1  $0F $00 $00 $10 $0F $17 $00 $10
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $00 $00 $0F $06 $00 $00
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $00 $10 $30 $0F $16 $10 $30
LevelInfo_PaletteCycles cycle $40 step 1  $0F $00 $00 $10 $0F $17 $00 $10
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $00 $00 $0F $06 $00 $00
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $00 $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  . # # # # # # . . . . . . . . .
$10  # # # # # # # # . . . . . . . .
$20  # # # # # # # # . . . . . . . .
$30  # # T # # . # # . . . . . . . .
$40  # # B # # # # # . . . . . . . .
$50  # # # # # # # # . . . . . . . .
$60  . # # # # # # . . . . . . . . .
$70  . # . # # . S . . . . . . . . .
```

# Level 1, Q2 (shown as LEVEL-1)

Block [`LevelInfoUW1`](../src/bank6/level-blocks-common-data.asm#L90) (`dat/LevelInfoUW1.dat`) patched with [`LevelInfoUWQ2Replacements1`](../src/bank6/quest2-replacements.asm#L84).

```
field                                     value
LevelInfo_StartRoomId                     $77
LevelInfo_TriforceRoomId                  $08
LevelInfo_BossRoomId                      $07
LevelInfo_LevelNumber                     $01
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $C9 $AC $89 $B7  = 0:$C0,$90  1:$A0,$C0  2:$80,$90  3:$B0,$70
LevelInfo_CellarRoomIdArray               $28 $FF $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_WorldFlagsAddr                  $06FF
LevelInfo_SubmenuMapRotation              $00
LevelInfo_StatusBarMapXOffset             $E0
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $00 $00 $00 $FF $DB $00 $00 $00 $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $65 $42 $FF $20 $85 $02 $FF $FB $20 $A5 $02 $FF $67 $20 $C5 $42 $FF $FF
  record                                  $2065 2: $FF repeated
  record                                  $2085 2: $FF $FB
  record                                  $20A5 2: $FF $67
  record                                  $20C5 2: $FF repeated
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $0C $1C $2C $0F $12 $1C $2C $0F $29 $27 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $0C $1C $2C $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $0C $1C $2C
  palette row 3                           $0F $12 $1C $2C
  palette row 4                           $0F $29 $27 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $0C $1C $2C
LevelInfo_PaletteCycles cycle $00 step 0  $0F $0C $1C $2C $0F $12 $1C $2C
LevelInfo_PaletteCycles cycle $00 step 1  $0F $0C $0C $1C $0F $11 $0C $1C
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $0C $0C $0F $02 $0C $0C
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $0C $1C $2C $0F $12 $1C $2C
LevelInfo_PaletteCycles cycle $40 step 1  $0F $0C $0C $1C $0F $11 $0C $1C
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $0C $0C $0F $02 $0C $0C
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $0C $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  . . . . . . . B T . . . . . . .
$10  . . . . . . . # # . . . . . . .
$20  . . . . . . . # . . . . . . . .
$30  . . . . . . . # # . . . . . . .
$40  . . . . . . . # # . . . . . . .
$50  . . . . . . . # . . . . . . . .
$60  . . . . . . . # # . . . . . . .
$70  . . . . . . . S # . . . . . . .
```

# Level 2, Q2 (shown as LEVEL-3)

Block [`LevelInfoUW2`](../src/bank6/level-blocks-common-data.asm#L94) (`dat/LevelInfoUW2.dat`) patched with [`LevelInfoUWQ2Replacements2`](../src/bank6/quest2-replacements.asm#L96).

```
field                                     value
LevelInfo_StartRoomId                     $75
LevelInfo_TriforceRoomId                  $20
LevelInfo_BossRoomId                      $30
LevelInfo_LevelNumber                     $03
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $C9 $AC $89 $87  = 0:$C0,$90  1:$A0,$C0  2:$80,$90  3:$80,$70
LevelInfo_CellarRoomIdArray               $56 $FF $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_WorldFlagsAddr                  $06FF
LevelInfo_SubmenuMapRotation              $05
LevelInfo_StatusBarMapXOffset             $00
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $00 $30 $00 $00 $00 $00 $7F $03 $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $67 $01 $FB $20 $82 $01 $FF $20 $87 $C3 $FF $20 $C8 $01 $FF $FF
  record                                  $2067 1: $FB
  record                                  $2082 1: $FF
  record                                  $2087 down 3: $FF repeated
  record                                  $20C8 1: $FF
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $02 $12 $22 $0F $16 $12 $22 $0F $29 $27 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $02 $12 $22 $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $02 $12 $22
  palette row 3                           $0F $16 $12 $22
  palette row 4                           $0F $29 $27 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $02 $12 $22
LevelInfo_PaletteCycles cycle $00 step 0  $0F $02 $12 $22 $0F $16 $12 $22
LevelInfo_PaletteCycles cycle $00 step 1  $0F $01 $02 $12 $0F $17 $02 $12
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $01 $02 $0F $06 $01 $02
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $02 $12 $22 $0F $16 $12 $22
LevelInfo_PaletteCycles cycle $40 step 1  $0F $01 $02 $12 $0F $17 $02 $12
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $01 $02 $0F $06 $01 $02
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $01 $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  . . . . . . . . . . . . . . . .
$10  . . . . . # . . . . . . . . . .
$20  T . . . . # . . . . . . . . . .
$30  B . . . . # . . . . . . . . . .
$40  . . . . . # . . . . . . . . . .
$50  . . . . . # . . . . . . . . . .
$60  . . . . . # # . . . . . . . . .
$70  . . . . . S # . . . . . . . . .
```

# Level 3, Q2 (shown as LEVEL-2)

Block [`LevelInfoUW3`](../src/bank6/level-blocks-common-data.asm#L98) (`dat/LevelInfoUW3.dat`) patched with [`LevelInfoUWQ2Replacements3`](../src/bank6/quest2-replacements.asm#L107).

```
field                                     value
LevelInfo_StartRoomId                     $79
LevelInfo_TriforceRoomId                  $1B
LevelInfo_BossRoomId                      $2B
LevelInfo_LevelNumber                     $02
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $C9 $AC $89 $37  = 0:$C0,$90  1:$A0,$C0  2:$80,$90  3:$30,$70
LevelInfo_CellarRoomIdArray               $09 $0B $FF $FF $FF $FF $FF $FF $FF $FF
LevelInfo_WorldFlagsAddr                  $06FF
LevelInfo_SubmenuMapRotation              $0D
LevelInfo_StatusBarMapXOffset             $C8
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $00 $00 $7F $EC $7F $00 $00 $00 $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $64 $03 $FB $FF $FB $20 $84 $03 $FF $67 $FF $20 $A4 $43 $FF $20 $C4 $03 $FF $24 $FF $FF
  record                                  $2064 3: $FB $FF $FB
  record                                  $2084 3: $FF $67 $FF
  record                                  $20A4 3: $FF repeated
  record                                  $20C4 3: $FF $24 $FF
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $0B $1B $2B $0F $16 $1B $2B $0F $29 $37 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $0B $1B $2B $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $0B $1B $2B
  palette row 3                           $0F $16 $1B $2B
  palette row 4                           $0F $29 $37 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $0B $1B $2B
LevelInfo_PaletteCycles cycle $00 step 0  $0F $0B $1B $2B $0F $16 $1B $2B
LevelInfo_PaletteCycles cycle $00 step 1  $0F $0B $0B $1B $0F $17 $0B $1B
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $0B $0B $0F $06 $0B $0B
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $0B $1B $2B $0F $16 $1B $2B
LevelInfo_PaletteCycles cycle $40 step 1  $0F $0B $0B $1B $0F $17 $0B $1B
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $0B $0B $0F $06 $0B $0B
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $0B $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  . . . . . . . . . . # . . . . .
$10  . . . . . . . . . # # T . . . .
$20  . . . . . . . . . # # B . . . .
$30  . . . . . . . . . # . # . . . .
$40  . . . . . . . . . # # # . . . .
$50  . . . . . . . . . # # # . . . .
$60  . . . . . . . . . # . # . . . .
$70  . . . . . . . . . S . # . . . .
```

# Level 4, Q2 (shown as LEVEL-5)

Block [`LevelInfoUW4`](../src/bank6/level-blocks-common-data.asm#L102) (`dat/LevelInfoUW4.dat`) patched with [`LevelInfoUWQ2Replacements4`](../src/bank6/quest2-replacements.asm#L119).

```
field                                     value
LevelInfo_StartRoomId                     $72
LevelInfo_TriforceRoomId                  $00
LevelInfo_BossRoomId                      $10
LevelInfo_LevelNumber                     $05
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $C9 $AC $89 $86  = 0:$C0,$90  1:$A0,$C0  2:$80,$90  3:$80,$60
LevelInfo_CellarRoomIdArray               $21 $58 $7A $FF $FF $FF $FF $FF $FF $FF
LevelInfo_WorldFlagsAddr                  $06FF
LevelInfo_SubmenuMapRotation              $06
LevelInfo_StatusBarMapXOffset             $10
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $00 $00 $CF $DB $F3 $00 $00 $00 $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $64 $43 $FF $20 $85 $02 $FB $FF $20 $A4 $02 $FF $67 $20 $C4 $43 $FF $FF
  record                                  $2064 3: $FF repeated
  record                                  $2085 2: $FB $FF
  record                                  $20A4 2: $FF $67
  record                                  $20C4 3: $FF repeated
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $08 $18 $28 $0F $12 $18 $28 $0F $29 $27 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $0F $18 $28 $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $08 $18 $28
  palette row 3                           $0F $12 $18 $28
  palette row 4                           $0F $29 $27 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $0F $18 $28
LevelInfo_PaletteCycles cycle $00 step 0  $0F $08 $18 $28 $0F $12 $18 $28
LevelInfo_PaletteCycles cycle $00 step 1  $0F $08 $08 $18 $0F $11 $08 $18
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $08 $08 $0F $02 $08 $08
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $08 $18 $28 $0F $12 $18 $28
LevelInfo_PaletteCycles cycle $40 step 1  $0F $08 $08 $18 $0F $11 $08 $18
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $08 $08 $0F $02 $08 $08
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $08 $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  T # # . . . . . . . . . . . . .
$10  B # # . . . . . . . . . . . . .
$20  . . # . . . . . . . . . . . . .
$30  . # # . . . . . . . . . . . . .
$40  # # . . . . . . . . . . . . . .
$50  # . . . . . . . . . . . . . . .
$60  # # # . . . . . . . . . . . . .
$70  # # S . . . . . . . . . . . . .
```

# Level 5, Q2 (shown as LEVEL-4)

Block [`LevelInfoUW5`](../src/bank6/level-blocks-common-data.asm#L106) (`dat/LevelInfoUW5.dat`) patched with [`LevelInfoUWQ2Replacements5`](../src/bank6/quest2-replacements.asm#L131).

```
field                                     value
LevelInfo_StartRoomId                     $7D
LevelInfo_TriforceRoomId                  $4F
LevelInfo_BossRoomId                      $5F
LevelInfo_LevelNumber                     $04
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $C9 $AC $89 $87  = 0:$C0,$90  1:$A0,$C0  2:$80,$90  3:$80,$70
LevelInfo_CellarRoomIdArray               $0F $6A $7F $FF $FF $FF $FF $FF $FF $FF
LevelInfo_WorldFlagsAddr                  $06FF
LevelInfo_SubmenuMapRotation              $0A
LevelInfo_StatusBarMapXOffset             $B0
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $00 $00 $FF $FF $E7 $7E $00 $00 $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $64 $04 $FF $FF $FF $FB $20 $84 $04 $FF $FF $67 $FF $20 $A4 $04 $FF $FF $FB $FF $20 $C4 $04 $FF $FF $FF $67 $FF
  record                                  $2064 4: $FF $FF $FF $FB
  record                                  $2084 4: $FF $FF $67 $FF
  record                                  $20A4 4: $FF $FF $FB $FF
  record                                  $20C4 4: $FF $FF $FF $67
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $0A $1A $2A $0F $16 $1A $2A $0F $29 $27 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $0A $1A $2A $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $0A $1A $2A
  palette row 3                           $0F $16 $1A $2A
  palette row 4                           $0F $29 $27 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $0A $1A $2A
LevelInfo_PaletteCycles cycle $00 step 0  $0F $0A $1A $2A $0F $16 $1A $2A
LevelInfo_PaletteCycles cycle $00 step 1  $0F $0A $0A $1A $0F $17 $0A $1A
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $0A $0A $0F $06 $0A $0A
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $0A $1A $2A $0F $16 $1A $2A
LevelInfo_PaletteCycles cycle $40 step 1  $0F $0A $0A $1A $0F $17 $0A $1A
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $0A $0A $0F $06 $0A $0A
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $0A $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  . . . . . . . . . . . . # # # .
$10  . . . . . . . . . . . . # # # #
$20  . . . . . . . . . . . . # # # #
$30  . . . . . . . . . . . . # # . #
$40  . . . . . . . . . . . . # # . T
$50  . . . . . . . . . . . . # # # B
$60  . . . . . . . . . . . . # # # #
$70  . . . . . . . . . . . . # S # .
```

# Level 6, Q2 (shown as LEVEL-6)

Block [`LevelInfoUW6`](../src/bank6/level-blocks-common-data.asm#L110) (`dat/LevelInfoUW6.dat`) patched with [`LevelInfoUWQ2Replacements6`](../src/bank6/quest2-replacements.asm#L145).

```
field                                     value
LevelInfo_StartRoomId                     $74
LevelInfo_TriforceRoomId                  $16
LevelInfo_BossRoomId                      $26
LevelInfo_LevelNumber                     $06
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $49 $79 $89 $56  = 0:$40,$90  1:$70,$90  2:$80,$90  3:$50,$60
LevelInfo_CellarRoomIdArray               $03 $73 $46 $FF $FF $FF $FF $FF $FF $FF
LevelInfo_WorldFlagsAddr                  $06FF
LevelInfo_SubmenuMapRotation              $04
LevelInfo_StatusBarMapXOffset             $00
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $00 $04 $0C $7E $FF $80 $F0 $00 $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $65 $03 $FB $FF $67 $20 $68 $C2 $FF $20 $86 $C3 $FF $20 $85 $83 $FF $FF $67 $20 $A3 $02 $FB $FF $FF
  record                                  $2065 3: $FB $FF $67
  record                                  $2068 down 2: $FF repeated
  record                                  $2086 down 3: $FF repeated
  record                                  $2085 down 3: $FF $FF $67
  record                                  $20A3 2: $FB $FF
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $08 $18 $28 $0F $16 $18 $28 $0F $29 $27 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $08 $18 $28 $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $08 $18 $28
  palette row 3                           $0F $16 $18 $28
  palette row 4                           $0F $29 $27 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $08 $18 $28
LevelInfo_PaletteCycles cycle $00 step 0  $0F $08 $18 $28 $0F $16 $18 $28
LevelInfo_PaletteCycles cycle $00 step 1  $0F $08 $08 $18 $0F $17 $08 $18
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $08 $08 $0F $06 $08 $08
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $08 $18 $28 $0F $16 $18 $28
LevelInfo_PaletteCycles cycle $40 step 1  $0F $08 $08 $18 $0F $17 $08 $18
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $08 $08 $0F $06 $08 $08
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $08 $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  . . . . # # # . . . . . . . . .
$10  . . . # # . T . . . . . . . . .
$20  . . . # # . B . . . . . . . . .
$30  . . . # # . # . . . . . . . . .
$40  . . # # # . . . . . . . . . . .
$50  . # # # # . . . . . . . . . . .
$60  . . . # # . . . . . . . . . . .
$70  . . . . S . . . . . . . . . . .
```

# Level 7, Q2 (shown as LEVEL-8)

Block [`LevelInfoUW7`](../src/bank6/level-blocks-common-data.asm#L114) (`dat/LevelInfoUW7.dat`) patched with [`LevelInfoUWQ2Replacements7`](../src/bank6/quest2-replacements.asm#L158).

```
field                                     value
LevelInfo_StartRoomId                     $7F
LevelInfo_TriforceRoomId                  $2D
LevelInfo_BossRoomId                      $3D
LevelInfo_LevelNumber                     $08
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $C9 $AC $89 $79  = 0:$C0,$90  1:$A0,$C0  2:$80,$90  3:$70,$90
LevelInfo_CellarRoomIdArray               $02 $03 $04 $05 $20 $21 $26 $2B $2C $FF
LevelInfo_WorldFlagsAddr                  $077F
LevelInfo_SubmenuMapRotation              $0C
LevelInfo_StatusBarMapXOffset             $C0
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $FE $FE $82 $82 $82 $BE $80 $FF $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $62 $C3 $FF $20 $63 $C3 $FF $20 $64 $45 $67 $20 $69 $C4 $FF $20 $87 $C2 $FF $20 $C2 $46 $67 $FF
  record                                  $2062 down 3: $FF repeated
  record                                  $2063 down 3: $FF repeated
  record                                  $2064 5: $67 repeated
  record                                  $2069 down 4: $FF repeated
  record                                  $2087 down 2: $FF repeated
  record                                  $20C2 6: $67 repeated
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $0A $1A $2A $0F $12 $1A $2A $0F $29 $27 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $0A $1A $2A $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $0A $1A $2A
  palette row 3                           $0F $12 $1A $2A
  palette row 4                           $0F $29 $27 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $0A $1A $2A
LevelInfo_PaletteCycles cycle $00 step 0  $0F $0A $1A $2A $0F $12 $1A $2A
LevelInfo_PaletteCycles cycle $00 step 1  $0F $0A $0A $1A $0F $11 $0A $1A
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $0A $0A $0F $02 $0A $0A
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $0A $1A $2A $0F $12 $1A $2A
LevelInfo_PaletteCycles cycle $40 step 1  $0F $0A $0A $1A $0F $11 $0A $1A
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $0A $0A $0F $02 $0A $0A
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $0A $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  . . . . . . . . # # # # # # # #
$10  . . . . . . . . # # . . . . . #
$20  . . . . . . . . # # . . . T . #
$30  . . . . . . . . # # . . . B . #
$40  . . . . . . . . # # . . . # . #
$50  . . . . . . . . # # . . . # . #
$60  . . . . . . . . # # # # # # . #
$70  . . . . . . . . . . . . . . . S
```

# Level 8, Q2 (shown as LEVEL-7)

Block [`LevelInfoUW8`](../src/bank6/level-blocks-common-data.asm#L118) (`dat/LevelInfoUW8.dat`) patched with [`LevelInfoUWQ2Replacements8`](../src/bank6/quest2-replacements.asm#L170).

```
field                                     value
LevelInfo_StartRoomId                     $79
LevelInfo_TriforceRoomId                  $1B
LevelInfo_BossRoomId                      $1C
LevelInfo_LevelNumber                     $07
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $C9 $AC $89 $57  = 0:$C0,$90  1:$A0,$C0  2:$80,$90  3:$50,$70
LevelInfo_CellarRoomIdArray               $27 $30 $37 $60 $67 $70 $FF $FF $FF $FF
LevelInfo_WorldFlagsAddr                  $077F
LevelInfo_SubmenuMapRotation              $0C
LevelInfo_StatusBarMapXOffset             $C0
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $01 $01 $7D $5D $5D $41 $7F $00 $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $64 $45 $FB $20 $84 $05 $FF $FB $FB $24 $FF $20 $A4 $43 $FF $20 $A8 $01 $FF $20 $C2 $46 $FB $20 $C8 $01 $FF $FF
  record                                  $2064 5: $FB repeated
  record                                  $2084 5: $FF $FB $FB $24 $FF
  record                                  $20A4 3: $FF repeated
  record                                  $20A8 1: $FF
  record                                  $20C2 6: $FB repeated
  record                                  $20C8 1: $FF
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $00 $10 $30 $0F $22 $10 $30 $0F $29 $27 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $00 $10 $30 $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $00 $10 $30
  palette row 3                           $0F $22 $10 $30
  palette row 4                           $0F $29 $27 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $00 step 0  $0F $00 $10 $30 $0F $22 $10 $30
LevelInfo_PaletteCycles cycle $00 step 1  $0F $00 $00 $10 $0F $12 $00 $10
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $00 $00 $0F $02 $00 $00
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $00 $10 $30 $0F $22 $10 $30
LevelInfo_PaletteCycles cycle $40 step 1  $0F $00 $00 $10 $0F $12 $00 $10
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $00 $00 $0F $02 $00 $00
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $00 $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  . . . . . . . . . . . . . . . .
$10  . . . . . . . . . . # T B # # .
$20  . . . . . . . . . . # . . . # .
$30  . . . . . . . . . . # # # . # .
$40  . . . . . . . . . . # # # . # .
$50  . . . . . . . . . . # # # . # .
$60  . . . . . . . . . . . . . . # .
$70  . . . . . . . . # S # # # # # .
```

# Level 9, Q2 (shown as LEVEL-9)

Block [`LevelInfoUW9`](../src/bank6/level-blocks-common-data.asm#L122) (`dat/LevelInfoUW9.dat`) patched with [`LevelInfoUWQ2Replacements9`](../src/bank6/quest2-replacements.asm#L183).

```
field                                     value
LevelInfo_StartRoomId                     $74
LevelInfo_TriforceRoomId                  $07
LevelInfo_BossRoomId                      $17
LevelInfo_LevelNumber                     $09
LevelInfo_StartY                          $DD
LevelInfo_FoeCounts                       $03 $05 $06 $08
LevelInfo_ShortcutOrItemPosArray          $C9 $AC $89 $B6  = 0:$C0,$90  1:$A0,$C0  2:$80,$90  3:$B0,$60
LevelInfo_CellarRoomIdArray               $71 $72 $75 $76 $77 $FF $FF $FF $FF $FF
LevelInfo_WorldFlagsAddr                  $077F
LevelInfo_SubmenuMapRotation              $04
LevelInfo_StatusBarMapXOffset             $00
LevelInfo_SubmenuMapMask                  $00 $00 $00 $00 $CC $DE $76 $7F $7F $76 $DE $CC $00 $00 $00 $00
LevelInfo_StatusBarMapTransferBuf         $20 $62 $48 $FF $20 $64 $44 $FB $20 $83 $46 $FB $20 $84 $44 $FF $20 $A2 $08 $FF $FF $FB $FF $FF $FB $FF $FF $20 $C3 $46 $67 $20 $C5 $42 $FF $FF
  record                                  $2062 8: $FF repeated
  record                                  $2064 4: $FB repeated
  record                                  $2083 6: $FB repeated
  record                                  $2084 4: $FF repeated
  record                                  $20A2 8: $FF $FF $FB $FF $FF $FB $FF $FF
  record                                  $20C3 6: $67 repeated
  record                                  $20C5 2: $FF repeated
LevelInfo_PalettesTransferBuf             $3F $00 $20 $0F $30 $00 $12 $0F $16 $27 $36 $0F $00 $10 $30 $0F $16 $10 $30 $0F $29 $27 $17 $0F $02 $22 $30 $0F $16 $27 $30 $0F $0F $10 $30 $FF
  palette row 0                           $0F $30 $00 $12
  palette row 1                           $0F $16 $27 $36
  palette row 2                           $0F $00 $10 $30
  palette row 3                           $0F $16 $10 $30
  palette row 4                           $0F $29 $27 $17
  palette row 5                           $0F $02 $22 $30
  palette row 6                           $0F $16 $27 $30
  palette row 7                           $0F $0F $10 $30
LevelInfo_PaletteCycles cycle $00 step 0  $0F $00 $10 $30 $0F $16 $10 $30
LevelInfo_PaletteCycles cycle $00 step 1  $0F $00 $00 $10 $0F $17 $00 $10
LevelInfo_PaletteCycles cycle $00 step 2  $0F $0F $00 $00 $0F $06 $00 $00
LevelInfo_PaletteCycles cycle $00 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $20 step 0  $0F $00 $10 $30 $0F $00 $10 $30
LevelInfo_PaletteCycles cycle $20 step 1  $0F $00 $00 $10 $0F $00 $00 $10
LevelInfo_PaletteCycles cycle $20 step 2  $0F $0F $00 $00 $0F $0F $00 $00
LevelInfo_PaletteCycles cycle $20 step 3  $0F $0F $0F $0F $0F $0F $0F $0F
LevelInfo_PaletteCycles cycle $40 step 0  $0F $00 $10 $30 $0F $16 $10 $30
LevelInfo_PaletteCycles cycle $40 step 1  $0F $00 $00 $10 $0F $17 $00 $10
LevelInfo_PaletteCycles cycle $40 step 2  $0F $0F $00 $00 $0F $06 $00 $00
LevelInfo_PaletteCycles cycle $40 step 3  $0F $0F $0F $00 $0F $0F $0F $0F
LevelInfo_DeathPaletteSeries step 0       $0F $06 $17 $16 $0F $06 $17 $16
LevelInfo_DeathPaletteSeries step 1       $0F $07 $06 $16 $0F $07 $06 $16
LevelInfo_DeathPaletteSeries step 2       $0F $0F $07 $06 $0F $0F $07 $06
LevelInfo_DeathPaletteSeries step 3       $0F $0F $0F $0F $0F $0F $0F $0F
```

Rooms on the level's submenu map (`#`), by room id; `S` start, `T` Triforce, `B` boss room
(marked also when the map does not show them):

```
     0 1 2 3 4 5 6 7 8 9 A B C D E F
$00  # # . . . . # T . . . . . . . .
$10  # # # # # # # B . . . . . . . .
$20  . . # # # # . . . . . . . . . .
$30  . # # # # # # . . . . . . . . .
$40  # # . # # . # # . . . . . . . .
$50  # # # # # # # # . . . . . . . .
$60  . # # # # # # . . . . . . . . .
$70  . . . # S . . . . . . . . . . .
```
