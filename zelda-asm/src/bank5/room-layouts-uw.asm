; ============================================================================
; FILE:     bank5/room-layouts-uw.asm    BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 4410-4535
; SUBJECTS: (none)
; TOPIC:    Dungeon wall tiles, door faces, layouts, column heaps, cellars (data)
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0030 WallTileList         data
;   L0043 DoorFaceTilesE       data  5 sets of 12 bytes laying out door faces facing E.
;   L0054 DoorFaceTilesW       data  5 sets of 12 bytes laying out door faces facing W.
;   L0065 DoorFaceTilesS       data  5 sets of 12 bytes laying out door faces facing S.
;   L0076 DoorFaceTilesN       data  5 sets of 12 bytes laying out door faces facing N.
;   L0086 RoomLayoutsUW        data
;   L0089 ColumnHeapUW0        data
;   L0093 ColumnHeapUW1        data
;   L0099 ColumnHeapUW2        data
;   L0104 ColumnHeapUW3        data
;   L0109 ColumnHeapUW4        data
;   L0114 ColumnHeapUW5        data
;   L0119 ColumnHeapUW6        data
;   L0124 ColumnHeapUW7        data
;   L0129 ColumnHeapUW8        data
;   L0135 ColumnHeapUW9        data
;   L0142 RoomLayoutUWCellar0  data
;   L0146 RoomLayoutUWCellar1  data
;   L0150 ColumnHeapUWCellar   data
; ============================================================================

WallTileList:
    .BYTE $E0, $F5, $F5, $F5, $F5, $B8, $F5, $D4
    .BYTE $F5, $F5, $F5, $C4, $DE, $DE, $BC, $C8
    .BYTE $DE, $BC, $DE, $DE, $F5, $DC, $C4, $DE
    .BYTE $C8, $DE, $BC, $C8, $DE, $DE, $F5, $DC
    .BYTE $DC, $00, $C0, $D0, $DC, $00, $F5, $DC
    .BYTE $CC, $00, $F5, $DC, $DC, $00, $F5, $CC
    .BYTE $D0, $00, $F5, $DC, $DC, $00, $C0, $D0
    .BYTE $DC, $00, $F5, $DC, $CC, $00, $F5, $DC
    .BYTE $DC, $00, $D8, $CC, $D0, $00, $F5, $DC
    .BYTE $DC, $00, $F5, $DC, $DC, $00

; 5 sets of 12 bytes laying out door faces facing E.
DoorFaceTilesE:
    .BYTE $88, $74, $8A, $24, $87, $87, $75, $89
    .BYTE $24, $8B, $87, $87, $88, $A4, $8A, $A6
    .BYTE $87, $87, $A5, $89, $A7, $8B, $87, $87
    .BYTE $88, $AC, $8A, $AE, $87, $87, $AD, $89
    .BYTE $AF, $8B, $87, $87, $DF, $DF, $DF, $DF
    .BYTE $F5, $F5, $DF, $DF, $DF, $DF, $F5, $F5
    .BYTE $DF, $24, $DF, $92, $F5, $F5, $24, $DF
    .BYTE $93, $DF, $F5, $F5

; 5 sets of 12 bytes laying out door faces facing W.
DoorFaceTilesW:
    .BYTE $82, $82, $83, $24, $85, $76, $82, $82
    .BYTE $24, $84, $77, $86, $82, $82, $83, $A0
    .BYTE $85, $A2, $82, $82, $A1, $84, $A3, $86
    .BYTE $82, $82, $83, $AC, $85, $AE, $82, $82
    .BYTE $AD, $84, $AF, $86, $F5, $F5, $DE, $DE
    .BYTE $DE, $DE, $F5, $F5, $DE, $DE, $DE, $DE
    .BYTE $F5, $F5, $DE, $90, $DE, $24, $F5, $F5
    .BYTE $91, $DE, $24, $DE

; 5 sets of 12 bytes laying out door faces facing S.
DoorFaceTilesS:
    .BYTE $7E, $7F, $7D, $76, $24, $7D, $74, $24
    .BYTE $7D, $80, $81, $7D, $7E, $7F, $7D, $9C
    .BYTE $9D, $7D, $9E, $9F, $7D, $80, $81, $7D
    .BYTE $7E, $7F, $7D, $A8, $A9, $7D, $AA, $AB
    .BYTE $7D, $80, $81, $7D, $DD, $DD, $F5, $DD
    .BYTE $DD, $F5, $DD, $DD, $F5, $DD, $DD, $F5
    .BYTE $DD, $DD, $F5, $24, $8E, $F5, $24, $8F
    .BYTE $F5, $DD, $DD, $F5

; 5 sets of 12 bytes laying out door faces facing N.
DoorFaceTilesN:
    .BYTE $78, $79, $7A, $78, $24, $77, $78, $24
    .BYTE $75, $78, $7B, $7C, $78, $79, $7A, $78
    .BYTE $98, $99, $78, $9A, $9B, $78, $7B, $7C
    .BYTE $78, $79, $7A, $78, $A8, $A9, $78, $AA
    .BYTE $AB, $78, $7B, $7C, $F5, $DC, $DC, $F5
    .BYTE $DC, $DC, $F5, $DC, $DC, $F5, $DC, $DC
    .BYTE $F5, $DC, $DC, $F5, $8C, $24, $F5, $8D
    .BYTE $24, $F5, $DC, $DC

RoomLayoutsUW:
    .INCBIN "dat/RoomLayoutsUW.dat"

ColumnHeapUW0:
    .BYTE $E1, $80, $C1, $00, $01, $A0, $04, $A0
    .BYTE $01, $A0, $31, $90, $21, $90, $21, $00

ColumnHeapUW1:
    .BYTE $81, $00, $01, $04, $81, $00, $01, $00
    .BYTE $01, $00, $81, $00, $21, $00, $81, $40
    .BYTE $81, $20, $01, $00, $81, $00, $04, $00
    .BYTE $01, $00, $01, $90, $41, $80

ColumnHeapUW2:
    .BYTE $D1, $80, $21, $00, $91, $20, $91, $00
    .BYTE $01, $00, $11, $D0, $81, $00, $41, $B1
    .BYTE $00

ColumnHeapUW3:
    .BYTE $91, $00, $A1, $00, $21, $B1, $06, $91
    .BYTE $00, $B1, $00, $81, $00, $21, $90, $17
    .BYTE $02

ColumnHeapUW4:
    .BYTE $91, $17, $01, $17, $81, $16, $B1, $07
    .BYTE $91, $06, $31, $87, $31, $17, $81, $00
    .BYTE $01, $03, $81, $00, $01, $02, $01

ColumnHeapUW5:
    .BYTE $80, $01, $10, $11, $83, $47, $83, $41
    .BYTE $03, $D1, $03, $90, $01, $10, $91, $00
    .BYTE $11, $90, $17, $03, $11, $C0, $11

ColumnHeapUW6:
    .BYTE $E0, $E6, $81, $C6, $81, $86, $01, $06
    .BYTE $01, $06, $81, $06, $21, $86, $01, $26
    .BYTE $01

ColumnHeapUW7:
    .BYTE $86, $41, $86, $01, $46, $D1, $02, $A6
    .BYTE $01, $26, $81, $56, $81, $16, $11, $16
    .BYTE $E7, $E5

ColumnHeapUW8:
    .BYTE $81, $10, $01, $10, $81, $27, $A1, $06
    .BYTE $21, $80, $01, $10, $11, $82, $47, $82
    .BYTE $41, $02, $80, $01, $10, $A1, $02, $A1
    .BYTE $03, $21

ColumnHeapUW9:
    .BYTE $80, $27, $21, $81, $00, $02, $11, $00
    .BYTE $81, $00, $03, $11, $00, $81, $02, $81
    .BYTE $02, $01, $02, $01, $02, $05, $C1, $15
    .BYTE $B1, $25, $81, $03, $81, $03, $01, $03
    .BYTE $01, $03, $05

RoomLayoutUWCellar0:
    .BYTE $04, $04, $00, $01, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $01, $00, $04, $04

RoomLayoutUWCellar1:
    .BYTE $04, $04, $00, $01, $00, $00, $00, $03
    .BYTE $03, $03, $03, $02, $03, $03, $04, $04

ColumnHeapUWCellar:
    .BYTE $82, $43, $43, $43, $02, $0C, $43, $80
    .BYTE $41, $41, $41, $41, $43, $82, $43, $42
    .BYTE $0C, $01, $41, $43, $82, $43, $42, $0C
    .BYTE $03, $02, $0C, $43, $82, $43, $43, $43
    .BYTE $43, $43
