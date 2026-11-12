; ============================================================================
; FILE:     bank5/room-layouts-ow.asm    BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 4228-4408
; SUBJECTS: (none)
; TOPIC:    Overworld room layouts, cave layouts, column heaps (data)
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0033 RoomLayoutsOW      data
;   L0036 RoomLayoutOWCave0  data
;   L0040 RoomLayoutOWCave1  data
;   L0044 RoomLayoutOWCave2  data
;   L0048 ColumnHeapOW0      data
;   L0057 ColumnHeapOW1      data
;   L0066 ColumnHeapOW2      data
;   L0077 ColumnHeapOW3      data
;   L0088 ColumnHeapOW4      data
;   L0097 ColumnHeapOW5      data
;   L0107 ColumnHeapOW6      data
;   L0117 ColumnHeapOW7      data
;   L0127 ColumnHeapOW8      data
;   L0137 ColumnHeapOW9      data
;   L0148 ColumnHeapOWA      data
;   L0159 ColumnHeapOWB      data
;   L0169 ColumnHeapOWC      data
;   L0178 ColumnHeapOWD      data
;   L0189 ColumnHeapOWE      data
;   L0198 ColumnHeapOWF      data
;   L0209 RoomLayoutsOWAddr  data
;   L0212 ColumnHeapOWAddr   data
; ============================================================================

RoomLayoutsOW:
    .INCBIN "dat/RoomLayoutsOW.dat"

RoomLayoutOWCave0:
    .BYTE $00, $00, $95, $95, $95, $95, $95, $C2
    .BYTE $C2, $95, $95, $95, $95, $95, $00, $00

RoomLayoutOWCave1:
    .BYTE $00, $00, $95, $95, $95, $F8, $95, $C2
    .BYTE $F8, $95, $95, $F8, $95, $95, $00, $00

RoomLayoutOWCave2:
    .BYTE $00, $A9, $64, $66, $02, $53, $54, $D1
    .BYTE $54, $54, $56, $02, $64, $66, $A8, $00

ColumnHeapOW0:
    .BYTE $DB, $5B, $5B, $DB, $5B, $1B, $0E, $1A
    .BYTE $5B, $DB, $4E, $4E, $4E, $0E, $1A, $1B
    .BYTE $DB, $4E, $32, $1B, $34, $4E, $1A, $1B
    .BYTE $DB, $5B, $1B, $4E, $4E, $4E, $DB, $0E
    .BYTE $32, $5B, $1B, $4E, $1A, $1B, $DB, $5B
    .BYTE $1B, $4E, $4E, $1A, $1B, $DB, $CE, $4E
    .BYTE $4E, $4E, $0E, $1A, $1B

ColumnHeapOW1:
    .BYTE $9B, $0C, $4E, $4E, $4E, $0E, $1A, $1B
    .BYTE $CA, $4E, $0E, $4E, $4E, $4A, $C5, $45
    .BYTE $05, $0B, $C5, $45, $05, $45, $45, $45
    .BYTE $D9, $28, $59, $D9, $59, $59, $19, $D9
    .BYTE $59, $4E, $0E, $59, $D9, $4E, $4E, $4E
    .BYTE $0E, $59, $8E, $4E, $4E, $CE, $4E, $4E
    .BYTE $0E

ColumnHeapOW2:
    .BYTE $D9, $59, $0E, $4E, $4E, $4E, $DB, $0E
    .BYTE $5B, $5B, $4E, $1A, $1B, $DB, $0E, $33
    .BYTE $5B, $1B, $4E, $1A, $1B, $CE, $4E, $4E
    .BYTE $4E, $0E, $59, $DB, $4E, $33, $1B, $35
    .BYTE $4E, $1A, $1B, $D9, $4E, $19, $0E, $19
    .BYTE $4E, $D9, $0E, $19, $0E, $19, $0E, $19
    .BYTE $0E, $D9, $4E, $0E, $19, $0E, $4E, $D9
    .BYTE $19, $4E, $4E, $0E, $19, $D9, $0E, $59
    .BYTE $59, $59

ColumnHeapOW3:
    .BYTE $D9, $0E, $59, $59, $19, $0E, $D9, $4E
    .BYTE $19, $0E, $28, $4E, $59, $DB, $4E, $13
    .BYTE $0E, $13, $4E, $1A, $1B, $DB, $0E, $13
    .BYTE $0E, $13, $0E, $13, $0E, $1A, $1B, $DB
    .BYTE $4E, $0E, $13, $4E, $0E, $1A, $1B, $C8
    .BYTE $48, $17, $4E, $4E, $1A, $1B, $C5, $45
    .BYTE $07, $4E, $4E, $1A, $1B, $C5, $45, $07
    .BYTE $0E, $1A, $DB, $5B, $1B, $4E, $4E, $4A
    .BYTE $DB, $4E, $4E, $15

ColumnHeapOW4:
    .BYTE $C8, $48, $17, $4E, $4E, $59, $C9, $49
    .BYTE $18, $4E, $4E, $D9, $59, $4E, $06, $C5
    .BYTE $45, $45, $C9, $49, $09, $0B, $49, $49
    .BYTE $09, $C8, $48, $08, $48, $C8, $48, $17
    .BYTE $4E, $4E, $4E, $DB, $4E, $4E, $06, $C5
    .BYTE $45, $07, $4E, $4E, $D9, $4E, $2C, $4E
    .BYTE $4E, $59

ColumnHeapOW5:
    .BYTE $DB, $5B, $1B, $0E, $4A, $4A, $0A, $DB
    .BYTE $5B, $4E, $06, $C5, $45, $07, $4E, $4E
    .BYTE $4E, $DB, $0E, $15, $48, $17, $4E, $1A
    .BYTE $1B, $DB, $0E, $06, $45, $07, $4E, $1A
    .BYTE $1B, $DB, $0E, $06, $05, $45, $45, $45
    .BYTE $DB, $0E, $16, $49, $18, $4E, $1A, $1B
    .BYTE $D9, $0E, $59, $59, $19, $0E, $CE, $4E
    .BYTE $4E, $06, $45, $45

ColumnHeapOW6:
    .BYTE $D9, $4E, $4E, $16, $49, $49, $D9, $4E
    .BYTE $4E, $06, $C5, $45, $07, $4E, $D9, $59
    .BYTE $4E, $15, $48, $48, $DB, $0E, $1C, $1F
    .BYTE $4E, $4E, $1A, $1B, $DB, $0E, $1D, $0C
    .BYTE $4E, $4E, $1A, $1B, $DB, $0E, $1E, $20
    .BYTE $4E, $4E, $1A, $1B, $DB, $0E, $21, $24
    .BYTE $4E, $4E, $1A, $1B, $DB, $0E, $22, $0C
    .BYTE $4E, $4E, $4A

ColumnHeapOW7:
    .BYTE $DB, $0E, $23, $25, $4E, $4E, $1A, $1B
    .BYTE $DB, $4E, $0E, $26, $4E, $0E, $1A, $1B
    .BYTE $DB, $5B, $07, $4E, $1A, $1B, $DB, $5B
    .BYTE $07, $4E, $4E, $1A, $1B, $CE, $0E, $59
    .BYTE $59, $19, $0E, $D9, $4E, $0E, $26, $4E
    .BYTE $0E, $D9, $4E, $4E, $4E, $4E, $0E, $DB
    .BYTE $4E, $19, $0E, $19, $4E, $1A, $1B, $DB
    .BYTE $5B, $5B, $45

ColumnHeapOW8:
    .BYTE $C5, $05, $5B, $5B, $DB, $5B, $27, $0E
    .BYTE $1A, $5B, $DB, $0E, $14, $0E, $14, $0E
    .BYTE $14, $4E, $0E, $CA, $4A, $0A, $4E, $4E
    .BYTE $1A, $1B, $DB, $1B, $35, $4E, $4E, $0E
    .BYTE $1A, $1B, $9B, $27, $4E, $4E, $06, $C5
    .BYTE $45, $05, $2F, $4E, $4E, $0E, $DB, $0E
    .BYTE $22, $0C, $4E, $4E, $4A, $DB, $0E, $14
    .BYTE $0E, $29, $0E, $14, $4E, $0E

ColumnHeapOW9:
    .BYTE $DB, $35, $4E, $4E, $4E, $1A, $1B, $DB
    .BYTE $1B, $34, $4E, $06, $C5, $45, $05, $2E
    .BYTE $4E, $4E, $0E, $CE, $0E, $14, $0E, $14
    .BYTE $0E, $14, $0E, $1A, $1B, $CE, $0E, $14
    .BYTE $0E, $14, $0E, $14, $4E, $0E, $DB, $4D
    .BYTE $4D, $4D, $0D, $1A, $1B, $C5, $05, $1B
    .BYTE $0C, $4E, $0E, $4A, $0A, $DB, $1B, $35
    .BYTE $4E, $06, $45, $C5, $4E, $4E, $4E, $0E
    .BYTE $45

ColumnHeapOWA:
    .BYTE $DB, $5B, $34, $4E, $4E, $1A, $1B, $DB
    .BYTE $5B, $35, $4E, $4E, $1A, $1B, $9B, $35
    .BYTE $4E, $4E, $4E, $0E, $33, $1B, $9B, $34
    .BYTE $4E, $4E, $4E, $0E, $32, $1B, $9B, $34
    .BYTE $4E, $4E, $4E, $0E, $1A, $1B, $9B, $35
    .BYTE $4E, $4E, $4E, $0E, $1A, $1B, $DB, $5B
    .BYTE $34, $0E, $32, $DB, $5B, $35, $0E, $33
    .BYTE $5B, $DB, $34, $4E, $4E, $0E, $32, $DB
    .BYTE $35, $4E, $4E, $0E, $33

ColumnHeapOWB:
    .BYTE $DB, $5B, $07, $4E, $4E, $4E, $DB, $0E
    .BYTE $06, $05, $45, $2F, $0E, $1A, $1B, $DB
    .BYTE $0E, $06, $05, $45, $2E, $0E, $1A, $1B
    .BYTE $CA, $4A, $0A, $0E, $1A, $DB, $5B, $15
    .BYTE $45, $45, $45, $DB, $4E, $4E, $4E, $0E
    .BYTE $CA, $4E, $4E, $4E, $0E, $1A, $1B, $DB
    .BYTE $4E, $2C, $0E, $2C, $4E, $1A, $1B, $CA
    .BYTE $4A, $4E, $06, $45, $45

ColumnHeapOWC:
    .BYTE $9B, $35, $4E, $0E, $13, $4E, $0E, $1A
    .BYTE $1B, $F6, $76, $0F, $4E, $1A, $1B, $5B
    .BYTE $DB, $4D, $4D, $4D, $4D, $0D, $CE, $4E
    .BYTE $0E, $30, $45, $45, $05, $DB, $F7, $F7
    .BYTE $77, $77, $37, $77, $1A, $1B, $DB, $1B
    .BYTE $34, $4E, $4E, $0E, $1A, $1B, $DB, $34
    .BYTE $0E, $4E, $4E, $0E, $1A, $1B

ColumnHeapOWD:
    .BYTE $C5, $45, $0B, $4E, $4E, $4E, $DB, $0E
    .BYTE $06, $45, $07, $4E, $4E, $9B, $27, $4E
    .BYTE $4E, $4E, $0E, $1A, $1B, $9B, $27, $4E
    .BYTE $4E, $4E, $4E, $0E, $DB, $1B, $27, $07
    .BYTE $4E, $1A, $5B, $1B, $CE, $4E, $4E, $4E
    .BYTE $33, $5B, $DB, $5B, $1B, $0E, $15, $48
    .BYTE $48, $DB, $0E, $2D, $0C, $4E, $4E, $4A
    .BYTE $DB, $0E, $1C, $1F, $0E, $1E, $20, $0E
    .BYTE $1A, $1B

ColumnHeapOWE:
    .BYTE $C8, $48, $08, $C5, $45, $45, $08, $48
    .BYTE $48, $DB, $35, $CE, $4E, $4E, $4E, $0E
    .BYTE $33, $1B, $DB, $0E, $33, $5B, $27, $4E
    .BYTE $1A, $1B, $DB, $4E, $4E, $1C, $1F, $0E
    .BYTE $1A, $1B, $DB, $5B, $1B, $35, $4E, $33
    .BYTE $DB, $5B, $1B, $34, $4E, $32, $DB, $4E
    .BYTE $4E, $4E, $0E, $33

ColumnHeapOWF:
    .BYTE $9B, $5B, $34, $4E, $0E, $32, $DB, $1B
    .BYTE $35, $4E, $0E, $33, $1B, $DB, $0E, $1E
    .BYTE $20, $0E, $1C, $1F, $0E, $1A, $1B, $DB
    .BYTE $4E, $4E, $1E, $20, $0E, $1A, $1B, $D9
    .BYTE $59, $19, $4E, $4E, $59, $DB, $0E, $2D
    .BYTE $0C, $4E, $4E, $1A, $1B, $CE, $4E, $0E
    .BYTE $31, $05, $C5, $45, $07, $4E, $28, $59
    .BYTE $19, $DB, $4D, $4D, $12, $4D, $1A, $1B
    .BYTE $DB, $77, $77, $77, $37, $1A, $1B

RoomLayoutsOWAddr:
    .ADDR RoomLayoutsOW

ColumnHeapOWAddr:
    .ADDR ColumnHeapOW0
