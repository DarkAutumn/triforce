; ============================================================================
; FILE:     bank0/song-data.asm          BANK: 0 (switchable, $8000-$BFFF)
; UPSTREAM: Z_00.asm 9-104
; SUBJECTS: (none)
; TOPIC:    Song table, song phrase headers, song script blobs
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0030 SongTable              data
;   L0052 SongHeaderDemo0        data  Description:
;   L0065 SongHeaderItemTaken0   data
;   L0068 SongHeaderEndLevel0    data
;   L0071 SongHeaderOverworld0   data
;   L0078 SongHeaderUnderworld0  data
;   L0082 SongHeaderLastLevel0   data
;   L0085 SongHeaderGanon0       data
;   L0088 SongHeaderEnding0      data
;   L0097 SongHeaderZelda        data
;   L0100 SongScriptItemTaken0   data
;   L0103 SongScriptOverworld0   data
;   L0106 SongScriptUnderworld0  data
;   L0109 SongScriptEndLevel0    data
;   L0112 SongScriptLastLevel0   data
;   L0115 SongScriptGanon0       data
;   L0118 SongScriptEnding0      data
;   L0121 SongScriptDemo0        data
;   L0124 SongScriptZelda0       data
; ============================================================================

SongTable:
    .BYTE $7D, $B5, $6E, $67, $7D, $AD, $64, $64
    .BYTE $75, $7D, $85, $95, $7D, $8D, $95, $9D
    .BYTE $A5, $BD, $C5, $CD, $D5, $DD, $D5, $E5
    .BYTE $ED, $24, $2C, $34, $3C, $44, $34, $4C
    .BYTE $54, $5C, $44, $F5

; Description:
; Each song phrase is described by a header in this format:
;
; 0: Note length table base
; 1: Song script address low
; 2: Song script address high
; 3: Triangle note offset in script
; 4: Square 0 note offset in script
; 5: Noise note offset in script
; 6: Envelope selector
; 7: TODO:
;
; Note that the last byte of the "Item taken" header overlaps
; "End level", and "End level" overlaps "Overworld".
;
SongHeaderDemo0:
    .BYTE $20, $8B, $94, $3B, $1D, $4F, $80, $01
    .BYTE $20, $DC, $94, $27, $57, $23, $01, $80
    .BYTE $20, $A1, $95, $38, $17, $B8, $80, $80
    .BYTE $20, $F1, $95, $6C, $26, $68, $80, $80
    .BYTE $20, $8D, $96, $3E, $25, $21, $80, $80
    .BYTE $20, $EB, $96, $19, $0D, $31, $80, $80
    .BYTE $20, $20, $97, $3F, $27, $7C, $80, $80
    .BYTE $20, $8F, $97, $1D, $11, $0D, $80, $80

; Unknown block
    .BYTE $10, $A1, $8E

SongHeaderItemTaken0:
    .BYTE $10, $5D, $8E, $0D, $07, $00, $80

SongHeaderEndLevel0:
    .BYTE $10, $A4, $91, $46, $22, $00, $80

SongHeaderOverworld0:
    .BYTE $10, $70, $8E, $32, $5D, $8E, $01, $80
    .BYTE $10, $0F, $8F, $35, $16, $CE, $01, $80
    .BYTE $10, $55, $8F, $60, $26, $88, $01, $80
    .BYTE $10, $32, $90, $59, $2D, $A4, $01, $80
    .BYTE $10, $E4, $8F, $3B, $1A, $F2, $01, $80

SongHeaderUnderworld0:
    .BYTE $00, $DD, $90, $45, $22, $00, $01, $01
    .BYTE $00, $3A, $91, $39, $1C, $00, $01, $01

SongHeaderLastLevel0:
    .BYTE $10, $FD, $91, $A5, $53, $CD, $80, $80

SongHeaderGanon0:
    .BYTE $10, $CC, $92, $22, $10, $00, $80, $01

SongHeaderEnding0:
    .BYTE $08, $F7, $92, $22, $50, $59, $01, $80
    .BYTE $08, $F7, $92, $2F, $50, $59, $01, $80
    .BYTE $08, $52, $93, $7A, $1B, $C2, $80, $80
    .BYTE $08, $86, $93, $46, $24, $8E, $01, $80
    .BYTE $08, $9D, $93, $44, $23, $77, $01, $80
    .BYTE $08, $ED, $93, $1B, $0E, $27, $01, $80
    .BYTE $08, $1A, $94, $40, $1A, $6B, $80, $80

SongHeaderZelda:
    .BYTE $10, $C4, $97, $3F, $20, $00, $80, $80

SongScriptItemTaken0:
    .INCBIN "dat/SongScriptItemTaken0.dat"

SongScriptOverworld0:
    .INCBIN "dat/SongScriptOverworld0.dat"

SongScriptUnderworld0:
    .INCBIN "dat/SongScriptUnderworld0.dat"

SongScriptEndLevel0:
    .INCBIN "dat/SongScriptEndLevel0.dat"

SongScriptLastLevel0:
    .INCBIN "dat/SongScriptLastLevel0.dat"

SongScriptGanon0:
    .INCBIN "dat/SongScriptGanon0.dat"

SongScriptEnding0:
    .INCBIN "dat/SongScriptEnding0.dat"

SongScriptDemo0:
    .INCBIN "dat/SongScriptDemo0.dat"

SongScriptZelda0:
    .INCBIN "dat/SongScriptZelda0.dat"
