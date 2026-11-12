; ============================================================================
; FILE:     bank6/quest2-replacements.asm BANK: 6 (switchable, $8000-$BFFF)
; UPSTREAM: Z_06.asm 265-388
; SUBJECTS: (none)
; TOPIC:    Second-quest replacement tables
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0024 LevelBlockAttrsBQ2ReplacementOffsets  data
;   L0027 LevelBlockAttrsBQ2ReplacementValues   data
;   L0030 LevelInfoUWQ2Replacements1            data
;   L0040 LevelInfoUWQ2Replacements2            data
;   L0049 LevelInfoUWQ2Replacements3            data
;   L0059 LevelInfoUWQ2Replacements4            data
;   L0069 LevelInfoUWQ2Replacements5            data
;   L0080 LevelInfoUWQ2Replacements6            data
;   L0090 LevelInfoUWQ2Replacements7            data
;   L0100 LevelInfoUWQ2Replacements8            data
;   L0111 LevelInfoUWQ2Replacements9            data
;   L0123 LevelInfoUWQ2ReplacementAddrs         data
;   L0134 LevelInfoUWQ2ReplacementSizes         data
; ============================================================================

LevelBlockAttrsBQ2ReplacementOffsets:
    .BYTE $0E, $0F, $22, $34, $3C, $45, $74, $8B

LevelBlockAttrsBQ2ReplacementValues:
    .BYTE $7B, $83, $84, $0F, $0B, $12, $7A, $2F

LevelInfoUWQ2Replacements1:
    .BYTE $C9, $AC, $89, $B7, $00, $E0, $77, $08
    .BYTE $FF, $06, $01, $28, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $07, $00, $00
    .BYTE $00, $00, $00, $00, $00, $FF, $DB, $00
    .BYTE $00, $00, $00, $00, $00, $00, $20, $65
    .BYTE $42, $FF, $20, $85, $02, $FF, $FB, $20
    .BYTE $A5, $02, $FF, $67, $20, $C5, $42, $FF
    .BYTE $FF

LevelInfoUWQ2Replacements2:
    .BYTE $C9, $AC, $89, $87, $05, $00, $75, $20
    .BYTE $FF, $06, $03, $56, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $30, $00, $00
    .BYTE $00, $00, $00, $30, $00, $00, $00, $00
    .BYTE $7F, $03, $00, $00, $00, $00, $20, $67
    .BYTE $01, $FB, $20, $82, $01, $FF, $20, $87
    .BYTE $C3, $FF, $20, $C8, $01, $FF, $FF

LevelInfoUWQ2Replacements3:
    .BYTE $C9, $AC, $89, $37, $0D, $C8, $79, $1B
    .BYTE $FF, $06, $02, $09, $0B, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $2B, $00, $00
    .BYTE $00, $00, $00, $00, $7F, $EC, $7F, $00
    .BYTE $00, $00, $00, $00, $00, $00, $20, $64
    .BYTE $03, $FB, $FF, $FB, $20, $84, $03, $FF
    .BYTE $67, $FF, $20, $A4, $43, $FF, $20, $C4
    .BYTE $03, $FF, $24, $FF, $FF

LevelInfoUWQ2Replacements4:
    .BYTE $C9, $AC, $89, $86, $06, $10, $72, $00
    .BYTE $FF, $06, $05, $21, $58, $7A, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $10, $00, $00
    .BYTE $00, $00, $00, $00, $CF, $DB, $F3, $00
    .BYTE $00, $00, $00, $00, $00, $00, $20, $64
    .BYTE $43, $FF, $20, $85, $02, $FB, $FF, $20
    .BYTE $A4, $02, $FF, $67, $20, $C4, $43, $FF
    .BYTE $FF

LevelInfoUWQ2Replacements5:
    .BYTE $C9, $AC, $89, $87, $0A, $B0, $7D, $4F
    .BYTE $FF, $06, $04, $0F, $6A, $7F, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $5F, $00, $00
    .BYTE $00, $00, $00, $00, $FF, $FF, $E7, $7E
    .BYTE $00, $00, $00, $00, $00, $00, $20, $64
    .BYTE $04, $FF, $FF, $FF, $FB, $20, $84, $04
    .BYTE $FF, $FF, $67, $FF, $20, $A4, $04, $FF
    .BYTE $FF, $FB, $FF, $20, $C4, $04, $FF, $FF
    .BYTE $FF, $67, $FF

LevelInfoUWQ2Replacements6:
    .BYTE $49, $79, $89, $56, $04, $00, $74, $16
    .BYTE $FF, $06, $06, $03, $73, $46, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $26, $00, $00
    .BYTE $00, $00, $00, $04, $0C, $7E, $FF, $80
    .BYTE $F0, $00, $00, $00, $00, $00, $20, $65
    .BYTE $03, $FB, $FF, $67, $20, $68, $C2, $FF
    .BYTE $20, $86, $C3, $FF, $20, $85, $83, $FF
    .BYTE $FF, $67, $20, $A3, $02, $FB, $FF, $FF

LevelInfoUWQ2Replacements7:
    .BYTE $C9, $AC, $89, $79, $0C, $C0, $7F, $2D
    .BYTE $7F, $07, $08, $02, $03, $04, $05, $20
    .BYTE $21, $26, $2B, $2C, $FF, $3D, $00, $00
    .BYTE $00, $00, $FE, $FE, $82, $82, $82, $BE
    .BYTE $80, $FF, $00, $00, $00, $00, $20, $62
    .BYTE $C3, $FF, $20, $63, $C3, $FF, $20, $64
    .BYTE $45, $67, $20, $69, $C4, $FF, $20, $87
    .BYTE $C2, $FF, $20, $C2, $46, $67, $FF

LevelInfoUWQ2Replacements8:
    .BYTE $C9, $AC, $89, $57, $0C, $C0, $79, $1B
    .BYTE $7F, $07, $07, $27, $30, $37, $60, $67
    .BYTE $70, $FF, $FF, $FF, $FF, $1C, $00, $00
    .BYTE $00, $00, $01, $01, $7D, $5D, $5D, $41
    .BYTE $7F, $00, $00, $00, $00, $00, $20, $64
    .BYTE $45, $FB, $20, $84, $05, $FF, $FB, $FB
    .BYTE $24, $FF, $20, $A4, $43, $FF, $20, $A8
    .BYTE $01, $FF, $20, $C2, $46, $FB, $20, $C8
    .BYTE $01, $FF, $FF

LevelInfoUWQ2Replacements9:
    .BYTE $C9, $AC, $89, $B6, $04, $00, $74, $07
    .BYTE $7F, $07, $09, $71, $72, $75, $76, $77
    .BYTE $FF, $FF, $FF, $FF, $FF, $17, $00, $00
    .BYTE $00, $00, $CC, $DE, $76, $7F, $7F, $76
    .BYTE $DE, $CC, $00, $00, $00, $00, $20, $62
    .BYTE $48, $FF, $20, $64, $44, $FB, $20, $83
    .BYTE $46, $FB, $20, $84, $44, $FF, $20, $A2
    .BYTE $08, $FF, $FF, $FB, $FF, $FF, $FB, $FF
    .BYTE $FF, $20, $C3, $46, $67, $20, $C5, $42
    .BYTE $FF, $FF

LevelInfoUWQ2ReplacementAddrs:
    .ADDR LevelInfoUWQ2Replacements1
    .ADDR LevelInfoUWQ2Replacements2
    .ADDR LevelInfoUWQ2Replacements3
    .ADDR LevelInfoUWQ2Replacements4
    .ADDR LevelInfoUWQ2Replacements5
    .ADDR LevelInfoUWQ2Replacements6
    .ADDR LevelInfoUWQ2Replacements7
    .ADDR LevelInfoUWQ2Replacements8
    .ADDR LevelInfoUWQ2Replacements9

LevelInfoUWQ2ReplacementSizes:
    .BYTE $39, $37, $3D, $39, $43, $40, $3F, $43
    .BYTE $4A

; Unknown block
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF
