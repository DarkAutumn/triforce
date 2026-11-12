; ============================================================================
; FILE:     bank6/level-blocks-common-data.asm BANK: 6 (switchable, $8000-$BFFF; the BANK_06_DATA part is copied to RAM $67F0-$687D)
; UPSTREAM: Z_06.asm 390-483
; SUBJECTS: (none)
; TOPIC:    Level block/info blobs, common data block
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0036 LevelBlockOW                 data
;   L0039 LevelBlockUW1Q1              data
;   L0042 LevelBlockUW2Q1              data
;   L0045 LevelBlockUW1Q2              data
;   L0048 LevelBlockUW2Q2              data
;   L0051 LevelInfoOW                  data
;   L0054 LevelInfoUW1                 data
;   L0057 LevelInfoUW2                 data
;   L0060 LevelInfoUW3                 data
;   L0063 LevelInfoUW4                 data
;   L0066 LevelInfoUW5                 data
;   L0069 LevelInfoUW6                 data
;   L0072 LevelInfoUW7                 data
;   L0075 LevelInfoUW8                 data
;   L0078 LevelInfoUW9                 data
;   L0080 CommonDataBlock_Bank6        data
;   L0092 MenuPalettesTransferBuf      data
;   L0099 LevelPaletteRow7TransferBuf  data
;   L0102 LevelNumberTransferBuf       data
;   L0106 ColumnDirectoryOW            data
;   L0112 TriforceRow0TransferBuf      data
;   L0115 TriforceRow1TransferBuf      data
;   L0119 TriforceRow2TransferBuf      data
;   L0123 TriforceRow3TransferBuf      data
;   L0127 TriforceTextTransferBuf      data
; ============================================================================

LevelBlockOW:
    .INCBIN "dat/LevelBlockOW.dat"

LevelBlockUW1Q1:
    .INCBIN "dat/LevelBlockUW1Q1.dat"

LevelBlockUW2Q1:
    .INCBIN "dat/LevelBlockUW2Q1.dat"

LevelBlockUW1Q2:
    .INCBIN "dat/LevelBlockUW1Q2.dat"

LevelBlockUW2Q2:
    .INCBIN "dat/LevelBlockUW2Q2.dat"

LevelInfoOW:
    .INCBIN "dat/LevelInfoOW.dat"

LevelInfoUW1:
    .INCBIN "dat/LevelInfoUW1.dat"

LevelInfoUW2:
    .INCBIN "dat/LevelInfoUW2.dat"

LevelInfoUW3:
    .INCBIN "dat/LevelInfoUW3.dat"

LevelInfoUW4:
    .INCBIN "dat/LevelInfoUW4.dat"

LevelInfoUW5:
    .INCBIN "dat/LevelInfoUW5.dat"

LevelInfoUW6:
    .INCBIN "dat/LevelInfoUW6.dat"

LevelInfoUW7:
    .INCBIN "dat/LevelInfoUW7.dat"

LevelInfoUW8:
    .INCBIN "dat/LevelInfoUW8.dat"

LevelInfoUW9:
    .INCBIN "dat/LevelInfoUW9.dat"
CommonDataBlock_Bank6:

.SEGMENT "BANK_06_DATA"


.EXPORT ColumnDirectoryOW
.EXPORT LevelNumberTransferBuf
.EXPORT LevelPaletteRow7TransferBuf
.EXPORT MenuPalettesTransferBuf
.EXPORT TriforceRow0TransferBuf


MenuPalettesTransferBuf:
    .BYTE $3F, $00, $20, $0F, $30, $00, $12, $0F
    .BYTE $16, $27, $36, $0F, $0C, $1C, $2C, $0F
    .BYTE $12, $1C, $2C, $0F, $29, $27, $07, $0F
    .BYTE $22, $27, $07, $0F, $26, $27, $07, $0F
    .BYTE $15, $27, $30, $FF

LevelPaletteRow7TransferBuf:
    .BYTE $3F, $1C, $04, $0F, $0F, $0F, $0F, $FF

LevelNumberTransferBuf:
    .BYTE $20, $42, $07, $15, $0E, $1F, $0E, $15
    .BYTE $62, $00, $FF

ColumnDirectoryOW:
    .BYTE $D8, $9B, $0D, $9C, $3E, $9C, $80, $9C
    .BYTE $C4, $9C, $F6, $9C, $32, $9D, $6D, $9D
    .BYTE $A8, $9D, $E6, $9D, $27, $9E, $6C, $9E
    .BYTE $A9, $9E, $DF, $9E, $21, $9F, $55, $9F

TriforceRow0TransferBuf:
    .BYTE $2A, $EE, $04, $ED, $E9, $EA, $EE, $FF

TriforceRow1TransferBuf:
    .BYTE $2B, $0D, $06, $ED, $E9, $24, $24, $EA
    .BYTE $EE, $FF

TriforceRow2TransferBuf:
    .BYTE $2B, $2C, $08, $ED, $E9, $24, $24, $24
    .BYTE $24, $EA, $EE, $FF

TriforceRow3TransferBuf:
    .BYTE $2B, $4B, $0A, $ED, $E9, $24, $24, $24
    .BYTE $24, $24, $24, $EA, $EE, $FF

TriforceTextTransferBuf:
    .BYTE $2B, $AC, $08, $1D, $1B, $12, $0F, $18
    .BYTE $1B, $0C, $0E
