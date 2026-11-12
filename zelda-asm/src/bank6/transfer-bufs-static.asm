; ============================================================================
; FILE:     bank6/transfer-bufs-static.asm BANK: 6 (switchable, $8000-$BFFF)
; UPSTREAM: Z_06.asm 655-874
; SUBJECTS: (none)
; TOPIC:    Static PPU transfer buffers; bank tail
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0052 Mode1TileTransferBuf                          data
;   L0071 ModeFCharsTransferBuf                         data
;   L0089 GanonPaletteRow7TransferBuf                   data
;   L0092 EndingPaletteTransferBuf                      data
;   L0097 BlankTextBoxLines                             data
;   L0101 BlankPersonWares                              data
;   L0105 SubmenuTriforceApexTransferBuf                data
;   L0108 SubmenuTriforceBottomTransferBuf              data
;   L0112 GhostPaletteRow7TransferBuf                   data
;   L0115 GreenBgPaletteRow7TransferBuf                 data
;   L0118 BrownBgPaletteRow7TransferBuf                 data
;   L0121 CaveBgPaletteRowsTransferBuf                  data
;   L0125 CellarAttrsTransferBuf                        data
;   L0129 WhitePaletteBottomHalfTransferBuf             data
;   L0133 RedArmosPaletteRow7TransferBuf                data
;   L0136 GleeokPaletteRow7TransferBuf                  data
;   L0139 AquamentusPaletteRow7TransferBuf              data
;   L0142 OrangeBossPaletteRow7TransferBuf              data
;   L0145 BombCapacityPriceTextTransferBuf              data
;   L0148 LifeOrMoneyCostTextTransferBuf                data
;   L0152 Mode8TextTileBuffer                           data
;   L0158 StatusBarStaticsTransferBuf                   data
;   L0169 InventoryTextTransferBuf                      data
;   L0173 SubmenuBoxesTopsTransferBuf                   data
;   L0178 SubmenuBoxesSidesTransferBuf                  data
;   L0183 SelectedItemBoxBottomTransferBuf              data
;   L0186 UseBButtonTextTransferBuf                     data
;   L0190 InventoryBoxBottomTransferBuf                 data
;   L0195 SubmenuMapRemainderTransferBuf                data
;   L0202 SheetMapBottomEdgeTransferBuf                 data
;   L0207 SubmenuAttrs1TransferBuf                      data
;   L0211 SubmenuAttrs2TransferBuf                      data
;   L0214 BlankBottomRowNT2TransferBuf                  data
;   L0217 BlankRowTransferBuf                           data
;   L0220 Mode11DeadLinkPalette                         data
;   L0223 GameOverTransferBuf                           data
;   L0228 Mode11BackgroundPaletteBottomHalfTransferBuf  data
;   L0232 Mode11PlayAreaAttrsTopHalfTransferBuf         data
;   L0235 Mode11PlayAreaAttrsBottomHalfTransferBuf      data
;   L0238 StoryTileAttrTransferBuf                      data
;   L0241 GameTitleTransferBuf                          data
; ============================================================================

Mode1TileTransferBuf:
    .BYTE $23, $C0, $7F, $00, $23, $D4, $03, $40
    .BYTE $50, $50, $23, $DC, $03, $44, $55, $55
    .BYTE $23, $E4, $03, $44, $55, $55, $20, $A8
    .BYTE $0F, $62, $24, $1C, $24, $0E, $24, $15
    .BYTE $24, $0E, $24, $0C, $24, $1D, $24, $62
    .BYTE $21, $03, $01, $69, $21, $04, $58, $6A
    .BYTE $21, $1C, $01, $6B, $21, $23, $D0, $6C
    .BYTE $21, $3C, $D0, $6C, $23, $23, $01, $6E
    .BYTE $23, $24, $58, $6A, $23, $3C, $01, $6D
    .BYTE $21, $0A, $06, $24, $17, $0A, $16, $0E
    .BYTE $24, $21, $13, $06, $24, $15, $12, $0F
    .BYTE $0E, $24, $22, $A6, $12, $1B, $0E, $10
    .BYTE $12, $1C, $1D, $0E, $1B, $24, $22, $18
    .BYTE $1E, $1B, $24, $17, $0A, $16, $0E, $22
    .BYTE $E6, $10, $0E, $15, $12, $16, $12, $17
    .BYTE $0A, $1D, $12, $18, $17, $24, $16, $18
    .BYTE $0D, $0E, $FF

ModeFCharsTransferBuf:
    .BYTE $22, $05, $01, $69, $22, $06, $55, $6A
    .BYTE $22, $1B, $01, $6B, $22, $25, $C7, $6C
    .BYTE $22, $3B, $C7, $6C, $23, $05, $01, $6E
    .BYTE $23, $06, $55, $6A, $23, $1B, $01, $6D
    .BYTE $22, $26, $15, $0A, $24, $0B, $24, $0C
    .BYTE $24, $0D, $24, $0E, $24, $0F, $24, $10
    .BYTE $24, $11, $24, $12, $24, $13, $24, $14
    .BYTE $22, $66, $15, $15, $24, $16, $24, $17
    .BYTE $24, $18, $24, $19, $24, $1A, $24, $1B
    .BYTE $24, $1C, $24, $1D, $24, $1E, $24, $1F
    .BYTE $22, $A6, $15, $20, $24, $21, $24, $22
    .BYTE $24, $23, $24, $62, $24, $63, $24, $28
    .BYTE $24, $29, $24, $2A, $24, $2B, $24, $2C
    .BYTE $22, $E6, $13, $00, $24, $01, $24, $02
    .BYTE $24, $03, $24, $04, $24, $05, $24, $06
    .BYTE $24, $07, $24, $08, $24, $09, $FF

GanonPaletteRow7TransferBuf:
    .BYTE $3F, $1C, $04, $0F, $16, $2C, $3C, $FF

EndingPaletteTransferBuf:
    .BYTE $3F, $08, $08, $0F, $22, $10, $00, $0F
    .BYTE $2A, $10, $00, $3F, $1C, $04, $0F, $27
    .BYTE $06, $16, $FF

BlankTextBoxLines:
    .BYTE $21, $A4, $58, $24, $21, $C4, $58, $24
    .BYTE $FF

BlankPersonWares:
    .BYTE $21, $E4, $58, $24, $22, $C8, $4D, $24
    .BYTE $FF

SubmenuTriforceApexTransferBuf:
    .BYTE $2A, $CF, $02, $ED, $EE, $FF

SubmenuTriforceBottomTransferBuf:
    .BYTE $2B, $6A, $0C, $EB, $EF, $F1, $F1, $F1
    .BYTE $F1, $F1, $F1, $F1, $F1, $F0, $EC, $FF

GhostPaletteRow7TransferBuf:
    .BYTE $3F, $1C, $04, $0F, $30, $00, $12, $FF

GreenBgPaletteRow7TransferBuf:
    .BYTE $3F, $1C, $04, $0F, $1A, $37, $12, $FF

BrownBgPaletteRow7TransferBuf:
    .BYTE $3F, $1C, $04, $0F, $17, $37, $12, $FF

CaveBgPaletteRowsTransferBuf:
    .BYTE $3F, $08, $08, $0F, $30, $00, $12, $0F
    .BYTE $07, $0F, $17, $FF

CellarAttrsTransferBuf:
    .BYTE $23, $D0, $60, $AA, $23, $F0, $50, $AA
    .BYTE $FF

WhitePaletteBottomHalfTransferBuf:
    .BYTE $3F, $08, $08, $0F, $30, $30, $30, $0F
    .BYTE $30, $30, $30, $FF

RedArmosPaletteRow7TransferBuf:
    .BYTE $3F, $1C, $04, $0F, $0F, $1C, $16, $FF

GleeokPaletteRow7TransferBuf:
    .BYTE $3F, $1C, $04, $0F, $2A, $1A, $0C, $FF

AquamentusPaletteRow7TransferBuf:
    .BYTE $3F, $1C, $04, $0F, $0A, $29, $30, $FF

OrangeBossPaletteRow7TransferBuf:
    .BYTE $3F, $1C, $04, $0F, $17, $27, $30, $FF

BombCapacityPriceTextTransferBuf:
    .BYTE $22, $CD, $04, $62, $01, $00, $00, $FF

LifeOrMoneyCostTextTransferBuf:
    .BYTE $22, $CB, $0A, $62, $01, $24, $24, $24
    .BYTE $24, $24, $62, $05, $00, $FF

Mode8TextTileBuffer:
    .BYTE $23, $C0, $7F, $00, $21, $4A, $08, $0C
    .BYTE $18, $17, $1D, $12, $17, $1E, $0E, $21
    .BYTE $AA, $04, $1C, $0A, $1F, $0E, $22, $0A
    .BYTE $05, $1B, $0E, $1D, $1B, $22, $FF

StatusBarStaticsTransferBuf:
    .BYTE $23, $C2, $0E, $40, $00, $00, $44, $55
    .BYTE $55, $00, $00, $04, $00, $00, $44, $55
    .BYTE $55, $20, $6F, $0E, $69, $0B, $6B, $69
    .BYTE $0A, $6B, $24, $24, $62, $15, $12, $0F
    .BYTE $0E, $62, $20, $CF, $06, $6E, $6A, $6D
    .BYTE $6E, $6A, $6D, $20, $8F, $C2, $6C, $20
    .BYTE $91, $C2, $6C, $20, $92, $C2, $6C, $20
    .BYTE $94, $C2, $6C, $20, $6B, $84, $F7, $24
    .BYTE $F9, $61, $FF

InventoryTextTransferBuf:
    .BYTE $29, $84, $09, $12, $17, $1F, $0E, $17
    .BYTE $1D, $18, $1B, $22, $FF

SubmenuBoxesTopsTransferBuf:
    .BYTE $29, $C7, $04, $69, $6A, $6A, $6B, $29
    .BYTE $CF, $01, $69, $29, $D0, $4B, $6A, $29
    .BYTE $DB, $01, $6B, $FF

SubmenuBoxesSidesTransferBuf:
    .BYTE $29, $E7, $C2, $6C, $29, $EA, $C2, $6C
    .BYTE $29, $EF, $C4, $6C, $29, $FB, $C4, $6C
    .BYTE $FF

SelectedItemBoxBottomTransferBuf:
    .BYTE $2A, $27, $04, $6E, $6A, $6A, $6D, $FF

UseBButtonTextTransferBuf:
    .BYTE $2A, $42, $0C, $1E, $1C, $0E, $24, $0B
    .BYTE $24, $0B, $1E, $1D, $1D, $18, $17, $FF

InventoryBoxBottomTransferBuf:
    .BYTE $2A, $64, $08, $0F, $18, $1B, $24, $1D
    .BYTE $11, $12, $1C, $2A, $6F, $01, $6E, $2A
    .BYTE $70, $4B, $6A, $2A, $7B, $01, $6D, $FF

SubmenuMapRemainderTransferBuf:
    .BYTE $2B, $43, $07, $0C, $18, $16, $19, $0A
    .BYTE $1C, $1C, $2A, $A5, $03, $16, $0A, $19
    .BYTE $2A, $8C, $10, $F5, $F5, $FD, $F5, $F5
    .BYTE $FD, $F5, $F5, $FD, $F5, $F5, $F5, $FD
    .BYTE $F5, $F5, $F5, $FF

SheetMapBottomEdgeTransferBuf:
    .BYTE $2B, $AC, $10, $F5, $FE, $F5, $F5, $F5
    .BYTE $FE, $F5, $F5, $F5, $F5, $FE, $F5, $F5
    .BYTE $F5, $FE, $F5, $FF

SubmenuAttrs1TransferBuf:
    .BYTE $2B, $D9, $43, $05, $2B, $DC, $4B, $00
    .BYTE $FF

SubmenuAttrs2TransferBuf:
    .BYTE $2B, $E9, $56, $55, $FF

BlankBottomRowNT2TransferBuf:
    .BYTE $2B, $A0, $60, $24, $FF

BlankRowTransferBuf:
    .BYTE $28, $E0, $60, $24, $FF

Mode11DeadLinkPalette:
    .BYTE $3F, $10, $04, $0F, $10, $30, $00, $FF

GameOverTransferBuf:
    .BYTE $23, $E3, $03, $0F, $0F, $CF, $22, $4C
    .BYTE $0A, $10, $0A, $16, $0E, $24, $18, $1F
    .BYTE $0E, $1B, $24, $22, $6C, $4A, $24, $FF

Mode11BackgroundPaletteBottomHalfTransferBuf:
    .BYTE $3F, $08, $08, $0F, $17, $16, $26, $0F
    .BYTE $17, $16, $26, $FF

Mode11PlayAreaAttrsTopHalfTransferBuf:
    .BYTE $23, $D0, $58, $FF, $FF

Mode11PlayAreaAttrsBottomHalfTransferBuf:
    .BYTE $23, $E8, $58, $FF, $FF

StoryTileAttrTransferBuf:
    .INCBIN "dat/StoryTileAttrTransferBuf.dat"

GameTitleTransferBuf:
    .INCBIN "dat/GameTitleTransferBuf.dat"

.SEGMENT "BANK_06_ISR"




; Unknown block
    .BYTE $78, $D8, $A9, $00, $8D, $00, $20, $A2
    .BYTE $FF, $9A, $AD, $02, $20, $29, $80, $F0
    .BYTE $F9, $AD, $02, $20, $29, $80, $F0, $F9
    .BYTE $09, $FF, $8D, $00, $80, $8D, $00, $A0
    .BYTE $8D, $00, $C0, $8D, $00, $E0, $A9, $0F
    .BYTE $20, $98, $BF, $A9, $00, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $A9, $07, $20, $AC, $BF, $4C, $40, $E4
    .BYTE $8D, $00, $80, $4A, $8D, $00, $80, $4A
    .BYTE $8D, $00, $80, $4A, $8D, $00, $80, $4A
    .BYTE $8D, $00, $80, $60, $8D, $00, $E0, $4A
    .BYTE $8D, $00, $E0, $4A, $8D, $00, $E0, $4A
    .BYTE $8D, $00, $E0, $4A, $8D, $00, $E0, $60

.SEGMENT "BANK_06_VEC"




; Unknown block
    .BYTE $84, $E4, $50, $BF, $F0, $BF
