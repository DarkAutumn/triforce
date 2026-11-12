; ============================================================================
; FILE:     bank6/transfer-buf-dispatch.asm BANK: 6 (switchable, $8000-$BFFF)
; UPSTREAM: Z_06.asm 485-653
; SUBJECTS: (none)
; TOPIC:    Transfer buffer selector table and PPU transfer loop
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0021 TransferBufAddrs         data
;   L0087 TransferCurTileBuf       code
;   L0108 ContinueTransferTileBuf  code
;   L0178 TransferTileBuf          code
; ============================================================================

.SEGMENT "BANK_06_DLIST"


.EXPORT TransferCurTileBuf


TransferBufAddrs:
    .ADDR DynTileBuf
    .ADDR StoryTileAttrTransferBuf
    .ADDR Mode8TextTileBuffer
    .ADDR LevelPaletteRow7TransferBuf
    .ADDR AquamentusPaletteRow7TransferBuf
    .ADDR OrangeBossPaletteRow7TransferBuf
    .ADDR LevelNumberTransferBuf
    .ADDR StatusBarStaticsTransferBuf
    .ADDR GameTitleTransferBuf
    .ADDR MenuPalettesTransferBuf
    .ADDR Mode1TileTransferBuf
    .ADDR ModeFCharsTransferBuf
    .ADDR LevelInfo_PalettesTransferBuf
    .ADDR DynTileBuf
    .ADDR DynTileBuf
    .ADDR BlankTextBoxLines
    .ADDR GhostPaletteRow7TransferBuf
    .ADDR GreenBgPaletteRow7TransferBuf
    .ADDR BrownBgPaletteRow7TransferBuf
    .ADDR CellarAttrsTransferBuf
    .ADDR DynTileBuf
    .ADDR BlankPersonWares
    .ADDR Mode11DeadLinkPalette
    .ADDR LevelNumberTransferBuf
    .ADDR InventoryTextTransferBuf
    .ADDR SubmenuBoxesTopsTransferBuf
    .ADDR SubmenuBoxesSidesTransferBuf
    .ADDR GanonPaletteRow7TransferBuf
    .ADDR SelectedItemBoxBottomTransferBuf
    .ADDR UseBButtonTextTransferBuf
    .ADDR InventoryBoxBottomTransferBuf
    .ADDR CaveBgPaletteRowsTransferBuf
    .ADDR SubmenuMapRemainderTransferBuf
    .ADDR SheetMapBottomEdgeTransferBuf
    .ADDR LevelInfo_StatusBarMapTransferBuf
    .ADDR GameOverTransferBuf
    .ADDR SubmenuAttrs1TransferBuf
    .ADDR SubmenuAttrs2TransferBuf
    .ADDR BlankBottomRowNT2TransferBuf
    .ADDR BlankRowTransferBuf
    .ADDR SubmenuTriforceApexTransferBuf
    .ADDR TriforceRow0TransferBuf
    .ADDR TriforceRow1TransferBuf
    .ADDR TriforceRow2TransferBuf
    .ADDR TriforceRow3TransferBuf
    .ADDR SubmenuTriforceBottomTransferBuf
    .ADDR TriforceTextTransferBuf
    .ADDR Mode11BackgroundPaletteBottomHalfTransferBuf
    .ADDR Mode11PlayAreaAttrsTopHalfTransferBuf
    .ADDR Mode11PlayAreaAttrsBottomHalfTransferBuf
    .ADDR DynTileBuf
    .ADDR DynTileBuf
    .ADDR DynTileBuf
    .ADDR EndingPaletteTransferBuf
    .ADDR BombCapacityPriceTextTransferBuf
    .ADDR DynTileBuf
    .ADDR DynTileBuf
    .ADDR DynTileBuf
    .ADDR DynTileBuf
    .ADDR LifeOrMoneyCostTextTransferBuf
    .ADDR WhitePaletteBottomHalfTransferBuf
    .ADDR RedArmosPaletteRow7TransferBuf
    .ADDR GleeokPaletteRow7TransferBuf
    .ADDR DynTileBuf

TransferCurTileBuf:
    LDX TileBufSelector
    LDA TransferBufAddrs, X
    STA $00
    LDA TransferBufAddrs+1, X
    STA $01
    JSR TransferTileBuf

    ; UNKNOWN:
    ; $3D is the maximum size of the dynamic transfer buf. Is this
    ; $3F related? [0300] is only ever written.
    LDA #$3F
    STA $0300
    LDX #$00
    STX TileBufSelector
    STX SwitchNameTablesReq
    STX DynTileBufLen
    DEX
    STX DynTileBuf              ; Empty the tile buffer.
    RTS

ContinueTransferTileBuf:
    ;
    ;
    ; Save VRAM address high byte.
    PHA
    STA PpuAddr_2006
    INY
    LDA ($00), Y                ; Read low byte of VRAM address.
    STA PpuAddr_2006
    INY
    LDA ($00), Y                ; Read count and attribute byte.
    ASL
    PHA
    LDA CurPpuControl_2000
    ORA #$04
    BCS :+                      ; If high bit is set, then auto-increment VRAM address by 32.
    AND #$FB
:
    STA PpuControl_2000
    STA CurPpuControl_2000
    PLA
    ASL
    PHP
    BCC :+                      ; If bit 6 is set, then repeat one tile.
    ORA #$02
    INY                         ; increment Y index to point to first byte of text.
:
    PLP

    ; If the count was 0 (bottom 6 bits),
    ; then make it 64.
    CLC
    BNE :+
    SEC
:
    ROR
    LSR

    ; We pulled the flags out, and we're left with a count in A.
    ; Move it to X.
    TAX

@Loop:
    BCS :+                      ; If the original bit 6 is clear,
    INY                         ; then increment Y index (not repeating).
:
    LDA ($00), Y
    STA PpuData_2007
    DEX
    BNE @Loop
    PLA                         ; Restore VRAM address high byte.

    ; If we wrote to $3Fxx, then set PPUADDR to $3F00, then $0000.
    CMP #$3F
    BNE @AdvanceSource
    STA PpuAddr_2006
    STX PpuAddr_2006
    STX PpuAddr_2006
    STX PpuAddr_2006

@AdvanceSource:
    ; Advance the source address to one after the last byte read.
    SEC
    TYA
    ADC $00
    STA $00
    LDA #$00
    ADC $01
    STA $01

TransferTileBuf:
    LDX PpuStatus_2002
    LDY #$00
    LDA ($00), Y                ; Read high byte of VRAM address.
    BPL ContinueTransferTileBuf ; End when we read a negative VRAM address.
    RTS
