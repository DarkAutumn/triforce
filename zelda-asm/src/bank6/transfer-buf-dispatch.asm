; ============================================================================
; FILE:     bank6/transfer-buf-dispatch.asm BANK: 6 (switchable, $8000-$BFFF)
; UPSTREAM: Z_06.asm 485-653
; SUBJECTS: (none)
; TOPIC:    How queued background, attribute and palette data reaches the PPU: the table mapping
;           TileBufSelector to a transfer buffer, and the loop that the NMI handler runs once per frame
;           to write that buffer's records to PPU memory.
; SEE ALSO: bank7/run-game-nmi.asm (IsrNmi calls TransferCurTileBuf every frame); Variables.inc
;           (TileBufSelector, DynTileBuf, DynTileBufLen, SwitchNameTablesReq);
;           bank6/transfer-bufs-static.asm and bank6/level-blocks-common-data.asm (the static buffers).
; KEY FACTS:
;   - One transfer buffer is sent per frame, in the NMI: the one TileBufSelector selects (an even byte
;     offset into TransferBufAddrs, $00-$7E). TransferCurTileBuf.
;   - After every transfer TileBufSelector returns to 0 (the dynamic buffer DynTileBuf) and DynTileBuf
;     is emptied, even when a static buffer was sent: records queued in DynTileBuf on a frame whose
;     selector is not 0 are discarded unsent. TransferCurTileBuf.
;   - Record format: PPU address high, low; control byte (bit 7 = address step 32 (write downwards)
;     instead of 1, bit 6 = repeat one data byte, bits 5-0 = count, 0 meaning 64); then count data bytes
;     (one if repeating). A byte with bit 7 set where an address high byte is expected ends the buffer.
;     TransferTileBuf, ContinueTransferTileBuf.
;   - After a record written to $3Fxx (palettes) the PPU address is set to $3F00 and then $0000.
;     ContinueTransferTileBuf.
;   - The address-step choice is kept in CurPpuControl bit 2 after the transfer. ContinueTransferTileBuf.
;   - 12 of the 64 selectors ($00, $1A, $1C, $28, $64-$68, $6E-$74, $7E) name the dynamic buffer.
;     TransferBufAddrs.
; CONTENTS: (generated; L = line in this file)
;   L0042 TransferBufAddrs         data  TABLE: transfer buffer addresses; index = TileBufSelector (even, $00-$7E); value = buffer address.
;   L0116 TransferCurTileBuf       code  Sends the transfer buffer chosen by TileBufSelector to the PPU, then resets the transfer state.
;   L0138 ContinueTransferTileBuf  code  Writes one transfer record (A = PPU address high byte, Y = 0), then falls into TransferTileBuf.
;   L0214 TransferTileBuf          code  Writes the transfer records at [00:01] to the PPU until a byte with bit 7 set is found.
; ============================================================================

.SEGMENT "BANK_06_DLIST"


.EXPORT TransferCurTileBuf


; TABLE: transfer buffer addresses; index = TileBufSelector (even, $00-$7E); value = buffer address.
; A buffer is a list of transfer records. Entries in RAM: DynTileBuf, LevelInfo_* (level info) and the
; common data block buffers at $67F0-$687D; the rest are in this bank's ROM.
TransferBufAddrs:
    .ADDR DynTileBuf                            ; $00 dynamic buffer (the default)
    .ADDR StoryTileAttrTransferBuf              ; $02 story screen tiles and attributes
    .ADDR Mode8TextTileBuffer                   ; $04 CONTINUE / SAVE / RETRY menu
    .ADDR LevelPaletteRow7TransferBuf           ; $06 sprite palette row 7 from the level's palettes
    .ADDR AquamentusPaletteRow7TransferBuf      ; $08 row 7 for Aquamentus
    .ADDR OrangeBossPaletteRow7TransferBuf      ; $0A row 7 for Digdogger and Dodongo
    .ADDR LevelNumberTransferBuf                ; $0C "LEVEL-n"
    .ADDR StatusBarStaticsTransferBuf           ; $0E status bar frame: B and A boxes, -LIFE-
    .ADDR GameTitleTransferBuf                  ; $10 title screen tiles and attributes
    .ADDR MenuPalettesTransferBuf               ; $12 file menu palettes
    .ADDR Mode1TileTransferBuf                  ; $14 file select screen
    .ADDR ModeFCharsTransferBuf                 ; $16 register/eliminate character board
    .ADDR LevelInfo_PalettesTransferBuf         ; $18 the level's palettes (level info in RAM)
    .ADDR DynTileBuf                            ; $1A dynamic buffer
    .ADDR DynTileBuf                            ; $1C dynamic buffer
    .ADDR BlankTextBoxLines                     ; $1E blank person text lines 1 and 2 ($21A4, $21C4)
    .ADDR GhostPaletteRow7TransferBuf           ; $20 row 7: gravestone ($65); armos $66, odd inner palette
    .ADDR GreenBgPaletteRow7TransferBuf         ; $22 row 7: rock ($62) or armos $66, even inner palette
    .ADDR BrownBgPaletteRow7TransferBuf         ; $24 row 7: rock ($62), odd inner palette
    .ADDR CellarAttrsTransferBuf                ; $26 cellar play area attributes
    .ADDR DynTileBuf                            ; $28 dynamic buffer
    .ADDR BlankPersonWares                      ; $2A blank text line 3 ($21E4) and the price row ($22C8)
    .ADDR Mode11DeadLinkPalette                 ; $2C Link's palette (row 4) while dying
    .ADDR LevelNumberTransferBuf                ; $2E "LEVEL-n" (same buffer as $0C)
    .ADDR InventoryTextTransferBuf              ; $30 submenu: INVENTORY
    .ADDR SubmenuBoxesTopsTransferBuf           ; $32 submenu: box tops
    .ADDR SubmenuBoxesSidesTransferBuf          ; $34 submenu: box sides
    .ADDR GanonPaletteRow7TransferBuf           ; $36 row 7 for Ganon
    .ADDR SelectedItemBoxBottomTransferBuf      ; $38 submenu: selected item box bottom
    .ADDR UseBButtonTextTransferBuf             ; $3A submenu: USE B BUTTON
    .ADDR InventoryBoxBottomTransferBuf         ; $3C submenu: FOR THIS, inventory box bottom
    .ADDR CaveBgPaletteRowsTransferBuf          ; $3E background palette rows 2-3 for caves
    .ADDR SubmenuMapRemainderTransferBuf        ; $40 submenu (dungeon): COMPASS, MAP, map sheet top
    .ADDR SheetMapBottomEdgeTransferBuf         ; $42 submenu (dungeon): map sheet bottom edge
    .ADDR LevelInfo_StatusBarMapTransferBuf     ; $44 status bar map (level info in RAM)
    .ADDR GameOverTransferBuf                   ; $46 GAME OVER
    .ADDR SubmenuAttrs1TransferBuf              ; $48 submenu attributes, part 1
    .ADDR SubmenuAttrs2TransferBuf              ; $4A submenu attributes, part 2
    .ADDR BlankBottomRowNT2TransferBuf          ; $4C blank row at $2BA0
    .ADDR BlankRowTransferBuf                   ; $4E blank row at $28E0
    .ADDR SubmenuTriforceApexTransferBuf        ; $50 submenu (overworld): Triforce outline apex
    .ADDR TriforceRow0TransferBuf               ; $52 Triforce outline row 0 (with held pieces)
    .ADDR TriforceRow1TransferBuf               ; $54 row 1
    .ADDR TriforceRow2TransferBuf               ; $56 row 2
    .ADDR TriforceRow3TransferBuf               ; $58 row 3
    .ADDR SubmenuTriforceBottomTransferBuf      ; $5A Triforce outline bottom edge
    .ADDR TriforceTextTransferBuf               ; $5C TRIFORCE
    .ADDR Mode11BackgroundPaletteBottomHalfTransferBuf  ; $5E death: background palette rows 2-3
    .ADDR Mode11PlayAreaAttrsTopHalfTransferBuf ; $60 death: play area attributes, upper part
    .ADDR Mode11PlayAreaAttrsBottomHalfTransferBuf  ; $62 death: play area attributes, lower part
    .ADDR DynTileBuf                            ; $64 dynamic buffer
    .ADDR DynTileBuf                            ; $66 dynamic buffer
    .ADDR DynTileBuf                            ; $68 dynamic buffer
    .ADDR EndingPaletteTransferBuf              ; $6A ending: background rows 2-3 and sprite row 7
    .ADDR BombCapacityPriceTextTransferBuf      ; $6C price text "-100" (more-bombs offer)
    .ADDR DynTileBuf                            ; $6E dynamic buffer
    .ADDR DynTileBuf                            ; $70 dynamic buffer
    .ADDR DynTileBuf                            ; $72 dynamic buffer
    .ADDR DynTileBuf                            ; $74 dynamic buffer
    .ADDR LifeOrMoneyCostTextTransferBuf        ; $76 text "-1" and "-50" (life-or-money man)
    .ADDR WhitePaletteBottomHalfTransferBuf     ; $78 background rows 2-3 all white ($30)
    .ADDR RedArmosPaletteRow7TransferBuf        ; $7A row 7: every other overworld room (InitMode5Play)
    .ADDR GleeokPaletteRow7TransferBuf          ; $7C row 7 for Gleeok types $43-$45
    .ADDR DynTileBuf                            ; $7E dynamic buffer

; ----------------------------------------------------------------------------
; Sends the transfer buffer chosen by TileBufSelector to the PPU, then resets the transfer state.
; IN:   TileBufSelector = even offset into TransferBufAddrs; called from IsrNmi with bank 6 switched in.
; OUT:  TileBufSelector = 0; SwitchNameTablesReq = 0; DynTileBufLen = 0; DynTileBuf[0] = $FF (empty);
;       $0300 = $3F; [00:01] = address of the buffer's end marker.
; NOTE: The dynamic buffer is emptied even when another buffer was sent, so a frame that selects a
;       static buffer loses whatever was queued in DynTileBuf.
; ----------------------------------------------------------------------------
TransferCurTileBuf:
    LDX TileBufSelector
    LDA TransferBufAddrs, X
    STA $00
    LDA TransferBufAddrs+1, X
    STA $01
    JSR TransferTileBuf

    ; Store $3F in $0300, the byte before DynTileBufLen. The only other writes to $0300 found are the
    ; two demo palette setups (value $23); no instruction reads it by address, so its purpose is
    ; (unverified).
    LDA #$3F
    STA $0300
    LDX #$00
    STX TileBufSelector
    STX SwitchNameTablesReq
    STX DynTileBufLen
    DEX
    STX DynTileBuf              ; $FF = end marker in the first byte: the dynamic buffer is empty.
    RTS

; Writes one transfer record (A = PPU address high byte, Y = 0), then falls into TransferTileBuf.
ContinueTransferTileBuf:
    ; Save the PPU address high byte for the palette test below.
    PHA
    STA PpuAddr
    INY
    LDA ($00), Y                ; PPU address low byte.
    STA PpuAddr
    INY
    LDA ($00), Y                ; Control byte: bit 7 step 32, bit 6 repeat, bits 5-0 count.
    ASL
    PHA
    LDA CurPpuControl
    ORA #$04
    BCS :+                      ; Bit 7 set: PPU address step 32 (write down a column); else step 1.
    AND #$FB
:
    STA PpuControl
    STA CurPpuControl
    PLA
    ASL
    PHP
    BCC :+                      ; Bit 6 set: repeat one data byte. Flag it in bit 1 of A, and
    ORA #$02
    INY                         ; point Y at that data byte now (it will not advance in the loop).
:
    PLP

    ; A = count * 4 (+2 if repeating); Z (from the shift) is set when the count is 0. The two shifts
    ; below leave the count in A and the repeat flag in carry; a count of 0 becomes 64 by shifting the
    ; carry set here into bit 6.
    CLC
    BNE :+
    SEC
:
    ROR
    LSR

    ; X = number of bytes to write (1-64); carry = repeat flag, unchanged through the loop.
    TAX

@Loop:
    BCS :+                      ; Repeating: keep reading the same byte.
    INY                         ; Not repeating: advance to the next data byte.
:
    LDA ($00), Y
    STA PpuData
    DEX
    BNE @Loop
    PLA                         ; PPU address high byte of this record.

    ; After a palette write ($3Fxx): set the PPU address to $3F00, then $0000 (X is 0 here).
    CMP #$3F
    BNE @AdvanceSource
    STA PpuAddr
    STX PpuAddr
    STX PpuAddr
    STX PpuAddr

@AdvanceSource:
    ; [00:01] += Y + 1: the byte after the last one read (Y = offset of that byte).
    SEC
    TYA
    ADC $00
    STA $00
    LDA #$00
    ADC $01
    STA $01

; ----------------------------------------------------------------------------
; Writes the transfer records at [00:01] to the PPU until a byte with bit 7 set is found.
; IN:   [00:01] = address of the first record.
; OUT:  [00:01] = address of the end marker; CurPpuControl bit 2 as set by the last record.
; NOTE: Only caller: TransferCurTileBuf, from the NMI handler. Record: address high, low, control (bit
;       7 step 32, bit 6 repeat one byte, bits 5-0 count, 0 = 64), data. Loops through
;       ContinueTransferTileBuf above, once per record.
; ----------------------------------------------------------------------------
TransferTileBuf:
    LDX PpuStatus               ; Reading the status resets the PPU address latch.
    LDY #$00
    LDA ($00), Y                ; PPU address high byte, or the end marker.
    BPL ContinueTransferTileBuf ; Bit 7 clear: a record follows. Bit 7 set: done.
    RTS
