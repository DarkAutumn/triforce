; ============================================================================
; FILE:     bank6/transfer-buf-dispatch.asm BANK: 6 (switchable, $8000-$BFFF)
; UPSTREAM: Z_06.asm 485-653
; SUBJECTS: Transfer
; TOPIC:    How queued background, attribute and palette data reaches the PPU: the table mapping
;           TileBufSelector to a transfer buffer, and the loop that the NMI handler runs once per frame
;           to write that buffer's records to PPU memory.
; SEE ALSO: bank7/run-game-nmi.asm (Nmi_Isr calls Transfer_SendCurBuf every frame); Variables.inc
;           (TileBufSelector, DynTileBuf, DynTileBufLen, SwitchNameTablesReq);
;           bank6/transfer-bufs-static.asm and bank6/level-blocks-common-data.asm (the static buffers).
; KEY FACTS:
;   - One transfer buffer is sent per frame, in the NMI: the one TileBufSelector selects (an even byte
;     offset into Transfer_BufAddrs, $00-$7E). Transfer_SendCurBuf.
;   - After every transfer TileBufSelector returns to 0 (the dynamic buffer DynTileBuf) and DynTileBuf
;     is emptied, even when a static buffer was sent: records queued in DynTileBuf on a frame whose
;     selector is not 0 are discarded unsent. Transfer_SendCurBuf.
;   - Record format: PPU address high, low; control byte (bit 7 = address step 32 (write downwards)
;     instead of 1, bit 6 = repeat one data byte, bits 5-0 = count, 0 meaning 64); then count data bytes
;     (one if repeating). A byte with bit 7 set where an address high byte is expected ends the buffer.
;     Transfer_WriteBuf, Transfer_WriteRecord.
;   - After a record written to $3Fxx (palettes) the PPU address is set to $3F00 and then $0000.
;     Transfer_WriteRecord.
;   - The address-step choice is kept in CurPpuControl bit 2 after the transfer. Transfer_WriteRecord.
;   - 12 of the 64 selectors ($00, $1A, $1C, $28, $64-$68, $6E-$74, $7E) name the dynamic buffer.
;     Transfer_BufAddrs.
; CONTENTS: (generated; L = line in this file)
;   L0042 Transfer_BufAddrs     data  TABLE: transfer buffer addresses; index = TileBufSelector (even, $00-$7E); value = buffer address.
;   L0116 Transfer_SendCurBuf   code  Sends the transfer buffer chosen by TileBufSelector to the PPU, then resets the transfer state.
;   L0141 Transfer_WriteRecord  code  Writes one transfer record (A = PPU address high byte, Y = 0), then falls into Transfer_WriteBuf.
;   L0217 Transfer_WriteBuf     code  Writes the transfer records at [00:01] to the PPU until a byte with bit 7 set is found.
; ============================================================================

.SEGMENT "BANK_06_DLIST"


.EXPORT Transfer_SendCurBuf


; TABLE: transfer buffer addresses; index = TileBufSelector (even, $00-$7E); value = buffer address.
; A buffer is a list of transfer records. Entries in RAM: DynTileBuf, LevelInfo_* (level info) and the
; common data block buffers at $67F0-$687D; the rest are in this bank's ROM.
Transfer_BufAddrs:
    .ADDR DynTileBuf                            ; $00 dynamic buffer (the default)
    .ADDR Transfer_StoryTileAttrBuf              ; $02 story screen tiles and attributes
    .ADDR Transfer_Mode8TextBuf                   ; $04 CONTINUE / SAVE / RETRY menu
    .ADDR Transfer_LevelPaletteRow7Buf           ; $06 sprite palette row 7 from the level's palettes
    .ADDR Transfer_AquamentusPaletteRow7Buf      ; $08 row 7 for Aquamentus
    .ADDR Transfer_OrangeBossPaletteRow7Buf      ; $0A row 7 for Digdogger and Dodongo
    .ADDR Transfer_LevelNumberBuf                ; $0C "LEVEL-n"
    .ADDR Transfer_StatusBarStaticsBuf           ; $0E status bar frame: B and A boxes, -LIFE-
    .ADDR Transfer_GameTitleBuf                  ; $10 title screen tiles and attributes
    .ADDR Transfer_MenuPalettesBuf               ; $12 file menu palettes
    .ADDR Transfer_Mode1TilesBuf                  ; $14 file select screen
    .ADDR Transfer_ModeEandFCharBoardBuf                 ; $16 register/eliminate character board
    .ADDR LevelInfo_PalettesTransferBuf         ; $18 the level's palettes (level info in RAM)
    .ADDR DynTileBuf                            ; $1A dynamic buffer
    .ADDR DynTileBuf                            ; $1C dynamic buffer
    .ADDR Transfer_BlankTextLines12Buf                     ; $1E blank person text lines 1 and 2 ($21A4, $21C4)
    .ADDR Transfer_GhostPaletteRow7Buf           ; $20 row 7: gravestone ($65); armos $66, odd inner palette
    .ADDR Transfer_GreenBgPaletteRow7Buf         ; $22 row 7: rock ($62) or armos $66, even inner palette
    .ADDR Transfer_BrownBgPaletteRow7Buf         ; $24 row 7: rock ($62), odd inner palette
    .ADDR Transfer_CellarAttrsBuf                ; $26 cellar play area attributes
    .ADDR DynTileBuf                            ; $28 dynamic buffer
    .ADDR Transfer_BlankTextLine3AndPricesBuf                      ; $2A blank text line 3 ($21E4) and the price row ($22C8)
    .ADDR Transfer_Mode11DeadLinkPaletteBuf                 ; $2C Link's palette (row 4) while dying
    .ADDR Transfer_LevelNumberBuf                ; $2E "LEVEL-n" (same buffer as $0C)
    .ADDR Transfer_InventoryTextBuf              ; $30 submenu: INVENTORY
    .ADDR Transfer_SubmenuBoxesTopsBuf           ; $32 submenu: box tops
    .ADDR Transfer_SubmenuBoxesSidesBuf          ; $34 submenu: box sides
    .ADDR Transfer_GanonPaletteRow7Buf           ; $36 row 7 for Ganon
    .ADDR Transfer_SelectedItemBoxBottomBuf      ; $38 submenu: selected item box bottom
    .ADDR Transfer_UseBButtonTextBuf             ; $3A submenu: USE B BUTTON
    .ADDR Transfer_InventoryBoxBottomBuf         ; $3C submenu: FOR THIS, inventory box bottom
    .ADDR Transfer_CaveBgPaletteRowsBuf          ; $3E background palette rows 2-3 for caves
    .ADDR Transfer_SubmenuMapTextsAndSheetTopBuf        ; $40 submenu (dungeon): COMPASS, MAP, map sheet top
    .ADDR Transfer_SubmenuMapSheetBottomEdgeBuf         ; $42 submenu (dungeon): map sheet bottom edge
    .ADDR LevelInfo_StatusBarMapTransferBuf     ; $44 status bar map (level info in RAM)
    .ADDR Transfer_GameOverBuf                   ; $46 GAME OVER
    .ADDR Transfer_SubmenuAttrs1Buf              ; $48 submenu attributes, part 1
    .ADDR Transfer_SubmenuAttrs2Buf              ; $4A submenu attributes, part 2
    .ADDR Transfer_BlankBottomRowNT2Buf          ; $4C blank row at $2BA0
    .ADDR Transfer_BlankRow7NT2Buf                   ; $4E blank row at $28E0
    .ADDR Transfer_SubmenuTriforceApexBuf        ; $50 submenu (overworld): Triforce outline apex
    .ADDR Transfer_TriforceRow0Buf               ; $52 Triforce outline row 0 (with held pieces)
    .ADDR Transfer_TriforceRow1Buf               ; $54 row 1
    .ADDR Transfer_TriforceRow2Buf               ; $56 row 2
    .ADDR Transfer_TriforceRow3Buf               ; $58 row 3
    .ADDR Transfer_SubmenuTriforceBottomBuf      ; $5A Triforce outline bottom edge
    .ADDR Transfer_TriforceTextBuf               ; $5C TRIFORCE
    .ADDR Transfer_Mode11BackgroundPaletteBottomHalfBuf  ; $5E death: background palette rows 2-3
    .ADDR Transfer_Mode11PlayAreaAttrsTopHalfBuf ; $60 death: play area attributes, upper part
    .ADDR Transfer_Mode11PlayAreaAttrsBottomHalfBuf  ; $62 death: play area attributes, lower part
    .ADDR DynTileBuf                            ; $64 dynamic buffer
    .ADDR DynTileBuf                            ; $66 dynamic buffer
    .ADDR DynTileBuf                            ; $68 dynamic buffer
    .ADDR Transfer_EndingPaletteBuf              ; $6A ending: background rows 2-3 and sprite row 7
    .ADDR Transfer_BombCapacityPriceTextBuf      ; $6C price text "-100" (more-bombs offer)
    .ADDR DynTileBuf                            ; $6E dynamic buffer
    .ADDR DynTileBuf                            ; $70 dynamic buffer
    .ADDR DynTileBuf                            ; $72 dynamic buffer
    .ADDR DynTileBuf                            ; $74 dynamic buffer
    .ADDR Transfer_LifeOrMoneyCostTextBuf        ; $76 text "-1" and "-50" (life-or-money man)
    .ADDR Transfer_WhitePaletteBottomHalfBuf     ; $78 background rows 2-3 all white ($30)
    .ADDR Transfer_DefaultOWPaletteRow7Buf        ; $7A row 7: every other overworld room (Mode5Play_Init)
    .ADDR Transfer_GleeokPaletteRow7Buf          ; $7C row 7 for Gleeok types $43-$45
    .ADDR DynTileBuf                            ; $7E dynamic buffer

; ----------------------------------------------------------------------------
; Sends the transfer buffer chosen by TileBufSelector to the PPU, then resets the transfer state.
; IN:   TileBufSelector = even offset into Transfer_BufAddrs; called from Nmi_Isr with bank 6 switched in.
; OUT:  TileBufSelector = 0; SwitchNameTablesReq = 0; DynTileBufLen = 0; DynTileBuf[0] = $FF (empty);
;       $0300 = $3F; [00:01] = address of the buffer's end marker.
; NOTE: The dynamic buffer is emptied even when another buffer was sent, so a frame that selects a
;       static buffer loses whatever was queued in DynTileBuf.
; ----------------------------------------------------------------------------
Transfer_SendCurBuf:
    LDX TileBufSelector
    LDA Transfer_BufAddrs, X
    STA $00
    LDA Transfer_BufAddrs+1, X
    STA $01
    JSR Transfer_WriteBuf

    ; Store $3F in $0300, the byte before DynTileBufLen. Nothing reads $0300: the only other
    ; instructions that name it are two more stores (the demo palette setups, value $23), and the
    ; only indexed array below it, Sprites, is never indexed past $02FF. A dead store.
    ; (trace) Overwriting $0300 with a random value before every frame for 1500 frames of play (two
    ; dungeon rooms with the scroll between them; an overworld room) changed no other RAM byte.
    ; What the byte was meant for is not in the code.
    LDA #$3F
    STA $0300
    LDX #$00
    STX TileBufSelector
    STX SwitchNameTablesReq
    STX DynTileBufLen
    DEX
    STX DynTileBuf              ; $FF = end marker in the first byte: the dynamic buffer is empty.
    RTS

; Writes one transfer record (A = PPU address high byte, Y = 0), then falls into Transfer_WriteBuf.
Transfer_WriteRecord:
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
; NOTE: Only caller: Transfer_SendCurBuf, from the NMI handler. Record: address high, low, control (bit
;       7 step 32, bit 6 repeat one byte, bits 5-0 count, 0 = 64), data. Loops through
;       Transfer_WriteRecord above, once per record.
; ----------------------------------------------------------------------------
Transfer_WriteBuf:
    LDX PpuStatus               ; Reading the status resets the PPU address latch.
    LDY #$00
    LDA ($00), Y                ; PPU address high byte, or the end marker.
    BPL Transfer_WriteRecord ; Bit 7 clear: a record follows. Bit 7 set: done.
    RTS
