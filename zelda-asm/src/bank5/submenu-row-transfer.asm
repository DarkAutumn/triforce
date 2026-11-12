; ============================================================================
; FILE:     bank5/submenu-row-transfer.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 6805-6917
; SUBJECTS: Submenu
; TOPIC:    Draws the item submenu into name table 2 while it scrolls in: each call sends one
;           tile row, either a row of blank tiles or a fixed piece (text, boxes, Triforce in the
;           overworld, the map sheet in a dungeon).
; SEE ALSO: bank5/submenu-states.asm (the submenu state machine that calls these every frame and
;           sets SubmenuScrollProgress to $2B); bank5/submenu-map.asm
;           (Submenu_WriteSheetMapRowTransferRecord: dungeon map rows $0D-$14);
;           bank6/transfer-buf-dispatch.asm (TransferBufAddrs: what each selector value sends);
;           bank6/transfer-bufs-static.asm (the static buffers themselves).
; KEY FACTS:
;   - SubmenuScrollProgress counts $2B down to 0, one per call: bits 7-1 = submenu row ($15 down
;     to 0), bit 0 = 1 for the row of blank tiles, 0 for the row's fixed piece. Each row is first
;     blanked, then drawn, from the bottom row up: $2C calls. Submenu_CueTransferRowUW,
;     Submenu_CueTransferRowOW
;   - A blank row is $20 tiles $24 at PPU address $28E0 + row * $20. PrepFullBlackRow
;   - Rows 3-8 are the same in both worlds: inventory text, box tops, box sides, selected item
;     box bottom, "USE B BUTTON" text, inventory box bottom. SubmenuTransferBufSelectorsUW,
;     SubmenuTransferBufSelectorsOW
;   - Dungeon: row $0C = map remainder, rows $0D-$14 = the eight rows of the map sheet built from
;     the room flags, row $15 = the sheet's bottom edge. Submenu_CueTransferRowUW
;   - Overworld: rows $0F-$14 = the Triforce; row $15 has no piece. Submenu_CueTransferRowOW
;   - A table value of 0 means the row has no piece: TileBufSelector is set to 0, which selects
;     the dynamic buffer. SubmenuTransferBufSelectorsUW, SubmenuTransferBufSelectorsOW
; CONTENTS: (generated; L = line in this file)
;   L0041 SubmenuTransferBufSelectorsUW   data  TABLE: dungeon submenu pieces; index = submenu row $00-$0C; value = TileBufSelector (0 = none).
;   L0050 SubmenuTransferBufSelectorsOW   data  TABLE: overworld submenu pieces; index = submenu row $00-$14; value = TileBufSelector (0 = none).
;   L0068 Submenu_CueTransferRowUW        code  Dungeon: cues the next row of the submenu for transfer and counts SubmenuScrollProgress down.
;   L0095 DecSubmenuScroll                code  Shared tail of the row routines: counts SubmenuScrollProgress down by one.
;   L0099 L1725A_Exit                     code  Shared exit of Submenu_CueTransferRowUW.
;   L0109 PrepFullBlackRow                code  Writes a record that fills one submenu row with blank tiles, then counts the progress down.
;   L0140 Submenu_CueTransferRowOW        code  Overworld: cues the next row of the submenu for transfer and counts SubmenuScrollProgress down.
;   L0156 SelectTransferBufAndDecCounter  code  Cues static transfer buffer A (TileBufSelector) and counts SubmenuScrollProgress down.
;   L0162 L17295_Exit                     code  Shared exit of Submenu_CueTransferRowOW.
; ============================================================================

; TABLE: dungeon submenu pieces; index = submenu row $00-$0C; value = TileBufSelector (0 = none).
; Rows $0D-$15 are handled in code (Submenu_CueTransferRowUW).
SubmenuTransferBufSelectorsUW:
    ; Rows 0-2 none; 3 $30 InventoryTextTransferBuf; 4 $32 SubmenuBoxesTopsTransferBuf;
    ; 5 $34 SubmenuBoxesSidesTransferBuf; 6 $38 SelectedItemBoxBottomTransferBuf;
    ; 7 $3A UseBButtonTextTransferBuf.
    .BYTE $00, $00, $00, $30, $32, $34, $38, $3A
    ; Row 8 $3C InventoryBoxBottomTransferBuf; 9-$0B none; $0C $40 SubmenuMapRemainderTransferBuf.
    .BYTE $3C, $00, $00, $00, $40

; TABLE: overworld submenu pieces; index = submenu row $00-$14; value = TileBufSelector (0 = none).
SubmenuTransferBufSelectorsOW:
    ; Rows 0-7: as in SubmenuTransferBufSelectorsUW.
    .BYTE $00, $00, $00, $30, $32, $34, $38, $3A
    ; Row 8 $3C InventoryBoxBottomTransferBuf; 9-$0E none; $0F $50 SubmenuTriforceApexTransferBuf.
    .BYTE $3C, $00, $00, $00, $00, $00, $00, $50
    ; Rows $10-$13: $52, $54, $56, $58 TriforceRow0TransferBuf to TriforceRow3TransferBuf;
    ; $14: $5A SubmenuTriforceBottomTransferBuf.
    .BYTE $52, $54, $56, $58, $5A

; ----------------------------------------------------------------------------
; Dungeon: cues the next row of the submenu for transfer and counts SubmenuScrollProgress down.
; IN:   SubmenuScrollProgress (negative = everything was sent: return)
; OUT:  TileBufSelector set, or a record written at the start of DynTileBuf; SubmenuScrollProgress
;       decremented.
; NOTE: Row = progress / 2. Odd progress: a row of blank tiles (PrepFullBlackRow). Even progress:
;       rows $00-$0C from SubmenuTransferBufSelectorsUW; rows $0D-$14 one row of the map sheet
;       (Submenu_WriteSheetMapRowTransferRecord); row $15 the sheet's bottom edge (selector $42).
; ----------------------------------------------------------------------------
Submenu_CueTransferRowUW:
    LDA SubmenuScrollProgress
    BMI L1725A_Exit             ; Negative: all rows were sent. Return.

    ; A = Y = submenu row (progress / 2); C = bit 0: 1 = send the row of blank tiles.
    LSR
    TAY
    BCS PrepFullBlackRow

    ; Rows $00-$0C are fixed text and boxes: take the static buffer selector from the table.
    CMP #$0D
    BCS @CheckBottomEdge
    LDA SubmenuTransferBufSelectorsUW, Y

@SelectTransferBuf:
    JMP SelectTransferBufAndDecCounter
@CheckBottomEdge:
    ; Row $15: the bottom edge of the map sheet.
    CMP #$15
    BNE @WriteMapRow
    LDA #$42                    ; Selector $42 = SheetMapBottomEdgeTransferBuf.
    JMP @SelectTransferBuf
@WriteMapRow:
    ; Rows $0D-$14: build one row of the map sheet in DynTileBuf.
    JSR Submenu_WriteSheetMapRowTransferRecord

; Shared tail of the row routines: counts SubmenuScrollProgress down by one.
DecSubmenuScroll:
    DEC SubmenuScrollProgress

; Shared exit of Submenu_CueTransferRowUW.
L1725A_Exit:
    RTS

; ----------------------------------------------------------------------------
; Writes a record that fills one submenu row with blank tiles, then counts the progress down.
; IN:   Y = submenu row ($00-$15)
; OUT:  DynTileBuf = PPU address $28E0 + row * $20 (name table 2), control $60 (repeat one tile
;       $20 times), tile $24, end marker $FF; SubmenuScrollProgress decremented. Y = $FF.
; NOTE: Written at offset 0 of DynTileBuf; DynTileBufLen is not updated.
; ----------------------------------------------------------------------------
PrepFullBlackRow:
    ; Address = $28C0 + $20 * (row + 1): add $20 once for each value of Y from row down to 0.
    LDA #$28
    STA DynTileBuf
    LDA #$C0

@Add20H:
    CLC
    ADC #$20
    BCC :+
    INC DynTileBuf              ; Carry into the address high byte.
:
    DEY
    BPL @Add20H
    STA DynTileBuf+1            ; Address low byte.
    LDA #$60
    STA DynTileBuf+2            ; Control: bit 6 = repeat one byte, count $20.
    LDA #$24
    STA DynTileBuf+3            ; The blank tile.
    LDA #$FF
    STA DynTileBuf+4            ; End marker.
    JMP DecSubmenuScroll

; ----------------------------------------------------------------------------
; Overworld: cues the next row of the submenu for transfer and counts SubmenuScrollProgress down.
; IN:   SubmenuScrollProgress (negative = everything was sent: return)
; OUT:  TileBufSelector set, or a blank row record in DynTileBuf; SubmenuScrollProgress
;       decremented.
; NOTE: Row = progress / 2. Odd progress: a row of blank tiles (PrepFullBlackRow). Even progress:
;       rows $00-$14 from SubmenuTransferBufSelectorsOW; row $15 sends nothing.
; ----------------------------------------------------------------------------
Submenu_CueTransferRowOW:
    LDA SubmenuScrollProgress
    BMI L17295_Exit             ; Negative: all rows were sent. Return.

    ; A = Y = submenu row (progress / 2); C = bit 0: 1 = send the row of blank tiles.
    LSR
    TAY
    BCS PrepFullBlackRow

    ; Row $15 has no piece: only count down (the unnamed label below). Rows $00-$14: take the
    ; static buffer selector from the table.
    CMP #$15
    BCS :+
    LDA SubmenuTransferBufSelectorsOW, Y

; Cues static transfer buffer A (TileBufSelector) and counts SubmenuScrollProgress down.
SelectTransferBufAndDecCounter:
    STA TileBufSelector
:
    DEC SubmenuScrollProgress

; Shared exit of Submenu_CueTransferRowOW.
L17295_Exit:
    RTS
