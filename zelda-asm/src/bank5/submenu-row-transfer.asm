; ============================================================================
; FILE:     bank5/submenu-row-transfer.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 6805-6917
; SUBJECTS: Submenu
; TOPIC:    Submenu row transfer while scrolling
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0020 SubmenuTransferBufSelectorsUW   data
;   L0024 SubmenuTransferBufSelectorsOW   data
;   L0029 Submenu_CueTransferRowUW        code
;   L0067 DecSubmenuScroll                code
;   L0071 L1725A_Exit                     code
;   L0074 PrepFullBlackRow                code
;   L0100 Submenu_CueTransferRowOW        code
;   L0125 SelectTransferBufAndDecCounter  code
;   L0131 L17295_Exit                     code
; ============================================================================

SubmenuTransferBufSelectorsUW:
    .BYTE $00, $00, $00, $30, $32, $34, $38, $3A
    .BYTE $3C, $00, $00, $00, $40

SubmenuTransferBufSelectorsOW:
    .BYTE $00, $00, $00, $30, $32, $34, $38, $3A
    .BYTE $3C, $00, $00, $00, $00, $00, $00, $50
    .BYTE $52, $54, $56, $58, $5A

Submenu_CueTransferRowUW:
    ; If menu scroll value is negative, then we transferred everything.
    ; So, return.
    LDA SubmenuScrollProgress
    BMI L1725A_Exit

    ; Shift right. The value in A is now (menu scroll value) / 2,
    ; and represents the current row of the submenu being processed.
    ; The bottom bit tells us (1) whether to transfer a full row of
    ; black tiles, or (0) a row of the map or other piece of the GUI.
    ;
    ; If the bottom bit is set, go prepare the full black row.
    LSR
    TAY
    BCS PrepFullBlackRow

    ; Submenu rows $D to $15 are parts of the map.
    ; Submenu rows 0 to $C are fixed text and boxes.
    ;
    ; If submenu row < $D, cue a transfer of the static transfer
    ; buffer for this row.
    CMP #$0D
    BCS :+
    LDA SubmenuTransferBufSelectorsUW, Y

@SelectTransferBuf:
    JMP SelectTransferBufAndDecCounter
:
    ; If map row = $15, go cue the transfer of the bottom edge
    ; of the big map sheet.
    CMP #$15
    BNE :+
    LDA #$42
    JMP @SelectTransferBuf
:
    ; Else row is between $D and $14. Prepare a map row.
    JSR Submenu_WriteSheetMapRowTransferRecord

DecSubmenuScroll:
    ; Decrease the counter for the next frame.
    DEC SubmenuScrollProgress

L1725A_Exit:
    RTS

PrepFullBlackRow:
    ; Prepare the dynamic transfer buf with a row of blank ($24) tiles
    ; to transfer to row (7 + Y).
    ;
    ; Calculate PPU address ($28E0 + Y*$20).
    LDA #$28
    STA DynTileBuf
    LDA #$C0

@Add20H:
    CLC
    ADC #$20
    BCC :+
    INC DynTileBuf              ; High PPU address
:
    DEY
    BPL @Add20H
    STA DynTileBuf+1            ; Low PPU address
    LDA #$60
    STA DynTileBuf+2            ; Repeating 1 byte $20 times
    LDA #$24
    STA DynTileBuf+3            ; Blank tile
    LDA #$FF
    STA DynTileBuf+4            ; End marker
    JMP DecSubmenuScroll

Submenu_CueTransferRowOW:
    ;
    ;
    ; If menu scroll value is negative, then we transferred everything.
    ; So, return.
    LDA SubmenuScrollProgress
    BMI L17295_Exit

    ; Shift right. The value in A is now (menu scroll value) / 2,
    ; and represents the current row of the submenu being processed.
    ; The bottom bit tells us (1) whether to transfer a full row of
    ; black tiles, or (0) a row of the triforce or other piece of the GUI.
    ;
    ; If the bottom bit is set, go prepare the full black row.
    LSR
    TAY
    BCS PrepFullBlackRow

    ; There's nothing to do for row $15.
    ; If submenu row < $15, cue a transfer of a triforce or other
    ; transfer buffer for this row.
    CMP #$15
    BCS :+
    LDA SubmenuTransferBufSelectorsOW, Y

SelectTransferBufAndDecCounter:
    STA TileBufSelector
:
    ; Decrease the counter for the next frame.
    DEC SubmenuScrollProgress

L17295_Exit:
    RTS
