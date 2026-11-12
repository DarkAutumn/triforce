; ============================================================================
; FILE:     bank2/credits.asm            BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 3672-4032
; SUBJECTS: (none)
; TOPIC:    Mode $13 update submode 3: the credits. The screen scrolls up half a pixel per frame and
;           one tile row (brick frame, text line, attributes) is drawn for every 8 pixels scrolled.
;           Holds the credits text lines and the tables that place them.
; SEE ALSO: bank2/mode13-win-game.asm (the jump table that reaches UpdateMode13WinGame_Sub3, and
;           submode 4 that follows); bank2/mode13-ending-init.asm (InitMode13_Sub4 presets
;           CreditsTileOffset); bank1/format-decimal-hearts.asm (FormatDecimalByte).
; KEY FACTS:
;   - Scroll speed is 0.5 pixel per frame: $80 is added to a fraction (VScrollAddrHi, reused) each
;     frame and the carry goes into CurVScroll and CreditsTileOffset. UpdateMode13WinGame_Sub3
;   - CurVScroll wraps at $F0; each wrap toggles the base nametable (SwitchNameTablesReq) and counts
;     one screen in VScrollAddrLo (reused; nothing in this file sets its start value).
;     UpdateMode13WinGame_Sub3
;   - The scroll ends, by the quest of the current save slot (QuestNumbers): Q1 at screen 2 with
;     CurVScroll $78, Q2 at screen 3 with CurVScroll 0. Then ObjTimer = $40 and submode 4 follows.
;     CreditsLastScreenList, CreditsLastVscrollList
;   - No controller input is read: the credits cannot be skipped. UpdateMode13WinGame_Sub3
;   - There are $17 text lines. Q1 shows lines $00-$0F; Q2 shows $00-$0B and $10-$16. DrawCredits
;   - Line $11 (Q2 only) gets the save slot's name (8 characters from SaveSlotNames, columns 9-16)
;     and its death count (DeathCounts, 3 decimal characters, columns 19-21). DrawCredits
;   - Which tile rows carry text is a bit mask per nametable page of 8 rows (6 rows in pages 3, 7
;     and $B), bit 7 = first row. CreditsPagesTextMasks
;   - Text line record: byte 0 = character count, byte 1 = starting column, then the tile codes.
;     CreditsTextLines, CreditsTextAddrsLo, CreditsTextAddrsHi
; CONTENTS: (generated; L = line in this file)
;   L0042 CreditsLastScreenList     data  TABLE: count of whole screens scrolled at which the credits end; index = 0 Q1, 1 Q2.
;   L0046 CreditsLastVscrollList    data  TABLE: CurVScroll within that last screen at which the credits end; index = 0 Q1, 1 Q2; pixels.
;   L0054 UpdateMode13WinGame_Sub3  code  Update submode 3: scrolls the credits 0.5 pixel per frame and draws a row per 8 pixels.
;   L0120 CreditLineVramAddrsHi     data  TABLE: high nametable address byte of a page of 8 tile rows; index = CreditsVramPage (0-$B).
;   L0125 CreditsPagesTextMasks     data  TABLE: which rows of a page take a text line; index = CreditsVramPage; bit 7 = row 0 .. bit 0 = row 7.
;   L0130 CreditsTextAddrsLo        data  TABLE: low address byte of each credits text line record; index = CreditsLineIndex ($00-$16).
;   L0134 CreditsTextAddrsHi        data  TABLE: high address byte of each credits text line record; index = CreditsLineIndex ($00-$16).
;   L0138 CreditsTextLines          data  DATA: credits text line records: character count, starting column, then that many tile codes.
;   L0142 CreditsAttrs              data  TABLE: attribute byte for the middle 6 blocks of an attribute row; index = CreditsRow / 4.
;   L0158 DrawCredits               code  Builds the transfer records for the next credits tile row: bricks, text line, attributes.
; ============================================================================

; TABLE: count of whole screens scrolled at which the credits end; index = 0 Q1, 1 Q2.
CreditsLastScreenList:
    .BYTE $02, $03

; TABLE: CurVScroll within that last screen at which the credits end; index = 0 Q1, 1 Q2; pixels.
CreditsLastVscrollList:
    .BYTE $78, $00

; ----------------------------------------------------------------------------
; Update submode 3: scrolls the credits 0.5 pixel per frame and draws a row per 8 pixels.
; OUT:  at the end position for the quest: GameSubmode + 1 and ObjTimer = $40.
; NOTE: VScrollAddrHi serves as the scroll fraction and VScrollAddrLo as the screen count here.
; ----------------------------------------------------------------------------
UpdateMode13WinGame_Sub3:
    ; After scrolling 8 pixels, draw another tile row.
    LDA CreditsTileOffset
    CMP #$08
    BMI :+
    LDA CreditsTileOffset
    SBC #$08
    STA CreditsTileOffset
    JSR DrawCredits
:
    ; Add $80 to the scroll speed fraction.
    LDA VScrollAddrHi
    CLC
    ADC #$80
    STA VScrollAddrHi

    ; Carry over to the tile offset and current V-scroll.
    BCC :+
    INC CreditsTileOffset
:
    LDA CurVScroll
    ADC #$00
    STA CurVScroll

    ; If we've reached the bottom of a nametable, then
    ; roll over current V-scroll to 0, and increase number of
    ; screens scrolled.
    CMP #$F0
    LDA #$00
    BCC :+
    STA CurVScroll
    INC a:VScrollAddrLo
:
    ; Roll the carry from the comparison above into bit 0.
    ; So, if reached the bottom of a nametable, switch nametables.
    ROL
    STA SwitchNameTablesReq

    ; Put the quest number in Y register.
    LDY #$00
    LDX CurSaveSlot
    LDA QuestNumbers, X
    BEQ :+
    INY
:
    ; If we're not showing the last screen, then return.
    LDA a:VScrollAddrLo
    CMP CreditsLastScreenList, Y
    BCC @Exit

    ; If we haven't scrolled the last amount in the last screen,
    ; then return.
    LDA CurVScroll
    CMP CreditsLastVscrollList, Y
    BCC @Exit

    ; But if we have, then go to the next submode, and set a timer
    ; to wait $40 frames in the next submode.
    INC GameSubmode
    LDA #$40
    STA ObjTimer

@Exit:
    RTS

; TABLE: high nametable address byte of a page of 8 tile rows; index = CreditsVramPage (0-$B).
CreditLineVramAddrsHi:
    .BYTE $28, $29, $2A, $2B, $20, $21, $22, $23
    .BYTE $28, $29, $2A, $2B

; TABLE: which rows of a page take a text line; index = CreditsVramPage; bit 7 = row 0 .. bit 0 = row 7.
CreditsPagesTextMasks:
    .BYTE $46, $10, $90, $84, $24, $30, $01, $48
    .BYTE $03, $25, $05, $40

; TABLE: low address byte of each credits text line record; index = CreditsLineIndex ($00-$16).
CreditsTextAddrsLo:
    .INCLUDE "dat/CreditsTextAddrsLo.inc"

; TABLE: high address byte of each credits text line record; index = CreditsLineIndex ($00-$16).
CreditsTextAddrsHi:
    .INCLUDE "dat/CreditsTextAddrsHi.inc"

; DATA: credits text line records: character count, starting column, then that many tile codes.
CreditsTextLines:
    .INCBIN "dat/CreditsTextLines.dat"

; TABLE: attribute byte for the middle 6 blocks of an attribute row; index = CreditsRow / 4.
CreditsAttrs:
    .BYTE $00, $AA, $FF, $FF, $55, $AA, $AA, $FF
    .BYTE $FF, $FF, $55, $00, $00, $00, $00, $00
    .BYTE $00, $00, $50, $00, $00, $00, $AA, $00

; One more byte without a label: it is CreditsAttrs+24, the entry DrawCredits would read for
; CreditsRow $60. If CreditsRow and the screen count start at 0 (not set in this file, (unverified)),
; the last row drawn is CreditsRow $5D in Q2 ($4E in Q1) and this byte is never read.
    .BYTE $00

; ----------------------------------------------------------------------------
; Builds the transfer records for the next credits tile row: bricks, text line, attributes.
; OUT:  DynTileBuf holds a $20-tile row record and, every 4th row, an 8-byte attribute record;
;       CreditsRow, CreditsVramLine, CreditsVramPage and CreditsLineIndex advanced.
; NOTE: CreditsRow skips $1E and $1F of every $20 (a nametable has 30 tile rows).
; ----------------------------------------------------------------------------
DrawCredits:
    ; Make a transfer record of a full row of blank tiles.
    LDY #$1F
    LDA #$24
:
    STA DynTileBuf+3, Y
    DEY
    BPL :-

    ; No wall tiles go in row 0.
    LDA CreditsRow
    BEQ @TerminateRecords

    ; Rows 1 and $2E get horizontal wall tiles.
    ; Rows 2 to $2D get side walls.
    CMP #$01
    BEQ @WriteHorizontalWalls
    CMP #$2E
    BCC @WriteSideWalls
    BNE @TerminateRecords

@WriteHorizontalWalls:
    LDY #$19
    LDA #$FA                    ; Wall bricks tile
:
    STA DynTileBuf+6, Y
    DEY
    BPL :-

@WriteSideWalls:
    LDA #$FA                    ; Wall bricks tile
    STA DynTileBuf+6
    STA DynTileBuf+31

@TerminateRecords:
    ; Put the buffer's end marker -- whether we transfer tiles and
    ; attributes or only tiles.
    LDA #$FF
    STA DynTileBuf+35           ; The end of the first record: tiles
    STA DynTileBuf+46           ; The end of the second record: NT attributes

    ; Specify that we're transferring $20 bytes/tiles.
    LDA #$20
    STA DynTileBuf+2

    ; Write the high byte of the current VRAM page.
    LDX CreditsVramPage
    LDA CreditLineVramAddrsHi, X
    STA DynTileBuf

    ; The line number will be used to shift the task mask below.
    LDA CreditsVramLine
    TAY

    ; Multiply the VRAM line number by $20 to get the low VRAM address.
    ASL
    ASL
    ASL
    ASL
    ASL
    STA DynTileBuf+1

    ; Get the mask for the current VRAM page.
    LDA CreditsPagesTextMasks, X
:
    ; Shift left as many times as the current VRAM line number,
    ; to get the bit that indicates whether this line has text.
    ASL
    DEY
    BPL :-
    BCC @IncVramLine

    ; The mask indicates that it should have text.
    ; But if line index >= $17, then it doesn't.
    LDY CreditsLineIndex
    CPY #$17
    BCS @IncVramLine

    ; In the first quest, don't consider line numbers >= $10.
    LDX CurSaveSlot
    LDA QuestNumbers, X
    BNE :+
    CPY #$10
    BCS @IncLine
:
    LDX CurSaveSlot
    LDA QuestNumbers, X

    ; In the second quest, skip lines $C to $F.
    BEQ :+
    CPY #$0C
    BCC :+
    CPY #$10
    BCC @IncLine
:
    LDA CreditsTextAddrsLo, Y
    STA $00
    LDA CreditsTextAddrsHi, Y
    STA $01

    ; First read the length of the string.
    LDY #$00
    LDA ($00), Y
    STA $02                     ; The length of the string

    ; Second, read the offset where the first character goes
    ; in the line.
    INY
    LDA ($00), Y
    TAX

    ; Loop over each character in the rest of the credits source record.
    INY
:
    LDA ($00), Y
    STA DynTileBuf+3, X
    INY                         ; Increment the source pointer.
    INX                         ; Increment the destination pointer.
    DEC $02                     ; Decrement the count remaining.
    BNE :-

    ; If line index <> $11, go increment.
    LDY CreditsLineIndex
    CPY #$0C
    BCC @IncLine
    CPY #$11
    BNE @IncLine

    ; Line index = $11, prepare to read the save slot's name.
    LDA CurSaveSlot
    ASL
    ASL
    ASL
    TAY

    ; Copy the save slot's name at offset 9 in string to transfer.
    LDX #$00
:
    LDA SaveSlotNames, Y
    STA DynTileBuf+12, X
    INY
    INX
    CPX #$08
    BCC :-

    ; Format the death count for the current save slot.
    LDY a:CurSaveSlot
    LDA DeathCounts, Y
    JSR FormatDecimalByte

    ; Copy the decimal death count at offset 19 in string to transfer.
    LDX #$02
:
    LDA $01, X
    STA DynTileBuf+22, X
    DEX
    BPL :-
    LDY CreditsLineIndex

@IncLine:
    INC CreditsLineIndex

@IncVramLine:
    INC CreditsVramLine

    ; In each VRAM page there are 8 lines, except in ones where
    ; (page MOD 4) = 3. This means $23xx and $2Bxx.
    ; These pages have 6 lines.
    LDA CreditsVramPage
    AND #$03
    CMP #$03
    LDA #$08
    BCC :+
    LDA #$06
:
    ; If the VRAM line number has not just been incremented to
    ; the reference line number count, then go write NT attributes.
    CMP CreditsVramLine
    BNE @WriteAttributes

    ; Else roll over the VRAM line to 0.
    LDA #$00
    STA CreditsVramLine

    ; Increment the VRAM page number.
    LDY CreditsVramPage
    INY

    ; If it has reached $C, then roll it over to 0.
    CPY #$0C
    BCC :+
    TAY
:
    STY CreditsVramPage

@WriteAttributes:
    ; Every 4 rows, we have to transfer NT attributes.
    ; So, when (counter MOD 4) <> 0, skip filling an attribute record.
    LDA CreditsRow
    LSR
    BCS @IncRow
    LSR
    BCS @IncRow

    ; Attributes for the left and right blocks are 0.
    LDX #$00
    STX DynTileBuf+38
    STX DynTileBuf+45

    ; The row was already divided by 4. So now it refers to an
    ; NT attribute block row. Look up the attribute byte to use
    ; for almost every block in this NT attribute block row.
    TAY
    LDA CreditsAttrs, Y

    ; Fill a record with this byte, so the whole NT attribute block
    ; row is changed.
    LDY #$05
:
    STA DynTileBuf+39, Y
    DEY
    BPL :-

    ; If tiles are being written to NT 1 (>= $2800), then
    ; write the high byte of NT 1's attribute space (>= $2BC0).
    ; Else write the high byte of NT 0's attribute space (>= $23C0).
    LDY #$23
    LDA DynTileBuf
    AND #$08
    BEQ :+
    LDY #$2B
:
    STY DynTileBuf+35

    ; Every attribute block row is at an offset (tile row * 2).
    LDA CreditsRow
    AND #$1F
    ASL

    ; Add the offset to the low byte of attribute space base
    ; ($23C0 or $2BC0); and write it to the record.
    ADC #$C0
    STA DynTileBuf+36
    LDA #$08                    ; 8 bytes in the attribute row record.
    STA DynTileBuf+37

@IncRow:
    ; Add 1 to the row.
    LDY CreditsRow
    INY

    ; Skip the last two rows in each cycle.
    ; When (row MOD $20) gets to $1E, add 2.
    ;
    ; Again, this is because there are
    ; 240 pixels vertically instead of 256.
    TYA
    AND #$1F
    CMP #$1E
    BCC :+
    INY
    INY
:
    STY CreditsRow
    RTS
