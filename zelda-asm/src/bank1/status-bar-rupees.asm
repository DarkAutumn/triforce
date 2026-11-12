; ============================================================================
; FILE:     bank1/status-bar-rupees.asm  BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 2818-2976
; SUBJECTS: Inv, Mode, StatusBar, Text
; TOPIC:    First code of the common code block: the switch from a mode's init to its update
;           (Mode_BeginUpdate), the rupee counter that moves InvRupees 1 at a time, and the status
;           bar text (hearts, rupee, key and bomb counts) written to the dynamic transfer buffer.
; SEE ALSO: bank7/mode5-play-update.asm (World_UpdateHeartsAndRupees calls Inv_ChangeRupees every frame
;           of the play modes); bank5/cave-cellar-enter.asm (World_FillHearts, called just before);
;           bank1/format-decimal-hearts.asm (Text_FormatDecimalByte, Text_FormatHeartRows);
;           bank1/take-item.asm (Item_TakeOneRupee) and bank1/cave-hint-money-game.asm (Inv_PostRupeeCredit,
;           Inv_PostRupeeDebit): where rupees to add or subtract are posted; bank5/mode11-death-init.asm
;           (calls StatusBar_FormatText directly); bank1/common-code-header.asm (why this code runs
;           from RAM).
; KEY FACTS:
;   - InvRupees never jumps: on every even frame (FrameCounter bit 0 clear) it gains 1 while
;     InvRupeesToAdd is nonzero and loses 1 while InvRupeesToSubtract is nonzero. That is 30 rupees
;     a second; a price of 100 takes 200 frames to count off. Inv_ChangeRupees.
;   - The count runs only where World_UpdateHeartsAndRupees is called: the play update of modes 5, 9, $A,
;     $B and $C (also while paused or in the submenu) and mode $12 submode 2. It does not run while
;     the flute tune plays (FluteTimer nonzero) or a dark room is brightening. Inv_ChangeRupees.
;   - A frame is skipped when a PPU transfer is already queued (TileBufSelector nonzero, or
;     DynTileBuf holding a record). Inv_ChangeRupees.
;   - With InvRupees = 0 a pending subtraction is thrown away; with InvRupees = $FF a pending
;     addition is thrown away. So rupees stop at 255 and a debt is never carried.
;     Inv_ChangeRupees.
;   - Each counted rupee requests tune 0 bit $10, the blip also used by the heart refill and by
;     typed text. Inv_ChangeRupees.
;   - After each count step the whole status bar text is rewritten from HeartValues, HeartPartial,
;     InvRupees, InvKeys (or InvMagicKey) and InvBombs. StatusBar_FormatText.
;   - Counts are an "X" (tile $21) and the digits, left-aligned: 123, X23, X3 and a blank. With the
;     magic key the key count reads "XA". Text_FormatDecimalCountByte, StatusBar_FormatText.
;   - Mode_BeginUpdate sets GameSubmode = 0 and makes IsUpdatingMode nonzero: from the next frame
;     the mode's update routine runs instead of its init routine.
; CONTENTS: (generated; L = line in this file)
;   L0053 Mode_BeginUpdate                      code  Ends a mode's init phase: resets the submode and makes the mode's update routine run from now on.
;   L0059 Mode_BeginUpdate_Return               code  Shared RTS: end of Mode_BeginUpdate and the early exits of Inv_ChangeRupees.
;   L0071 StatusBar_TextTransferBufTemplate     data  TABLE: transfer buffer template for the status bar text, $29 bytes; index = byte offset, as in
;   L0095 Inv_ChangeRupees                      code  Counts pending rupees into or out of InvRupees, 1 per even frame, then rewrites the status bar.
;   L0148 StatusBar_FormatText                  code  Writes the status bar text (hearts, rupee, key and bomb counts) into the dynamic transfer buffer.
;   L0203 Text_FormatDecimalCountByteInTextBuf  code  Formats a count as three tiles ("X" and digits) and writes them into the dynamic transfer buffer.
;   L0212 Text_CopyTripletToTextBuf             code  Copies the three tiles in [01], [02], [03] into the dynamic transfer buffer.
;   L0230 Text_FormatDecimalCountByte           code  Formats a count as three tiles: the number itself if it has three digits, else "X" and digits.
;   L0247 Text_FormatCharDoublet                code  Puts one tile in the middle position and a blank in the last position of the tile triplet.
; ============================================================================

; ----------------------------------------------------------------------------
; Ends a mode's init phase: resets the submode and makes the mode's update routine run from now on.
; OUT:  GameSubmode = 0; IsUpdatingMode incremented (0 -> 1); A = 0 (callers store it)
; NOTE: Nmi_Isr runs the mode's init routine while IsUpdatingMode = 0 and its update routine
;       otherwise, one of the two per frame; so the first update happens on the next frame.
; ----------------------------------------------------------------------------
Mode_BeginUpdate:
    LDA #$00
    STA GameSubmode
    INC IsUpdatingMode

; Shared RTS: end of Mode_BeginUpdate and the early exits of Inv_ChangeRupees.
Mode_BeginUpdate_Return:
    RTS

; TABLE: transfer buffer template for the status bar text, $29 bytes; index = byte offset, as in
; DynTileBuf; value = PPU address high, low, tile count, then that many tiles; $FF ends it.
;   +$00 $20B6 (row 5, column 22), 8 tiles at +$03..+$0A: top row of hearts (hearts 9-16)
;   +$0B $20D6 (row 6, column 22), 8 tiles at +$0E..+$15: bottom row of hearts (hearts 1-8)
;   +$16 $206C (row 3, column 12), 3 tiles at +$19..+$1B: rupee count
;   +$1C $20AC (row 5, column 12), 3 tiles at +$1F..+$21: key count
;   +$22 $20CC (row 6, column 12), 3 tiles at +$25..+$27: bomb count
;   +$28 $FF end marker
; Tiles: $24 blank, $21 "X", $00-$09 the digits. The template itself reads "X0" for each count.
StatusBar_TextTransferBufTemplate:
    .BYTE $20, $B6, $08, $24, $24, $24, $24, $24    ; $20B6, 8 tiles: top heart row ...
    .BYTE $24, $24, $24, $20, $D6, $08, $24, $24    ; ... $20D6, 8 tiles: bottom heart row ...
    .BYTE $24, $24, $24, $24, $24, $24, $20, $6C    ; ... $206C,
    .BYTE $03, $21, $00, $24, $20, $AC, $03, $21    ; 3 tiles: rupees; $20AC, 3 tiles: keys
    .BYTE $00, $24, $20, $CC, $03, $21, $00, $24    ; $20CC, 3 tiles: bombs
    .BYTE $FF                                       ; End marker.

; ----------------------------------------------------------------------------
; Counts pending rupees into or out of InvRupees, 1 per even frame, then rewrites the status bar.
; OUT:  InvRupees, InvRupeesToAdd, InvRupeesToSubtract stepped; tune 0 request $10 per counted
;       rupee; DynTileBuf = the status bar text (by falling into StatusBar_FormatText)
; NOTE: Returns without doing anything when a PPU transfer is already queued for this frame:
;       TileBufSelector nonzero, or a record in DynTileBuf (first byte with bit 7 clear).
; NOTE: Before counting, every frame: InvRupees = 0 zeroes InvRupeesToSubtract (a debt is dropped);
;       InvRupees = $FF zeroes InvRupeesToAdd (the rest of a gain is dropped). This is the only cap:
;       rupees stop at 255.
; NOTE: Counting happens only when FrameCounter is even: 1 rupee per 2 frames (30 a second) in
;       each direction. An addition and a subtraction pending together both step in the same frame.
; NOTE: Called by World_UpdateHeartsAndRupees: every frame of the play update (modes 5, 9, $A, $B, $C),
;       also while paused or in the submenu, and in mode $12 submode 2. Mode5Play_Update returns
;       before it while FluteTimer is nonzero and while IsRoomBrightening is set. In other modes
;       pending rupees wait.
; ----------------------------------------------------------------------------
Inv_ChangeRupees:
    LDA TileBufSelector
    BNE Mode_BeginUpdate_Return              ; A static transfer buffer is queued for this frame: return.
    LDA DynTileBuf
    BPL Mode_BeginUpdate_Return              ; The dynamic buffer holds a record (empty = $FF): return.

    ; If InvRupees = 0, zero InvRupeesToSubtract (item slot $27).
    ; If InvRupees = $FF, zero InvRupeesToAdd (item slot $26).
    LDY #$27
    LDA InvRupees
    BEQ @ResetSlot
    LDY #$26
    CMP #$FF
    BNE :+                      ; 0 < InvRupees < $FF: zero neither.

@ResetSlot:
    LDA #$00
    STA InvItems, Y
:
    ; Count only on even frames.
    LDA FrameCounter
    LSR
    BCS Mode_BeginUpdate_Return              ; Odd frame: return.

    ; One rupee to add, if any is pending.
    LDA InvRupeesToAdd
    BEQ @CheckSubtract
    DEC InvRupeesToAdd
    INC InvRupees

    ; Request tune 0 bit $10 (the blip also used by the heart refill and by typed text), not the
    ; "rupee taken" tune of tune channel 1.
    LDA #$10
    STA Tune0Request

@CheckSubtract:
    ; One rupee to subtract, if any is pending. Either way the status bar text is rewritten.
    LDA InvRupeesToSubtract
    BEQ StatusBar_FormatText
    DEC InvRupeesToSubtract
    DEC InvRupees
    LDA #$10                    ; The same blip as for an added rupee.
    STA Tune0Request

; ----------------------------------------------------------------------------
; Writes the status bar text (hearts, rupee, key and bomb counts) into the dynamic transfer buffer.
; OUT:  DynTileBuf+$00..+$28 = StatusBar_TextTransferBufTemplate with the current values filled in;
;       uses [00]-[03] and [0B]-[0F]; DynTileBufLen is not changed
; NOTE: Inv_ChangeRupees falls into this on every even frame in which it runs; the death mode
;       init calls it directly. It writes from offset 0 without testing whether the buffer is
;       free (Inv_ChangeRupees tests that first).
; NOTE: With InvMagicKey nonzero the key count is replaced by "XA" and InvKeys is not shown.
; ----------------------------------------------------------------------------
StatusBar_FormatText:
    LDY #$28                    ; Copy the template, $29 bytes, from its end marker down.
:
    LDA StatusBar_TextTransferBufTemplate, Y
    STA DynTileBuf, Y
    DEY
    BPL :-

    ; Hearts: the top row starts at offset 3 (the bottom row follows at offset $0E).
    LDY #$03
    LDA HeartValues
    STA $0E
    LDA HeartPartial
    STA $0F
    JSR Text_FormatHeartRows

    ; Rupee count: its third tile is at offset $1B.
    LDX #$02                    ; Not used: the callee does not read X and overwrites it.
    LDA InvRupees
    LDY #$1B
    JSR Text_FormatDecimalCountByteInTextBuf

    ; Key count: its third tile is at offset $21.
    LDY #$21
    LDA InvMagicKey
    BEQ @FormatKeyCount         ; No magic key: show InvKeys.

    ; Magic key: write "XA" and a blank instead of a number.
    STY $00
    LDA #$21                    ; [01] = tile $21, the "X".
    STA $01
    LDA #$0A                    ; [02] = tile $0A, the letter A; [03] = blank.
    JSR Text_FormatCharDoublet
    LDX #$08                    ; Not used: Text_CopyTripletToTextBuf loads X itself.
    JSR Text_CopyTripletToTextBuf
    JMP @FormatBombCount

@FormatKeyCount:
    LDX #$08                    ; Not used (see above).
    LDA InvKeys
    JSR Text_FormatDecimalCountByteInTextBuf

@FormatBombCount:
    ; Bomb count: its third tile is at offset $27.
    LDX #$0E                    ; Not used (see above).
    LDA InvBombs
    LDY #$27
    JMP Text_FormatDecimalCountByteInTextBuf

; ----------------------------------------------------------------------------
; Formats a count as three tiles ("X" and digits) and writes them into the dynamic transfer buffer.
; IN:   A = value (0-255); Y = offset in DynTileBuf of the third (last) of the three tiles
; OUT:  DynTileBuf+Y-2..+Y = the three tiles; [00] = Y; [01]-[03] = the tiles; X = Y - 3; Y = $FF
; NOTE: Falls into Text_CopyTripletToTextBuf.
; ----------------------------------------------------------------------------
Text_FormatDecimalCountByteInTextBuf:
    STY $00
    JSR Text_FormatDecimalCountByte

; ----------------------------------------------------------------------------
; Copies the three tiles in [01], [02], [03] into the dynamic transfer buffer.
; IN:   [00] = offset in DynTileBuf of the third (last) tile; [01]-[03] = tiles, left to right
; OUT:  DynTileBuf+[00]-2..+[00] = [01], [02], [03]; X = [00] - 3; Y = $FF
; ----------------------------------------------------------------------------
Text_CopyTripletToTextBuf:
    LDY #$02
    LDX $00
:
    LDA $0001, Y                ; [01+Y], copied from the last tile back to the first.
    STA DynTileBuf, X
    DEX
    DEY
    BPL :-
    RTS

; ----------------------------------------------------------------------------
; Formats a count as three tiles: the number itself if it has three digits, else "X" and digits.
; IN:   A = value (0-255)
; OUT:  [01], [02], [03] = the tiles, left to right: 123 -> "123"; 23 -> "X23"; 3 -> "X3" and a
;       blank ($24). "X" is tile $21; a digit's tile is its value.
; NOTE: For values below 10 it falls into Text_FormatCharDoublet to move the digit left.
; ----------------------------------------------------------------------------
Text_FormatDecimalCountByte:
    JSR Text_FormatDecimalByte       ; [01]-[03] = digits with leading blanks; Y = [01]; A = [02].
    CPY #$24
    BNE :+                      ; Three digits: keep the hundreds digit.
    LDY #$21                    ; No hundreds digit: put the "X" there.
:
    STY $01
    CMP #$24
    BNE :+                      ; There is a tens digit: done (the RTS below).
    LDA $03                     ; One digit only: A = that digit, to go in the middle.

; ----------------------------------------------------------------------------
; Puts one tile in the middle position and a blank in the last position of the tile triplet.
; IN:   A = tile
; OUT:  [02] = A; [03] = $24 (blank); A = $24
; NOTE: The RTS is also the exit of Text_FormatDecimalCountByte for values of 10 and more.
; ----------------------------------------------------------------------------
Text_FormatCharDoublet:
    STA $02
    LDA #$24
    STA $03
:
    RTS
