; ============================================================================
; FILE:     bank1/format-decimal-hearts.asm BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 3153-3304
; SUBJECTS: (none)
; TOPIC:    Turns a byte into three decimal digit tiles (leading blanks), and turns the heart bytes into
;           the two rows of eight heart tiles shown in the status bar and on the file-select screen.
; SEE ALSO: bank1/status-bar-rupees.asm (FormatStatusBarText: hearts, rupee, key and bomb counts);
;           bank1/cave-person-prices.asm (prices); bank2/mode1-slot-tiles.asm (save-slot hearts and deaths);
;           bank2/credits.asm (death count).
; KEY FACTS:
;   - Heart byte format: high nibble = heart containers - 1, low nibble = number of completely full
;     hearts; the next heart is drawn from the partial byte: 0 empty, $01-$7F half, $80-$FF full
;     (FormatHeartsInTextBuf).
;   - A heart byte of 0 draws 16 blank tiles, not one container (FormatHeartsInTextBuf).
;   - Hearts 1-8 are the bottom row, 9-16 the top row, both left to right (FormatHeartsInTextBuf).
;   - Tiles: $24 blank, $F2 full heart, $65 half heart, $66 empty heart (FormatHeartsInTextBuf).
;   - Drawing a half heart writes 0 to IsSwordShotForced (FormatHeartsInTextBuf).
;   - FormatDecimalByte gives hundreds, tens, ones in [01], [02], [03]; leading zeros become tile $24.
; CONTENTS: (generated; L = line in this file)
;   L0032 FormatDecimalByte      code  Converts a byte to three decimal digit tiles, with blank tiles ($24) in place of leading zeros.
;   L0058 DivideBy10             code  Divides A by 10 by repeated subtraction.
;   L0081 FormatHeartsInTextBuf  code  Writes the two rows of eight heart tiles for a heart byte and partial byte into DynTileBuf.
; ============================================================================

; ----------------------------------------------------------------------------
; Converts a byte to three decimal digit tiles, with blank tiles ($24) in place of leading zeros.
; IN:   A = value (0-255)
; OUT:  [01] = hundreds digit or $24; [02] = tens digit or $24; [03] = ones digit (always a digit);
;       Y = [01]; A = [02]
; NOTE: Digit tiles are the digit values themselves ($00-$09).
; ----------------------------------------------------------------------------
FormatDecimalByte:
    JSR DivideBy10
    STA $03                     ; Ones digit.
    TYA
    JSR DivideBy10              ; A = tens digit, Y = hundreds digit.
    CPY #$00
    BNE @WriteHighChars         ; Hundreds digit is not 0: keep both digits.
    LDY #$24                    ; Blank tile for the hundreds place.
    CMP #$00
    BNE @WriteHighChars         ; Tens digit is not 0: keep it.
    TYA                         ; Value < 10: blank tile for the tens place too.

@WriteHighChars:
    STA $02
    STY $01
:
    ; This RTS also ends DivideBy10 (its loop branches back here).
    RTS

; ----------------------------------------------------------------------------
; Divides A by 10 by repeated subtraction.
; IN:   A = value
; OUT:  Y = quotient; A = remainder
; NOTE: Returns through the RTS at the end of FormatDecimalByte. The BNE below is always taken (the
;       quotient is at most 25), so execution never runs on into FormatHeartsInTextBuf.
; ----------------------------------------------------------------------------
DivideBy10:
    LDY #$00

@Loop:
    CMP #$0A
    BCC :-                      ; Less than 10 left: return.
    SBC #$0A                    ; Carry is set here, so this subtracts exactly 10.
    INY
    BNE @Loop

; ----------------------------------------------------------------------------
; Writes the two rows of eight heart tiles for a heart byte and partial byte into DynTileBuf.
; IN:   [0E] = heart byte (high nibble = heart containers - 1, low nibble = full hearts);
;       [0F] = partial heart byte; Y = offset in DynTileBuf of the first (leftmost) tile of the top row
; OUT:  writes DynTileBuf+Y .. +Y+7 (top row, hearts 9-16) and DynTileBuf+Y+$0B .. +Y+$12 (bottom row,
;       hearts 1-8); uses [00], [01], [0B], [0C], [0D]; X = $10
; NOTE: The rows are in two transfer records: 8 top-row tiles, the 3-byte header of the next record,
;       then 8 bottom-row tiles, so the last bottom-row tile is 7 + 3 + 8 = $12 bytes after the first
;       top-row tile. The caller's buffer template must already hold both record headers.
; NOTE: With low nibble n, hearts 1..n are full, heart n+1 follows the partial byte (0 = empty,
;       $01-$7F = half, $80-$FF = full) and hearts above it are empty. A heart byte of 0 draws only blank
;       tiles. Drawing a half heart also writes 0 to IsSwordShotForced.
; ----------------------------------------------------------------------------
FormatHeartsInTextBuf:
    STY $0D                     ; [0D] = starting offset in DynTileBuf.
    LDA $0E
    PHA
    AND #$0F                    ; [00] = $F - low nibble (full hearts).
    STA $00
    LDA #$0F
    SEC
    SBC $00
    STA $00
    PLA
    LSR                         ; [01] = $F - high nibble (heart containers - 1).
    LSR
    LSR
    LSR
    STA $01
    LDA #$0F
    SEC
    SBC $01
    STA $01

    ; The loop below walks X = 0..$F over the 16 heart positions from heart 16 down to heart 1
    ; (heart number = $10 - X), writing tiles right to left: first the top row, then the bottom row.
    ;   X <  [01]: no container in this position (blank tile)
    ;   X <  [00]: container without health (empty heart)
    ;   X =  [00]: the heart described by the partial byte
    ;   X >  [00]: full heart
    LDX #$00
    TYA
    CLC
    ADC #$07
    STA $0B                     ; [0B] = offset of the last (rightmost) tile of the top row.

    ; Y counts the tiles left in the top row; it passes 0 to $FF after the eighth tile.
    LDY #$07

@LoopHeartSlot:
    CPY #$FF
    BNE @FinishedFirstRow       ; Top row not finished yet (or already in the bottom row).
    LDA $0D                     ; Top row finished: move to the bottom row.
    CLC
    ADC #$12
    STA $0B                     ; [0B] = offset of the last (rightmost) tile of the bottom row.
    LDY #$12                    ; Any value that will not reach $FF again in 8 more steps.

@FinishedFirstRow:
    ; Blank tile if the heart byte is 0 (empty save slot), or if this position is above the number
    ; of heart containers.
    LDA $0E
    BEQ @EmitSpace
    CPX $01
    BCS @CheckOccupiedSlot

@EmitSpace:
    LDA #$24                    ; Blank tile.
    BNE @EmitChar               ; Always taken.

@CheckOccupiedSlot:
    CPX $00
    BEQ @CheckPartial           ; This is the heart described by the partial byte.
    BCC @EmitEmptyHeart         ; Above it: empty heart. Below it: full heart.

@EmitFullHeart:
    LDA #$F2                    ; Full heart tile.
    BNE @EmitChar               ; Always taken.

@CheckPartial:
    LDA $0F
    BEQ @EmitEmptyHeart         ; Partial byte 0: empty heart.
    CMP #$80
    BCS @EmitFullHeart          ; Partial byte >= $80: full heart.
    LDA #$00                    ; Partial byte $01-$7F: half heart;
    STA IsSwordShotForced       ; and write 0 to IsSwordShotForced (no code writes a nonzero value).
    LDA #$65                    ; Half heart tile.
    BNE @EmitChar               ; Always taken.

@EmitEmptyHeart:
    LDA #$66                    ; Empty heart tile.

@EmitChar:
    STY $0C                     ; Save the row counter.
    LDY $0B
    STA DynTileBuf, Y           ; Write the tile, then step one tile left.
    DEC $0B
    LDY $0C
    DEY
    INX
    CPX #$10
    BNE @LoopHeartSlot
    RTS
