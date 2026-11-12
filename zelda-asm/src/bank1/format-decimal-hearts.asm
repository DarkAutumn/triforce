; ============================================================================
; FILE:     bank1/format-decimal-hearts.asm BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 3153-3304
; SUBJECTS: (none)
; TOPIC:    Decimal formatting, heart row tiles
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0026 FormatDecimalByte      code  Formats a decimal byte. Spaces are used for missing digits.
;   L0051 DivideBy10             code  Params:
;   L0061 FormatHeartsInTextBuf  code
; ============================================================================

; Formats a decimal byte. Spaces are used for missing digits.
;
; Params:
; A: value
;
; Returns:
; Y: First character
; A: Second character
; [$01]: First character
; [$02]: Second character
; [$03]: Third character
;
FormatDecimalByte:
    JSR DivideBy10
    STA $03
    TYA
    JSR DivideBy10
    CPY #$00
    BNE @WriteHighChars
    LDY #$24
    CMP #$00
    BNE @WriteHighChars
    TYA

@WriteHighChars:
    STA $02
    STY $01
:
    RTS

; Params:
; A: value
;
; Returns:
; Y: result
; A: remainder
;
DivideBy10:
    LDY #$00

@Loop:
    CMP #$0A
    BCC :-
    SBC #$0A
    INY
    BNE @Loop

FormatHeartsInTextBuf:
    ; [$0E]: hearts value
    ; [$0F]: heart partial
    ; Y: The offset of first character in buffer
    ;
    ; The end of the bottom row is $12 bytes after the
    ; start of the top row. We're writing to two transfer
    ; records. We write the top row in the first transfer
    ; record wherever the caller chooses. The second
    ; transfer record has to begin with the heart tiles.
    ; So, 7 (offset to end of first row) + 3 (size of transfer
    ; header) + 8 (length of bottom row) = 18 ($12).
    ;
    ; Store the starting offset in [$0D].
    STY $0D
    LDA $0E
    PHA
    AND #$0F                    ; Put ($F - hearts) in [$00].
    STA $00
    LDA #$0F
    SEC
    SBC $00
    STA $00
    PLA
    LSR                         ; Put ($F - heart containers) in [$01].
    LSR
    LSR
    LSR
    STA $01
    LDA #$0F
    SEC
    SBC $01
    STA $01

    ; At this point,
    ; [$00] = $F - hearts
    ; [$01] = $F - heart containers
    ; [$0D] = Y parameter (starting offset in tile buf)
    ;
    ; Also, Y is still set to the value passed in.
    LDX #$00
    TYA
    CLC
    ADC #$07
    STA $0B                     ; Set [$0B] to offset of last heart in top row.

    ; Y keeps track of index of heart in row.
    ; Start at the end of the first row.
    LDY #$07

@LoopHeartSlot:
    ; For X in 0..$10 (all hearts spots):
    CPY #$FF
    BNE @FinishedFirstRow       ; If finished first row,
    LDA $0D
    CLC
    ADC #$12
    STA $0B                     ; Set [$0B] to offset of last heart in bottom row.
    LDY #$12

@FinishedFirstRow:
    ; If there are no hearts,
    ; or still processing missing heart containers;
    LDA $0E
    BEQ @EmitSpace
    CPX $01
    BCS @CheckOccupiedSlot

@EmitSpace:
    LDA #$24                    ; Use a blank tile.
    BNE @EmitChar               ; Go emit it.

@CheckOccupiedSlot:
    CPX $00
    BEQ @CheckPartial           ; At the boundary heart, go check the partial heart.
    BCC @EmitEmptyHeart         ; If no longer processing missing hearts,

@EmitFullHeart:
    LDA #$F2                    ; Use a full heart tile.
    BNE @EmitChar               ; Go emit it.

@CheckPartial:
    LDA $0F
    BEQ @EmitEmptyHeart         ; If partial heart is 0, go emit an empty heart tile.
    CMP #$80
    BCS @EmitFullHeart          ; If partial heart at least half full, go emit a full heart tile.
    LDA #$00
    STA IsSwordShotForced          ; else, clear [$0529]
    LDA #$65                    ; and use a half heart tile.
    BNE @EmitChar

@EmitEmptyHeart:
    LDA #$66                    ; Use an empty heart tile.

@EmitChar:
    STY $0C
    LDY $0B                     ; Emit chosen tile to buf.
    STA DynTileBuf, Y
    DEC $0B
    LDY $0C
    DEY
    INX
    CPX #$10
    BNE @LoopHeartSlot
    RTS
