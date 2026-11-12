; ============================================================================
; FILE:     bank1/status-bar-rupees.asm  BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 2818-2976
; SUBJECTS: (none)
; TOPIC:    BeginUpdateMode, rupee count ticking, status bar text
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0023 BeginUpdateMode                  code  Returns:
;   L0028 L6506_Exit                       code
;   L0031 StatusBarTransferBufTemplate     data
;   L0039 World_ChangeRupees               code
;   L0081 FormatStatusBarText              code
;   L0124 FormatDecimalCountByteInTextBuf  code  Params:
;   L0131 CopyTripletToTextBuf             code  Params:
;   L0155 FormatDecimalCountByte           code  Formats a byte:
;   L0173 FormatCharDoublet                code  Params:
; ============================================================================

; Returns:
; A: 0
;
BeginUpdateMode:
    LDA #$00
    STA GameSubmode
    INC IsUpdatingMode

L6506_Exit:
    RTS

StatusBarTransferBufTemplate:
    .BYTE $20, $B6, $08, $24, $24, $24, $24, $24
    .BYTE $24, $24, $24, $20, $D6, $08, $24, $24
    .BYTE $24, $24, $24, $24, $24, $24, $20, $6C
    .BYTE $03, $21, $00, $24, $20, $AC, $03, $21
    .BYTE $00, $24, $20, $CC, $03, $21, $00, $24
    .BYTE $FF

World_ChangeRupees:
    LDA TileBufSelector
    BNE L6506_Exit              ; If a static transfer buf is already chosen, then return.
    LDA DynTileBuf
    BPL L6506_Exit              ; If the dynamic transfer buf is already used, then return.

    ; Reset RupeesToAdd or RupeesToSubtract as appropriate,
    ; if reached 0 or max.
    ;
    ; Index of RupeesToSubtract item slot.
    LDY #$27
    LDA InvRupees
    BEQ @ResetSlot
    LDY #$26                    ; Index of RupeesToAdd item slot.
    CMP #$FF
    BNE :+                      ; If 0 < rupees < $FF, don't reset anything.

@ResetSlot:
    LDA #$00                    ; Reset the slot we chose.
    STA Items, Y
:
    LDA FrameCounter
    LSR
    BCS L6506_Exit              ; Every two frames, return.
    LDA RupeesToAdd
    BEQ @CheckSubtract          ; If RupeesToAdd <> 0,
    DEC RupeesToAdd             ; then add one rupee.
    INC InvRupees

    ; Play "heart taken" tune, probably because it's more
    ; pleasing for a continuous process than "rupee taken".
    LDA #$10
    STA Tune0Request

@CheckSubtract:
    LDA RupeesToSubtract
    BEQ FormatStatusBarText     ; If RupeesToSubtract = 0, then skip it, and go format header text.
    DEC RupeesToSubtract
    DEC InvRupees               ; Subtract one rupee.
    LDA #$10                    ; Play the tune for this.
    STA Tune0Request

FormatStatusBarText:
    LDY #$28                    ; Copy transfer buf template for status bar text to dynamic buf.
:
    LDA StatusBarTransferBufTemplate, Y
    STA DynTileBuf, Y
    DEY
    BPL :-
    LDY #$03
    LDA HeartValues
    STA $0E
    LDA HeartPartial
    STA $0F
    JSR FormatHeartsInTextBuf
    LDX #$02
    LDA InvRupees
    LDY #$1B
    JSR FormatDecimalCountByteInTextBuf
    LDY #$21                    ; Key count goes at offset $21.
    LDA InvMagicKey
    BEQ @FormatKeyCount         ; If don't have the magic key, then go format key count.
    STY $00
    LDA #$21                    ; Put an "X" in [$01].
    STA $01
    LDA #$0A                    ; Put an "A" in formatting buffer.
    JSR FormatCharDoublet
    LDX #$08                    ; UNKNOWN: Why? X is overwritten in the procedure.
    JSR CopyTripletToTextBuf
    JMP @FormatBombCount        ; Have the magic key. Skip formatting key count.

@FormatKeyCount:
    LDX #$08
    LDA InvKeys
    JSR FormatDecimalCountByteInTextBuf

@FormatBombCount:
    LDX #$0E
    LDA InvBombs
    LDY #$27
    JMP FormatDecimalCountByteInTextBuf

; Params:
; A: value
; Y: offset of first character in buffer.
FormatDecimalCountByteInTextBuf:
    STY $00
    JSR FormatDecimalCountByte

; Params:
; [$00]: offset to first character in buffer
;
CopyTripletToTextBuf:
    LDY #$02
    LDX $00
:
    LDA $0001, Y
    STA DynTileBuf, X
    DEX
    DEY
    BPL :-
    RTS

; Formats a byte:
; 123 -> "123"
;  23 -> "X23"
;   3 -> "X3 "
;
; Params:
; A: value
;
; Returns:
; [$01]: First character
; [$02]: Second character
; [$03]: Third character
;
FormatDecimalCountByte:
    JSR FormatDecimalByte
    CPY #$24
    BNE :+
    LDY #$21
:
    STY $01
    CMP #$24
    BNE :+
    LDA $03

; Params:
; A: character
;
; Returns:
; [$02]: character
; [$03]: space
;
FormatCharDoublet:
    STA $02
    LDA #$24
    STA $03
:
    RTS
