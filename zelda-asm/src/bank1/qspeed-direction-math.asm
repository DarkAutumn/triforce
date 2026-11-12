; ============================================================================
; FILE:     bank1/qspeed-direction-math.asm BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 3511-3636
; SUBJECTS: (none)
; TOPIC:    Q-speed position fraction, opposite direction, Abs, Negate
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0024 AddQSpeedToPositionFraction    code  Params:
;   L0063 SubQSpeedFromPositionFraction  code  Params:
;   L0095 OppositeDirs                   data
;   L0110 GetOppositeDir                 code  Params:
;   L0128 Abs                            code  Params:
;   L0137 Negate                         code  Params:
; ============================================================================

; Params:
; [010E]: PositiveGridCellSize
; [010F]: NegativeGridCellSize
;
; Returns:
; C: 1 if fractional position reached a whole pixel
;
AddQSpeedToPositionFraction:
    LDA ObjPosFrac, X
    CLC
    ADC ObjQSpeedFrac, X
    STA ObjPosFrac, X
    PHP                         ; Save carry flag.

    ; Don't let positions go past the limits
    ; If we're at a limit, then clear carry.
    LDA ObjGridOffset, X
    CMP PositiveGridCellSize
    BEQ @ClearCarry
    CMP NegativeGridCellSize
    BNE :+

@ClearCarry:
    PLP                         ; Replace saved (pushed) carry flag with 0.
    CLC
    PHP
:
    PLP                         ; Restore carry flag to use in addition below.
    PHP                         ; But save it again in order to return it at the end.

    ; Add only carry to the grid offset.
    ; Carry represents whether the fractional position reached
    ; a whole pixel (C=1).
    LDA ObjGridOffset, X
    ADC #$00
    STA ObjGridOffset, X
    PLP                         ; Return carry flag from speed addition, or 0 if we reached a limit.
    RTS

; Params:
; [010E]: PositiveGridCellSize
; [010F]: NegativeGridCellSize
;
; Returns:
; C: 0 if fractional position reached a whole pixel
;
SubQSpeedFromPositionFraction:
    LDA ObjPosFrac, X
    SEC
    SBC ObjQSpeedFrac, X
    STA ObjPosFrac, X
    PHP                         ; Save carry flag.

    ; Don't let positions go past the limits
    ; If we're at a limit, then set carry.
    LDA ObjGridOffset, X
    CMP PositiveGridCellSize
    BEQ @SetCarry
    CMP NegativeGridCellSize
    BNE :+

@SetCarry:
    PLP                         ; Replace saved (pushed) carry flag with 1.
    SEC
    PHP
:
    PLP                         ; Restore carry flag to use in addition below.
    PHP                         ; But save it again in order to return it at the end.

    ; Subtract only carry from the grid offset.
    ; Carry represents whether the fractional position reached
    ; a whole pixel (C=0).
    LDA ObjGridOffset, X
    SBC #$00
    STA ObjGridOffset, X
    PLP                         ; Return carry flag from speed subtraction, or 1 if we reached a limit.
    RTS

OppositeDirs:
    .BYTE $04, $08, $01, $02

; Params:
; A: direction
;
; Returns:
; A: opposite direction
; Y: a reverse direction index
;
; The reverse direction index returned has this mapping:
; In Dir: 1 2 4 8
;         -------
; Index:  3 2 1 0
;
GetOppositeDir:
    LDY #$03

@Loop:
    LSR
    BCS :+
    DEY
    BPL @Loop
:
    LDA OppositeDirs, Y
    RTS

; Params:
; A: value
;
; Returns:
; A: absolute value
;
Abs:
    BPL :+

; Params:
; A: value
;
; Returns:
; A: result
;
Negate:
    EOR #$FF
    CLC
    ADC #$01
:
    RTS
