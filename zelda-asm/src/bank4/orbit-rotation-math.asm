; ============================================================================
; FILE:     bank4/orbit-rotation-math.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 11870-12071
; SUBJECTS: (none)
; TOPIC:    Orbit movement for Patra's children and the pond fairy's hearts: sine table, per-frame step
;           along a circle, shift multiply, angle decrease. Then bank 4's tail: reset stub and vectors.
; SEE ALSO: bank4/patra.asm (PatraChild_State1: angle step and bit counts per maneuver);
;           bank4/pond-fairy.asm (the orbiting hearts); bank7/reset-mmc1-vectors.asm (IsrReset, the
;           labelled original of the reset stub at the end of this file).
; KEY FACTS:
;   - An orbit angle has 32 steps a circle (ObjAngleWhole $00-$1F) plus a fraction byte; the callers
;     subtract $60 or $70 from the 16-bit angle each frame: 85.3 or 73.1 frames a circle.
;     DecreaseObjectAngle
;   - The object is not placed on a circle around a centre: each frame it moves by a velocity
;     (+sin, -cos of the angle) scaled by ObjQSpeedFrac, so the radius follows from speed and angle
;     rate. RotateObjectLocation
;   - The step per axis is q-speed * (sine >> (8 - bits)) / 256 pixels; the caller chooses the bit
;     count per axis (5 or 6), 6 giving twice the step of 5. RotateObjectLocation, ShiftMultiply
;   - The orbit keeps its X and Y position fractions in $0412,X and $041F,X, the bytes flyers use as
;     Flyer_ObjPosFrac and Flyer_ObjSpeed. RotateObjectLocation
;   - The new Y is returned in A and stored by the caller; the new X is stored here.
;     RotateObjectLocation
;   - The bytes after DecreaseObjectAngle are bank 4's copy of the reset handler ($BF50) and of the
;     interrupt vectors ($BFFA); they have no labels.
; CONTENTS: (generated; L = line in this file)
;   L0034 PatraSines            data  TABLE: half sine wave, amplitude $80; index = angle AND $0F (16 steps = 180 degrees);
;   L0051 RotateObjectLocation  code  Moves an orbiting object one frame along its circle, from its angle and q-speed.
;   L0151 ShiftMultiply         code  Multiplies A by the top Y bits of [00], giving a 16-bit product in [02] (low) and [03] (high).
;   L0181 DecreaseObjectAngle   code  Subtracts a 16-bit amount from an orbiting object's angle and wraps it to $00-$1F.
; ============================================================================

; TABLE: half sine wave, amplitude $80; index = angle AND $0F (16 steps = 180 degrees);
;        value = $80 * sin(index * 11.25 degrees). The sign comes from the angle's bit 4.
PatraSines:
    .BYTE $00, $18, $30, $47, $5A, $6A, $76, $7D
    .BYTE $80, $7D, $76, $6A, $5A, $47, $30, $18

; ----------------------------------------------------------------------------
; Moves an orbiting object one frame along its circle, from its angle and q-speed.
; IN:   X = object slot; ObjAngleWhole,X = angle $00-$1F; ObjQSpeedFrac,X = speed factor;
;       Y = number of high bits of the sine to use for the X step (callers pass 5 or 6);
;       A = number of high bits of the cosine to use for the Y step.
; OUT:  ObjX,X updated; A = new Y coordinate, which the caller must store in ObjY,X.
;       Uses [00]-[03], [05], [06]. The X and Y position fractions are kept in $0412,X and $041F,X.
; NOTE: X step = +sin(angle), Y step = -cos(angle), each = q-speed * (table value >> (8 - bits)) / 256
;       pixels. With q-speed $20 the largest step is 4 pixels with 6 bits and 2 pixels with 5 bits.
;       Angle $18 moves left, $00 up, $08 right, $10 down; the callers decrease the angle each
;       frame (DecreaseObjectAngle), which makes the path counter-clockwise on screen. Different bit
;       counts for X and Y give an ellipse. Used by Patra's children and the pond fairy's hearts.
; ----------------------------------------------------------------------------
RotateObjectLocation:
    STA $06                     ; [06] = bits of cosine to use (Y step)
    STY $05                     ; [05] = bits of sine to use (X step)

    ; X step first. [00] = |sin(angle)| from the table.
    LDA ObjAngleWhole, X
    AND #$0F
    TAY
    LDA PatraSines, Y
    STA $00

    ; [02:03] = q-speed * (high [05] bits of [00]): [02] fraction, [03] whole pixels.
    ; (This routine does not set the q-speed; it is whatever the object's slot holds.)
    LDA ObjQSpeedFrac, X
    LDY $05
    JSR ShiftMultiply

    ; Sign of the X step: angle $10-$1F (sine negative) moves left, $00-$0F moves right.
    ; AND #$18 keeps bits 3-4, enough for the compare with $10.
    LDA ObjAngleWhole, X
    AND #$18
    CMP #$10
    BCC @AddToX

    ; Moving left: subtract the fraction from the X fraction ($0412,X) ...
    LDA $0412, X
    SEC
    SBC $02
    STA $0412, X

    ; ... and the whole pixels, with borrow, from ObjX.
    LDA ObjX, X
    SBC $03
    JMP @SetXRotateY

@AddToX:
    ; Moving right: add the fraction to the X fraction ($0412,X) ...
    LDA $0412, X
    CLC
    ADC $02
    STA $0412, X

    ; ... and the whole pixels, with carry, to ObjX.
    LDA ObjX, X
    ADC $03

@SetXRotateY:
    STA ObjX, X

    ; Y step. [00] = |cos(angle)|: the sine table read a quarter circle (8 steps) ahead.
    LDA ObjAngleWhole, X
    CLC
    ADC #$08
    AND #$0F
    TAY
    LDA PatraSines, Y
    STA $00

    ; [02:03] = q-speed * (high [06] bits of [00]).
    LDA ObjQSpeedFrac, X
    LDY $06
    JSR ShiftMultiply

    ; Sign of the Y step: (angle - 8) AND $1F in $10-$1F, that is angle $18-$1F or $00-$07
    ; (cosine positive), moves up; otherwise down. The AND also discards the borrow of angles < 8.
    LDA ObjAngleWhole, X
    SEC
    SBC #$08
    AND #$18
    CMP #$10
    BCC @AddToY

    ; Moving up: subtract the fraction from the Y fraction ($041F,X) ...
    LDA $041F, X
    SEC
    SBC $02
    STA $041F, X

    ; ... and the whole pixels, with borrow, from ObjY. The result is returned in A, not stored.
    LDA ObjY, X
    SBC $03
    RTS

@AddToY:
    ; Moving down: add the fraction to the Y fraction ($041F,X) ...
    LDA $041F, X
    CLC
    ADC $02
    STA $041F, X

    ; ... and the whole pixels, with carry, to ObjY. The result is returned in A, not stored.
    LDA ObjY, X
    ADC $03
    RTS

; ----------------------------------------------------------------------------
; Multiplies A by the top Y bits of [00], giving a 16-bit product in [02] (low) and [03] (high).
; IN:   A = multiplicand; [00] = multiplier (destroyed); Y = how many of its high bits to use (>= 1).
; OUT:  [03:02] = A * ([00] >> (8 - Y)); [01] = multiplicand; Y = 0.
; ----------------------------------------------------------------------------
ShiftMultiply:
    STA $01
    LDA #$00
    STA $02
    STA $03

@Loop:
    ASL $02
    ROL $03
    ASL $00
    BCC @Next
    LDA $02
    CLC
    ADC $01
    STA $02
    BCC @Next
    INC $03                     ; Carry from the low byte into the high byte.

@Next:
    DEY
    BNE @Loop
    RTS

; ----------------------------------------------------------------------------
; Subtracts a 16-bit amount from an orbiting object's angle and wraps it to $00-$1F.
; IN:   X = object slot; A = low byte of the amount (1/256 of an angle step); [0B] = high byte.
; OUT:  ObjAngleFrac,X and ObjAngleWhole,X (masked with $1F: 32 steps a circle); uses [0A].
; NOTE: The callers pass [0B] = 0 and A = $70 (Patra child type $25) or $60 (type $26 and the pond
;       fairy's hearts): a full circle takes 32 * 256 / $70 = 73.1 or 32 * 256 / $60 = 85.3 frames.
; ----------------------------------------------------------------------------
DecreaseObjectAngle:
    STA $0A
    LDA ObjAngleFrac, X
    SEC
    SBC $0A
    STA ObjAngleFrac, X

    ; Subtract the high byte, with borrow, from the whole angle.
    LDA ObjAngleWhole, X
    SBC $0B

    ; Wrap to $00-$1F.
    AND #$1F
    STA ObjAngleWhole, X
    RTS

.SEGMENT "BANK_04_ISR"




; DATA: bank 4's copy of the reset handler, at $BF50 (no labels; stored as bytes). Same instructions
;       as the fixed bank's IsrReset (a copy for the case that this bank holds the vectors at reset):
;       SEI, CLD, PPU control = 0, stack = $01FF, wait for two vertical blanks ($2002 bit 7),
;       write $FF to $8000/$A000/$C000/$E000 (reset the MMC1 shift registers),
;       A = $0F, JSR $BF98 (MMC1 control), write 0 to $A000 five times (CHR bank 0),
;       A = 7, JSR $BFAC (switch to PRG bank 7), JMP $E440 (an address in the fixed bank; that it is
;       the game's entry routine is unverified here: no label, raw address).
;       $BF98 (row 10): five serial writes of A to $8000, RTS. $BFAC (row 12, 5th byte): five serial
;       writes of A to $E000, RTS.
    .BYTE $78, $D8, $A9, $00, $8D, $00, $20, $A2
    .BYTE $FF, $9A, $AD, $02, $20, $29, $80, $F0
    .BYTE $F9, $AD, $02, $20, $29, $80, $F0, $F9
    .BYTE $09, $FF, $8D, $00, $80, $8D, $00, $A0
    .BYTE $8D, $00, $C0, $8D, $00, $E0, $A9, $0F
    .BYTE $20, $98, $BF, $A9, $00, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $A9, $07, $20, $AC, $BF, $4C, $40, $E4
    .BYTE $8D, $00, $80, $4A, $8D, $00, $80, $4A
    .BYTE $8D, $00, $80, $4A, $8D, $00, $80, $4A
    .BYTE $8D, $00, $80, $60, $8D, $00, $E0, $4A
    .BYTE $8D, $00, $E0, $4A, $8D, $00, $E0, $4A
    .BYTE $8D, $00, $E0, $4A, $8D, $00, $E0, $60

.SEGMENT "BANK_04_VEC"




; DATA: bank 4's copy of the interrupt vectors at $BFFA: NMI $E484, reset $BF50 (the code above),
;       IRQ $BFF0.
    .BYTE $84, $E4, $50, $BF, $F0, $BF
