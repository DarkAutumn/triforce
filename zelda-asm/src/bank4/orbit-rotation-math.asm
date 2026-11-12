; ============================================================================
; FILE:     bank4/orbit-rotation-math.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 11870-12071
; SUBJECTS: (none)
; TOPIC:    Rotation math; bank tail
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0015 PatraSines            data
;   L0028 RotateObjectLocation  code  Params:
;   L0141 ShiftMultiply         code  Params:
;   L0173 DecreaseObjectAngle   code  Params:
; ============================================================================

PatraSines:
    .BYTE $00, $18, $30, $47, $5A, $6A, $76, $7D
    .BYTE $80, $7D, $76, $6A, $5A, $47, $30, $18

; Params:
; A: high bits of cosine to use in calculating Y increment
; Y: high bits of sine to use in calculating X increment
; X: object index
;
; Returns:
; A: new Y coordinate of object
;
; [06] holds the number of high bits of cosine to use
RotateObjectLocation:
    STA $06
    STY $05                     ; [05] holds the number of high bits of sine to use

    ; First, rotate the X coordinate.
    ;
    ; Look up the sine for the current angle, and store it in [00].
    LDA ObjAngleWhole, X
    AND #$0F
    TAY
    LDA PatraSines, Y
    STA $00

    ; Mutliply q-speed by [00].
    ;
    ; Q-speed has the default value $20 set when entering the room.
    LDA ObjQSpeedFrac, X
    LDY $05
    JSR ShiftMultiply

    ; The absolute X increment for the current angle is now in [02:03].
    ;
    ; If angle >= $10, then the monster is in the top half of circle going left.
    LDA ObjAngleWhole, X
    AND #$18                    ; UNKNOWN: Why AND with $18?
    CMP #$10
    BCC @AddToX

    ; Subtract the product's fraction from the X coordinate fraction.
    LDA $0412, X
    SEC
    SBC $02
    STA $0412, X

    ; And subtract the product's whole part from the X coordinate with carry.
    LDA ObjX, X
    SBC $03
    JMP @SetXRotateY

@AddToX:
    ; Else angle < $10. So, the monster is in the bottom half of circle going right.
    ; Add the product's fraction to the X coordinate fraction.
    LDA $0412, X
    CLC
    ADC $02
    STA $0412, X

    ; And add the product's whole part to the X coordinate with carry.
    LDA ObjX, X
    ADC $03

@SetXRotateY:
    STA ObjX, X

    ; Now rotate the Y coordinate.
    ;
    ; Look up the cosine for the current angle, and store it in [00].
    ; The cosine corresponds to the sine offset by a quarter circle (8).
    LDA ObjAngleWhole, X
    CLC
    ADC #$08
    AND #$0F
    TAY
    LDA PatraSines, Y
    STA $00

    ; Multiply q-speed by [00].
    LDA ObjQSpeedFrac, X
    LDY $06
    JSR ShiftMultiply

    ; The absolute Y increment for the current angle is now in [02:03].
    ;
    ; If (angle - 8) >= $10, then the monster is in the right half of circle going up.
    LDA ObjAngleWhole, X
    SEC
    SBC #$08
    AND #$18
    CMP #$10
    BCC @AddToY

    ; Subtract the product's fraction from the Y coordinate fraction.
    LDA $041F, X
    SEC
    SBC $02
    STA $041F, X

    ; And subtract the product's whole part from the Y coordinate.
    LDA ObjY, X
    SBC $03
    RTS

@AddToY:
    ; Else (angle - 8) < $10. So, the monster is in the left half of circle going down.
    ; Add the product's fraction to the Y coordinate fraction.
    LDA $041F, X
    CLC
    ADC $02
    STA $041F, X

    ; And add the product's whole part to the Y coordinate with carry.
    LDA ObjY, X
    ADC $03
    RTS

; Params:
; A: multiplicand
; Y: number of high bits of [00] to use
; [00]: multiplier (high Y bits are used)
;
; Returns:
; [02:03]: product
;
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
    ; Begin unverified code 13453
    INC $03
    ; End unverified code

@Next:
    DEY
    BNE @Loop
    RTS

; Params:
; A: low amount byte
; X: object index
; [0B]: high amount byte
;
; Subtract the low amount byte from low angle byte.
;
DecreaseObjectAngle:
    STA $0A
    LDA ObjAngleFrac, X
    SEC
    SBC $0A
    STA ObjAngleFrac, X

    ; Subtract the high amount byte from high angle byte with borrow.
    LDA ObjAngleWhole, X
    SBC $0B

    ; Cap the high byte at $1F.
    AND #$1F
    STA ObjAngleWhole, X
    RTS

.SEGMENT "BANK_04_ISR"




; Unknown block
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




; Unknown block
    .BYTE $84, $E4, $50, $BF, $F0, $BF
