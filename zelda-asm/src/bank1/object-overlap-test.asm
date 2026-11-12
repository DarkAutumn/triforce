; ============================================================================
; FILE:     bank1/object-overlap-test.asm BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 6351-6493
; SUBJECTS: (none)
; TOPIC:    Weapon hotspot and object overlap tests
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0029 CheckMonsterSlenderWeaponCollision   code  Params:
;   L0047 CheckMonsterSlenderWeaponCollision2  code  Params:
;   L0100 DoObjectsCollide                     code  Params:
;   L0123 DoObjectsCollideWithThresholds       code  Params:
; ============================================================================

; Params:
; A: damage points
; X: monster object slot
; [00]: weapon object slot
; [02]: monster object mid X
; [03]: monster object mid Y
; [09]: damage type
; [0D]: horizontal threshold
;
; Returns:
; [06]: 1 if objects collide
; [07]: damage points
; [0E]: vertical threshold
;
CheckMonsterSlenderWeaponCollision:
    STA $07
    LDA $0D
    STA $0E

; Params:
; X: monster object slot
; [00]: weapon object slot
; [02]: monster object mid X
; [03]: monster object mid Y
; [07]: damage points
; [09]: damage type
; [0D]: horizontal threshold
; [0E]: vertical threshold
;
; Returns:
; [06]: 1 if objects collide
;
CheckMonsterSlenderWeaponCollision2:
    LDY $00

    ; If Link is facing vertically, then:
    ; [04] := (weapon X + 6)
    ; A    := (weapon Y + 8)
    ;
    ; Shouldn't it be based on the weapon's direction?
    LDA ObjDir
    AND #$0C
    BEQ @CheckHorizontal
    LDA a:ObjX, Y
    CLC
    ADC #$06
    STA $04
    LDA a:ObjY, Y
    CLC
    ADC #$08
    JMP :+

@CheckHorizontal:
    ; Else Link is facing horizontally.
    ; [04] := (weapon X + 8)
    ; A    := (weapon Y + 6)
    ;
    ; Shouldn't it be based on the weapon's direction?
    LDA a:ObjX, Y
    CLC
    ADC #$08
    STA $04
    LDA a:ObjY, Y
    CLC
    ADC #$06
:
    ; The A register will be copied to [05] and used here.
    JMP CheckMonsterWeaponCollision

; Params:
; A: collision distance threshold (objects collide if nearer in X and Y)
; [00]: weapon object slot (unused)
; [02]: object 1 mid X
; [03]: object 1 mid Y
; [04]: object 2 mid X
; [05]: object 2 mid Y
;
; Returns:
; A: 1 if objects collide
; Y: weapon object slot
; Z: 0 if objects collide
; [06]: 1 if objects collide
; [0A]: horizontal distance
; [0B]: vertical distance
;
DoObjectsCollide:
    ; Use the same threshold value horizontally and vertically.
    ; Copy it to [0D] for horizontal threshold, and [0E] for vertical one.
    STA $0D
    STA $0E

; Params:
; [00]: weapon or Link object slot (unused)
; [02]: object 1 mid X
; [03]: object 1 mid Y
; [04]: object 2 mid X
; [05]: object 2 mid Y
; [0D]: horizontal threshold
; [0E]: vertical threshold
;
; Returns:
; A: 1 if objects collide
; Y: weapon or Link object slot (copied from [00])
; Z: 0 if objects collide
; [06]: 1 if objects collide
; [0A]: horizontal distance
; [0B]: vertical distance
;
DoObjectsCollideWithThresholds:
    ; Reset [06] to indicate no collision by default.
    LDA #$00
    STA $06
    LDY $00

    ; Store in [0A] the horizontal distance between the two objects.
    LDA $02
    SEC
    SBC $04
    JSR Abs
    STA $0A

    ; If distance >= horizontal threshold in [0D], return 0.
    CMP $0D
    BCS @ReturnValue

    ; Store in [0B] the vertical distance between the two objects.
    LDA $03
    SEC
    SBC $05
    JSR Abs
    STA $0B

    ; If distance >= vertical threshold in [0E], return 0.
    CMP $0E
    BCS @ReturnValue

    ; If we get here, then the objects are close enough in both axes.
    ; Return 1.
    INC $06

@ReturnValue:
    LDA $06
    RTS
