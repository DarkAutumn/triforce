; ============================================================================
; FILE:     bank4/digdogger.asm          BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 5155-5594
; SUBJECTS: (none)
; TOPIC:    Digdogger: flute split, update, move, collisions, draw
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0032 DigdoggerCornerOffsetsX             data
;   L0035 DigdoggerCornerOffsetsY             data
;   L0043 L_Digdogger_AfterFlute              code  Params:
;   L0140 L_Digdogger_TurnTowardLink          code
;   L0144 UpdateDigdogger                     code
;   L0156 L_Digdogger_Turn                    code
;   L0170 L_Digdogger_Move                    code
;   L0173 L_Digdogger_DrawAndCheckCollisions  code
;   L0181 CheckBigDigdoggerCollisions         code
;   L0228 L_Digdogger_DrawAsLittle            code
;   L0271 Digdogger_ChangeSpeed               code
;   L0275 Digdogger_ChangeSpeed_JumpTable     data
;   L0279 Digdogger_SpeedUp                   code
;   L0300 Digdogger_SlowDown                  code
;   L0324 SetTargetSpeed                      code
;   L0334 L116EA_Exit                         code
;   L0337 Digdogger_Move                      code
;   L0411 DigdoggerSpriteOffsetsX             data
;   L0414 DigdoggerSpriteOffsetsY             data
;   L0417 DigdoggerSpriteAttrs                data
;   L0420 Digdogger_Draw                      code
; ============================================================================

DigdoggerCornerOffsetsX:
    .BYTE $00, $10, $00, $F0

DigdoggerCornerOffsetsY:
    .BYTE $00, $10, $F0, $10

; Params:
; Y: flute state
;
; If flute state = 2, go switch between big and little appearance.
;
L_Digdogger_AfterFlute:
    DEY
    BNE @SplitUp

    ; Flute state = 1.
    ;
    ; If this is a child digdogger, then go turn and move as usual.
    LDA Digdogger_ObjIsChild, X
    BNE L_Digdogger_Turn

    ; Set object timer to $40, and increase flute state to 2.
    ; Then go check collisions as a big digdogger.
    LDA #$40
    STA ObjTimer, X
    INC UsedFlute
    JMP CheckBigDigdoggerCollisions

@SplitUp:
    ; If the timer = 0, then skip the block below, and go split up.
    LDA ObjTimer, X
    BEQ @MakeChildren

    ; Every 8 frames, switch between behaving as a big and little digdogger.
    ;
    ; UNKNOWN:
    ; But is this possible, given the following?
    ; a. all object timers run concurrently
    ; b. all object updates are paused until the flute timer expires
    ; c. the flute timer lasts longer
    ; Begin unverified code 1159D
    AND #$07
    BNE :+
    LDA Digdogger_ObjIsChild, X
    EOR #$01
    STA Digdogger_ObjIsChild, X
    BEQ CheckBigDigdoggerCollisions
:
    JMP L_Digdogger_DrawAsLittle
    ; End unverified code

@MakeChildren:
    ; Decrement flute state to 1.
    DEC UsedFlute

    ; Loop to make a number of child digdoggers.
    LDA ChildDigdoggerCount
    STA $00                     ; [00] is the loop counter
    STA RoomObjCount            ; Replace the object count with the number of children.
    TXA                         ; Save the current object index in X register.
    PHA

@LoopMakeChild:
    ; Initialize a child digdogger (type $18) in the next object slot.
    INX
    JSR InitDigdogger1
    LDA #$18
    STA ObjType, X

    ; Start with a target speed of $0100.
    ; The low byte is still 0 from when room data was reset.
    INC Digdogger_ObjTargetSpeedWhole, X

    ; Flag this a child digdogger.
    LDA #$01
    STA Digdogger_ObjIsChild, X

    ; The speed will increase.
    LDA #$00
    STA Digdogger_ObjSpeedFlag, X

    ; Copy the parent's coordinates to the child.
    LDA ObjX+1
    STA ObjX, X
    LDA ObjY+1
    STA ObjY, X

    ; Bottom of the loop.
    ; Decrement [00] until 0.
    DEC $00
    BNE @LoopMakeChild
    PLA                         ; Restore the current object index in X register.
    TAX
    JSR PlayBossDeathCry

    ; Destroy the big digdogger by resetting the object type.
    LDA #$00
    STA ObjType, X

    ; Don't let the digdogger disappear until the next frame when
    ; the children can draw themselves. Go draw the big digdogger
    ; as a little one.
    ;
    ; This will work, even though the object type was reset. No
    ; other object data has not been touched. Also, the code block
    ; we'll jump to temporarily sets the object type.
    JMP L_Digdogger_DrawAsLittle

L_Digdogger_TurnTowardLink:
    JSR TurnTowardsPlayer8
    JMP L_Digdogger_Move

UpdateDigdogger:
    ; If we have the magic clock, or the monster is stunned,
    ; then go check room boundaries and object collisions, and draw.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE L_Digdogger_DrawAndCheckCollisions

    ; If the flute was used, then go start splitting up,
    ; or handling a child digdogger.
    LDY UsedFlute
    BNE L_Digdogger_AfterFlute

L_Digdogger_Turn:
    JSR Digdogger_ChangeSpeed

    ; If the object timer has expired, then set it to $10,
    ; and randomly choose to turn toward Link or randomly.
    LDA ObjTimer, X
    BNE L_Digdogger_Move
    LDA #$10
    STA ObjTimer, X
    LDA Random, X
    CMP #$80
    BCS L_Digdogger_TurnTowardLink
    JSR TurnRandomlyDir8

L_Digdogger_Move:
    JSR Digdogger_Move

L_Digdogger_DrawAndCheckCollisions:
    ; If this is a child, then finish updating this little digdogger.
    LDA Digdogger_ObjIsChild, X
    BEQ CheckBigDigdoggerCollisions
    JSR BoundFlyer
    JSR CheckMonsterCollisions
    JMP Digdogger_Draw

CheckBigDigdoggerCollisions:
    ; Save the original coordinates.
    LDA ObjX, X
    PHA
    LDA ObjY, X
    PHA

    ; Reset the index of the current part.
    LDA #$00
    STA Digdogger_ObjCurPart, X

@LoopCornerCollision:
    ; Digdogger is big, but room boundary checks and
    ; object collision checks assume a 16x16 monster or smaller.
    ; Therefore, we have to check these things four times,
    ; in different corners of the boss.
    ;
    ; Loop 4 times, from 0 to 3.
    LDY Digdogger_ObjCurPart, X

    ; Change the coordinates to one corner.
    LDA ObjX, X
    CLC
    ADC DigdoggerCornerOffsetsX, Y
    STA ObjX, X
    LDA ObjY, X
    CLC
    ADC DigdoggerCornerOffsetsY, Y
    STA ObjY, X

    ; Check the room boundary and object collisions in this temporary location.
    JSR BoundFlyer
    JSR CheckMonsterCollisions

    ; Increase the loop index, and loop again until it = 4.
    INC Digdogger_ObjCurPart, X
    LDA Digdogger_ObjCurPart, X
    CMP #$04
    BCC @LoopCornerCollision

    ; Restore the original coordinates.
    PLA
    STA ObjY, X
    PLA
    STA ObjX, X
    JSR Digdogger_Draw

L_Digdogger_DrawAsLittle:
    ; Save the original coordinates.
    LDA ObjX, X
    PHA
    LDA ObjY, X
    PHA

    ; Add 8 to Y and X.
    CLC
    ADC #$08
    STA ObjY, X
    LDA ObjX, X
    CLC
    ADC #$08
    STA ObjX, X

    ; Save the child flag and object type.
    LDA Digdogger_ObjIsChild, X
    PHA
    LDA ObjType, X
    PHA

    ; Change object type to $18 (Little Digdogger),
    ; and child flag to 1, in order to draw as a little digdogger.
    LDA #$18
    STA ObjType, X
    LDA #$01
    STA Digdogger_ObjIsChild, X
    JSR Digdogger_Draw

    ; Restore the child flag and object type.
    PLA
    STA ObjType, X
    PLA
    STA Digdogger_ObjIsChild, X

    ; Restore the original coordinates.
    PLA
    STA ObjY, X
    PLA
    STA ObjX, X
    RTS

Digdogger_ChangeSpeed:
    LDA Digdogger_ObjSpeedFlag, X
    JSR TableJump

Digdogger_ChangeSpeed_JumpTable:
    .ADDR Digdogger_SpeedUp
    .ADDR Digdogger_SlowDown

Digdogger_SpeedUp:
    ; Add 1 to 16-bit speed.
    INC Digdogger_ObjSpeedFrac, X
    BNE :+
    INC Digdogger_ObjSpeedWhole, X
:
    ; If the 16-bit speed has not reached the 16-bit target speed, then return.
    LDA Digdogger_ObjSpeedFrac, X
    CMP Digdogger_ObjTargetSpeedFrac, X
    BNE L116EA_Exit
    LDA Digdogger_ObjSpeedWhole, X
    CMP Digdogger_ObjTargetSpeedWhole, X
    BNE L116EA_Exit

    ; Set speed flag to 1 meaning decelerate.
    INC Digdogger_ObjSpeedFlag, X

    ; Go set target speed to $0040 (or $0140 if this is a child).
    LDA #$40
    JMP SetTargetSpeed

Digdogger_SlowDown:
    ; Subtract 1 from 16-bit speed.
    DEC Digdogger_ObjSpeedFrac, X
    LDA Digdogger_ObjSpeedFrac, X
    CMP #$FF
    BNE :+
    ; Begin unverified code 116C2
    DEC Digdogger_ObjSpeedWhole, X
    ; End unverified code
:
    ; If the 16-bit speed has not reached the 16-bit target speed, then return.
    LDA Digdogger_ObjSpeedFrac, X
    CMP Digdogger_ObjTargetSpeedFrac, X
    BNE L116EA_Exit
    LDA Digdogger_ObjSpeedWhole, X
    CMP Digdogger_ObjTargetSpeedWhole, X
    BNE L116EA_Exit

    ; Set speed flag to 0 meaning accelerate.
    DEC Digdogger_ObjSpeedFlag, X

    ; Set target speed to $0080.
    LDA #$80

SetTargetSpeed:
    STA Digdogger_ObjTargetSpeedFrac, X
    LDA #$00
    STA Digdogger_ObjTargetSpeedWhole, X

    ; But if this is a child, then make the target speed $0180.
    LDA Digdogger_ObjIsChild, X
    BEQ L116EA_Exit
    INC Digdogger_ObjTargetSpeedWhole, X

L116EA_Exit:
    RTS

Digdogger_Move:
    ; This calculation is the same one used in Manhandla to
    ; manifest the fractional speed over several frames.
    ;
    ; To summarize, the formula is:
    ;
    ;   (SA=speed accumulator, SH=speed high, SL=speed low)
    ;
    ;   SA := SA + (SL AND $E0)
    ;   [03] := SH + carry
    ;
    ; See 04:A28E Manhandla_Move.
    LDA Digdogger_ObjSpeedFrac, X
    AND #$E0
    CLC                         ; Add the low speed byte to speed accumulator.
    ADC $0412, X
    STA $0412, X

    ; Assign (high speed byte + carry) to [03].
    LDA Digdogger_ObjSpeedWhole, X
    ADC #$00
    STA $03

    ; Change coordinates by the speed amount/offset in [03]
    ; according to the direction.
    ;
    ; Start with a mask of $A1 in [02]:
    ; - low nibble represents the right direction
    ; - high nibble is used to set or clear carry as we go along
    LDA #$A1
    STA $02

    ; If direction has a right component (1), then add offset to X coordinate.
    LDA ObjDir, X
    BIT $02
    BEQ @Left
    LDA ObjX, X
    CLC
    ADC $03
    STA ObjX, X

@Left:
    ; If direction has a left component (2), then subtract offset from X coordinate.
    LDA ObjDir, X
    ASL $02                     ; Bit 7 of mask $A1 sets Carry now.
    BIT $02
    BEQ @Down
    LDA ObjX, X
    SBC $03
    STA ObjX, X

@Down:
    ; If direction has a down component (4), then add offset to Y coordinate.
    LDA ObjDir, X
    ASL $02                     ; Bit 6 of mask $A1 clears Carry now.
    BIT $02
    BEQ @Up
    LDA ObjY, X
    ADC $03
    STA ObjY, X

@Up:
    ; If direction has an up component (8), then subtract offset from Y coordinate.
    LDA ObjDir, X
    ASL $02                     ; Bit 5 of mask $A1 sets Carry now.
    BIT $02
    BEQ @End
    LDA ObjY, X
    SBC $03
    STA ObjY, X

@End:
    JMP Anim_FetchObjPosForSpriteDescriptor

DigdoggerSpriteOffsetsX:
    .BYTE $00, $10, $00, $10

DigdoggerSpriteOffsetsY:
    .BYTE $00, $00, $10, $10

DigdoggerSpriteAttrs:
    .BYTE $03, $03, $83, $83

Digdogger_Draw:
    LDA #$06                    ; 6 animation frames a screen frame
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor

    ; If this is a child digdogger, then go draw it.
    LDA Digdogger_ObjIsChild, X
    BNE @DrawLittle

    ; For each part of the big digdogger, from 0 to 3,
    ; indexed by Y register:
    LDY #$00

@LoopPart:
    ; Store in [00] the X coordinate offset for the current corner.
    LDA ObjX, X
    CLC
    ADC DigdoggerSpriteOffsetsX, Y
    STA $00

    ; Store in [01] the Y coordinate offset for the current corner.
    LDA ObjY, X
    CLC
    ADC DigdoggerSpriteOffsetsY, Y
    STA $01

    ; Load and set the sprite attributes for this part.
    ; Vertical flipping is set appropriately here.
    LDA DigdoggerSpriteAttrs, Y
    JSR Anim_SetSpriteDescriptorAttributes
    TYA                         ; Save the loop index.
    PHA

    ; Parts at odd indexes are flipped horizontally.
    AND #$01
    STA $0F
    LDA #$00                    ; There's only 1 frame image.
    JSR DrawObjectNotMirrored
    PLA                         ; Restore the loop index.
    TAY

    ; Bottom of the loop.
    ; Increment Y register until = 4.
    INY
    CPY #$04
    BNE @LoopPart
    RTS

@DrawLittle:
    ; A little digdogger is easy to draw.
    JSR Anim_SetSpriteDescriptorLevelPaletteRow
    LDA ObjAnimFrame, X
    JMP DrawObjectMirrored
