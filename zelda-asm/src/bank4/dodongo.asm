; ============================================================================
; FILE:     bank4/dodongo.asm            BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 5855-6447
; SUBJECTS: Anim, Dodongo
; TOPIC:    Dodongo states, bloat/stun, bomb eating, collisions, draw
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0042 UpdateDodongo                            code
;   L0051 DodongoBloatedWaitTimes                  data
;   L0054 UpdateDodongoState                       code
;   L0058 UpdateDodongoState_JumpTable             data
;   L0063 UpdateDodongoState0_Move                 code
;   L0107 UpdateDodongoState2_Stunned              code
;   L0121 UpdateDodongoState1_Bloated              code
;   L0125 UpdateDodongoState1_Bloated_JumpTable    data
;   L0132 UpdateDodongoState1_Bloated_Sub_Wait     code
;   L0187 L_Dodongo_DecrementBloatedTimer          code
;   L0192 UpdateDodongoState1_Bloated_Sub_Die      code
;   L0196 UpdateDodongoState1_Bloated_Sub_End      code
;   L0201 Dodongo_CheckCollisions                  code
;   L0239 L11989_Exit                              code
;   L0242 Dodongo_CheckCollisionsStandardSize      code
;   L0260 DodongoMouthNegativeLimits0              data
;   L0263 DodongoMouthPositiveLimits0              data
;   L0266 DodongoMouthNegativeLimits1              data
;   L0269 DodongoMouthPositiveLimits1              data
;   L0272 Dodongo_CheckBombHit                     code
;   L0332 Dodongo_TryEatBomb                       code
;   L0397 DodongoBombPositiveLimits                data
;   L0400 DodongoBombNegativeLimits                data
;   L0418 Dodongo_IsBombInRange                    code  Params:
;   L0471 DodongoFrameImages                       data  Two sets of frame image numbers: 1 set for each animation frame.
;   L0475 DodongoFrameHFlips                       data
;   L0479 DodongoFrameImagesBloated                data
;   L0483 DodongoFrameHFlipsBloated                data
;   L0487 Dodongo_Draw                             code
;   L0623 Anim_SetSpriteDescriptorLevelPaletteRow  code  Returns:
;   L0627 L11B00_Exit                              code
; ============================================================================

UpdateDodongo:
    JSR UpdateDodongoState
    JSR Dodongo_CheckCollisions
    JSR Dodongo_CheckBombHit
    JMP Dodongo_Draw

; Unknown block
    .BYTE $10, $F0, $10, $FF, $F0

DodongoBloatedWaitTimes:
    .BYTE $20, $40, $40

UpdateDodongoState:
    LDA ObjState, X
    JSR TableJump

UpdateDodongoState_JumpTable:
    .ADDR UpdateDodongoState0_Move
    .ADDR UpdateDodongoState1_Bloated
    .ADDR UpdateDodongoState2_Stunned

UpdateDodongoState0_Move:
    ; If facing left, then push X offset 0 on the stack.
    ; Else push X offset -$10, and add $10 to X coordinate.
    ;
    ; We would like to use one set of coordinates only.
    ; But movement would break down, if we did; because
    ; Dodongo is so long.
    ;
    ; So, when not facing left, we temporarily shift the
    ; X coordinate to be closer to the other side of the monster.
    LDY #$00
    LDA ObjDir, X
    AND #$0D
    BEQ @Move
    LDA ObjX, X
    CLC
    ADC #$10
    STA ObjX, X
    LDY #$F0

@Move:
    TYA
    PHA
    LDA #$20                    ; Turn rate $20
    STA Wanderer_ObjTurnRate, X
    JSR Wanderer_TargetPlayer

    ; Pop the offset and add it to the X coordinate.
    ;
    ; This brings the X coordinate back to the unique value
    ; that represents the position.
    PLA
    CLC
    ADC ObjX, X
    STA ObjX, X

    ; If X coordinate < $20, then face right.
    CMP #$20
    BCS :+
    LDA #$01
    STA ObjDir, X
:
    RTS

UpdateDodongoState2_Stunned:
    ; Take action depending on the value of stun timer:
    ; 0:     set stun timer to $20 
    ; 1:     go back to state 0
    ; other: return
    LDY a:ObjStunTimer, X
    DEY
    BEQ UpdateDodongoState1_Bloated_Sub_End
    BPL :+
    LDA #$20
    STA a:ObjStunTimer, X
:
    RTS

UpdateDodongoState1_Bloated:
    LDA Dodongo_ObjBloatedSubstate, X
    JSR TableJump

UpdateDodongoState1_Bloated_JumpTable:
    .ADDR UpdateDodongoState1_Bloated_Sub_Wait
    .ADDR UpdateDodongoState1_Bloated_Sub_Wait
    .ADDR UpdateDodongoState1_Bloated_Sub_Wait
    .ADDR UpdateDodongoState1_Bloated_Sub_Die
    .ADDR UpdateDodongoState1_Bloated_Sub_End

UpdateDodongoState1_Bloated_Sub_Wait:
    ; Bloated timer =>
    ; 0:     fall thru to set a new timer and check bomb hits
    ; 1:     go advance the substate
    ; other: go decrement Bloated timer
    LDY Dodongo_ObjBloatedTimer, X
    DEY
    BEQ @AdvanceSubstate
    BPL L_Dodongo_DecrementBloatedTimer

    ; Set the bloated timer according to the substate.
    LDY Dodongo_ObjBloatedSubstate, X
    LDA DodongoBloatedWaitTimes, Y
    STA Dodongo_ObjBloatedTimer, X

    ; If substate <> 0, go decrement bloated timer.
    CPY #$00
    BNE L_Dodongo_DecrementBloatedTimer

    ; Deactivate the bomb in the first bomb slot ($10).
    LDY #$10
    LDA #$00
    STA a:ObjState, Y

    ; Increment the number of bomb hits.
    LDA Dodongo_ObjBombHits, X
    CLC
    ADC #$01
    STA Dodongo_ObjBombHits, X
    JMP L_Dodongo_DecrementBloatedTimer

@AdvanceSubstate:
    ; Advance the substate.
    INC Dodongo_ObjBloatedSubstate, X

    ; If new bloated substate < 2, then go decrement the bloated timer.
    LDA Dodongo_ObjBloatedSubstate, X
    CMP #$02
    BCC L_Dodongo_DecrementBloatedTimer

    ; New bloated substate >= 2.
    ;
    ; If bomb hits < 2, then set bloated substate 4,
    ; which will transition to state 0 (moving).
    ;
    ; Note that if the bomb hits >= 2, then:
    ; - If new substate = 2, then substate 2 will wait,
    ;   and then transition to substate 3.
    ; - If new substate = 3, then the next frame the monster will die.
    LDY Dodongo_ObjBombHits, X
    CPY #$02
    BCS L_Dodongo_DecrementBloatedTimer
    LDA #$04
    STA Dodongo_ObjBloatedSubstate, X

L_Dodongo_DecrementBloatedTimer:
    ; Either way, decrement the bloated timer.
    DEC Dodongo_ObjBloatedTimer, X
    RTS

UpdateDodongoState1_Bloated_Sub_Die:
    JSR UpdateDeadDummy
    JSR PlayBossDeathCry

UpdateDodongoState1_Bloated_Sub_End:
    JSR ResetObjState           ; Go to state 0.
    STA Dodongo_ObjBloatedSubstate, X    ; Reset bloated substate.
    RTS

Dodongo_CheckCollisions:
    ; Check for collisions using a regular monster size.
    JSR Dodongo_CheckCollisionsStandardSize

    ; If harmed, then go die.
    LDA ObjInvincibilityTimer, X
    BNE @Die

    ; If facing vertically, then return.
    LDA ObjDir, X
    CMP #$04
    BCS L11989_Exit

    ; Save the X coordinate.
    ; Then add $10 to check the right-hand side.
    LDA ObjX, X
    PHA
    CLC
    ADC #$10
    STA ObjX, X
    JSR Dodongo_CheckCollisionsStandardSize

    ; Restore the X coordinate.
    PLA
    STA ObjX, X

    ; If not harmed, then return.
    LDA ObjInvincibilityTimer, X
    BEQ L11989_Exit

@Die:
    JSR UpdateDodongoState1_Bloated_Sub_Die

    ; Drop a bomb from this fight.
    LDA #$0A
    STA HelpDropCount
    STA IsHelpDropBomb

L11989_Exit:
    RTS

Dodongo_CheckCollisionsStandardSize:
    LDA #$FF
    STA ObjInvincibilityMask, X
    JSR CheckMonsterCollisions

    ; If state <> 2 (stunned), then return.
    LDA ObjState, X
    CMP #$02
    BNE L11989_Exit

    ; Take out the bit for the sword in the invincibility mask.
    ; Check for collision with Link's sword.
    LDA #$FE
    STA ObjInvincibilityMask, X
    JSR GetObjectMiddle
    LDY #$0D
    JMP CheckMonsterSwordCollision

DodongoMouthNegativeLimits0:
    .BYTE $F0, $00, $F8, $FF, $F8

DodongoMouthPositiveLimits0:
    .BYTE $00, $10, $08, $FF, $08

DodongoMouthNegativeLimits1:
    .BYTE $FC, $FC, $F0, $FF, $00

DodongoMouthPositiveLimits1:
    .BYTE $04, $04, $00, $FF, $10

Dodongo_CheckBombHit:
    ; If state <> 0 (not moving), then return.
    LDA ObjState, X
    BNE @Exit

    ; If facing vertically, then store (X + 8) in [00].
    ; Else store (X + $10).
    LDA ObjX, X
    CLC
    ADC #$08
    LDY ObjDir, X
    CPY #$04
    BCS :+
    ADC #$08
:
    STA $00

    ; Store (Y + 8) in [01].
    LDA ObjY, X
    ADC #$08
    STA $01

    ; Only check the first bomb slot ($10).
    ;
    ; Store the bomb's (X + 8) in [02].
    LDY #$10
    LDA a:ObjX, Y
    ADC #$08
    STA $02

    ; Store the bomb's (Y + 8) in [03].
    LDA a:ObjY, Y
    ADC #$08
    STA $03

    ; But, if there's no bomb, then return.
    LDA a:ObjState, Y
    BEQ @Exit

    ; If the bomb has not exploded (state $12), then
    ; go see if Dodongo will eat it.
    CMP #$12
    BEQ Dodongo_TryEatBomb

    ; If the object in the slot is actually a fire, then return.
    CMP #$20
    BCS @Exit

    ; The bomb already blew up.
    ; If the dust cloud is in range of the monster,
    ; then set stunned state (2).
    LDY #$00
    JSR Dodongo_IsBombInRange
    BNE @Exit
    LDA #$02
    STA ObjState, X

@Exit:
    RTS

Dodongo_TryEatBomb:
    ; Try to eat the bomb.
    ;
    ; If the bomb is not in range of the monster, then return.
    ;
    ; This is a coarse test, to see if the bomb is over any part of the monster.
    LDY #$01
    JSR Dodongo_IsBombInRange
    BNE @Exit

    ; Will loop from 1 to 0, indexed by [00].
    LDA #$01
    STA $00

    ; Shift facing direction right 1 to make an index. Value 3 is unused.
    LDA ObjDir, X
    LSR
    TAY

    ; Now we have to see if it's near the mouth. But the mouth
    ; position depends on the orientation of the monster.
    ;
    ; Start with the horizontal distance.
    LDA $04

@LoopLimit:
    ; Look up the limits for the given direction.
    ;
    ; If the distance from the bomb < negative limit,
    ; or >= positive limit, then return.
    ;
    ; Note that these are simple signed comparisons that work,
    ; because the operands are in a small range.
    CMP DodongoMouthNegativeLimits0, Y
    BMI @Exit
    CMP DodongoMouthPositiveLimits0, Y
    BPL @Exit

    ; Add $A to the index to look into the second set for vertical limits.
    TYA
    CLC
    ADC #$0A
    TAY

    ; Bottom of the loop.
    ;
    ; Load the vertical distance for the next iteration.
    ; Decrement loop counter.
    LDA $05
    DEC $00
    BPL @LoopLimit

    ; If we get here, then the bomb was within range of the mouth.
    ; Go to state 1 (bloated).
    INC ObjState, X

    ; Deactivate the bomb, and reset the bloated substate.
    LDY #$10
    LDA #$00
    STA a:ObjState, Y
    STA Dodongo_ObjBloatedSubstate, X

@Exit:
    RTS

DodongoBombPositiveLimits:
    .BYTE $0C, $11

DodongoBombNegativeLimits:
    .BYTE $F4, $F0

; Params:
; Y: limit index (0 for dust cloud, 1 for the bomb)
; [00]: monster hotspot X
; [01]: monster hotspot Y
; [02]: bomb hotspot X
; [03]: bomb hotspot Y
;
; Returns:
; A: 0 if the bomb is in range of the monster
; Z: 1 if the bomb is in range of the monster
; [04]: horizontal distance
; [05]: vertical distance
;
; Store the positive limit in [06], and the negative one in [07].
;
Dodongo_IsBombInRange:
    LDA DodongoBombPositiveLimits, Y
    STA $06
    LDA DodongoBombNegativeLimits, Y
    STA $07

    ; Set [08] to 3. It will be shifted right once for each axis that
    ; the bomb is near enough the monster.
    LDA #$03
    STA $08

    ; For each coordinate (Y/X), from 1 to 0, indexed by Y register:
    LDY #$01

@LoopAxis:
    ; Subtract the bomb's coordinate from the boss's,
    ; The loop index determines the coordinate.
    LDA $0000, Y
    SEC
    SBC $0002, Y

    ; If the distance >= positive offset, or < negative offset;
    ; then loop again.
    CMP $06
    BPL @NextLoopAxis
    CMP $07                     ; [07] negative offset
    BMI @NextLoopAxis

    ; The distance is close enough in this axis.
    ;
    ; Store the distance in this axis in [05] or [04],
    ; depending on the loop index.
    STA $0004, Y

    ; Shift [08] right, because we're in range in this axis.
    LSR $08

@NextLoopAxis:
    DEY
    BPL @LoopAxis

    ; If close enough in both axes, then [08] will be 0.
    LDA $08
    RTS

; Two sets of frame image numbers: 1 set for each animation frame.
; Each set contains 5 frame image numbers, indexed by direction.
;
; But these are not typical direction indexes. Instead a
; 5-way index is used.
; A direction shifted right once becomes a 5-way index.
; The fourth element is unused.
;
DodongoFrameImages:
    .BYTE $00, $01, $06, $FF, $08, $02, $03, $06
    .BYTE $FF, $08

DodongoFrameHFlips:
    .BYTE $00, $40, $00, $FF, $00, $00, $40, $40
    .BYTE $FF, $40

DodongoFrameImagesBloated:
    .BYTE $04, $05, $07, $FF, $09, $04, $05, $07
    .BYTE $FF, $09

DodongoFrameHFlipsBloated:
    .BYTE $00, $40, $00, $FF, $00, $00, $40, $00
    .BYTE $FF, $00

Dodongo_Draw:
    JSR Anim_SetSpriteDescriptorLevelPaletteRow

    ; Make a 5-way direction index out of the facing direction
    ; by shifting right once, and store it in [00].
    LDA ObjDir, X
    LSR
    STA $00

    ; If state = 0, go handle animating the two frames of this state.
    LDY ObjState, X
    BEQ @DrawWalkingFast

    ; If state > 1, then the monster is stunned (state 2).
    ; So, go switch frames every $20 screen frames.
    LDA #$20
    DEY
    BNE @DrawWalking

    ; State = 1. Bloated.
    ;
    ; In substate 0, go animate as usual for walking.
    LDY Dodongo_ObjBloatedSubstate, X
    BEQ @DrawWalkingFast

    ; If in bloated substates 2 or 3 (last wait or die), then
    ; go fade the boss; so that some frames are not drawn.
    CPY #$02
    BEQ @DrawFaded
    CPY #$03
    BEQ @DrawFaded

    ; State 1, substate 1
    ;
    ; Add $14 to the frame image index to index into the bloated
    ; frame images and H-flips.
    LDA $00
    CLC
    ADC #$14
    TAY
    JMP @PrepareToDraw

@DrawFaded:
    ; In bloated substates 2 and 3:
    ; Every two screen frames, don't draw for two frames. Return.
    LDA FrameCounter
    AND #$02
    BEQ L11B00_Exit
    LDY $00
    JMP @PrepareToDraw

@DrawWalkingFast:
    ; Every 8 screen frames, switch between 2 animation frames.
    LDA #$08

@DrawWalking:
    LDY $00
    AND FrameCounter
    BEQ @PrepareToDraw

    ; For animation frame 1, add 5 to the index in Y register,
    ; in order to index into the second set of frame image numbers.
    TYA
    CLC
    ADC #$05
    TAY

@PrepareToDraw:
    ; Start preparing sprites.
    JSR Anim_FetchObjPosForSpriteDescriptor
    TYA                         ; Save the frame image index (not frame image number).
    PHA

    ; Get the horizontal flipping and frame image number for this frame image.
    LDA DodongoFrameHFlips, Y
    STA $0F                     ; [0F] horizontal flipping
    LDA DodongoFrameImages, Y

    ; Frame image numbers 7 and 9 are vertical bloated frame images.
    ; They are drawn mirrored.
    CMP #$07
    BEQ :+
    CMP #$09
    BNE @DrawLeftSide
:
    JSR DrawObjectMirrored
    JMP @DrawRightSide

@DrawLeftSide:
    ; No other frame images are mirrored.
    JSR DrawObjectNotMirrored

@DrawRightSide:
    PLA                         ; Restore the frame image index
    TAY

    ; If facing vertically, then return. There's no other side to draw.
    LDA ObjDir, X
    AND #$03
    BEQ L11B00_Exit

    ; Else draw the right side.
    ;
    ; Keep the Y coordinate in [01].
    ; But, store (X + $10) in [00].
    LDA ObjX, X
    CLC
    ADC #$10
    STA $00
    JSR Anim_SetSpriteDescriptorLevelPaletteRow

    ; XOR'ing the low bit swaps the left and right frame numbers.
    ;
    ; So, if we had animation frame 1, direction left:
    ; - 5-way direction index = 1 (2 >> 1)
    ; - frame image index = 6 (1 + 5)
    ;   (dodongo/walk/right/right half/anim frame 1)
    ; - frame number = 3 (look up element 6 in frame image array)
    ; - horizontal flip = true (loop up element 6 in H-flip array)
    ;
    ; So, 3 would have been the frame number used above for
    ; the left side.
    ;
    ; Here, the frame number is 2 (3 XOR 1), meaning
    ; (dodongo/walk/right/left half/anim frame 1). The horizontal
    ; flipping flag is unchanged. So, this left half frame is shown correctly
    ; on the right.
    LDA DodongoFrameImages, Y
    EOR #$01
    JMP DrawObjectNotMirrored

; Returns:
; A: 3
; [04]: left sprite attributes
; [05]: right sprite attributes
;
Anim_SetSpriteDescriptorLevelPaletteRow:
    LDA #$03
    JSR Anim_SetSpriteDescriptorAttributes

L11B00_Exit:
    RTS

; Unknown block
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF
