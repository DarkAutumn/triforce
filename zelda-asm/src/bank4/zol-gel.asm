; ============================================================================
; FILE:     bank4/zol-gel.asm            BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 1239-1569
; SUBJECTS: Gel, Zol
; TOPIC:    Zol states and split into Gels; Gel
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0026 UpdateZol                 code
;   L0041 UpdateZolState            code
;   L0045 UpdateZolState_JumpTable  data
;   L0050 UpdateZolState0_Wander    code
;   L0054 UpdateZolState1_Shove     code
;   L0062 UpdateZolState2_Split     code
;   L0087 CreateChildGel            code
;   L0103 Zol_CheckCollisions       code
;   L0166 InitGel                   code
;   L0171 UpdateGel                 code
;   L0200 Gel_Move                  code
;   L0263 UpdateNormalZolOrGel      code  Params:
;   L0304 ZolGelDelays              data
;   L0312 Gel_MoveSplitting         code  Returns:
;   L0349 Gel_CheckCollisions       code
; ============================================================================

UpdateZol:
    JSR UpdateZolState
    JSR Zol_CheckCollisions
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Every 8 screen frames, switch between frame image 0 and 1.
    LDY #$00
    LDA FrameCounter
    AND #$08
    BNE :+
    INY
:
    TYA
    JMP DrawObjectMirrored

UpdateZolState:
    LDA ObjState, X
    JSR TableJump

UpdateZolState_JumpTable:
    .ADDR UpdateZolState0_Wander
    .ADDR UpdateZolState1_Shove
    .ADDR UpdateZolState2_Split

UpdateZolState0_Wander:
    LDA #$18                    ; QSpeed $18 (0.375 pixels a frame)
    JMP UpdateNormalZolOrGel

UpdateZolState1_Shove:
    ; Move straight and fast until blocked. Then go to state 2.
    JSR Gel_MoveSplitting
    BCC :+
    INC ObjState, X
:
    RTS

UpdateZolState2_Split:
    ; We'll create two gels, but destroy the zol. So, increase
    ; the object count by 1 only.
    INC RoomObjCount
    JSR DestroyMonster
    JSR CreateChildGel

    ; If the zol's direction is vertical, then set the first child gel's
    ; direction to left, else up.
    LDA ObjDir, X
    CMP #$04
    LDA #$02
    BCS :+
    LDA #$08
:
    STA a:ObjDir, Y
    PHA                         ; Save the direction of the first gel.
    JSR CreateChildGel
    PLA                         ; Restore the direction of the first gel.

    ; Set the second child gel's direction to the opposite of the first one's.
    LSR
    STA a:ObjDir, Y
    RTS

CreateChildGel:
    LDA #$14                    ; Child Gel object type

    ; Use the shooting operation to create the gel object.
    STA $00
    JSR ShootLimited

    ; Shots start in state $10. But Child Gel needs to start in state 0.
    LDA #$00
    STA a:ObjState, Y

    ; Give the child the same grid offset as the parent.
    LDA ObjGridOffset, X
    STA ObjGridOffset, Y
    RTS

Zol_CheckCollisions:
    ; If not in state 0, then return.
    LDA ObjState, X
    BNE @Exit

    ; Check collisions.
    ;
    ; If killed, or got hit but didn't get hurt (invincibility timer = 0),
    ; then return.
    JSR Gel_CheckCollisions
    LDA ObjMetastate, X
    BNE @Exit
    LDA ObjInvincibilityTimer, X
    BEQ @Exit

    ; If vertically aligned with the square grid, then
    ; use a horizontal direction mask (3).
    ;
    ; Keep in mind that most objects are aligned vertically when
    ; the low nibble of their Y is $D.
    LDY #$00
    LDA ObjY, X
    AND #$0F
    CMP #$0D
    BNE :+
    LDY #$03
:
    STY $00

    ; If horizontally aligned with the square grid, then
    ; use a vertical direction mask ($C).
    LDY #$00
    LDA ObjX, X
    AND #$0F
    BNE :+
    LDY #$0C
:
    TYA

    ; Combine the two masks. At most one will be 0.
    ORA $00

    ; The point is to see if the zol is facing in a way that's aligned
    ; with the grid.
    ;
    ; If so, then go to state 1 to do a "big shove".
    ; Else go to state 2, to immediately split in two.
    ;
    ; Examples:
    ; It would be aligned, if moving left along a tile as usual.
    ; It would *not* be aligned, if while moving left, it was hit downward.
    ; - The collision check routines would have made the zol
    ;   face in the direction of the weapon.
    ; - See CheckMonsterWeaponCollision.
    AND ObjDir
    BNE :+
    INC ObjState, X
:
    INC ObjState, X

@Exit:
    RTS

InitGel:
    LDA #$02                    ; Gels that aren't split from a Zol begin in state 2.
    STA ObjState, X
    JMP InitWalker

UpdateGel:
    JSR Gel_Move
    JSR Gel_CheckCollisions

    ; Gels are narrow. So, temporarily shift the X coordinate
    ; 4 pixels to the right.
    LDA ObjX, X
    PHA                         ; Save the original X coordinate.
    CLC
    ADC #$04
    STA ObjX, X
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Every two screen frames, switch between frame image 0 and 1.
    LDY #$00
    LDA FrameCounter
    AND #$02
    BNE :+
    INY
:
    ; Set sprite attributes 3: normal sprite, palette row "level".
    LDA #$03
    JSR Anim_SetSpriteDescriptorAttributes
    TYA                         ; Pass the frame image in A register.
    JSR DrawObjectNotMirrored
    PLA                         ; Restore the original X coordinate.
    STA ObjX, X
    RTS

Gel_Move:
    ; Go handle state 0 or 2 as appropriate.
    LDY ObjState, X
    BEQ @State0
    DEY
    BNE @State2

    ; State 1.
    ;
    ; If the timer has not expired, then move.
    LDA ObjTimer, X
    BEQ @Move
    JSR Gel_MoveSplitting

    ; If not blocked while moving, then return.
    BCC @Exit

@Move:
    ; Blocked or timer expired.
    ;
    ; Set X coordinate to ((X + 8) AND $F0).
    LDA ObjX, X
    CLC
    ADC #$08
    AND #$F0
    STA ObjX, X

    ; Set Y coordinate to (((Y + 8) AND $F0) OR $D).
    LDA ObjY, X
    ADC #$08
    AND #$F0
    ORA #$0D
    STA ObjY, X

    ; Reset the grid offset, and go to state 2.
    LDA #$00
    STA ObjGridOffset, X
    INC ObjState, X

@Exit:
    RTS

@State0:
    ; State 0.
    ;
    ; Immediately go to state 1 with QSpeed $20 (half a pixel a frame),
    ; and 5 frame timer.
    LDA #$20
    STA ObjQSpeedFrac, X
    LDA #$05
    STA ObjTimer, X
    INC ObjState, X
    RTS

@State2:
    ; State 2.
    ;
    ; Set QSpeed $40 (1 pixel a frame).
    LDA #$40

; Params:
; A: qspeed
;
UpdateNormalZolOrGel:
    STA ObjQSpeedFrac, X

    ; If object timer >= 5, then return.
    LDA ObjTimer, X
    CMP #$05
    BCS @Exit
    LDA #$20                    ; Turn rate $20
    STA Wanderer_ObjTurnRate, X
    JSR Wanderer_TargetPlayer

    ; If the monster is between squares, or timer <> 0; then return.
    LDA ObjGridOffset, X
    ORA ObjTimer, X
    BNE @Exit

    ; Choose a random index (0 to 3) to look up an amount of time.
    LDA Random, X
    AND #$03
    TAY

    ; If the monster is Zol, then go use this index as is.
    LDA ObjType, X
    CMP #$13
    BEQ @SetDelay

    ; Else it's a gel. Add 4 to the index to look in the set of times
    ; for Gel.
    INY
    INY
    INY
    INY

@SetDelay:
    ; Set a delay at a tile edge for the next time to move.
    LDA ZolGelDelays, Y
    STA ObjTimer, X

@Exit:
    RTS

ZolGelDelays:
    .BYTE $18, $28, $38, $48, $08, $18, $28, $38

; Returns:
; C: 1 if blocked
;
; Set the speed to maximum.
;
Gel_MoveSplitting:
    LDA #$FF
    STA ObjQSpeedFrac, X

    ; Set moving direction to facing direction.
    LDA ObjDir, X
    STA $0F

    ; If at a tile boundary, then check for a tile collision.
    LDA ObjGridOffset, X
    BNE @CheckRoomBoundary
    JSR GetCollidingTileMoving

    ; If blocked by a tile, then return C=1.
    CMP RoomFirstUnwalkableTile
    BCS @Exit

@CheckRoomBoundary:
    ; If blocked by the room boundary, then return C=1.
    JSR BoundByRoom
    SEC
    BEQ @Exit

    ; Move.
    JSR MoveObject

    ; Mask the grid offset with $F (to a square length),
    ; and return C=0.
    LDA ObjGridOffset, X
    AND #$0F
    CLC
    BNE @Exit
    STA ObjGridOffset, X

@Exit:
    RTS

Gel_CheckCollisions:
    ; CheckMonsterCollisions calls GetObjectMiddle which
    ; overwrites [02] and [03]. So, these assignments might be redundant.
    LDA ObjX, X
    STA $02
    LDA ObjY, X
    STA $03
    JMP CheckMonsterCollisions
