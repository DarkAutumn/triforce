; ============================================================================
; FILE:     bank4/leever.asm             BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 2592-2966
; SUBJECTS: (none)
; TOPIC:    Blue and Red Leever; shared burrower code (Zora)
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0023 BlueLeeverStateQSpeeds                      data
;   L0026 BlueLeeverStateTimes                        data
;   L0029 BlueLeeverStateAnimTimes                    data
;   L0032 UpdateBlueLeever                            code
;   L0037 UpdateBurrower                              code
;   L0095 Burrower_AnimateDrawAndCheckCollisions      code  Params:
;   L0166 RedLeeverStateQSpeeds                       data
;   L0169 RedLeeverStateTimes                         data
;   L0172 RedLeeverStateAnimTimes                     data
;   L0175 UpdateRedLeever                             code
;   L0369 RedLeever_CycleStateDrawAndCheckCollisions  code
;   L0392 RedLeever_AnimateAndCheckCollisions         code
; ============================================================================

BlueLeeverStateQSpeeds:
    .BYTE $08, $0A, $10, $20, $10, $0A

BlueLeeverStateTimes:
    .BYTE $80, $20, $0F, $FF, $10, $60

BlueLeeverStateAnimTimes:
    .BYTE $10, $0B, $01, $05, $01, $0B

UpdateBlueLeever:
    LDA #$A0                    ; Turn rate $A0
    STA ObjTurnRate, X
    JSR Wanderer_TargetPlayer

UpdateBurrower:
    ; If object timer <> 0, then go animate, draw, and check for collisions.
    LDA ObjTimer, X
    BNE @Animate

    ; If not zora or state <> 1, go cycle the state.
    LDA ObjType, X
    CMP #$11
    BNE @CycleState
    LDY ObjState, X
    DEY
    BNE @CycleState

    ; Else is zora and state = 1.
    ;
    ; If zora's Y >= Link's Y, then set zora's frame image to
    ; 3 (back), else 2 (front).
    ;
    ; The frame image number is being assigned to ObjDir.
    LDY #$03
    LDA ObjY, X
    CMP ObjY
    BCS :+
    DEY
:
    STY ObjDir, X

@CycleState:
    ; Cycle the state between 0 and 5.
    LDA ObjState, X
    CLC
    ADC #$01
    CMP #$06
    BCC :+
    LDA #$00
:
    STA ObjState, X

    ; Look up and set the speed and timer for the current state.
    TAY
    LDA BlueLeeverStateQSpeeds, Y
    STA ObjQSpeedFrac, X
    LDA BlueLeeverStateTimes, Y
    STA ObjTimer, X

@Animate:
    ; Advance the animation counter; possibly setting a new
    ; value according to the new state.
    LDY ObjState, X
    LDA BlueLeeverStateAnimTimes, Y

; Params:
; A: new value for animation counter, in case it rolls over
; X: object index
;
; Note:
; Advances the animation counter first.
;
Burrower_AnimateDrawAndCheckCollisions:
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor

    ; If state = 0, return.
    LDA ObjState, X
    BEQ @Exit

    ; If is zora, and state >= 2 and < 5, then use frame image 2 or 3
    ; (front or back), which was stored in ObjDir above.
    LDY ObjType, X
    CPY #$11
    BNE @CalcFrameImage
    LDY ObjState, X
    CPY #$02
    BCC @CalcFrameImage
    CPY #$05
    BCS @CalcFrameImage
    LDA ObjDir, X
    BNE @Draw

@CalcFrameImage:
    ; Or else if is zora in state 1 or 5, then use one of
    ; the mound frame images:
    ;
    ; frame image := ((state - 1) * 2) + movement frame
    ;
    ; This yields 0/1 or 8/9, depending on the state.
    ;
    ; This formula is used for states 1 to 5 of non-zora burrowers.
    SEC
    SBC #$01
    ASL
    CLC
    ADC ObjAnimFrame, X

@Draw:
    JSR DrawObjectMirrored

    ; If not zora and state <> 3, 
    ; or zora and state <> 2, 3, 4;
    ; then return.
    LDA ObjType, X
    CMP #$11
    BNE @NoState3Collisions
    LDA ObjState, X
    CMP #$02
    BEQ @CheckCollisions
    CMP #$04
    BEQ @CheckCollisions

@NoState3Collisions:
    LDA ObjState, X
    CMP #$03
    BNE @Exit

@CheckCollisions:
    JSR CheckMonsterCollisions

    ; If not dying, then return.
    LDA ObjMetastate, X
    BEQ @Exit

    ; If monster is a red leever, then decrement the red leever count.
    LDA ObjType, X
    CMP #$10
    BNE @Exit
    DEC ActiveRedLeeverCount

@Exit:
    RTS

RedLeeverStateQSpeeds:
    .BYTE $00, $00, $00, $20, $00, $00

RedLeeverStateTimes:
    .BYTE $00, $10, $08, $FF, $08, $10

RedLeeverStateAnimTimes:
    .BYTE $10, $08, $08, $05, $08, $08

UpdateRedLeever:
    LDA ObjState, X
    BEQ :+
    JMP @CheckOtherStates
:
    ; State 0.
    ;
    ; In state we determine whether to burst out of the ground.
    ; If the long timer has not expired, then return.
    LDA RedLeeverLongTimer
    BNE @Exit

    ; Only allow two red leevers at a time.
    LDA ActiveRedLeeverCount
    CMP #$02
    BCS @Exit

    ; Face in the same direction as Link.
    LDA ObjDir
    STA ObjDir, X

    ; Randomly (>= $C0) face the opposite direction -- toward Link.
    LDA Random+1, X
    CMP #$C0
    BCC :+
    JSR ReverseObjDir
:
    ; If facing horizontally, go place the leever to the left or right of Link.
    LDA ObjDir, X
    AND #$0C
    BEQ @PlaceHorizontally

    ; Begin with the leever's X set to Link's X.
    LDA ObjX
    STA ObjX, X

    ; If facing down, then use Y offset $28.
    ; Else store Y offset -$28 in [00].
    LDY #$28
    LDA ObjDir, X
    AND #$08
    BEQ :+
    LDY #$D8
:
    STY $00

    ; Add Link's Y and the vertical offset.
    LDA ObjY
    CLC
    ADC $00

    ; Sanitize the Y coordinate:
    ; (Y AND $F0) OR $D
    AND #$F0
    ORA #$0D
    STA ObjY, X

    ; If Y coordinate >= $5D (top of the third square row), then
    ; go see if the tile is valid to come out of.
    ;
    ; Else it's too close to the top edge of the screen. So, return.
    CMP #$5D
    BCS @CheckSafeToSpawn

@Exit:
    RTS

@PlaceHorizontally:
    ; Begin with the leever's Y set to Link's Y.
    LDA ObjY
    STA ObjY, X

    ; If Y coordinate < $5D (top of the third square row), then
    ; it's too close to the top edge of the screen. So, return.
    CMP #$5D
    BCC @Exit

    ; If facing right, then use X offset $28.
    ; Else store X offset -$28 in [00].
    LDY #$28
    LDA ObjDir, X
    AND #$02
    BEQ :+
    LDY #$D8
:
    STY $00

    ; [01] holds Link's X coordinate.
    LDA ObjX
    STA $01

    ; Add Link's X and the horizontal offset.
    CLC
    ADC $00

    ; AND the result with $F8, so it's aligned with a tile horizontally.
    AND #$F8

    ; Assign that result to the leever's X coordinate and [02].
    STA ObjX, X
    STA $02

    ; If the leever's X [02] >= Link's X [01], then swap [02] and [01].
    CMP $01
    BCC @Subtract
    PHA
    LDA $01
    STA $02
    PLA
    STA $01

@Subtract:
    ; The point is to get the absolute distance between the two.
    LDA $01
    SEC
    SBC $02

    ; If the absolute distance >= $30, then return.
    ; It means that the leever's X coordinate wrapped around
    ; the left or right edge of the screen.
    CMP #$30
    BCS @Exit

@CheckSafeToSpawn:
    ; The leever can't come out of an unwalkable tile. So, return.
    JSR GetCollidableTileStill
    CMP ObjectFirstUnwalkableTile
    BCS @Exit
    INC ActiveRedLeeverCount

    ; Set animation counter to 1, so that it will roll over this frame
    ; and pick up whatever the new value is for state 1.
    LDA #$01
    STA ObjAnimCounter, X

    ; Set the long timer to 2, meaning 20 frames.
    ASL
    STA RedLeeverLongTimer
    JSR RedLeever_CycleStateDrawAndCheckCollisions

    ; The leever has to face Link in order to move toward him.
    JSR ReverseObjDir

@CheckOtherStates:
    ; If not in state 3, go check the object timer in order to draw and such.
    LDA ObjState, X
    CMP #$03
    BNE @AnimateIfTime

    ; State 3.
    ;
    ; If being shoved, go shove, animate, and draw the leever.
    LDA ObjShoveDir, X
    BEQ @CheckStunned
    JSR Obj_Shove
    JMP RedLeever_AnimateAndCheckCollisions

@CheckStunned:
    ; If we have the magic clock or the leever is stunned, then
    ; go animate, draw, and check for collisions.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE RedLeever_AnimateAndCheckCollisions

    ; If the leever is blocked by a tile or the room boundary,
    ; then go to the next state, in addition to animating, drawing,
    ; and checking for collisions.
    LDA ObjDir, X
    STA $0F
    JSR GetCollidingTileMoving
    CMP ObjectFirstUnwalkableTile
    BCS RedLeever_CycleStateDrawAndCheckCollisions
    JSR BoundByRoom
    BEQ RedLeever_CycleStateDrawAndCheckCollisions

    ; Move.
    JSR MoveObject

    ; Truncate the grid offset.
    LDA ObjGridOffset, X
    AND #$0F
    BNE :+
    STA ObjGridOffset, X
:
    ; Set timer to $FF.
    LDA #$FF
    STA ObjTimer, X

@AnimateIfTime:
    ; If object timer has not expired, then
    ; go animate, draw, and check for collisions.
    LDA ObjTimer, X
    BNE RedLeever_AnimateAndCheckCollisions

RedLeever_CycleStateDrawAndCheckCollisions:
    ; Cycle the state between 0 and 5.
    ;
    ; If a cycle was completed (wrapped around to 0),
    ; then decrease the active red leever count.
    LDA ObjState, X
    CLC
    ADC #$01
    CMP #$06
    BCC :+
    DEC ActiveRedLeeverCount
    LDA #$00
:
    STA ObjState, X

    ; Look up and set the speed, timer, and animation counter
    ; for the current state.
    TAY
    LDA RedLeeverStateQSpeeds, Y
    STA ObjQSpeedFrac, X
    LDA RedLeeverStateTimes, Y
    STA ObjTimer, X

RedLeever_AnimateAndCheckCollisions:
    LDY ObjState, X
    LDA RedLeeverStateAnimTimes, Y
    JMP Burrower_AnimateDrawAndCheckCollisions
    ; Begin unverified code 10BFE
    RTS
