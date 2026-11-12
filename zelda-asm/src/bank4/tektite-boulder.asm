; ============================================================================
; FILE:     bank4/tektite-boulder.asm    BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 2167-2590
; SUBJECTS: Jumper
; TOPIC:    Boulder spawner; Tektite/Boulder jumper physics
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0024 UpdateBoulderSet                  code
;   L0084 JumperYOffsets                    data
;   L0095 JumperYAccelerations              data  Logically, this array would be 33 bytes: 3 sets of 11 bytes,
;   L0101 JumperStartSpeedsHi               data
;   L0104 JumperYAccelerationBaseOffsets    data
;   L0107 UpdateTektiteOrBoulder            code
;   L0322 Jumper_AnimateAndCheckCollisions  code
;   L0378 Jumper_GetKind                    code  Returns:
;   L0387 L10A10_Exit                       code
;   L0390 Jumper_PointBoulderDownward       code
;   L0406 Jumper_MoveY                      code  Params:
;   L0442 Jumper_ResetVSpeedFrac            code
;   L0446 L10A52_Exit                       code
; ============================================================================

UpdateBoulderSet:
    ; If the object timer has not expired, then return.
    LDA ObjTimer, X
    BNE @Exit

    ; If the maximum number (3) of boulders are active, then
    ; go set the timer to wait a random amount of time.
    LDA ActiveBoulderCount
    CMP #$03
    BEQ @RandomizeTimer

    ; If no empty monster slot is found, then return.
    JSR FindEmptyMonsterSlot
    BEQ @Exit

    ; Set the boulder monster type, and increase the number
    ; of active boulders.
    LDX EmptyMonsterSlot
    INC ActiveBoulderCount
    LDA #$20
    JSR SetTypeAndClearObject

    ; Give the boulder a random X coordinate in the same half
    ; of the screen as the chase target.
    LDA Random+1, X
    LDY ChaseTargetX
    CPY #$80                    ; If chase target X < $80, mask Random with $7F.
    BCS @RightHalf
    AND #$7F
    JMP @SetX

@RightHalf:
    ORA #$80                    ; Else make it >= $80.

@SetX:
    STA ObjX, X

    ; Start at the top edge of the screen ($40).
    LDA #$40
    STA ObjY, X

    ; Set the boulder set's timer to ((Random + 8) AND $1F).
    LDX CurObjIndex
    LDA #$08
    CLC
    ADC Random+1, X
    AND #$1F
    STA ObjTimer, X
    RTS

@RandomizeTimer:
    ; Randomize the object timer.
    LDA ObjTimer, X
    CLC
    ADC Random+1, X
    STA ObjTimer, X

@Exit:
    RTS

JumperYOffsets:
    .BYTE $00, $00, $00, $00, $00, $20, $20, $00
    .BYTE $00, $E0, $E0

; Logically, this array would be 33 bytes: 3 sets of 11 bytes,
; 1 byte for each 8-way direction
;
; But because Boulder does not need the upward accelerations,
; the elements for those are used for the starting speeds
; at 04:88F2.
;
JumperYAccelerations:
    .BYTE $00, $40, $40, $00, $00, $40, $40, $00
    .BYTE $00, $30, $30, $00, $80, $80, $00, $00
    .BYTE $80, $80, $00, $00, $50, $50, $60, $60
    .BYTE $60, $60, $60, $60, $60

JumperStartSpeedsHi:
    .BYTE $FD, $FC, $FE

JumperYAccelerationBaseOffsets:
    .BYTE $00, $0B, $16

UpdateTektiteOrBoulder:
    ; If the monster is shoved, then keep shoving it.
    LDA ObjShoveDir, X
    BEQ @CheckStunned
    ; Begin unverified code 108FC
    JMP Obj_Shove
    ; End unverified code

@DrawAndCheckCollisions:
    JMP Jumper_AnimateAndCheckCollisions

@CheckStunned:
    ; If we have the magic clock, or the monster is stunned; 
    ; then go animate, draw, and check collisions.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE @DrawAndCheckCollisions

    ; If state <> 0, go update the jump.
    LDA ObjState, X
    BNE @State1

    ; State 0. On the ground.
    ;
    ; If timer <> 0, then go animate, draw, and check collisions.
    LDA ObjTimer, X
    BNE @Animate

    ; State = 0 and object timer expired. Time to jump.
    ;
    ; First, turn 1 step toward the chase target.
    JSR TurnTowardsPlayer8

    ; If the facing direction has no horizontal component, then
    ; find the horizontal direction toward the chase target.
    ;
    ; Then combine it with the facing direction.
    LDA ObjDir, X
    AND #$03
    BNE @SetStateJump
    LDY #$02
    LDA ChaseTargetX
    CMP ObjX, X
    BCC :+
    DEY
:
    TYA
    ORA ObjDir, X
    STA ObjDir, X

@SetStateJump:
    ; Set state 1 to jump.
    INC ObjState, X

@SetUpJump:
    ; If reversal count >= 2, then
    ; 1. invert the horizontal component of facing direction,
    ;    to face away from the chase target
    ; 2. reset the reversal count to 0.
    ;
    ; This prevents a jumper from getting stuck in a corner.
    ;
    ; For example, if a tektite's direction is down and right,
    ; and jumps into the top right corner of the room:
    ; 1. The right room boundary is hit. So the direction is
    ;    automatically reversed to up and left.
    ; 2. The next frame, the top room boundary is hit.
    ;    The direction is reversed to down and right.
    ;
    ; As you can see, an endless loop has been made.
    LDA Jumper_ObjReversalCount, X
    CMP #$02
    BCC @FinishSetUpJump
    LDA ObjDir, X
    EOR #$03
    STA ObjDir, X
    LDA #$00
    STA Jumper_ObjReversalCount, X

@FinishSetUpJump:
    ; If the object is a boulder, then make sure its facing direction
    ; has a down vertical component.
    JSR Jumper_PointBoulderDownward

    ; For the given direction, get the Y offset of the destination
    ; of the jump.
    ;
    ; Add it to the object's Y, and store this destination Y.
    LDY ObjDir, X
    LDA ObjY, X
    CLC
    ADC JumperYOffsets, Y
    STA Jumper_ObjTargetY, X

    ; Set the starting speed, depending on the object type.
    JSR Jumper_GetKind
    LDA JumperStartSpeedsHi, Y
    STA Jumper_ObjSpeedWholeY, X

    ; Reset the vertical speed low byte.
    ; Then go animate, draw, and check collisions.
    JSR Jumper_ResetVSpeedFrac

@Animate:
    JMP Jumper_AnimateAndCheckCollisions

@State1:
    ; State 1. Jumping.
    ;
    ; Keep the object inside the bounds of the room.
    JSR BoundFlyer

    ; If the object was blocked at the room boundary, then
    ; increase boundary reversal count, and go set up another jump.
    LDA $0F
    BNE :+
    INC Jumper_ObjReversalCount, X
    JMP @SetUpJump
:
    JSR Jumper_PointBoulderDownward

    ; This step in the jump succeeded. So, reset the reversal count.
    LDA #$00
    STA Jumper_ObjReversalCount, X

    ; Get the base offset for the kind of jumper the object is.
    JSR Jumper_GetKind
    LDA JumperYAccelerationBaseOffsets, Y

    ; Get the acceleration by indexing with (base offset + direction).
    ;
    ; Note that the set of accelerations of Boulder (the third set)
    ; does not have $B elements like the tektites' sets.
    ;
    ; Boulder does not need the upward accelerations.
    ; So, the elements for those are used for the starting speeds
    ; at 04:88F2.
    CLC
    ADC ObjDir, X
    TAY
    LDA JumperYAccelerations, Y
    LDY #$02                    ; Max speed 2 pixels going down
    JSR Jumper_MoveY

    ; If the facing direction has a left component, then
    ; subtract 1 from the object's X.
    ;
    ; Else add 1.
    LDY #$FF
    LDA ObjDir, X
    AND #$02
    BNE :+
    LDY #$01
:
    TYA
    CLC
    ADC ObjX, X
    STA ObjX, X

    ; If going up (negative v-speed high byte), then
    ; go animate, draw, and check collisions.
    LDA Jumper_ObjSpeedWholeY, X
    BMI Jumper_AnimateAndCheckCollisions

    ; Going down. We have to consider the target Y.
    ;
    ; If the absolute vertical distance to the target position >= 3,
    ; then go animate, draw, and check collisions.
    LDA ObjY, X
    SEC
    SBC Jumper_ObjTargetY, X
    JSR Abs
    CMP #$03
    BCS Jumper_AnimateAndCheckCollisions

    ; Distance < 3. Go to state 0.
    ;
    ; If the object is a boulder, then go reset timer, animate, draw.
    JSR ResetObjState
    LDY ObjType, X
    CPY #$20
    BEQ @SetTimerAndAnimate

    ; Else calculate a timer value.
    ;
    ; t := (Random + $10)
    ; if t < $20 then
    ;   t := t - $40
    LDA #$10
    CLC
    ADC Random+1, X
    CMP #$20
    BCS :+
    SEC
    SBC #$40
:
    ; If the object is a blue tektite, then go use this t.
    LDY ObjType, X
    CPY #$0D
    BEQ @SetTimerAndAnimate

    ; t := t AND $7F
    AND #$7F

    ; If Random < $A0, then go use this t.
    LDY Random+1, X
    CPY #$A0
    BCC @SetTimerAndAnimate

    ; t := t AND $F
    AND #$0F

@SetTimerAndAnimate:
    STA ObjTimer, X

Jumper_AnimateAndCheckCollisions:
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; UNKNOWN:
    ; [0D] doesn't seem to be used. The next place that it's read is in
    ; DrawObjectWithAnim, but after overwriting it.
    STA $0D

    ; If the monster is a boulder, then go animate, draw, check collision.
    LDA ObjType, X
    CMP #$20
    BEQ @DrawBoulder

    ; Draw a tektite.
    ;
    ; If state = 1, then go draw frame image 1, and check collisions.
    LDA ObjState, X
    BNE @DrawTektite

    ; If object timer < $21, go draw frame image 0, and check collisions.
    LDY ObjTimer, X
    CPY #$21
    BCC @DrawTektite

    ; Else cycle the animation counter, draw, and check collisions.
    ;
    ; The movement frame determines the frame image.
    LDA #$10
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor
    LDA ObjAnimFrame, X

@DrawTektite:
    JMP DrawObjectMirroredAndCheckCollisions

@DrawBoulder:
    ; Draw a boulder.
    LDA #$06
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor
    LDA ObjAnimFrame, X         ; The animation frame determines the frame image.
    JSR DrawObjectNotMirroredAndCheckLinkCollision

    ; If the boulder's Y < $F0, return.
    LDA ObjY, X
    CMP #$F0
    BCC L10A10_Exit

    ; Else the boulder is close enough to the bottom of the screen
    ; to destroy it.
    DEC ActiveBoulderCount
    JMP DestroyMonster

; Returns:
; Y: 0: blue tektite
;    1: red tektite
;    2: boulder
;
Jumper_GetKind:
    LDA ObjType, X
    SEC
    SBC #$0D
    TAY
    CPY #$02
    BCC L10A10_Exit
    LDY #$02

L10A10_Exit:
    RTS

Jumper_PointBoulderDownward:
    LDA ObjType, X
    CMP #$20
    BNE @Exit
    LDA ObjDir, X
    AND #$03
    ORA #$04
    STA ObjDir, X

@Exit:
    RTS

; Params:
; A: acceleration
; Y: max speed high byte
;
Jumper_MoveY:
    STA $00                     ; [00] holds the acceleration
    STY $02                     ; [02] holds the max speed high byte

    ; Add the high speed byte to the Y coordinate.
    LDA Jumper_ObjSpeedWholeY, X
    ADC ObjY, X
    STA ObjY, X

    ; Add the acceleration to the speed fraction (low byte).
    LDA Jumper_ObjSpeedFracY, X
    CLC
    ADC $00                     ; [00] acceleration
    STA Jumper_ObjSpeedFracY, X

    ; Carry over to the high speed byte.
    LDA Jumper_ObjSpeedWholeY, X
    ADC #$00
    STA Jumper_ObjSpeedWholeY, X

    ; *Signed comparison*:
    ; If the high speed byte < high max speed byte, then return.
    ;
    ; Note that this is not a general purpose signed comparison,
    ; which would be more complicated. But it works because
    ; the deviation from 0 of both values is small.
    CMP $02
    BMI L10A52_Exit

    ; Limit the speed to the max speed.
    LDA Jumper_ObjSpeedFracY, X
    CMP #$80
    BCC L10A52_Exit
    LDA $02
    STA Jumper_ObjSpeedWholeY, X

Jumper_ResetVSpeedFrac:
    LDA #$00
    STA Jumper_ObjSpeedFracY, X

L10A52_Exit:
    RTS
