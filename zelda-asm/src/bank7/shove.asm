; ============================================================================
; FILE:     bank7/shove.asm              BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 2323-2494
; SUBJECTS: (none)
; TOPIC:    Knockback movement
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0015 Obj_Shove          code
;   L0075 ResetShoveInfo     code  Returns:
;   L0084 SetShoveInfoWith0  code  Params:
;   L0089 ShoveMoveMin       code
; ============================================================================

Obj_Shove:
    ; If this is not the first call to this routine for this instance of shoving (high bit is clear),
    ; then go handle it separately.
    LDA ObjShoveDir, X
    ASL
    BCC @MoveIfNotDone

    ; Clear the high bit, so we don't repeat this initialization.
    LSR
    STA ObjShoveDir, X

    ; If the object faces horizontally, go check which axis shove
    ; direction is on.
    LDY ObjDir, X
    CPY #$03
    BCC @FacingHorizontally

    ; The object faces vertically.
    ;
    ; If the shove direction is vertical, return. We're OK to shove.
    AND #$03
    BEQ @Exit

@CheckPerpendicularShove:
    ; Allow a perpendicular shove, if grid offset = 0.
    LDA ObjGridOffset, X
    BEQ @Exit

    ; Else change the shove direction.
    ;
    ; If the object is not Link, then reset it.
    CPX #$00
    BNE ResetShoveInfo

    ; Since it's Link, shove in the opposite direction that he's facing.
    ; Begin unverified code 1EED3
    LDA ObjDir
    JSR GetOppositeDir
    STA ObjShoveDir
    ; End unverified code

@Exit:
    RTS

@FacingHorizontally:
    ; If shove direction is horizontal, return. We're OK to shove.
    ; Else go check the object's grid offset.
    AND #$0C
    BNE @CheckPerpendicularShove
    RTS

@MoveIfNotDone:
    ; If shove distance hasn't gone down to 0, go move some more.
    ; Else reset shove info.
    LDA ObjShoveDistance, X
    BNE ShoveMoveMin

; Returns:
; A: 0
;
ResetShoveInfo:
    LDA #$00

; Params:
; A: 0
;
; Returns:
; A: 0
;
SetShoveInfoWith0:
    STA ObjShoveDir, X
    STA ObjShoveDistance, X
    RTS

ShoveMoveMin:
    ; Try to move 4 pixels. [03] is the counter.
    LDA #$04
    STA $03

@LoopShovePixel:
    ; If the object's grid offset = 0, make sure it's aligned to the grid.
    ; Then, stop shoving if the object hits a tile.
    LDA ObjGridOffset, X
    BNE @CheckBoundary
    JSR EnsureObjectAligned
    LDA ObjShoveDir, X
    AND #$0F
    STA $0F
    JSR GetCollidingTileMoving
    CMP ObjectFirstUnwalkableTile
    BCS ResetShoveInfo

@CheckBoundary:
    ; Regardless of the grid offset, if the object runs into the
    ; bounds of the room, then stop shoving.
    LDA ObjShoveDir, X
    AND #$0F
    JSR BoundByRoomWithA
    BEQ ResetShoveInfo

    ; If there is a grumble moblin or person in the room (regardless
    ; of which object is being shoved), then see if it's blocked by
    ; the person. Stop shoving if it is blocked.
    LDA ObjType+1
    CMP #$36
    BEQ @CheckBlocked
    CMP #$4B
    BCC @ChooseSpeed
    CMP #$53
    BCS @ChooseSpeed

@CheckBlocked:
    JSR CheckPersonBlocking
    LDA $0F
    BEQ ResetShoveInfo

@ChooseSpeed:
    ; If the direction is right or down, store 1 in [02], else -1.
    ; This is the amount to change the relevant coordinate by.
    LDY #$01
    LDA ObjShoveDir, X
    AND #$05
    BNE :+
    LDY #$FF
:
    STY $02

    ; We're OK to change the position. So, decrease the distance
    ; that's left to move.
    DEC ObjShoveDistance, X

    ; Change the grid offset by the amount in [02] (1, -1).
    LDA ObjGridOffset, X
    CLC
    ADC $02
    STA ObjGridOffset, X

    ; If grid offset is a multiple of $10, or the object is Link and grid offset
    ; is a multiple of 8, then reset grid offset.
    AND #$0F
    BEQ @SetGridOffset
    CPX #$00
    BNE @ApplySpeed
    AND #$07
    BNE @ApplySpeed

@SetGridOffset:
    STA ObjGridOffset, X

@ApplySpeed:
    ; If the direction is horizontal, add the amount in [02] to X.
    LDA ObjShoveDir, X
    AND #$03
    BEQ @AddToY
    LDA ObjX, X
    CLC
    ADC $02
    STA ObjX, X
    JMP @NextShovePixel

@AddToY:
    ; Else add the amount to Y coordinate.
    LDA ObjY, X
    CLC
    ADC $02
    STA ObjY, X

@NextShovePixel:
    ; Loop again if [03] is not going down to 0.
    DEC $03
    BNE @LoopShovePixel
    RTS
