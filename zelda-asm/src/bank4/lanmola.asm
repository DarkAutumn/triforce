; ============================================================================
; FILE:     bank4/lanmola.asm            BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 9671-10027
; SUBJECTS: (none)
; TOPIC:    Lanmola segments and head
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0015 UpdateLamnola       code
;   L0123 L12A6F_Exit         code
;   L0126 Lamnola_UpdateHead  code
;   L0320 Lamnola_Move        code
; ============================================================================

UpdateLamnola:
    ; If the direction of this segment is not set, then it has not started
    ; moving yet. So, return.
    LDA ObjDir, X
    BEQ L129EA_Exit

    ; If we have the magic clock, then go draw and check collisions.
    LDA InvClock
    BNE @DrawAndCheckCollisions
    JSR Lamnola_Move

    ; If the current segment is a head, then update the direction it's facing.
    CPX #$05
    BEQ :+
    CPX #$0A
    BNE @DrawAndCheckCollisions
:
    JSR Lamnola_UpdateHead

@DrawAndCheckCollisions:
    LDA ObjX, X
    PHA                         ; Save the X coordinate.

    ; Each segment is a full 16x16 object. But, it's drawn as a narrow,
    ; half-width object. So, center it by temporarily adding 4 to X.
    CLC
    ADC #$04
    STA ObjX, X

    ; The speed is 1 or 2. By swapping them, you get sprite attributes
    ; to store in [03].
    LDA Lanmola_Speed
    EOR #$03
    STA $03

    ; If the segment is a head, then use tile $9E, else $A0.
    LDA #$9E
    CPX #$05
    BEQ @Draw
    CPX #$0A
    BEQ @Draw
    LDA #$A0

@Draw:
    JSR Anim_WriteSprite
    PLA                         ; Restore the X coordinate.
    STA ObjX, X
    LDA ObjDir, X               ; Save the facing direction.
    PHA
    JSR CheckMonsterCollisions
    PLA                         ; Restore the facing direction.
    STA ObjDir, X

    ; If still alive, then return.
    LDA ObjMetastate, X
    BEQ L129EA_Exit

    ; Otherwise, the segment is dead.
    JSR ResetShoveInfo

    ; Set the current segment's HP to the initial value in anticipation
    ; of bringing this segment back to life.
    LDA #$20
    STA ObjHP, X

    ; Find the tail of the lamnola that the current segment belongs to.
    ;
    ; From the lowest slot, look for one that has the original lamnola
    ; object type.
    ;
    ; The goal is to swap the current dead segment with the tail segment.
    LDY #$FF
    CPX #$06
    BCC @FindTail
    LDY #$04

@FindTail:
    INY
    LDA ObjType+1, Y
    CMP Lanmola_Type
    BNE @FindTail

    ; Set the timer for the dead dummy object that will replace the tail.
    LDA #$11
    STA a:ObjTimer+1, Y

    ; Copy the current dead segment's invincibility timer, X, and Y
    ; to the tail segment.
    LDA ObjInvincibilityTimer, X
    STA ObjInvincibilityTimer+1, Y
    LDA ObjX, X
    STA a:ObjX+1, Y
    LDA ObjY, X
    STA a:ObjY+1, Y

    ; If the tail segment found is a head, then the whole lamnola died.
    ; So, return and leave it dead.
    CPY #$04
    BEQ L129EA_Exit
    CPY #$09
    BEQ L129EA_Exit

    ; Change the tail segment to the dead dummy object, and
    ; bring the current segment back to life.
    LDA #$5D
    STA ObjType+1, Y
    JMP ResetObjMetastate

L12A6F_Exit:
    RTS

Lamnola_UpdateHead:
    ; If X coordinate is not a multiple of 8, then return.
    LDA ObjX, X
    AND #$07
    BNE L12A6F_Exit

    ; If (Y + 3) is not a multiple of 8, then return.
    ; We had to account for the usual offset of 3.
    LDA ObjY, X
    CLC
    ADC #$03
    AND #$07
    BNE L12A6F_Exit

    ; Will loop 4 times, propagating directions down the chain,
    ; by pulling from the bottom.
    LDA #$04
    STA $00

    ; Choose the tail index of the lamnola that the current segment belongs to.
    LDY #$00
    CPX #$05
    BEQ @PropagateDirs
    LDY #$05

@PropagateDirs:
    ; Loop over every segment under the head, starting from the tail.
    ; Copy the next segment's direction to the current one in this loop.
    LDA a:ObjDir+2, Y
    STA a:ObjDir+1, Y
    INY
    DEC $00
    BNE @PropagateDirs

    ; If this head segment is not aligned with a square, then return.
    LDA ObjX, X
    AND #$0F
    BNE @Exit

    ; If (Y + 3) is not a multiple of $10, then return.
    ; We had to account for the usual offset of 3.
    LDA ObjY, X
    CLC
    ADC #$03
    AND #$0F
    BNE @Exit

    ; Store the opposite of the facing direction in [00].
    LDA ObjDir, X
    LSR
    AND #$05
    STA $00
    LDA ObjDir, X
    ASL
    AND #$0A
    ORA $00

    ; Make a mask of the inverted bits of the opposite direction.
    EOR #$0F
    STA LanmolaViableDirMask

    ; There's a 50% chance of going to try to turn toward Link.
    LDA Random, X
    CMP #$80
    BCC @TurnTowardLink

    ; Otherwise, we randomly choose a perpendicular direction.
    ; Load the monster's facing direction.
    LDA ObjDir, X

    ; Load the next random number into Y register.
    LDY Random+1, X

    ; Based on it, there's a 50% chance of skipping the loop,
    ; to keep moving straight.
    CPY #$80
    BCS @SetDirAndCheckTiles

@LoopFindPerpendicular:
    ; Loop to rotate the direction bit over the mask.
    ; The goal is to calculate a direction perpendicular to the monster's
    ; facing direction.
    ;
    ; Shift right the copy of the direction.
    ; If the direction bit rolled off the low end, then roll it onto the high end.
    LSR
    BCC :+
    LDA #$08
:
    ; If the current direction bit is masked off, then we've reached
    ; the opposite direction. Loop again.
    BIT LanmolaViableDirMask
    BEQ @LoopFindPerpendicular

    ; If Y register >= $40, then break out of the loop.
    ; So, 75% of the time, if horizontal, then turn up, else left.
    CPY #$40
    BCS @SetDirAndCheckTiles

    ; Set Y register to $40 in order to break out of the loop next time;
    ; and loop again now.
    ; So, 25% of the time, if horizontal, then turn down, else right.
    LDY #$40
    BCC @LoopFindPerpendicular

@SetDirAndCheckTiles:
    ; Change the facing direction to the direction found.
    STA ObjDir, X

    ; Set moving direction [0F] for boundary and tile checks.
    STA $0F
    JSR BoundByRoom

    ; If not blocked by the room boundary, then go check tile collision.
    LDA $0F
    BNE @CheckTileCollision

@FindNonOpposite:
    ; Else it was blocked. So, load the facing direction in preparation
    ; for looking for any non-opposite direction that is not blocked.
    LDA ObjDir, X

@LoopFindNonOpposite:
    ; Shift right the copy of the direction.
    ; If the direction bit rolled off the low end, then roll it onto the high end.
    LSR
    BCC :+
    LDA #$08
:
    ; If the current direction bit is not masked off, then go check the
    ; room boundary in this direction.
    ; Else we've reached the opposite direction. Loop again.
    BIT LanmolaViableDirMask
    BNE @SetDirAndCheckTiles
    BEQ @LoopFindNonOpposite

@CheckTileCollision:
    ; We found a direction that was not blocked by the room boundary.
    ; Check tile collision. If blocked by a tile, then go look for another direction.
    JSR GetCollidingTileMoving
    CMP RoomFirstUnwalkableTile
    BCS @FindNonOpposite

@Exit:
    ; If there were a room that had lamnolas, and they could get
    ; surrounded on 3 sides, then the loop above would get stuck in
    ; an infinite loop.
    ;
    ; But, the only room with that configuration has those blocks
    ; blocked off with a push block, which can only be pushed after
    ; all monsters are killed.
    ;
    ; I confirmed this by using the memory view to move the lamnolas
    ; in room $57 of Q1/L9 over the stairs in the middle of the room.
    RTS

@TurnTowardLink:
    ; Find the horizontal direction toward Link and store it in [02].
    LDA #$01
    STA $02
    LDA ObjX
    SEC
    SBC ObjX, X
    BCS :+
    ASL $02
:
    ; Find the vertical direction toward Link and store it in [03].
    LDA #$04
    STA $03
    LDA ObjY
    SEC
    SBC ObjY, X
    BCS :+
    ASL $03
:
    ; If the horizontal direction toward Link is not allowed, then
    ; go check the vertical direction.
    LDA $02
    BIT LanmolaViableDirMask
    BEQ @SetVertical

    ; If the horizontal direction toward Link is the same direction
    ; that Link is facing, then go test it for tile and room boundary collision.
    BIT ObjDir
    BNE :+

@SetVertical:
    ; Load the vertical direction toward Link in order to test it for
    ; room boundary and tile collision.
    LDA $03
:
    ; Go test this direction for collisions with tiles or the room boundary.
    JMP @SetDirAndCheckTiles

Lamnola_Move:
    ; This code is similar to 04:96EB Digdogger_Move.
    ;
    ; Change coordinates by the appropriate speed according to the direction.
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
    ADC Lanmola_Speed
    STA ObjX, X

@Left:
    ; If direction has a left component (2), then subtract offset from X coordinate.
    LDA ObjDir, X
    ASL $02                     ; Bit 7 of mask $A1 sets Carry now.
    BIT $02
    BEQ @Down
    LDA ObjX, X
    SBC Lanmola_Speed
    STA ObjX, X

@Down:
    ; If direction has a down component (4), then add offset to Y coordinate.
    LDA ObjDir, X
    ASL $02                     ; Bit 6 of mask $A1 clears Carry now.
    BIT $02
    BEQ @Up
    LDA ObjY, X
    ADC Lanmola_Speed
    STA ObjY, X

@Up:
    ; If direction has an up component (8), then subtract offset from Y coordinate.
    LDA ObjDir, X
    ASL $02                     ; Bit 5 of mask $A1 sets Carry now.
    BIT $02
    BEQ @Exit
    LDA ObjY, X
    SBC Lanmola_Speed
    STA ObjY, X

@Exit:
    RTS
