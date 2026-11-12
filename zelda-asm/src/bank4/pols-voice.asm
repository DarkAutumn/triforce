; ============================================================================
; FILE:     bank4/pols-voice.asm         BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 6516-6814
; SUBJECTS: PolsVoice
; TOPIC:    Pols Voice walk/jump
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0022 PolsVoiceWalkSpeedsX                data
;   L0026 PolsVoiceWalkSpeedsY                data
;   L0030 PolsVoiceInitialJumpSpeeds          data
;   L0033 PolsVoiceDestinationYOffsets        data
;   L0036 PolsVoiceDirections                 data
;   L0039 UpdatePolsVoice                     code
;   L0162 L_PolsVoice_DrawAndCheckCollisions  code
;   L0177 UpdatePolsVoiceState1_Jumping       code
;   L0255 PolsVoice_IsSquareWalkable          code  Returns:
;   L0296 PolsVoice_GetCollidingTile          code  Returns:
;   L0313 PolsVoice_MoveX                     code  Returns:
; ============================================================================

PolsVoiceWalkSpeedsX:
    .BYTE $01, $FF, $00, $00, $01, $FF, $00, $00
    .BYTE $01, $FF

PolsVoiceWalkSpeedsY:
    .BYTE $00, $00, $00, $01, $01, $01, $00, $FF
    .BYTE $FF, $FF

PolsVoiceInitialJumpSpeeds:
    .BYTE $FD, $FD, $FF, $FF, $FF, $FF, $FF, $FC

PolsVoiceDestinationYOffsets:
    .BYTE $00, $00, $20, $20, $20, $20, $20, $E0

PolsVoiceDirections:
    .BYTE $01, $02, $04, $08

UpdatePolsVoice:
    CPX #$01
    BNE :+
:
    ; If we have the magic clock, or is stunned, then
    ; go draw and check collisions.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE @DrawAndCheckCollisions

    ; Every odd screen frame, go straight to drawing
    ; and checking collisions.
    LDA FrameCounter
    LSR
    BCS L_PolsVoice_DrawAndCheckCollisions

    ; Otherwise move horizontally.
    JSR PolsVoice_MoveX

    ; If state <> 0, then the monster is jumping.
    ; Go handle vertical movement.
    LDA ObjState, X
    BEQ @State0
    JSR UpdatePolsVoiceState1_Jumping
    JMP @CheckWalkability

@State0:
    ; State 0. Walking.
    ;
    ; If grid offset = 0, then no more distance to cover.
    ; So, go change to state 1 -- jumping.
    LDA ObjRemDistance, X
    BEQ @SetState1

    ; Decrement grid offset / distance remaining.
    DEC ObjRemDistance, X

    ; Move vertically.
    ; Add the Y offset for the current direction to the Y coordinate.
    LDA ObjY, X
    CLC
    ADC PolsVoiceWalkSpeedsY, Y
    STA ObjY, X

@CheckWalkability:
    ; If the square the monster is over is walkable, then
    ; go draw and check object collisions.
    JSR PolsVoice_IsSquareWalkable
    BCC @DrawAndCheckCollisions

    ; If the unwalkable tile is a block, or >= $F4 (water, screen edge bricks),
    ; then go change to state 1.
    LDA PolsVoice_ObjLastTile, X
    AND #$FC
    CMP #$B0
    BEQ @SetState1
    CMP #$F4
    BCS @SetState1

    ; Else blocked by something else.
    ;
    ; If facing horizontally, then go flip the direction, move, and draw.
    LDA ObjDir, X
    AND #$03
    BNE @FlipHorizontally

    ; Else facing vertically. Flip the direction.
    ; Go draw.
    LDA ObjDir, X
    EOR #$0C
    STA ObjDir, X
    JMP L_PolsVoice_DrawAndCheckCollisions

@FlipHorizontally:
    ; Facing horizontally. Flip the direction.
    ; Move two pixels, and go draw.
    EOR #$03
    STA ObjDir, X
    JSR PolsVoice_MoveX
    JSR PolsVoice_MoveX

@DrawAndCheckCollisions:
    JMP L_PolsVoice_DrawAndCheckCollisions

@SetState1:
    ; If already in state 1, then go draw and check collisions.
    LDA ObjState, X
    BNE L_PolsVoice_DrawAndCheckCollisions

    ; Go to state 1.
    INC ObjState, X

    ; Subtract 1 from facing direction to make an index.
    LDY ObjDir, X
    DEY

    ; If Y coordinate < $78, make the index 3 for down.
    LDA ObjY, X
    CMP #$78
    BCS :+
    LDY #$03
:
    ; If Y coordinate >= $A8, make the index 7 for up.
    CMP #$A8
    BCC :+
    LDY #$07
:
    ; Set the vertical speed for the beginning of the jump.
    LDA PolsVoiceInitialJumpSpeeds, Y
    STA PolsVoice_ObjSpeedWhole, X

    ; Look up the Y offset for the current direction.
    ; Add it to object Y to set the destination Y of the jump.
    LDA ObjY, X
    CLC
    ADC PolsVoiceDestinationYOffsets, Y
    STA PolsVoice_ObjTargetY, X

    ; Turn the index into a direction again by adding 1.
    ; Store the facing direction.
    INY
    STY ObjDir, X

L_PolsVoice_DrawAndCheckCollisions:
    LDA #$08                    ; Animation counter 8
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor
    LDA ObjAnimFrame, X
    JSR DrawObjectMirrored

    ; Pols Voice is invincible to everything but the sword,
    ; during normal collision detection.
    ;
    ; The arrow is a special case.
    LDA #$FE
    STA ObjInvincibilityMask, X
    JSR CheckMonsterCollisions
    RTS

UpdatePolsVoiceState1_Jumping:
    ; Add acceleration $38 to low speed byte (vertical).
    LDA PolsVoice_ObjSpeedFrac, X
    CLC
    ADC #$38
    STA PolsVoice_ObjSpeedFrac, X

    ; Add Carry to high speed byte (vertical).
    LDA PolsVoice_ObjSpeedWhole, X
    ADC #$00
    STA PolsVoice_ObjSpeedWhole, X

    ; Add high speed byte to Y coordinate.
    CLC
    ADC ObjY, X
    STA ObjY, X

    ; If the speed is still negative, then return.
    LDA PolsVoice_ObjSpeedWhole, X
    BMI @Exit

    ; If Y coordinate still < target Y, then return.
    LDA ObjY, X
    CMP PolsVoice_ObjTargetY, X
    BCC @Exit

    ; Else reached the destination. Go to state 0.
    LDA #$00
    STA ObjState, X

    ; Reset 16-bit vertical speed.
    STA PolsVoice_ObjSpeedFrac, X
    STA PolsVoice_ObjSpeedWhole, X

    ; Choose a random direction to face in.
    LDA Random, X
    AND #$03
    TAY
    LDA PolsVoiceDirections, Y
    STA ObjDir, X

    ; Randomly set distance-to-move to $30 or $70.
    LDA Random, X
    AND #$40
    ADC #$30
    STA ObjRemDistance, X

    ; Align the monster with a square horizontally.
    ;
    ; If in the left half of a square, align with that square.
    ; If in the right half, align with the next square.
    LDA ObjX, X
    CLC
    ADC #$08
    AND #$F0
    STA ObjX, X

    ; Align the monster with a square vertically.
    ;
    ; If in the top half of a square, align with that square.
    ; If in the bottom half, align with the next square.
    LDA ObjY, X
    CLC
    ADC #$08
    AND #$F0

    ; Make sure the lower nibble of Y is $D as usual.
    SEC
    SBC #$03
    STA ObjY, X

@Exit:
    RTS

; Returns:
; C: 0 if walkable
; [041F][X]: tile
;
PolsVoice_IsSquareWalkable:
    JSR PolsVoice_GetCollidingTile

    ; If there's an unwalkable tile at the usual hotspot, then return.
    BCS @Exit

    ; Save the original object X, and temporarily add $E to it.
    LDA ObjX, X
    PHA
    CLC
    ADC #$0E
    STA ObjX, X

    ; Save the original Y, and temporarily add 6 to it.
    LDA ObjY, X
    PHA
    CLC
    ADC #$06
    STA ObjY, X

    ; Check again at relative hotspot ($E, $E).
    JSR PolsVoice_GetCollidingTile

    ; Restore the original coordinates.
    PLA
    STA ObjY, X
    PLA
    STA ObjX, X

@Exit:
    RTS

; Returns:
; C: 0 if walkable
; [041F][X]: tile
;
; Use offset 0 from the hotspot.
;
; It looks like [0F] is still 0 from when ObjUninitialized was copied
; to it inside UpdateObject.
;
PolsVoice_GetCollidingTile:
    LDY #$00
    JSR GetCollidableTile
    LDA ObjCollidedTile, X

    ; Return C=0 if walkable.
    CMP RoomFirstUnwalkableTile

    ; Remember the tile that the monster is over.
    STA PolsVoice_ObjLastTile, X
    RTS

; Returns:
; Y: facing direction - 1 (so, 0 to 7)
;
; Add the X offset for the current direction to the X coordinate.
;
PolsVoice_MoveX:
    LDY ObjDir, X
    DEY
    LDA ObjX, X
    CLC
    ADC PolsVoiceWalkSpeedsX, Y
    STA ObjX, X
    RTS
