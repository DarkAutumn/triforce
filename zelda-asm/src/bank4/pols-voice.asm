; ============================================================================
; FILE:     bank4/pols-voice.asm         BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 6516-6814
; SUBJECTS: PolsVoice
; TOPIC:    Pols Voice (object type $16): its update routine, the walk and jump tables, the jump physics
;           and its own two-point tile test. It moves only on even frames and dies to any arrow.
; SEE ALSO: bank1/sword-bomb-arrow-collision.asm (CheckMonsterShotCollision: an arrow sets its HP to 0);
;           bank1/monster-collisions.asm (CheckMonsterCollisions, ObjTypeToDamagePoints);
;           bank7/object-update-table.asm (attributes, HP pairs, update table);
;           bank7/object-init-table.asm (type $16 is initialized by InitWalker);
;           bank7/tile-collision.asm (GetCollidableTile); bank4/drop-items.asm (drop group lists).
; KEY FACTS:
;   - Invincibility mask $FE is set every frame: only damage type 1 (sword, sword shot, rod swing) goes
;     through the normal weapon tests; boomerang, bomb, fire and magic shot do nothing
;     (L_PolsVoice_DrawAndCheckCollisions).
;   - An arrow is the exception: on contact CheckMonsterShotCollision (bank 1) sets ObjHP to 0 and deals
;     the damage, so any arrow kills it; the arrow is not stopped and flies on.
;   - HP $A0 (nibble $A in ObjectTypeToHpPairs): 10 wooden, 5 white or 3 magic sword hits.
;   - It moves only on frames where FrameCounter is even, and not at all while InvClock is set or it is
;     stunned (UpdatePolsVoice). 1 pixel per moving frame = 0.5 pixels per frame.
;   - State 0 walks $31 or $71 moving frames (98 or 226 frames) in a random direction, then jumps
;     (UpdatePolsVoiceState1_Jumping sets the distance; the carry makes it $31/$71, not $30/$70).
;   - Jump: start speed -3 (sideways), -1 (down) or -4 (up) pixels per moving frame, gravity $38/256 per
;     moving frame; lands level, $20 pixels lower or $20 pixels higher (PolsVoiceInitialJumpSpeeds,
;     PolsVoiceDestinationYOffsets). Above Y $78 it always jumps down, at Y $A8 or below always up.
;   - Tiles $B0-$B3 (block) and $F4 and above start a jump instead of turning it around; any other
;     unwalkable tile reverses its direction (UpdatePolsVoice@CheckWalkability).
;   - It never calls Walker_Move or Obj_Shove: no grid movement, and a hit does not knock it back.
;     A sword hit still starts its invincibility timer ($10, BeginShove in bank 1); while that runs
;     CheckMonsterCollisions tests no weapon, so an arrow passes through it without killing it.
;   - Its walk distance is 0 when the room is entered (the slot arrays are zeroed), so its first
;     moving frame starts a jump (UpdatePolsVoice@State0).
; CONTENTS: (generated; L = line in this file)
;   L0048 PolsVoiceWalkSpeedsX                data  TABLE: Pols Voice X step; index = ObjDir - 1 (0-9); value = signed pixels per moving frame.
;   L0054 PolsVoiceWalkSpeedsY                data  TABLE: Pols Voice Y step while walking (state 0); index = ObjDir - 1 (0-9); value = signed pixels per
;   L0060 PolsVoiceInitialJumpSpeeds          data  TABLE: Pols Voice jump start speed; index = jump direction - 1 (0-7); value = signed pixels per moving
;   L0065 PolsVoiceDestinationYOffsets        data  TABLE: Pols Voice jump landing height; index = jump direction - 1 (0-7); value = signed pixels added
;   L0069 PolsVoiceDirections                 data  TABLE: direction chosen after a jump; index = Random,X AND 3; value = direction bits.
;   L0082 UpdatePolsVoice                     code  Updates a Pols Voice: walks or jumps on even frames, turns at walls, draws it and checks collisions.
;   L0196 L_PolsVoice_DrawAndCheckCollisions  code  Shared tail of UpdatePolsVoice: animates and draws the Pols Voice, sets its mask, checks collisions.
;   L0219 UpdatePolsVoiceState1_Jumping       code  Moves a jumping Pols Voice vertically and, when it lands, picks its next direction and walk distance.
;   L0295 PolsVoice_IsSquareWalkable          code  Tests the tiles under a Pols Voice at two points; C set if either one is unwalkable.
;   L0335 PolsVoice_GetCollidingTile          code  Reads the tile at (ObjX, ObjY+$0B) and compares it with the first unwalkable tile.
;   L0352 PolsVoice_MoveX                     code  Adds the X step for the Pols Voice's facing direction to its X coordinate.
; ============================================================================

; TABLE: Pols Voice X step; index = ObjDir - 1 (0-9); value = signed pixels per moving frame.
PolsVoiceWalkSpeedsX:
    .BYTE $01, $FF, $00, $00, $01, $FF, $00, $00   ; dir 1 right +1, 2 left -1, 3 -, 4 down 0, 5 +1, 6 -1,
    .BYTE $01, $FF                                 ; 7 -, 8 up 0, 9 +1, $A -1 (5, 6, 9, $A = diagonals).

; TABLE: Pols Voice Y step while walking (state 0); index = ObjDir - 1 (0-9); value = signed pixels per
;        moving frame. Not used during a jump.
PolsVoiceWalkSpeedsY:
    .BYTE $00, $00, $00, $01, $01, $01, $00, $FF   ; dir 1 right 0, 2 left 0, 3 -, 4 down +1, 5 +1, 6 +1,
    .BYTE $FF, $FF                                 ; 7 -, 8 up -1, 9 -1, $A -1.

; TABLE: Pols Voice jump start speed; index = jump direction - 1 (0-7); value = signed pixels per moving
;        frame (whole part of the vertical speed; negative = up).
PolsVoiceInitialJumpSpeeds:
    .BYTE $FD, $FD, $FF, $FF, $FF, $FF, $FF, $FC   ; right -3, left -3, -, down -1, -, -, -, up -4

; TABLE: Pols Voice jump landing height; index = jump direction - 1 (0-7); value = signed pixels added
;        to the Y at take-off to give PolsVoice_ObjTargetY.
PolsVoiceDestinationYOffsets:
    .BYTE $00, $00, $20, $20, $20, $20, $20, $E0   ; right 0, left 0, -, down +$20, -, -, -, up -$20

; TABLE: direction chosen after a jump; index = Random,X AND 3; value = direction bits.
PolsVoiceDirections:
    .BYTE $01, $02, $04, $08                       ; right, left, down, up

; ----------------------------------------------------------------------------
; Updates a Pols Voice: walks or jumps on even frames, turns at walls, draws it and checks collisions.
; IN:   X = object slot.
; OUT:  writes ObjX,X, ObjY,X, ObjDir,X, ObjState,X (0 walking, 1 jumping), ObjRemDistance,X.
; NOTE: It moves only when FrameCounter is even, and not while InvClock is set or it is stunned. It
;       does not use Walker_Move or Obj_Shove, so it ignores the grid and is never knocked back.
;       The walkability test runs during a jump too: a wall reverses it in mid-air, a block (tiles
;       $B0-$B3) or a tile >= $F4 does not. ObjRemDistance is 0 after room entry, so a new Pols
;       Voice jumps on its first moving frame.
; ----------------------------------------------------------------------------
UpdatePolsVoice:
    ; Leftover test of slot 1: both outcomes continue at the same place.
    CPX #$01
    BNE :+
:
    ; If Link has the clock or this monster is stunned, do not move: only draw and check collisions.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE @DrawAndCheckCollisions

    ; On odd frames do not move either.
    LDA FrameCounter
    LSR
    BCS L_PolsVoice_DrawAndCheckCollisions

    ; Move horizontally by the X step of the facing direction (0 when facing up or down).
    ; Leaves Y = ObjDir - 1 for the table lookup below.
    JSR PolsVoice_MoveX

    ; State 1 = jumping: the jump code moves it vertically.
    LDA ObjState, X
    BEQ @State0
    JSR UpdatePolsVoiceState1_Jumping
    JMP @CheckWalkability

@State0:
    ; State 0, walking. When the distance left reaches 0, start a jump.
    LDA ObjRemDistance, X
    BEQ @SetState1

    ; One moving frame less to walk.
    DEC ObjRemDistance, X

    ; Move vertically by the Y step of the facing direction (Y = ObjDir - 1 from PolsVoice_MoveX).
    LDA ObjY, X
    CLC
    ADC PolsVoiceWalkSpeedsY, Y
    STA ObjY, X

@CheckWalkability:
    ; C clear = both test points are over walkable tiles: nothing more to do.
    JSR PolsVoice_IsSquareWalkable
    BCC @DrawAndCheckCollisions

    ; Unwalkable. Tiles $B0-$B3 (the block) and tiles >= $F4 do not turn it around: they start a jump
    ; if it is walking, and are ignored if it is already jumping. $B0 and $F4 are the first tiles of
    ; dungeon floor squares 0 (block) and 6 (PrimarySquaresUW); that square 6 is water is (unverified).
    LDA PolsVoice_ObjLastTile, X
    AND #$FC
    CMP #$B0
    BEQ @SetState1
    CMP #$F4
    BCS @SetState1

    ; Blocked by any other tile: reverse the facing direction.
    ; A = left/right bits of the direction; nonzero = facing horizontally.
    LDA ObjDir, X
    AND #$03
    BNE @FlipHorizontally

    ; Facing vertically: swap up and down.
    LDA ObjDir, X
    EOR #$0C
    STA ObjDir, X
    JMP L_PolsVoice_DrawAndCheckCollisions

@FlipHorizontally:
    ; Swap left and right (A = ObjDir AND 3), then step 2 pixels in the new direction.
    EOR #$03
    STA ObjDir, X
    JSR PolsVoice_MoveX
    JSR PolsVoice_MoveX

@DrawAndCheckCollisions:
    JMP L_PolsVoice_DrawAndCheckCollisions

@SetState1:
    ; A jump is already under way: leave it alone.
    LDA ObjState, X
    BNE L_PolsVoice_DrawAndCheckCollisions

    ; Start a jump: state 1.
    INC ObjState, X

    ; Y = jump direction - 1, at first from the facing direction.
    LDY ObjDir, X
    DEY

    ; Above Y $78 the jump always goes down (index 3).
    LDA ObjY, X
    CMP #$78
    BCS :+
    LDY #$03
:
    ; At Y $A8 or below it always goes up (index 7). A still holds ObjY.
    CMP #$A8
    BCC :+
    LDY #$07
:
    ; Start speed of the jump (whole pixels per moving frame, negative = up).
    LDA PolsVoiceInitialJumpSpeeds, Y
    STA PolsVoice_ObjSpeedWhole, X

    ; The jump ends at the take-off Y plus the offset for the jump direction.
    LDA ObjY, X
    CLC
    ADC PolsVoiceDestinationYOffsets, Y
    STA PolsVoice_ObjTargetY, X

    ; The jump direction becomes the facing direction (index + 1).
    INY
    STY ObjDir, X

; Shared tail of UpdatePolsVoice: animates and draws the Pols Voice, sets its mask, checks collisions.
L_PolsVoice_DrawAndCheckCollisions:
    LDA #$08                    ; Animation counter 8
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor
    LDA ObjAnimFrame, X
    JSR DrawObjectMirrored

    ; Mask $FE every frame: only damage type 1 (sword, sword shot, rod swing) can hurt it in the normal
    ; weapon tests. An arrow kills it anyway: the arrow collision code tests for object type $16, sets
    ; ObjHP to 0 and deals the damage without consulting the mask.
    LDA #$FE
    STA ObjInvincibilityMask, X
    JSR CheckMonsterCollisions
    RTS

; ----------------------------------------------------------------------------
; Moves a jumping Pols Voice vertically and, when it lands, picks its next direction and walk distance.
; IN:   X = object slot; called once per moving frame (every second frame) in state 1.
; OUT:  on landing: ObjState,X = 0, speed cleared, ObjDir,X random, ObjRemDistance,X = $31 or $71,
;       ObjX,X and ObjY,X snapped to the nearest square (Y low nibble $D).
; NOTE: Gravity $38/256 pixels per moving frame. With the table speeds a sideways or upward jump
;       takes 31 moving frames (62 frames) and rises 26 or 44 pixels; a downward jump takes 25 (50
;       frames). RNG: one byte, Random,X: bits 0-1 choose the direction, bit 6 the distance.
; ----------------------------------------------------------------------------
UpdatePolsVoiceState1_Jumping:
    ; Vertical speed (16-bit, whole:fraction) += $0038.
    LDA PolsVoice_ObjSpeedFrac, X
    CLC
    ADC #$38
    STA PolsVoice_ObjSpeedFrac, X

    ; Carry into the whole part.
    LDA PolsVoice_ObjSpeedWhole, X
    ADC #$00
    STA PolsVoice_ObjSpeedWhole, X

    ; Y += whole part of the speed (A still holds it).
    CLC
    ADC ObjY, X
    STA ObjY, X

    ; Still rising: not landed.
    LDA PolsVoice_ObjSpeedWhole, X
    BMI @Exit

    ; Falling but still above the landing height (unsigned compare): not landed.
    LDA ObjY, X
    CMP PolsVoice_ObjTargetY, X
    BCC @Exit

    ; Landed: back to state 0 (walking). The CMP above left C set.
    LDA #$00
    STA ObjState, X

    ; Clear the vertical speed.
    STA PolsVoice_ObjSpeedFrac, X
    STA PolsVoice_ObjSpeedWhole, X

    ; Face a random direction: Random,X AND 3 -> right, left, down, up.
    LDA Random, X
    AND #$03
    TAY
    LDA PolsVoiceDirections, Y
    STA ObjDir, X

    ; Walk distance: bit 6 of the same random byte, plus $30, plus the carry still set by the landing
    ; compare: $31 or $71 moving frames.
    LDA Random, X
    AND #$40
    ADC #$30
    STA ObjRemDistance, X

    ; Snap X to the nearest square: round to a multiple of $10.
    LDA ObjX, X
    CLC
    ADC #$08
    AND #$F0
    STA ObjX, X

    ; Snap Y to the nearest square row the same way ...
    LDA ObjY, X
    CLC
    ADC #$08
    AND #$F0

    ; ... then subtract 3, so the low nibble of Y is $D as for other monsters.
    SEC
    SBC #$03
    STA ObjY, X

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Tests the tiles under a Pols Voice at two points; C set if either one is unwalkable.
; IN:   X = object slot.
; OUT:  C = 0 if both points are walkable; PolsVoice_ObjLastTile,X = tile at the last point tested.
; NOTE: Points tested: (ObjX, ObjY+$0B), then (ObjX+$0E, ObjY+$11). The second is skipped when the
;       first is unwalkable. ObjX and ObjY are changed for the second test and restored.
; ----------------------------------------------------------------------------
PolsVoice_IsSquareWalkable:
    JSR PolsVoice_GetCollidingTile

    ; First point unwalkable: return C = 1.
    BCS @Exit

    ; Save X and add $0E to it for the second point.
    LDA ObjX, X
    PHA
    CLC
    ADC #$0E
    STA ObjX, X

    ; Save Y and add 6 to it.
    LDA ObjY, X
    PHA
    CLC
    ADC #$06
    STA ObjY, X

    ; Test the second point: $0E right of and 6 below the first.
    JSR PolsVoice_GetCollidingTile

    ; Restore the coordinates (PLA and STA leave C alone).
    PLA
    STA ObjY, X
    PLA
    STA ObjX, X

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Reads the tile at (ObjX, ObjY+$0B) and compares it with the first unwalkable tile.
; IN:   X = object slot; [0F] = 0 (see NOTE).
; OUT:  C = 0 if walkable (tile < RoomFirstUnwalkableTile); A and PolsVoice_ObjLastTile,X = the tile.
; NOTE: GetCollidableTile is called with hotspot offset 0 and treats [0F] as a direction. [0F] is
;       still 0 here: UpdateObject stored the object's "uninitialized" flag in it, which is 0 for an
;       object being updated, and nothing in UpdatePolsVoice changes it before this call.
; ----------------------------------------------------------------------------
PolsVoice_GetCollidingTile:
    LDY #$00
    JSR GetCollidableTile
    LDA ObjCollidedTile, X

    ; C = 0 if the tile is walkable.
    CMP RoomFirstUnwalkableTile

    ; Remember the tile (STA and RTS leave C alone).
    STA PolsVoice_ObjLastTile, X
    RTS

; ----------------------------------------------------------------------------
; Adds the X step for the Pols Voice's facing direction to its X coordinate.
; IN:   X = object slot.
; OUT:  Y = ObjDir,X - 1 (index into the Pols Voice direction tables); ObjX,X updated.
; ----------------------------------------------------------------------------
PolsVoice_MoveX:
    LDY ObjDir, X
    DEY
    LDA ObjX, X
    CLC
    ADC PolsVoiceWalkSpeedsX, Y
    STA ObjX, X
    RTS
