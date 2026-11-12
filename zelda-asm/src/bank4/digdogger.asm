; ============================================================================
; FILE:     bank4/digdogger.asm          BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 5155-5594
; SUBJECTS: Digdogger
; TOPIC:    Digdogger update (big types $38/$39 and small type $18): shrinking and splitting after the
;           flute, speed oscillation, 8-way movement, four-corner collisions of the big body, drawing.
; SEE ALSO: bank4/aquamentus-digdogger-dodongo-inits.asm (InitDigdogger1/2: start speed, child count);
;           bank7/flute.asm (WieldFlute sets FluteUsedState and the $98-frame freeze);
;           bank4/fairy-flyer-common.asm (BoundFlyer, TurnTowardsPlayer8, TurnRandomlyDir8, Directions8);
;           bank1/monster-collisions.asm (CheckMonsterCollisions, ObjTypeToDamagePoints);
;           bank7/object-update-table.asm (ObjectTypeToAttributes, ObjectTypeToHpPairs).
; KEY FACTS:
;   - The big Digdogger cannot be hit by any weapon: its ObjAttr is $A9, and bit $20 makes
;     CheckMonsterCollisions skip all weapon tests (ObjectTypeToAttributes, types $38/$39). Only the
;     small one (type $18: ObjAttr $81, HP $80) can be harmed. No code here sets an invincibility
;     mask; it is 0 from the room-entry RAM clear unless an earlier object in the slot set it.
;   - The flute in a dungeon sets FluteUsedState to 1 (WieldFlute). After the $98-frame flute freeze the
;     big Digdogger starts a $40-frame shrink (FluteUsedState 2), then splits (L_Digdogger_AfterFlute).
;   - The split makes ChildDigdoggerCount small Digdoggers (3 from InitDigdogger1, 1 from
;     InitDigdogger2) in the object slots right after the parent's, without testing that they are free,
;     sets RoomObjCount to that number and clears the parent's ObjType (L_Digdogger_AfterFlute).
;   - A child slot not used since room entry is still flagged uninitialized with ObjMetastate 1, so
;     the child first runs InitObject (ObjAttr $81, ObjHP $80) and a spawn cloud (UpdateObject). An
;     occupied slot is overwritten and keeps its old ObjAttr, ObjHP and metastate
;     (L_Digdogger_AfterFlute@LoopMakeChild).
;   - The FluteUsedState = 2 path does not test the size: while one big Digdogger shrinks, any other
;     Digdogger in the room, big or small, also runs it and splits when its own ObjTimer reads 0
;     (L_Digdogger_AfterFlute@SplitUp).
;   - While InvClock is set or the Digdogger is stunned, UpdateDigdogger skips the flute handling, so
;     the split does not start or progress.
;   - During the shrink the Digdogger does not move, and it tests collisions only on the frame the
;     shrink starts and on the 3 frames it is shown big again (L_Digdogger_AfterFlute@SplitUp).
;   - Speed is a 16-bit value changed by 1 per frame. It starts at $003F and rises to the target; from
;     then on it swings between $0040 and $0080 (big, 0.25 to 0.5 pixels per frame) or $0140 and $0180
;     (small, 1.25 to 1.5) (Digdogger_SpeedUp, Digdogger_SlowDown, SetTargetSpeed). Only bits 5-7 of
;     the low byte count, and the speed applies to each axis of the 8-way direction (Digdogger_Move).
;   - Every $10 frames it turns at most one 8-way step: by TurnTowardsPlayer8 (steers at the chase
;     target) when Random,X >= $80, else at random (L_Digdogger_Turn).
;   - The big body is 32x32 pixels: bounds and collisions are tested at four 16x16 corners
;     (CheckBigDigdoggerCollisions). Touching either size costs Link $0200 = 2 hearts before rings
;     (ObjTypeToDamagePoints entries $18, $38, $39 are $02).
;   - The big Digdogger is always drawn with a small one on top of its centre
;     (CheckBigDigdoggerCollisions falls into L_Digdogger_DrawAsLittle).
; CONTENTS: (generated; L = line in this file)
;   L0071 DigdoggerCornerOffsetsX             data  TABLE: X steps between the corners of the big Digdogger tested for collisions; index = corner 0-3;
;   L0076 DigdoggerCornerOffsetsY             data  TABLE: Y steps between the tested corners of the big Digdogger; index = corner 0-3; value = signed
;   L0090 L_Digdogger_AfterFlute              code  Runs a Digdogger after the flute was played: shrink for $40 frames, then split into small ones.
;   L0191 L_Digdogger_TurnTowardLink          code  Digdogger turn, Random,X >= $80: TurnTowardsPlayer8 (at most one 8-way step, steering at the
;   L0201 UpdateDigdogger                     code  Updates a Digdogger of either size (object types $18, $38, $39) for one frame.
;   L0214 L_Digdogger_Turn                    code  Normal movement of a Digdogger: change speed, turn every $10 frames, move, check collisions, draw.
;   L0231 L_Digdogger_Move                    code  Moves the Digdogger, then checks collisions and draws it.
;   L0235 L_Digdogger_DrawAndCheckCollisions  code  Checks room bounds and collisions and draws the Digdogger without moving it.
;   L0252 CheckBigDigdoggerCollisions         code  Bounds and collision-tests the big Digdogger at its four corners, then draws it.
;   L0303 L_Digdogger_DrawAsLittle            code  Draws a small Digdogger 8 pixels right of and below the object's position (the big one's centre).
;   L0349 Digdogger_ChangeSpeed               code  Moves a Digdogger's 16-bit speed 1 unit toward its target: up when the speed flag is 0, else down.
;   L0354 Digdogger_ChangeSpeed_JumpTable     data  TABLE: speed routines; index = Digdogger_ObjSpeedFlag (0 speeding up, 1 slowing down).
;   L0363 Digdogger_SpeedUp                   code  Adds 1 to the 16-bit speed; at the target speed, switches to slowing down toward $0040/$0140.
;   L0388 Digdogger_SlowDown                  code  Subtracts 1 from the 16-bit speed; at the target speed, switches to speeding up toward $0080/$0180.
;   L0413 SetTargetSpeed                      code  Sets the target speed to A (low byte), with high byte 0 for the big Digdogger and 1 for a small one.
;   L0424 L116EA_Exit                         code  Shared exit of Digdogger_SpeedUp, Digdogger_SlowDown and SetTargetSpeed.
;   L0436 Digdogger_Move                      code  Moves a Digdogger by its 16-bit speed along each axis of its 8-way direction.
;   L0504 DigdoggerSpriteOffsetsX             data  TABLE: X offsets of the four 16x16 parts of the big Digdogger's image; index = part 0-3
;   L0508 DigdoggerSpriteOffsetsY             data  TABLE: Y offsets of the four parts of the big Digdogger's image; index = part 0-3; value = pixels.
;   L0514 DigdoggerSpriteAttrs                data  TABLE: sprite attributes of the four parts of the big Digdogger; index = part 0-3;
;   L0524 Digdogger_Draw                      code  Draws a Digdogger: four 16x16 parts when big, one mirrored 16x16 image when small.
; ============================================================================

; TABLE: X steps between the corners of the big Digdogger tested for collisions; index = corner 0-3;
; value = signed pixels added to the previous corner's X. The corners visited are, relative to the
; object position: (0,0), ($10,$10), ($10,0), (0,$10).
DigdoggerCornerOffsetsX:
    .BYTE $00, $10, $00, $F0

; TABLE: Y steps between the tested corners of the big Digdogger; index = corner 0-3; value = signed
; pixels added to the previous corner's Y.
DigdoggerCornerOffsetsY:
    .BYTE $00, $10, $F0, $10

; ----------------------------------------------------------------------------
; Runs a Digdogger after the flute was played: shrink for $40 frames, then split into small ones.
; IN:   X = object slot; Y = FluteUsedState (1 or 2)
; OUT:  FluteUsedState 1 -> 2 when the shrink starts, 2 -> 1 at the split; RoomObjCount = number of
;       children; ObjType,X = 0 at the split.
; NOTE: With FluteUsedState = 1 a small Digdogger passes straight on to the normal turn and move
;       code. With FluteUsedState = 2 the size is not tested: every Digdogger in the room takes the
;       shrink path and splits when its own ObjTimer reads 0. The children are put in slots X+1..
;       whether or not those are free, at the position of slot 1 (ObjX+1, ObjY+1), not slot X.
;       FluteUsedState is one byte for the room (zeroed on room entry), shared by every Digdogger.
; ----------------------------------------------------------------------------
L_Digdogger_AfterFlute:
    DEY
    BNE @SplitUp

    ; FluteUsedState = 1: the flute was played and its freeze is over.
    ;
    ; A small Digdogger ignores the flute: go turn and move as usual.
    LDA Digdogger_ObjIsChild, X
    BNE L_Digdogger_Turn

    ; Big Digdogger: start the shrink. Object timer := $40 frames, FluteUsedState := 2.
    ; This frame only checks collisions and draws it big; it does not move.
    LDA #$40
    STA ObjTimer, X
    INC FluteUsedState
    JMP CheckBigDigdoggerCollisions

@SplitUp:
    ; FluteUsedState = 2 (a shrink is in progress). When this object's timer reads 0, split up.
    ; The size of the object is not tested on this path.
    LDA ObjTimer, X
    BEQ @MakeChildren

    ; Whenever the timer is a multiple of 8 ($38, $30, ... $08), toggle the child flag.
    ; On the frames where the flag becomes 0 (timer $30, $20, $10) the Digdogger is treated as big:
    ; corner collisions, big drawing. On every other frame of the shrink it is only drawn small,
    ; with no movement and no collision test. The flag is 1 when the timer runs out.
    ; (ObjTimer loses 1 per frame, so the split comes $40 frames after the shrink started.)
    AND #$07
    BNE :+
    LDA Digdogger_ObjIsChild, X
    EOR #$01
    STA Digdogger_ObjIsChild, X
    BEQ CheckBigDigdoggerCollisions
:
    JMP L_Digdogger_DrawAsLittle

@MakeChildren:
    ; FluteUsedState := 1, so the children behave as normal small Digdoggers.
    DEC FluteUsedState

    ; Make ChildDigdoggerCount children in the slots after this one.
    LDA ChildDigdoggerCount
    STA $00                     ; [00] is the loop counter
    STA RoomObjCount            ; Replace the object count with the number of children.
    TXA                         ; Save the current object index in X register.
    PHA

@LoopMakeChild:
    ; Set up the next object slot as a small Digdogger (type $18). InitDigdogger1 gives it a random
    ; 8-way direction, speed low byte $3F and target low byte $80 (it also requests the roar sample
    ; and rewrites ChildDigdoggerCount with 3; the loop counts with [00], so 3 is what the variable
    ; holds after any split).
    ; The slot is not tested for being free. If it was not used since room entry, it is still
    ; flagged uninitialized with ObjMetastate 1: on its first update UpdateObject runs InitObject
    ; (ObjAttr $81, ObjHP $80, ObjTimer = slot number; type $18's init routine is DoNothing) and
    ; then the spawn cloud, before UpdateDigdogger runs for it. ObjAttr and ObjHP are not written
    ; here, so a slot that was in use keeps its old values.
    INX
    JSR InitDigdogger1
    LDA #$18
    STA ObjType, X

    ; Target speed high byte +1: $0180 if it was 0 (low byte $80 from InitDigdogger1). This byte and
    ; the child's speed high byte are not written anywhere else here; both are 0 from the room-entry
    ; RAM clear ($0300-$051F) unless another object used these bytes of the slot in this room.
    INC Digdogger_ObjTargetSpeedWhole, X

    ; Flag it small.
    LDA #$01
    STA Digdogger_ObjIsChild, X

    ; Speed flag 0: speeding up.
    LDA #$00
    STA Digdogger_ObjSpeedFlag, X

    ; Place the child at the coordinates of object slot 1 (absolute, not slot X).
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

    ; Remove the big Digdogger by clearing its object type.
    LDA #$00
    STA ObjType, X

    ; Draw it once more as a small one, so that something is shown on this frame. This works with
    ; the type cleared: the other object bytes are untouched, and L_Digdogger_DrawAsLittle sets
    ; type $18 for the duration of the drawing (and then restores the 0).
    JMP L_Digdogger_DrawAsLittle

; Digdogger turn, Random,X >= $80: TurnTowardsPlayer8 (at most one 8-way step, steering at the
; chase target), then move.
L_Digdogger_TurnTowardLink:
    JSR TurnTowardsPlayer8
    JMP L_Digdogger_Move

; ----------------------------------------------------------------------------
; Updates a Digdogger of either size (object types $18, $38, $39) for one frame.
; IN:   X = object slot
; NOTE: With InvClock set or the monster stunned it does not move, and the flute handling is skipped
;       too, so a big Digdogger neither starts nor continues its shrink.
; ----------------------------------------------------------------------------
UpdateDigdogger:
    ; If Link has the clock, or the monster is stunned,
    ; then only check room bounds and collisions, and draw.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE L_Digdogger_DrawAndCheckCollisions

    ; If the flute was played in this room (FluteUsedState 1 or 2), go handle the shrink and split.
    ; With FluteUsedState = 1, small Digdoggers come back to L_Digdogger_Turn from there.
    LDY FluteUsedState
    BNE L_Digdogger_AfterFlute

; Normal movement of a Digdogger: change speed, turn every $10 frames, move, check collisions, draw.
L_Digdogger_Turn:
    JSR Digdogger_ChangeSpeed

    ; When the object timer has run out, set it to $10 frames and turn:
    ; Random,X >= $80 (1 in 2): by TurnTowardsPlayer8, steering at the chase target; else by
    ; TurnRandomlyDir8 (Random+1,X >= $A0 no turn, >= $50 one step clockwise, else one step
    ; counterclockwise).
    LDA ObjTimer, X
    BNE L_Digdogger_Move
    LDA #$10
    STA ObjTimer, X
    LDA Random, X
    CMP #$80
    BCS L_Digdogger_TurnTowardLink
    JSR TurnRandomlyDir8

; Moves the Digdogger, then checks collisions and draws it.
L_Digdogger_Move:
    JSR Digdogger_Move

; Checks room bounds and collisions and draws the Digdogger without moving it.
L_Digdogger_DrawAndCheckCollisions:
    ; A small Digdogger is one 16x16 object: bound it, check collisions, draw.
    ; BoundFlyer reverses its 8-way direction when a room bound blocks it.
    LDA Digdogger_ObjIsChild, X
    BEQ CheckBigDigdoggerCollisions
    JSR BoundFlyer
    JSR CheckMonsterCollisions
    JMP Digdogger_Draw

; ----------------------------------------------------------------------------
; Bounds and collision-tests the big Digdogger at its four corners, then draws it.
; IN:   X = object slot
; NOTE: BoundFlyer runs once per corner and reverses the 8-way direction each time a corner is
;       blocked (two blocked corners cancel out). With ObjAttr bit $20 set, CheckMonsterCollisions
;       tests only the touch with Link, once per corner.
;       Falls into L_Digdogger_DrawAsLittle, which draws a small Digdogger over the centre.
; ----------------------------------------------------------------------------
CheckBigDigdoggerCollisions:
    ; Save the original coordinates.
    LDA ObjX, X
    PHA
    LDA ObjY, X
    PHA

    ; Start with corner 0.
    LDA #$00
    STA Digdogger_ObjCurPart, X

@LoopCornerCollision:
    ; The big Digdogger is 32x32 pixels, but the room bound and collision routines treat an object
    ; as 16x16. So they are run four times, with the object moved to each 16x16 corner in turn.
    ;
    ; Loop over corners 0 to 3.
    LDY Digdogger_ObjCurPart, X

    ; Step the coordinates to this corner (the table values are steps from the previous corner).
    LDA ObjX, X
    CLC
    ADC DigdoggerCornerOffsetsX, Y
    STA ObjX, X
    LDA ObjY, X
    CLC
    ADC DigdoggerCornerOffsetsY, Y
    STA ObjY, X

    ; Check the room bounds and collisions at this temporary position.
    JSR BoundFlyer
    JSR CheckMonsterCollisions

    ; Next corner, until 4.
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

; ----------------------------------------------------------------------------
; Draws a small Digdogger 8 pixels right of and below the object's position (the big one's centre).
; IN:   X = object slot
; NOTE: Temporarily sets ObjType,X = $18 and the child flag = 1; restores both and the coordinates.
;       No collision test is made here.
; ----------------------------------------------------------------------------
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

    ; Set object type $18 (small Digdogger) and child flag 1 while drawing.
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

; ----------------------------------------------------------------------------
; Moves a Digdogger's 16-bit speed 1 unit toward its target: up when the speed flag is 0, else down.
; IN:   X = object slot
; ----------------------------------------------------------------------------
Digdogger_ChangeSpeed:
    LDA Digdogger_ObjSpeedFlag, X
    JSR TableJump

; TABLE: speed routines; index = Digdogger_ObjSpeedFlag (0 speeding up, 1 slowing down).
Digdogger_ChangeSpeed_JumpTable:
    .ADDR Digdogger_SpeedUp
    .ADDR Digdogger_SlowDown

; ----------------------------------------------------------------------------
; Adds 1 to the 16-bit speed; at the target speed, switches to slowing down toward $0040/$0140.
; IN:   X = object slot
; NOTE: The target test is for equality only.
; ----------------------------------------------------------------------------
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

    ; Speed flag := 1: slow down from now on.
    INC Digdogger_ObjSpeedFlag, X

    ; Go set the target speed to $0040 (big) or $0140 (small).
    LDA #$40
    JMP SetTargetSpeed

; ----------------------------------------------------------------------------
; Subtracts 1 from the 16-bit speed; at the target speed, switches to speeding up toward $0080/$0180.
; IN:   X = object slot
; ----------------------------------------------------------------------------
Digdogger_SlowDown:
    ; Subtract 1 from 16-bit speed.
    DEC Digdogger_ObjSpeedFrac, X
    LDA Digdogger_ObjSpeedFrac, X
    CMP #$FF
    BNE :+
    ; Borrow from the high byte. With the targets set in this file ($xx40 and $xx80) the low byte
    ; never wraps while slowing down, so this instruction is not reached in normal play.
    DEC Digdogger_ObjSpeedWhole, X
:
    ; If the 16-bit speed has not reached the 16-bit target speed, then return.
    LDA Digdogger_ObjSpeedFrac, X
    CMP Digdogger_ObjTargetSpeedFrac, X
    BNE L116EA_Exit
    LDA Digdogger_ObjSpeedWhole, X
    CMP Digdogger_ObjTargetSpeedWhole, X
    BNE L116EA_Exit

    ; Speed flag := 0: speed up from now on.
    DEC Digdogger_ObjSpeedFlag, X

    ; Set the target speed to $0080 (big) or $0180 (small).
    LDA #$80

; Sets the target speed to A (low byte), with high byte 0 for the big Digdogger and 1 for a small one.
SetTargetSpeed:
    STA Digdogger_ObjTargetSpeedFrac, X
    LDA #$00
    STA Digdogger_ObjTargetSpeedWhole, X

    ; A small Digdogger gets high byte 1: 1 more pixel per frame.
    LDA Digdogger_ObjIsChild, X
    BEQ L116EA_Exit
    INC Digdogger_ObjTargetSpeedWhole, X

; Shared exit of Digdogger_SpeedUp, Digdogger_SlowDown and SetTargetSpeed.
L116EA_Exit:
    RTS

; ----------------------------------------------------------------------------
; Moves a Digdogger by its 16-bit speed along each axis of its 8-way direction.
; IN:   X = object slot; ObjDir,X = direction bits (1 right, 2 left, 4 down, 8 up; diagonals combine)
; OUT:  ObjX,X / ObjY,X changed; [00], [01] = new X, Y; [0F] = 0; A = 0
; NOTE: Only bits 5-7 of the speed low byte count, so the big Digdogger's speed steps through
;       $40, $60, $80 (0.25, 0.375, 0.5 pixels per frame), after $20 (0.125) while the speed is
;       still $003F at the start. A diagonal moves the full amount on both axes. No tile
;       collision: only the room bounds (BoundFlyer, in the callers) confine it.
; ----------------------------------------------------------------------------
Digdogger_Move:
    ; Add the top 3 bits of the speed low byte to the position fraction at $412,X.
    ; Pixels to move this frame: [03] := speed high byte + carry.
    ; Manhandla_Move does the same calculation.
    LDA Digdogger_ObjSpeedFrac, X
    AND #$E0
    CLC
    ADC $0412, X
    STA $0412, X

    ; [03] := speed high byte + carry.
    LDA Digdogger_ObjSpeedWhole, X
    ADC #$00
    STA $03

    ; Change the coordinates by [03] pixels for each direction bit that is set.
    ;
    ; [02] starts as the mask $A1 and is shifted left before each later test:
    ; - its low nibble is the direction bit being tested (1, 2, 4, 8);
    ; - the bit shifted out of the top sets Carry for a subtraction (left, up)
    ;   or clears it for an addition (down).
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

; TABLE: X offsets of the four 16x16 parts of the big Digdogger's image; index = part 0-3
; (top left, top right, bottom left, bottom right); value = pixels.
DigdoggerSpriteOffsetsX:
    .BYTE $00, $10, $00, $10

; TABLE: Y offsets of the four parts of the big Digdogger's image; index = part 0-3; value = pixels.
DigdoggerSpriteOffsetsY:
    .BYTE $00, $00, $10, $10

; TABLE: sprite attributes of the four parts of the big Digdogger; index = part 0-3;
; value = OAM attribute byte: $03 = sprite palette 3 (the level's palette row), $80 = flipped
; vertically (the bottom parts).
DigdoggerSpriteAttrs:
    .BYTE $03, $03, $83, $83

; ----------------------------------------------------------------------------
; Draws a Digdogger: four 16x16 parts when big, one mirrored 16x16 image when small.
; IN:   X = object slot; Digdogger_ObjIsChild,X selects the size
; NOTE: Each call counts ObjAnimCounter down once (reload 6) and flips ObjAnimFrame at 0. The big
;       Digdogger is drawn through this routine twice per frame (body, then the small overlay).
;       Only the small image uses ObjAnimFrame.
; ----------------------------------------------------------------------------
Digdogger_Draw:
    LDA #$06                    ; Animation counter reload value.
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor

    ; If this is a small Digdogger, go draw it.
    LDA Digdogger_ObjIsChild, X
    BNE @DrawLittle

    ; For each part of the big Digdogger, from 0 to 3, indexed by Y register:
    LDY #$00

@LoopPart:
    ; [00] := X of the current part.
    LDA ObjX, X
    CLC
    ADC DigdoggerSpriteOffsetsX, Y
    STA $00

    ; [01] := Y of the current part.
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

    ; Parts at odd indexes (the right column) are flipped horizontally: [0F] := 1.
    AND #$01
    STA $0F
    LDA #$00                    ; Frame image 0: the big body has a single image.
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
    ; Small Digdogger: one mirrored image in the level's palette row, frame from ObjAnimFrame.
    JSR Anim_SetSpriteDescriptorLevelPaletteRow
    LDA ObjAnimFrame, X
    JMP DrawObjectMirrored
