; ============================================================================
; FILE:     bank4/manhandla.asm          BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 7832-8191
; SUBJECTS: Manhandla
; TOPIC:    Manhandla (object type $3C) per-frame update: the base in object slot 5 steers and speeds up
;           all five segments, the four hands in slots 1-4 shoot fireballs; movement, collisions, the
;           death of hands and base, and drawing.
; SEE ALSO: bank4/gleeok-manhandla-gohma-inits.asm (InitManhandla: segment layout, HP copy, invincibility
;           mask $E2, start speed); bank4/fairy-flyer-common.asm (TurnRandomlyDir8, TurnTowardsPlayer8,
;           BoundFlyer, Directions8); bank4/monster-shots-fireballs.asm (ShootFireball, UpdateFireball);
;           bank1/monster-collisions.asm (CheckMonsterCollisions); bank1/weapon-damage-death.asm
;           (DealDamage, HandleMonsterDied); bank7/object-update-table.asm (UpdateMetaObjectEnd: what a
;           dying segment turns into).
; KEY FACTS:
;   - Five objects of type $3C: hands in slots 1-4 (up, down, left, right of the base), base in slot 5.
;     UpdateManhandla runs once per segment per frame; slot 5 runs before slots 4-1.
;   - Only hands can be killed. A killed hand that is not the last becomes a dead dummy ($5D: no
;     RoomKillCount or WorldKillCycle step, no drop). When the last hand dies the base becomes a dying
;     dead dummy and the last hand keeps type $3C, so it is the one death counted there. Every hand
;     death adds 1 to WorldKillCount and HelpDropCount (HandleMonsterDied). Manhandla_CheckCollisions
;   - The base takes weapon hits but its death is cancelled (metastate reset to 0) and its invincibility
;     timer is cleared every frame, so a weapon overlapping it hits every frame. Each hit after its HP
;     ran out still runs HandleMonsterDied (WorldKillCount, HelpDropCount). Manhandla_CheckCollisions
;   - A hand that was hit has ObjInvincibilityTimer $10 (about 32 frames); meanwhile it is tested
;     against neither weapons nor Link (CheckMonsterCollisions returns early). The base always is.
;   - Speed is 16-bit, pixels.fraction per frame on each axis of the direction: $0080 (0.5) at start,
;     +$0080 after a frame in which any hands died: 1.0, 1.5, 2.0 with 3, 2, 1 hands left. Two hands
;     dying in the same frame add only one step. UpdateManhandla, Manhandla_Move
;   - Every $10 frames the base turns: Random,X >= $80 toward the chase target (TurnTowardsPlayer8;
;     ChaseTargetX/Y, Link's position unless monsters are chasing the decoy point), else
;     TurnRandomlyDir8 (one 45-degree step or none). All segments copy the base's direction.
;   - A segment that the room boundary turns around (BoundFlyer) stores its new direction in
;     Manhandla_BounceDir; the base gives it to all segments on its next update. UpdateManhandla
;   - A hand shoots fireball type $56 (the shield cannot block it) when its animation frame changes
;     to 0, Random+1,X >= $E0 (1 in 8) and object slot 7 is empty. UpdateManhandla
;   - Segments are never moved by knockback: nothing here calls the shove routine. Manhandla_Move
; CONTENTS: (generated; L = line in this file)
;   L0060 UpdateManhandla                    code  Updates one Manhandla segment: the base steers and speeds up all, each moves, collides, draws.
;   L0206 Manhandla_CheckCollisions          code  Checks one segment against Link and his weapons; handles the death of a hand and of the base.
;   L0300 Manhandla_SetAllSegmentsDirection  code  Sets the facing direction of all five segments (object slots 1-5).
;   L0316 Manhandla_Move                     code  Moves a segment by its 16-bit speed along its direction and turns it around at the room boundary.
;   L0412 Manhandla_Draw                     code  Draws one segment from its "frame image and sprite attributes" byte (Manhandla_ObjFrame).
; ============================================================================

; ----------------------------------------------------------------------------
; Updates one Manhandla segment: the base steers and speeds up all, each moves, collides, draws.
; IN:   X = object slot (1-4 a hand, 5 the base)
; NOTE: The base (slot 5) is updated before the hands in the same frame, so the direction it sets is
;       the one all hands move in this frame.
; NOTE: Speed-up: one step of $0080 (0.5 pixel per frame) per frame in which Manhandla_SegmentJustDied
;       is nonzero, however many hands died since the base last looked.
; NOTE: Turns come every $10 frames (ObjTimer of slot 5); 1 in 2 toward the chase target (Random,X >=
;       $80; ChaseTargetX/Y), else TurnRandomlyDir8: Random+1,X >= $A0 keep, >= $50 one step
;       clockwise, else one step counterclockwise in Directions8.
; NOTE: Hand fireballs (type $56): tested only on the frame the hand's frame byte changes to animation
;       frame 0; needs Random+1,X >= $E0 and ObjType of slot 7 = 0. ShootFireball takes the highest
;       empty slot from $0B down, so at most slots $0B-$07 hold fireballs (5), fewer once other objects
;       sit in those slots.
; ----------------------------------------------------------------------------
UpdateManhandla:
    ; Only the base (slot 5) speeds up and turns the segments.
    CPX #$05
    BNE @MoveBase

    ; If no hand died since the last check, then skip increasing the speed.
    LDA Manhandla_SegmentJustDied
    BEQ @BounceIfNeeded

    ; For each segment, Y = 4 to 0 (object slots 5 to 1):
    LDY #$04

@LoopSegment:
    ; Add $0080 to the segment's 16-bit speed (0.5 pixel per frame).
    LDA Manhandla_ObjSpeedFrac+1, Y
    CLC
    ADC #$80
    STA Manhandla_ObjSpeedFrac+1, Y
    LDA Manhandla_ObjSpeedWhole+1, Y
    ADC #$00
    STA Manhandla_ObjSpeedWhole+1, Y
    DEY
    BPL @LoopSegment

    ; Reset the "a hand died" count. It may have been more than 1.
    LDA #$00
    STA Manhandla_SegmentJustDied

@BounceIfNeeded:
    ; If there's a bounce direction, then assign it to the facing
    ; direction of all segments. It is 0 only until the first turn or boundary reversal.
    LDA Manhandla_BounceDir
    BEQ @TurnIfNeeded
    JSR Manhandla_SetAllSegmentsDirection

@TurnIfNeeded:
    ; If the base's object timer = 0, then
    ; 1. set the timer to $10 frames
    ; 2. turn toward Link (Random,X >= $80) or turn randomly
    ; 3. assign the base's direction to the bounce direction and all segments
    LDA ObjTimer, X
    BNE @MoveBase
    LDA #$10
    STA ObjTimer, X
    LDA Random, X
    CMP #$80
    BCS @TurnTowardLink
    JSR TurnRandomlyDir8

@CopyDirToHands:
    LDA ObjDir+5
    STA Manhandla_BounceDir
    JSR Manhandla_SetAllSegmentsDirection

@MoveBase:
    ; All segments get here. If this is the base, then remember its
    ; facing direction before moving in $0384. The hands, updated later
    ; in the frame, compare against the same saved value.
    CPX #$05
    BNE :+
    LDA ObjDir, X
    STA $0384
:
    JSR Manhandla_Move
    JSR Manhandla_CheckCollisions

    ; If this segment's direction now differs from the base's direction
    ; before moving (the room boundary reversed it), then
    ; copy it to the bounce direction.
    LDA ObjDir, X
    CMP $0384
    BEQ :+
    STA Manhandla_BounceDir
:
    ; [00] := animation frame = bit 4 of the frame accumulator.
    LDA Manhandla_ObjFrameAccum, X
    AND #$10
    LSR
    LSR
    LSR
    LSR
    STA $00

    ; Copy the animation frame into bit 0 of the
    ; "frame image and sprite attributes" byte.
    LDA Manhandla_ObjFrame, X
    AND #$FE
    ORA $00
    STA Manhandla_ObjFrame, X

    ; The base doesn't shoot. Go draw.
    CPX #$05
    BEQ @Draw

    ; If the hand's frame byte did not change this frame, then go draw.
    ;
    ; The previous frame value begins with value 0, because all
    ; room data was reset on entry.
    LDA Manhandla_ObjFrame, X
    CMP Manhandla_ObjPrevFrame, X
    BEQ @Draw

    ; Remember the new value.
    STA Manhandla_ObjPrevFrame, X

    ; If it changed to animation frame 1 (bit 0), then go draw.
    LSR
    BCS @Draw

    ; Changed to animation frame 0. Shoot only if the random value of
    ; the next slot up >= $E0 (1 in 8) and object slot 7 is empty.
    ;
    ; The second condition caps the fireballs: new objects take the
    ; highest empty slot from $0B down (FindEmptyMonsterSlot).
    LDA Random+1, X
    CMP #$E0
    BCC @Draw
    LDA ObjType+7
    BNE @Draw
    LDA #$56                    ; Fireball object type $56 (the shield cannot block it)
    JSR ShootFireball

@Draw:
    JMP Manhandla_Draw

@TurnTowardLink:
    JSR TurnTowardsPlayer8
    JMP @CopyDirToHands

; ----------------------------------------------------------------------------
; Checks one segment against Link and his weapons; handles the death of a hand and of the base.
; IN:   X = object slot (1-4 a hand, 5 the base)
; OUT:  ObjDir,X and ObjTimer,X unchanged; Manhandla_SegmentJustDied incremented when a hand died.
; NOTE: The base cannot die by itself: its dying metastate is reset to 0 at once. Its invincibility
;       timer is cleared every frame, so it has no invincibility frames; a weapon touching it deals
;       damage every frame, and once its HP is used up every such hit runs HandleMonsterDied again
;       (WorldKillCount and HelpDropCount go up, death sound).
; NOTE: A hand that dies while other hands (type $3C in slots 1-4) remain becomes type $5D (dead dummy)
;       in its dying metastate: UpdateMetaObjectEnd then steps neither RoomKillCount nor
;       WorldKillCycle, and $5D never drops an item. The last hand keeps type $3C (counted there, and
;       handed to SetUpDroppedItem as type $3C) and turns the base into a dead dummy with metastate $10.
;       WorldKillCount and HelpDropCount went up for every hand when it died (HandleMonsterDied).
; NOTE: A hit hand is skipped by CheckMonsterCollisions (weapons and Link's touch) while its
;       ObjInvincibilityTimer runs ($10, minus 1 every second frame).
; NOTE: Killing two hands in one frame is handled per hand; only the speed-up step is shared.
; ----------------------------------------------------------------------------
Manhandla_CheckCollisions:
    ; Check collisions, but never let them change the direction or the timer.
    LDA ObjDir, X
    PHA                         ; Save the direction
    LDA ObjTimer, X
    PHA                         ; Save the timer
    JSR CheckMonsterCollisions
    PLA                         ; Restore the timer
    STA ObjTimer, X
    PLA                         ; Restore the direction
    STA ObjDir, X

    ; Don't let the base segment be temporarily invincible.
    CPX #$05
    BNE :+
    LDA #$00
    STA ObjInvincibilityTimer, X
:
    ; Request the boss cry if the invincibility timer = $10 (just hit).
    ; Never true for the base, whose timer was just cleared.
    JSR PlayBossHitCryIfNeeded

    ; If still alive (metastate 0), then return.
    LDA ObjMetastate, X
    BEQ @Exit
    JSR ResetShoveInfo          ; Returns A = 0.

    ; Don't let the base die on its own.
    ; If this is the base, then reset metastate to 0, and return.
    ;
    ; The base only dies once the last hand dies.
    CPX #$05
    BNE @CountHands
    STA ObjMetastate, X
    JMP @Exit

@CountHands:
    ; At this point, we have a hand that just died.
    ;
    ; Count in [00] the slots 4 to 1 (Y = 3 to 0) that still hold type $3C.
    ; This includes the hand that just died and the other living ones;
    ; hands that died earlier are type $5D or gone.
    LDY #$03
    LDA #$00
    STA $00

@LoopHand:
    LDA ObjType+1, Y
    CMP #$3C
    BNE :+
    INC $00
:
    DEY
    BPL @LoopHand

    ; Decrease by 1, to account for the current dead one.
    DEC $00

    ; If living hands remain (count > 0), then
    ; go change this hand object to the dead dummy.
    BMI :+
    BNE @MakeDeadDummy
:
    ; The last hand died. So, kill the base.
    ;
    ; Change the base segment's object type to the dead dummy.
    ; Immediately set metastate to $10 (first death sparkle step), instead
    ; of waiting for the dead dummy object's update routine to do it.
    ; Also set its timer to $10 frames.
    ; This hand keeps type $3C while it dies.
    JSR PlayBossDeathCry
    LDA #$5D
    STA ObjType+5
    LDA #$10
    STA ObjMetastate+5
    STA ObjTimer+5

@FlagHandDied:
    ; Count the death for the base's speed-up.
    INC Manhandla_SegmentJustDied

@Exit:
    RTS

@MakeDeadDummy:
    LDA #$5D                    ; Dead dummy object type
    STA ObjType, X
    JMP @FlagHandDied

; ----------------------------------------------------------------------------
; Sets the facing direction of all five segments (object slots 1-5).
; IN:   A = direction
; OUT:  Y = $FF
; ----------------------------------------------------------------------------
Manhandla_SetAllSegmentsDirection:
    LDY #$04
:
    STA a:ObjDir+1, Y
    DEY
    BPL :-
    RTS

; ----------------------------------------------------------------------------
; Moves a segment by its 16-bit speed along its direction and turns it around at the room boundary.
; IN:   X = object slot
; OUT:  [03] = pixels moved this frame on each axis of the direction; ObjDir,X reversed by BoundFlyer
;       when the room boundary blocked it; [00],[01] = ObjX,X, ObjY,X; [0F] = 0
; NOTE: A diagonal direction moves the full amount on both axes (no diagonal slow-down).
; NOTE: No tile collision and no knockback: only the room boundary limits a segment.
; ----------------------------------------------------------------------------
Manhandla_Move:
    ; The speed is 16 bits: pixels per frame in the high byte, 1/256
    ; pixel in the low byte. The position fraction accumulates the low
    ; byte; each carry out of it moves one extra pixel:
    ;
    ;   fraction := fraction + (speed low AND $E0)
    ;   [03] := speed high + carry
    ;
    ; The speed low byte is only ever $00 or $80, so the AND changes nothing.
    LDA Manhandla_ObjSpeedFrac, X
    AND #$E0
    CLC
    ADC Manhandla_ObjPosFrac, X
    STA Manhandla_ObjPosFrac, X

    ; Assign (high speed byte + carry) to [03].
    LDA Manhandla_ObjSpeedWhole, X
    ADC #$00
    STA $03

    ; Change coordinates by the amount in [03] according to the direction.
    ;
    ; [02] is a mask shifted left once per direction bit, starting at $A1:
    ; - the low nibble selects the direction bit to test (the object's
    ;   direction never has high-nibble bits, so those are ignored by BIT)
    ; - bits 7, 6, 5 are shifted out into Carry: 1, 0, 1, which is the
    ;   Carry that SBC, ADC, SBC below need
    LDA #$A1
    STA $02

    ; If direction has a right component (1), then add the amount to X.
    LDA ObjDir, X
    BIT $02
    BEQ @Left
    LDA ObjX, X
    CLC
    ADC $03
    STA ObjX, X

@Left:
    ; If direction has a left component (2), then subtract the amount from X.
    LDA ObjDir, X
    ASL $02                     ; Bit 7 of mask $A1 sets Carry now.
    BIT $02
    BEQ @Down
    LDA ObjX, X
    SBC $03
    STA ObjX, X

@Down:
    ; If direction has a down component (4), then add the amount to Y.
    LDA ObjDir, X
    ASL $02                     ; Bit 6 of mask $A1 clears Carry now.
    BIT $02
    BEQ @Up
    LDA ObjY, X
    ADC $03
    STA ObjY, X

@Up:
    ; If direction has an up component (8), then subtract the amount from Y.
    LDA ObjDir, X
    ASL $02                     ; Bit 5 of mask $A1 sets Carry now.
    BIT $02
    BEQ @IncFrameAccum
    LDA ObjY, X
    SBC $03
    STA ObjY, X

@IncFrameAccum:
    ; To the frame accumulator, add:
    ; - a random value between 0 and 3
    ; - [03], the pixels moved this frame
    ; Bit 4 of the accumulator is the animation frame, so a segment
    ; animates faster the faster it moves.
    LDA Random+1, X
    AND #$03
    CLC
    ADC $03
    ADC Manhandla_ObjFrameAccum, X
    STA Manhandla_ObjFrameAccum, X

    ; Reverse the direction if the room boundary blocks it.
    JSR BoundFlyer

    ; Not needed (an RTS would do): the callers' next steps do not use
    ; [00], [01], [0F], and Manhandla_Draw fetches the position again.
    JMP Anim_FetchObjPosForSpriteDescriptor

; ----------------------------------------------------------------------------
; Draws one segment from its "frame image and sprite attributes" byte (Manhandla_ObjFrame).
; IN:   X = object slot
; NOTE: Manhandla_ObjFrame: low nibble = frame image (0/1 vertical hand, 2/3 horizontal hand, 4/5
;       base), bit 6 = flip horizontally, bit 7 = flip vertically. All segments use sprite palette
;       row 1.
; ----------------------------------------------------------------------------
Manhandla_Draw:
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Push the "frame image and sprite attributes" value.
    LDA Manhandla_ObjFrame, X
    PHA

    ; Bit 7 is the sprite attribute for vertical flipping, and can be
    ; passed along as is.
    AND #$80
    ORA #$01                    ; Sprite palette row 1.
    JSR Anim_SetSpriteDescriptorAttributes
    PLA                         ; Get the "frame image and sprite attributes" value again.
    PHA

    ; If bit 6 is set, then enable horizontal flipping in [0F].
    AND #$40
    BEQ :+
    INC $0F
:
    PLA                         ; Pop the "frame image and sprite attributes" value.

    ; If the low nibble = 2 or 3, then go draw not mirrored.
    ; These are the 2 frame images of the horizontal hands.
    ;
    ; Else draw the base and vertical hands mirrored.
    AND #$0F
    CMP #$02
    BEQ :+
    CMP #$03
    BEQ :+
    JMP DrawObjectMirrored
:
    JMP DrawObjectNotMirrored
