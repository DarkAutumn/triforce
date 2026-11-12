; ============================================================================
; FILE:     bank4/manhandla.asm          BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 7832-8191
; SUBJECTS: (none)
; TOPIC:    Manhandla
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0016 UpdateManhandla                    code
;   L0152 Manhandla_CheckCollisions          code
;   L0243 Manhandla_SetAllSegmentsDirection  code  Params:
;   L0251 Manhandla_Move                     code
;   L0342 Manhandla_Draw                     code
; ============================================================================

UpdateManhandla:
    ; If this segment is not the base, then skip turning and speeding up.
    CPX #$05
    BNE @MoveBase

    ; If no segment just died, then skip increasing the speed.
    LDA Manhandla_SegmentJustDied
    BEQ @BounceIfNeeded

    ; For each segment from 4 to 0, indexed by Y register:
    LDY #$04

@LoopSegment:
    ; Add $80 to the low speed byte of this segment.
    LDA Manhandla_ObjSpeedFrac+1, Y
    CLC
    ADC #$80
    STA Manhandla_ObjSpeedFrac+1, Y

    ; Carry to the high speed byte of this segment.
    LDA Manhandla_ObjSpeedWhole+1, Y
    ADC #$00
    STA Manhandla_ObjSpeedWhole+1, Y
    DEY
    BPL @LoopSegment

    ; Reset the "a segment just died" flag.
    LDA #$00
    STA Manhandla_SegmentJustDied

@BounceIfNeeded:
    ; If there's a bounce direction, then assign it to the facing
    ; direction of all segments.
    LDA Manhandla_BounceDir
    BEQ @TurnIfNeeded
    JSR Manhandla_SetAllSegmentsDirection

@TurnIfNeeded:
    ; If object timer = 0, then
    ; 1. set timer to $10
    ; 2. randomly choose to turn toward Link or turn randomly
    ; 3. assign the base's direction to all segments and bounce direction
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
    ; If this segment is the base, then remember the
    ; facing direction before moving.
    ;
    ; This is a continuation of the code above,
    ; which is also only for the base.
    CPX #$05
    BNE :+
    LDA ObjDir, X
    STA $0384
:
    JSR Manhandla_Move
    JSR Manhandla_CheckCollisions

    ; If the direction changed after moving, then
    ; copy it to the bounce direction.
    LDA ObjDir, X
    CMP $0384
    BEQ :+
    STA Manhandla_BounceDir
:
    ; Store in [00] the animation frame that you get from the frame accumulator:
    ; frame := ((accumulator AND $10) >> 4
    LDA Manhandla_ObjFrameAccum, X
    AND #$10
    LSR
    LSR
    LSR
    LSR
    STA $00

    ; Copy the animation frame bit into the "frame image and
    ; sprite attributes" byte.
    LDA Manhandla_ObjFrame, X
    AND #$FE
    ORA $00
    STA Manhandla_ObjFrame, X

    ; If this segment is the base, then go draw.
    CPX #$05
    BEQ @Draw

    ; If the old and new "frame image and sprite attributes" variables
    ; are the same, then go draw.
    ;
    ; The previous frame value begins with value 0, because all
    ; room data was reset on entry.
    LDA Manhandla_ObjFrame, X
    CMP Manhandla_ObjPrevFrame, X
    BEQ @Draw

    ; Copy the new value to the old variable, so that they're the same.
    STA Manhandla_ObjPrevFrame, X

    ; If animation frame = 1, then go draw.
    ;
    ; The low bit of "frame image and sprite attributes" byte
    ; represents the animation frame.
    LSR
    BCS @Draw

    ; If the random value for the next slot up < $E0,
    ; or there's an object in slot 7; then go draw.
    ;
    ; The second condition keeps the number of fireballs <= 4.
    LDA Random+1, X
    CMP #$E0
    BCC @Draw
    LDA ObjType+7
    BNE @Draw
    LDA #$56                    ; Fireball object type $56 (unblockable)
    JSR ShootFireball

@Draw:
    JMP Manhandla_Draw

@TurnTowardLink:
    JSR TurnTowardsPlayer8
    JMP @CopyDirToHands

Manhandla_CheckCollisions:
    ; Check collisions, but never let them interfere with movement.
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
    JSR PlayBossHitCryIfNeeded

    ; If still alive, then return.
    LDA ObjMetastate, X
    BEQ @Exit
    JSR ResetShoveInfo

    ; Don't let the base die on its own.
    ; If this is the base, then reset metastate, and return.
    ;
    ; The base only dies once the last hand dies.
    CPX #$05
    BNE @CountHands
    STA ObjMetastate, X
    JMP @Exit

@CountHands:
    ; At this point, we have a hand that just died.
    ;
    ; For each hand, from 3 to 0, indexed by 0:
    LDY #$03

    ; Count how many are allocated.
    ; This includes the current dead one, and the other living ones.
    ; Store the count in [00].
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

    ; If the count >= 0 (signed comparison), then
    ; go change this hand object to the dead dummy.
    BMI :+
    BNE @MakeDeadDummy
:
    ; The last hand died. So, kill the base.
    ;
    ; Change the base segment's object type to the dead dummy.
    ; Immediately set metastate to $10, instead of waiting for
    ; the dead dummy object's update routine to do it.
    ; Also set its timer to $10.
    JSR PlayBossDeathCry
    LDA #$5D
    STA ObjType+5
    LDA #$10
    STA ObjMetastate+5
    STA ObjTimer+5

@FlagHandDied:
    ; Set the "segment just died" flag.
    INC Manhandla_SegmentJustDied

@Exit:
    RTS

@MakeDeadDummy:
    LDA #$5D                    ; Dead dummy object type
    STA ObjType, X
    JMP @FlagHandDied

; Params:
; A: direction
;
Manhandla_SetAllSegmentsDirection:
    LDY #$04
:
    STA a:ObjDir+1, Y
    DEY
    BPL :-
    RTS

Manhandla_Move:
    ; Implement a way of manifesting the fractional part of the speed
    ; in the movement.
    ;
    ; Let's say the 16-bit speed is $0140. If we only took the high
    ; byte to add to the coordinates, then the effective speed would
    ; be 1 pixel a frame instead of 1.25.
    ;
    ; By keeping a speed accumulator, we can turn speed $0140
    ; into 1.25 pixels a frame over four frames, using this calculation:
    ;
    ;   (SA=speed accumulator, SH=speed high, SL=speed low)
    ;
    ;   SA := SA + (SL AND $E0)
    ;   [03] := SH + carry
    ;
    ; AND'ing the low speed byte with $E0 isn't strictly needed.
    LDA Manhandla_ObjSpeedFrac, X
    AND #$E0
    CLC                         ; Add the low speed byte to speed accumulator.
    ADC Manhandla_ObjSpeedAccum, X
    STA Manhandla_ObjSpeedAccum, X

    ; Assign (high speed byte + carry) to [03].
    LDA Manhandla_ObjSpeedWhole, X
    ADC #$00
    STA $03

    ; Change coordinates by the speed amount/offset in [03]
    ; according to the direction.
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
    BEQ @IncFrameAccum
    LDA ObjY, X
    SBC $03
    STA ObjY, X

@IncFrameAccum:
    ; To the frame accumulator, add:
    ; - a random value between 0 and 3
    ; - [03] the effective high speed byte calculated for this screen frame
    LDA Random+1, X
    AND #$03
    CLC
    ADC $03
    ADC Manhandla_ObjFrameAccum, X
    STA Manhandla_ObjFrameAccum, X
    JSR BoundFlyer

    ; UNKNOWN: It seems that an RTS would be enough.
    JMP Anim_FetchObjPosForSpriteDescriptor

Manhandla_Draw:
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Push the "frame image and sprite attributes" value.
    LDA Manhandla_ObjFrame, X
    PHA

    ; Bit 7 is the sprite attribute for vertical flipping, and can be
    ; passed along as is.
    AND #$80
    ORA #$01                    ; Use blue palette row.
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
