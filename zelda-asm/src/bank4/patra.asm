; ============================================================================
; FILE:     bank4/patra.asm              BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 10029-10282
; SUBJECTS: Flyer
; TOPIC:    Patra (types $47, $48): its flight, the rule that it can be hurt only when no child is
;           left, the orbit-pattern switch; its 8 children (types $25, $26): staggered start, orbit.
; SEE ALSO: bank4/patra-ganon-inits.asm (InitPatra: masks, start spot, speed, makes the children);
;           bank4/orbit-rotation-math.asm (RotateObjectLocation, DecreaseObjectAngle, PatraSines);
;           bank4/fairy-flyer-common.asm (Flyer_SpeedUp, Flyer_Chase, Flyer_Wander, MoveFlyer);
;           bank1/monster-collisions.asm (CheckMonsterCollisions); bank1/link-collision-harm.asm.
; KEY FACTS:
;   - Patra is in slot $01, its children in slots $02-$09. While any slot $02-$09 holds type $25 or
;     $26, Patra is tested only against Link, not against weapons (UpdatePatra@LoopChildren).
;   - Children appear one at a time above Patra, each when the child in slot $02 reaches the angle
;     in PatraChildStartAngles; that leaves them 4 angle steps (1/8 circle) apart (UpdatePatraChild).
;   - Children test no collisions at all (weapons or Link) until the last one (slot $09) has
;     appeared and set Patra's ObjState to 1 (UpdatePatraChild@Ready, PatraChild_State1).
;   - A killed child becomes a dead dummy ($5D): no kill count, no drop (PatraChild_State1).
;   - Type $25 children (Patra $47) turn $70/256 of an angle step per frame, type $26 (Patra $48)
;     $60/256; a circle is 32 steps, so about 73 and 85 frames per orbit (PatraChild_State1).
;   - Patra_ObjManeuverIndex of slot $01 selects the orbit shape: type $25: 0 = circle, 1 = ring
;     half as tall; type $26: 0 = small circle, 1 = circle twice as wide (the three bit-count
;     tables overlap: PatraChild1RotationCosineBits, PatraChild2RotationBits,
;     PatraChild1RotationSineBits).
;   - The index flips when ObjTimer of slot $02 is 0 and the angle of slot $03 is 0; the timer is
;     then reloaded with $FF from PatraManeuverTimes (entry 1, $50, is never read)
;     (UpdatePatra@TryChangeManeuver).
;   - Patra flies as a flyer: speed up once, then chase (Random,X >= $40) or wander: turn counter
;     8 = 7 turns $10 frames apart, then decide again (ControlPatraFlight, Flyer_PatraDecideState).
;   - Children follow Patra by adding its per-frame displacement (Flyer_ObjOffsetX/Y of slot $01)
;     to their own position before rotating (UpdatePatra, PatraChild_State1).
; CONTENTS: (generated; L = line in this file)
;   L0050 PatraManeuverTimes             data  TABLE: frames until the next orbit-pattern switch is allowed; index = Y left by the collision
;   L0060 UpdatePatra                    code  Updates Patra: flies, draws, checks collisions, and switches the children's orbit pattern.
;   L0123 ControlPatraFlight             code  Runs Patra's flight state (Flyer_ObjFlyingState,X) through the table below.
;   L0128 ControlPatraFlight_JumpTable   data  TABLE: Patra's flight state handlers; index = Flyer_ObjFlyingState,X.
;   L0142 Flyer_PatraDecideState         code  Flight state 1: picks chase (Random,X >= $40) or wander (Random,X < $40) with turn counter 8.
;   L0158 PatraChildStartAngles          data  TABLE: angle the child in slot $02 must have for a later child to appear above Patra;
;   L0163 PatraChild1RotationCosineBits  data  TABLE: type $25 child: number of high bits of the cosine used for the Y step; index =
;   L0168 PatraChild2RotationBits        data  TABLE: type $26 child: number of high sine and cosine bits used for both steps; index =
;   L0175 PatraChild1RotationSineBits    data  TABLE: type $25 child: number of high bits of the sine used for the X step; index =
;   L0185 UpdatePatraChild               code  Updates a Patra child: state 0 waits for its turn to appear above Patra, state 1 orbits.
;   L0253 PatraChild_State1              code  Patra child state 1: follows Patra, steps its orbit angle, moves along the orbit, draws, collides.
;   L0323 PatraChild_Draw                code  Draws a Patra child with sprite palette 2, a new animation frame every 2 frames.
; ============================================================================

; TABLE: frames until the next orbit-pattern switch is allowed; index = Y left by the collision
; calls in UpdatePatra; value = frames. Y is 0 there (entry 0, $FF), or $10 on the frame Patra is
; killed (reads a code byte past the table), so entry 1 ($50) is never read.
PatraManeuverTimes:
    .BYTE $FF, $50

; ----------------------------------------------------------------------------
; Updates Patra: flies, draws, checks collisions, and switches the children's orbit pattern.
; IN:   X = object slot (Patra is made in slot $01).
; OUT:  Flyer_ObjOffsetX,X / Flyer_ObjOffsetY,X = pixels moved this frame, for the children.
; NOTE: Link's weapons are tested against Patra only when no child (type $25 or $26) is left in
;       slots $02-$09. Patra's touch harms Link either way.
; ----------------------------------------------------------------------------
UpdatePatra:
    JSR ControlPatraFlight

    ; Zero the displacement counters; MoveFlyer then leaves in them the pixels moved this frame
    ; (-1, 0 or 1 per axis), which the children add to their positions.
    LDA #$00
    STA Flyer_ObjOffsetX, X
    STA Flyer_ObjOffsetY, X
    JSR MoveFlyer
    LDA #$02                    ; Animation counter reload: a new animation frame every 2 frames.
    JSR AnimateAndDrawCommonObject

    ; Look for a child in slots $09 down to $02 (ObjType+1,Y with Y = 8..1).
    LDY #$08

@LoopChildren:
    LDA ObjType+1, Y
    CMP #$25                    ; Patra child, type 1
    BEQ @CheckLinkCollision
    CMP #$26                    ; Patra child, type 2
    BEQ @CheckLinkCollision
    DEY
    BNE @LoopChildren

    ; No children left: Patra can be hit by weapons (and still harms Link).
    JSR CheckMonsterCollisions
    JSR PlayBossHitCryIfNeeded
    JSR PlayBossDeathCryIfNeeded
    JMP @TryChangeManeuver

@CheckLinkCollision:
    ; A child is left: only test Patra against Link.
    JSR CheckLinkCollision

@TryChangeManeuver:
    ; Flip the maneuver index when the timer of the next slot (ObjTimer+1,X: slot $02 for Patra
    ; in slot $01) is 0 and the angle of the child slot $03 is 0. InitPatra sets that timer to
    ; $FF, but the init of the child in slot $02 (ResetObjMetastateAndTimer) zeroes it again if
    ; it runs after InitPatra, which it does when the slot was empty until InitPatra filled it.
    LDA ObjTimer+1, X
    ORA ObjAngleWhole+3
    BNE @Exit
    LDA Patra_ObjManeuverIndex, X
    EOR #$01
    STA Patra_ObjManeuverIndex, X
    TYA

    ; Reload the timer. Y is not the maneuver index (the TYA above has no effect; A is reloaded):
    ; it is whatever the collision calls left. CheckLinkCollision loads Y with 0, and harming
    ; Link ends with Y = 0 too (the ring loop in Link_BeHarmed). CheckMonsterCollisions returns
    ; Y = 0 when Patra's invincibility timer runs (Y from the loop above), else ObjMetastate,X:
    ; 0 while alive, $10 on the frame Patra is killed (a code byte past the table is read then).
    ; So the reload is $FF. The path on which Link dies was not traced (unverified).
    LDA PatraManeuverTimes, Y
    STA ObjTimer+1, X

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Runs Patra's flight state (Flyer_ObjFlyingState,X) through the table below.
; IN:   X = object slot.
; ----------------------------------------------------------------------------
ControlPatraFlight:
    LDA Flyer_ObjFlyingState, X
    JSR TableJump

; TABLE: Patra's flight state handlers; index = Flyer_ObjFlyingState,X.
ControlPatraFlight_JumpTable:
    .ADDR Flyer_SpeedUp             ; 0: speed +1 per frame until (speed AND $E0) >= FlyerMaxSpeedFrac
    .ADDR Flyer_PatraDecideState    ; 1: choose state 2 or 3
    .ADDR Flyer_Chase               ; 2: turn toward the chase target, $10 frames per turn
    .ADDR Flyer_Wander              ; 3: turn at random, $10 frames per turn

; ----------------------------------------------------------------------------
; Flight state 1: picks chase (Random,X >= $40) or wander (Random,X < $40) with turn counter 8.
; IN:   X = object slot.
; OUT:  Flyer_ObjFlyingState,X = 2 or 3; Flyer_ObjTurns,X = 8.
; NOTE: Counter 8 gives 7 turns $10 frames apart (Flyer_Chase, Flyer_Wander decrement before
;       turning), then state 1 again. Nothing returns to state 0, so Patra never slows down
;       after its first speed-up.
; ----------------------------------------------------------------------------
Flyer_PatraDecideState:
    LDY #$02                    ; State 2: chase.
    LDA Random, X
    CMP #$40
    BCS :+
    INY                         ; State 3: wander.
:
    TYA
    STA Flyer_ObjFlyingState, X
    LDA #$08
    STA Flyer_ObjTurns, X
    RTS

; TABLE: angle the child in slot $02 must have for a later child to appear above Patra;
; index = child's slot - 3 (slots $03-$09); value = ObjAngleWhole ($00-$1F, $18 = top).
; Slot $02 starts at $18 and counts down, so the children appear in slot order, 4 steps apart.
PatraChildStartAngles:
    .BYTE $14, $10, $0C, $08, $04, $00, $1C

; TABLE: type $25 child: number of high bits of the cosine used for the Y step; index =
; Patra_ObjManeuverIndex of slot $01. Entry 1 is the next byte (PatraChild2RotationBits): 6, 5.
PatraChild1RotationCosineBits:
    .BYTE $06

; TABLE: type $26 child: number of high sine and cosine bits used for both steps; index =
; Patra_ObjManeuverIndex of slot $01. Entry 1 is the next byte: 5, 6.
PatraChild2RotationBits:
    .BYTE $05

; TABLE: type $25 child: number of high bits of the sine used for the X step; index =
; Patra_ObjManeuverIndex of slot $01: 6, 6.
; Each bit more doubles the step: with 6 bits the largest step is 4 pixels per frame, with 5
; bits 2 (q-speed $20 times the sine's top bits, see RotateObjectLocation).
PatraChild1RotationSineBits:
    .BYTE $06, $06

; ----------------------------------------------------------------------------
; Updates a Patra child: state 0 waits for its turn to appear above Patra, state 1 orbits.
; IN:   X = object slot ($02-$09).
; OUT:  on appearing: ObjState,X = 1, ObjAngleWhole,X = $18, position above Patra (slot $01);
;       the child in slot $09 also sets Patra's ObjState to 1.
; NOTE: A child in state 0 is not drawn and tests no collisions.
; ----------------------------------------------------------------------------
UpdatePatraChild:
    ; State 1: go orbit.
    LDA ObjState, X
    BNE PatraChild_State1

    ; State 0: waiting to appear. The child in slot $02 appears at once.
    CPX #$02
    BEQ @Ready

    ; Slots are updated from high to low, so in the first frame the other children see slot
    ; $02 still in state 0 and wait.
    LDA ObjState+2
    BEQ @Exit

    ; Later frames: the child appears when the child in slot $02 is at the angle listed for
    ; this slot (index = slot - 3).
    TXA
    SEC
    SBC #$03
    TAY
    LDA ObjAngleWhole+2
    CMP PatraChildStartAngles, Y
    BNE @Exit

@Ready:
    ; The last child (slot $09) sets Patra's state to 1; from then on the children test
    ; collisions (see PatraChild_State1).
    CPX #$09
    BNE :+
    INC ObjState+1
:
    ; Go to state 1 (orbit).
    INC ObjState, X

    ; Direction $80: no reader of this value was found (unverified).
    LDA #$80
    STA ObjDir, X

    ; Start at angle $18 (top of the circle), straight above Patra.
    LDA #$18
    STA ObjAngleWhole, X
    LDA ObjX+1
    STA ObjX, X

    ; [00] = start radius: $2C pixels for type $25, $18 for type $26.
    LDA #$2C
    LDY ObjType, X
    CPY #$25
    BEQ :+
    LDA #$18
:
    STA $00

    ; Y = Patra's Y - radius.
    LDA ObjY+1
    SEC
    SBC $00
    STA ObjY, X

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Patra child state 1: follows Patra, steps its orbit angle, moves along the orbit, draws, collides.
; IN:   X = object slot ($02-$09).
; OUT:  ObjX,X / ObjY,X; ObjAngleFrac,X / ObjAngleWhole,X; a killed child becomes type $5D.
; NOTE: Weapon and Link collisions are tested only once Patra's ObjState (slot $01) is nonzero.
; ----------------------------------------------------------------------------
PatraChild_State1:
    ; Follow Patra: add the pixels it moved this frame (Patra, slot $01, is updated last, so
    ; this is its movement of the previous frame).
    LDA ObjX, X
    CLC
    ADC Flyer_ObjOffsetX+1
    STA ObjX, X
    LDA ObjY, X
    CLC
    ADC Flyer_ObjOffsetY+1
    STA ObjY, X

    ; [0B] = high byte of the angle decrement: 0.
    LDA #$00
    STA $0B

    ; Low byte of the angle decrement: $70 for type $25, $60 for type $26 (in 1/256 of an
    ; angle step per frame; 32 steps make a circle).
    LDA #$70
    LDY ObjType, X
    CPY #$25
    BEQ :+
    LDA #$60
:
    JSR DecreaseObjectAngle

    ; Y = Patra's maneuver index (0 or 1): selects the bit counts below.
    LDY Patra_ObjManeuverIndex+1

    ; Type $25: separate bit counts for the Y step (A, cosine) and the X step (Y, sine):
    ; maneuver 0 = (6, 6), a circle; maneuver 1 = (5, 6), a ring half as tall.
    LDA ObjType, X
    CMP #$25
    BNE @UsePatraChild2Bits
    LDA PatraChild1RotationCosineBits, Y
    PHA
    LDA PatraChild1RotationSineBits, Y
    TAY
    PLA
    JMP @Rotate

@UsePatraChild2Bits:
    ; Type $26: one bit count for both steps: maneuver 0 = 5 (small circle), maneuver 1 = 6
    ; (circle twice as wide).
    LDA PatraChild2RotationBits, Y
    TAY

@Rotate:
    ; Sets ObjX,X itself and returns the new Y coordinate in A.
    JSR RotateObjectLocation
    STA ObjY, X
    JSR PatraChild_Draw

    ; While Patra's state is 0 the last child has not appeared: no collision tests yet.
    LDA ObjState+1
    BEQ @Exit
    JSR CheckMonsterCollisions

    ; A child that was killed (metastate nonzero) becomes a dead dummy ($5D) for its death.
    LDA ObjMetastate, X
    BEQ @Exit
    JSR SetDeadDummyObjType

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Draws a Patra child with sprite palette 2, a new animation frame every 2 frames.
; IN:   X = object slot.
; ----------------------------------------------------------------------------
PatraChild_Draw:
    JSR Anim_SetSpriteDescriptorRedPaletteRow

    ; The routine above returns A = 2; it is passed on as the animation counter reload value.
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor
    LDA ObjAnimFrame, X
    JMP DrawObjectNotMirrored
