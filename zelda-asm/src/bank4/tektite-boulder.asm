; ============================================================================
; FILE:     bank4/tektite-boulder.asm    BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 2167-2590
; SUBJECTS: Boulder, Jumper
; TOPIC:    The boulder spawner (type $1F) that drops up to 3 Boulders (type $20), and the jumper code shared
;           by Tektites (types $0D blue, $0E red) and Boulders: wait on the ground, hop along a parabola.
; SEE ALSO: bank4/zora-ow-inits.asm (Tektite_Init, Boulder_Init, Boulder_InitSpawner: starting direction and timer);
;           bank4/fairy-flyer-common.asm (Dir_TurnObjTowardTarget8, Flyer_BoundByRoom: aiming and room bounds);
;           bank1/monster-collisions.asm (Collision_CheckMonsterVsWeaponsAndLink; Damage_ObjTypeDamageToLink);
;           bank1/link-collision-harm.asm (Link_CheckMonsterTouch, the only collision a Boulder has);
;           bank7/object-update-table.asm (Obj_Update_JumpTable entries $0D, $0E, $1F, $20).
; KEY FACTS:
;   - A jumper has two states: 0 on the ground until ObjTimer runs out, 1 in the air
;     (Jumper_Update).
;   - A jump always has a horizontal part: X changes by exactly 1 pixel per frame in the air, left or right
;     (Jumper_Update@MoveX).
;   - A jump aims 0 pixels (sideways), $20 pixels lower (down diagonal) or $20 pixels higher (up diagonal)
;     than it began (Jumper_TargetYOffsets) and ends on the first frame the jumper is no longer rising and is
;     within 3 pixels of that Y. Y is not snapped to the aim (Jumper_Update@MoveX).
;   - Start speed and gravity: blue Tektite -3 px/frame, red -4, Boulder -2 (Jumper_StartSpeedsWholeY); gravity
;     in 1/256 px/frame per frame from Jumper_YAccelerations; falling speed is capped at 2 px/frame
;     (Jumper_MoveY).
;   - Wait between jumps: blue Tektite $20-$FF frames; red Tektite $00-$7F frames, or $00-$0F when its
;     random byte is >= $A0 (3 in 8); Boulder 0 (Jumper_Update@Landed).
;   - A waiting Tektite alternates frame images 0 and 1 every $10 frames while its timer is above $20
;     and holds frame image 0 for the last $20 frames before the jump; in the air it shows frame image 1
;     (Jumper_AnimateAndCheckCollisions).
;   - The clock or a stun freezes a jumper where it is, in the air too (Jumper_Update@CheckStunned).
;   - Jumpers ignore tiles; only the room bounds turn them around (Flyer_BoundByRoom). A bound crossed on two
;     frames in a row flips the horizontal direction (Jumper_Update@SetUpJump).
;   - Boulders always go down, are not bounded vertically, check only the collision with Link (weapons do
;     not touch them) and are destroyed at Y >= $F0 (Jumper_PointBoulderDownward,
;     Jumper_AnimateAndCheckCollisions@DrawBoulder).
;   - The spawner keeps at most 3 Boulders alive; a new one appears at Y = $40 at a random X in the half
;     of the screen that holds the chase target, then the spawner waits 0-$1F frames (Boulder_UpdateSpawner).
; CONTENTS: (generated; L = line in this file)
;   L0062 Boulder_UpdateSpawner             code  Updates the boulder spawner (type $1F): makes a new Boulder whenever its timer runs out.
;   L0122 Jumper_TargetYOffsets             data  TABLE: Y distance from the start of a jump to its end; index = ObjDir (0-$A); value = signed pixels.
;   L0131 Jumper_YAccelerations             data  TABLE: gravity of a jump; index = Jumper_YAccelerationBaseOffsets[kind] + ObjDir; value = 1/256 px/frame
;   L0143 Jumper_StartSpeedsWholeY          data  TABLE: whole part of the vertical speed at the start of a jump; index = kind (Jumper_GetKind);
;   L0147 Jumper_YAccelerationBaseOffsets   data  TABLE: start of each kind's set in Jumper_YAccelerations; index = kind (Jumper_GetKind).
;   L0166 Jumper_Update                     code  Updates a Tektite (types $0D, $0E) or a Boulder (type $20): wait on the ground, then jump.
;   L0390 Jumper_AnimateAndCheckCollisions  code  Draws a Tektite or a Boulder and checks its collisions; destroys a Boulder at the bottom of the screen.
;   L0444 Jumper_GetKind                    code  Gets the kind of jumper an object is, which indexes the jumper tables.
;   L0454 Jumper_GetKind_Return             code  Shared exit of Jumper_AnimateAndCheckCollisions and Jumper_GetKind.
;   L0461 Jumper_PointBoulderDownward       code  Makes a Boulder (type $20) point down: keeps the horizontal bits of ObjDir and sets down (4).
;   L0483 Jumper_MoveY                      code  Moves a jumper vertically by its speed, then adds gravity to the speed and caps the falling speed.
;   L0520 Jumper_ResetSpeedFracY            code  Clears a jumper's vertical speed fraction; also the tail of Jumper_MoveY.
;   L0525 Jumper_MoveY_Return               code  Shared exit of Jumper_MoveY.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates the boulder spawner (type $1F): makes a new Boulder whenever its timer runs out.
; IN:   X = object slot of the spawner.
; OUT:  X = the spawner's slot again (reloaded from CurObjIndex after a spawn).
; NOTE: At most 3 Boulders (ActiveBoulderCount) exist. With 3 alive the spawner's timer is set to a
;       random 0-$FF frames; with no free slot in $01-$0B it retries every frame. After a spawn it waits
;       (Random + 8) AND $1F frames. The spawner draws nothing and checks no collisions, and does not
;       test the clock. RNG: Random+1 indexed by the new Boulder's slot for X, by the spawner's slot
;       for the timers.
; ----------------------------------------------------------------------------
Boulder_UpdateSpawner:
    ; If the object timer has not expired, then return.
    LDA ObjTimer, X
    BNE @Exit

    ; If 3 boulders are alive, then go wait a random time.
    LDA ActiveBoulderCount
    CMP #$03
    BEQ @RandomizeTimer

    ; If no empty monster slot is found (Z set), then return.
    JSR Obj_FindEmptyMonsterSlot
    BEQ @Exit

    ; Make a Boulder (type $20) in the empty slot, and count it.
    ; It is initialized on its first update (Boulder_Init).
    LDX EmptyMonsterSlot
    INC ActiveBoulderCount
    LDA #$20
    JSR Obj_SetTypeAndClear

    ; Give the boulder a random X coordinate in the same half
    ; of the screen as the chase target.
    LDA Random+1, X
    LDY ChaseTargetX
    CPY #$80
    BCS @RightHalf
    AND #$7F                    ; Chase target X < $80: X := $00-$7F.
    JMP @SetX

@RightHalf:
    ORA #$80                    ; Chase target X >= $80: X := $80-$FF.

@SetX:
    STA ObjX, X

    ; Start at Y = $40.
    LDA #$40
    STA ObjY, X

    ; Set the spawner's timer to ((Random + 8) AND $1F): 0-$1F frames.
    LDX CurObjIndex
    LDA #$08
    CLC
    ADC Random+1, X
    AND #$1F
    STA ObjTimer, X
    RTS

@RandomizeTimer:
    ; The timer is 0 here, so this sets it to the random byte: 0-$FF frames.
    LDA ObjTimer, X
    CLC
    ADC Random+1, X
    STA ObjTimer, X

@Exit:
    RTS

; TABLE: Y distance from the start of a jump to its end; index = ObjDir (0-$A); value = signed pixels.
Jumper_TargetYOffsets:
    .BYTE $00, $00, $00, $00, $00, $20, $20, $00    ; 1 right, 2 left: 0. 5, 6 down diagonals: +$20.
    .BYTE $00, $E0, $E0                             ; 9, $A up diagonals: -$20.

; TABLE: gravity of a jump; index = Jumper_YAccelerationBaseOffsets[kind] + ObjDir; value = 1/256 px/frame
; added to the vertical speed each frame.
; Three sets of 11 bytes would be 33 bytes, but the Boulder's set stops after direction 6: a Boulder
; always points down (5 or 6), so the bytes for directions 7-$A are not stored and the index would run
; into Jumper_StartSpeedsWholeY.
Jumper_YAccelerations:
    ; Blue Tektite, directions 0-7: sideways (1, 2) $40, down diagonals (5, 6) $40.
    .BYTE $00, $40, $40, $00, $00, $40, $40, $00
    ; Blue Tektite 8-$A: up diagonals (9, $A) $30. Then red Tektite 0-4: sideways $80.
    .BYTE $00, $30, $30, $00, $80, $80, $00, $00
    ; Red Tektite 5-$A: down diagonals $80, up diagonals $50. Then Boulder 0-1.
    .BYTE $80, $80, $00, $00, $50, $50, $60, $60
    ; Boulder 2-6: $60 in every direction.
    .BYTE $60, $60, $60, $60, $60

; TABLE: whole part of the vertical speed at the start of a jump; index = kind (Jumper_GetKind);
; value = signed pixels per frame.
Jumper_StartSpeedsWholeY:
    .BYTE $FD, $FC, $FE         ; blue Tektite -3, red Tektite -4, Boulder -2

; TABLE: start of each kind's set in Jumper_YAccelerations; index = kind (Jumper_GetKind).
Jumper_YAccelerationBaseOffsets:
    .BYTE $00, $0B, $16         ; blue Tektite, red Tektite, Boulder

; ----------------------------------------------------------------------------
; Updates a Tektite (types $0D, $0E) or a Boulder (type $20): wait on the ground, then jump.
; IN:   X = object slot.
; NOTE: ObjState 0 = on the ground until ObjTimer is 0; 1 = in the air. At take-off Dir_TurnObjTowardTarget8
;       turns the jumper at most one step of 45 degrees (see its header for when the step is toward the
;       chase target), and a direction without a horizontal part gets left or right toward the chase
;       target's X. The take-off frame does not move the jumper. In the air X changes 1 pixel per frame
;       and Y follows Jumper_MoveY; tiles are ignored.
;       Without hitting the room bounds a blue Tektite's jump lasts 29 frames sideways (rising 24
;       pixels), 45 down-diagonal, 16 up-diagonal; a red Tektite's 20 (rising 20 pixels), 36, 13; a
;       Boulder's bounce 29 frames, then 1 frame on the ground. (trace) Frames in state 1 counted
;       for red Tektites (rooms $1A, $2C, $74), Boulders (room $17) and a red Tektite with its
;       type changed by hand to $0D: the same counts. A sideways jump ends 2 pixels above its
;       start, a down-diagonal one or a bounce 30 below, an up-diagonal one 33 or 34 above.
;       The clock (InvClock) or a stun freezes the jumper in place, in the air too.
; ----------------------------------------------------------------------------
Jumper_Update:
    ; If the monster is shoved, then keep shoving it; it is not drawn in such a frame.
    LDA ObjShoveDir, X
    BEQ @CheckStunned
    ; Never taken (upstream marked this jump as never seen executing). Only a weapon hit sets a
    ; monster's ObjShoveDir (Shove_Begin). A Boulder is not tested against weapons, and every
    ; weapon that shoves deals at least $10, a Tektite's whole HP, so the hit Tektite is in its
    ; death sparkle and this update no longer runs; the boomerang (damage 0) does not shove.
    JMP Shove_Update

@DrawAndCheckCollisions:
    JMP Jumper_AnimateAndCheckCollisions

@CheckStunned:
    ; If Link has the clock, or the monster is stunned;
    ; then only animate, draw, and check collisions.
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
    ; First, turn at most 1 step (45 degrees); see Dir_TurnObjTowardTarget8 for the rule.
    JSR Dir_TurnObjTowardTarget8

    ; If the facing direction has no horizontal component, then
    ; add the horizontal direction toward the chase target:
    ; 2 (left) if the target's X < the jumper's X, else 1 (right).
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
    ; 1. flip the horizontal component of the facing direction
    ;    (left <-> right)
    ; 2. reset the reversal count to 0.
    ;
    ; This prevents a jumper from getting stuck in a corner.
    ;
    ; For example, if a tektite's direction is down and right,
    ; and it jumps into the top right corner of the room:
    ; 1. The right room boundary is hit. So the direction is
    ;    reversed to up and left (Flyer_BoundByRoom).
    ; 2. The next frame, the top room boundary is hit.
    ;    The direction is reversed to down and right.
    ;
    ; Without the flip this would repeat without end.
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
    ; of the jump (0, +$20 or -$20).
    ;
    ; Add it to the object's Y, and store this destination Y.
    LDY ObjDir, X
    LDA ObjY, X
    CLC
    ADC Jumper_TargetYOffsets, Y
    STA Jumper_ObjTargetY, X

    ; Set the starting speed, depending on the kind of jumper.
    JSR Jumper_GetKind
    LDA Jumper_StartSpeedsWholeY, Y
    STA Jumper_ObjSpeedWholeY, X

    ; Reset the vertical speed fraction.
    ; Then go animate, draw, and check collisions.
    JSR Jumper_ResetSpeedFracY

@Animate:
    JMP Jumper_AnimateAndCheckCollisions

@State1:
    ; State 1. Jumping.
    ;
    ; Test the room bounds: horizontal for all, vertical for Tektites only.
    ; When a bound is crossed, Flyer_BoundByRoom reverses ObjDir and returns [0F] = 0.
    JSR Flyer_BoundByRoom

    ; If the object was blocked at the room boundary, then
    ; increase the reversal count, and go set up a new jump from here
    ; in the reversed direction. The object does not move this frame.
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
    LDA Jumper_YAccelerationBaseOffsets, Y

    ; Get the acceleration by indexing with (base offset + direction).
    ; The Boulder's set is short (see Jumper_YAccelerations); its direction is 5 or 6 here.
    ; The addition leaves carry clear, which Jumper_MoveY relies on.
    CLC
    ADC ObjDir, X
    TAY
    LDA Jumper_YAccelerations, Y
    LDY #$02                    ; Falling speed cap: 2 pixels per frame.
    JSR Jumper_MoveY

@MoveX:
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

    ; If still rising (negative whole speed), then
    ; go animate, draw, and check collisions.
    LDA Jumper_ObjSpeedWholeY, X
    BMI Jumper_AnimateAndCheckCollisions

    ; At the top or falling. We have to consider the target Y.
    ;
    ; If the absolute vertical distance to the target Y >= 3,
    ; then go animate, draw, and check collisions.
    LDA ObjY, X
    SEC
    SBC Jumper_ObjTargetY, X
    JSR Math_Abs
    CMP #$03
    BCS Jumper_AnimateAndCheckCollisions

@Landed:
    ; Distance < 3. Go to state 0. Obj_ResetState returns A = 0.
    ;
    ; If the object is a boulder, then go set the timer to 0:
    ; it bounces again on the next frame.
    JSR Obj_ResetState
    LDY ObjType, X
    CPY #$20
    BEQ @SetTimerAndAnimate

    ; Else calculate the wait until the next jump.
    ;
    ; t := (Random + $10) AND $FF
    ; if t < $20 then
    ;   t := t - $40            ($C0-$DF)
    LDA #$10
    CLC
    ADC Random+1, X
    CMP #$20
    BCS :+
    SEC
    SBC #$40
:
    ; If the object is a blue tektite, then go use this t ($20-$FF frames).
    LDY ObjType, X
    CPY #$0D
    BEQ @SetTimerAndAnimate

    ; Red tektite: t := t AND $7F
    AND #$7F

    ; If Random < $A0 (5 in 8), then go use this t ($00-$7F frames).
    LDY Random+1, X
    CPY #$A0
    BCC @SetTimerAndAnimate

    ; Else t := t AND $F ($00-$0F frames): a quick second jump.
    AND #$0F

@SetTimerAndAnimate:
    STA ObjTimer, X

; ----------------------------------------------------------------------------
; Draws a Tektite or a Boulder and checks its collisions; destroys a Boulder at the bottom of the screen.
; IN:   X = object slot.
; NOTE: Tektite: collisions with Link and his weapons (Collision_CheckMonsterVsWeaponsAndLink). Frame image 1 in the
;       air, 0 during the last $20 frames on the ground, and alternating 0/1 every $10 frames before that.
;       Boulder: 2 frame images alternating every 6 frames; only the collision with Link is checked, so
;       no weapon touches it. A Boulder at Y >= $F0 is destroyed and ActiveBoulderCount decremented.
; ----------------------------------------------------------------------------
Jumper_AnimateAndCheckCollisions:
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; [0D] := 0 (A is 0 here). The value is never read: both draw tails below go through
    ; Anim_DrawObjectWithAnim, which stores the frame image in [0D] before anything reads it.
    STA $0D

    ; If the monster is a boulder, then go animate, draw, check collision.
    LDA ObjType, X
    CMP #$20
    BEQ @DrawBoulder

    ; Draw a tektite.
    ;
    ; If state = 1 (in the air), then go draw frame image 1 (A = state), and check collisions.
    LDA ObjState, X
    BNE @DrawTektite

    ; If object timer < $21, go draw frame image 0 (A = state = 0), and check collisions.
    LDY ObjTimer, X
    CPY #$21
    BCC @DrawTektite

    ; Else advance the animation counter ($10 frames per animation frame), draw, and check collisions.
    ;
    ; The animation frame is the frame image.
    LDA #$10
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor
    LDA ObjAnimFrame, X

@DrawTektite:
    JMP Obj_DrawMirroredAndCheckCollisions

@DrawBoulder:
    ; Draw a boulder: 6 frames per animation frame.
    LDA #$06
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor
    LDA ObjAnimFrame, X         ; The animation frame is the frame image.
    JSR Obj_DrawNotMirroredAndCheckLinkCollision

    ; If the boulder's Y < $F0, return.
    LDA ObjY, X
    CMP #$F0
    BCC Jumper_GetKind_Return

    ; Else the boulder has reached the bottom of the screen: destroy it.
    DEC ActiveBoulderCount
    JMP Obj_DestroyMonster

; ----------------------------------------------------------------------------
; Gets the kind of jumper an object is, which indexes the jumper tables.
; IN:   X = object slot.
; OUT:  Y = 0 blue Tektite (type $0D), 1 red Tektite (type $0E), 2 anything else (Boulder, type $20).
; ----------------------------------------------------------------------------
Jumper_GetKind:
    LDA ObjType, X
    SEC
    SBC #$0D
    TAY
    CPY #$02
    BCC Jumper_GetKind_Return
    LDY #$02

; Shared exit of Jumper_AnimateAndCheckCollisions and Jumper_GetKind.
Jumper_GetKind_Return:
    RTS

; ----------------------------------------------------------------------------
; Makes a Boulder (type $20) point down: keeps the horizontal bits of ObjDir and sets down (4).
; IN:   X = object slot. Other object types are left alone.
; ----------------------------------------------------------------------------
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

; ----------------------------------------------------------------------------
; Moves a jumper vertically by its speed, then adds gravity to the speed and caps the falling speed.
; IN:   X = object slot; A = acceleration (1/256 px/frame per frame); Y = speed cap, whole pixels per
;       frame; carry clear (the first ADC has no CLC; the only caller arrives with carry clear).
; OUT:  ObjY,X moved by the whole speed; Jumper_ObjSpeedWholeY,X and Jumper_ObjSpeedFracY,X updated.
; NOTE: Only the whole part of the speed moves the object; the fraction only accumulates gravity. The
;       cap is applied when the whole speed has reached the cap and the fraction is >= $80: the speed is
;       then set to exactly the cap and the fraction to 0, so the whole speed never passes the cap.
;       [00] and [02] are used as temporaries.
; ----------------------------------------------------------------------------
Jumper_MoveY:
    STA $00                     ; [00] holds the acceleration
    STY $02                     ; [02] holds the speed cap (whole pixels per frame)

    ; Add the whole speed to the Y coordinate.
    LDA Jumper_ObjSpeedWholeY, X
    ADC ObjY, X
    STA ObjY, X

    ; Add the acceleration to the speed fraction.
    LDA Jumper_ObjSpeedFracY, X
    CLC
    ADC $00
    STA Jumper_ObjSpeedFracY, X

    ; Carry over to the whole speed.
    LDA Jumper_ObjSpeedWholeY, X
    ADC #$00
    STA Jumper_ObjSpeedWholeY, X

    ; Signed comparison:
    ; If the whole speed < the cap, then return.
    ;
    ; This is not a general purpose signed comparison (it tests only the sign
    ; of the difference), but it works because both values are near 0.
    CMP $02
    BMI Jumper_MoveY_Return

    ; At the cap or above. Once the fraction reaches $80, set the speed to exactly the cap and
    ; (falling into Jumper_ResetSpeedFracY) clear the fraction.
    LDA Jumper_ObjSpeedFracY, X
    CMP #$80
    BCC Jumper_MoveY_Return
    LDA $02
    STA Jumper_ObjSpeedWholeY, X

; Clears a jumper's vertical speed fraction; also the tail of Jumper_MoveY.
Jumper_ResetSpeedFracY:
    LDA #$00
    STA Jumper_ObjSpeedFracY, X

; Shared exit of Jumper_MoveY.
Jumper_MoveY_Return:
    RTS
