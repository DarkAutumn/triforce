; ============================================================================
; FILE:     bank4/octorok.asm            BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 2967-3068
; SUBJECTS: Obj, Octorok
; TOPIC:    Octorok ($07-$0A) per-frame update: turn toward Link, spit rocks, pick the sprite frame; plus
;           the shared "draw mirrored, then check collisions" tail.
; SEE ALSO: bank4/zora-ow-inits.asm (Octorok_InitSlowOrGhini, Octorok_InitFast: starting speed and spawn
;           delay); bank4/walker-wanderer-common.asm (Wanderer_MoveAndTurn: movement and turning);
;           bank4/moblin-lynel-shooting.asm (Shooter_RunShootTimer: the shoot timer); bank4/shoot-helpers-candle.asm
;           (Shooter_ShootLimited: the 4-shot limit); bank4/monster-shots-fireballs.asm (the flying rock, type $53);
;           bank1/monster-collisions.asm (Collision_CheckMonsterVsWeaponsAndLink, Damage_ObjTypeDamageToLink).
; KEY FACTS:
;   - Types $07 and $09 are slow, $08 and $0A fast; $07/$08 have 1 HP and $09/$0A 2 HP (Obj_TypeHpPairs).
;     Octorok_Update tests the type for turn rate (>= $09) and for speed ($07, $09).
;   - Turn rate (Wanderer_ObjTurnRate) is $70 for types $07/$08 and $A0 for $09/$0A. At a grid point the
;     Octorok faces the chase target (and wants to shoot) if turn rate >= Random+1,X (113 or 161 of 256
;     values) and the target is less than 9 pixels off on one axis (Octorok_Update, Wanderer_MoveAndTurn).
;   - Walking q-speed handed to Shooter_RunShootTimer: $20 (0.5 px/frame) slow, $40 (1 px/frame) fast. The init
;     routine starts a fast one at $30 (0.75 px/frame); it keeps that until Shooter_RunShootTimer first gets past
;     its random gate and stores the new speed (Octorok_Update).
;   - The shot is a flying rock, object type $53, fired in the facing direction from the Octorok's own
;     position (Octorok_Update, Shooter_RunShootTimer).
;   - Types $09/$0A run the shoot logic every frame; types $07/$08 only on frames where Random,X >= $F8
;     (8 in 256) or while their shoot timer is already running (Shooter_RunShootTimer).
;   - A shot stops the Octorok for $30 frames; the rock leaves when the timer reaches $10, not with
;     the clock or while stunned; being hit cancels it (Shooter_RunShootTimer).
;   - Touching an Octorok or its rock costs $0080 damage (half a heart before rings)
;     (Damage_ObjTypeDamageToLink, types $07-$0A and $53).
;   - Drops: types $07/$08 are drop group 0, $09/$0A group 2 (Drop_Group0Types,
;     Drop_Group2Types).
; CONTENTS: (generated; L = line in this file)
;   L0048 Octorok_Update                      code  Updates an Octorok: walks and turns toward Link, tries to spit a rock, animates, draws, checks hits.
;   L0139 Obj_DrawMirroredAndCheckCollisions  code  Draws the object mirrored, then checks it against Link's weapons and Link (shared tail).
; ============================================================================

; ----------------------------------------------------------------------------
; Updates an Octorok: walks and turns toward Link, tries to spit a rock, animates, draws, checks hits.
; IN:   X = object slot.
; NOTE: Types: $07 slow, $08 fast (1 HP each); $09 slow, $0A fast (2 HP each).
; NOTE: The walking q-speed is only stored when Shooter_RunShootTimer gets past its random gate, which types
;       $07/$08 pass on 8 frames in 256; until then the speed set by the init routine stays ($20 slow,
;       $30 fast). Types $09/$0A store it every frame.
; NOTE: While the shoot timer runs ($30 frames, rock leaves when it reaches $10) the Octorok stands still
;       and, with q-speed 0, does not turn either (Wanderer_MoveAndTurn turns only when q-speed <> 0).
; NOTE: No clock or stun test here: Walker_Move keeps it from moving then, and Shooter_RunShootTimer skips the
;       rock, but the shoot timer still counts.
; ----------------------------------------------------------------------------
Octorok_Update:
    ; Types $09 and $0A (2 HP) use turn rate $A0; types $07 and $08 use $70.
    ; At a grid point the monster tests for the chase target in line when the turn rate >= Random+1,X.
    LDA #$A0
    LDY ObjType, X
    CPY #$09
    BCS :+
    LDA #$70
:
    STA Wanderer_ObjTurnRate, X
    JSR Wanderer_MoveAndTurn   ; Moves, turns, and sets or clears the "wants to shoot" flag.

    ; Slow types ($07, $09) walk at q-speed $20 (0.5 px/frame).
    ; Fast types ($08, $0A) double it to $40 (1 px/frame).
    LDA #$20
    LDY ObjType, X
    CPY #$07
    BEQ @Shoot
    CPY #$09
    BEQ @Shoot
    ASL

@Shoot:
    ; A = walking q-speed, Y = shot type $53 (flying rock).
    ; Shooter_RunShootTimer stores A as the speed, or 0 while the shoot timer runs.
    LDY #$53
    JSR Shooter_RunShootTimer
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Choose the frame image offset for the facing direction: 0 horizontal, 1 up, 2 down.
    LDA ObjDir, X
    TAY
    AND #$0C
    BEQ @CheckHorizontal
    LDA #$02
    CPY #$08
    BNE @SaveOffset

    ; Facing up.
    LDA #$01
    BNE @SaveOffset

@CheckHorizontal:
    ; Facing right: set [0F] to flip the left-facing image horizontally.
    CPY #$01
    BNE :+
    INC $0F
:
    ; Facing left or right uses frame image offset 0.
    LDA #$00

@SaveOffset:
    PHA                         ; Save the frame image offset for the direction.

    ; Count down the animation counter; reload it with 6 frames when it reaches 0.
    DEC ObjAnimCounter, X
    BNE @CalcFinalOffset
    LDA #$06
    STA ObjAnimCounter, X

    ; There are 6 frame images in 2 sets of 3 (left, up, down), one set per animation frame.
    ; ObjAnimFrame holds the base offset of the current set, 0 or 3: A = 6 >> 1 = 3 toggles it.
    LSR
    EOR ObjAnimFrame, X
    STA ObjAnimFrame, X

@CalcFinalOffset:
    ; Frame image = set base (0 or 3) + directional offset (0-2).
    PLA
    CLC
    ADC ObjAnimFrame, X
    PHA

    ; Facing up or down: go draw with Anim_DrawObjectMirrored.
    LDA ObjDir, X
    AND #$0C
    BNE @DrawMirrored

    ; Facing left or right: draw with Anim_DrawObjectNotMirrored ([0F] flips it when facing right).
    PLA
    JSR Anim_DrawObjectNotMirrored
    JMP Collision_CheckMonsterVsWeaponsAndLink

@DrawMirrored:
    PLA                         ; Restore the frame image.

; ----------------------------------------------------------------------------
; Draws the object mirrored, then checks it against Link's weapons and Link (shared tail).
; IN:   A = frame image; X = object slot; [00] = object X; [01] = object Y.
; NOTE: Also the draw tail of the Tektite update.
; ----------------------------------------------------------------------------
Obj_DrawMirroredAndCheckCollisions:
    JSR Anim_DrawObjectMirrored
    JMP Collision_CheckMonsterVsWeaponsAndLink
