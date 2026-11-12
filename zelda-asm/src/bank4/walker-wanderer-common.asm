; ============================================================================
; FILE:     bank4/walker-wanderer-common.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 184-432
; SUBJECTS: MonsterShot, Sample, StandingFire, Unused, Walker, Wanderer
; TOPIC:    Boss hit and death cries; init of monster shots (q-speed) and of walking monsters (first
;           facing direction); standing fire update; the wanderer AI shared by most walking monsters
;           (move one grid square at a time, turn toward the chase target or perpendicular to it).
; SEE ALSO: bank4/goriya.asm (Walker_SetInputDirAndTryThrowBoomerang, where the wanderer ends);
;           bank7/walker-move.asm (Walker_Move, Obj_MoveByQSpeed); bank7/shove.asm (Shove_Update);
;           bank7/object-init-table.asm (which types use Walker_Init and MonsterShot_Init);
;           bank4/drop-items.asm (Sample_PlayBossDeathCry); bank4/monster-shots-fireballs.asm (shot updates);
;           bank7/mode5-play-update.asm (how ChaseTargetX/Y are chosen).
; KEY FACTS:
;   - A wanderer decides only on a grid point (ObjGridOffset a multiple of $10 pixels) and only when
;     its q-speed is not 0; between points it keeps walking in ObjDir. Wanderer_MoveAndTurn
;   - On a grid point it homes on the chase target when turn rate >= Random+1,X (chance about
;     (rate+1)/256; rate $FF always) and the target is within 8 pixels on one axis; it then faces the
;     target along the other axis. Wanderer_Return@TurnVertically, Wanderer_Return@TurnHorizontally
;   - Otherwise it turns only when Wanderer_ObjTurnTimer is 0, and then perpendicular to its facing,
;     toward the chase target. Every turn reloads the timer from Random,X (0-$FF frames).
;     Wanderer_Return@TurnIfTime, Wanderer_Return@SetDirTowardTarget
;   - Every turn sets Shooter_ObjWantsToShoot to 1; a grid point without a turn clears it.
;     Wanderer_Return@SetDirTowardTarget, Wanderer_Return@TurnIfTime
;   - The target is ChaseTargetX/Y, not Link's position: wanderers follow the decoy point or the food
;     when the chase target is one of those. Wanderer_MoveAndTurn
;   - With the clock (InvClock) or while stunned a wanderer does nothing at all, and its turn timer
;     does not run. Wanderer_Update
;   - A new walker faces vertically toward the chase target, unless the vertical difference (as an
;     unsigned byte) is >= the horizontal one: then it faces horizontally AWAY from it. Walker_Init
;   - Monster shots start at q-speed $C0 (3 pixels a frame); type $54 at $E0 (3.5). The type $54
;     init runs for any type $54 object (Obj_Init_JumpTable), but no instruction in the program
;     loads $54 as an object type to create. MonsterShot_Init, MonsterShot_InitType54
;   - The boss hit cry (sample $02) starts when the boss's invincibility timer is exactly $10, the
;     value a hit sets. Sample_PlayBossHitCryIfHit
;   - Standing fire (type $40) only tests contact with Link; it never looks at Link's weapons.
;     StandingFire_Update
; CONTENTS: (generated; L = line in this file)
;   L0057 Sample_PlayBossHitCryIfHit         code  Starts the boss cry sample on the frame a boss has just been hit.
;   L0070 Sample_PlayBossDeathCryIfDying     code  Plays the boss death cry when the object is dying (ObjMetastate <> 0).
;   L0082 MonsterShot_Init                   code  Init for monster shot types $53 and $55-$5A: q-speed $C0, and the shot updates at once.
;   L0091 MonsterShot_InitType54             code  Init for shot type $54: as MonsterShot_Init, but q-speed $E0 (3.5 pixels a frame).
;   L0098 Unused_Walker_InitWithTimer80      code  Dead code (nothing refers to it): sets the object timer to $80 frames, then falls into Walker_Init.
;   L0113 Walker_Init                        code  Init for walking monsters: chooses the first facing direction, unless one is already set.
;   L0160 StandingFire_Update                code  Standing fire (type $40): hurts Link on contact, animates and draws the flame.
;   L0185 Wanderer_Update                    code  Wanderer update: knockback if shoved; nothing with the clock or while stunned; else move and turn.
;   L0199 Unused_Wanderer_SetTurnRateByType  code  Dead code (nothing refers to it): a wanderer entry with turn rate $70 for type $05, else $A0.
;   L0224 Wanderer_MoveAndTurn               code  Moves a wanderer one frame and, on a grid point, chooses its next facing direction.
;   L0235 Wanderer_Return                    code  Shared exit of the wanderer routines above and below.
; ============================================================================

; ----------------------------------------------------------------------------
; Starts the boss cry sample on the frame a boss has just been hit.
; IN:   X = object slot
; NOTE: A hit sets a monster's ObjInvincibilityTimer to $10 (it then drops by 1 every second frame);
;       the test is for exactly $10. SampleRequest $02 is the boss cry.
; ----------------------------------------------------------------------------
Sample_PlayBossHitCryIfHit:
    LDA ObjInvincibilityTimer, X
    CMP #$10
    BNE :+
    LDA #$02
    STA SampleRequest
:
    RTS

; ----------------------------------------------------------------------------
; Plays the boss death cry when the object is dying (ObjMetastate <> 0).
; IN:   X = object slot
; ----------------------------------------------------------------------------
Sample_PlayBossDeathCryIfDying:
    LDA ObjMetastate, X
    BEQ :-                      ; Not dying: return through the RTS above.
    JMP Sample_PlayBossDeathCry

; ----------------------------------------------------------------------------
; Init for monster shot types $53 and $55-$5A: q-speed $C0, and the shot updates at once.
; IN:   X = object slot
; OUT:  ObjQSpeedFrac,X = $C0 (3 pixels a frame); ObjMetastate,X = 0 (no spawn cloud)
; NOTE: Fireballs ($55, $56) replace this q-speed with their own per-axis speeds when they move
;       (Fireball_MoveOneAxis).
; ----------------------------------------------------------------------------
MonsterShot_Init:
    LDA #$C0                    ; Q-speed $C0 (3 pixels a frame)
    BNE :+                      ; Always taken.

; Init for shot type $54: as MonsterShot_Init, but q-speed $E0 (3.5 pixels a frame).
; Its only reference is the Obj_Init_JumpTable entry of type $54 (that type's update is
; MonsterShot_Update), and type $54 is never created: every shot type handed to Shooter_Shoot or
; Fireball_Shoot is an immediate ($53, $55-$59, $5B, $5C), no room has a monster list id from $53
; to $61, and no monster list contains $54. The same holds for type $5A.
MonsterShot_InitType54:
    LDA #$E0                    ; Q-speed $E0 (3.5 pixels a frame)
:
    STA ObjQSpeedFrac, X
    JMP Obj_ResetMetastate

; Dead code (nothing refers to it): sets the object timer to $80 frames, then falls into Walker_Init.
Unused_Walker_InitWithTimer80:
    LDA #$80
    STA ObjTimer, X

; ----------------------------------------------------------------------------
; Init for walking monsters: chooses the first facing direction, unless one is already set.
; IN:   X = object slot; ChaseTargetX/Y
; OUT:  ObjDir,X; uses [00], [01]
; NOTE: Faces vertically toward the chase target when the vertical difference is smaller than the
;       horizontal one; otherwise horizontally away from the target. The differences are
;       (target - monster) as unsigned bytes, not absolute distances, so a target up or left of the
;       monster counts as a large difference.
;       Init routine of types $01-$06, $12-$14, $16, $17, $2A, $30, and the tail of several other
;       inits (Darknut, Rope, Bubble, Gel $15, Octoroks, Leevers, Ghini).
; ----------------------------------------------------------------------------
Walker_Init:
    ; If the facing direction was already set, then return.
    LDA ObjDir, X
    BNE @Exit

    ; [00] = chase target X - monster X (unsigned byte).
    ; Y = 2 (left) if the target is at the same X or to the right, else 1 (right):
    ; the horizontal direction away from the target.
    LDY #$02
    LDA ChaseTargetX
    SEC
    SBC ObjX, X
    BCS :+
    DEY
:
    STA $00

    ; [01] = that horizontal direction. It is also written to ObjDir, but the vertical
    ; direction below overwrites it at once.
    STY $01
    STY ObjDir, X

    ; Face vertically toward the chase target: down (4) if it is at the same Y or below, else up (8).
    LDY #$04
    LDA ChaseTargetY
    SEC
    SBC ObjY, X
    BCS :+
    LDY #$08
:
    STY ObjDir, X

    ; A = chase target Y - monster Y (unsigned byte). If it is >= the horizontal difference [00],
    ; then face in the horizontal direction [01] instead.
    CMP $00
    BCC @Exit
    LDA $01
    STA ObjDir, X

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Standing fire (type $40): hurts Link on contact, animates and draws the flame.
; IN:   X = object slot
; NOTE: Only Link_CheckMonsterTouch is called: Link's weapons are never tested against it.
; ----------------------------------------------------------------------------
StandingFire_Update:
    JSR Link_CheckMonsterTouch
    LDA #$02                    ; Sprite attributes: sprite palette 2 (palette row 6), no flipping
    JSR Anim_SetSpriteDescriptorAttributes
    LDA #$08                    ; Face up, so that the walking animation flips the image each 6 frames.
    STA ObjDir, X
    JSR Anim_AnimateObjWalking

    ; A type other than $40 would never be flipped horizontally ([0F] = 0). The branch below is
    ; always taken, because only type $40 has this update routine.
    LDA ObjType, X
    CMP #$40
    BEQ :+
    LDA #$00
    STA $0F
:
    LDA #$00                    ; Frame image 0
    JMP Anim_DrawObjectNotMirrored

; ----------------------------------------------------------------------------
; Wanderer update: knockback if shoved; nothing with the clock or while stunned; else move and turn.
; IN:   A = turn rate ($00-$FF; higher = homes on the chase target more often); X = object slot
; OUT:  Wanderer_ObjTurnRate,X = A
; NOTE: Callers pass $40 (Bubble), $80 (Stalfos, Gibdo, Darknut, Like-Like, Vire), $FF (Ghini).
; ----------------------------------------------------------------------------
Wanderer_Update:
    STA Wanderer_ObjTurnRate, X
    LDA ObjShoveDir, X
    BEQ :+
    JMP Shove_Update
:
    ; If Link has the clock or the monster is stunned, then return.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE Wanderer_Return
    JMP Wanderer_MoveAndTurn

; Dead code (nothing refers to it): a wanderer entry with turn rate $70 for type $05, else $A0.
; Returns if ObjState bit 7 is set, else falls into Wanderer_MoveAndTurn.
Unused_Wanderer_SetTurnRateByType:
    LDA #$70
    LDY ObjType, X
    CPY #$05
    BEQ :+
    LDA #$A0
:
    STA Wanderer_ObjTurnRate, X
    LDA ObjState, X
    BMI Wanderer_Return

; ----------------------------------------------------------------------------
; Moves a wanderer one frame and, on a grid point, chooses its next facing direction.
; IN:   X = object slot; Wanderer_ObjTurnRate,X; ChaseTargetX/Y
; OUT:  ObjDir,X, ObjInputDir,X, Wanderer_ObjTurnTimer,X, Shooter_ObjWantsToShoot,X
; NOTE: Entry point of the monsters that set their own turn rate (Octorok, Moblin, Leever, Zol,
;       Dodongo). It does not test the clock or the stun timer itself; Walker_Move does, for moving.
;       Decision on a grid point (grid offset a multiple of $10, q-speed <> 0):
;       1. If turn rate >= Random+1,X and Link's state is not $FF: when the chase target is within
;          8 pixels in X, face it vertically; else when within 8 pixels in Y, face it horizontally.
;       2. Otherwise, when the turn timer is 0, turn perpendicular to the facing direction, toward
;          the chase target.
;       Either turn sets the turn timer to Random,X frames and "wants to shoot" to 1; no turn
;       clears "wants to shoot". Ends in Walker_SetInputDirAndTryThrowBoomerang.
; ----------------------------------------------------------------------------
Wanderer_MoveAndTurn:
    ; If turn timer <> 0, then decrement it.
    LDA Wanderer_ObjTurnTimer, X
    BEQ :+
    DEC Wanderer_ObjTurnTimer, X
:
    JSR Walker_Move
    LDA ObjShoveDir, X
    BEQ :+                      ; Not shoved: go on below.

; Shared exit of the wanderer routines above and below.
Wanderer_Return:
    RTS
:
    ; If q-speed = 0, or the monster is between grid points;
    ; then go set input direction to facing direction.
    LDA ObjQSpeedFrac, X
    BEQ @SetInputDir
    LDA ObjGridOffset, X
    AND #$0F
    BNE @SetInputDir

    ; On a grid point (offset 0, $10 or $F0): reset the grid offset to 0.
    STA ObjGridOffset, X

    ; If turn rate < the random byte, or Link's state = $FF;
    ; then go turn only if the turn timer has expired.
    ; The state test is never true: no code sets bit 7 of Link's state (see Goriya_Update).
    LDA Wanderer_ObjTurnRate, X
    CMP Random+1, X
    BCC @TurnIfTime
    LDA ObjState
    CMP #$FF
    BEQ @TurnIfTime

    ; A = absolute horizontal distance between the monster and the chase target
    ; (the difference is treated as a signed byte).
    LDA ChaseTargetX
    SEC
    SBC ObjX, X
    BPL :+
    EOR #$FF
    CLC
    ADC #$01
:
    ; If distance >= 9, then go check the vertical distance.
    CMP #$09
    BCS @CheckVerticalDistance

@TurnVertically:
    ; Choose the vertical direction toward the chase target:
    ; up (8) if the target is above, else down (4).
    LDY #$08
    LDA ChaseTargetY
    CMP ObjY, X
    BCC @SetDirTowardTarget
    LDY #$04
    BNE @SetDirTowardTarget     ; Always taken.

@CheckVerticalDistance:
    ; A = absolute vertical distance between the monster and the chase target.
    LDA ChaseTargetY
    SEC
    SBC ObjY, X
    BPL :+
    EOR #$FF
    CLC
    ADC #$01
:
    ; If distance >= 9, then go turn if turn timer has expired.
    CMP #$09
    BCS @TurnIfTime

@TurnHorizontally:
    ; Choose the horizontal direction toward the chase target:
    ; right (1) if the target is at the same X or to the right, else left (2).
    LDY #$01
    LDA ChaseTargetX
    CMP ObjX, X
    BCS @SetDirTowardTarget
    INY

@SetDirTowardTarget:
    ; Face in the chosen direction.
    STY ObjDir, X

    ; Set turn timer to a random value (0-$FF frames).
    LDA Random, X
    STA Wanderer_ObjTurnTimer, X

    ; Set "wants to shoot" flag to 1.
    LDA #$01
    STA Shooter_ObjWantsToShoot, X

    ; Go set input direction to facing direction.
    JMP Walker_SetInputDirAndTryThrowBoomerang

@TurnIfTime:
    ; Reset "wants to shoot" flag.
    LDA #$00
    STA Shooter_ObjWantsToShoot, X

    ; If turn timer <> 0, then keep the facing direction.
    LDA Wanderer_ObjTurnTimer, X
    BNE @SetInputDir

    ; Turn toward the chase target in a direction perpendicular to the facing direction:
    ; facing up or down, choose left or right; facing left or right, choose up or down.
    LDA ObjDir, X
    AND #$0C
    BNE @TurnHorizontally
    BEQ @TurnVertically

@SetInputDir:
    JMP Walker_SetInputDirAndTryThrowBoomerang
