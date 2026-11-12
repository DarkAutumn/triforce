; ============================================================================
; FILE:     bank4/walker-wanderer-common.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 184-432
; SUBJECTS: (none)
; TOPIC:    Boss cry helpers, InitMonsterShot, InitWalker, standing fire, common wanderer AI
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0022 PlayBossHitCryIfNeeded      code
;   L0031 PlayBossDeathCryIfNeeded    code
;   L0036 InitMonsterShot             code
;   L0040 _InitMonsterShot_Unknown54  code
;   L0048 Unused_10021                code
;   L0054 InitWalker                  code
;   L0102 UpdateStandingFire          code
;   L0127 UpdateCommonWanderer        code  Params:
;   L0139 Unused_10082                code
;   L0152 Wanderer_TargetPlayer       code
;   L0162 Wanderer_Return             code
; ============================================================================

PlayBossHitCryIfNeeded:
    LDA ObjInvincibilityTimer, X
    CMP #$10
    BNE :+
    LDA #$02
    STA SampleRequest
:
    RTS

PlayBossDeathCryIfNeeded:
    LDA ObjMetastate, X
    BEQ :-
    JMP PlayBossDeathCry

InitMonsterShot:
    LDA #$C0                    ; QSpeed $C0 (3 pixels a frame)
    BNE :+

_InitMonsterShot_Unknown54:
    ; Begin unverified code 10019
    LDA #$E0                    ; QSpeed $E0 (3.5 pixels a frame)
    ; End unverified code
:
    STA ObjQSpeedFrac, X
    JMP ResetObjMetastate

Unused_10021:
    ; Begin unverified code 10021
    LDA #$80
    STA ObjTimer, X
    ; End unverified code

InitWalker:
    ; If the facing direction was already set, then return.
    LDA ObjDir, X
    BNE @Exit

    ; Calculate horizontal distance and direction away from chase target.
    ; Store the distance in [00].
    LDY #$02
    LDA ChaseTargetX
    SEC
    SBC ObjX, X
    BCS :+
    DEY
:
    STA $00

    ; If chase target is to the right of the object, then
    ; store left (2) in [01], else right (1).
    ;
    ; Also, make this the object's facing direction, even though
    ; it will be clobbered shortly.
    STY $01
    STY ObjDir, X

    ; Calculate vertical distance and direction toward the chase target.
    ; Make this vertical direction the facing direction.
    LDY #$04
    LDA ChaseTargetY
    SEC
    SBC ObjY, X
    BCS :+
    LDY #$08
:
    STY ObjDir, X

    ; If vertical difference >= horizontal difference, then
    ; set the facing direction to the horizontal direction in [01].
    ;
    ; Note that an unsigned comparison is used, even though
    ; the values can be considered signed.
    CMP $00
    BCC @Exit
    LDA $01
    STA ObjDir, X

@Exit:
    RTS

UpdateStandingFire:
    JSR CheckLinkCollision
    LDA #$02                    ; Normal sprites and palette row 6 (red)
    JSR Anim_SetSpriteDescriptorAttributes
    LDA #$08                    ; Up
    STA ObjDir, X
    JSR AnimateObjectWalking

    ; If type <> $40, then never use horizontal mirroring. Dead code?
    LDA ObjType, X
    CMP #$40
    BEQ :+
    ; Begin unverified code 10065
    LDA #$00
    STA $0F
    ; End unverified code
:
    LDA #$00                    ; Frame image 0
    JMP DrawObjectNotMirrored

; Params:
; A: turn rate
;
; Set the turn rate. If being shoved, then shove and return.
;
UpdateCommonWanderer:
    STA ObjTurnRate, X
    LDA ObjShoveDir, X
    BEQ :+
    JMP Obj_Shove
:
    ; If we have the magic clock or the monster is stunned, then return.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE Wanderer_Return
    JMP Wanderer_TargetPlayer

Unused_10082:
    ; Begin unverified code 10082
    LDA #$70
    LDY ObjType, X
    CPY #$05
    BEQ :+
    LDA #$A0
:
    STA ObjTurnRate, X
    LDA ObjState, X
    BMI Wanderer_Return
    ; End unverified code

Wanderer_TargetPlayer:
    ; If turn timer <> 0, then decrement it.
    LDA ObjTurnTimer, X
    BEQ :+
    DEC ObjTurnTimer, X
:
    JSR Walker_Move
    LDA ObjShoveDir, X
    BEQ :+

Wanderer_Return:
    RTS
:
    ; If speed = 0, or the object is between squares;
    ; then go set input direction to facing direction.
    LDA ObjQSpeedFrac, X
    BEQ @SetInputDir
    LDA ObjGridOffset, X
    AND #$0F
    BNE @SetInputDir

    ; Set truncated grid offset.
    STA ObjGridOffset, X

    ; If turn rate < a random value, or Link's state = $FF;
    ; then go turn if turn timer has expired.
    LDA ObjTurnRate, X
    CMP Random+1, X
    BCC @TurnIfTime
    LDA ObjState
    CMP #$FF
    BEQ @TurnIfTime

    ; Get the absolute horizontal distance between
    ; the monster and the chase target.
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
    ; Choose a vertical direction toward the chase target.
    ; Then go face in this direction, and flag to shoot.
    LDY #$08
    LDA ChaseTargetY
    CMP ObjY, X
    BCC @SetDirTowardTarget
    LDY #$04
    BNE @SetDirTowardTarget

@CheckVerticalDistance:
    ; Get the absolute vertical distance between
    ; the monster and the chase target.
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
    ; Choose a horizontal direction toward the chase target.
    LDY #$01
    LDA ChaseTargetX
    CMP ObjX, X
    BCS @SetDirTowardTarget
    INY

@SetDirTowardTarget:
    ; Set the monster's facing direction to the chosen direction
    ; toward the chase target.
    STY ObjDir, X

    ; Set turn timer to a random value.
    LDA Random, X
    STA ObjTurnTimer, X

    ; Set "wants to shoot" flag to 1.
    LDA #$01
    STA ObjWantsToShoot, X

    ; Go set input direction to facing direction.
    JMP L_Walker_SetInputDirAndTryShootingBoomerang

@TurnIfTime:
    ; Reset "wants to shoot" flag.
    LDA #$00
    STA ObjWantsToShoot, X

    ; If turn timer <> 0, then go set input direction to facing direction.
    LDA ObjTurnTimer, X
    BNE @SetInputDir

    ; Turn toward the chase target in a direction perpendicular to
    ; facing direction.
    ;
    ; If facing vertically, then go choose a horizontal direction
    ; toward the chase target.
    ;
    ; Else go choose a vertical direction toward the chase target.
    LDA ObjDir, X
    AND #$0C
    BNE @TurnHorizontally
    BEQ @TurnVertically

@SetInputDir:
    JMP L_Walker_SetInputDirAndTryShootingBoomerang
