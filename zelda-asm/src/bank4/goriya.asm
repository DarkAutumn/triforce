; ============================================================================
; FILE:     bank4/goriya.asm             BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 434-624
; SUBJECTS: (none)
; TOPIC:    Goriya update (shared by Lynel, Armos); boomerang throwing
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0013 UpdateGoriya                                 code
;   L0115 L_Walker_SetInputDirAndTryShootingBoomerang  code
; ============================================================================

UpdateGoriya:
    LDA ObjType+1, X

    ; If not an armos and high bit of state is set, then return.
    ; This means the monster is delaying after shooting.
    CMP #$1E
    BEQ :+
    LDA ObjState, X
    BMI @Exit
:
    JSR Walker_Move

    ; If not being shoved, then go handle other activities.
    LDA ObjShoveDir, X
    BEQ @AfterMove

@Exit:
    RTS

@AfterMove:
    ; If speed = 0 or is between squares, then go try shooting.
    LDA ObjQSpeedFrac, X
    BEQ L_Walker_SetInputDirAndTryShootingBoomerang
    LDA ObjGridOffset, X
    AND #$0F
    BNE L_Walker_SetInputDirAndTryShootingBoomerang

    ; Store the truncated grid offset.
    STA ObjGridOffset, X

    ; If Link's state = $FF, go try shooting.
    ;
    ; UNKNOWN: I don't see how Link can get in this state.
    LDA ObjState
    CMP #$FF
    BEQ L_Walker_SetInputDirAndTryShootingBoomerang

    ; Store in [00] the absolute vertical distance between
    ; the chase target and the walker.
    ;
    ; Store in [02] the vertical direction from the walker to
    ; the chase target.
    LDA #$04
    STA $02
    LDA ChaseTargetY
    LDY ObjY, X
    CMP ObjY, X
    BCS :+
    LDA ObjY, X
    LDY ChaseTargetY
    ASL $02
:
    STY $0E
    SEC
    SBC $0E
    STA $00

    ; Store in [01] the absolute horizontal distance between
    ; the chase target and the walker.
    ;
    ; Store in [03] the horizontal direction from the walker to
    ; the chase target.
    LDA #$01
    STA $03
    LDA ChaseTargetX
    LDY ObjX, X
    CMP ObjX, X
    BCS :+
    LDA ObjX, X
    LDY ChaseTargetX
    ASL $03
:
    STY $0E
    SEC
    SBC $0E
    STA $01

    ; Whichever distance is longer determines which combination
    ; of distance and direction will be used.
    ;
    ; If the vertical distance is longer, then use index 0.
    ; Else use index 1 for horizontal distance and direction.
    LDY #$00
    LDA $00
    CMP $01
    BCS :+
    INY
:
    ; Reset the "wants to shoot" flag.
    LDA #$00
    STA ObjWantsToShoot, X

    ; If the distance chosen < $51, then
    ; set the "wants to shoot" flag to 1,
    ; and face in the chosen direction.
    LDA $0000, Y
    CMP #$51
    BCS L_Walker_SetInputDirAndTryShootingBoomerang
    INC ObjWantsToShoot, X
    LDA $0002, Y
    STA ObjDir, X

L_Walker_SetInputDirAndTryShootingBoomerang:
    ; Set the input direction to the facing direction.
    LDA ObjDir, X
    STA ObjInputDir, X

    ; Goriya will throw a boomerang (object type $5C).
    LDA #$5C

    ; If the monster is a blue goriya, go check the timer.
    LDY ObjType, X
    CPY #$05
    BEQ @CheckTimerToShoot

    ; If it's not a red goriya either, then return.
    CPY #$06
    BNE @Exit

    ; If a random value <> $23 nor $77, then return.
    LDA Random, X
    CMP #$23
    BEQ @CheckTimerToShootBoomerang
    CMP #$77
    BNE @Exit

@CheckTimerToShootBoomerang:
    ; Goriya will throw a boomerang (object type $5C).
    LDA #$5C

@CheckTimerToShoot:
    ; If object timer <> 0, return.
    LDY ObjTimer, X
    BNE @Exit

    ; Store in [00] the type of projectile the monster will throw.
    STA $00

    ; If we have the magic clock, or the monster is stunned, then return.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE @Exit

    ; Try to shoot. Return, if it failed.
    LDA $00
    JSR _ShootIfWanted
    BCC @Exit

    ; Set monster's state to $80 for delaying after shooting.
    LDA #$80
    STA ObjState, X

    ; Reset the "wants to shoot" flag.
    LDA #$00
    STA ObjWantsToShoot, X

    ; Have the shot keep track of the monster.
    TXA
    STA ObjRefId, Y

    ; Have the monster keep track of the shot.
    TYA
    STA ObjRefId, X

    ; Set the shot's state to $10, which is the flying state for them.
    LDA #$10
    STA a:ObjState, Y

    ; Fly at q-speed $A0 (2.5 pixels a frame).
    LDA #$A0
    STA ObjQSpeedFrac, Y

    ; For boomerangs, move up to $51 pixels (horizontally or vertically).
    LDA #$51
    STA ObjMovingLimit, Y

    ; Make it autonomous (ready to update on its own).
    LDA #$00
    STA ObjMetastate, Y

    ; The first animation frame lasts 3 screen frames.
    LDA #$03
    STA ObjAnimCounter, Y

    ; Have the monster wait a random amount of time, up to $3F frames.
    LDA Random, X
    AND #$3F
    STA ObjTimer, X

@Exit:
    RTS
