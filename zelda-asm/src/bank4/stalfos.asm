; ============================================================================
; FILE:     bank4/stalfos.asm            BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 4670-4769
; SUBJECTS: (none)
; TOPIC:    Stalfos; second-quest sword shot
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0012 UpdateStalfos  code
; ============================================================================

UpdateStalfos:
    LDA #$80                    ; Turn rate $80
    JSR UpdateCommonWanderer
    JSR CheckMonsterCollisions
    LDA #$08                    ; Animation counter 8
    JSR AnimateAndDrawCommonObject
    LDA #$20                    ; QSpeed $20
    STA $01                     ; [01] holds qspeed $20

    ; If in the first quest, return. Stalfos can't shoot in this quest.
    LDY CurSaveSlot
    LDA QuestNumbers, Y
    BEQ @Exit

    ; Second quest.
    ;
    ; If shoot timer = 0 and Random < $F8, then return.
    LDA ObjShootTimer, X
    BNE @PrepareToShoot
    LDA Random, X
    CMP #$F8
    BCC @Exit

@PrepareToShoot:
    LDA $01                     ; Save the qspeed.
    PHA

    ; If temporarily invincible, then go reset shoot timer.
    LDY #$00
    LDA ObjInvincibilityTimer, X
    BNE @SetShootTimer

    ; If shoot timer >= 1, then decrement and go set shoot timer.
    LDY ObjShootTimer, X
    DEY
    BPL @SetShootTimer

    ; If stalfos does not want to shoot, then go set the qspeed.
    LDA Shooter_ObjWantsToShoot, X
    BEQ @SetSpeed

    ; Else not temporarily invincible, and shoot timer = 0,
    ; and wants to shoot.
    ;
    ; Set shoot timer to $30.
    LDY #$30

@SetShootTimer:
    TYA
    STA ObjShootTimer, X

    ; If shoot timer = 0, then we're not shooting.
    ; Go set the qspeed.
    BEQ @SetSpeed

    ; If shoot timer <> $10, or we have the magic clock, or stunned;
    ; then go reset qspeed.
    CPY #$10
    BNE @ZeroSpeed
    LDA InvClock
    ORA ObjStunTimer, X
    BNE @ZeroSpeed

    ; Try to shoot a sword shot.
    LDA #$57
    STA $00
    JSR _ShootIfWanted

    ; If failed, then go set the qspeed.
    BCC @CheckResult

    ; Else set timer to $80, and decrement [0437][X].
    ;
    ; UNKNOWN: [0437] only ever seems to be decremented.
    ; Also, Stalfos doesn't care about the timer. This block seems
    ; copied and pasted from 8832 ("ShootIfWanted2").
    LDA #$80
    STA ObjTimer, X
    DEC $0437, X

@CheckResult:
    ; We jump here if shooting failed. Jump again to set the qspeed.
    BCC @SetSpeed

    ; Shooting succeeded.
    ; Reset "wants to shoot" flag and the speed.
    LDA #$00
    STA Shooter_ObjWantsToShoot, X

@ZeroSpeed:
    PLA                         ; Replace qspeed on the stack with 0.
    LDA #$00
    PHA

@SetSpeed:
    PLA                         ; Pop and set the qspeed (0 or the speed for stalfos).
    STA ObjQSpeedFrac, X

@Exit:
    RTS
