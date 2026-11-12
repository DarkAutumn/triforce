; ============================================================================
; FILE:     bank4/moblin-lynel-shooting.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 1950-2165
; SUBJECTS: (none)
; TOPIC:    Moblin, Lynel; shoot timer logic; monster arrow
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0017 UpdateMoblin        code
;   L0025 UpdateLynel         code
;   L0036 _TryShooting        code  Params:
;   L0151 _ShootIfWanted2     code  Params:
;   L0162 L1083E_Exit         code
;   L0165 UpdateMonsterArrow  code
; ============================================================================

UpdateMoblin:
    ; Set turn rate to $A0.
    LDA #$A0
    STA ObjTurnRate, X
    JSR Wanderer_TargetPlayer
    LDY #$5B                    ; Arrow object type
    JMP :+

UpdateLynel:
    JSR UpdateGoriya
    LDY #$57                    ; Sword shot object type
:
    LDA #$20                    ; QSpeed $20 (.5 pixels a frame)

; Params:
; A: q-speed of the monster
; Y: shot object type
;
; [01] holds the q-speed of the monster
_TryShooting:
    STA $01

    ; If the monster is not a blue walking OW monster, and shoot timer = 0,
    LDA ObjType, X
    CMP #$03                    ; Blue moblin
    BEQ @TryShootingNow
    CMP #$01                    ; Blue lynel
    BEQ @TryShootingNow
    CMP #$09                    ; Blue slow octorock
    BEQ @TryShootingNow
    CMP #$0A                    ; Blue fast octorock
    BEQ @TryShootingNow
    LDA ObjShootTimer, X
    BNE @TryShootingNow

    ; and a random value < $F8, then return.
    LDA Random, X
    CMP #$F8
    BCC @Exit

@TryShootingNow:
    ; Summary
    ;
    ; if temporarily invincible then
    ;   shoot timer := 0
    ; elif shoot timer > 0 then
    ;   decrement shoot timer
    ; elif wants to shoot then
    ;   shoot timer := $30
    ;
    ; qspeed := [01]
    ;
    ; if shoot timer <> 0 then
    ;   if shoot timer = $10 and no clock and not stunned then
    ;     shoot
    ;     if succeeded then
    ;       shoot timer := 0
    ;       qspeed := 0
    ;   else
    ;     qspeed := 0
    ;
    ; Push a 0 q-speed, in case the monster is in shooting time.
    LDA $01
    STY $00                     ; Store shot object type in [00] to pass to shooting routine.
    PHA
    LDY #$00

    ; If not temporarily invincible,
    LDA ObjInvincibilityTimer, X
    BNE @SetShootTimer

    ; then Y := shoot timer - 1
    LDY ObjShootTimer, X
    DEY

    ; If Y register < 0, then shoot timer is 0.
    ; So, if "wants to shoot" flag is set, then set shoot timer to $30.
    ; Else go set a non-zero speed and return.
    BPL @SetShootTimer
    LDA ObjWantsToShoot, X
    BEQ @SetSpeed
    LDY #$30

@SetShootTimer:
    ; shoot timer := Y
    TYA
    STA ObjShootTimer, X

    ; If new shoot timer = 0, go set a non-zero speed.
    BEQ @SetSpeed

    ; shoot timer <> 0
    ; If shoot timer <> $10, then go set a zero speed.
    CPY #$10
    BNE @ZeroSpeed

    ; If we have the magic clock, or the monster is stunned, then
    ; go set a zero speed.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE @ZeroSpeed

    ; Shoot.
    LDA $00
    JSR _ShootIfWanted2

    ; If it failed, go set a non-zero speed.
    BCC @SetSpeed

    ; It succeeded, so reset "wants to shoot" flag and the speed.
    LDA #$00
    STA ObjWantsToShoot, X

@ZeroSpeed:
    PLA                         ; Replace the speed that was pushed with 0.
    LDA #$00
    PHA

@SetSpeed:
    ; Set the speed to whatever was determined: 0 or the value passed in [01].
    PLA
    STA ObjQSpeedFrac, X

@Exit:
    RTS

; Params:
; A: shot object type
; X: shooter object index
;
; Returns:
; C: 1 if succeeded
; Y: shot slot
;
_ShootIfWanted2:
    JSR _ShootIfWanted

    ; If succeeded, then set object timer to $80, and decrement [0437][X].
    ;
    ; UNKNOWN: [0437][X] only ever seems to be decremented.
    BCC L1083E_Exit
    LDA #$80
    STA ObjTimer, X
    DEC $0437, X

L1083E_Exit:
    RTS

UpdateMonsterArrow:
    ; Set q-speed $80 (2 pixels a frame).
    LDA #$80
    STA ObjQSpeedFrac, X

    ; If the timer has not expired, then go draw,
    ; and see if the shooter's still alive.
    LDA ObjTimer, X
    BNE @CheckShooter

    ; If the arrow is sparking, then go update the base arrow.
    LDA a:ObjState, X
    AND #$F0
    CMP #$20
    BEQ @UpdateBase

    ; If the arrow is not flying, then
    ; go handle it bouncing off of Link's shield.
    CMP #$10
    BNE @CheckBounce

    ; Set the moving direction in [0F].
    ; Then go update the base arrow.
    ;
    ; UNKNOWN:
    ; This doesn't seem to be needed. Removing the assignment
    ; doesn't affect the movement of the arrow.
    LDA ObjDir, X
    STA $0F
    JMP @UpdateBase

@CheckShooter:
    ; If the shooter is gone, then reset the arrow's object timer;
    ; so that it flies right away.
    ; Begin unverified code 1085C
    LDY ObjRefId, X
    LDA ObjType, Y
    BNE :+
    STA ObjTimer, X
:
    JMP DrawArrow
    ; End unverified code

@UpdateBase:
    JSR UpdateArrowOrBoomerang

    ; If the arrow is no longer flying, then
    ; go handle it sparking or bouncing.
    LDA a:ObjState, X
    AND #$F0
    CMP #$10
    BNE @CheckBounce
    JSR CheckShotLinkCollision

    ; If [06] = 0, then there was no collision with Link that harmed him
    ; Either there was no collision at all, or it bounced off of the shield.
    ; So, return.
    LDA $06
    BEQ L1083E_Exit

    ; Else there was a collision with Link that harmed him.
    ; So, destroy the arrow.
    JMP DestroyCountedMonsterShot

@CheckBounce:
    CMP #$30
    BNE L1083E_Exit
    JMP BounceShot
