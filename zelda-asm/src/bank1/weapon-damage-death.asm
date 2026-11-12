; ============================================================================
; FILE:     bank1/weapon-damage-death.asm BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 5830-6035
; SUBJECTS: (none)
; TOPIC:    Boomerang stun, weapon hits, DealDamage, monster death, help-drop counters
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0028 CheckMonsterBoomerangOrFoodCollision  code  Params:
;   L0071 CheckMonsterWeaponCollision           code  Params:
;   L0114 HandleMonsterWeaponCollision          code
;   L0165 DealDamage                            code  TRIFORCE: Subtracts [$07] from ObjHP[X]. Since both HP and damage are multiples of $10,
;   L0180 L74D8_Exit                            code
;   L0183 L_PlayParrySoundForDamageType         code
;   L0192 HandleMonsterDied                     code
;   L0221 ResetShoveInfoAndInvincibilityTimer   code  Params:
; ============================================================================

; Params:
; X: monster slot
; Y: weapon slot
; [02]: monster mid X
; [03]: monster mid Y
;
; Returns:
; [00]: weapon slot
;
CheckMonsterBoomerangOrFoodCollision:
    ; If the weapon slot holds food (high bit of state is set), then return.
    LDA a:ObjState, Y
    ASL
    BCS Link_BeHarmed_Return
    STY $00                     ; [00] holds the weapon slot

    ; Set boomerang damage type (2) in [09].
    LDA #$02
    STA $09

    ; Set collision threshold $A in both axes.
    LDA #$0A
    STA $0D
    STA $0E

    ; The boomerang's middle X is 4 pixels to the right. Store in [04].
    LDA a:ObjX, Y
    CLC
    ADC #$04
    STA $04

    ; The boomerang's middle Y is 8 pixels down. Store in [05].
    LDA a:ObjY, Y
    CLC
    ADC #$08

; Params:
; A: weapon object mid Y
; X: monster object slot
; Y: weapon object slot
; [00]: weapon object slot
; [02]: monster object mid X
; [03]: monster object mid Y
; [04]: weapon object mid X
; [07]: damage points
; [09]: damage type
; [0D]: horizontal threshold
; [0E]: vertical threshold
;
; Returns:
; [06]: 1 if objects collide
;
CheckMonsterWeaponCollision:
    STA $05

    ; Reset [06] for no collision.
    LDA #$00
    STA $06

    ; If the weapon's not active (state = 0), then return.
    LDY $00
    LDA a:ObjState, Y
    BEQ L74D8_Exit

    ; If the objects do not collide, then return.
    JSR DoObjectsCollideWithThresholds
    BEQ L74D8_Exit

    ; If the weapon is a boomerang ...
    CPY #$0F
    BNE HandleMonsterWeaponCollision

    ; ... and the monster is invincible to it,
    ; then play the parry sound.
    LDA ObjInvincibilityMask, X
    AND $09
    BEQ :+
    JSR PlayParryTune
:
    ; Set the boomerang state to return fast to the thrower.
    LDA #$50
    STA a:ObjState, Y

    ; If the monster is invincible to the boomerang, then return.
    LDA ObjInvincibilityMask, X
    AND $09
    BNE L74D8_Exit

    ; Still the boomerang.
    ; Reset [07] damage points, and set the monster's stun timer to $10 ($A0 frames).
    LDA #$00
    STA $07
    LDA #$10
    STA ObjStunTimer, X

HandleMonsterWeaponCollision:
    ; If the monster is invincible to this damage type, then
    ; go play parry sound if needed, and return.
    LDA ObjInvincibilityMask, X
    AND $09
    BNE L_PlayParrySoundForDamageType

    ; If the monster is Blue Gohma or Red Gohma, then
    ; let it handle the collision.
    LDA ObjType, X
    CMP #$33
    BEQ :+
    CMP #$34
    BNE @CheckZolVire
:
    JMP Gohma_HandleWeaponCollision

@CheckZolVire:
    ; If the monster is Zol or Vire, then
    ;   If not hit by the boomerang, then
    ;     Set monster's direction to the weapon's
    ;   Go deal damage
    CMP #$13
    BEQ :+
    CMP #$12
    BNE @CheckDarknut
:
    CPY #$0F
    BEQ DealDamage
    LDA a:ObjDir, Y
    STA ObjDir, X
    JMP DealDamage

@CheckDarknut:
    ; If the monster is a red or blue darknut, and it and the weapon
    ; are facing opposite directions, then only play the parry sound.
    CMP #$0B
    BEQ :+
    CMP #$0C
    BNE DealDamage
:
    LDA a:ObjDir, Y
    ORA ObjDir, X
    CMP #$0C
    BEQ L_PlayParrySoundForDamageType
    CMP #$03
    BEQ L_PlayParrySoundForDamageType

; TRIFORCE: Subtracts [$07] from ObjHP[X]. Since both HP and damage are multiples of $10,
; TRIFORCE: the low nibble of ObjHP is always 0. Python reads health as ObjHP >> 4.
; TRIFORCE: Death occurs when HP < damage (BCC) or HP - damage == 0 (BEQ).
DealDamage:
    ; Play the "harmed" sound.
    LDA #$02
    STA Tune0Request

    ; Subtract the damage points from HP; if damage points >= HP,
    ; then go handle the monster dying.
    LDA ObjHP, X
    CMP $07                     ; [07] damage points
    BCC HandleMonsterDied
    SEC
    SBC $07                     ; [07] damage points
    STA ObjHP, X
    BEQ HandleMonsterDied

L74D8_Exit:
    RTS

L_PlayParrySoundForDamageType:
    ; Play the parry sound for all but fire and bomb damage types.
    LDA $09
    CMP #$20                    ; Fire damage type
    BEQ L74D8_Exit
    CMP #$08                    ; Bomb damage type
    BEQ L74D8_Exit
    JMP PlayParryTune

HandleMonsterDied:
    ; Handle the monster dying.
    INC WorldKillCount

    ; Increase the help drop counter, if not at max ($A).
    LDA HelpDropCount
    CMP #$0A
    BCS @SetDyingMetastate
    INC HelpDropCount

    ; If the help drop counter has reached the max, and damage type = bomb (8);
    ; then set [51] HelpDropValue in order to drop a bomb next time.
    LDA HelpDropCount
    CMP #$0A
    BNE @SetDyingMetastate
    LDA $09
    CMP #$08
    BNE @SetDyingMetastate
    INC HelpDropValue

@SetDyingMetastate:
    ; Set the dying metastate, and reset some object info.
    JSR UpdateDeadDummy
    LDA #$00
    STA ObjStunTimer, X

; Params:
; A: 0
;
ResetShoveInfoAndInvincibilityTimer:
    JSR SetShoveInfoWith0
    STA ObjInvincibilityTimer, X
    RTS
