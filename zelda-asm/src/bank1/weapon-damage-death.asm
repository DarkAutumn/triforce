; ============================================================================
; FILE:     bank1/weapon-damage-death.asm BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 5830-6035
; SUBJECTS: (none)
; TOPIC:    A weapon of Link's touching a monster: the boomerang's stun, the invincibility mask, the Gohma,
;           Zol/Vire and Darknut special cases, HP subtraction, monster death and the kill counters.
; SEE ALSO: bank1/monster-collisions.asm (CheckMonsterCollisions: the order the weapon slots are tested);
;           bank1/sword-bomb-arrow-collision.asm (the other weapons' entry routines, PlayParryTune);
;           bank1/object-overlap-test.asm (DoObjectsCollideWithThresholds); bank1/shove-knockback.asm
;           (BeginShove); bank4/gohma.asm (Gohma_HandleWeaponCollision); bank4/drop-items.asm (how the kill
;           counters choose a drop); bank1/link-collision-harm.asm (where the counters are zeroed).
; KEY FACTS:
;   - HP is ObjHP (HP x $10); a hit subtracts the damage points in [07]; the monster dies when HP < damage
;     or the result is 0 (DealDamage).
;   - Damage types in [09], tested against ObjInvincibilityMask: 1 sword/rod/sword shot, 2 boomerang,
;     4 arrow, 8 bomb, $10 magic shot, $20 fire (HandleMonsterWeaponCollision).
;   - The boomerang (slot $0F) does 0 damage and sets the monster's stun timer to $10; a monster whose HP
;     is already 0 dies from it. It always turns back (state $50) on touching a monster
;     (CheckMonsterWeaponCollision, DealDamage).
;   - Food in slot $0F (state bit 7) never collides (CheckMonsterBoomerangOrFoodCollision).
;   - A Darknut ($0B, $0C) takes no damage when the weapon's direction and its own are exact opposites
;     (HandleMonsterWeaponCollision@CheckDarknut).
;   - A Zol ($13) or Vire ($12) hit by anything but the boomerang takes the weapon's direction
;     (HandleMonsterWeaponCollision@CheckZolVire).
;   - Each death adds 1 to WorldKillCount (no cap) and to HelpDropCount (stops at $0A); IsHelpDropBomb is
;     set only when the kill that brings HelpDropCount to exactly $0A is a bomb kill (HandleMonsterDied).
;   - Death sets ObjMetastate to $10 and clears the stun timer, knockback and invincibility timer;
;     WorldKillCycle is not changed here (HandleMonsterDied, ResetShoveInfoAndInvincibilityTimer).
;   - An immune hit plays the parry tune, except for bomb and fire (L_PlayParrySoundForDamageType).
; CONTENTS: (generated; L = line in this file)
;   L0048 CheckMonsterBoomerangOrFoodCollision  code  Tests Link's boomerang (not food) in slot Y against the monster in slot X; a hit stuns the monster.
;   L0082 CheckMonsterWeaponCollision           code  Tests a weapon point against a monster and, on overlap, applies the hit (stun, damage or parry).
;   L0127 HandleMonsterWeaponCollision          code  Applies a weapon hit to a monster: immunity mask, Gohma, Zol/Vire turning, Darknut parry, then damage.
;   L0181 DealDamage                            code  Subtracts the damage points in [07] from the monster's HP and starts its death when none is left.
;   L0195 L74D8_Exit                            code  Shared exit of the weapon collision routines in this file.
;   L0199 L_PlayParrySoundForDamageType         code  Plays the parry tune for a hit that did no damage, unless the damage type in [09] is bomb or fire.
;   L0216 HandleMonsterDied                     code  Counts a kill in the drop counters and puts the monster into the dying metastate.
;   L0245 ResetShoveInfoAndInvincibilityTimer   code  Clears the slot's knockback direction, knockback distance and invincibility timer.
; ============================================================================

; ----------------------------------------------------------------------------
; Tests Link's boomerang (not food) in slot Y against the monster in slot X; a hit stuns the monster.
; IN:   X = monster slot; Y = weapon slot ($0F); [02],[03] = monster's middle.
; OUT:  [00] = weapon slot; [06] = 1 when they overlap (left unchanged when the slot holds food).
; NOTE: Box: the boomerang's point (X+4, Y+8) within less than $0A pixels of the monster's middle on both
;       axes. Falls into CheckMonsterWeaponCollision.
; ----------------------------------------------------------------------------
CheckMonsterBoomerangOrFoodCollision:
    ; Food has bit 7 of its state set: it never collides. Return.
    LDA a:ObjState, Y
    ASL
    BCS Link_BeHarmed_Return
    STY $00                     ; [00] := weapon slot

    LDA #$02
    STA $09                     ; [09] := damage type 2 (boomerang)

    LDA #$0A
    STA $0D                     ; [0D] := horizontal threshold $0A
    STA $0E                     ; [0E] := vertical threshold $0A

    ; [04] := boomerang X + 4; A := boomerang Y + 8 (stored in [05] below).
    LDA a:ObjX, Y
    CLC
    ADC #$04
    STA $04

    LDA a:ObjY, Y
    CLC
    ADC #$08

; ----------------------------------------------------------------------------
; Tests a weapon point against a monster and, on overlap, applies the hit (stun, damage or parry).
; IN:   A = weapon point Y; X = monster slot; [00] = weapon slot; [02],[03] = monster's middle;
;       [04] = weapon point X; [07] = damage points; [09] = damage type; [0D],[0E] = horizontal and
;       vertical thresholds.
; OUT:  [06] = 1 when the weapon is active and overlaps the monster (also when the hit was parried or the
;       monster is immune), else 0; Y = weapon slot.
; NOTE: Boomerang (slot $0F): it is set to return fast (state $50) on any overlap; if the monster is not
;       immune to type 2 its stun timer becomes $10 and the damage is 0.
; ----------------------------------------------------------------------------
CheckMonsterWeaponCollision:
    STA $05

    LDA #$00
    STA $06                     ; [06] := 0: no collision

    ; Return if the weapon slot is inactive (state 0).
    LDY $00
    LDA a:ObjState, Y
    BEQ L74D8_Exit

    ; Return if the two points are not within the thresholds. Sets [06] = 1 when they are.
    JSR DoObjectsCollideWithThresholds
    BEQ L74D8_Exit

    ; Every weapon but the boomerang goes straight to the hit handling.
    CPY #$0F
    BNE HandleMonsterWeaponCollision

    ; Boomerang. If the monster is immune to it, play the parry tune.
    LDA ObjInvincibilityMask, X
    AND $09
    BEQ :+
    JSR PlayParryTune
:
    ; Immune or not, the boomerang turns back: state $50 (returning fast).
    LDA #$50
    STA a:ObjState, Y

    ; If the monster is immune, that is all.
    LDA ObjInvincibilityMask, X
    AND $09
    BNE L74D8_Exit

    ; Damage := 0, and stun the monster: stun timer := $10 (counted in 10-frame units).
    LDA #$00
    STA $07
    LDA #$10
    STA ObjStunTimer, X

; ----------------------------------------------------------------------------
; Applies a weapon hit to a monster: immunity mask, Gohma, Zol/Vire turning, Darknut parry, then damage.
; IN:   X = monster slot; Y = weapon slot; [07] = damage points; [09] = damage type.
; NOTE: Entered directly by the bomb and fire test; the other weapons fall in from above.
; ----------------------------------------------------------------------------
HandleMonsterWeaponCollision:
    ; If the monster's invincibility mask holds this damage type, no damage: play the parry tune
    ; (not for bomb or fire) and return.
    LDA ObjInvincibilityMask, X
    AND $09
    BNE L_PlayParrySoundForDamageType

    ; Gohma ($33, $34) has its own handler (only an arrow in the open eye hurts it).
    LDA ObjType, X
    CMP #$33
    BEQ :+
    CMP #$34
    BNE @CheckZolVire
:
    JMP Gohma_HandleWeaponCollision

@CheckZolVire:
    ; Zol ($13) or Vire ($12): unless the weapon is the boomerang, copy the weapon's direction into
    ; the monster's direction. Then deal damage. (The Zol's update reads its direction when it splits.)
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
    ; Darknut ($0B red, $0C blue): if the weapon's direction and the Darknut's direction together are
    ; exactly up + down ($0C) or left + right ($03), the hit is blocked: parry tune (not for bomb or
    ; fire) and no damage. A hit from the side or from behind goes through.
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

; ----------------------------------------------------------------------------
; Subtracts the damage points in [07] from the monster's HP and starts its death when none is left.
; IN:   X = monster slot; [07] = damage points; [09] = damage type (read only if the monster dies).
; NOTE: ObjHP holds HP x $10 and every damage value passed in by the weapon routines is a multiple of
;       $10: sword $10/$20/$40, rod $20, magic shot $20, arrow $20/$40, bomb $40, fire $10, boomerang 0.
;       The monster dies when HP < damage or HP - damage = 0, so damage 0 kills a monster whose HP is 0.
;       No immunity test here: callers that jump in directly bypass the invincibility mask.
; ----------------------------------------------------------------------------
DealDamage:
    ; Request the "monster harmed" tune.
    LDA #$02
    STA Tune0Request

    LDA ObjHP, X
    CMP $07                     ; HP < damage: dies (HP is left as it was).
    BCC HandleMonsterDied
    SEC
    SBC $07
    STA ObjHP, X
    BEQ HandleMonsterDied       ; HP reached 0: dies.

; Shared exit of the weapon collision routines in this file.
L74D8_Exit:
    RTS

; Plays the parry tune for a hit that did no damage, unless the damage type in [09] is bomb or fire.
L_PlayParrySoundForDamageType:
    LDA $09
    CMP #$20                    ; Fire damage type
    BEQ L74D8_Exit
    CMP #$08                    ; Bomb damage type
    BEQ L74D8_Exit
    JMP PlayParryTune

; ----------------------------------------------------------------------------
; Counts a kill in the drop counters and puts the monster into the dying metastate.
; IN:   X = monster slot; [09] = damage type of the killing hit.
; OUT:  WorldKillCount + 1; HelpDropCount + 1 up to $0A; IsHelpDropBomb + 1 when HelpDropCount has just
;       become $0A and the damage type is 8 (bomb); ObjMetastate,X = $10; stun timer, knockback and
;       invincibility timer of the slot = 0.
; NOTE: WorldKillCount has no cap here. Once HelpDropCount is $0A neither it nor IsHelpDropBomb changes
;       again until the forced drop is made or Link is harmed. WorldKillCycle is not touched here.
; ----------------------------------------------------------------------------
HandleMonsterDied:
    INC WorldKillCount

    ; Count toward the help drop, up to $0A.
    LDA HelpDropCount
    CMP #$0A
    BCS @SetDyingMetastate
    INC HelpDropCount

    ; If that made it $0A and the killing hit was a bomb (damage type 8), the forced drop will be a
    ; bomb instead of 5 rupees.
    LDA HelpDropCount
    CMP #$0A
    BNE @SetDyingMetastate
    LDA $09
    CMP #$08
    BNE @SetDyingMetastate
    INC IsHelpDropBomb

@SetDyingMetastate:
    ; Death tune, ObjMetastate := $10 (first step of the death sparkle); clear the stun timer.
    JSR UpdateDeadDummy
    LDA #$00
    STA ObjStunTimer, X

; ----------------------------------------------------------------------------
; Clears the slot's knockback direction, knockback distance and invincibility timer.
; IN:   A = 0; X = object slot.
; ----------------------------------------------------------------------------
ResetShoveInfoAndInvincibilityTimer:
    JSR SetShoveInfoWith0
    STA ObjInvincibilityTimer, X
    RTS
