; ============================================================================
; FILE:     bank1/link-collision-harm.asm BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 5592-5828
; SUBJECTS: Link
; TOPIC:    A monster or monster shot touching Link: the overlap test, shield parry rules, damage lookup,
;           ring halving, heart subtraction and the switch to the death mode.
; SEE ALSO: bank1/monster-collisions.asm (Collision_CheckMonsterVsWeaponsAndLink calls this; Obj_GetMiddle;
;           Damage_ObjTypeDamageToLink); bank1/shove-knockback.asm (Shove_Begin: Link's knockback and invincibility
;           timer); bank1/object-overlap-test.asm (Collision_TestPoints); bank7/fire.asm (candle fire harming
;           Link through Link_BeHarmed); bank4/drop-items.asm (readers of the counters reset here).
; KEY FACTS:
;   - No touch is tested while Link's invincibility timer, the clock, Link's stun timer or the monster's
;     stun timer is nonzero, while Link's state is exactly $40 (halted) or a Like-Like holds him
;     (Link_CheckMonsterTouch, Link_CheckMonsterTouch_NoReset).
;   - Touch box: Link's middle (X+8, Y+8) against the object's middle, hit when both axis distances are
;     under 9 pixels (Link_CheckMonsterTouch_NoReset).
;   - Object types $53 and up are shots; they touch Link only in states $10-$1F (Link_CheckMonsterTouch_NoReset).
;   - A shot is parried only when Link's state is $00-$0F (not swinging or using an item) and Link and the
;     shot face opposite ways on one axis. Types $53, $54, $5B, $5C need no magic shield; $55, $57, $58,
;     $59 need it; $56 and $5A are never parried (Link_CheckMonsterTouch_NoReset@FacingOpposite).
;   - Damage = Damage_ObjTypeDamageToLink[type]: low nibble = whole hearts, high nibble = sixteenths of a heart
;     ($80 = half a heart) (Link_HarmByObjType).
;   - Each ring grade halves the damage: blue 1/2, red 1/4 (Link_BeHarmed).
;   - Every call of Link_BeHarmed zeroes WorldKillCount, HelpDropCount and IsHelpDropBomb, even when the
;     damage is 0 (Link_BeHarmed@ResetHelp).
;   - Link dies when the whole-heart damage exceeds his full hearts or a borrow is needed with no full
;     heart left; then GameMode becomes $11 (Link_BeHarmed@HandleDied).
;   - A borrow from the full hearts costs 1/256 heart extra (Link_BeHarmed@BorrowHeart).
; CONTENTS: (generated; L = line in this file)
;   L0048 Link_CheckMonsterTouch          code  Tests whether the monster or shot in slot X touches Link and, unless parried, harms and shoves him.
;   L0073 Link_CheckMonsterTouch_NoReset  code  Second entry: the caller has set [02],[03] (object middle), [00] = 0 and cleared the results itself.
;   L0159 Link_CheckMonsterTouch_Return   code  Shared exit of Link_CheckMonsterTouch.
;   L0164 Link_HarmByObjType              code  Starts Link's knockback and harms him by the touching object's damage from Damage_ObjTypeDamageToLink.
;   L0193 Link_BeHarmed                   code  Takes damage [0D]:[0E], halved per ring grade, from Link's hearts; starts death mode when he dies.
;   L0268 Link_BeHarmed_Return            code  Shared exit: the RTS of Link_BeHarmed, also used by Boomerang_CheckMonsterCollision.
; ============================================================================

; ----------------------------------------------------------------------------
; Tests whether the monster or shot in slot X touches Link and, unless parried, harms and shoves him.
; IN:   X = object slot of the monster or shot.
; OUT:  [06] = 1 when the object overlaps Link and was not parried, else 0; [0C] = 1 when Link was harmed
;       (Link_HarmByObjType ran), else 0; IsShotCollidingWithLink = 1 when a shot (type >= $53) overlaps Link, parried
;       or not; [00] = 0 (Link's slot); [09] = 0 (no damage type); [02],[03] = the object's middle;
;       [04],[05] = Link's middle when the test got that far.
; NOTE: Skipped entirely (no hit) while any of these is nonzero: Link's invincibility timer, InvClock,
;       Link's stun timer, the object's stun timer. A stunned monster cannot hurt Link by touch.
; ----------------------------------------------------------------------------
Link_CheckMonsterTouch:
    JSR Obj_GetMiddle         ; [02],[03] := the object's middle

    ; Clear the results.
    LDA #$00
    STA IsShotCollidingWithLink
    STA $06
    STA $09
    STA $0C
    LDY #$00
    STY $00

    ; Return no collision if Link is invincible, the clock is held, Link is stunned
    ; or the object is stunned.
    LDA ObjInvincibilityTimer
    ORA InvClock
    ORA ObjStunTimer
    ORA ObjStunTimer, X
    BNE Link_CheckMonsterTouch_Return

; Second entry: the caller has set [02],[03] (object middle), [00] = 0 and cleared the results itself.
; It skips the invincibility, clock and stun tests above. Ganon enters here with his own middle point
; after testing Link's invincibility timer himself (not InvClock or the stun timers). The clock still
; protects Link from Ganon on nearly every frame: while InvClock is set, Mode5Play_UpdateWorld adds $10
; to Link's invincibility timer every frame, and by that arithmetic the byte reads 0 on 1 frame in 33.
Link_CheckMonsterTouch_NoReset:
    ; Return no collision if Link's state is exactly $40 (halted) or a Like-Like holds him.
    LDA ObjState
    CMP #$40
    BEQ Link_CheckMonsterTouch_Return
    LDA IsLinkParalyzed
    BNE Link_CheckMonsterTouch_Return

    ; Object types $53 and up are shots. A shot can touch Link only in states $10-$1F.
    LDA ObjType, X
    CMP #$53
    BCC :+
    LDA ObjState, X
    AND #$F0
    CMP #$10
    BNE Link_CheckMonsterTouch_Return
:
    ; [04],[05] := Link's middle (X+8, Y+8).
    LDA ObjX
    CLC
    ADC #$08
    STA $04
    LDA ObjY
    CLC
    ADC #$08
    STA $05

    ; No collision unless both axis distances are under 9 pixels.
    LDA #$09
    JSR Collision_TestPoints
    BEQ Link_CheckMonsterTouch_Return

    ; They overlap ([06] = 1). A monster (type < $53) harms Link at once.
    LDA ObjType, X
    CMP #$53
    BCC Link_HarmByObjType

    ; The object is a shot. Flag that a shot touched Link (set even if it is parried below).
    INC IsShotCollidingWithLink

    ; Shot types $56 and $5A can never be parried. Nor can any shot while Link's state is $10 or
    ; above (swinging the sword or using an item).
    CMP #$56
    BEQ Link_HarmByObjType
    CMP #$5A
    BEQ Link_HarmByObjType
    LDA ObjState
    AND #$F0
    BNE Link_HarmByObjType

    ; Link must face against the shot: the two directions together must hold both vertical bits
    ; (up + down) or both horizontal bits (left + right). Otherwise harm Link.
    ; The masks pick one axis at a time, so a diagonal shot is parried if either of its components
    ; opposes Link's direction.
    LDA ObjDir
    ORA ObjDir, X
    AND #$0C
    CMP #$0C
    BEQ @FacingOpposite
    LDA ObjDir
    ORA ObjDir, X
    AND #$03
    CMP #$03
    BNE Link_HarmByObjType

@FacingOpposite:
    ; Types below $55 ($53 flying rock, $54) and $5B and up ($5B monster arrow, $5C monster boomerang)
    ; are parried by any shield.
    LDA ObjType, X
    CMP #$55
    BCC @Parry
    CMP #$5B
    BCS @Parry

    ; The rest ($55 fireball, $57 sword shot, $58/$59 magic shot) need the magic shield.
    LDA InvMagicShield
    BEQ Link_HarmByObjType

@Parry:
    ; Play the parry tune and cancel the collision ([06] := 0). IsShotCollidingWithLink stays 1.
    LDA #$01
    STA Tune0Request
    LDA #$00
    STA $06

; Shared exit of Link_CheckMonsterTouch.
Link_CheckMonsterTouch_Return:
    RTS

; Starts Link's knockback and harms him by the touching object's damage from Damage_ObjTypeDamageToLink.
; IN: X = object slot of the attacker. Sets [0C] = 1, then falls into Link_BeHarmed.
Link_HarmByObjType:
    ; Knockback: $20 pixels away from the attacker, and Link's invincibility timer := $18 (Shove_Begin).
    JSR Shove_Begin

    ; [0C] := 1: Link was harmed. (Wallmaster and Like-Like capture Link on this flag.)
    INC $0C

    ; Unpack the damage byte of the attacker's type into a 16-bit amount:
    ; [0D] = low nibble = whole hearts; [0E] = high nibble in place = 1/256 hearts ($80 = half a heart).
    LDY ObjType, X
    LDA Damage_ObjTypeDamageToLink, Y
    PHA
    AND #$0F
    STA $0D
    PLA
    AND #$F0
    STA $0E

; ----------------------------------------------------------------------------
; Takes damage [0D]:[0E], halved per ring grade, from Link's hearts; starts death mode when he dies.
; IN:   X = object slot of the attacker; [0D] = whole hearts of damage; [0E] = 1/256 hearts of damage.
; OUT:  HeartValues, HeartPartial reduced; WorldKillCount, HelpDropCount, IsHelpDropBomb = 0.
;       On death: GameMode = $11, Link's state 0, facing down, full hearts and HeartPartial 0.
; NOTE: Does not test Link's invincibility timer; the callers do. The counters are zeroed on every call,
;       before the hearts are touched, even when the damage is 0 (the whirlwind, type $2E, has damage 0).
;       Also entered directly by Link's own candle fire with damage $00:$80 (half a heart) and X = the
;       fire's slot ($10 or $11); ObjType has only 13 entries, so the whirlwind test below then reads
;       RoomObjTemplateType ($35F) for slot $10 and the byte after it ($360) for slot $11.
; ----------------------------------------------------------------------------
Link_BeHarmed:
    ; Request the "hurt" sample (request bit $08) unless the attacker is the whirlwind (type $2E).
    LDY ObjType, X
    CPY #$2E
    BEQ :+
    LDA #$08
    JSR Sample_Play
:
    ; Halve the 16-bit damage once per ring grade: blue ring (1) = 1/2, red ring (2) = 1/4.
    LDY InvRing
    BEQ @ResetHelp
:
    LSR $0D
    ROR $0E
    DEY
    BNE :-

@ResetHelp:
    ; Zero the kill counters behind the fairy drop (WorldKillCount) and the help drop
    ; (HelpDropCount, IsHelpDropBomb). Reached a second time after a borrow.
    LDA #$00
    STA WorldKillCount
    STA HelpDropCount
    STA IsHelpDropBomb

    ; If HeartPartial >= the fraction byte, subtract it. Else go borrow from the full hearts.
    LDA HeartPartial
    CMP $0E
    BCC @BorrowHeart
    SEC
    SBC $0E
    STA HeartPartial

    ; If full hearts (low nibble) >= the whole-heart byte, subtract it and return.
    ; Else Link dies.
    LDA HeartValues
    AND #$0F
    CMP $0D
    BCC @HandleDied
    LDA HeartValues
    SEC
    SBC $0D
    STA HeartValues
    RTS

@BorrowHeart:
    ; The fraction left to take is [0E] - HeartPartial; one full heart is turned into a partial
    ; heart of $FF (not $100), so a borrow costs 1/256 heart more than plain 16-bit subtraction.
    ; Example: 4 full hearts + $70, damage $80 -> 3 full hearts + $EF (plain subtraction: $F0).
    LDA $0E
    SEC
    SBC HeartPartial
    STA $0E
    LDA HeartValues
    AND #$0F
    BEQ @HandleDied             ; No full heart to borrow from: Link dies.
    DEC HeartValues
    LDA #$FF
    STA HeartPartial
    BNE @ResetHelp              ; Always taken: subtract the remainder.

@HandleDied:
    ; Link died: full hearts := 0 (the container count in the high nibble is kept), HeartPartial := 0,
    ; Link's state := 0, GameMode := $11 (death) with submode 0, Link faces down (4).
    LDA HeartValues
    AND #$F0
    STA HeartValues
    JSR Mode_End             ; Returns A = 0.
    STA HeartPartial
    STA ObjState
    LDA #$11
    STA GameMode
    LDA #$04
    STA ObjDir
; Shared exit: the RTS of Link_BeHarmed, also used by Boomerang_CheckMonsterCollision.
Link_BeHarmed_Return:
    RTS
