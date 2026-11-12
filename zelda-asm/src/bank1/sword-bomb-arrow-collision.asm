; ============================================================================
; FILE:     bank1/sword-bomb-arrow-collision.asm BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 6037-6349
; SUBJECTS: (none)
; TOPIC:    Per-weapon entry routines of the monster-vs-weapon test: sword shot and magic shot, bomb blast
;           and fire, sword, rod, arrow. Each sets damage type, damage points and hit box; then the
;           Pols Voice arrow kill, the Darknut parry and the knockback.
; SEE ALSO: bank1/monster-collisions.asm (CheckMonsterCollisions calls these per weapon slot);
;           bank1/weapon-damage-death.asm (HandleMonsterWeaponCollision, DealDamage: mask, special
;           monsters, HP); bank1/object-overlap-test.asm (hit points and box test);
;           bank1/shove-knockback.asm (BeginShove); bank7/sword-shot-magic-shot.asm (HandleShotBlocked).
; KEY FACTS:
;   - Sword: damage type 1; damage $10 wooden, $20 white, $40 magic; hits only while the sword's state
;     is 2 (SwordDamagePoints, CheckMonsterSwordCollision).
;   - Sword shot: damage type 1, same damage as the sword held; magic shot (state bit 7): type $10,
;     damage $20; a spreading sword shot (state bit 0) hits nothing
;     (CheckMonsterSwordShotOrMagicShotCollision).
;   - A shot that overlaps a monster ends there even if the monster is immune: the sword shot spreads, the
;     magic shot vanishes or becomes a fire with the book (CheckMonsterSwordShotOrMagicShotCollision).
;   - Bomb: type 8, damage $40, only in bomb state $13, box under $18 pixels; fire (state >= $20): type
;     $20, damage $10, box under $0E pixels (CheckMonsterBombOrFireCollision).
;   - Rod swing (slot $12, state >= $30): type 1 (sword), damage $20; unlike the sword it is tested in
;     every state of the swing (CheckMonsterArrowOrRodCollision).
;   - Arrow (slot $12, state < $20): type 4, damage $20 wooden, $40 silver, box under $0B pixels
;     (CheckMonsterArrowOrRodCollision).
;   - An arrow that overlaps a Pols Voice ($16) kills it whatever its HP or invincibility mask, and
;     keeps flying (CheckMonsterShotCollision).
;   - An arrow that overlaps any other monster turns into a spark (state $20), immune monster or not
;     (CheckMonsterShotCollision).
;   - Sword and rod box: $0C pixels across Link's facing axis, $10 along it (CheckMonsterStabbingCollision).
;   - A Darknut ($0B, $0C) facing exactly against the weapon's direction is not knocked back; the parry
;     tune plays instead (ParryOrShove).
;   - Bomb and fire knock the monster back unless its mask holds the damage type
;     (CheckMonsterBombOrFireCollision).
; CONTENTS: (generated; L = line in this file)
;   L0058 CheckMonsterSwordShotOrMagicShotCollision  code  Tests Link's sword shot or magic shot (slot Y = $0E) against the monster in slot X.
;   L0120 CheckMonsterBombOrFireCollision            code  Tests a detonating bomb or a fire in slot Y ($10 or $11) against the monster in slot X.
;   L0171 L7595_Exit                                 code  Shared exit of the bomb/fire, sword and stabbing tests.
;   L0176 SwordDamagePoints                          data  TABLE: sword damage points; index = InvSword - 1 (0 wooden, 1 white, 2 magic); value = damage in ObjHP
;   L0186 CheckMonsterSwordCollision                 code  Tests Link's sword (slot Y = $0D) against the monster in slot X.
;   L0209 CheckMonsterStabbingCollision              code  Tests a sword or rod swing against a monster with a box that is longer along Link's facing axis.
;   L0243 CheckMonsterArrowOrRodCollision            code  Tests the rod swing or the flying arrow in slot Y = $12 against the monster in slot X.
;   L0281 CheckMonsterShotCollision                  code  Tests an arrow, sword shot or magic shot against a monster; handles Pols Voice and the arrow's spark.
;   L0312 ParryOrShove                               code  Knocks the monster in slot X back in the weapon's direction, or plays the parry tune for a Darknut.
;   L0333 PlayParryTune                              code  Requests the parry tune (Tune0Request := 1).
;   L0338 L763A_Exit                                 code  Shared exit of the arrow and shot tests.
; ============================================================================

; ----------------------------------------------------------------------------
; Tests Link's sword shot or magic shot (slot Y = $0E) against the monster in slot X.
; IN:   X = monster slot; Y = weapon slot ($0E); [02],[03] = monster's middle.
; OUT:  [00] = weapon slot; [06] = 1 when they overlap (not written when the shot is spreading).
; NOTE: Sword shot: damage type 1, damage by the sword held ($10/$20/$40). Magic shot (state bit 7):
;       type $10, damage $20. Box: under $0C pixels on both axes. On any overlap, damage or not, the shot
;       ends (HandleShotBlocked): a sword shot starts spreading; a magic shot is removed, or with the
;       book turns into a fire if a fire slot is free.
; ----------------------------------------------------------------------------
CheckMonsterSwordShotOrMagicShotCollision:
    STY $00                     ; [00] := weapon slot

    LDA #$10
    STA $09                     ; [09] := damage type $10 (magic shot), changed below for a sword shot

    ; Bit 0 of the state set = a sword shot that is spreading out: it cannot hit. Return.
    LDA a:ObjState, Y
    LSR
    BCS @Exit

    LDA #$0C
    STA $0D                     ; [0D] := threshold $0C

    ; Bit 7 of the state set = magic shot: damage $20 (carried in Y).
    LDA a:ObjState, Y
    LDY #$20
    ASL
    BCS @CheckCollision

    ; Sword shot: damage type 1 (sword).
    LDA #$01
    STA $09

    ; Damage by InvSword: 3 magic sword $40, 2 white sword $20, else (wooden) $10.
    LDY #$40
    LDA InvSword
    CMP #$03
    BEQ @CheckCollision
    LDY #$20
    CMP #$02
    BEQ @CheckCollision
    LDY #$10

@CheckCollision:
    TYA                         ; A := damage points
    JSR CheckMonsterShotCollision

    ; If the shot did not overlap the monster, return.
    LDA $06
    BEQ @Exit

    ; End the shot (spread, vanish or turn into a fire), keeping the monster slot in X.
    TXA
    PHA
    LDX #$0E                    ; Shot object slot
    JSR HandleShotBlocked
    PLA
    TAX

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Tests a detonating bomb or a fire in slot Y ($10 or $11) against the monster in slot X.
; IN:   X = monster slot; Y = weapon slot; [02],[03] = monster's middle.
; OUT:  [00] = weapon slot; [06] = 1 when they overlap.
; NOTE: Fire (state >= $20): damage type $20, damage $10, box under $0E pixels on both axes.
;       Bomb: only in state $13; type 8, damage $40, box under $18 pixels. Other bomb states do nothing.
;       The point tested is the weapon's (X+8, Y+8). After the hit the monster is knocked back unless its
;       mask holds the damage type; a Darknut that blocked the hit is still knocked back.
; ----------------------------------------------------------------------------
CheckMonsterBombOrFireCollision:
    STY $00                     ; [00] := weapon slot

    ; Start with the values for a fire.
    LDA #$20
    STA $09                     ; [09] := damage type $20 (fire)
    LDA #$10
    STA $07                     ; [07] := damage $10
    LDA #$0E
    STA $0D                     ; [0D] := threshold $0E

    ; State $20 and up = fire. Below that the slot holds a bomb (or nothing): only state $13, the
    ; blast, can hit.
    LDA a:ObjState, Y
    CMP #$20
    BCS @SetHotspot
    CMP #$13
    BNE L7595_Exit

    ; Values for a bomb.
    LDA #$08
    STA $09                     ; [09] := damage type 8 (bomb)
    LDA #$40
    STA $07                     ; [07] := damage $40
    LDA #$18
    STA $0D                     ; [0D] := threshold $18

@SetHotspot:
    ; [04],[05] := the weapon's middle (X+8, Y+8).
    LDA a:ObjX, Y
    CLC
    ADC #$08
    STA $04
    LDA a:ObjY, Y
    CLC
    ADC #$08
    STA $05
    LDA $0D                     ; A := threshold for both axes

    ; Return if they do not overlap.
    JSR DoObjectsCollide
    BEQ L7595_Exit
    JSR HandleMonsterWeaponCollision

    ; Knock the monster back unless its mask holds this damage type.
    LDA ObjInvincibilityMask, X
    AND $09
    BNE L7595_Exit
    JSR BeginShove

; Shared exit of the bomb/fire, sword and stabbing tests.
L7595_Exit:
    RTS

; TABLE: sword damage points; index = InvSword - 1 (0 wooden, 1 white, 2 magic); value = damage in ObjHP
; units (HP x $10).
SwordDamagePoints:
    .BYTE $10, $20, $40

; ----------------------------------------------------------------------------
; Tests Link's sword (slot Y = $0D) against the monster in slot X.
; IN:   X = monster slot; Y = weapon slot ($0D); [02],[03] = monster's middle.
; OUT:  [00] = weapon slot; [06] = 1 when they overlap (not written unless the sword's state is 2).
; NOTE: The sword hits only in state 2 (fully extended). Damage type 1; damage $10 wooden, $20 white,
;       $40 magic. Falls into CheckMonsterStabbingCollision.
; ----------------------------------------------------------------------------
CheckMonsterSwordCollision:
    STY $00                     ; [00] := weapon slot

    LDA #$01
    STA $09                     ; [09] := damage type 1 (sword)

    ; Return unless the sword's state is 2.
    LDA a:ObjState, Y
    CMP #$02
    BNE L7595_Exit

    ; A := damage points for the sword held.
    LDY InvSword
    LDA SwordDamagePoints-1, Y

; ----------------------------------------------------------------------------
; Tests a sword or rod swing against a monster with a box that is longer along Link's facing axis.
; IN:   A = damage points; X = monster slot; [00] = weapon slot; [02],[03] = monster's middle;
;       [09] = damage type.
; OUT:  [06] = 1 when they overlap.
; NOTE: Link facing up or down: under $0C pixels horizontally and $10 vertically; facing left or right:
;       $10 horizontally and $0C vertically. On overlap goes to ParryOrShove (knockback or Darknut parry).
; ----------------------------------------------------------------------------
CheckMonsterStabbingCollision:
    STA $07                     ; [07] := damage points

    ; Link facing up or down: [0D] := $0C, [0E] := $10.
    LDA ObjDir
    AND #$0C
    BEQ @Horizontal
    LDA #$0C
    STA $0D
    LDA #$10
    JMP @SetVerticalThreshold

@Horizontal:
    ; Link facing left or right: [0D] := $10, [0E] := $0C.
    LDA #$10
    STA $0D
    LDA #$0C
@SetVerticalThreshold:
    STA $0E
    JSR CheckMonsterSlenderWeaponCollision2

    ; If no overlap, return.
    LDA $06
    BEQ L7595_Exit
    JMP ParryOrShove

; ----------------------------------------------------------------------------
; Tests the rod swing or the flying arrow in slot Y = $12 against the monster in slot X.
; IN:   X = monster slot; Y = weapon slot ($12); [02],[03] = monster's middle.
; OUT:  [00] = weapon slot; [06] = 1 when they overlap (not written for an arrow in state $20-$2F).
; NOTE: State >= $30 is the rod: damage type 1 (sword), damage $20, tested like the sword but in every
;       rod state. State < $20 is an arrow in flight: type 4, damage $20 wooden (InvArrow = 1), else $40
;       (silver); box under $0B pixels on both axes. A sparking arrow ($20-$2F) does not hit.
; ----------------------------------------------------------------------------
CheckMonsterArrowOrRodCollision:
    STY $00                     ; [00] := weapon slot
    LDA a:ObjState, Y

    ; State $30 and up: the rod.
    CMP #$30
    BCC :+
    LDA #$01
    STA $09                     ; [09] := damage type 1 (sword)
    LDA #$20                    ; A := damage $20
    BNE CheckMonsterStabbingCollision   ; Always taken.
:
    ; State $20-$2F: the arrow is a spark. Return.
    CMP #$20
    BCS L763A_Exit

    LDA #$04
    STA $09                     ; [09] := damage type 4 (arrow)

    ; A := damage $20 for the wooden arrow (InvArrow = 1), doubled to $40 otherwise (silver).
    LDA #$20
    LDY InvArrow
    CPY #$01
    BEQ :+
    ASL
:
    LDY #$0B
    STY $0D                     ; [0D] := threshold $0B

; ----------------------------------------------------------------------------
; Tests an arrow, sword shot or magic shot against a monster; handles Pols Voice and the arrow's spark.
; IN:   A = damage points; X = monster slot; [00] = weapon slot; [02],[03] = monster's middle;
;       [09] = damage type; [0D] = threshold for both axes.
; OUT:  [06] = 1 when they overlap.
; NOTE: An arrow overlapping a Pols Voice (type $16) kills it outright: its HP is set to 0 and DealDamage
;       is called directly, past the invincibility mask; the arrow is not stopped. On any other monster
;       the arrow becomes a spark (state $20) even when the monster is immune to arrows.
; ----------------------------------------------------------------------------
CheckMonsterShotCollision:
    JSR CheckMonsterSlenderWeaponCollision

    ; If no overlap, return.
    LDA $06
    BEQ L763A_Exit

    ; Y = weapon slot. Slot $12 here is always an arrow (the rod took the other branch above); the
    ; shots in slot $0E go straight to the knockback.
    CPY #$12
    BNE ParryOrShove

    ; Arrow on a Pols Voice: HP := 0 and deal damage again, so that it dies whatever the normal hit
    ; handling above did. Returns from DealDamage: no spark, no knockback, and no invincibility timer
    ; (that is set only by the knockback).
    LDA ObjType, X
    CMP #$16                    ; Pols Voice
    BNE :+
    LDA #$00
    STA ObjHP, X
    JMP DealDamage
:
    ; Arrow on any other monster: the arrow's state := $20 (spark), its animation counter := 3.
    LDA #$20
    STA a:ObjState, Y
    LDA #$03
    STA ObjAnimCounter, Y

; Knocks the monster in slot X back in the weapon's direction, or plays the parry tune for a Darknut.
; IN: X = monster slot; Y = weapon slot; [00] = weapon slot; [09] = damage type.
; Runs after the hit was already applied; this part decides only between knockback and parry tune.
ParryOrShove:
    ; Darknut ($0B red, $0C blue) whose direction and the weapon's direction together are exactly
    ; up + down ($0C) or left + right ($03): parry tune, no knockback.
    LDA ObjType, X
    CMP #$0B                    ; Red Darknut
    BEQ :+
    CMP #$0C                    ; Blue Darknut
    BNE @Shove
:
    LDA a:ObjDir, Y
    ORA ObjDir, X
    CMP #$0C
    BEQ PlayParryTune
    CMP #$03
    BEQ PlayParryTune
@Shove:
    ; BeginShove itself does nothing when the monster's mask holds the damage type or its
    ; invincibility timer is running.
    JMP BeginShove

; Requests the parry tune (Tune0Request := 1).
PlayParryTune:
    LDA #$01
    STA Tune0Request

; Shared exit of the arrow and shot tests.
L763A_Exit:
    RTS
