; ============================================================================
; FILE:     bank1/sword-bomb-arrow-collision.asm BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 6037-6349
; SUBJECTS: (none)
; TOPIC:    Per-weapon damage: beam, magic, bomb, fire, sword, rod, arrow; parry
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0031 CheckMonsterSwordShotOrMagicShotCollision  code  Params:
;   L0101 CheckMonsterBombOrFireCollision            code  Params:
;   L0152 L7595_Exit                                 code
;   L0157 SwordDamagePoints                          data  TRIFORCE: Sword damage per level: Wood=$10(1HP), White=$20(2HP), Magic=$40(4HP).
;   L0170 CheckMonsterSwordCollision                 code  Params:
;   L0197 CheckMonsterStabbingCollision              code  Params:
;   L0237 CheckMonsterArrowOrRodCollision            code  Params:
;   L0280 CheckMonsterShotCollision                  code  Params:
;   L0308 ParryOrShove                               code
;   L0329 PlayParryTune                              code
;   L0333 L763A_Exit                                 code
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
CheckMonsterSwordShotOrMagicShotCollision:
    STY $00                     ; [00] holds the weapon slot

    ; Set magic shot damage type ($10) in [09].
    LDA #$10
    STA $09

    ; If the shot is a sword shot that's spreading out, then return.
    LDA a:ObjState, Y
    LSR
    BCS @Exit

    ; Set horizontal collision threshold $C.
    LDA #$0C
    STA $0D

    ; If the weapon is a magic shot, go use $20 damage points.
    LDA a:ObjState, Y
    LDY #$20
    ASL
    BCS @CheckCollision

    ; The weapon is a sword shot.
    ;
    ; First, change the damage type to sword (1).
    LDA #$01
    STA $09

    ; Determine the damage points for the kind of sword.
    ; - wood sword:   $10
    ; - white sword:  $20
    ; - master sword: $40
    LDY #$40
    LDA Items
    CMP #$03
    BEQ @CheckCollision
    LDY #$20
    CMP #$02
    BEQ @CheckCollision
    LDY #$10

@CheckCollision:
    TYA
    JSR CheckMonsterShotCollision

    ; If no collision, then return.
    LDA $06
    BEQ @Exit

    ; Else handle blocking the shot.
    TXA
    PHA
    LDX #$0E                    ; Shot object slot
    JSR HandleShotBlocked
    PLA
    TAX

@Exit:
    RTS

; Params:
; X: monster slot
; Y: weapon slot
; [02]: monster mid X
; [03]: monster mid Y
;
; Returns:
; [00]: weapon slot
;
; TRIFORCE: Fire damage=$10(1HP), Bomb damage=$40(4HP). Both multiples of $10.
CheckMonsterBombOrFireCollision:
    STY $00                     ; [00] holds the weapon slot

    ; Set up parameters for a fire.
    LDA #$20
    STA $09                     ; [09] holds fire damage type
    LDA #$10
    STA $07                     ; [07] holds $10 damage points
    LDA #$0E
    STA $0D                     ; [0D] holds collision threshold $E

    ; If the weapon is a fire, then go set the hotspot/midpoint.
    ; But if the bomb is not detonating, then return.
    LDA a:ObjState, Y
    CMP #$20
    BCS @SetHotspot
    CMP #$13
    BNE L7595_Exit

    ; Else set up parameters for a bomb.
    LDA #$08
    STA $09                     ; [09] holds bomb damage type
    LDA #$40
    STA $07                     ; [07] holds $40 damage points
    LDA #$18
    STA $0D                     ; [0D] holds collision threshold $18

@SetHotspot:
    ; The weapon's midpoint/hotspot is at (X + 8, Y + 8).
    ; Pass it in [04] and [05].
    LDA a:ObjX, Y
    CLC
    ADC #$08
    STA $04
    LDA a:ObjY, Y
    CLC
    ADC #$08
    STA $05
    LDA $0D                     ; Pass collision threshold in A.

    ; If the objects do not collide, then return.
    JSR DoObjectsCollide
    BEQ L7595_Exit
    JSR HandleMonsterWeaponCollision

    ; If the monster is weak to the damage type, then shove it.
    LDA ObjInvincibilityMask, X
    AND $09
    BNE L7595_Exit
    JSR BeginShove

L7595_Exit:
    RTS

; TRIFORCE: Sword damage per level: Wood=$10(1HP), White=$20(2HP), Magic=$40(4HP).
; TRIFORCE: All multiples of $10 so ObjHP low nibble stays 0 after subtraction.
SwordDamagePoints:
    .BYTE $10, $20, $40

; Params:
; X: monster slot
; Y: weapon slot
; [02]: monster mid X
; [03]: monster mid Y
;
; Returns:
; [00]: weapon slot
; [06]: 1 if objects collide
;
CheckMonsterSwordCollision:
    STY $00                     ; [00] holds the weapon slot

    ;Set sword damage type (1) in [09].
    LDA #$01
    STA $09

    ; If the sword is not fully extended, then return.
    LDA a:ObjState, Y
    CMP #$02
    BNE L7595_Exit

    ; Look up and set the damage points for the sword type.
    LDY Items
    LDA SwordDamagePoints-1, Y

; Params:
; A: damage points
; X: monster slot
; Y: weapon slot
; [02]: monster mid X
; [03]: monster mid Y
; [09]: damage type
;
; Returns:
; [06]: 1 if objects collide
;
CheckMonsterStabbingCollision:
    STA $07                     ; [07] damage points

    ; If Link's direction is vertical, then set collision thresholds accordingly:
    ; [0D] := $C
    ; [0E] := $10
    LDA ObjDir
    AND #$0C
    BEQ @Horizontal
    LDA #$0C
    STA $0D
    LDA #$10
    JMP :+

@Horizontal:
    ; Else switch it:
    ; [0D] := $10
    ; [0E] := $C
    LDA #$10
    STA $0D
    LDA #$0C
:
    STA $0E
    JSR CheckMonsterSlenderWeaponCollision2

    ; If no collision, then return.
    LDA $06
    BEQ L7595_Exit
    JMP ParryOrShove

; Params:
; X: monster slot
; Y: weapon slot
; [02]: monster mid X
; [03]: monster mid Y
;
; Returns:
; [00]: weapon slot
; [06]: 1 if objects collide
;
CheckMonsterArrowOrRodCollision:
    STY $00                     ; [00] holds the weapon slot
    LDA a:ObjState, Y

    ; If the weapon is a rod, then
    ; go check a stabbing collision using rod parameters.
    CMP #$30
    BCC :+
    LDA #$01
    STA $09                     ; [09] holds sword damage type
    LDA #$20                    ; Pass $20 damage points.
    BNE CheckMonsterStabbingCollision
:
    ; We have an arrow. If it's no longer flying, then return.
    CMP #$20
    BCS L763A_Exit

    ; Set up arrow parameters.
    LDA #$04
    STA $09                     ; [09] holds arrow damage type

    ; Use $20 damage points for wooden arrows, and $40 for silver ones.
    LDA #$20
    LDY InvArrow
    CPY #$01
    BEQ :+
    ASL
:
    LDY #$0B
    STY $0D                     ; [0D] holds collision threshold $B

; Params:
; A: damage points
; X: monster object slot
; [00]: weapon object slot
; [02]: monster object mid X
; [03]: monster object mid Y
; [09]: damage type
; [0D]: horizontal threshold
;
; Returns:
; [06]: 1 if objects collide
;
CheckMonsterShotCollision:
    JSR CheckMonsterSlenderWeaponCollision

    ; If no collision, return.
    LDA $06
    BEQ L763A_Exit

    ; $12 is the arrow and rod slot. But this routine will only
    ; be called if it's an arrow.
    ;
    ; If the weapon is an arrow and the monster is Pol's Voice;
    ; then set HP to 0, and deal damage, even if the collision
    ; checking routine above dealt some. This way it dies for sure.
    CPY #$12
    BNE ParryOrShove
    LDA ObjType, X
    CMP #$16                    ; Pol's Voice
    BNE :+
    LDA #$00
    STA ObjHP, X
    JMP DealDamage
:
    ; If the weapon is an arrow, set its state to spark ($20).
    LDA #$20
    STA a:ObjState, Y
    LDA #$03
    STA ObjAnimCounter, Y

ParryOrShove:
    ; For any weapon type:
    ;
    ; If the monster is a darknut and both are facing opposite directions,
    ; then go parry.
    LDA ObjType, X
    CMP #$0B                    ; Red Darknut
    BEQ :+
    CMP #$0C                    ; Blue Darknut
    BNE :++
:
    LDA a:ObjDir, Y
    ORA ObjDir, X
    CMP #$0C
    BEQ PlayParryTune
    CMP #$03
    BEQ PlayParryTune
:
    ; Else shove the monster.
    JMP BeginShove

PlayParryTune:
    LDA #$01
    STA Tune0Request

L763A_Exit:
    RTS
