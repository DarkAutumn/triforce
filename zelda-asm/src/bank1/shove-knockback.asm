; ============================================================================
; FILE:     bank1/shove-knockback.asm    BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 6495-6749
; SUBJECTS: Filler
; TOPIC:    Knockback and invincibility frames; bank tail
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0032 BeginShove         code  Params:
;   L0225 Filler_7751        data
;   L0250 SwitchBank_Local1  code
; ============================================================================

; Params:
; X: caller's object index, a monster
;    the attacker, if monster attacks Link
;    else defender, if weapon attacks monster
; [00]: 0 if monster attacks Link (defender),
;       else weapon slot (attacker)
; [09]: damage type,
;       only if a monster is defending
; [0B]: direction,
;       only if a monster is defending
;
; Only monsters (and fire) call this routine.
;
; When monsters attack Link, they set [00] to 0 for Link's slot.
;
; On the other hand, if a weapon attacks a monster,
; the monster sets [00] to the weapon's slot.
;
BeginShove:
    ; If a weapon attacks a monster, and the weapon's damage
    ; type matches the monster's invincibility mask;
    ; then return without shoving.
    ;
    ; If a monster attacks Link, then damage type will be 0.
    LDY $00
    CPX #$0D
    BCS :+
    LDA ObjInvincibilityMask, X
    AND $09
    BNE @Exit
:
    ; [08], [04], and [05] are used in determining the direction to push Link,
    ; if he is defending. The goal is to push Link away from the monster.
    ;
    ; If instead a monster is defending, then all this calculation
    ; won't matter, because the shove direction will be set to the
    ; weapon's direction.
    ;
    ; As such, calculate a direction from the monster to the defender,
    ; assuming it's Link.
    ;
    ; Start with an assumption that the direction is vertical:
    ; Store 8 (up) in [08], monster Y in [04], and other Y in [05].
    LDA #$08
    STA $08
    LDA ObjY, X
    STA $04
    LDA a:ObjY, Y
    STA $05

    ; If the defender is Link and his grid offset <> 0 then
    ; use his usual object direction variable to determine
    ; which axis to check.
    CPY #$00
    BNE :+
    LDA ObjGridOffset
    BEQ :+
    LDA ObjDir
    AND #$03
    BNE @CheckHorizontal
    BEQ @CheckVertical
:
    ; For monsters defending, or Link with grid offset = 0,
    ; check direction in [0B].
    ;
    ; For monsters defending, this won't matter. For Link with grid
    ; offset = 0, I believe the result will be arbitrary.
    ;
    ; UNKNOWN:
    ; [0B] isn't always set, ifLink's grid offset = 0.
    ; For example, if fire attacks him.
    LDA $0B
    CMP #$04
    BCS @CheckVertical

@CheckHorizontal:
    ; If the direction checked is left or right,
    ; set [08] to 2 (left), monster X in [04] and other X in [05].
    LDA #$02
    STA $08
    LDA ObjX, X
    STA $04
    LDA a:ObjX, Y
    STA $05

@CheckVertical:
    ; If the monster is down or right of the other object,
    ; then use the opposite direction already in [08].
    ; Else shift [08] right to turn it into the opposite
    ; direction up or down.
    ;
    ; Again assuming Link is the defender; after this, [08] points
    ; Link away from the monster.
    LDA $04
    CMP $05
    BCS :+
    LSR $08
:
    ; If a monster is the defender, go handle it separately.
    CPY #$00
    BNE @MonsterDefender

    ; Link is the defender.
    ;
    ; If Link is invincible, return without shoving.
    LDA ObjInvincibilityTimer
    BNE @Exit

    ; Store the shove direction we determined.
    ;
    ; Turn on the high bit to indicate that the next time we try to
    ; move by shoving, it will be the first time for this shove action.
    LDA $08
    ORA #$80
    STA ObjShoveDir

    ; Mark Link invincible $18 frames, and should move $20 pixels.
    LDA #$18
    STA ObjInvincibilityTimer
    LDA #$20
    STA ObjShoveDistance

    ; If Link was shoved by one of his own weapons, then
    ; nothing else to do. Return.
    CPX #$0D
    BCS @Exit

    ; Link was shoved by a monster.
    ;
    ; If the monster's object attribute "reverse after hit Link" is set,
    ; then return.
    LDA ObjAttr, X
    AND #$80
    BNE @Exit

    ; If the attacker is a vire, then return.
    LDA ObjType, X
    CMP #$12
    BEQ @Exit

    ; Turn the attacker in the opposite direction.
    LDA ObjDir, X
    JSR GetOppositeDir
    STA ObjDir, X

@Exit:
    RTS

@MonsterDefender:
    ; A monster is the defender. A weapon is the attacker.
    ;
    ; Set the shove direction to the weapon's direction,
    ; instead of the direction passed in [0B].
    LDA a:ObjDir, Y
    STA $08

    ; If the monster's attribute "reverse after hit Link" is set, then
    ; combine the shove direction with $40.
    ;
    ; UNKNOWN:
    ; But the shoving routing doesn't read this flag.
    LDA ObjAttr, X
    AND #$80
    BEQ :+
    LDA $08
    ORA #$40
    STA $08
:
    ; If the monster is invincible, then return without shoving.
    LDA ObjInvincibilityTimer, X
    BNE @Exit2

    ; If the monster is not a Gohma, then go store the shove direction.
    LDA ObjType, X
    CMP #$33
    BEQ @Gohma
    CMP #$34
    BNE @SetShoveDir

@Gohma:
    ; This a Gohma, but if the current part [0F] is not an eye part (3 or 4),
    ; then return, so that the hit doesn't register.
    LDA $0F
    CMP #$03
    BEQ :+
    CMP #$04
    BNE @Exit2
:
    ; If Gohma's eye state/frame image <> 3, then return.
    LDA $046B, X
    CMP #$03
    BNE @Exit2

@SetShoveDir:
    ; Store the shove direction for the monster.
    ;
    ; Turn on the high bit to indicate that the next time we try to
    ; move by shoving, it will be the first time for this shove action.
    LDA $08
    ORA #$80
    STA ObjShoveDir, X

    ; Mark the monster invincible $10 frames, and should move $40 pixels.
    LDA #$40
    STA ObjShoveDistance, X
    LDA #$10
    STA ObjInvincibilityTimer, X

@Exit2:
    RTS

Filler_7751:
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF

.SEGMENT "BANK_01_ISR"




; Unknown block
    .BYTE $78, $D8, $A9, $00, $8D, $00, $20, $A2
    .BYTE $FF, $9A, $AD, $02, $20, $29, $80, $F0
    .BYTE $F9, $AD, $02, $20, $29, $80, $F0, $F9
    .BYTE $09, $FF, $8D, $00, $80, $8D, $00, $A0
    .BYTE $8D, $00, $C0, $8D, $00, $E0, $A9, $0F
    .BYTE $20, $98, $BF, $A9, $00, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $A9, $07, $20, $AC, $BF, $4C, $40, $E4
    .BYTE $8D, $00, $80, $4A, $8D, $00, $80, $4A
    .BYTE $8D, $00, $80, $4A, $8D, $00, $80, $4A
    .BYTE $8D, $00, $80, $60

SwitchBank_Local1:
    STA $E000
    LSR
    STA $E000
    LSR
    STA $E000
    LSR
    STA $E000
    LSR
    STA $E000
    RTS

.SEGMENT "BANK_01_VEC"




; Unknown block
    .BYTE $84, $E4, $50, $BF, $F0, $BF
